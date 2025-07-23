pub mod gguf;
pub mod json;
pub mod llm;
pub mod onnx;

use crate::{
    Element,
    layers::{convolution::conv2d_shape, pooling::maxpool2d_shape},
    model::Model,
    padding::pad_model,
    quantization::{AbsoluteMax, ModelMetadata, ScalingStrategy},
    tensor::Shape,
};
use anyhow::{Context, Error, Result, bail, ensure};
use itertools::Either;
use tracing::debug;
use tract_onnx::{pb::ModelProto, prelude::*};

/// Utility struct for loading a onnx model with float weights and producing a quantized model
/// that can be used for inference and proving.
#[derive(Debug)]
pub struct FloatOnnxLoader<'a, S: ScalingStrategy> {
    /// Either a path to model file or memmap'd bytes
    model: Either<String, &'a [u8]>,
    scaling_strategy: S,
    model_type: Option<ModelType>,
    keep_float: bool,
}

pub type DefaultFloatOnnxLoader<'a> = FloatOnnxLoader<'a, AbsoluteMax>;

impl DefaultFloatOnnxLoader<'_> {
    pub fn new(model_path: &str) -> Self {
        Self::new_with_scaling_strategy(model_path, AbsoluteMax::new())
    }
}

impl<'a, S: ScalingStrategy> FloatOnnxLoader<'a, S> {
    pub fn new_with_scaling_strategy(model_path: &str, scaling_strategy: S) -> Self {
        Self {
            model: Either::Left(model_path.to_string()),
            scaling_strategy,
            model_type: None,
            keep_float: false,
        }
    }
    pub fn from_bytes_with_scaling_strategy(model_bytes: &'a [u8], scaling_strategy: S) -> Self {
        Self {
            model: Either::Right(model_bytes),
            scaling_strategy,
            model_type: None,
            keep_float: false,
        }
    }

    pub fn with_scaling_strategy(mut self, scaling_strategy: S) -> Self {
        self.scaling_strategy = scaling_strategy;
        self
    }

    pub fn with_model_type(mut self, model_type: ModelType) -> Self {
        self.model_type = Some(model_type);
        self
    }

    pub fn with_keep_float(mut self, keep_float: bool) -> Self {
        self.keep_float = keep_float;
        self
    }

    pub fn build(self) -> Result<(Model<Element>, ModelMetadata)> {
        let proto = match self.model {
            Either::Left(path) => load_proto_from_path(&path)?,
            Either::Right(bytes) => {
                use prost_tract_compat::Message;
                ModelProto::decode(bytes)
                    .map_err(|e| Error::msg(format!("Failed to load model: {e:?}")))?
            }
        };
        if let Some(model_type) = self.model_type {
            model_type.validate_proto(&proto)?
        }
        let float_model = load_float_model(&proto)?;
        debug!("Input shape: {:?}", float_model.input_shapes());
        let mut kept_float = None;
        if self.keep_float {
            kept_float = Some(float_model.clone());
        }
        let (quantized_model, mut md) = self.scaling_strategy.quantize(float_model)?;
        let padded_model = pad_model(quantized_model)?;
        md.float_model = kept_float;
        Ok((padded_model, md))
    }
}

fn load_proto_from_path(path: &str) -> Result<ModelProto> {
    tract_onnx::onnx()
        .proto_model_for_path(path)
        .map_err(|e| Error::msg(format!("Failed to load model: {e:?}")))
}
// Supported operators
const ACTIVATION: [&str; 2] = ["Relu", "Sigmoid"];
const CONVOLUTION: [&str; 1] = ["Conv"];
const DOWNSAMPLING: [&str; 1] = ["MaxPool"];
const LINEAR_ALG: [&str; 2] = ["Gemm", "MatMul"];
const RESHAPE: [&str; 2] = ["Flatten", "Reshape"];

fn is_mlp(model: &ModelProto) -> Result<bool> {
    let mut prev_was_gemm_or_matmul = false;
    let graph = model.graph.as_ref().context("Model has no graph")?;

    for node in graph.node.iter() {
        if LINEAR_ALG.contains(&node.op_type.as_str()) {
            if prev_was_gemm_or_matmul {
                return Ok(false);
            }
            prev_was_gemm_or_matmul = true;
        } else if ACTIVATION.contains(&node.op_type.as_str()) {
            if !prev_was_gemm_or_matmul {
                return Ok(false);
            }
            prev_was_gemm_or_matmul = false;
        } else {
            return Err(Error::msg(format!(
                "Operator '{}' unsupported, yet.",
                node.op_type.as_str()
            )));
        }
    }

    Ok(true)
}

fn is_cnn(model: &ModelProto) -> Result<bool> {
    let mut is_cnn = true;
    let mut found_lin = false;

    let graph = model.graph.as_ref().context("Model has no graph")?;
    let mut previous_op = "";

    for node in graph.node.iter() {
        let op_type = node.op_type.as_str();

        if !CONVOLUTION.contains(&op_type)
            && !DOWNSAMPLING.contains(&op_type)
            && !ACTIVATION.contains(&op_type)
            && !LINEAR_ALG.contains(&op_type)
            && !RESHAPE.contains(&op_type)
        {
            return Err(Error::msg(format!(
                "Operator '{op_type}' unsupported, yet."
            )));
        }

        if ACTIVATION.contains(&op_type) {
            is_cnn =
                is_cnn && (LINEAR_ALG.contains(&previous_op) || CONVOLUTION.contains(&previous_op));
        }

        if DOWNSAMPLING.contains(&op_type) {
            is_cnn = is_cnn && ACTIVATION.contains(&previous_op);
        }

        // Check for dense layers
        if LINEAR_ALG.contains(&op_type) {
            found_lin = true;
        }

        // Conv layers should appear before dense layers
        if found_lin && CONVOLUTION.contains(&op_type) {
            is_cnn = false;
        }
        previous_op = op_type;
        if !is_cnn {
            break;
        }
    }

    Ok(is_cnn)
}

pub fn safe_conv2d_shape(input_shape: &Shape, filter_shape: &Shape) -> Result<Shape> {
    let result = check_filter(filter_shape);
    assert!(result.is_ok(), "conv2d: Failed {:?}", result.unwrap_err());

    check_cnn_input(input_shape).context("conv2d: invalid input shape")?;

    Ok(conv2d_shape(input_shape, filter_shape))
}

pub fn check_filter(filter_shape: &Shape) -> Result<()> {
    ensure!(filter_shape.len() == 4, "Filter should be 4D tensor.");
    ensure!(
        filter_shape[2] == filter_shape[3],
        "Filter should be square."
    );
    Ok(())
}

pub fn check_cnn_input(input_shape: &Shape) -> Result<()> {
    ensure!(input_shape.len() == 3, "input should be 3d tensor");
    ensure!(input_shape[1] == input_shape[2], "input should be square");
    Ok(())
}

pub fn safe_maxpool2d_shape(input_shape: &Shape) -> Result<Shape> {
    check_cnn_input(input_shape).context("maxpool2d: invalid input shape")?;
    Ok(maxpool2d_shape(input_shape))
}

/// Enum representing the different types of models that can be loaded
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    MLP,
    CNN,
}

impl ModelType {
    /// Analyze the given filepath and determine if it matches this model type
    pub fn validate_file(&self, filepath: &str) -> Result<()> {
        let model = load_proto_from_path(filepath)?;
        self.validate_proto(&model)
    }

    /// Analyze the `ModelProto` and determine if it matches this model type
    pub fn validate_proto(&self, model: &ModelProto) -> Result<()> {
        match self {
            ModelType::CNN => {
                if !is_cnn(model)? {
                    bail!("Model is not a valid CNN architecture");
                }
            }
            ModelType::MLP => {
                if !is_mlp(model)? {
                    bail!("Model is not a valid MLP architecture");
                }
            }
        }
        Ok(())
    }

    pub fn from_onnx(filepath: &str) -> Result<ModelType> {
        let model = load_proto_from_path(filepath)?;
        let is_mlp = is_mlp(&model);
        if is_mlp.is_ok() {
            return Ok(ModelType::MLP);
        }
        let is_cnn = is_cnn(&model);
        if is_cnn.is_ok() {
            return Ok(ModelType::CNN);
        }
        bail!(
            "Model is not a valid MLP or CNN architecture: not mlp: {} and not cnn: {}",
            is_mlp.unwrap_err(),
            is_cnn.unwrap_err()
        )
    }
}

/// Unified model loading function that handles both MLP and CNN models
pub fn load_float_model(model: &ModelProto) -> Result<Model<f32>> {
    let model = onnx::from_proto(model)?;
    model.describe();
    Ok(model)
}

// Module for caching downloaded files
#[cfg(test)]
pub mod file_cache {
    use anyhow::{Context as _, anyhow, bail};
    use hex;
    use reqwest;
    use sha2::{Digest, Sha256};
    use std::{
        fs::{self, File},
        io::{ErrorKind, Write},
        path::{Path, PathBuf},
        sync::LazyLock,
        thread,
        time::Duration,
    };

    // Directory to store cached files.
    static CACHE_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
        let dir = PathBuf::from("target").join("test_assets_cache");
        if !dir.exists() {
            fs::create_dir_all(&dir).expect("Failed to create cache directory for test assets");
        }
        dir
    });

    fn generate_filename_from_url(url: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(url.as_bytes());
        let result = hasher.finalize();
        // Append a common extension or a marker for GGUF if all files are such
        format!("{}.gguf", hex::encode(result))
    }

    struct FileLockGuard {
        path: PathBuf,
        acquired: bool,
    }

    impl FileLockGuard {
        fn acquire(path: &Path) -> anyhow::Result<Self> {
            match File::options().write(true).create_new(true).open(path) {
                Ok(_) => Ok(FileLockGuard {
                    path: path.to_path_buf(),
                    acquired: true,
                }),
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                    // Lock already exists, we didn't acquire it.
                    Ok(FileLockGuard {
                        path: path.to_path_buf(),
                        acquired: false,
                    })
                }
                Err(e) => {
                    Err(anyhow!(e)
                        .context(format!("Failed to create lock file {}", path.display())))
                }
            }
        }

        #[inline]
        fn is_acquired(&self) -> bool {
            self.acquired
        }
    }

    impl Drop for FileLockGuard {
        fn drop(&mut self) {
            if self.acquired {
                if let Err(e) = fs::remove_file(&self.path) {
                    // Log error, but don't panic in drop.
                    eprintln!(
                        "Warning: Failed to remove lock file {}: {}",
                        self.path.display(),
                        e
                    );
                }
            }
        }
    }

    pub fn ensure_downloaded(url: &str) -> anyhow::Result<PathBuf> {
        let base_filename = generate_filename_from_url(url); // e.g., hash.gguf
        let local_file_path = CACHE_DIR.join(&base_filename);
        let lock_file_path = CACHE_DIR.join(format!("{}.lock", base_filename));

        const MAX_RETRIES: u32 = 60; // Approx 60 seconds total timeout
        const RETRY_DELAY_MS: u64 = 1000;

        for attempt in 0..MAX_RETRIES {
            // Check 1: File exists and no lock. This is the ideal fast path.
            if local_file_path.exists() && !lock_file_path.exists() {
                return Ok(local_file_path);
            }

            // Check 2: Lock file exists. Someone else might be working or left a stale lock.
            if lock_file_path.exists() {
                if attempt == MAX_RETRIES - 1 {
                    // Last attempt, if lock is still there but main file appeared, warn and return.
                    if local_file_path.exists() {
                        eprintln!(
                            "Warning: Lock file {} still exists, but target file {} is present. Proceeding with cached file.",
                            lock_file_path.display(),
                            local_file_path.display()
                        );
                        return Ok(local_file_path);
                    }
                    bail!(
                        "Lock file {} persisted after {} retries. Target file {} not found.",
                        lock_file_path.display(),
                        MAX_RETRIES,
                        local_file_path.display()
                    );
                }
                thread::sleep(Duration::from_millis(RETRY_DELAY_MS));
                continue; // Go to next retry iteration
            }

            // Check 3: No lock file present. Attempt to acquire lock if data file is also missing.
            // (If data file exists here, and no lock, Check 1 would have caught it).
            if !local_file_path.exists() {
                let lock_guard = FileLockGuard::acquire(&lock_file_path)?;

                if lock_guard.is_acquired() {
                    // We got the lock. Critical section starts.
                    // Re-check: did another thread create the file *just* before we got the lock?
                    if local_file_path.exists() {
                        // Yes, file now exists. No need to download. Guard will release lock.
                        return Ok(local_file_path);
                    }

                    println!(
                        "Acquired lock for {}. Downloading {} to {}...",
                        base_filename,
                        url,
                        local_file_path.display()
                    );

                    let temp_download_path =
                        CACHE_DIR.join(format!("{}.tmp_download", base_filename));

                    // Perform the download. Lock_guard ensures lock removal on success or panic/error.
                    match (|| -> anyhow::Result<()> {
                        let response = reqwest::blocking::get(url)
                            .with_context(|| format!("Download: Failed to GET URL: {}", url))?;

                        if !response.status().is_success() {
                            bail!(
                                "Download: Failed for URL: {}. Server status: {}",
                                url,
                                response.status()
                            );
                        }

                        let mut dest_file =
                            File::create(&temp_download_path).with_context(|| {
                                format!(
                                    "Download: Failed to create temporary file: {}",
                                    temp_download_path.display()
                                )
                            })?;

                        let content = response.bytes().with_context(|| {
                            format!("Download: Failed to read response bytes from URL: {}", url)
                        })?;

                        dest_file.write_all(&content).with_context(|| {
                            format!(
                                "Download: Failed to write content to temporary file: {}",
                                temp_download_path.display()
                            )
                        })?;

                        fs::rename(&temp_download_path, &local_file_path).with_context(|| {
                            format!(
                                "Download: Failed to move temp file {} to final location {}",
                                temp_download_path.display(),
                                local_file_path.display()
                            )
                        })?;
                        Ok(())
                    })() {
                        Ok(_) => {
                            println!(
                                "Successfully downloaded and cached {} to {}",
                                url,
                                local_file_path.display()
                            );
                            // lock_guard will release the lock.
                            return Ok(local_file_path);
                        }
                        Err(e) => {
                            // Download or rename failed. Clean up temp file if it exists.
                            if temp_download_path.exists() {
                                if let Err(remove_err) = fs::remove_file(&temp_download_path) {
                                    eprintln!(
                                        "Error: Failed to remove temporary download file {}: {}",
                                        temp_download_path.display(),
                                        remove_err
                                    );
                                }
                            }
                            // lock_guard will release the lock. Propagate the error.
                            return Err(e);
                        }
                    }
                } else {
                    // Lock acquisition failed (lock_path was created by another thread/process
                    // between our check and our attempt to create it).
                    // Sleep briefly and let the loop retry.
                    thread::sleep(Duration::from_millis(RETRY_DELAY_MS / 2)); // Shorter sleep
                    continue;
                }
            }
            // If local_file_path.exists() but we didn't hit Check 1 (because lock_path also existed
            // or appeared), the loop will continue, sleep if lock_path is still there, and re-evaluate.
        }

        bail!(
            "Failed to ensure file {} (from URL {}) is downloaded after {} retries. Last state: file exists: {}, lock exists: {}.",
            local_file_path.display(),
            url,
            MAX_RETRIES,
            local_file_path.exists(),
            lock_file_path.exists()
        );
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use crate::{
        Context, Prover, ScalingFactor, init_test_logging_default, quantization::InferenceObserver,
        testing::Pcs, verify,
    };
    use ff_ext::GoldilocksExt2;
    use tracing::info;
    use transcript::BasicTranscript;

    type F = GoldilocksExt2;

    // cargo test --release --package zkml -- onnx_parse::tests::test_tract --nocapture

    #[test]
    fn test_load_mlp() {
        let filepath = "assets/scripts/MLP/mlp-iris-01.onnx";
        let result = FloatOnnxLoader::new(&filepath)
            .with_model_type(ModelType::MLP)
            .build();

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());
    }

    #[test]
    fn test_mlp_model_run() {
        init_test_logging_default();
        let filepath = "assets/scripts/MLP/mlp-iris-01.onnx";
        let (model, md) = FloatOnnxLoader::new(&filepath)
            .with_model_type(ModelType::MLP)
            .build()
            .unwrap();
        let input =
            crate::tensor::Tensor::<f32>::random(&model.input_shapes()[0]).quantize(&md.input[0]);
        let input = model.prepare_inputs(vec![input]).unwrap();
        let trace = model.run::<F>(&input).unwrap();
        println!("Result: {:?}", trace.outputs());
    }

    #[test]
    fn test_quantize() {
        let input: [f32; 2] = [0.09039914, -0.07716653];
        let scaling = ScalingFactor::from_span(1.0, -1.0, None);
        println!("Result: {} => {:?}", input[0], scaling.quantize(&input[0]));
        println!("Result: {} => {:?}", input[1], scaling.quantize(&input[0]));
        println!("Result: {} => {:?}", 0, scaling.quantize(&0.0));
        println!("Result: {} => {:?}", -1.0, scaling.quantize(&-1.0));
        println!("Result: {} => {:?}", 1.0, scaling.quantize(&1.0));
    }
    #[test]
    #[ignore]
    fn test_covid_cnn() {
        let subscriber = tracing_subscriber::fmt::Subscriber::builder()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("Failed to set global subscriber");

        let filepath = "assets/scripts/covid/cnn-covid.onnx";
        let result = FloatOnnxLoader::new(&filepath)
            .with_model_type(ModelType::CNN)
            .build();

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());

        info!("CREAting random tensor input");
        let (model, md) = result.unwrap();
        let inputs = model
            .unpadded_input_shapes()
            .into_iter()
            .zip(&md.input)
            .map(|(shape, s)| crate::tensor::Tensor::<f32>::random(&shape).quantize(s))
            .collect();
        let input = model.prepare_inputs(inputs).unwrap();
        info!("RUNNING MODEL...");
        let trace = model.run::<F>(&input).unwrap();
        info!("RUNNING MODEL DONE...");
        println!("Result: {:?}", trace.outputs());

        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"m2vec");
        info!("GENERATING CONTEXT...");
        let ctx = Context::<GoldilocksExt2, Pcs<GoldilocksExt2>>::generate(&model, None, None)
            .expect("Unable to generate context");
        info!("GENERATING CONTEXT DONE...");
        let io = trace.to_verifier_io();
        info!("GENERATING Proof...");
        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>, _> =
            Prover::new(&ctx, &mut tr);
        let proof = prover.prove(trace).expect("unable to generate proof");
        info!("GENERATING Proof DONE...");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");

        verify::<_, _, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_is_cnn() {
        let filepath = "assets/scripts/CNN/cnn-cifar-01.onnx";
        let result = is_cnn(&load_proto_from_path(&filepath).unwrap());

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());
    }
    #[test]
    fn test_load_cnn() {
        init_test_logging_default();
        let filepath = "assets/scripts/CNN/cnn-cifar-01.onnx";
        ModelType::CNN.validate_file(filepath).unwrap();
        let result =
            FloatOnnxLoader::new_with_scaling_strategy(&filepath, InferenceObserver::new())
                .with_model_type(ModelType::CNN)
                .build();

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());

        let (model, md) = result.unwrap();
        model.describe();
        let native_input = model
            .unpadded_input_shapes()
            .into_iter()
            .zip(&md.input)
            .map(|(shape, s)| crate::tensor::Tensor::<f32>::random(&shape).quantize(s))
            .collect();
        let input = model.prepare_inputs(native_input).unwrap();
        let trace = model.run::<F>(&input).unwrap();
        println!("Result: {:?}", trace.outputs());

        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"m2vec");
        let ctx = Context::<GoldilocksExt2, Pcs<GoldilocksExt2>>::generate(&model, None, None)
            .expect("Unable to generate context");

        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>, _> =
            Prover::new(&ctx, &mut tr);
        let io = trace.to_verifier_io();
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");
        verify::<_, _, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_tract() {
        let filepath = "assets/scripts/CNN/cnn-cifar-01.onnx";
        let model = tract_onnx::onnx()
            .model_for_path(filepath)
            .map_err(|e| Error::msg(format!("Failed to load model: {:?}", e)))
            .unwrap();
        for symbol in model.symbols.all_symbols().iter() {
            println!("symbol: {:?}", symbol);
        }
        let opt = model.into_typed().unwrap();

        let eval_order = opt.eval_order().unwrap();
        eval_order.into_iter().for_each(|id| {
            let node = opt.node(id);
            let outputs = &node.outputs;
            for (i, output) in outputs.iter().enumerate() {
                println!(
                    "Cluttered Node: {},  Output {} shape: {:?}",
                    node,
                    i,
                    output.fact.shape.dims()
                );
            }
            // for node_input in &node.inputs {
            //     let label = opt.outlet_label(*node_input);
            //     println!("Node input label: {:?}", label);
            // }
        });

        let opt = opt.into_decluttered().unwrap();

        let eval_order = opt.eval_order().unwrap();

        eval_order.into_iter().for_each(|id| {
            let node = opt.node(id);
            let outputs = &node.outputs;

            for (i, output) in outputs.iter().enumerate() {
                println!(
                    "Node {}: {},  Output {} shape: {:?}",
                    id,
                    node,
                    i,
                    output.fact.shape.dims()
                );
            }
            // for node_input in &node.inputs {
            //     let label = opt.outlet_label(*node_input);
            //     println!("Node input label: {:?}", label);
            // }
        });

        let mut values = SymbolValues::default();
        let symbol = opt.sym("batch_size");
        values.set(&symbol, 1);

        let opt = opt.concretize_dims(&values).unwrap();

        // let eval_order = opt.eval_order().unwrap();

        // eval_order.into_iter().for_each(|id| {
        //     let node = opt.node(id);
        //     let outputs = &node.outputs;
        //     for (i, output) in outputs.iter().enumerate() {
        //         println!(
        //             "Node: {},  Output {} shape: {:?}",
        //             node,
        //             i,
        //             output.fact.shape.dims()
        //         );
        //     }
        // });

        let plan = SimplePlan::new(opt).unwrap();

        for node_id in plan.order_without_consts() {
            let node = plan.model().node(*node_id);
            println!(
                "planned node {}:{}: input {:?} -> op{:?}",
                node_id,
                node.name,
                node.inputs,
                node.op()
            );
        }
    }
}
