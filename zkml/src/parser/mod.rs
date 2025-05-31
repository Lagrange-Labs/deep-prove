pub mod gguf;
pub mod onnx;
pub mod json;

use crate::{
    Element,
    layers::{convolution::conv2d_shape, pooling::maxpool2d_shape},
    model::Model,
    padding::pad_model,
    parser::onnx::from_path,
    quantization::{AbsoluteMax, ModelMetadata, ScalingStrategy},
};
use anyhow::{Context, Error, Result, bail, ensure};
use tract_onnx::prelude::*;

/// Utility struct for loading a onnx model with float weights and producing a quantized model
/// that can be used for inference and proving.
#[derive(Debug)]
pub struct FloatOnnxLoader<S: ScalingStrategy> {
    model_path: String,
    scaling_strategy: S,
    model_type: Option<ModelType>,
    keep_float: bool,
}

pub type DefaultFloatOnnxLoader = FloatOnnxLoader<AbsoluteMax>;

impl DefaultFloatOnnxLoader {
    pub fn new(model_path: &str) -> Self {
        Self::new_with_scaling_strategy(model_path, AbsoluteMax::new())
    }
}

impl<S: ScalingStrategy> FloatOnnxLoader<S> {
    pub fn new_with_scaling_strategy(model_path: &str, scaling_strategy: S) -> Self {
        Self {
            model_path: model_path.to_string(),
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
        if let Some(model_type) = self.model_type {
            model_type.validate(&self.model_path)?;
        }
        let float_model = load_float_model(&self.model_path)?;
        println!("Input shape: {:?}", float_model.input_shapes());
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
// Supported operators
const ACTIVATION: [&str; 2] = ["Relu", "Sigmoid"];
const CONVOLUTION: [&str; 1] = ["Conv"];
const DOWNSAMPLING: [&str; 1] = ["MaxPool"];
const LINEAR_ALG: [&str; 2] = ["Gemm", "MatMul"];
const RESHAPE: [&str; 2] = ["Flatten", "Reshape"];

fn is_mlp(filepath: &str) -> Result<bool> {
    let is_mlp = true;
    let mut prev_was_gemm_or_matmul = false;

    let model = tract_onnx::onnx()
        .proto_model_for_path(filepath)
        .map_err(|e| Error::msg(format!("Failed to load model: {e:?}")))?;
    let graph = model.graph.unwrap();

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

    Ok(is_mlp)
}

fn is_cnn(filepath: &str) -> Result<bool> {
    let mut is_cnn = true;
    let mut found_lin = false;

    // Load the ONNX model
    let model = tract_onnx::onnx()
        .proto_model_for_path(filepath)
        .map_err(|e| Error::msg(format!("Failed to load model: {e:?}")))?;

    let graph = model.graph.unwrap();
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
            break;
        }
        previous_op = op_type;
    }

    Ok(is_cnn)
}

pub fn safe_conv2d_shape(input_shape: &[usize], filter_shape: &[usize]) -> Result<Vec<usize>> {
    let result = check_filter(filter_shape);
    assert!(result.is_ok(), "conv2d: Failed {:?}", result.unwrap_err());

    check_cnn_input(input_shape).context("conv2d: invalid input shape")?;

    Ok(conv2d_shape(input_shape, filter_shape))
}

pub fn check_filter(filter_shape: &[usize]) -> Result<()> {
    ensure!(filter_shape.len() == 4, "Filter should be 4D tensor.");
    ensure!(
        filter_shape[2] == filter_shape[3],
        "Filter should be square."
    );
    Ok(())
}

pub fn check_cnn_input(input_shape: &[usize]) -> Result<()> {
    ensure!(input_shape.len() == 3, "input should be 3d tensor");
    ensure!(input_shape[1] == input_shape[2], "input should be square");
    Ok(())
}

pub fn safe_maxpool2d_shape(input_shape: &[usize]) -> Result<Vec<usize>> {
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
    pub fn validate(&self, filepath: &str) -> Result<()> {
        match self {
            ModelType::CNN => {
                if !is_cnn(filepath)? {
                    bail!("Model is not a valid CNN architecture");
                }
                Ok(())
            }
            ModelType::MLP => {
                if !is_mlp(filepath)? {
                    bail!("Model is not a valid MLP architecture");
                }
                Ok(())
            }
        }
    }
    pub fn from_onnx(filepath: &str) -> Result<ModelType> {
        let is_mlp = is_mlp(filepath);
        if is_mlp.is_ok() {
            return Ok(ModelType::MLP);
        }
        let is_cnn = is_cnn(filepath);
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
pub fn load_float_model(filepath: &str) -> Result<Model<f32>> {
    let model = from_path(filepath)?;
    model.describe();
    Ok(model)
}

#[cfg(test)]
mod tests {

    use super::*;

    use crate::{
        Context, Prover, ScalingFactor, init_test_logging, quantization::InferenceObserver, verify,
    };
    use goldilocks::GoldilocksExt2;
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
        init_test_logging();
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
        let ctx =
            Context::<GoldilocksExt2>::generate(&model, None).expect("Unable to generate context");
        info!("GENERATING CONTEXT DONE...");
        let io = trace.to_verifier_io();
        info!("GENERATING Proof...");
        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>> =
            Prover::new(&ctx, &mut tr);
        let proof = prover.prove(trace).expect("unable to generate proof");
        info!("GENERATING Proof DONE...");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");

        verify::<_, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_is_cnn() {
        let filepath = "assets/scripts/CNN/cnn-cifar-01.onnx";
        let result = is_cnn(&filepath);

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());
    }
    #[test]
    fn test_load_cnn() {
        init_test_logging();
        let filepath = "assets/scripts/CNN/cnn-cifar-01.onnx";
        ModelType::CNN.validate(filepath).unwrap();
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
        let ctx =
            Context::<GoldilocksExt2>::generate(&model, None).expect("Unable to generate context");

        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>> =
            Prover::new(&ctx, &mut tr);
        let io = trace.to_verifier_io();
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");
        verify::<_, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
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
