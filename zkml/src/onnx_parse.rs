use crate::{
    Element,
    layers::{
        Layer,
        activation::{Activation, Relu},
        convolution::{Convolution, conv2d_shape},
        dense::Dense,
        pooling::{MAXPOOL2D_KERNEL_SIZE, Maxpool2D, Pooling, maxpool2d_shape},
        reshape::Reshape,
    },
    padding::pad_model,
    quantization::{AbsoluteMax, ModelMetadata, ScalingStrategy},
};
use anyhow::{Context, Error, Result, bail, ensure};
use itertools::Itertools;
use std::{collections::HashMap, i8, time::Instant};
use tracing::{debug, info, trace, warn};
use tract_onnx::{
    pb::{GraphProto, NodeProto},
    prelude::*,
};

use tract_onnx::pb::{
    tensor_shape_proto::dimension::Value::{DimParam, DimValue},
    type_proto::Value,
};

use crate::model::Model;

/// Utility struct for loading a onnx model with float weights and producing a quantized model
/// that can be used for inference and proving.
#[derive(Debug)]
pub struct FloatOnnxLoader {
    model_path: String,
    scaling_strategy: Box<dyn ScalingStrategy>,
    model_type: Option<ModelType>,
    keep_float: bool,
}

impl FloatOnnxLoader {
    pub fn new(model_path: &str) -> Self {
        Self {
            model_path: model_path.to_string(),
            scaling_strategy: Box::new(AbsoluteMax::new()),
            model_type: None,
            keep_float: false,
        }
    }
    pub fn with_scaling_strategy(mut self, scaling_strategy: Box<dyn ScalingStrategy>) -> Self {
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

// Given serialized data and its tract DatumType, build a tract tensor.
fn create_tensor(shape: Vec<usize>, dt: DatumType, data: &[u8]) -> TractResult<Tensor> {
    unsafe {
        match dt {
            DatumType::U8 => Tensor::from_raw::<u8>(&shape, data),
            DatumType::U16 => Tensor::from_raw::<u16>(&shape, data),
            DatumType::U32 => Tensor::from_raw::<u32>(&shape, data),
            DatumType::U64 => Tensor::from_raw::<u64>(&shape, data),
            DatumType::I8 => Tensor::from_raw::<i8>(&shape, data),
            DatumType::I16 => Tensor::from_raw::<i16>(&shape, data),
            DatumType::I32 => Tensor::from_raw::<i32>(&shape, data),
            DatumType::I64 => Tensor::from_raw::<i64>(&shape, data),
            DatumType::F16 => Tensor::from_raw::<f16>(&shape, data),
            DatumType::F32 => Tensor::from_raw::<f32>(&shape, data),
            DatumType::F64 => Tensor::from_raw::<f64>(&shape, data),
            DatumType::Bool => Ok(Tensor::from_raw::<u8>(&shape, data)?
                .into_array::<u8>()?
                .mapv(|x| x != 0)
                .into()),
            _ => unimplemented!("create_tensor: Failed"),
        }
    }
}

fn is_mlp(filepath: &str) -> Result<bool> {
    let is_mlp = true;
    let mut prev_was_gemm_or_matmul = false;

    let model = tract_onnx::onnx()
        .proto_model_for_path(filepath)
        .map_err(|e| Error::msg(format!("Failed to load model: {:?}", e)))?;
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
        .map_err(|e| Error::msg(format!("Failed to load model: {:?}", e)))?;

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
                "Operator '{}' unsupported, yet.",
                op_type
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

fn model_input_shape(graph: &GraphProto) -> Vec<usize> {
    let mut input_shape: Vec<usize> = Vec::new();
    for input in graph.input.iter() {
        let fact = input.r#type.as_ref().unwrap().value.as_ref().unwrap();
        match fact {
            Value::TensorType(result) => match &result.shape {
                Some(shape) => {
                    for dim in &shape.dim {
                        match &dim.value {
                            Some(value) => match value {
                                DimValue(size) => {
                                    input_shape.push(*size as usize);
                                }
                                DimParam(_param) => {
                                    input_shape.push(1 as usize);
                                }
                            },
                            None => {
                                warn!("  Dimension not specified");
                            }
                        }
                    }
                }
                None => {
                    warn!("No shape information available");
                }
            },
        }
    }
    input_shape
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
    let model_type = ModelType::from_onnx(filepath).context("can't prove unknown model:")?;
    info!("Model type detected: {:?}", model_type);

    // Continue with model loading
    let model = tract_onnx::onnx()
        .proto_model_for_path(filepath)
        .map_err(|e| Error::msg(format!("Failed to load model: {:?}", e)))?;

    let graph = model.graph.unwrap();

    let mut input_shape = model_input_shape(&graph);

    assert!(
        input_shape[0] == 1,
        "First dimension of the CNNs or MLP's input should 1."
    );
    // We force the input shape to be for a single inference and not a batch inference.
    input_shape.remove(0);
    if model_type == ModelType::CNN {
        assert!(input_shape.len() == 3);
    } else {
        assert!(input_shape.len() == 1);
    }

    let mut input_shape_og = input_shape.clone();
    let mut initializers: HashMap<String, Tensor> = HashMap::new();
    for item in graph.initializer {
        let dt = tract_onnx::pb::tensor_proto::DataType::from_i32(item.data_type)
            .context("can't load from onnx")?
            .try_into()?;
        let shape: Vec<usize> = item.dims.iter().map(|&i| i as usize).collect();
        let value = create_tensor(shape, dt, &item.raw_data).unwrap();
        let key = item.name.to_string();
        initializers.insert(key, value);
    }
    debug!(
        "Initializers: TENSOR NAMES: {:?}",
        initializers.keys().collect_vec()
    );
    let mut layers: Vec<Layer<f32>> = Vec::with_capacity(graph.node.len());
    // we need to keep track of the last shape because when we pad to next power of two one layer, we need to make sure
    // the next layer's expected input matches.
    info!("load_model:BEFORE loop of graph nodes extraction");
    for (i, node) in graph.node.iter().enumerate() {
        match node.op_type.as_str() {
            op if LINEAR_ALG.contains(&op) => {
                let WeightBiasInfo { weights, bias } = fetch_weight_and_bias(node, &initializers)?;
                ensure!(bias.get_shape().len() == 1, "bias is not a vector");
                input_shape_og = vec![weights.get_shape()[0]];
                let nrows = weights.get_shape()[0];
                ensure!(
                    bias.get_data().len() == nrows,
                    "bias length {} does not match matrix width {}",
                    bias.get_data().len(),
                    nrows
                );
                ensure!(
                    bias.get_shape()[0] == nrows,
                    "bias length {} does not match matrix width {}",
                    bias.get_shape()[0],
                    nrows
                );
                debug!("layer idx {} -> final shape {:?}", i, weights.get_shape());
                layers.push(Layer::Dense(Dense::new_from_weights(weights, bias)));
            }
            op if ACTIVATION.contains(&op) => {
                let layer = Layer::Activation(Activation::Relu(Relu::new()));
                layers.push(layer);
            }
            op if CONVOLUTION.contains(&op) => {
                let now = Instant::now();
                let _ = fetch_conv2d_attributes(node)?;
                let WeightBiasInfo { weights, bias } = fetch_weight_and_bias(node, &initializers)?;
                input_shape_og = safe_conv2d_shape(&input_shape_og, &weights.get_shape())?;
                let weight_shape = weights.get_shape();
                // Perform basic sanity checks on the tensor dimensions
                let shape_test = check_filter(&weight_shape);
                assert!(shape_test.is_ok(), "Failed: {:?}", shape_test.unwrap_err());
                assert!(
                    weight_shape[0] == bias.get_shape()[0],
                    "Bias length doesn't match filter shape"
                );

                let layer = Layer::Convolution(Convolution::new(weights, bias));
                layers.push(layer);
                debug!("EXTRACTIONG conv2d time: {:?}", now.elapsed());
            }
            op if DOWNSAMPLING.contains(&op) => {
                input_shape_og = safe_maxpool2d_shape(&input_shape_og)?;
                let _ = fetch_maxpool_attributes(node)?;
                let layer = Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default()));
                layers.push(layer);
            }
            op if RESHAPE.contains(&op) => {
                layers.push(Layer::Reshape(Reshape));
                input_shape_og = vec![input_shape_og.iter().product()];
            }
            _ => bail!("Unsupported operation"),
        };
        println!(
            "{}. {}'s output shape: {:?}",
            i + 1,
            node.op_type.as_str(),
            // input_shape_padded,
            input_shape_og
        );
    }

    info!("load_model:AFTER loop of graph nodes extraction");

    // Create and return the model
    let mut model = Model::new(&input_shape);
    for layer in layers {
        debug!("Added the layer: {}", layer.describe());
        model.add_layer(layer);
    }
    model.describe();
    Ok(model)
}

/// Common function to extract tensor data from a node
///
/// This function handles finding the tensor by name, applying alpha/beta multipliers,
/// and extracting the raw f32 data and shape for further processing.
fn extract_tensor_f32_data(
    weight_or_bias: &str,
    node: &NodeProto,
    initializers: &HashMap<String, Tensor>,
) -> Result<Option<(Vec<f32>, Vec<usize>)>> {
    ensure!(weight_or_bias == "weight" || weight_or_bias == "bias");
    let selector = match weight_or_bias {
        "weight" => vec!["weight", "MatMul"],
        "bias" => vec!["bias"],
        _ => bail!("Invalid weight_or_bias: {}", weight_or_bias),
    };
    let is_good_selector = |name: &str| -> bool { selector.iter().any(|s| name.contains(s)) };

    // Handle multipliers (alpha/beta) from Gemm operations
    let mut alpha_or_beta: f32 = 1.0;
    if node.op_type == "Gemm" || node.op_type == "MatMul" {
        let result = node
            .attribute
            .iter()
            .filter(|x| {
                x.name.contains(match weight_or_bias {
                    "weight" | "MatMul" => "alpha",
                    _ => "beta",
                })
            })
            .map(|x| x.f)
            .collect_vec();

        if !result.is_empty() {
            alpha_or_beta = result[0];
        }
    }

    // Find tensor by name pattern
    let tensor_vec = node
        .input
        .iter()
        .filter(|x| is_good_selector(x))
        .filter_map(|key| initializers.get(key).cloned())
        .collect_vec();

    trace!(
        "node input: {:?} - match {:?}-> tensor_vec len() {:?}",
        node.input,
        node.input
            .iter()
            .map(|name| initializers.get(name))
            .collect::<Vec<_>>(),
        tensor_vec.len()
    );

    // If no matching tensor found, return None
    if tensor_vec.is_empty() {
        return Ok(None);
    }

    // Get the tensor data
    let tensor_t = &tensor_vec[0];
    let mut tensor_shape = tensor_t.shape().to_vec();
    let tensor_data = tensor_t.as_slice::<f32>()?.to_vec();

    let tensor_t_f32 = if node.op_type == "MatMul" && weight_or_bias == "weight" {
        tensor_shape.reverse();
        let (m, n) = (tensor_shape[0], tensor_shape[1]);
        let mut transposed_data = vec![0.0; tensor_data.len()];

        // Transpose the data matrix
        for i in 0..m {
            for j in 0..n {
                transposed_data[i * n + j] = tensor_data[j * m + i] * alpha_or_beta;
            }
        }
        transposed_data
    } else {
        tensor_data.into_iter().map(|x| x * alpha_or_beta).collect()
    };

    Ok(Some((tensor_t_f32, tensor_shape)))
}

struct WeightBiasInfo {
    weights: crate::Tensor<f32>,
    bias: crate::Tensor<f32>,
}

fn fetch_weight_and_bias(
    node: &NodeProto,
    initializers: &HashMap<String, Tensor>,
) -> Result<WeightBiasInfo> {
    // Extract the tensor data using the common function
    let (data, shape) = match extract_tensor_f32_data("weight", node, initializers)? {
        Some(data) => data,
        None => bail!("No weight tensor found for node {}", node.name),
    };
    let (bias, bias_shape) = match extract_tensor_f32_data("bias", node, initializers)? {
        Some(data) => data,
        None => {
            warn!("No bias tensor found for node {}", node.name);
            (vec![0.0; shape[0]], vec![shape[0]])
        }
    };

    let weights = crate::Tensor::new(shape, data);
    let bias = crate::Tensor::new(bias_shape, bias);
    Ok(WeightBiasInfo { weights, bias })
}

/// Get the conv2d attributes and assert if supported by DeepProve
fn fetch_conv2d_attributes(node: &NodeProto) -> Result<()> {
    let get_attr = |name: &str| -> Vec<i64> {
        node.attribute
            .iter()
            .find(|x| x.name.contains(name))
            .map_or_else(Vec::new, |x| x.ints.clone())
    };

    let (strides, pads, _kernel_shape, dilations) = (
        get_attr("strides"),
        get_attr("pads"),
        get_attr("kernel_shape"),
        get_attr("dilations"),
    );

    assert!(strides.iter().all(|&x| x == 1), "Strides must be {}", 1);
    assert!(pads.iter().all(|&x| x == 0), "Padding must be 0s");
    assert!(
        dilations.iter().all(|&x| x == 1),
        "Dilations shape must be 1"
    );

    Ok(())
}

/// Get the maxpool attributes and assert if supported by DeepProve
fn fetch_maxpool_attributes(node: &NodeProto) -> Result<()> {
    let get_attr = |name: &str| -> Vec<i64> {
        node.attribute
            .iter()
            .find(|x| x.name.contains(name))
            .map_or_else(Vec::new, |x| x.ints.clone())
    };

    let (strides, pads, kernel_shape, dilations) = (
        get_attr("strides"),
        get_attr("pads"),
        get_attr("kernel_shape"),
        get_attr("dilations"),
    );

    // println!(
    //     "strides: {:?}, pads: {:?}, kernel_shape: {:?}, dilation: {:?}",
    //     strides, pads, kernel_shape, dilations
    // );

    let expected_value: i64 = MAXPOOL2D_KERNEL_SIZE.try_into()?;

    assert!(
        strides.iter().all(|&x| x == expected_value),
        "Strides must be {}",
        expected_value
    );
    assert!(pads.iter().all(|&x| x == 0), "Padding must be 0s");
    assert!(
        kernel_shape.iter().all(|&x| x == expected_value),
        "Kernel shape must be {}",
        expected_value
    );
    assert!(
        dilations.iter().all(|&x| x == 1),
        "Dilations shape must be 1"
    );

    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;

    use crate::{
        Context, IO, Prover, ScalingFactor,
        commit::Pcs,
        init_test_logging,
        quantization::{InferenceObserver, TensorFielder},
        verify,
    };
    use goldilocks::GoldilocksExt2;
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
        // let filepath = "assets/scripts/MLP/mlp-iris-01.onnx";
        let filepath = "assets/scripts/MLP/mlp-iris-01.onnx";
        let (model, md) = FloatOnnxLoader::new(&filepath)
            .with_model_type(ModelType::MLP)
            .build()
            .unwrap();
        let input =
            crate::tensor::Tensor::<f32>::random(&vec![model.input_shape()[0]]).quantize(&md.input);
        let input = model.prepare_input(input);
        let trace = model.run::<F>(input.clone()).unwrap();
        println!("Result: {:?}", trace.final_output());
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
        let input = crate::tensor::Tensor::<f32>::random(&model.input_shape()).quantize(&md.input);
        info!("random input tensor CREATED : {:?}", input.get_shape());
        let input = model.prepare_input(input);
        info!("RUNNING MODEL...");
        let trace = model.run::<F>(input.clone()).unwrap();
        info!("RUNNING MODEL DONE...");
        // println!("Result: {:?}", trace.final_output());

        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"m2vec");
        info!("GENERATING CONTEXT...");
        let ctx = Context::<GoldilocksExt2, Pcs<F>>::generate(&model, Some(input.get_shape()))
            .expect("Unable to generate context");
        info!("GENERATING CONTEXT DONE...");
        let output = trace.final_output().clone();
        info!("GENERATING Proof...");
        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>, _> =
            Prover::new(&ctx, &mut tr);
        let proof = prover.prove(trace).expect("unable to generate proof");
        info!("GENERATING Proof DONE...");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");
        let io = IO::new(input.to_fields(), output.to_fields());
        verify::<_, _, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_is_cnn() {
        let filepath = "assets/scripts/CNN/cnn-cifar-01.onnx";
        let result = is_cnn(&filepath);

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());
    }

    #[test]
    fn test_load_cnn() {
        let filepath = "assets/scripts/CNN/cnn-cifar-01.onnx";
        // let filepath = "bench/model.onnx";
        ModelType::CNN.validate(filepath).unwrap();
        let result = FloatOnnxLoader::new(&filepath)
            .with_model_type(ModelType::CNN)
            .with_scaling_strategy(Box::new(InferenceObserver::new()))
            .build();

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());

        let (model, md) = result.unwrap();
        let native_input =
            crate::tensor::Tensor::<f32>::random(&model.unpadded_input_shape()).quantize(&md.input);
        let input = model.prepare_input(native_input);
        let trace = model.run::<F>(input.clone()).unwrap();
        println!("Result: {:?}", trace.final_output());

        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"m2vec");
        let ctx = Context::<GoldilocksExt2, Pcs<F>>::generate(&model, Some(input.get_shape()))
            .expect("Unable to generate context");
        let output = trace.final_output().clone();

        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>, _> =
            Prover::new(&ctx, &mut tr);
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");
        let io = IO::new(input.to_fields(), output.to_fields());
        verify::<_, _, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
    }
}
