use crate::{convolution::Convolution, dense::Dense};
use anyhow::{Context, Error, Result, bail, ensure};
use goldilocks::GoldilocksExt2;
use itertools::Itertools;
use std::{collections::HashMap, i8, path::Path};
use tracing::{debug, info};
use tract_onnx::{
    pb::{GraphProto, NodeProto},
    prelude::*,
};

use tract_onnx::pb::{
    tensor_shape_proto::dimension::Value::{DimParam, DimValue},
    type_proto::Value,
};

type F = GoldilocksExt2;

use crate::{
    Element,
    activation::{Activation, Relu},
    model::{Layer, Model},
    pooling::{MAXPOOL2D_KERNEL_SIZE, Maxpool2D, Pooling},
    quantization::Quantizer,
};

// Supported operators
const ACTIVATION: [&str; 1] = ["Relu"];
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
                    debug!("Input tensor shape dimensions:");
                    for dim in &shape.dim {
                        match &dim.value {
                            Some(value) => match value {
                                DimValue(size) => {
                                    debug!("  Fixed dimension: {}", size);
                                    input_shape.push(*size as usize);
                                }
                                DimParam(param) => {
                                    debug!("  Symbolic dimension: {}", param);
                                    input_shape.push(1 as usize);
                                }
                            },
                            None => {
                                debug!("  Dimension not specified");
                            }
                        }
                    }
                }
                None => {
                    debug!("No shape information available");
                }
            },
        }
    }
    input_shape
}

fn check_filter(filter_shape: &[usize]) -> Result<bool> {
    let result = true;
    if !(filter_shape.len() == 4) {
        return Err(Error::msg(format!("Filter should be 4D tensor.")));
    }
    if !(filter_shape[2] == filter_shape[3]) {
        return Err(Error::msg(format!("Filter should be square.")));
    }
    Ok(result)
}

fn check_cnn_input(input_shape: &[usize]) -> Result<bool> {
    let result = true;
    if !(input_shape.len() == 3) {
        return Err(Error::msg(format!("Input should be 3D tensor.")));
    }
    if !(input_shape[1] == input_shape[2]) {
        return Err(Error::msg(format!("Input should be square.")));
    }
    Ok(result)
}

/// Assumes stride=1, padding=0, and dilation=1
/// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
fn conv2d_shape(input_shape: &[usize], filter_shape: &[usize]) -> Vec<usize> {
    let result = check_filter(filter_shape);
    assert!(result.is_ok(), "conv2d: Failed {:?}", result.unwrap_err());

    let result = check_cnn_input(input_shape);
    assert!(result.is_ok(), "conv2d: Failed {:?}", result.unwrap_err());

    let stride = 1usize;
    let padding = 0usize;
    let dilation = 1usize;

    let h_in = input_shape[2];
    let kernel = filter_shape[2];
    let h_out = (h_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
    vec![filter_shape[0], h_out, h_out]
}

/// Assumes kernel=2, stride=2, padding=0, and dilation=1
/// https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
fn maxpool2d_shape(input_shape: &[usize]) -> Vec<usize> {
    let result = check_cnn_input(input_shape);
    assert!(
        result.is_ok(),
        "maxpool2d: Failed {:?}",
        result.unwrap_err()
    );

    let stride = 2usize;
    let padding = 0usize;
    let kernel = 2usize;
    let dilation = 1usize;

    let d1 = input_shape[0];
    let d2 = (input_shape[1] + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;

    vec![d1, d2, d2]
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
pub fn load_model<Q: Quantizer<Element>>(filepath: &str) -> Result<Model> {
    // Validate that the model matches the expected type
    let model_type = ModelType::from_onnx(filepath).context("can't prove unknown model:")?;
    info!("Model type detected: {:?}", model_type);

    // Get global weight ranges first
    let (global_min, global_max) = analyze_model_weight_ranges(filepath)?;
    let global_max_abs = global_min.abs().max(global_max.abs());
    debug!(
        "Using global weight range for quantization: [{}, {}], max_abs={}",
        global_min, global_max, global_max_abs
    );

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
    input_shape.remove(0);
    if model_type == ModelType::CNN {
        assert!(input_shape.len() == 3);
    } else {
        assert!(input_shape.len() == 1);
    }

    let mut ignore_garbage_pad: Option<(Vec<usize>, Vec<usize>)> = None;
    let mut input_shape_og = input_shape.clone();
    let mut input_shape_padded = input_shape
        .iter()
        .map(|i| i.next_power_of_two())
        .collect_vec();
    debug!("Input shape: {:?}", input_shape);
    debug!("Padded input shape: {:?}", input_shape_padded);
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

    let mut layers: Vec<Layer> = Vec::new();
    // we need to keep track of the last shape because when we pad to next power of two one layer, we need to make sure
    // the next layer's expected input matches.
    for (i, node) in graph.node.iter().enumerate() {
        match node.op_type.as_str() {
            op if LINEAR_ALG.contains(&op) => {
                let mut weight = fetch_weight_bias_as_tensor::<Q>(
                    "weight",
                    node,
                    &initializers,
                    global_max_abs,
                )?;
                let bias =
                    fetch_weight_bias_as_tensor::<Q>("bias", node, &initializers, global_max_abs)?;
                ensure!(bias.get_shape().len() == 1, "bias is not a vector");
                input_shape_og = vec![weight.get_shape()[0]];
                let nrows = weight.get_shape()[0];
                ensure!(
                    bias.get_data().len() == nrows,
                    "bias length {} does not match matrix width {}",
                    bias.get_data().len(),
                    nrows
                );
                assert!(input_shape_padded.iter().all(|d| d.is_power_of_two()));
                assert!(input_shape_padded.len() == 1);

                let mut new_cols = weight.ncols_2d();
                if weight.ncols_2d() != input_shape_padded[0] {
                    if weight.ncols_2d() < input_shape_padded[0] {
                        new_cols = input_shape_padded[0];
                    } else {
                        // If we have too many columns, we can't shrink without losing information
                        panic!(
                            "Matrix has more columns ({}) than previous layer output size ({}).
                            Cannot shrink without losing information.",
                            weight.ncols_2d(),
                            input_shape_padded[0]
                        );
                    }
                }

                let ncols = new_cols.next_power_of_two();
                let nrows = weight.nrows_2d().next_power_of_two();
                // Pad to power of two dimensions

                if let Some(ref conv_shapes) = ignore_garbage_pad.clone() {
                    let conv_shape_og = conv_shapes.0.clone();
                    let conv_shape_pad = conv_shapes.1.clone();
                    weight = weight.pad_matrix_to_ignore_garbage(
                        &conv_shape_og,
                        &conv_shape_pad,
                        &vec![nrows, ncols],
                    );
                    ignore_garbage_pad = None;
                } else {
                    weight.reshape_to_fit_inplace_2d(vec![nrows, ncols]);
                }

                let bias = bias.pad_1d(nrows);
                input_shape_padded = vec![nrows];

                debug!("layer idx {} -> final shape {:?}", i, weight.get_shape());
                layers.push(Layer::Dense(Dense::new(weight, bias)));
            }
            op if ACTIVATION.contains(&op) => {
                assert!(input_shape_padded.iter().all(|d| d.is_power_of_two()));
                let layer = Layer::Activation(Activation::Relu(Relu::new()));
                layers.push(layer);
            }
            op if CONVOLUTION.contains(&op) => {
                let mut weight = fetch_weight_bias_as_tensor::<Q>(
                    "weight",
                    node,
                    &initializers,
                    global_max_abs,
                )?;
                let mut bias =
                    fetch_weight_bias_as_tensor::<Q>("bias", node, &initializers, global_max_abs)?;

                input_shape_og = conv2d_shape(&input_shape_og, &weight.get_shape());
                let weight_shape = weight.get_shape();
                // Perform basic sanity checks on the tensor dimensions
                let shape_test = check_filter(&weight_shape);
                assert!(shape_test.is_ok(), "Failed: {:?}", shape_test.unwrap_err());
                assert!(
                    weight_shape[0] == bias.get_shape()[0],
                    "Bias length doesn't match filter shape"
                );

                // Pad the tensors to the next power of two.
                weight = weight.pad_next_power_of_two();
                bias = bias.pad_next_power_of_two();

                // Make sure that input shape is already padded and is well formed
                assert!(input_shape_padded.iter().all(|d| d.is_power_of_two()));
                assert!(input_shape_padded.len() == 3);

                // Since we are doing an FFT based conv, we need to pad the last two dimensions of the filter to match the input.
                let weight_shape = weight.get_shape();
                let (filter_height, filter_weight) = (weight_shape[2], weight_shape[3]);
                let (input_height, input_weight) = (input_shape_padded[1], input_shape_padded[2]);

                assert!(
                    filter_height <= input_height && filter_weight <= input_weight,
                    "Filter dimensions have to be smaller than input dimensions"
                );

                // weight = weight.pad_last_two_dimensions(vec![input_height, input_weight]);

                // Filter need to know the shape of the input
                // weight.update_input_shape(&input_shape_padded);

                let dims = weight.get_shape();
                let weight = crate::tensors::Tensor::new_conv(
                    weight.get_shape(),
                    input_shape_padded.clone(),
                    weight.get_data().to_vec(),
                );
                // let dims = weight.dims(); // save the shape of the filter to compute the output shape

                let layer = Layer::Convolution(Convolution::new(weight, bias));
                // let layer = Layer::SchoolBookConvolution(Convolution::new(weight, _bias));

                layers.push(layer);

                let output_shape = conv2d_shape(&input_shape_padded, &dims);
                input_shape_padded = output_shape
                    .iter()
                    .map(|i| i.next_power_of_two())
                    .collect_vec();
            }
            op if DOWNSAMPLING.contains(&op) => {
                input_shape_og = maxpool2d_shape(&input_shape_og);
                // Make sure that input shape is already padded and is well formed
                assert!(input_shape_padded.iter().all(|d| d.is_power_of_two()));

                let _ = fetch_maxpool_attributes(node)?;
                let layer = Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default()));
                layers.push(layer);
                input_shape_padded = maxpool2d_shape(&input_shape_padded);
            }
            op if RESHAPE.contains(&op) => {
                ignore_garbage_pad = Some((input_shape_og.clone(), input_shape_padded.clone()));

                input_shape_og = vec![input_shape_og.iter().product()];
                assert!(input_shape_padded.iter().all(|d| d.is_power_of_two()));
                input_shape_padded = vec![input_shape_padded.iter().product()];
            }
            _ => bail!("Unsupported operation"),
        };
        debug!(
            "{}. {}'s output shape: {:?}. {:?}",
            i + 1,
            node.op_type.as_str(),
            input_shape_padded,
            input_shape_og
        );
    }

    // Create and return the model
    let mut model = Model::new();
    model.set_input_shape(input_shape);
    for layer in layers {
        debug!("Added the layer: {}", layer.describe());
        model.add_layer::<F>(layer);
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

    // Handle multipliers (alpha/beta) from Gemm operations
    let mut alpha_or_beta: f32 = 1.0;
    if node.op_type == "Gemm" {
        let result = node
            .attribute
            .iter()
            .filter(|x| {
                x.name.contains(match weight_or_bias {
                    "weight" => "alpha",
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
        .filter(|x| x.contains(weight_or_bias))
        .filter_map(|key| initializers.get(key).cloned())
        .collect_vec();

    // If no matching tensor found, return None
    if tensor_vec.is_empty() {
        return Ok(None);
    }

    // Get the tensor data
    let tensor_t = tensor_vec[0].clone();
    let tensor_shape = tensor_t.shape().to_vec();
    let tensor_t_f32 = tensor_t.as_slice::<f32>().unwrap().to_vec();

    // Apply alpha/beta multiplier
    let tensor_t_f32 = tensor_t_f32.iter().map(|x| x * alpha_or_beta).collect_vec();

    Ok(Some((tensor_t_f32, tensor_shape)))
}

/// Extracts the min and max values from a specific tensor in a node
fn extract_node_weight_range(
    weight_or_bias: &str,
    node: &NodeProto,
    initializers: &HashMap<String, Tensor>,
) -> Result<Option<(f32, f32)>> {
    // Extract the tensor data using the common function
    let (tensor_data, _) = match extract_tensor_f32_data(weight_or_bias, node, initializers)? {
        Some(data) => data,
        None => return Ok(None),
    };

    // Find min and max values
    let min_val = tensor_data
        .iter()
        .fold(f32::MAX, |min_so_far, &val| min_so_far.min(val));
    let max_val = tensor_data
        .iter()
        .fold(f32::MIN, |max_so_far, &val| max_so_far.max(val));

    Ok(Some((min_val, max_val)))
}

fn fetch_weight_bias_as_tensor<Q: Quantizer<Element>>(
    weight_or_bias: &str,
    node: &NodeProto,
    initializers: &HashMap<String, Tensor>,
    global_max_abs: f32,
) -> Result<crate::tensors::Tensor<Element>> {
    // Extract the tensor data using the common function
    let (tensor_data, tensor_shape) =
        match extract_tensor_f32_data(weight_or_bias, node, initializers)? {
            Some(data) => data,
            None => bail!("No {} tensor found for node {}", weight_or_bias, node.name),
        };

    // For debugging, calculate the local range
    let local_max_abs = tensor_data
        .iter()
        .fold(0.0f32, |max_so_far, &val| max_so_far.max(val.abs()));
    let min_val = tensor_data
        .iter()
        .fold(f32::MAX, |min_so_far, &val| min_so_far.min(val));
    let max_val = tensor_data
        .iter()
        .fold(f32::MIN, |max_so_far, &val| max_so_far.max(val));

    debug!(
        "Tensor {}: local range=[{}, {}], abs={}, using global_max_abs={}",
        weight_or_bias, min_val, max_val, local_max_abs, global_max_abs
    );

    // Quantize using the global max_abs
    let tensor_f = tensor_data
        .iter()
        .map(|x| Q::from_f32_unsafe_clamp(x, global_max_abs as f64))
        //.map(|x| Q::from_f32_unsafe_clamp(x, local_max_abs as f64))
        .collect_vec();

    let tensor_result = crate::tensors::Tensor::new(tensor_shape, tensor_f);

    Ok(tensor_result)
}

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

/// Analyzes all weights from supported layers (Dense and Conv2D)
/// and returns the global min and max values.
///
/// This is useful for determining quantization ranges for the entire model.
pub fn analyze_model_weight_ranges(filepath: &str) -> Result<(f32, f32)> {
    if !Path::new(filepath).exists() {
        return Err(Error::msg(format!("File '{}' does not exist", filepath)));
    }

    // Load the ONNX model
    let model = tract_onnx::onnx()
        .proto_model_for_path(filepath)
        .map_err(|e| Error::msg(format!("Failed to load model: {:?}", e)))?;

    let graph = model.graph.unwrap();

    // Build map of initializers
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

    // Track global min and max values
    let mut global_min = f32::MAX;
    let mut global_max = f32::MIN;

    // Examine all nodes in the graph
    for node in graph.node.iter() {
        let op_type = node.op_type.as_str();

        // Only process layers we support
        if LINEAR_ALG.contains(&op_type) || CONVOLUTION.contains(&op_type) {
            // Process weights
            if let Some(weight_min_max) = extract_node_weight_range("weight", node, &initializers)?
            {
                global_min = global_min.min(weight_min_max.0);
                global_max = global_max.max(weight_min_max.1);
                debug!(
                    "Node {}: weight range [{}, {}]",
                    node.name, weight_min_max.0, weight_min_max.1
                );
            }

            // Process bias if present
            if let Some(bias_min_max) = extract_node_weight_range("bias", node, &initializers)? {
                global_min = global_min.min(bias_min_max.0);
                global_max = global_max.max(bias_min_max.1);
                debug!(
                    "Node {}: bias range [{}, {}]",
                    node.name, bias_min_max.0, bias_min_max.1
                );
            }
        }
    }

    // Handle case where no weights were found
    if global_min == f32::MAX || global_max == f32::MIN {
        return Err(Error::msg(
            "No supported layers with weights found in model",
        ));
    }

    debug!(
        "Global weight range: min={}, max={}",
        global_min, global_max
    );
    Ok((global_min, global_max))
}

#[cfg(test)]
mod tests {

    use super::*;

    use crate::{Context, IO, Prover, lookup, quantization::TensorFielder, verify};
    use goldilocks::GoldilocksExt2;
    use transcript::BasicTranscript;

    type F = GoldilocksExt2;

    // cargo test --release --package zkml -- onnx_parse::tests::test_tract --nocapture

    #[test]
    fn test_load_mlp() {
        let filepath = "assets/scripts/MLP/mlp-iris-01.onnx";
        ModelType::MLP.validate(filepath).unwrap();
        let result = load_model::<Element>(&filepath);

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());
    }

    #[test]
    fn test_mlp_model_run() {
        let filepath = "assets/scripts/MLP/mlp-iris-01.onnx";
        ModelType::MLP.validate(filepath).unwrap();
        let model = load_model::<Element>(&filepath).unwrap();
        let input = crate::tensors::Tensor::random(vec![model.input_shape()[0]]);
        let input = model.prepare_input(input);
        let trace = model.run::<F>(input.clone());
        println!("Result: {:?}", trace.final_output());
    }

    #[test]
    fn test_quantize() {
        let input = [0.09039914, -0.07716653];

        println!(
            "Result: {} => {:?}",
            input[0],
            <Element as Quantizer<Element>>::from_f32_unsafe(&input[0])
        );
        println!(
            "Result: {} => {:?}",
            input[1],
            <Element as Quantizer<Element>>::from_f32_unsafe(&input[1])
        );
        println!(
            "Result: {} => {:?}",
            0,
            <Element as Quantizer<Element>>::from_f32_unsafe(&0.0)
        );
        println!(
            "Result: {} => {:?}",
            -1.0,
            <Element as Quantizer<Element>>::from_f32_unsafe(&-1.0)
        );
        println!(
            "Result: {} => {:?}",
            1.0,
            <Element as Quantizer<Element>>::from_f32_unsafe(&1.0)
        );
    }

    #[test]
    fn test_is_cnn() {
        let filepath = "assets/scripts/CNN/cnn-cifar-01.onnx";
        let result = is_cnn(&filepath);

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());
    }

    #[test]
    fn test_load_cnn() {
        // let filepath = "assets/scripts/CNN/lenet-mnist-01.onnx";
        let filepath = "assets/scripts/CNN/cnn-cifar-01.onnx";
        ModelType::CNN.validate(filepath).unwrap();
        let result = load_model::<Element>(&filepath);

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());

        let model = result.unwrap();
        let input = crate::tensors::Tensor::random(model.input_shape());
        let input = model.prepare_input(input);
        let trace = model.run::<F>(input.clone());
        // println!("Result: {:?}", trace.final_output());

        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"m2vec");
        let ctx = Context::<GoldilocksExt2>::generate(&model, Some(input.get_shape()))
            .expect("Unable to generate context");
        let output = trace.final_output().clone();

        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>, lookup::LogUp> =
            Prover::new(&ctx, &mut tr);
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");
        let io = IO::new(input.to_fields(), output.to_fields());
        verify::<_, _, lookup::LogUp>(ctx, proof, io, &mut verifier_transcript).unwrap();
    }
}
