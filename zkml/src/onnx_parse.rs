use crate::dense::Dense;
use anyhow::{Context, Error, Result, bail, ensure};
use goldilocks::GoldilocksExt2;
use itertools::Itertools;
use std::{collections::HashMap, i8, path::Path};
use tracing::debug;
use tract_onnx::{pb::NodeProto, prelude::*};

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
}

/// Unified model loading function that handles both MLP and CNN models
pub fn load_model<Q: Quantizer<Element>>(filepath: &str, model_type: ModelType) -> Result<Model> {
    // Validate that the model matches the expected type
    model_type.validate(filepath)?;

    // Get global weight ranges first
    let (global_min, global_max) = analyze_model_weight_ranges(filepath)?;
    let global_max_abs = global_min.abs().max(global_max.abs());
    println!(
        "Using global weight range for quantization: [{}, {}], max_abs={}",
        global_min, global_max, global_max_abs
    );

    // Continue with model loading
    let model = tract_onnx::onnx()
        .proto_model_for_path(filepath)
        .map_err(|e| Error::msg(format!("Failed to load model: {:?}", e)))?;

    let graph = model.graph.unwrap();
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
    let mut prev_layer_shape: Option<Vec<usize>> = None;
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
                ensure!(bias.dims().len() == 1, "bias is not a vector");
                let nrows = weight.dims()[0];
                ensure!(
                    bias.get_data().len() == nrows,
                    "bias length {} does not match matrix width {}",
                    bias.get_data().len(),
                    nrows
                );
                let mut new_cols = weight.ncols_2d();
                if let Some(prev_shape) = prev_layer_shape {
                    assert!(prev_shape.iter().all(|d| d.is_power_of_two()));
                    // Check if previous output's vector length is equal to the number of columns of this matrix
                    if weight.ncols_2d() != prev_shape[0] {
                        if weight.ncols_2d() < prev_shape[0] {
                            new_cols = prev_shape[0];
                        } else {
                            // If we have too many columns, we can't shrink without losing information
                            panic!(
                                "Matrix has more columns ({}) than previous layer output size ({}).
                                Cannot shrink without losing information.",
                                weight.ncols_2d(),
                                prev_shape[0]
                            );
                        }
                    }
                }

                let ncols = new_cols.next_power_of_two();
                let nrows = weight.nrows_2d().next_power_of_two();

                // Pad to power of two dimensions
                weight.reshape_to_fit_inplace_2d(vec![nrows, ncols]);
                let bias = bias.pad_1d(nrows);
                // Update prev_output_size to reflect the padded size
                prev_layer_shape = Some(weight.dims());
                debug!("layer idx {} -> final shape {:?}", i, weight.dims());
                layers.push(Layer::Dense(Dense::new(weight, bias)));
            }
            op if ACTIVATION.contains(&op) => {
                let layer = Layer::Activation(Activation::Relu(Relu::new()));
                layers.push(layer);
            }
            op if CONVOLUTION.contains(&op) => {
                let _weight = fetch_weight_bias_as_tensor::<Q>(
                    "weight",
                    node,
                    &initializers,
                    global_max_abs,
                )?;
                let _bias =
                    fetch_weight_bias_as_tensor::<Q>("bias", node, &initializers, global_max_abs)?;
                // CNN-specific implementation
                unimplemented!("CNN convolution layer processing not yet implemented")
            }
            op if DOWNSAMPLING.contains(&op) => {
                let _ = fetch_maxpool_attributes(node)?;
                let layer = Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default()));
                layers.push(layer);
                unimplemented!("CNN pooling layer processing not yet implemented")
            }
            op if RESHAPE.contains(&op) => {
                // Most likely this is for flattening after CNN layers and before Dense layers
                unimplemented!("CNN reshape layer processing not yet implemented")
            }
            _ => (),
        };
    }

    // Create and return the model
    let mut model = Model::new();
    for layer in layers {
        model.add_layer::<F>(layer);
    }
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
) -> Result<crate::tensor::Tensor<Element>> {
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

    println!(
        "Tensor {}: local range=[{}, {}], abs={}, using global_max_abs={}",
        weight_or_bias, min_val, max_val, local_max_abs, global_max_abs
    );

    // Quantize using the global max_abs
    let tensor_f = tensor_data
        .iter()
        .map(|x| Q::from_f32_unsafe_clamp(x, global_max_abs as f64))
        //.map(|x| Q::from_f32_unsafe_clamp(x, local_max_abs as f64))
        .collect_vec();

    let tensor_result = crate::tensor::Tensor::new(tensor_shape, tensor_f);

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

    println!(
        "Global weight range: min={}, max={}",
        global_min, global_max
    );
    Ok((global_min, global_max))
}

#[cfg(test)]
mod tests {

    use super::*;

    use goldilocks::GoldilocksExt2;

    type F = GoldilocksExt2;

    // cargo test --release --package zkml -- onnx_parse::tests::test_tract --nocapture

    #[test]
    fn test_tract() {
        let filepath = "assets/scripts/MLP/mlp-iris-01.onnx";
        let result = load_model::<Element>(&filepath, ModelType::MLP);

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());
    }

    #[test]
    fn test_model_run() {
        let filepath = "assets/scripts/MLP/mlp-iris-01.onnx";

        let model = load_model::<Element>(&filepath, ModelType::MLP).unwrap();
        let input = crate::tensor::Tensor::random(vec![model.input_shape()[0]]);
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

    // #[test]
    // fn test_load_cnn() {
    //    // let filepath = "assets/scripts/CNN/lenet-mnist-01.onnx";
    //    let filepath = "assets/scripts/CNN/cnn-cifar-01.onnx";
    //    let result = load_model::<Element>(&filepath, ModelType::CNN);

    //    assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());
    //}
}
