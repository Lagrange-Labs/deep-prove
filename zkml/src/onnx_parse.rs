use crate::dense::Dense;
use anyhow::{Context, Error, Result, bail, ensure};
use itertools::Itertools;
use log::debug;
use std::{collections::HashMap, i8, path::Path};
use tract_onnx::{pb::NodeProto, prelude::*};

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

pub fn load_mlp<Q: Quantizer<Element>>(filepath: &str) -> Result<Model> {
    if !Path::new(filepath).exists() {
        return Err(Error::msg(format!("File '{}' does not exist", filepath)));
    }
    // TODO: Re-enable. Was disabled to test the bench binary but only dense layer were working
    // assert!(is_mlp(filepath)?, "is_mlp: Failed");

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
    for (i, node) in graph.node.iter().enumerate() {
        match node.op_type.as_str() {
            op if LINEAR_ALG.contains(&op) => {
                let matrix_weight =
                    fetch_weight_bias_as_tensor::<Q>(1.0, "weight", node, &initializers)?;
                let matrix_bias =
                    fetch_weight_bias_as_tensor::<Q>(1.0, "bias", node, &initializers)?;

                // Concatenate bias as an extra column
                let matrix = matrix_weight.concat_matvec_col(&matrix_bias);

                debug!("layer idx {} -> unprocessed matrix {:?}", i, matrix.dims());
                layers.push(Layer::Dense(Dense::new(matrix_weight, matrix_bias)));
            }
            op if ACTIVATION.contains(&op) => {
                let layer = Layer::Activation(Activation::Relu(Relu::new()));
                layers.push(layer);
            }
            _ => (),
        };
    }

    println!(" DONE READING LAYERS - MOVING ON TO PADDING");
    // Process the layers to ensure consistent dimensions
    let mut processed_layers: Vec<Layer> = Vec::new();
    let mut prev_layer_shape: Option<Vec<usize>> = None;
    let last = layers.len() - 1;
    for (i, layer) in layers.into_iter().enumerate() {
        if let Layer::Dense(dense) = layer {
            let Dense { mut matrix, bias } = dense;
            let mut new_cols = matrix.ncols_2d();
            if let Some(prev_shape) = prev_layer_shape {
                assert!(prev_shape.iter().all(|d| d.is_power_of_two()));
                // Check if previous output's vector length is equal to the number of columns of this matrix
                if matrix.ncols_2d() != prev_shape[0] {
                    if matrix.ncols_2d() < prev_shape[0] {
                        new_cols = prev_shape[0];
                    } else {
                        // If we have too many columns, we can't shrink without losing information
                        panic!(
                            "Matrix has more columns ({}) than previous layer output size ({}).
                            Cannot shrink without losing information.",
                            matrix.ncols_2d(),
                            prev_shape[0]
                        );
                    }
                }
            }

            let ncols = new_cols.next_power_of_two();
            let nrows = matrix.nrows_2d().next_power_of_two();

            // println!("layer idx {} -> from ({:?} to ({},{})",i,matrix.shape(),
            //                 nrows.next_power_of_two(),
            //                 new_cols.next_power_of_two());
            // Pad to power of two dimensions
            matrix.reshape_to_fit_inplace_2d(vec![nrows, ncols]);
            let bias = bias.pad_1d(nrows);
            // Update prev_output_size to reflect the padded size
            prev_layer_shape = Some(matrix.dims());
            debug!("layer idx {} -> final shape {:?}", i, matrix.dims());
            processed_layers.push(Layer::Dense(Dense::new(matrix, bias)));
        } else {
            // prev_layer_shape = Some(layer.shape()); // TODO: Need to double check
            processed_layers.push(layer);
        }
    }

    let mut model = Model::new();
    for layer in processed_layers {
        model.add_layer(layer);
    }

    Ok(model)
}

// TODO: Need to get max_abs by looking at the range of the weights and biases during calibration.
fn fetch_weight_bias_as_tensor<Q: Quantizer<Element>>(
    max_abs: f64,
    weight_or_bias: &str,
    node: &NodeProto,
    initializers: &HashMap<String, Tensor>,
) -> Result<crate::tensor::Tensor<Element>> {
    ensure!(weight_or_bias == "weight" || weight_or_bias == "bias");

    let mut alpha_or_beta: f32 = 1.0;
    if node.name.contains("Gemm") {
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
        alpha_or_beta = result[0];
    }

    let tensor_vec = node
        .input
        .iter()
        .filter(|x| x.contains(weight_or_bias))
        .filter_map(|key| initializers.get(key).cloned())
        .collect_vec();

    // If a node is Gemm, then it has only one tensor of the form "fcN.weight"
    let tensor_t = tensor_vec[0].clone();
    let tensor_t_f32 = tensor_t.as_slice::<f32>().unwrap().to_vec();
    let tensor_t_f32 = tensor_t_f32.iter().map(|x| x * alpha_or_beta).collect_vec();
    
    // Calculate both min and max values, not just max_abs
    let max_abs = tensor_t_f32.iter().fold(0.0f32, |max_so_far, &val| max_so_far.max(val.abs()));
    let min_val = tensor_t_f32.iter().fold(f32::MAX, |min_so_far, &val| min_so_far.min(val));
    let max_val = tensor_t_f32.iter().fold(f32::MIN, |max_so_far, &val| max_so_far.max(val));
    println!("Tensor: min={}, max={}, abs={}", min_val, max_val, max_abs);
    // Use the new quantization method with both min and max
    let tensor_f = tensor_t_f32
        .iter()
        .map(|x| Q::from_f32_unsafe_clamp(x, max_abs as f64))
        .collect_vec();
    
    let tensor_result = crate::tensor::Tensor::new(tensor_t.shape().to_vec(), tensor_f);

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

pub fn load_cnn<Q: Quantizer<Element>>(filepath: &str) -> Result<Model> {
    if !Path::new(filepath).exists() {
        return Err(Error::msg(format!("File '{}' does not exist", filepath)));
    }
    let result = is_cnn(filepath)?;
    if !result {
        bail!("is_cnn: Failed");
    }

    let model = tract_onnx::onnx()
        .proto_model_for_path(filepath)
        .map_err(|e| Error::msg(format!("Failed to load model: {:?}", e)))?;

    let graph = model.graph.unwrap();
    let mut initializers: HashMap<String, Tensor> = HashMap::new();
    for item in graph.initializer {
        let dt = tract_onnx::pb::tensor_proto::DataType::from_i32(item.data_type)
            .unwrap()
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
                let mut weight =
                    fetch_weight_bias_as_tensor::<Q>(1.0, "weight", node, &initializers)?;
                let bias = fetch_weight_bias_as_tensor::<Q>(1.0, "bias", node, &initializers)?;
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

                // println!("layer idx {} -> from ({:?} to ({},{})",i,matrix.shape(),
                //                 nrows.next_power_of_two(),
                //                 new_cols.next_power_of_two());
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
                // TODO
                let weight = fetch_weight_bias_as_tensor::<Q>(1.0, "weight", node, &initializers)?;
                let bias = fetch_weight_bias_as_tensor::<Q>(1.0, "bias", node, &initializers)?;
                unimplemented!()
            }
            op if DOWNSAMPLING.contains(&op) => {
                // TODO
                let _ = fetch_maxpool_attributes(node)?;
                let layer = Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default()));
                layers.push(layer);
                unimplemented!()
            }
            op if RESHAPE.contains(&op) => {
                // TODO
                unimplemented!()
            }
            _ => (),
        };
    }

    let mut model = Model::new();
    for layer in layers {
        model.add_layer(layer);
    }
    Ok(model)
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
        let result = load_mlp::<Element>(&filepath);

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());
    }

    #[test]
    fn test_model_run() {
        let filepath = "assets/scripts/MLP/mlp-iris-01.onnx";

        let model = load_mlp::<Element>(&filepath).unwrap();
        let input = crate::tensor::Tensor::random(vec![model.input_shape()[0]]);
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
    fn test_load_cnn() {
        // let filepath = "assets/scripts/CNN/lenet-mnist-01.onnx";
        let filepath = "assets/scripts/CNN/cnn-cifar-01.onnx";
        let result = load_cnn::<Element>(&filepath);

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());
    }
}
