use crate::dense::Dense;
use anyhow::{bail, ensure, Context, Error, Result};
use itertools::Itertools;
use log::debug;
use std::{collections::HashMap, i8, path::Path};
use tract_onnx::{pb::NodeProto, prelude::*};

use crate::{
    Element,
    activation::{Activation, Relu},
    model::{Layer, Model},
    quantization::Quantizer,
};

#[derive(Debug, Clone)]
struct Gemm {
    name: String,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
}

// Given a ONNX node, build a struct which contains information about the Gemm
fn build_gemm(node: &NodeProto) -> Result<Gemm> {
    let name = node.name.to_string();
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(1.);
    let beta = node.get_attr_opt("beta")?.unwrap_or(1.);
    let trans_a = node.get_attr_opt("transA")?.unwrap_or(false);
    let trans_b = node.get_attr_opt("transB")?.unwrap_or(false);
    let gemm = Gemm {
        name,
        alpha,
        beta,
        trans_a,
        trans_b,
    };
    Ok(gemm)
}

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
    let activation_functions = ["Relu", "Sigmoid", "Tanh", "LeakyRelu", "Elu", "Selu"];

    let mut is_mlp = true;
    let mut prev_was_gemm_or_matmul = false;

    let model = tract_onnx::onnx()
        .proto_model_for_path(filepath)
        .map_err(|e| Error::msg(format!("Failed to load model: {:?}", e)))?;
    let graph = model.graph.unwrap();

    for node in graph.node.iter() {
        if node.op_type == "Gemm" || node.op_type == "MatMul" {
            if prev_was_gemm_or_matmul {
                is_mlp = false;
                break;
            }
            prev_was_gemm_or_matmul = true;
        } else if activation_functions.contains(&node.op_type.as_str()) {
            if !prev_was_gemm_or_matmul {
                is_mlp = false;
                break;
            }
            prev_was_gemm_or_matmul = false;
        } else {
            is_mlp = false;
            break;
        }
    }

    Ok(is_mlp)
}

// Given a flat vector, converts it to matrix in row major form
fn reshape<T: Clone>(flat_vec: Vec<T>, rows: usize, cols: usize) -> Option<Vec<Vec<T>>> {
    if flat_vec.len() != rows * cols {
        return None; // Return None if dimensions don't match the number of elements
    }

    Some(flat_vec.chunks(cols).map(|chunk| chunk.to_vec()).collect())
}

fn fetch_weight_bias_as_mat<Q: Quantizer<Element>>(
    weight_or_bias: &str,
    node: &NodeProto,
    initializers: &HashMap<String, Tensor>,
) -> Result<Vec<Vec<Element>>> {
    ensure!(weight_or_bias == "weight" || weight_or_bias == "bias");

    let alpha_or_beta = node
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
    let alpha_or_beta = alpha_or_beta[0];

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
    let tensor_f = tensor_t_f32.iter().map(Q::from_f32_unsafe).collect_vec();

    let (rows, cols) = match tensor_t.shape().len() {
        1 => (tensor_t.shape()[0], 1),
        2 => (tensor_t.shape()[0], tensor_t.shape()[1]),
        _ => bail!("Invalid tensor shape: expected 1D or 2D tensor"),
    };

    let field_matrix = reshape(tensor_f, rows, cols).unwrap();

    Ok(field_matrix)
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
        let dt = tract_onnx::pb::tensor_proto::DataType::from_i32(item.data_type).context("can't load from onnx")?.try_into()?;
        let shape: Vec<usize> = item.dims.iter().map(|&i| i as usize).collect();
        let value = create_tensor(shape, dt, &item.raw_data).unwrap();
        let key = item.name.to_string();
        initializers.insert(key, value);
    }

    let mut layers: Vec<Layer> = Vec::new();
    for (i, node) in graph.node.iter().enumerate() {
        match node.op_type.as_str() {
            "Gemm" => {
                let matrix_weight = fetch_weight_bias_as_mat::<Q>("weight", node, &initializers)?;
                let matrix_bias = fetch_weight_bias_as_mat::<Q>("bias", node, &initializers)?;
                ensure!(matrix_bias.iter().all(|r| r.len() == 1), "bias is not a vector");
                let flat_bias = matrix_bias.into_iter().map(|r| r[0]).collect_vec();
                let nrows = matrix_weight.len();
                ensure!(flat_bias.len() == nrows, "bias length {} does not match matrix width {}", flat_bias.len(), nrows);

                // Create matrix and transpose (PyTorch stores as output_size x input_size)
                let matrix =
                    crate::tensor::Tensor::<Element>::from_coeffs_2d(matrix_weight).unwrap();
                let bias =
                    crate::tensor::Tensor::<Element>::new(vec![nrows], flat_bias);
                debug!("layer idx {} -> unprocessed matrix {:?}", i, matrix.dims());
                layers.push(Layer::Dense(Dense::new(matrix, bias)));
            }
            "Relu" => {
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

#[cfg(test)]
mod tests {

    use super::*;

    use goldilocks::GoldilocksExt2;

    // cargo test --release --package zkml -- onnx_parse::tests::test_tract --nocapture

    #[test]
    fn test_tract() {
        let filepath = "assets/model.onnx";
        let result = load_mlp::<Element>(&filepath);

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());
    }

    #[test]
    fn test_model_run() {
        let filepath = "assets/model.onnx";

        let model = load_mlp::<Element>(&filepath).unwrap();
        let input = crate::tensor::Tensor::random(vec![model.input_shape()[0]]);
        // random_vector::<QuantInteger>(model.input_shape()[0])
        //     .into_iter()
        //     .map(|x| x as Element)
        //     .collect_vec();

        let trace = model.run(input.clone());
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
}
