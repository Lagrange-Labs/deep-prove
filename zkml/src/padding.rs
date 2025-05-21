use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, ensure};

use crate::{
    layers::{
        convolution::Convolution, dense::Dense, flatten::Flatten, matrix_mul::{MatMul, OperandMatrix}, pooling::Pooling, provable::{Node, NodeId, OpInfo}
    }, model::{Model, ToIterator}, onnx_parse::{check_filter, safe_conv2d_shape, safe_maxpool2d_shape}, try_unzip, Element
};
type GarbagePad = Option<(Vec<usize>, Vec<usize>)>;
type Shape = Vec<usize>;

#[derive(thiserror::Error, Debug)]
pub enum PaddingError {
    #[error("Shape mismatch when padding model: {0}")]
    ShapeMismatch(String),
    #[error("Geenric error when padding model: {0}")]
    GenericError(anyhow::Error),
}

impl From<anyhow::Error> for PaddingError {
    fn from(error: anyhow::Error) -> Self {
        PaddingError::GenericError(error)
    }
}

#[derive(Clone, Debug, Copy)]
pub enum PaddingMode {
    NoPadding,
    Padding,
}

#[derive(Clone, Debug)]
pub struct ShapeInfo {
    shapes: Vec<ShapeData>,
}

#[derive(Clone, Debug)]
pub struct ShapeData {
    input_shape_padded: Shape,
    ignore_garbage_pad: GarbagePad,
    input_shape_og: Shape,
}

pub fn pad_model(mut model: Model<Element>) -> Result<Model<Element>, PaddingError> {
    let input_si = ShapeInfo {
        shapes: model
            .unpadded_input_shapes()
            .into_iter()
            .zip(model.padded_input_shapes())
            .map(|(unpadded_shape, padded_shape)| ShapeData {
                input_shape_padded: padded_shape,
                ignore_garbage_pad: None,
                input_shape_og: unpadded_shape,
            })
            .collect(),
    };
    let mut shape_infos: HashMap<NodeId, ShapeInfo> = HashMap::new();
    let unpadded_input_shapes = model.unpadded_input_shapes();
    let mut catch_err = Ok(());
    let nodes = model
        .into_forward_iterator()
        .map(|(node_id, node)| -> Result<(NodeId, Node<Element>)> {
            let shapes = node
                .inputs
                .iter()
                .map(|edge| {
                    if let Some(n) = edge.node {
                        let si = shape_infos
                            .get(&n)
                            .ok_or(anyhow!("Shapes for node {n} not found"))?;
                        ensure!(
                            edge.index < si.shapes.len(),
                            "Shape for input {} requested, but node {n} has only {} inputs",
                            edge.index,
                            si.shapes.len(),
                        );
                        Ok(si.shapes[edge.index].clone())
                    } else {
                        ensure!(
                            edge.index < input_si.shapes.len(),
                            "Shape for input {} requested, but model has only {} inputs",
                            edge.index,
                            input_si.shapes.len(),
                        );
                        Ok(input_si.shapes[edge.index].clone())
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let mut si = ShapeInfo { shapes };
            let node = node.pad_node(&mut si)?;
            shape_infos.insert(node_id, si);
            Ok((node_id, node))
        })
        .map_while(|n| {
            if n.is_err() {
                catch_err = Err(n.unwrap_err());
                None
            } else {
                Some(n.unwrap())
            }
        });
    model = Model::<Element>::new(unpadded_input_shapes, PaddingMode::Padding, nodes);
    catch_err?;
    Ok(model)
}

pub(crate) fn reshape(si: &mut ShapeInfo) -> Result<Flatten, PaddingError> {
    si.shapes.iter_mut().for_each(|sd| {
        sd.ignore_garbage_pad = Some((sd.input_shape_og.clone(), sd.input_shape_padded.clone()))
    });
    Ok(Flatten)
}

pub(crate) fn pooling(p: Pooling, si: &mut ShapeInfo) -> Result<Pooling, PaddingError> {
    for sd in si.shapes.iter_mut() {
        // Make sure that input shape is already padded and is well formed
        if !sd.input_shape_padded.iter().all(|d| d.is_power_of_two()) {
            return Err(PaddingError::ShapeMismatch(
                "Input shape for max pool is not padded".to_string(),
            ));
        }
        sd.input_shape_og = safe_maxpool2d_shape(&sd.input_shape_og)?;
        sd.input_shape_padded = safe_maxpool2d_shape(&sd.input_shape_padded)?;
    }
    Ok(p)
}

pub(crate) fn pad_conv(
    c: Convolution<Element>,
    si: &mut ShapeInfo,
) -> Result<Convolution<Element>, PaddingError> {
    // convolution layer currently expects 1 input, so we check there is only 1 input shape
    if si.shapes.len() != 1 {
        return Err(PaddingError::ShapeMismatch(
            "More than 1 input shape found for convolution layer".to_string(),
        ));
    }
    let sd = si.shapes.first_mut().unwrap();
    sd.input_shape_og = safe_conv2d_shape(&sd.input_shape_og, &c.filter.get_shape())?;
    let weight_shape = c.filter.get_shape();
    // Perform basic sanity checks on the tensor dimensions
    check_filter(&weight_shape).context("filter shape test failed:")?;
    if weight_shape[0] != c.bias.get_shape()[0] {
        return Err(PaddingError::ShapeMismatch(
            "Bias length doesn't match filter shape".to_string(),
        ));
    }
    // Make sure that input shape is already padded and is well formed
    if !sd.input_shape_padded.iter().all(|d| d.is_power_of_two()) {
        return Err(PaddingError::ShapeMismatch(
            "Input shape for convolution is not padded".to_string(),
        ));
    }
    if sd.input_shape_padded.len() != 3 {
        return Err(PaddingError::ShapeMismatch(
            "Input shape for convolution is not 3D".to_string(),
        ));
    }
    let new_conv_good = c.clone();
    // Since we are doing an FFT based conv, we need to pad the last two dimensions of the filter to match the input.
    let weight_shape = c.filter.pad_next_power_of_two().get_shape();
    let (filter_height, filter_width) = (weight_shape[2], weight_shape[3]);
    let (input_height, input_width) = (sd.input_shape_padded[1], sd.input_shape_padded[2]);

    if filter_height > input_height || filter_width > input_width {
        return Err(PaddingError::ShapeMismatch(
            "Filter dimensions have to be smaller than input dimensions".to_string(),
        ));
    }

    let new_conv = new_conv_good.into_padded_and_ffted(&sd.input_shape_og);
    let output_shape = safe_conv2d_shape(&sd.input_shape_padded, &weight_shape)?;
    sd.input_shape_padded = output_shape
        .iter()
        .map(|i| i.next_power_of_two())
        .collect::<Vec<_>>();
    Ok(new_conv)
}

pub(crate) fn pad_dense(
    mut d: Dense<Element>,
    si: &mut ShapeInfo,
) -> Result<Dense<Element>, PaddingError> {
    // dense layer currently expects 1 input, so we check there is only 1 input shape
    if si.shapes.len() != 1 {
        return Err(PaddingError::ShapeMismatch(
            "More than 1 input shape found for dense layer".to_string(),
        ));
    }
    let sd = si.shapes.first_mut().unwrap();
    let nrows = d.matrix.get_shape()[0];
    sd.input_shape_og = vec![nrows];
    if d.bias.get_data().len() != nrows {
        return Err(PaddingError::ShapeMismatch(format!(
            "bias length {} does not match matrix width {}",
            d.bias.get_data().len(),
            nrows
        )));
    }
    if !sd.input_shape_padded.iter().all(|d| d.is_power_of_two()) {
        return Err(PaddingError::ShapeMismatch(
            "Input shape for dense is not padded".to_string(),
        ));
    }
    if sd.input_shape_padded.len() != 1 {
        sd.input_shape_padded = vec![sd.input_shape_padded.iter().product()];
        sd.input_shape_og = vec![sd.input_shape_og.iter().product()];
    }
    let mut new_cols = d.matrix.ncols_2d();
    if d.matrix.ncols_2d() != sd.input_shape_padded[0] {
        if d.matrix.ncols_2d() < sd.input_shape_padded[0] {
            new_cols = sd.input_shape_padded[0];
        } else {
            // If we have too many columns, we can't shrink without losing information
            return Err(PaddingError::GenericError(anyhow!(
                "Matrix has more columns ({}) than previous layer output size ({}).
                            Cannot shrink without losing information.",
                d.matrix.ncols_2d(),
                sd.input_shape_padded[0]
            )));
        }
    }
    // The reason to pad to a minimum of 4 is that any subsequent activation function will
    // be needing at least input shape of total size 4 due to usage of lookups.
    // current logup gkr implementation requires at least 2 variables for poly.
    let ncols = pad_minimum(new_cols);
    let nrows = pad_minimum(d.matrix.nrows_2d());

    if let Some(ref previous_shape) = sd.ignore_garbage_pad.as_ref() {
        let previous_input_shape_og = previous_shape.0.clone();
        let previous_input_shape_padded = previous_shape.1.clone();
        d.matrix = d.matrix.pad_matrix_to_ignore_garbage(
            &previous_input_shape_og,
            &previous_input_shape_padded,
            &vec![nrows, ncols],
        );
        sd.ignore_garbage_pad = None;
    } else {
        d.matrix.reshape_to_fit_inplace_2d(vec![nrows, ncols]);
    }
    d.bias = d.bias.pad_1d(nrows);
    sd.input_shape_padded = vec![nrows];
    Ok(d)
}

pub(crate) fn pad_matmul( 
    mut mat: MatMul<Element>,
    si: &mut ShapeInfo,
) -> Result<MatMul<Element>, PaddingError> {
    let expected_num_inputs = mat.num_inputs();
    if si.shapes.len() != expected_num_inputs {
        return Err(PaddingError::ShapeMismatch(
            format!("Expected {expected_num_inputs} input shapes for MatMul, found {}",
        si.shapes.len(),
        )));
    }
    let (unpadded_input_shapes, mut padded_input_shapes): (Vec<_>, Vec<_>) = try_unzip(si.shapes.iter().map(|s| {
        if s.input_shape_og.len() != 2 {
            return Err(PaddingError::ShapeMismatch(
                "Unpadded input shape for MatMul is not 2D".to_string(),
            ));
        }
        if s.input_shape_padded.len() != 2 {
            return Err(PaddingError::ShapeMismatch(
                "Padded input shape for MatMul is not 2D".to_string(),
            ));
        }
        Ok((s.input_shape_og.clone(), s.input_shape_padded.clone()))
    }))?;


    let mut unpadded_output_shapes = mat.output_shapes(&unpadded_input_shapes, PaddingMode::NoPadding);
    if unpadded_output_shapes.len() != 1 {
        return Err(PaddingError::ShapeMismatch(
            format!("Expected 1 unpadded output shape for MatMul, found {}", unpadded_output_shapes.len()).to_string()
        ));
    }
    let unpadded_output_shape = unpadded_output_shapes.pop().unwrap();
    let (left_shape, right_shape) = match (&mut mat.left_matrix, &mut mat.right_matrix) {
        (OperandMatrix::Weigth(m), OperandMatrix::Input) => {
            let nrows = pad_minimum(m.tensor.nrows_2d());
            let ncols = padded_input_shapes[0][0];
            m.tensor.reshape_to_fit_inplace_2d(vec![nrows, ncols]);
            (
                m.tensor.get_shape(),
                padded_input_shapes.pop().unwrap(), // safe to unwrap since we checked the number of inputs at the beginning
            )
        },
        (OperandMatrix::Input, OperandMatrix::Weigth(m)) => {
            let nrows = padded_input_shapes[0][1];
            let ncols = pad_minimum(m.tensor.ncols_2d());
            m.tensor.reshape_to_fit_inplace_2d(vec![nrows, ncols]);
            (
                padded_input_shapes.pop().unwrap(),
                m.tensor.get_shape(),
            )
        },
        (OperandMatrix::Input, OperandMatrix::Input) => {
            let right_shape = padded_input_shapes.pop().unwrap();
            let left_shape = padded_input_shapes.pop().unwrap();
            (
                left_shape,
                right_shape,
            )
        }
        (OperandMatrix::Weigth(_), OperandMatrix::Weigth(_)) => unreachable!("Found MatMul layer with 2 weight matrices"),  
    };
    if left_shape[1] != right_shape[0] {
        return Err(PaddingError::ShapeMismatch(
                format!("Number of columns in left matrix ({}) does not match with number of rows in right matrix ({})",
                left_shape[1],
                right_shape[0],
            ).to_string(),
        ));
    }
    if !si.shapes.iter().all(|sd| 
        sd.ignore_garbage_pad.is_none()
    ) {
        return Err(PaddingError::ShapeMismatch(
            "MatMul layer has garbage padding to be removed".to_string(),
        ));
    }
    si.shapes = vec![ShapeData {
        input_shape_og: unpadded_output_shape,
        input_shape_padded: vec![left_shape[0], right_shape[1]],
        ignore_garbage_pad: None,
    }];
    Ok(mat)
}

fn pad_minimum(dim: usize) -> usize {
    let r = dim.next_power_of_two();
    if r < 4 { 4 } else { r }
}
