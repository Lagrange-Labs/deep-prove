use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::{
    Element,
    layers::{
        convolution::Convolution,
        dense::Dense,
        flatten::Flatten,
        matrix_mul::{MatMul, OperandMatrix},
        pooling::Pooling,
        provable::{Node, NodeId, OpInfo},
    },
    model::{Model, ToIterator},
    parser::{check_filter, safe_conv2d_shape, safe_maxpool2d_shape},
    tensor::Shape,
};
type GarbagePad = Option<(Shape, Shape)>;

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

pub fn pad_model(mut model: Model<Element>) -> Result<Model<Element>> {
    let input_si = ShapeInfo {
        shapes: model
            .unpadded_input_shapes()
            .into_iter()
            .zip(model.padded_input_shapes())
            .map(|(unpadded_shape, padded_shape)| ShapeData {
                input_shape_padded: padded_shape.into(),
                ignore_garbage_pad: None,
                input_shape_og: unpadded_shape.into(),
            })
            .collect(),
    };
    let mut shape_infos: HashMap<NodeId, ShapeInfo> = HashMap::new();
    let unpadded_input_shapes = model.unpadded_input_shapes();
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
        .collect::<Result<_>>()?;
    model = Model::<Element>::new(unpadded_input_shapes, PaddingMode::Padding, nodes);
    Ok(model)
}

pub(crate) fn reshape(si: &mut ShapeInfo) -> Result<Flatten> {
    si.shapes.iter_mut().for_each(|sd| {
        sd.ignore_garbage_pad = Some((sd.input_shape_og.clone(), sd.input_shape_padded.clone()))
    });
    Ok(Flatten)
}

pub(crate) fn pooling(p: Pooling, si: &mut ShapeInfo) -> Result<Pooling> {
    for sd in si.shapes.iter_mut() {
        // Make sure that input shape is already padded and is well formed
        ensure!(
            sd.input_shape_padded.is_power_of_two(),
            "Input shape for max pool is not padded"
        );
        sd.input_shape_og = safe_maxpool2d_shape(&sd.input_shape_og)?.into();
        sd.input_shape_padded = safe_maxpool2d_shape(&sd.input_shape_padded)?.into();
    }
    Ok(p)
}

pub(crate) fn pad_conv(
    c: Convolution<Element>,
    si: &mut ShapeInfo,
) -> Result<Convolution<Element>> {
    // convolution layer currently expects 1 input, so we check there is only 1 input shape
    ensure!(
        si.shapes.len() == 1,
        "More than 1 input shape found when padding convolution layer"
    );
    let sd = si.shapes.first_mut().unwrap();
    sd.input_shape_og = safe_conv2d_shape(&sd.input_shape_og, &c.filter.get_shape())?.into();
    let weight_shape = c.filter.get_shape();
    // Perform basic sanity checks on the tensor dimensions
    check_filter(&weight_shape).context("filter shape test failed:")?;
    ensure!(
        weight_shape[0] == c.bias.get_shape()[0],
        "Bias length doesn't match filter shape"
    );
    // Make sure that input shape is already padded and is well formed
    ensure!(
        sd.input_shape_padded.is_power_of_two(),
        "Input shape for convolution is not padded",
    );
    ensure!(
        sd.input_shape_padded.rank() == 3,
        "Input shape for convolution is not 3D"
    );
    let new_conv_good = c.clone();
    // Since we are doing an FFT based conv, we need to pad the last two dimensions of the filter to match the input.
    let weight_shape = c.filter.pad_next_power_of_two().get_shape();
    let (filter_height, filter_width) = (weight_shape[2], weight_shape[3]);
    let (input_height, input_width) = (sd.input_shape_padded.dim(1), sd.input_shape_padded.dim(2));

    ensure!(
        filter_height <= input_height && filter_width <= input_width,
        "Filter dimensions in convolution have to be smaller than input dimensions",
    );

    let new_conv = new_conv_good.into_padded_and_ffted(&sd.input_shape_og);
    let output_shape: Shape = safe_conv2d_shape(&sd.input_shape_padded, &weight_shape)?.into();
    sd.input_shape_padded = output_shape.next_power_of_two();
    Ok(new_conv)
}

pub(crate) fn pad_dense(mut d: Dense<Element>, si: &mut ShapeInfo) -> Result<Dense<Element>> {
    // dense layer currently expects 1 input, so we check there is only 1 input shape
    ensure!(
        si.shapes.len() == 1,
        "More than 1 input shape found when padding dense layer"
    );
    let sd = si.shapes.first_mut().unwrap();
    let matrix_shape: Shape = d.matrix.get_shape().into();
    let nrows = matrix_shape.nrows();
    sd.input_shape_og = vec![nrows].into();
    ensure!(
        d.bias.get_data().len() == nrows,
        "Bias length {} does not match matrix width {}",
        d.bias.get_data().len(),
        nrows,
    );
    ensure!(
        sd.input_shape_padded.is_power_of_two(),
        "Input shape for dense is not padded"
    );
    if sd.input_shape_padded.rank() != 1 {
        sd.input_shape_padded = vec![sd.input_shape_padded.numel()].into();
        sd.input_shape_og = vec![sd.input_shape_og.numel()].into();
    }
    let mut new_cols = d.matrix.ncols_2d();
    if d.matrix.ncols_2d() != sd.input_shape_padded.dim(0) {
        if d.matrix.ncols_2d() < sd.input_shape_padded.dim(0) {
            new_cols = sd.input_shape_padded.dim(0);
        } else {
            // If we have too many columns, we can't shrink without losing information
            bail!(
                "Dense layer matrix has more columns ({}) than previous layer output size ({}).
                            Cannot shrink without losing information.",
                d.matrix.ncols_2d(),
                sd.input_shape_padded.dim(0)
            );
        }
    }
    // The reason to pad to a minimum of 4 is that any subsequent activation function will
    // be needing at least input shape of total size 4 due to usage of lookups.
    // current logup gkr implementation requires at least 2 variables for poly.
    let ncols = pad_minimum(new_cols);
    let nrows = pad_minimum(d.matrix.nrows_2d());

    if let Some(previous_shape) = sd.ignore_garbage_pad.as_ref() {
        let previous_input_shape_og = previous_shape.0.clone();
        let previous_input_shape_padded = previous_shape.1.clone();
        d.matrix = d.matrix.pad_matrix_to_ignore_garbage(
            previous_input_shape_og.as_ref(),
            previous_input_shape_padded.as_ref(),
            &[nrows, ncols],
        );
        sd.ignore_garbage_pad = None;
    } else {
        d.matrix
            .reshape_to_fit_inplace_2d(vec![nrows, ncols].into());
    }
    d.bias = d.bias.pad_1d(nrows);
    sd.input_shape_padded = vec![nrows].into();
    Ok(d)
}

pub(crate) fn pad_matmul(mut mat: MatMul<Element>, si: &mut ShapeInfo) -> Result<MatMul<Element>> {
    let expected_num_inputs = mat.num_inputs();
    ensure!(
        si.shapes.len() == expected_num_inputs,
        "Expected {expected_num_inputs} input shapes when padding MatMul, found {}",
        si.shapes.len(),
    );

    ensure!(
        si.shapes
            .iter()
            .all(|s| s.input_shape_og.rank() == 2 && s.input_shape_padded.rank() == 2),
        "Unpadded input shape for MatMul is not 2D"
    );
    let (unpadded_input_shapes, mut padded_input_shapes): (Vec<Shape>, Vec<Shape>) = si
        .shapes
        .iter()
        .map(|s| (s.input_shape_og.clone(), s.input_shape_padded.clone()))
        .collect();
    let mut unpadded_output_shapes =
        mat.output_shapes(&unpadded_input_shapes, PaddingMode::NoPadding);
    ensure!(
        unpadded_output_shapes.len() == 1,
        "Expected 1 unpadded output shape for MatMul, found {}",
        unpadded_output_shapes.len(),
    );
    let unpadded_output_shape = unpadded_output_shapes.pop().unwrap();
    let (left_shape, mut right_shape) = match (&mut mat.left_matrix, &mut mat.right_matrix) {
        (OperandMatrix::Weigth(m), OperandMatrix::Input) => {
            let nrows = pad_minimum(m.tensor.nrows_2d());
            let ncols = padded_input_shapes[0][0];
            m.tensor
                .reshape_to_fit_inplace_2d(vec![nrows, ncols].into());
            (
                m.tensor.get_shape(),
                padded_input_shapes.pop().unwrap(), /* safe to unwrap since we checked the number of inputs at the beginning */
            )
        }
        (OperandMatrix::Input, OperandMatrix::Weigth(m)) => {
            let nrows = padded_input_shapes[0][1];
            let ncols = pad_minimum(m.tensor.ncols_2d());
            m.tensor
                .reshape_to_fit_inplace_2d(vec![nrows, ncols].into());
            (padded_input_shapes.pop().unwrap(), m.tensor.get_shape())
        }
        (OperandMatrix::Input, OperandMatrix::Input) => {
            let right_shape = padded_input_shapes.pop().unwrap();
            let left_shape = padded_input_shapes.pop().unwrap();
            (left_shape, right_shape)
        }
        (OperandMatrix::Weigth(_), OperandMatrix::Weigth(_)) => {
            unreachable!("Found MatMul layer with 2 weight matrices")
        }
    };
    if mat.is_right_transposed() {
        right_shape.reverse();
    }
    ensure!(
        left_shape[1] == right_shape[0],
        "While padding MatMul layer. number of columns in left matrix ({}) does not match with number of rows in right matrix ({})",
        left_shape[1],
        right_shape[0],
    );
    ensure!(
        si.shapes.iter().all(|sd| sd.ignore_garbage_pad.is_none()),
        "MatMul layer has garbage padding to be removed",
    );
    si.shapes = vec![ShapeData {
        input_shape_og: unpadded_output_shape.into(),
        input_shape_padded: vec![left_shape[0], right_shape[1]].into(),
        ignore_garbage_pad: None,
    }];
    Ok(mat)
}

fn pad_minimum(dim: usize) -> usize {
    let r = dim.next_power_of_two();
    if r < 4 { 4 } else { r }
}
