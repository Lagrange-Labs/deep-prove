use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail, ensure};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{
    Element, Tensor,
    layers::{
        concat_matmul::ConcatMatMul,
        convolution::Convolution,
        dense::Dense,
        flatten::Flatten,
        matrix_mul::{MatMul, OperandMatrix},
        pooling::Pooling,
        provable::{Node, NodeId, OpInfo},
        reshape::Reshape,
        transformer::{mha::pad_matrix_to_ignore_mha_garbage, qkv::QKV},
    },
    model::{Model, ToIterator},
    parser::{check_filter, safe_conv2d_shape, safe_maxpool2d_shape},
    tensor::Shape,
};

#[derive(Clone, Debug)]
pub enum GarbagePad {
    Convolution((Shape, Shape)),
    MHA((Shape, Shape)),
}

impl GarbagePad {
    fn pad_matrix_to_ignore_garbage(
        &self,
        matrix: &mut Tensor<Element>,
        padded_matrix_shape: Shape,
    ) -> Result<()> {
        match self {
            GarbagePad::Convolution(previous_shape) => {
                let previous_input_shape_og = previous_shape.0.clone();
                let previous_input_shape_padded = previous_shape.1.clone();
                *matrix = matrix.pad_matrix_to_ignore_garbage(
                    previous_input_shape_og.as_ref(),
                    previous_input_shape_padded.as_ref(),
                    &padded_matrix_shape,
                );
            }
            GarbagePad::MHA(previous_shape) => {
                *matrix = pad_matrix_to_ignore_mha_garbage(
                    matrix,
                    &previous_shape.0,
                    &previous_shape.1,
                    padded_matrix_shape,
                )?;
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub enum PaddingMode {
    NoPadding,
    Padding,
}

#[derive(Clone, Debug)]
pub struct ShapeInfo {
    pub(crate) shapes: Vec<ShapeData>,
}

impl ShapeInfo {
    pub fn unpadded_input_shapes(&self) -> Vec<Shape> {
        self.shapes
            .iter()
            .map(|sd| sd.input_shape_og.clone())
            .collect()
    }

    pub fn padded_input_shapes(&self) -> Vec<Shape> {
        self.shapes
            .iter()
            .map(|sd| sd.input_shape_padded.clone())
            .collect()
    }
}

impl From<&[ShapeData]> for ShapeInfo {
    fn from(value: &[ShapeData]) -> Self {
        Self {
            shapes: value.to_vec(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ShapeData {
    pub(crate) input_shape_padded: Shape,
    pub(crate) ignore_garbage_pad: Option<GarbagePad>,
    pub(crate) input_shape_og: Shape,
}

impl ShapeData {
    /// Build new shape data for an input tensor of a layer, given the unpadded input shape
    pub fn new(unpadded_input_shape: Shape) -> Self {
        Self {
            input_shape_padded: unpadded_input_shape.next_power_of_two(),
            ignore_garbage_pad: None,
            input_shape_og: unpadded_input_shape,
        }
    }

    pub fn new_with_garbage_pad(unpadded_input_shape: Shape, garbage_pad: GarbagePad) -> Self {
        Self {
            input_shape_padded: unpadded_input_shape.next_power_of_two(),
            ignore_garbage_pad: Some(garbage_pad),
            input_shape_og: unpadded_input_shape,
        }
    }

    pub(crate) fn with_garbage_pad(self, garbage_pad: GarbagePad) -> Self {
        Self {
            input_shape_padded: self.input_shape_padded,
            ignore_garbage_pad: Some(garbage_pad),
            input_shape_og: self.input_shape_og,
        }
    }
}

pub fn pad_model(mut model: Model<Element>) -> Result<Model<Element>> {
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
        sd.ignore_garbage_pad = Some(GarbagePad::Convolution((
            sd.input_shape_og.clone(),
            sd.input_shape_padded.clone(),
        )))
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
        sd.input_shape_og = safe_maxpool2d_shape(&sd.input_shape_og)?;
        sd.input_shape_padded = safe_maxpool2d_shape(&sd.input_shape_padded)?;
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
    sd.input_shape_og = safe_conv2d_shape(&sd.input_shape_og, &c.filter.get_shape())?;
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
    let output_shape: Shape = safe_conv2d_shape(&sd.input_shape_padded, &weight_shape)?;
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
    let matrix_shape: Shape = d.matrix.get_shape();
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
        sd.input_shape_padded = vec![sd.input_shape_padded.product()].into();
        sd.input_shape_og = vec![sd.input_shape_og.product()].into();
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

    if let Some(garbage_pad) = sd.ignore_garbage_pad.as_ref() {
        garbage_pad.pad_matrix_to_ignore_garbage(&mut d.matrix, vec![nrows, ncols].into())?;
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
        (OperandMatrix::Weight(m), OperandMatrix::Input) => {
            let nrows = pad_minimum(m.tensor.nrows_2d());
            let ncols = padded_input_shapes[0][0];
            m.tensor
                .reshape_to_fit_inplace_2d(vec![nrows, ncols].into());
            (
                m.tensor.get_shape(),
                padded_input_shapes.pop().unwrap(), /* safe to unwrap since we checked the number of inputs at the beginning */
            )
        }
        (OperandMatrix::Input, OperandMatrix::Weight(m)) => {
            let nrows = padded_input_shapes[0][1];
            let ncols = pad_minimum(m.tensor.ncols_2d());
            let padded_matrix_shape = vec![nrows, ncols].into();
            // check if there is garbage pad: this is the only case we support in matrix mul where there
            // could be garbage pad
            if let Some(garbage_pad) = &si.shapes[0].ignore_garbage_pad {
                garbage_pad.pad_matrix_to_ignore_garbage(&mut m.tensor, padded_matrix_shape)?;
                si.shapes[0].ignore_garbage_pad = None;
            } else {
                m.tensor.reshape_to_fit_inplace_2d(padded_matrix_shape)
            };
            (padded_input_shapes.pop().unwrap(), m.tensor.get_shape())
        }
        (OperandMatrix::Input, OperandMatrix::Input) => {
            let right_shape = padded_input_shapes.pop().unwrap();
            let left_shape = padded_input_shapes.pop().unwrap();
            (left_shape, right_shape)
        }
        (OperandMatrix::Weight(_), OperandMatrix::Weight(_)) => {
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
        input_shape_og: unpadded_output_shape,
        input_shape_padded: vec![left_shape[0], right_shape[1]].into(),
        ignore_garbage_pad: None,
    }];
    if let Some(mut bias) = mat.bias {
        bias.pad_to_shape(right_shape.slice(1..));
        mat.bias = Some(bias);
    }
    Ok(mat)
}

pub(crate) fn pad_qkv(mut qkv: QKV<Element>, si: &mut ShapeInfo) -> Result<QKV<Element>> {
    // qkv layer currently expects 1 input, so we check there is only 1 input shape
    ensure!(
        si.shapes.len() == 1,
        "More than 1 input shape found when padding qkv layer"
    );
    let sd = si.shapes.first_mut().unwrap();

    ensure!(
        sd.input_shape_og.rank() == 2,
        "Unpadded input shape for QKV is not 2D"
    );
    ensure!(
        sd.input_shape_padded.rank() == 2,
        "Padded input shape for QKV is not 2D"
    );

    let unpadded_output_shapes = qkv.output_shapes(
        std::slice::from_ref(&sd.input_shape_og),
        PaddingMode::NoPadding,
    );
    let expected_num_outputs = qkv.num_outputs(1);
    ensure!(
        unpadded_output_shapes.len() == expected_num_outputs,
        "Expected {expected_num_outputs} unpadded output shapes for QKV layer, found {}",
        unpadded_output_shapes.len(),
    );

    ensure!(
        sd.input_shape_padded
            .as_ref()
            .iter()
            .all(|d| d.is_power_of_two()),
        "Padded input shapes for QKV layer are not a power of 2"
    );

    // Pad weight matrices
    let head_dim = qkv.head_dim;
    let padded_head_dim = pad_minimum(head_dim);
    let padded_num_heads = pad_minimum(qkv.num_heads);
    [&mut qkv.q, &mut qkv.k, &mut qkv.v].into_iter().try_for_each(|weight_mat| {
        ensure!(weight_mat.nrows_2d() <= sd.input_shape_padded.dim(1),
            "Weight matrices in QKV layer has more rows than the number of columns of padded input shapes: Expected at most {} rows, found {}",
            sd.input_shape_padded.dim(1), weight_mat.nrows_2d(),
        );

        weight_mat.reshape_in_place(Shape::new(vec![
            weight_mat.nrows_2d(),
            qkv.num_heads,
            head_dim,
        ]));
        let nrows = pad_minimum(sd.input_shape_padded.dim(1));
        weight_mat.pad_to_shape(
            vec![nrows, padded_num_heads, padded_head_dim].into()
        );
        weight_mat.reshape_in_place(Shape::new(vec![
            nrows,
            padded_num_heads*padded_head_dim,
        ]));
        Ok(())
    })?;

    // Pad bias vectors
    [&mut qkv.q_bias, &mut qkv.k_bias, &mut qkv.v_bias]
        .into_iter()
        .for_each(|bias_vec| {
            bias_vec.reshape_in_place(Shape::new(vec![qkv.num_heads, head_dim]));
            bias_vec.pad_to_shape(vec![padded_num_heads, padded_head_dim].into());
            bias_vec.reshape_in_place(Shape::new(vec![padded_num_heads * padded_head_dim]))
        });

    let padded_output_shapes = qkv.output_shapes(
        std::slice::from_ref(&sd.input_shape_padded),
        PaddingMode::Padding,
    );
    ensure!(
        unpadded_output_shapes.len() == padded_output_shapes.len(),
        "Number of unpadded output shapes different from number of padded output shapes for QKV layer"
    );

    ensure!(
        sd.ignore_garbage_pad.is_none(),
        "QKV layer has garbage padding to be removed",
    );

    si.shapes = unpadded_output_shapes
        .into_iter()
        .zip(padded_output_shapes)
        .map(|(unpadded_shape, padded_shape)| ShapeData {
            input_shape_padded: padded_shape,
            ignore_garbage_pad: None,
            input_shape_og: unpadded_shape,
        })
        .collect();

    Ok(qkv)
}

pub(crate) fn pad_concat_mat_mul(mat: ConcatMatMul, si: &mut ShapeInfo) -> Result<ConcatMatMul> {
    // no padding is needed since we don't have constant matrices in this layer
    // So, we check input shapes are padded, and we update shape info
    ensure!(
        si.shapes.len() == 2,
        "Expected 2 input shapes when padding ConcatMatMul layer, found {}",
        si.shapes.len(),
    );
    let unpadded_input_shapes = si.unpadded_input_shapes();

    mat.ensure_shape_consistency(&unpadded_input_shapes)?;

    let unpadded_output_shapes = mat.output_shapes(&unpadded_input_shapes, PaddingMode::NoPadding);
    let expected_num_outputs = mat.num_outputs(2);
    ensure!(
        unpadded_output_shapes.len() == expected_num_outputs,
        "Expected {expected_num_outputs} unpadded output shapes when padding ConcatMatMul, found {}",
        unpadded_output_shapes.len(),
    );

    let padded_input_shapes = si.padded_input_shapes();

    mat.ensure_shape_consistency(&padded_input_shapes)?;

    padded_input_shapes.iter().try_for_each(|s| {
        ensure!(
            s.is_power_of_two(),
            "Padded input shape for ConcatMatMul is not properly padded"
        );
        Ok(())
    })?;

    let padded_output_shapes = mat.output_shapes(&padded_input_shapes, PaddingMode::Padding);

    ensure!(
        padded_output_shapes.len() == expected_num_outputs,
        "Expected {expected_num_outputs} padded output shapes when padding ConcatMatMul, found {}",
        unpadded_output_shapes.len(),
    );

    ensure!(
        si.shapes.iter().all(|sd| sd.ignore_garbage_pad.is_none()),
        "ConcatMatMul layer has garbage padding to be removed",
    );

    si.shapes = unpadded_output_shapes
        .into_iter()
        .zip(padded_output_shapes)
        .map(|(unpadded, padded)| ShapeData {
            input_shape_padded: padded,
            ignore_garbage_pad: None,
            input_shape_og: unpadded,
        })
        .collect_vec();

    Ok(mat)
}

pub(crate) fn pad_reshape_layer(reshape: Reshape, si: &mut ShapeInfo) -> Result<Reshape> {
    let unpadded_output_shapes =
        reshape.output_shapes(&si.unpadded_input_shapes(), PaddingMode::NoPadding);

    let padded_output_shapes =
        reshape.output_shapes(&si.padded_input_shapes(), PaddingMode::Padding);

    ensure!(
        unpadded_output_shapes.len() == padded_output_shapes.len(),
        "Different number of unpadded output shapes and padded output shapes: {} vs {}",
        unpadded_output_shapes.len(),
        padded_output_shapes.len(),
    );

    // pad reshape depending on the type of reshape operation
    let reshape = reshape.to_padded_reshape();

    si.shapes
        .iter_mut()
        .zip(unpadded_output_shapes)
        .zip(padded_output_shapes)
        .for_each(|((sd, unpadded_shape), padded_shape)| {
            sd.input_shape_og = unpadded_shape;
            sd.input_shape_padded = padded_shape;
        });

    Ok(reshape)
}

fn pad_minimum(dim: usize) -> usize {
    let r = dim.next_power_of_two();
    if r < 4 { 4 } else { r }
}
