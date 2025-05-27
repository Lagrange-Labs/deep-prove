use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::{
    Element,
    layers::{
        convolution::Convolution,
        dense::Dense,
        flatten::Flatten,
        pooling::Pooling,
        provable::{Node, NodeId},
    },
    model::{Model, ToIterator},
    parser::{check_filter, safe_conv2d_shape, safe_maxpool2d_shape},
};
type GarbagePad = Option<(Vec<usize>, Vec<usize>)>;
type Shape = Vec<usize>;

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
        sd.ignore_garbage_pad = Some((sd.input_shape_og.clone(), sd.input_shape_padded.clone()))
    });
    Ok(Flatten)
}

pub(crate) fn pooling(p: Pooling, si: &mut ShapeInfo) -> Result<Pooling> {
    for sd in si.shapes.iter_mut() {
        // Make sure that input shape is already padded and is well formed
        ensure!(
            sd.input_shape_padded.iter().all(|d| d.is_power_of_two()),
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
        sd.input_shape_padded.iter().all(|d| d.is_power_of_two()),
        "Input shape for convolution is not padded",
    );
    ensure!(
        sd.input_shape_padded.len() == 3,
        "Input shape for convolution is not 3D"
    );
    let new_conv_good = c.clone();
    // Since we are doing an FFT based conv, we need to pad the last two dimensions of the filter to match the input.
    let weight_shape = c.filter.pad_next_power_of_two().get_shape();
    let (filter_height, filter_width) = (weight_shape[2], weight_shape[3]);
    let (input_height, input_width) = (sd.input_shape_padded[1], sd.input_shape_padded[2]);

    ensure!(
        filter_height <= input_height && filter_width <= input_width,
        "Filter dimensions in convolution have to be smaller than input dimensions",
    );

    let new_conv = new_conv_good.into_padded_and_ffted(&sd.input_shape_og);
    let output_shape = safe_conv2d_shape(&sd.input_shape_padded, &weight_shape)?;
    sd.input_shape_padded = output_shape
        .iter()
        .map(|i| i.next_power_of_two())
        .collect::<Vec<_>>();
    Ok(new_conv)
}

pub(crate) fn pad_dense(mut d: Dense<Element>, si: &mut ShapeInfo) -> Result<Dense<Element>> {
    // dense layer currently expects 1 input, so we check there is only 1 input shape
    ensure!(
        si.shapes.len() == 1,
        "More than 1 input shape found when padding dense layer"
    );
    let sd = si.shapes.first_mut().unwrap();
    let nrows = d.matrix.get_shape()[0];
    sd.input_shape_og = vec![nrows];
    ensure!(
        d.bias.get_data().len() == nrows,
        "Bias length {} does not match matrix width {}",
        d.bias.get_data().len(),
        nrows,
    );
    ensure!(
        sd.input_shape_padded.iter().all(|d| d.is_power_of_two()),
        "Input shape for dense is not padded"
    );
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
            bail!(
                "Dense layer matrix has more columns ({}) than previous layer output size ({}).
                            Cannot shrink without losing information.",
                d.matrix.ncols_2d(),
                sd.input_shape_padded[0]
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
            &previous_input_shape_og,
            &previous_input_shape_padded,
            &[nrows, ncols],
        );
        sd.ignore_garbage_pad = None;
    } else {
        d.matrix.reshape_to_fit_inplace_2d(vec![nrows, ncols]);
    }
    d.bias = d.bias.pad_1d(nrows);
    sd.input_shape_padded = vec![nrows];
    Ok(d)
}

fn pad_minimum(dim: usize) -> usize {
    let r = dim.next_power_of_two();
    if r < 4 { 4 } else { r }
}
