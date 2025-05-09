use anyhow::{Context, Result, ensure};

use crate::{
    Element,
    layers::{Layer, convolution::Convolution, dense::Dense, flatten::Flatten},
    model::Model,
    onnx_parse::{check_filter, safe_conv2d_shape, safe_maxpool2d_shape},
};
type GarbagePad = Option<(Vec<usize>, Vec<usize>)>;
type Shape = Vec<usize>;

#[derive(Clone, Debug, Copy)]
pub enum PaddingMode {
    NoPadding,
    Padding,
}

#[derive(Clone, Debug)]
struct ShapeInfo {
    input_shape_padded: Shape,
    ignore_garbage_pad: GarbagePad,
    input_shape_og: Shape,
}

pub fn pad_model(mut model: Model<Element>) -> Result<Model<Element>> {
    let mut si = ShapeInfo {
        input_shape_padded: model
            .unpadded_input
            .iter()
            .map(|i| i.next_power_of_two())
            .collect::<Vec<_>>(),
        ignore_garbage_pad: None,
        input_shape_og: model.unpadded_input.clone(),
    };
    model.layers = model
        .layers
        .into_iter()
        .enumerate()
        .map(|(_i, layer)| {
            match layer {
                Layer::Dense(d) => Ok(Layer::Dense(pad_dense(d, &mut si)?)),
                Layer::Convolution(c) => Ok(Layer::Convolution(pad_conv(c, &mut si)?)),
                Layer::Pooling(m) => {
                    // Make sure that input shape is already padded and is well formed
                    assert!(si.input_shape_padded.iter().all(|d| d.is_power_of_two()));
                    si.input_shape_og = safe_maxpool2d_shape(&si.input_shape_og)?;
                    si.input_shape_padded = safe_maxpool2d_shape(&si.input_shape_padded)?;
                    Ok(Layer::Pooling(m))
                }
                Layer::Flatten(_) => Ok(Layer::<Element>::Flatten(reshape(&mut si)?)),
                e => Ok(e),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(model)
}

fn reshape(si: &mut ShapeInfo) -> Result<Flatten> {
    si.ignore_garbage_pad = Some((si.input_shape_og.clone(), si.input_shape_padded.clone()));
    Ok(Flatten)
}

fn pad_conv(c: Convolution<Element>, si: &mut ShapeInfo) -> Result<Convolution<Element>> {
    si.input_shape_og = safe_conv2d_shape(&si.input_shape_og, &c.filter.get_shape())?;
    let weight_shape = c.filter.get_shape();
    // Perform basic sanity checks on the tensor dimensions
    check_filter(&weight_shape).context("filter shape test failed:")?;
    ensure!(
        weight_shape[0] == c.bias.get_shape()[0],
        "Bias length doesn't match filter shape"
    );
    // Make sure that input shape is already padded and is well formed
    ensure!(si.input_shape_padded.iter().all(|d| d.is_power_of_two()));
    ensure!(si.input_shape_padded.len() == 3);

    let new_conv_good = c.clone();
    // Since we are doing an FFT based conv, we need to pad the last two dimensions of the filter to match the input.
    let weight_shape = c.filter.pad_next_power_of_two().get_shape();
    let (filter_height, filter_width) = (weight_shape[2], weight_shape[3]);
    let (input_height, input_width) = (si.input_shape_padded[1], si.input_shape_padded[2]);

    ensure!(
        filter_height <= input_height && filter_width <= input_width,
        "Filter dimensions have to be smaller than input dimensions"
    );

    let new_conv = new_conv_good.into_padded_and_ffted(&si.input_shape_og);
    let output_shape = safe_conv2d_shape(&si.input_shape_padded, &weight_shape)?;
    si.input_shape_padded = output_shape
        .iter()
        .map(|i| i.next_power_of_two())
        .collect::<Vec<_>>();
    Ok(new_conv)
}

fn pad_dense(mut d: Dense<Element>, si: &mut ShapeInfo) -> Result<Dense<Element>> {
    let nrows = d.matrix.get_shape()[0];
    si.input_shape_og = vec![nrows];
    ensure!(
        d.bias.get_data().len() == nrows,
        "bias length {} does not match matrix width {}",
        d.bias.get_data().len(),
        nrows
    );
    ensure!(si.input_shape_padded.iter().all(|d| d.is_power_of_two()));
    if si.input_shape_padded.len() != 1 {
        si.input_shape_padded = vec![si.input_shape_padded.iter().product()];
        si.input_shape_og = vec![si.input_shape_og.iter().product()];
    }
    let mut new_cols = d.matrix.ncols_2d();
    if d.matrix.ncols_2d() != si.input_shape_padded[0] {
        if d.matrix.ncols_2d() < si.input_shape_padded[0] {
            new_cols = si.input_shape_padded[0];
        } else {
            // If we have too many columns, we can't shrink without losing information
            anyhow::bail!(
                "Matrix has more columns ({}) than previous layer output size ({}).
                            Cannot shrink without losing information.",
                d.matrix.ncols_2d(),
                si.input_shape_padded[0]
            );
        }
    }
    let ncols = new_cols.next_power_of_two();
    let nrows = d.matrix.nrows_2d().next_power_of_two();

    if let Some(ref previous_shape) = si.ignore_garbage_pad.as_ref() {
        let previous_input_shape_og = previous_shape.0.clone();
        let previous_input_shape_padded = previous_shape.1.clone();
        d.matrix = d.matrix.pad_matrix_to_ignore_garbage(
            &previous_input_shape_og,
            &previous_input_shape_padded,
            &vec![nrows, ncols],
        );
        si.ignore_garbage_pad = None;
    } else {
        d.matrix.reshape_to_fit_inplace_2d(vec![nrows, ncols]);
    }
    d.bias = d.bias.pad_1d(nrows);
    si.input_shape_padded = vec![nrows];
    Ok(d)
}
