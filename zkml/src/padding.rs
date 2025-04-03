use anyhow::{ensure, Context, Result};
use tracing::debug;

use crate::{layers::{convolution::Convolution, dense::Dense, Layer}, model::Model, onnx_parse::{check_cnn_input, check_filter, conv2d_shape, maxpool2d_shape}, Element};
type GarbagePad = Option<(Vec<usize>, Vec<usize>)>;
type Shape = Vec<usize>;

#[derive(Clone, Debug)]
struct ShapeInfo {
    input_shape_padded: Shape,
    ignore_garbage_pad: GarbagePad,
    input_shape_og: Shape,
}

pub fn pad_model(mut model: Model<Element>) -> Result<Model<Element>> {

    let mut si = ShapeInfo {
        input_shape_padded: model
            .input_not_padded
            .iter()
            .map(|i| i.next_power_of_two())
            .collect::<Vec<_>>(),
        ignore_garbage_pad: None,
        input_shape_og: model.input_not_padded.clone(),
    };
    println!("\n\nPADDING DENSE\n\n");
    println!("{:?}",model.describe());
    println!("input shape {:?}",si);
    model.layers = model
        .layers
        .into_iter()
        .enumerate()
        .map(|(i, layer)| {
            match layer {
            Layer::Dense(d) => Ok(Layer::Dense(pad_dense(d, &mut si)?)),
            Layer::Convolution(c) => Ok(Layer::Convolution(pad_conv(c, &mut si)?)),
            Layer::Pooling(m) =>  {
                println!("PAD POOLING: input shape {:?}",si);
                // Make sure that input shape is already padded and is well formed
                assert!(si.input_shape_padded.iter().all(|d| d.is_power_of_two()));
                si.input_shape_og = maxpool2d_shape(&si.input_shape_og)?;
                si.input_shape_padded = maxpool2d_shape(&si.input_shape_padded)?;
                Ok(Layer::Pooling(m))
            }
            e => Ok(e)
        }
        }).collect::<Result<Vec<_>>>()?;
    Ok(model)
}


fn pad_conv(mut c: Convolution<Element>, si: &mut ShapeInfo) -> Result<Convolution<Element>> {
    println!("PAD CONV: input shape {:?}",si);
    si.input_shape_og = conv2d_shape(&si.input_shape_og, &c.filter.get_shape())?;
    let weight_shape = c.filter.get_shape();
    // Perform basic sanity checks on the tensor dimensions
    check_filter(&weight_shape).context("filter shape test failed:")?;
    assert!(
        weight_shape[0] == c.bias.get_shape()[0],
        "Bias length doesn't match filter shape"
    );

    // Pad the tensors to the next power of two.
    c.filter = c.filter.pad_next_power_of_two();
    c.bias = c.bias.pad_next_power_of_two();

    // Make sure that input shape is already padded and is well formed
    assert!(si.input_shape_padded.iter().all(|d| d.is_power_of_two()));
    assert!(si.input_shape_padded.len() == 3);

    // Since we are doing an FFT based conv, we need to pad the last two dimensions of the filter to match the input.
    let weight_shape = c.filter.get_shape();
    let (filter_height, filter_weight) = (weight_shape[2], weight_shape[3]);
    let (input_height, input_weight) = (si.input_shape_padded[1], si.input_shape_padded[2]);

    assert!(
        filter_height <= input_height && filter_weight <= input_weight,
        "Filter dimensions have to be smaller than input dimensions"
    );

    // weight = weight.pad_last_two_dimensions(vec![input_height, input_weight]);

    // Filter need to know the shape of the input
    // weight.update_input_shape(&input_shape_padded);

    let dims = c.filter.get_shape();
    // NOTE: This is a bit of a hack but given we compute the new padded filter shape at the same time we do the fft
    // we just do both here.
    c.filter = crate::tensor::Tensor::new_conv(
        c.filter.get_shape(),
        si.input_shape_padded.clone(),
        c.filter.get_data().to_vec(),
    );

    // let layer = Layer::SchoolBookConvolution(Convolution::new(weight, _bias));

    let output_shape = conv2d_shape(&si.input_shape_padded, &dims)?;
    si.input_shape_padded = output_shape
        .iter()
        .map(|i| i.next_power_of_two())
        .collect::<Vec<_>>();
    Ok(c)
}

fn pad_dense(mut d: Dense<Element>, si: &mut ShapeInfo) -> Result<Dense<Element>> {
    println!("PAD DENSE: input shape {:?}",si);
    let nrows = d.matrix.get_shape()[0];
    si.input_shape_og = vec![nrows];
    ensure!(
        d.bias.get_data().len() == nrows,
        "bias length {} does not match matrix width {}",
        d.bias.get_data().len(),
        nrows
    );
    assert!(si.input_shape_padded.iter().all(|d| d.is_power_of_two()));
    /// NOTE: this eventually should be handled by a reshape layer
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
            panic!(
                "Matrix has more columns ({}) than previous layer output size ({}).
                            Cannot shrink without losing information.",
                d.matrix.ncols_2d(),
                si.input_shape_padded[0]
            );
        }
    }
    let ncols = new_cols.next_power_of_two();
    let nrows = d.matrix.nrows_2d().next_power_of_two();
    if let Some(ref conv_shapes) = si.ignore_garbage_pad.as_ref() {
        let conv_shape_og = conv_shapes.0.clone();
        let conv_shape_pad = conv_shapes.1.clone();
        d.matrix = d
            .matrix
            .pad_matrix_to_ignore_garbage(&conv_shape_og, &conv_shape_pad, &vec![nrows, ncols]);
        si.ignore_garbage_pad = None;
    } else {
        d.matrix.reshape_to_fit_inplace_2d(vec![nrows, ncols]);
    }
    d.bias = d.bias.pad_1d(nrows);
    si.input_shape_padded = vec![nrows];
    Ok(d)
}