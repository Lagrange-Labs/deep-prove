use crate::{
    layers::activation::Activation,
    quantization::{
        BIT_LEN,
        metadata::{MetadataBuilder, ModelMetadata},
    },
};
use std::collections::HashMap;

use crate::{
    Element, Tensor,
    layers::{Layer, LayerOutput, requant::Requant},
    model::Model,
    quantization,
};
use anyhow::{Result, bail, ensure};
use statrs::statistics::{Data, Distribution, Max, Min, OrderStatistics};
use tracing::{debug, info, warn};
use tract_onnx::tract_core::ops::matmul::quant;

use super::ScalingFactor;

/// Trait for quantizing a float-based model into a quantized model. The current implementation
/// simply looks at the absolute maximum value of the model and uses that as the scaling factor
/// to quantize the model, one scaling factor per layer.
pub trait ScalingStrategy: std::fmt::Debug {
    fn quantize(&self, model: Model<f32>) -> Result<(Model<Element>, ModelMetadata)>;
    fn name(&self) -> String;
}

/// Quantization strategy that observes the inference of the model with different inputs and uses the
/// min/max values of the output to determine the output scaling factor of each layer that needs
/// requantization afterwards.
#[derive(Debug)]
pub struct InferenceObserver {
    inputs: Vec<Vec<f32>>,
}

impl InferenceObserver {
    pub fn new_with_representative_input(inputs: Vec<Vec<f32>>) -> Self {
        Self { inputs }
    }
}

const INPUT_TRACKING_ID: usize = 10_000;
impl ScalingStrategy for InferenceObserver {
    fn name(&self) -> String {
        format!("inference [{},{}]", *quantization::MIN, *quantization::MAX)
    }
    fn quantize(&self, model: Model<f32>) -> Result<(Model<Element>, ModelMetadata)> {
        let mut tracker = InferenceTracker::new();
        let input_shape = model.input_shape();
        let input_not_padded_shape = model.input_not_padded();
        // 1. Run the inference multiple times with different inputs
        // TODO: integrate that within model.rs in a more elegant way with inference step - currently problematic
        // because of the generics and FFT requirement to take a field
        for (i, input) in self.inputs.iter().enumerate() {
            // let input_tensor = model.load_input_flat(input.clone());
            let input_tensor = Tensor::new(model.input_not_padded.clone(), input.clone());
            // ensure!(
            //    model.input_shape() == input_tensor.get_shape(),
            //    "input shape mismatch: expected {:?}, got {:?}",
            //    model.input_shape(),
            //    input_tensor.get_shape()
            //);
            let mut last_output = input_tensor;
            tracker.track(INPUT_TRACKING_ID, last_output.clone());
            for (id, layer) in model.layers.iter().enumerate() {
                // debug!(
                //    "Inference Observer: inference run #{}: running layer {}",
                //    i,
                //    layer.describe()
                //);
                last_output = run_layer(layer, &last_output);
                tracker.track(id, last_output.clone());
            }
        }
        let mut md = MetadataBuilder::new();
        // 2. get the scaling factor of the input
        let (input_min, input_max) = tracker.distribution_info(INPUT_TRACKING_ID);
        let input_scaling =
            ScalingFactor::from_absolute_max(input_min.abs().max(input_max.abs()), None);
        md.set_input_scaling(input_scaling);
        let mut last_input_scaling = input_scaling.clone();
        // 2. Create the requant layers from the infered data
        let quantized_layers = model
            .layers
            .into_iter()
            .enumerate()
            .map(|(id, layer)| {
                match layer {
                    Layer::Dense(dense) => {
                        let model_scaling = ScalingFactor::from_absolute_max(dense.max_abs_weight(), None);
                        let (min, max) = tracker.distribution_info(id);
                        let output_scaling = ScalingFactor::from_absolute_max(min.abs().max(max.abs()), None);
                        let scale = last_input_scaling.m(&model_scaling, &output_scaling);
                        let bias_scaling = {
                                // bias has to be quantized over integers with double bit length
                                let min_quantized = -(1 << (2*(*BIT_LEN) - 1)) + 1;
                                let max_quantized = (1 << (2*(*BIT_LEN) - 1)) - 1;
                                ScalingFactor::from_scale(
                                    last_input_scaling.scale() * model_scaling.scale(),
                                    Some((min_quantized, max_quantized))
                                )
                        };
                        println!("InferenceObserver: DENSE {} layer model max weight abs {:?}", id, dense.max_abs_weight());
                        println!("InferenceObserver: DENSE {} layer output range [{:?}, {:?}]", id, min,max);
                        println!("InferenceObserver: DENSE {} layer scales: input {:?}, model {:?}, bias {:?}, output {:?} -> scale {:?}", id, last_input_scaling.scale(), model_scaling.scale(), bias_scaling.scale(), output_scaling.scale(), scale);
                        let shift = last_input_scaling.shift(&model_scaling, &output_scaling);
                        let quantized_dense = dense.quantize(&model_scaling, Some(&bias_scaling));
                        let (quantized_min, _quantized_max) =
                            quantized_dense.output_range(*quantization::MIN, *quantization::MAX);
                        let requant = Requant {
                            right_shift: shift,
                            range: quantized_min.abs() as usize,
                            after_range: 1 << *quantization::BIT_LEN,
                            multiplier: scale,
                        };
                        md.set_layers_scaling(id, output_scaling);
                        last_input_scaling = output_scaling;
                        vec![Layer::Dense(quantized_dense), Layer::Requant(requant)]
                    }
                    Layer::SchoolBookConvolution(conv) => {
                        let model_scaling = ScalingFactor::from_absolute_max(conv.max_abs_weight(), None);
                        let (min, max) = tracker.distribution_info(id);
                        let output_scaling = ScalingFactor::from_absolute_max(min.abs().max(max.abs()), None);
                        let scale = last_input_scaling.scale() * model_scaling.scale() / output_scaling.scale();
                        let bias_scaling = {
                            // bias has to be quantized over integers with double bit length
                            let min_quantized = -(1 << (2*(*BIT_LEN) - 1)) + 1;
                            let max_quantized = (1 << (2*(*BIT_LEN) - 1)) - 1;
                            ScalingFactor::from_scale(
                                last_input_scaling.scale() * model_scaling.scale(),
                                Some((min_quantized, max_quantized))
                            )
                        };
                        println!("InferenceObserver: CONV {} layer model max weight abs {:?}", id, conv.max_abs_weight());
                        println!("InferenceObserver: CONV {} layer output range [{:?}, {:?}]", id, min,max);
                        println!("InferenceObserver: CONV {} layer scales: input {:?}, model {:?}, bias {:?}, output {:?} -> scale {:?}", id, last_input_scaling.scale(), model_scaling.scale(), bias_scaling.scale(), output_scaling.scale(), scale);
                        let quantized_conv = conv.quantize(&model_scaling, Some(&bias_scaling));
                        let shift = last_input_scaling.shift(&model_scaling, &output_scaling);
                        let (quantized_min, _quantized_max) =
                            quantized_conv.output_range(*quantization::MIN, *quantization::MAX);
                        md.set_layers_scaling(id, output_scaling);
                        let requant = Requant {
                            right_shift: shift,
                            range: quantized_min.abs() as usize,
                            after_range: 1 << *quantization::BIT_LEN,
                            multiplier: scale,
                        };
                        last_input_scaling = output_scaling;
                        vec![Layer::Convolution(quantized_conv), Layer::Requant(requant)]
                    }
                    //Layer::Activation(Activation::Relu(r)) => {
                    //    let max_abs = last_input_scaling.max;
                    //    // HERE we use the half of the real span but still symmetric in the quantized space
                    //    last_input_scaling = ScalingFactor::from_span(0.0, max_abs);
                    //    vec![Layer::Activation(Activation::Relu(r))]
                    //}
                    a => return vec![a.quantize(
                        &last_input_scaling,
                        None // no scaling factor for bias needed for this layer
                    )],
                }
            })
            .flatten()
            .collect::<Vec<_>>();
        Ok((
            Model::<Element>::new_from(quantized_layers, input_not_padded_shape, input_shape),
            md.build(),
        ))
    }
}

fn run_layer(layer: &Layer<f32>, input: &Tensor<f32>) -> Tensor<f32> {
    match layer {
        Layer::Dense(ref dense) => dense.op(input),
        Layer::Activation(activation) => activation.op(input),
        Layer::Convolution(ref conv_pair) => {
            // println!("RUN CONV: input shape {:?} (4d {:?}), filter shape {:?}, bias shape {:?} -- filter 4d {:?}", input.get_shape(),input.get4d(), conv_pair.filter.get_shape(), conv_pair.bias.get_shape(),conv_pair.filter.get4d());
            // if input.get_shape().len() == 3 {
            //    Tensor::new(input.get_shape(),input.get_data().to_vec());
            //    let input_shape_prod = input.get_shape().iter().product::<usize>();
            //    assert_eq!(input_shape_prod,input.get_data().len());
            //    let augmented_shape = vec![1, input.get_shape()[0], input.get_shape()[1], input.get_shape()[2]];
            //    let new_input_shape_prod = augmented_shape.iter().product::<usize>();
            //    assert_eq!(new_input_shape_prod,input_shape_prod,"augmented input shape {:?} vs original shape {:?}",augmented_shape,input.get_shape());
            //    assert_eq!(augmented_shape.iter().product::<usize>(),input.get_data().len());
            //    println!("\t Extending input shape to 4d to {:?}",augmented_shape);
            //    let conv_input = Tensor::new(augmented_shape, input.get_data().to_vec());
            //    conv_input.conv2d(&conv_pair.filter, &conv_pair.bias, 1)
            //} else {
            input.conv2d(&conv_pair.filter, &conv_pair.bias, 1)
            //}
        }
        Layer::Pooling(info) => info.op(input),
        // Traditional convolution is used for debug purposes. That is because the actual convolution
        // we use relies on the FFT algorithm. This convolution does not have a snark implementation.
        Layer::SchoolBookConvolution(ref conv_pair) => {
            input.conv2d(&conv_pair.filter, &conv_pair.bias, 1)
        }
        Layer::Requant(_) => {
            panic!(
                "InferenceObserver: requantization layer found while observing inference on float !?"
            );
        }
    }
}
struct InferenceTracker {
    /// For each layer of interest, we track all the outputs of that layer
    data: HashMap<usize, Vec<f64>>,
}

impl InferenceTracker {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
    fn track(&mut self, layer_index: usize, output: Tensor<f32>) {
        self.data
            .entry(layer_index)
            .or_insert(Vec::new())
            .extend(output.get_data().iter().map(|x| *x as f64));
    }

    /// Returns the 0.05 and 0.95 quantiles of the distribution of the output values of the layer.
    fn distribution_info(&self, layer_index: usize) -> (f32, f32) {
        let mut d: Data<Vec<f64>> = Data::new(self.data[&layer_index].clone());
        let min = d.percentile(5) as f32;
        let max = d.percentile(95) as f32;
        assert!(min <= max);
        //(min, max)
        (d.min() as f32, d.max() as f32)
        // let mean = d.mean().unwrap();
        // let std_dev = d.std_dev().unwrap();
        // let upper_bound = mean + 3.0 * std_dev;
        // let lower_bound = mean - 3.0 * std_dev;
        //(lower_bound as f32, upper_bound as f32)
    }
}

#[derive(Debug)]
pub struct AbsoluteMax(Option<Vec<f32>>);

impl AbsoluteMax {
    pub fn new_with_representative_input(input: Vec<f32>) -> Self {
        Self(Some(input))
    }
    pub fn new() -> Self {
        Self(None)
    }
}
impl ScalingStrategy for AbsoluteMax {
    fn name(&self) -> String {
        "absolute_max".to_string()
    }
    fn quantize(&self, model: Model<f32>) -> Result<(Model<Element>, ModelMetadata)> {
        let mut last_input_scaling_factor = if let Some(ref input) = self.0 {
            let input_tensor = model.load_input_flat(input.clone());
            ensure!(
                model.input_shape() == input_tensor.get_shape(),
                "input shape mismatch: expected {:?}, got {:?}",
                model.input_shape(),
                input_tensor.get_shape()
            );
            ScalingFactor::from_absolute_max(input_tensor.max_abs_output(), None)
        } else {
            ScalingFactor::default()
        };
        let mut md = MetadataBuilder::new();
        md.set_input_scaling(last_input_scaling_factor.clone());
        let input_shape = model.input_shape();
        let input_not_padded_shape = model.input_not_padded();
        let quantized_layers = model
            .layers
            .into_iter()
            .enumerate()
            .flat_map(|(id,l)| {
                // If a layer requires a requantization step the current layer, this method returns the
                // next layer, e.g. requantization layer, as well as the scaling factor of the output. This is
                // given to the next layer as input scaling factor.
                match l {
                    Layer::Dense(d) => {
                        let max_weight = d.max_abs_weight();
                        let model_scaling = ScalingFactor::from_absolute_max(max_weight, None);
                        //let (min_output,max_output) = d.output_range(-last_input_scaling_factor.abs_max,last_input_scaling_factor.abs_max);
                        let bias_scaling = {
                            // bias has to be quantized over integers with double bit length
                            let min_quantized = -(1 << (2*(*BIT_LEN) - 1)) + 1;
                            let max_quantized = (1 << (2*(*BIT_LEN) - 1)) - 1;
                            ScalingFactor::from_scale(
                                last_input_scaling_factor.scale() * model_scaling.scale(),
                                Some((min_quantized, max_quantized))
                            )
                        };
                        let quantized_dense= d.quantize(&model_scaling, Some(&bias_scaling));
                        let (quant_min_output,_quant_max_output) = quantized_dense.output_range(*quantization::MIN,*quantization::MAX);
                        //let abs_max = min_output.abs().max(max_output.abs());
                        //let output_scaling = ScalingFactor::new(abs_max);
                        // TODO: remove this is broken
                        let output_scaling = ScalingFactor::default();
                        last_input_scaling_factor = output_scaling;
                        md.set_layers_scaling(id, output_scaling);
                        let shift = last_input_scaling_factor.shift(&model_scaling, &output_scaling);
                        println!("Scaling: AbsoluteMax: CONV max_weight {:?}, max_output: {:?} - adding requant", max_weight, 1.0);
                        let requant = Requant {
                            right_shift: shift,
                            range: quant_min_output.abs() as usize,
                            after_range: 1 << *quantization::BIT_LEN,
                            multiplier: 1.0,
                        };
                        vec![Layer::Dense(quantized_dense), Layer::Requant(requant)]

                    }
                    Layer::Convolution(d) => {
                        let max_weight = d.max_abs_weight();
                        let model_scaling = ScalingFactor::from_absolute_max(max_weight, None);
                        //let (min_output,max_output) = d.output_range(-last_input_scaling_factor.abs_max,last_input_scaling_factor.abs_max);
                        let bias_scaling = {
                            // bias has to be quantized over integers with double bit length
                            let min_quantized = -(1 << (2*(*BIT_LEN) - 1)) + 1;
                            let max_quantized = (1 << (2*(*BIT_LEN) - 1)) - 1;
                            ScalingFactor::from_scale(
                                last_input_scaling_factor.scale() * model_scaling.scale(),
                                Some((min_quantized, max_quantized))
                            )
                        };
                        let quantized_conv= d.quantize(&model_scaling, Some(&bias_scaling));
                        let (quant_min_output,_quant_max_output) = quantized_conv.output_range(*quantization::MIN,*quantization::MAX);
                        //let abs_max = min_output.abs().max(max_output.abs());
                        //let output_scaling = ScalingFactor::new(abs_max);
                        // TODO: remove this is broken
                        let output_scaling = ScalingFactor::default();
                        last_input_scaling_factor = output_scaling;
                        md.set_layers_scaling(id, output_scaling);
                        let shift = last_input_scaling_factor.shift(&model_scaling, &output_scaling);
                        println!("Scaling: AbsoluteMax: CONV max_weight {:?}, max_output: {:?} - adding requant", max_weight, 1.0);
                        let requant = Requant {
                            right_shift: shift,
                            range: quant_min_output.abs() as usize,
                            after_range: 1 << *quantization::BIT_LEN,
                            multiplier: 1.0,
                        };
                        vec![Layer::Convolution(quantized_conv), Layer::Requant(requant)]
                    }
                    a => return vec![a.quantize(
                        &last_input_scaling_factor,
                        None // no scaling factor for bias needed for this layer
                    )],
                }
            })
            .collect::<Vec<Layer<Element>>>();
        Ok((
            Model::<Element>::new_from(quantized_layers, input_not_padded_shape, input_shape),
            md.build(),
        ))
    }
}
