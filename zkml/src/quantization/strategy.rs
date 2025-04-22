use crate::{
    layers::requant::FullRequant,
    quantization::{
        BIT_LEN,
        metadata::{MetadataBuilder, ModelMetadata},
    },
    tensor::Number,
};
use std::collections::HashMap;

use crate::{
    Element, Tensor,
    layers::Layer,
    model::Model,
    quantization,
};
use anyhow::{Result, ensure};
use ark_std::rand;
use itertools::Itertools;
use multilinear_extensions::util::ceil_log2;
use statrs::statistics::{Data, Max, Min, OrderStatistics};
use tracing::{debug, info};

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
    pub fn new() -> Self {
        Self { inputs: vec![] }
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
        let inputs = if self.inputs.is_empty() {
            let size = input_not_padded_shape.iter().product();
            (0..10)
                .map(|_| {
                    (0..size)
                        .map(|_| <f32 as Number>::random(&mut rand::thread_rng()))
                        .collect_vec()
                })
                .collect_vec()
        } else {
            self.inputs.clone()
        };
        // 1. Run the inference multiple times with different inputs
        // TODO: integrate that within model.rs in a more elegant way with inference step - currently problematic
        // because of the generics and FFT requirement to take a field
        let mut nsamples = 0;
        for (i, input) in inputs.iter().enumerate() {
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
                debug!(
                    "Inference Observer: inference run #{}: running layer {}",
                    i,
                    layer.describe()
                );
                last_output = layer.run(&last_output);
                tracker.track(id, last_output.clone());
            }
            nsamples += 1;
        }
        info!("InferenceObserver: {} samples observed", nsamples);
        let mut md = MetadataBuilder::new();
        // 2. get the scaling factor of the input
        let (input_min, input_max) = tracker.distribution_info(INPUT_TRACKING_ID);
        let input_scaling =
            ScalingFactor::from_absolute_max(input_min.abs().max(input_max.abs()), None);
        md.set_input_scaling(input_scaling);
        let mut last_input_scaling = input_scaling;
        // 2. Create the requant layers from the infered data
        // manually take care of updating the step_idx since we are adding layers here
        let mut step_idx = 0;
        let quantized_layers = model
            .layers
            .into_iter()
            .enumerate()
            .map(|(id,layer)| {
                match layer {
                    Layer::Dense(dense) => {
                        let model_scaling =
                            ScalingFactor::from_absolute_max(dense.max_abs_weight(), None);
                        let (min, max) = tracker.distribution_info(id);
                        let output_scaling =
                            ScalingFactor::from_absolute_max(min.abs().max(max.abs()), None);
                        let _bias_scaling = {
                            // bias has to be quantized over integers with double bit length
                            let min_quantized = -(1 << (2 * (*BIT_LEN) - 1)) + 1;
                            let max_quantized = (1 << (2 * (*BIT_LEN) - 1)) - 1;
                            ScalingFactor::from_scale(
                                last_input_scaling.scale() * model_scaling.scale(),
                                Some((min_quantized, max_quantized)),
                            )
                        };
                        println!("InferenceObserver: DENSE {} layer",step_idx);

                        // println!("InferenceObserver: DENSE {} layer model max weight abs {:?}", id, dense.max_abs_weight());
                        // println!("InferenceObserver: DENSE {} layer output range [{:?}, {:?}]", id, min,max);
                        // println!("InferenceObserver: DENSE {} layer scales: input {:?}, model {:?}, bias {:?}, output {:?} -> scale {:?}", id, last_input_scaling.scale(), model_scaling.scale(), bias_scaling.scale(), output_scaling.scale(), scale);
                        
                        // let quantized_dense = dense.quantize(&model_scaling, Some(&_bias_scaling));
                        let quantized_dense = dense.quantize(&model_scaling, None);
                        let (quantized_min, _quantized_max) =
                            quantized_dense.output_range(*quantization::MIN, *quantization::MAX);
                        
                        // let scale = last_input_scaling.m(&model_scaling, &output_scaling);
                        // requant.set_test_multiplier(scale);
                        let intermediate_bit_size = ceil_log2(quantized_min.abs() as usize);
                        let full_requant = FullRequant::from_scaling_factors(last_input_scaling, model_scaling, output_scaling, intermediate_bit_size);
                        md.set_layers_scaling(step_idx, output_scaling);
                        last_input_scaling = output_scaling;
                        // because we are adding a new layer
                        step_idx += 2;
                        // vec![Layer::Dense(quantized_dense), Layer::Requant(requant)]
                        vec![Layer::Dense(quantized_dense), Layer::FullRequant(full_requant)]
                    }
                    Layer::Convolution(conv) => {
                        let model_scaling =
                            ScalingFactor::from_absolute_max(conv.max_abs_weight(), None);
                        let (min, max) = tracker.distribution_info(id);
                        let output_scaling =
                            ScalingFactor::from_absolute_max(min.abs().max(max.abs()), None);
                                                let _bias_scaling = {
                            // bias has to be quantized over integers with double bit length
                            let min_quantized = -(1 << (2 * (*BIT_LEN) - 1)) + 1;
                            let max_quantized = (1 << (2 * (*BIT_LEN) - 1)) - 1;
                            ScalingFactor::from_scale(
                                last_input_scaling.scale() * model_scaling.scale(),
                                Some((min_quantized, max_quantized)),
                            )
                        };
                        // println!("InferenceObserver: CONV {} layer model max weight abs {:?}", id, conv.max_abs_weight());
                        // println!("InferenceObserver: CONV {} layer output range [{:?}, {:?}]", id, min,max);
                        // println!("InferenceObserver: CONV {} layer scales: input {:?}, model {:?}, bias {:?}, output {:?} -> scale {:?}", id, last_input_scaling.scale(), model_scaling.scale(), bias_scaling.scale(), output_scaling.scale(), scale);
                        // let quantized_conv = conv.quantize(&model_scaling, Some(&_bias_scaling));
                        let quantized_conv = conv.quantize(&model_scaling, None);
                        
                        let (quantized_min, _quantized_max) =
                            quantized_conv.output_range(*quantization::MIN, *quantization::MAX);
                        //println!(
                        //    "InferenceObserver: CONV {} layer f32 output range [{:?}, {:?}], quantized output range [{:?}, {:?}]",
                        //    step_idx, min,max, quantized_min, _quantized_max
                        //);
                        let intermediate_bit_size = ceil_log2(quantized_min.abs() as usize);
                        let full_requant = FullRequant::from_scaling_factors(last_input_scaling, model_scaling, output_scaling, intermediate_bit_size);
                        md.set_layers_scaling(step_idx, output_scaling);
                        
                        {
                            // TEST
                            let all_data = tracker.data_for(id);
                            let quantized_data = all_data.iter().map(|x| output_scaling.quantize(x)).collect_vec();
                            let max_quantized = quantized_data.iter().max().unwrap();
                            let min_quantized = quantized_data.iter().min().unwrap();
                            println!("InferenceObserver: CONV {} layer f32 output quantized: min {},max {}", step_idx, min_quantized, max_quantized);
                            let max_weight = quantized_conv.filter.max_value();
                            let min_weight = quantized_conv.filter.min_value();
                            let max_bias = quantized_conv.bias.max_value();
                            let min_bias = quantized_conv.bias.min_value();
                            println!("InferenceObserver: CONV {} layer f32 weight quantized: min {},max {}", step_idx, min_weight, max_weight);
                            println!("InferenceObserver: CONV {} layer f32 bias quantized: min {},max {}", step_idx, min_bias, max_bias);
                        }
                        // let scale = last_input_scaling.scale() * model_scaling.scale()
                        //  / output_scaling.scale();
                        // requant.set_test_multiplier(scale);
                        last_input_scaling = output_scaling;
                        // because we are adding a new layer
                        step_idx += 2;
                        // vec![Layer::Convolution(quantized_conv), Layer::Requant(requant)]
                        vec![Layer::Convolution(quantized_conv), Layer::FullRequant(full_requant)]
                    }
                    // Layer::Activation(Activation::Relu(r)) => {
                    //    let max_abs = last_input_scaling.max;
                    //    // HERE we use the half of the real span but still symmetric in the quantized space
                    //    last_input_scaling = ScalingFactor::from_span(0.0, max_abs);
                    //    vec![Layer::Activation(Activation::Relu(r))]
                    //}
                    a => {
                        println!("InferenceObserver: ANY {} layer {}",step_idx,a.describe());
                        step_idx += 1;
                        return vec![a.quantize(
                            &last_input_scaling,
                            None, // no scaling factor for bias needed for this layer
                        )];
                    }
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

    fn data_for(&self, layer_index: usize) -> Vec<f32> {
        self.data
            .get(&layer_index)
            .expect(&format!("No data for layer {:?}", layer_index))
            .iter()
            .map(|x| *x as f32)
            .collect_vec()
    }

    /// Returns the 0.05 and 0.95 quantiles of the distribution of the output values of the layer.
    fn distribution_info(&self, layer_index: usize) -> (f32, f32) {
        let mut d: Data<Vec<f64>> = Data::new(
            self.data
                .get(&layer_index)
                .expect(&format!("No data for layer {:?}", layer_index))
                .clone(),
        );
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
                        let _bias_scaling = Some({
                            // bias has to be quantized over integers with double bit length
                            let min_quantized = -(1 << (2*(*BIT_LEN) - 1)) + 1;
                            let max_quantized = (1 << (2*(*BIT_LEN) - 1)) - 1;
                            ScalingFactor::from_scale(
                                last_input_scaling_factor.scale() * model_scaling.scale(),
                                Some((min_quantized, max_quantized))
                            )
                        });
                        let bias_scaling = None;
                        let quantized_dense= d.quantize(&model_scaling, bias_scaling);
                        let (quant_min_output,_quant_max_output) = quantized_dense.output_range(*quantization::MIN,*quantization::MAX);
                        //let abs_max = min_output.abs().max(max_output.abs());
                        //let output_scaling = ScalingFactor::new(abs_max);
                        // TODO: remove this is broken
                        let output_scaling = ScalingFactor::default();
                        last_input_scaling_factor = output_scaling;
                        md.set_layers_scaling(id, output_scaling);
                        
                        println!("Scaling: AbsoluteMax: DENSE max_weight {:?}, max_output: {:?} - adding requant", max_weight, 1.0);
                        
                        let intermediate_bit_size = ceil_log2(quant_min_output.abs() as usize);
                        let full_requant = FullRequant::from_scaling_factors(last_input_scaling_factor, model_scaling, output_scaling, intermediate_bit_size);
                        // vec![Layer::Dense(quantized_dense), Layer::Requant(requant)]
                        vec![Layer::Dense(quantized_dense), Layer::FullRequant(full_requant)]

                    }
                    Layer::Convolution(d) => {
                        let max_weight = d.max_abs_weight();
                        let model_scaling = ScalingFactor::from_absolute_max(max_weight, None);
                        //let (min_output,max_output) = d.output_range(-last_input_scaling_factor.abs_max,last_input_scaling_factor.abs_max);
                        let _bias_scaling = {
                            // bias has to be quantized over integers with double bit length
                            let min_quantized = -(1 << (2*(*BIT_LEN) - 1)) + 1;
                            let max_quantized = (1 << (2*(*BIT_LEN) - 1)) - 1;
                            ScalingFactor::from_scale(
                                last_input_scaling_factor.scale() * model_scaling.scale(),
                                Some((min_quantized, max_quantized))
                            )
                        };
                        let bias_scaling = None;
                        let quantized_conv= d.quantize(&model_scaling, bias_scaling);
                        let (quant_min_output,_quant_max_output) = quantized_conv.output_range(*quantization::MIN,*quantization::MAX);
                        //let abs_max = min_output.abs().max(max_output.abs());
                        //let output_scaling = ScalingFactor::new(abs_max);
                        // TODO: remove this is broken
                        let output_scaling = ScalingFactor::default();
                        md.set_layers_scaling(id, output_scaling);
                        
                        last_input_scaling_factor = output_scaling;
                        println!("Scaling: AbsoluteMax: CONV max_weight {:?}, max_output: {:?} - adding requant", max_weight, 1.0);
                        
                        let intermediate_bit_size = ceil_log2(quant_min_output.abs() as usize);
                        let full_requant = FullRequant::from_scaling_factors(last_input_scaling_factor, model_scaling, output_scaling, intermediate_bit_size);
                        // vec![Layer::Convolution(quantized_conv), Layer::Requant(requant)]
                        vec![Layer::Convolution(quantized_conv), Layer::FullRequant(full_requant)]
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
