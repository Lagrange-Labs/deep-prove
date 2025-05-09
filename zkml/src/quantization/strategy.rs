use crate::{
    quantization::{
        BIT_LEN,
        metadata::{MetadataBuilder, ModelMetadata},
    },
    tensor::Number,
};
use std::collections::HashMap;

use crate::{
    Element, Tensor,
    layers::{Layer, requant::Requant},
    model::Model,
    quantization,
};
use anyhow::{Result, ensure};
use ark_std::rand;
use itertools::Itertools;
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
        let input_not_padded_shape = model.unpadded_input_shape();
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
            let input_tensor = Tensor::new(model.unpadded_input.clone(), input.clone());
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
        // 2. get the scaling factor of the input
        let (input_min, input_max) = tracker.distribution_info(INPUT_TRACKING_ID);
        let input_scaling =
            ScalingFactor::from_absolute_max(input_min.abs().max(input_max.abs()), None);
        let mut md = MetadataBuilder::new(input_scaling.clone());
        let mut last_input_scaling = input_scaling.clone();
        // 2. Create the requant layers from the infered data
        // manually take care of updating the step_idx since we are adding layers here
        let mut step_idx = 0;
        let quantized_layers = model
            .layers
            .into_iter()
            .enumerate()
            .map(|(id, layer)| {
                match layer {
                    Layer::Dense(dense) => {
                        let model_scaling =
                            ScalingFactor::from_absolute_max(dense.max_abs_weight(), None);
                        let (min, max) = tracker.distribution_info(id);
                        let output_scaling =
                            ScalingFactor::from_absolute_max(min.abs().max(max.abs()), None);
                        let bias_scaling = {
                            // bias has to be quantized over integers with double bit length
                            let min_quantized = -(1 << (2 * (*BIT_LEN) - 1)) + 1;
                            let max_quantized = (1 << (2 * (*BIT_LEN) - 1)) - 1;
                            ScalingFactor::from_scale(
                                last_input_scaling.scale() * model_scaling.scale(),
                                Some((min_quantized, max_quantized)),
                            )
                        };

                        // let quantized_dense = dense.quantize(&model_scaling, Some(&_bias_scaling));
                        let quantized_dense = dense.quantize(&model_scaling, &bias_scaling);
                        let intermediate_bit_size = quantized_dense.output_bitsize();
                        let requant = Requant::from_scaling_factors(
                            last_input_scaling,
                            model_scaling,
                            output_scaling,
                            intermediate_bit_size,
                        );
                        // requant.set_test_multiplier(scale);
                        md.set_layers_scaling(step_idx, output_scaling);
                        last_input_scaling = output_scaling;
                        // because we are adding a new layer
                        step_idx += 2;
                        vec![Layer::Dense(quantized_dense), Layer::Requant(requant)]
                    }
                    Layer::Convolution(conv) => {
                        let model_scaling =
                            ScalingFactor::from_absolute_max(conv.max_abs_weight(), None);
                        let (min, max) = tracker.distribution_info(id);
                        let output_scaling =
                            ScalingFactor::from_absolute_max(min.abs().max(max.abs()), None);
                        let bias_scaling = {
                            // bias has to be quantized over integers with double bit length
                            let min_quantized = -(1 << (2 * (*BIT_LEN) - 1)) + 1;
                            let max_quantized = (1 << (2 * (*BIT_LEN) - 1)) - 1;
                            ScalingFactor::from_scale(
                                last_input_scaling.scale() * model_scaling.scale(),
                                Some((min_quantized, max_quantized)),
                            )
                        };
                        let quantized_conv = conv.quantize(&model_scaling, &bias_scaling);

                        let intermediate_bit_size = quantized_conv.output_bitsize();
                        md.set_layers_scaling(step_idx, output_scaling);
                        let requant = Requant::from_scaling_factors(
                            last_input_scaling,
                            model_scaling,
                            output_scaling,
                            intermediate_bit_size,
                        );
                        // requant.set_test_multiplier(scale);
                        last_input_scaling = output_scaling;
                        // because we are adding a new layer
                        step_idx += 2;
                        vec![Layer::Convolution(quantized_conv), Layer::Requant(requant)]
                    }
                    a => {
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
        let mut md = MetadataBuilder::new(last_input_scaling_factor.clone());
        let input_shape = model.input_shape();
        let input_not_padded_shape = model.unpadded_input_shape();
        let quantized_layers = model
            .layers
            .into_iter()
            .enumerate()
            .flat_map(|(id, l)| {
                // If a layer requires a requantization step the current layer, this method returns the
                // next layer, e.g. requantization layer, as well as the scaling factor of the output. This is
                // given to the next layer as input scaling factor.
                match l {
                    Layer::Dense(d) => {
                        let max_weight = d.max_abs_weight();
                        let model_scaling = ScalingFactor::from_absolute_max(max_weight, None);
                        let bias_scaling = {
                            // bias has to be quantized over integers with double bit length
                            let min_quantized = -(1 << (2 * (*BIT_LEN) - 1)) + 1;
                            let max_quantized = (1 << (2 * (*BIT_LEN) - 1)) - 1;
                            ScalingFactor::from_scale(
                                last_input_scaling_factor.scale() * model_scaling.scale(),
                                Some((min_quantized, max_quantized)),
                            )
                        };
                        let quantized_dense = d.quantize(&model_scaling, &bias_scaling);

                        let output_scaling = ScalingFactor::default();
                        let intermediate_bit_size = quantized_dense.output_bitsize();
                        let requant = Requant::from_scaling_factors(
                            last_input_scaling_factor,
                            model_scaling,
                            output_scaling,
                            intermediate_bit_size,
                        );
                        last_input_scaling_factor = output_scaling;
                        md.set_layers_scaling(id, output_scaling);

                        vec![Layer::Dense(quantized_dense), Layer::Requant(requant)]
                    }
                    Layer::Convolution(d) => {
                        let max_weight = d.max_abs_weight();
                        let model_scaling = ScalingFactor::from_absolute_max(max_weight, None);
                        let bias_scaling = {
                            // bias has to be quantized over integers with double bit length
                            let min_quantized = -(1 << (2 * (*BIT_LEN) - 1)) + 1;
                            let max_quantized = (1 << (2 * (*BIT_LEN) - 1)) - 1;
                            ScalingFactor::from_scale(
                                last_input_scaling_factor.scale() * model_scaling.scale(),
                                Some((min_quantized, max_quantized)),
                            )
                        };
                        let quantized_conv = d.quantize(&model_scaling, &bias_scaling);

                        let output_scaling = ScalingFactor::default();
                        let intermediate_bit_size = quantized_conv.output_bitsize();
                        let requant = Requant::from_scaling_factors(
                            last_input_scaling_factor,
                            model_scaling,
                            output_scaling,
                            intermediate_bit_size,
                        );
                        md.set_layers_scaling(id, output_scaling);

                        last_input_scaling_factor = output_scaling;

                        vec![Layer::Convolution(quantized_conv), Layer::Requant(requant)]
                    }
                    a => {
                        return vec![a.quantize(
                            &last_input_scaling_factor,
                            None, // no scaling factor for bias needed for this layer
                        )];
                    }
                }
            })
            .collect::<Vec<Layer<Element>>>();
        Ok((
            Model::<Element>::new_from(quantized_layers, input_not_padded_shape, input_shape),
            md.build(),
        ))
    }
}
