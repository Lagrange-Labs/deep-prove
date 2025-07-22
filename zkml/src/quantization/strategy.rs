use crate::{
    layers::provable::{Node, NodeId, QuantizeOp},
    model::{Model, ToIterator},
    quantization::metadata::{MetadataBuilder, ModelMetadata},
    rng_from_env_or_random,
    tensor::Number,
};
use std::collections::HashMap;

use crate::{Element, Tensor, quantization};
use anyhow::{Result, anyhow, ensure};
use ff_ext::GoldilocksExt2;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use statrs::statistics::{Data, Max, Min};
use tracing::{debug, info, warn};

use super::ScalingFactor;

/// Trait for quantizing a float-based model into a quantized model. The current implementation
/// simply looks at the absolute maximum value of the model and uses that as the scaling factor
/// to quantize the model, one scaling factor per layer.
pub trait ScalingStrategy: std::fmt::Debug {
    type AuxData: Sized;

    fn quantize(&self, model: Model<f32>) -> Result<(Model<Element>, ModelMetadata)>;

    /// Returns the scaling factors for the outputs of the node with the given ID. The number of
    /// outputs is given by the `num_outputs` parameter. The scaling factors are computed based on
    /// the auxiliary data provided.
    fn scaling_factors_for_node(
        data: &Self::AuxData,
        node_id: NodeId,
        num_outputs: usize,
    ) -> Vec<ScalingFactor>;

    fn name(&self) -> String;
}

/// Implementors of [`ScalingStrategy`]
#[derive(Debug, Clone, Copy, derive_more::Display, Serialize, Deserialize)]
pub enum ScalingStrategyKind {
    InferenceObserver,
    AbsoluteMax,
}

/// Quantization strategy that observes the inference of the model with different inputs and uses the
/// min/max values of the output to determine the output scaling factor of each layer that needs
/// requantization afterwards.
#[derive(Debug)]
pub struct InferenceObserver {
    inputs: Vec<Vec<Vec<f32>>>,
}

impl Default for InferenceObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceObserver {
    pub fn new_with_representative_input(inputs: Vec<Vec<Vec<f32>>>) -> Self {
        Self { inputs }
    }
    pub fn new() -> Self {
        Self { inputs: vec![] }
    }
}

const INPUT_TRACKING_ID: usize = 10_000;

impl ScalingStrategy for InferenceObserver {
    type AuxData = InferenceTracker;

    fn name(&self) -> String {
        format!("inference [{},{}]", *quantization::MIN, *quantization::MAX)
    }

    fn quantize(&self, model: Model<f32>) -> Result<(Model<Element>, ModelMetadata)> {
        let mut tracker = InferenceTracker::new();
        let input_shapes = model.input_shapes();
        let input_not_padded_shapes = model.unpadded_input_shapes();
        let inputs = if self.inputs.is_empty() {
            let mut rng = rng_from_env_or_random();
            warn!("No representative inputs provided, generating random ones");
            (0..10)
                .map(|_| {
                    input_shapes
                        .iter()
                        .map(|shape| {
                            let size = shape.product();
                            (0..size)
                                .map(|_| <f32 as Number>::random(&mut rng))
                                .collect_vec()
                        })
                        .collect_vec()
                })
                .collect()
        } else {
            debug!("Using provided representative inputs");
            self.inputs.clone()
        };
        // 1. Run the inference multiple times with different inputs
        // TODO: integrate that within model.rs in a more elegant way with inference step - currently problematic
        // because of the generics and FFT requirement to take a field
        let mut nsamples = 0;
        for input in inputs.into_iter() {
            let input_tensors = input
                .into_iter()
                .zip(model.unpadded_input_shapes())
                .enumerate()
                .map(|(i, (inp, shape))| {
                    let input_tensor = Tensor::new(shape, inp);
                    tracker.track(INPUT_TRACKING_ID as NodeId, i, input_tensor.clone());
                    input_tensor
                })
                .collect_vec();
            model.run_with_tracker::<GoldilocksExt2>(&input_tensors, Some(&mut tracker))?;
            nsamples += 1;
        }
        info!("InferenceObserver: {} samples observed", nsamples);
        // 2. get the scaling factor of the input
        let num_model_inputs = input_not_padded_shapes.len();
        let input_scaling = (0..num_model_inputs)
            .map(|i| {
                let (input_min, input_max) =
                    tracker.distribution_info(INPUT_TRACKING_ID as NodeId, i);
                ScalingFactor::from_absolute_max(input_min.abs().max(input_max.abs()), None)
            })
            .collect_vec();
        quantize_model::<InferenceObserver>(model, tracker, input_scaling)
    }

    fn scaling_factors_for_node(
        tracker: &InferenceTracker,
        node_id: NodeId,
        num_outputs: usize,
    ) -> Vec<ScalingFactor> {
        (0..num_outputs)
            .map(|i| {
                let (min, max) = tracker.distribution_info(node_id, i);
                ScalingFactor::from_absolute_max(min.abs().max(max.abs()), None)
            })
            .collect()
    }
}

pub struct InferenceTracker {
    /// For each output of each node in the model of interest, we track all the values of the tensor
    data: HashMap<(NodeId, usize), Vec<f64>>,
}

impl InferenceTracker {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
    pub(crate) fn track(&mut self, node_id: NodeId, output_index: usize, output: Tensor<f32>) {
        self.data
            .entry((node_id, output_index))
            .or_default()
            .extend(output.get_data().iter().map(|x| *x as f64));
    }

    /// Returns the 0.05 and 0.95 quantiles of the distribution of the output values of the layer.
    pub(crate) fn distribution_info(&self, node_id: NodeId, output_index: usize) -> (f32, f32) {
        let d: Data<Vec<f64>> = Data::new(
            self.data
                .get(&(node_id, output_index))
                .unwrap_or_else(|| {
                    panic!("No data for output tensor {output_index} of node {node_id}")
                })
                .clone(),
        );
        // let min = d.percentile(5) as f32;
        // let max = d.percentile(95) as f32;
        // assert!(min <= max);
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
pub struct AbsoluteMax(Option<Vec<Vec<f32>>>);

impl Default for AbsoluteMax {
    fn default() -> Self {
        Self::new()
    }
}

impl AbsoluteMax {
    pub fn new_with_representative_input(input: Vec<Vec<f32>>) -> Self {
        Self(Some(input))
    }
    pub fn new() -> Self {
        Self(None)
    }
}

impl ScalingStrategy for AbsoluteMax {
    type AuxData = ();

    fn name(&self) -> String {
        "absolute_max".to_string()
    }

    fn quantize(&self, model: Model<f32>) -> Result<(Model<Element>, ModelMetadata)> {
        let input_scaling_factor = if let Some(ref input) = self.0 {
            let input_tensor = model.load_input_flat(input.clone())?;
            model
                .input_shapes()
                .into_iter()
                .zip(&input_tensor)
                .try_for_each(|(shape, input)| {
                    ensure!(
                        shape == input.get_shape(),
                        "input shape mismatch: expected {:?}, got {:?}",
                        shape,
                        input.get_shape()
                    );
                    Ok(())
                })?;
            input_tensor
                .into_iter()
                .map(|input| ScalingFactor::from_absolute_max(input.max_abs_output(), None))
                .collect_vec()
        } else {
            (0..model.num_inputs())
                .map(|_| ScalingFactor::default())
                .collect_vec()
        };
        quantize_model::<AbsoluteMax>(model, (), input_scaling_factor)
    }

    fn scaling_factors_for_node(
        _data: &Self::AuxData,
        _node_id: NodeId,
        num_outputs: usize,
    ) -> Vec<ScalingFactor> {
        vec![ScalingFactor::default(); num_outputs]
    }
}

fn quantize_model<S: ScalingStrategy>(
    model: Model<f32>,
    data: S::AuxData,
    input_scaling: Vec<ScalingFactor>,
) -> anyhow::Result<(Model<Element>, ModelMetadata)> {
    let input_shapes = model.input_shapes();
    let input_not_padded_shapes = model.unpadded_input_shapes();
    let mut md = MetadataBuilder::new(input_scaling);
    // 2. Create the requant layers from the inferred data
    let mut requant_layers = vec![];
    let nodes = model
        .into_forward_iterator()
        .map(|(node_id, node)| {
            let input_scaling = md.compute_input_scaling(&node.inputs)?;
            let quantized_out = node
                .operation
                .quantize_op::<S>(&data, node_id, &input_scaling)?;
            md.set_layers_scaling(node_id, quantized_out.output_scalings, input_scaling);
            if let Some(requant) = quantized_out.requant_layer {
                requant_layers.push((node_id, requant));
            }
            let quantized_node =
                Node::new_with_outputs(node.inputs, quantized_out.quantized_op, node.outputs);
            Ok((node_id, quantized_node))
        })
        .collect::<Result<_>>()?;
    let mut model = Model::new_from_shapes(input_not_padded_shapes, input_shapes, nodes);
    for (input_node_id, requant) in requant_layers {
        let requant_ids = model.add_requant_nodes(requant, input_node_id)?;
        // add scaling factor to `md` for requant layers: the scaling factors of the inputs correspond to
        // the scaling factors of the outputs of the previous node
        let input_scaling = md.get_output_layer_scaling(&input_node_id).ok_or(anyhow!(
            "Scaling factors not found for node {input_node_id}"
        ))?;
        ensure!(
            requant_ids.len() == input_scaling.len(),
            "Number of requant layers must match number of output scalings"
        );
        for (node_id, scaling_factor) in requant_ids.into_iter().zip(input_scaling.to_vec()) {
            md.set_layers_scaling(node_id, vec![scaling_factor], vec![scaling_factor]);
        }
    }
    let out_nodes = model.output_nodes();
    let md = md.build(out_nodes)?;
    Ok((model, md))
}
