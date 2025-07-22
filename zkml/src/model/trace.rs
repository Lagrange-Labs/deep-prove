use std::collections::HashMap;

use ff_ext::ExtensionField;

use crate::{
    Element, IO, Tensor,
    layers::{
        Layer,
        provable::{LayerOut, NodeId},
    },
    quantization::{Fieldizer, ModelMetadata, TensorFielder},
    tensor::Shape,
};

#[derive(Debug, Clone, Default)]
pub struct Trace<'a, E: ExtensionField, N, D> {
    pub(crate) steps: HashMap<NodeId, InferenceStep<'a, E, N, D>>,
    pub(crate) input: Vec<Tensor<D>>,
    pub(crate) output: Vec<Tensor<D>>,
}
// The trace produce by running the model during inference
pub type InferenceTrace<'a, E, N> = Trace<'a, E, N, N>;
// The trace used to prove the model
pub type ProvingTrace<'a, E, N> = Trace<'a, E, N, E>;

impl<'a, E: ExtensionField, N, D> Trace<'a, E, N, D> {
    /// Get the trace data for node `node_id`, if any
    pub(crate) fn get_step(&self, node_id: &NodeId) -> Option<&InferenceStep<'a, E, N, D>> {
        self.steps.get(node_id)
    }

    /// Insert the trace data `step` about node `node_id` in the trace
    pub(crate) fn new_step(&mut self, node_id: NodeId, step: InferenceStep<'a, E, N, D>) {
        self.steps.insert(node_id, step);
    }

    /// Compute the inputs and outputs tensors from the trace, which are necessary
    /// for the verifier to verify the proof of the model inference
    pub fn to_verifier_io(&self) -> IO<E>
    where
        D: Fieldizer<E>,
    {
        let input = self.input.iter().map(|inp| inp.to_fields()).collect();
        let output = self.output.iter().map(|out| out.to_fields()).collect();
        IO::new(input, output)
    }

    /// Convert an inference trace computed over integers to a trace over field elements, which is
    /// needed to prove the inference
    pub(crate) fn into_fields(self) -> ProvingTrace<'a, E, N>
    where
        D: Fieldizer<E>,
    {
        let input = self.input.into_iter().map(|inp| inp.to_fields()).collect();
        let output = self.output.into_iter().map(|out| out.to_fields()).collect();
        let field_steps = self
            .steps
            .into_iter()
            .map(|(id, step)| {
                (
                    id,
                    InferenceStep {
                        op: step.op,
                        step_data: StepData {
                            inputs: step
                                .step_data
                                .inputs
                                .into_iter()
                                .map(|inp| inp.to_fields())
                                .collect(),
                            outputs: LayerOut {
                                outputs: step
                                    .step_data
                                    .outputs
                                    .outputs
                                    .into_iter()
                                    .map(|out| out.to_fields())
                                    .collect(),
                                proving_data: step.step_data.outputs.proving_data,
                            },
                            unpadded_output_shapes: step.step_data.unpadded_output_shapes,
                        },
                    },
                )
            })
            .collect();
        Trace {
            steps: field_steps,
            input,
            output,
        }
    }

    /// Get the output tensors of the inference represented by this trace
    pub fn outputs(&self) -> anyhow::Result<Vec<&Tensor<D>>> {
        Ok(self.output.iter().collect())
    }
}

impl<'a, E: ExtensionField> InferenceTrace<'a, E, Element> {
    /// Given as input a trace over quantized values, compute the equivalent
    /// trace with dequantized values
    pub fn dequantized(&self, md: &ModelMetadata) -> Trace<'a, E, Element, f32> {
        let inputs = self
            .input
            .iter()
            .zip(&md.input)
            .map(|(input, s)| input.dequantize(s))
            .collect();
        let outputs = self
            .output
            .iter()
            .zip(&md.output)
            .map(|(out, s)| out.dequantize(s))
            .collect();
        let steps = self
            .steps
            .iter()
            .map(|(node_id, step)| {
                let input_scaling = md.layer_input_scaling_factor(*node_id);
                let inputs = step
                    .step_data
                    .inputs
                    .iter()
                    .zip(input_scaling)
                    .map(|(inp, s)| inp.dequantize(s))
                    .collect();
                let output_scaling = md.layer_output_scaling_factor(*node_id);
                let outputs = step
                    .step_data
                    .outputs
                    .outputs()
                    .into_iter()
                    .zip(output_scaling)
                    .map(|(out, s)| out.dequantize(s))
                    .collect();
                (
                    *node_id,
                    InferenceStep {
                        op: step.op,
                        step_data: StepData {
                            inputs,
                            outputs: LayerOut {
                                outputs,
                                proving_data: step.step_data.outputs.proving_data.clone(),
                            },
                            unpadded_output_shapes: step.step_data.unpadded_output_shapes.clone(),
                        },
                    },
                )
            })
            .collect();
        Trace {
            steps,
            input: inputs,
            output: outputs,
        }
    }
}

/// Data found in the trace for each node of the model
#[derive(Debug, Clone)]
pub struct InferenceStep<'a, E: ExtensionField, N, D> {
    pub(crate) op: &'a Layer<N>,
    pub(crate) step_data: StepData<D, E>,
}

impl<'a, E: ExtensionField, N, D> InferenceStep<'a, E, N, D> {
    /// Returns the output tensors of the node
    pub fn outputs(&self) -> Vec<&Tensor<D>> {
        self.step_data.outputs.outputs()
    }
}

/// Data about the input and output tensors in a trace
/// for each node in the model
#[derive(Debug, Clone)]
pub struct StepData<D, E: ExtensionField> {
    pub(crate) inputs: Vec<Tensor<D>>,
    pub(crate) outputs: LayerOut<D, E>,
    pub(crate) unpadded_output_shapes: Vec<Shape>,
}
