use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow, ensure};
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;

use crate::{
    IO, Tensor,
    layers::{
        Layer, LayerCtx,
        provable::{Edge, OpInfo},
    },
    padding::PaddingMode,
    quantization::{Fieldizer, TensorFielder},
    tensor::Number,
    try_unzip,
};

use super::{Evaluate, InferenceStep, NodeCtx, NodeEgdes, NodeId, Op, ProvableNode, StepData};

#[derive(Debug)]
pub struct ProvableModel<N> {
    pub(crate) nodes: HashMap<NodeId, ProvableNode<N>>,
    pub(crate) input_shapes: Vec<Vec<usize>>,
    pub(crate) unpadded_input_shapes: Vec<Vec<usize>>,
}

pub struct Trace<'a, E: ExtensionField, N, D> {
    pub(crate) steps: HashMap<NodeId, InferenceStep<'a, E, N, D>>,
    pub(crate) input: Vec<Tensor<D>>,
    pub(crate) output: Vec<Tensor<D>>,
}
// The trace produce by running the model during inference
pub type InferenceTrace<'a, E: ExtensionField, N> = Trace<'a, E, N, N>;
// The trace used to prove the model
pub type ProvingTrace<'a, E: ExtensionField, N> = Trace<'a, E, N, E>;

impl<N> ProvableModel<N> {
    pub fn to_forward_iterator(&self) -> ModelForwardIterator<N> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().cloned().collect(),
            nodes: &self.nodes,
        }
    }

    pub fn to_backward_iterator(&self) -> ModelBackwardIterator<N> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().cloned().collect(),
            nodes: &self.nodes,
        }
    }

    /// Returns an iterator over the nodes in the model, in arbitrary order.
    /// It is more efficient then `ForwardIterator` and `BackwardIterator`, so it
    /// can be used to iterate over the nodes when the order does not matter
    pub fn to_unstable_iterator(&self) -> impl Iterator<Item = (&NodeId, &ProvableNode<N>)> {
        self.nodes.iter()
    }

    pub(crate) fn new_from_input_shapes(unpadded_input_shapes: Vec<Vec<usize>>) -> Self {
        let input_shapes = unpadded_input_shapes
            .iter()
            .map(|shape| {
                shape
                    .into_iter()
                    .map(|dim| dim.next_power_of_two())
                    .collect()
            })
            .collect();
        Self {
            nodes: HashMap::new(),
            input_shapes,
            unpadded_input_shapes,
        }
    }

    pub(crate) fn unpadded_input_shapes(&self) -> Vec<Vec<usize>> {
        self.unpadded_input_shapes.clone()
    }

    pub(crate) fn input_shapes(&self, padding: PaddingMode) -> Vec<Vec<usize>> {
        match padding {
            PaddingMode::NoPadding => self.unpadded_input_shapes.clone(),
            PaddingMode::Padding => self.input_shapes.clone(),
        }
    }

    pub(crate) fn add_node(&mut self, node: ProvableNode<N>) -> anyhow::Result<NodeId> {
        let node_id = self.nodes.len() as NodeId;
        self.add_node_with_id(node_id, node)?;
        Ok(node_id)
    }

    pub(crate) fn add_node_with_id(
        &mut self,
        node_id: NodeId,
        node: ProvableNode<N>,
    ) -> anyhow::Result<()> {
        // iterate over the inputs of the node and add the edges to the outputs of
        // corresponding nodes already in the model
        for (i, input) in node.inputs.iter().enumerate() {
            if let Some(input_node_id) = &input.node {
                let input_node = self.nodes.get_mut(input_node_id).ok_or(anyhow!(
                    "Node {} for input {} of new node not found in model",
                    input_node_id,
                    i,
                ))?;
                ensure!(
                    input.index < input_node.outputs.len(),
                    "Specified output number {} for node {}, which has only {} outputs",
                    input.index,
                    input_node_id,
                    input_node.outputs.len(),
                );
                input_node.outputs[input.index].edges.push(Edge {
                    node: Some(node_id),
                    index: i,
                });
            }
        }

        self.nodes.insert(node_id, node);

        Ok(())
    }

    pub(crate) fn route_output(&mut self, output_edges: Vec<Edge>) -> Result<()> {
        for (out_index, edge) in output_edges.iter().enumerate() {
            let out_node_id = edge
                .node
                .ok_or(anyhow!("Provided output edge with no input node"))?;
            let out_node = self
                .nodes
                .get_mut(&out_node_id)
                .ok_or(anyhow!("Node {out_node_id} not found"))?;
            ensure!(
                edge.index < out_node.outputs.len(),
                "Specified output {} for node {out_node_id}, but only {} outputs found",
                edge.index,
                out_node.outputs.len()
            );
            out_node.outputs[edge.index].edges.push(Edge {
                node: None,
                index: out_index,
            })
        }

        Ok(())
    }

    pub(crate) fn output_node(&self) -> NodeId {
        self.nodes
            .iter()
            .find_map(|(id, node)| {
                if node
                    .outputs
                    .iter()
                    .all(|wire| wire.edges.iter().all(|edge| edge.node.is_none()))
                {
                    Some(*id)
                } else {
                    None
                }
            })
            .expect("No output node found")
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.nodes.len()
    }
}
// Creates an iterator over the nodes in the model, starting from the inputs and
// yielding nodes in order according to whether their inputs all come from nodes
// already visited by the iterator. This is useful for traversing the model when
// evaluating it at interence time
pub type ModelForwardIterator<'a, N> = NodeIterator<'a, ProvableNode<N>, true>;
// Creates an iterator over the nodes in the model, starting from the outputs and
// yielding nodes in order according to whether their outputs all come from nodes
// already visited by the iterator. This is useful for traversing the model when
// proving
pub type ModelBackwardIterator<'a, N> = NodeIterator<'a, ProvableNode<N>, false>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct ModelCtx<E: ExtensionField + DeserializeOwned>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    pub(crate) nodes: HashMap<NodeId, NodeCtx<E>>,
}

impl<E: ExtensionField + DeserializeOwned> ModelCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    pub(crate) fn to_forward_iterator(&self) -> ModelCtxForwardIterator<E> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().cloned().collect(),
            nodes: &self.nodes,
        }
    }

    pub(crate) fn to_backward_iterator(&self) -> ModelCtxBackwardIterator<E> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().cloned().collect(),
            nodes: &self.nodes,
        }
    }

    pub(crate) fn contexts(&self) -> impl Iterator<Item = &LayerCtx<E>> {
        self.nodes.iter().map(|(_, nodes)| &nodes.ctx)
    }
}

pub type ModelCtxForwardIterator<'a, E> = NodeIterator<'a, NodeCtx<E>, true>;
pub type ModelCtxBackwardIterator<'a, E> = NodeIterator<'a, NodeCtx<E>, false>;

pub struct NodeIterator<'a, E: NodeEgdes, const FORWARD: bool> {
    pub(crate) unvisited_nodes: HashSet<NodeId>,
    pub(crate) nodes: &'a HashMap<NodeId, E>,
}

impl<'a, E: NodeEgdes, const FORWARD: bool> Iterator for NodeIterator<'a, E, FORWARD> {
    type Item = (NodeId, &'a E);

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.unvisited_nodes.iter().find_map(|node_id| {
            let node = self.nodes.get(node_id).unwrap(); // safe to unwrap since this should contain only nodes in the model
            let is_node_next = if FORWARD {
                node.inputs().iter().all(|edge| {
                    edge.node.is_none()
                        || !self.unvisited_nodes.contains(edge.node.as_ref().unwrap())
                })
            } else {
                node.outputs()
                    .iter()
                    .flat_map(|output| &output.edges)
                    .all(|edge| {
                        edge.node.is_none()
                            || !self.unvisited_nodes.contains(edge.node.as_ref().unwrap())
                    })
            };
            if is_node_next {
                Some((*node_id, node))
            } else {
                None
            }
        });
        if let Some((node_id, _)) = &node {
            // remove the node from the unvisited nodes set
            self.unvisited_nodes.remove(node_id);
        }
        node
    }
}

impl<'a, E: ExtensionField, N, D> Trace<'a, E, N, D> {
    pub(crate) fn get_step(&self, node_id: &NodeId) -> Option<&InferenceStep<'a, E, N, D>> {
        self.steps.get(node_id)
    }

    pub(crate) fn new_step(&mut self, node_id: NodeId, step: InferenceStep<'a, E, N, D>) {
        self.steps.insert(node_id, step);
    }

    pub(crate) fn steps(&self) -> impl Iterator<Item = (&NodeId, &InferenceStep<'a, E, N, D>)> {
        self.steps.iter()
    }

    pub(crate) fn to_verifier_io(&self) -> IO<E>
    where
        D: Fieldizer<E>,
    {
        let input = self.input.iter().map(|inp| inp.to_fields()).collect();
        let output = self.output.iter().map(|out| out.to_fields()).collect();
        IO::new(input, output)
    }

    pub(crate) fn to_field(self) -> ProvingTrace<'a, E, N>
    where
        D: Fieldizer<E>,
    {
        let input = self.input.into_iter().map(|inp| inp.to_fields()).collect();
        let output = self.output.into_iter().map(|out| out.to_fields()).collect();
        let field_steps = self
            .steps
            .into_iter()
            .map(|(id, step)| {
                (id, InferenceStep {
                    op: step.op,
                    step_data: StepData {
                        inputs: step
                            .step_data
                            .inputs
                            .into_iter()
                            .map(|inp| inp.to_fields())
                            .collect(),
                        outputs: super::LayerOut {
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
                })
            })
            .collect();
        Trace {
            steps: field_steps,
            input,
            output,
        }
    }

    pub(crate) fn output(&self) -> Result<&Tensor<D>> {
        // we assume for now there is only one output tensor
        ensure!(
            self.output.len() == 1,
            "Found more than 1 output tensor for the model"
        );
        Ok(&self.output[0])
    }
}

impl<N: Number> ProvableModel<N> {
    pub(crate) fn run<E: ExtensionField>(
        &self,
        input: &[Tensor<N>],
    ) -> anyhow::Result<InferenceTrace<'_, E, N>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: Serialize + DeserializeOwned,
        Layer<N>: Evaluate<N>,
    {
        let mut trace = Trace {
            steps: HashMap::new(),
            input: input.to_vec(),
            output: vec![],
        };
        let iter = self.to_forward_iterator();
        for (node_id, node) in iter {
            let (inputs, shapes): (Vec<_>, Vec<_>) = try_unzip(node.inputs.iter().map(|edge| {
                Ok(if let Some(n) = &edge.node {
                    let step = trace.get_step(n);
                    ensure!(step.is_some(), "Node {} not found in trace", n);
                    let step = step.unwrap();
                    let outputs = step.step_data.outputs.outputs();
                    ensure!(
                        edge.index < outputs.len(),
                        "Requested output {} for node {}, which has only {} outputs",
                        edge.index,
                        n,
                        outputs.len()
                    );
                    // get shape of this output
                    let out_shapes = &step.step_data.unpadded_output_shapes;
                    (outputs[edge.index], out_shapes[edge.index].clone())
                } else {
                    (
                        &input[edge.index],
                        self.unpadded_input_shapes()[edge.index].clone(),
                    )
                })
            }))?;
            let output_shapes = node
                .operation
                .output_shapes(&shapes, PaddingMode::NoPadding);
            let output = node.run(inputs.as_slice(), shapes)?;
            let new_step = StepData {
                inputs: inputs.into_iter().cloned().collect(),
                outputs: output,
                unpadded_output_shapes: output_shapes,
            };
            trace.new_step(node_id, InferenceStep {
                op: &node.operation,
                step_data: new_step,
            });
        }

        // compute the output tensor from the outputs of the output nodes
        let output_node = self.output_node();
        let output: Vec<Tensor<N>> = trace
            .get_step(&output_node)
            .ok_or(anyhow!("Output node {} not found in trace", output_node))?
            .outputs()
            .into_iter()
            .cloned()
            .collect();
        ensure!(output.len() == 1, "Unexpected number of output tensors");

        trace.output = output;

        Ok(trace)
    }

    pub(crate) fn provable_nodes(&self) -> impl Iterator<Item = (&NodeId, &ProvableNode<N>)> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.operation.is_provable())
    }
}

#[cfg(test)]
mod tests {
    use goldilocks::GoldilocksExt2;
    use transcript::BasicTranscript;

    use crate::{
        Context, Element, IO, Prover, Tensor,
        layers::{
            Layer,
            activation::{Activation, Relu},
            dense::Dense,
            provable::{Edge, OpInfo, ProvableNode},
        },
        padding::PaddingMode,
        quantization::TensorFielder,
        tensor::Number,
        testing::random_vector,
        verify,
    };

    use super::ProvableModel;

    type E = GoldilocksExt2;
    type T = BasicTranscript<GoldilocksExt2>;
    type N = Element;

    fn build_test_model<N: Number, const INPUT_SIZE: usize, const PADDED: bool>() -> ProvableModel<N>
    {
        let input_shape = vec![INPUT_SIZE];
        let mut model = ProvableModel::<N>::new_from_input_shapes(vec![input_shape.clone()]);
        // add input dense layer
        // generate random dense matrix
        let ncols = input_shape[0];
        let nrows = 42;
        let dense = Dense::random(vec![nrows, ncols]);
        let dense_out_shape = &dense.output_shapes(
            &model.input_shapes(PaddingMode::NoPadding),
            PaddingMode::NoPadding,
        )[0];
        let dense = if PADDED {
            dense.pad_next_power_of_two()
        } else {
            dense
        };
        let input_node = model
            .add_node(ProvableNode::new(
                vec![Edge {
                    node: None, // it's the input tensor for the model
                    index: 0,   // it's the first (and only) input tensor
                }],
                Layer::Dense(dense),
            ))
            .unwrap();
        // add activation layer
        let relu = Activation::Relu(Relu::new());
        let relu_node = model
            .add_node(ProvableNode::new(
                vec![Edge {
                    node: Some(input_node),
                    index: 0, // it's the first (and only) output tensor of the previous layer
                }],
                Layer::Activation(relu),
            ))
            .unwrap();
        // add another dense layer as output
        let nrows = 37;
        let ncols = dense_out_shape[0]; // it's a vector, so it has only one dimension
        let dense = Dense::random(vec![nrows, ncols]);
        let dense = if PADDED {
            dense.pad_next_power_of_two()
        } else {
            dense
        };
        let output_node = model
            .add_node(ProvableNode::new(
                vec![Edge {
                    node: Some(relu_node),
                    index: 0, // it's the first (and only) output tensor of the previous layer
                }],
                Layer::Dense(dense),
            ))
            .unwrap();
        model
            .route_output(vec![Edge {
                node: Some(output_node),
                index: 0,
            }])
            .unwrap();

        assert_eq!(model.output_node(), output_node);

        model
    }

    #[test]
    fn test_model_inference() {
        const INPUT_SIZE: usize = 45;
        let model = build_test_model::<N, INPUT_SIZE, false>();
        let input_shape = model.input_shapes(PaddingMode::NoPadding)[0].clone();

        let input = random_vector(input_shape.iter().product());
        let input_tensor = Tensor::new(input_shape, input);
        let trace = model.run::<E>(&[input_tensor]).unwrap();
        assert_eq!(trace.steps.len(), 3);
    }

    #[test]
    fn test_model_float_inference() {
        const INPUT_SIZE: usize = 45;
        let model = build_test_model::<f32, INPUT_SIZE, false>();
        let input_shape = model.input_shapes(PaddingMode::NoPadding)[0].clone();

        let input_tensor = Tensor::random(&input_shape);
        let trace = model.run::<E>(&[input_tensor]).unwrap();
        assert_eq!(trace.steps.len(), 3);
    }

    #[test]
    fn test_model_proving() {
        const INPUT_SIZE: usize = 57;
        let model = build_test_model::<N, INPUT_SIZE, true>();
        let input_shape = model.input_shapes(PaddingMode::Padding)[0].clone();

        let input_tensor = Tensor::random(&input_shape);
        let trace = model.run(&[input_tensor]).unwrap();
        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"model");
        let ctx =
            Context::<GoldilocksExt2>::generate(&model, None).expect("Unable to generate context");
        let prover: Prover<'_, E, T> = Prover::new(&ctx, &mut tr);
        let io = trace.to_verifier_io();
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"model");
        verify::<_, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
    }
}
