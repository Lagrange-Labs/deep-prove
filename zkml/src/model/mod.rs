use std::collections::{BTreeMap, HashMap};

use anyhow::{Result, anyhow, ensure};
use ff_ext::ExtensionField;
use goldilocks::GoldilocksExt2;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use trace::Trace;
use tracing::{debug, info};

use crate::{
    Tensor,
    layers::{
        Layer,
        provable::{Edge, Evaluate, Node, NodeCtx, NodeId, OpInfo},
        requant::Requant,
    },
    padding::PaddingMode,
    quantization::InferenceTracker,
    tensor::Number,
    try_unzip,
};

pub(crate) mod iterator;
pub(crate) mod trace;

pub use iterator::ToIterator;
pub use trace::{InferenceStep, InferenceTrace, StepData};

/// Represents a model
#[derive(Debug, Clone)]
pub struct Model<N> {
    pub(crate) nodes: HashMap<NodeId, Node<N>>,
    pub(crate) input_shapes: Vec<Vec<usize>>,
    pub(crate) unpadded_input_shapes: Vec<Vec<usize>>,
}

impl<N> Model<N>
where
    N: Number,
{
    /// Returns an iterator over the nodes in the model, in arbitrary order.
    /// It is more efficient then `ForwardIterator` and `BackwardIterator`, so it
    /// can be used to iterate over the nodes when the order does not matter
    pub fn to_unstable_iterator(&self) -> impl Iterator<Item = (&NodeId, &Node<N>)> {
        self.nodes.iter()
    }

    /// Utility method to pad the inputs shapes to the next power of two
    fn compute_padded_input_shapes(unpadded_input_shapes: &[Vec<usize>]) -> Vec<Vec<usize>> {
        unpadded_input_shapes
            .iter()
            .map(|shape| shape.iter().map(|dim| dim.next_power_of_two()).collect())
            .collect()
    }

    /// Instantiate a model with the given input shape: the `padding` input specifies whether
    /// the provided inputs shapes should be padded or not
    pub fn new_from_input_shapes(
        unpadded_input_shapes: Vec<Vec<usize>>,
        padding: PaddingMode,
    ) -> Self {
        let input_shapes = match padding {
            PaddingMode::NoPadding => unpadded_input_shapes.clone(),
            PaddingMode::Padding => Self::compute_padded_input_shapes(&unpadded_input_shapes),
        };
        Self {
            nodes: HashMap::new(),
            input_shapes,
            unpadded_input_shapes,
        }
    }

    pub(crate) fn new(
        unpadded_input_shapes: Vec<Vec<usize>>,
        padding: PaddingMode,
        nodes: HashMap<NodeId, Node<N>>,
    ) -> Self {
        let mut model = Self::new_from_input_shapes(unpadded_input_shapes, padding);
        model.nodes = nodes;

        model
    }

    /// Instantiate a model from the set of nodes and the input shapes.
    /// `actual_input_shapes` correspond to the expected shape of the input
    /// tensors for the model; therefore, `actual_input_shapes` can be the same
    /// as `unpadded_input_shapes` if the input tensors of the model are
    /// not expected to be padded
    pub fn new_from_shapes(
        unpadded_input_shapes: Vec<Vec<usize>>,
        actual_input_shapes: Vec<Vec<usize>>,
        nodes: HashMap<NodeId, Node<N>>,
    ) -> Self {
        Self {
            unpadded_input_shapes,
            input_shapes: actual_input_shapes,
            nodes,
        }
    }

    /// Get the shapes of the input tensors, not padded
    pub(crate) fn unpadded_input_shapes(&self) -> Vec<Vec<usize>> {
        self.unpadded_input_shapes.clone()
    }

    /// Get the actual input shapes, which could be padded or unpadded
    /// depending on how the model was instantiated
    pub fn input_shapes(&self) -> Vec<Vec<usize>> {
        self.input_shapes.clone()
    }

    pub fn num_inputs(&self) -> usize {
        self.input_shapes.len()
    }

    /// Prepare the input tensors to be provided to the model according to the
    /// actual input shapes expected by the model
    pub fn prepare_inputs(&self, inputs: Vec<Tensor<N>>) -> Result<Vec<Tensor<N>>> {
        let input_shapes = self.input_shapes.clone();
        ensure!(
            input_shapes.len() == inputs.len(),
            "Unexpected number of inputs tensors: expected {}, found {}",
            input_shapes.len(),
            inputs.len()
        );
        Ok(inputs
            .into_iter()
            .zip(input_shapes)
            .map(|(mut input, shape)| {
                if input.get_shape().clone() == shape {
                    // no need to pad, simply return the input
                    input
                } else {
                    input.pad_to_shape(shape);
                    input
                }
            })
            .collect())
    }

    /// Build the inputs tensors, according to the expected input shapes,
    /// from a set of flat data
    pub fn load_input_flat(&self, input: Vec<Vec<N>>) -> Result<Vec<Tensor<N>>> {
        let input_tensor = input
            .into_iter()
            .zip(self.unpadded_input_shapes())
            .map(|(inp, shape)| Tensor::new(shape, inp))
            .collect();
        self.prepare_inputs(input_tensor)
    }

    /// Compute the input shapes padded to the next power of two
    pub(crate) fn padded_input_shapes(&self) -> Vec<Vec<usize>> {
        Self::compute_padded_input_shapes(&self.unpadded_input_shapes)
    }

    /// Textual description of the model
    pub fn describe(&self) {
        info!("Model description:");
        info!("Unpadded input shapes: {:?}", self.unpadded_input_shapes);
        info!("Padded input shapes: {:?}", self.padded_input_shapes());
        for (idx, layer) in self.to_forward_iterator() {
            info!("\t- {}: {:?}", idx, layer.inputs);
            info!("\t- {}: {}", idx, layer.operation.describe());
        }
        info!("Output nodes:");
        for (idx, node) in self.output_nodes() {
            info!("\t- {}:{:?}", idx, node.outputs);
        }
    }

    /// Add re-quantization nodes to the model after the node with id `input_node_id`
    /// It creates as many requant layers as there are output wires of the input node
    pub(crate) fn add_requant_nodes(
        &mut self,
        requants: Vec<Requant>,
        input_node_id: NodeId,
    ) -> anyhow::Result<Vec<NodeId>> {
        let input_node = self
            .nodes
            .get(&input_node_id)
            .ok_or(anyhow!("Node {input_node_id} not found in the model"))?;
        let num_outputs = input_node.outputs.len();
        // we want to create new requant nodes for each output of the input node. That means we need to
        // create one output edge from input_node to new requant_node and need to copy the associated output wire
        let requant_nodes = input_node
            .outputs
            .iter()
            .enumerate()
            .zip(requants.into_iter())
            .map(|((i, wire), requant)| {
                // INPUT EDGES: for each output wire, we simply copy the index i, and set the node to be input_node_id
                let input_edges = wire
                    .edges
                    .iter()
                    .map(|_| Edge::new(input_node_id, i))
                    .collect();
                // OUTPUT EDGES: We simply copy the output wires of input_node since they are the same.
                // NOTE here we enforce that one requant  == one output wire. Later we might want to revisit that assumption if needed.
                let output_wires = wire.clone();
                Ok(Node::new_with_outputs(
                    input_edges,
                    Layer::Requant(requant),
                    vec![output_wires],
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        debug!(
            "Requant insertion: from input node {}: inputs: {:?}, outputs: {:?}",
            input_node_id,
            self.nodes.get(&input_node_id).unwrap().inputs,
            self.nodes.get(&input_node_id).unwrap().outputs
        );
        // remove edges from outputs of `input_node` - BEFORE adding the requant nodes to the model, since
        // that action will append to the input_node.outputs.
        // safe unwrap because already did it before - redo it here for borrowing safety reasons
        self.nodes.get_mut(&input_node_id).unwrap().outputs = vec![Default::default(); num_outputs];
        let requant_ids = requant_nodes
            .into_iter()
            .map(|node| self.add_node(node))
            .collect::<Result<Vec<_>>>()?;
        debug!(
            "Requant insertion: requant nodes: {:?}",
            requant_ids
                .iter()
                .map(|id| {
                    let requant_node = self.nodes.get(&id).unwrap();
                    format!(
                        "id: {:?}, inputs: {:?}, outputs: {:?}",
                        id, requant_node.inputs, requant_node.outputs
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        );
        // route inputs of the nodes using outputs of `input_node_id` to the newly inserted
        // requant node
        for requant_id in requant_ids.iter() {
            let requant_node = self.nodes.get(&requant_id).ok_or(anyhow!(
                "Requant node {requant_id} just inserted not found in the model"
            ))?;
            for (i, wire) in requant_node.outputs.clone().iter().enumerate() {
                // change inputs of each node using this output wire
                wire.edges.iter().filter(|edge| edge.node.is_some()).try_for_each(|edge|{
                    let node_id = edge.node.unwrap();
                    let node = self.nodes.get_mut(&node_id).ok_or(
                        anyhow!("Node {node_id}, which should use an output of requant node {requant_id}, not found in model")
                    )?;
                    ensure!(edge.index < node.inputs.len(),
                        "Node {node_id} has {} inputs, so cannot access input {}",
                        node.inputs.len(),
                        edge.index,
                    );
                    // check that this input was indeed referring to an output of input_node_id
                    let input_edge = &mut node.inputs[edge.index];
                    ensure!(input_edge.node.ok_or(
                        anyhow!("{} input of node {node_id} should not be an input of the model", edge.index)
                    )? == input_node_id,
                        "{} input of node {node_id} should be {input_node_id}", edge.index
                    );
                    // replace `input_node_id` with `requant_id`
                    input_edge.node = Some(*requant_id);
                    input_edge.index = i;
                    Ok(())
                })?;
            }
        }
        Ok(requant_ids)
    }

    /// Corner-case method to add a node whose inputs correspond to the outputs of a node already inserted in the model
    /// The `NodeId` of the already inserted node is the `previous_node_id` input; if no id is provided, it is assumed
    /// that the inputs of the node correspond to the inputs of the model
    pub fn add_consecutive_layer(
        &mut self,
        layer: Layer<N>,
        previous_node_id: Option<NodeId>,
    ) -> anyhow::Result<NodeId> {
        let num_outputs = if let Some(id) = &previous_node_id {
            let previous_node = self
                .nodes
                .get(id)
                .ok_or(anyhow!("Node {id} not found in model"))?;
            previous_node.outputs.len()
        } else {
            // correspond to inputs of the model
            self.input_shapes.len()
        };

        let new_node = Node::new(
            (0..num_outputs)
                .map(|i| Edge {
                    node: previous_node_id,
                    index: i,
                })
                .collect(),
            layer,
        );
        self.add_node(new_node)
    }

    /// Add the node provided as input to the model. The id of the added node is
    /// computed inside this method and returned as output
    pub fn add_node(&mut self, node: Node<N>) -> anyhow::Result<NodeId> {
        let node_id = (0..self.nodes.len() + 1)
            .find(|i| !self.nodes.contains_key(i))
            .ok_or(anyhow!("No valid node id found for new node"))?;
        self.add_node_with_id(node_id, node)?;
        Ok(node_id)
    }

    /// Add the node provided as input to the model, binding it to the `node id`
    /// provided as input
    pub fn add_node_with_id(&mut self, node_id: NodeId, node: Node<N>) -> anyhow::Result<()> {
        // iterate over the inputs of the node and add the edges to the outputs of
        // corresponding nodes already in the model
        for (i, input_edge) in node.inputs.iter().enumerate() {
            if let Some(input_node_id) = &input_edge.node {
                let input_node = self.nodes.get_mut(input_node_id).ok_or(anyhow!(
                    "Node {} for input {} of new node not found in model",
                    input_node_id,
                    i,
                ))?;
                ensure!(
                    input_edge.index < input_node.outputs.len(),
                    "Specified output number {} for node {}, which has only {} outputs",
                    input_edge.index,
                    input_node_id,
                    input_node.outputs.len(),
                );
                input_node.outputs[input_edge.index].edges.push(Edge {
                    node: Some(node_id),
                    index: i,
                });
            }
        }

        self.nodes.insert(node_id, node);

        Ok(())
    }

    // Label the edges provided as input as the output edges of the model. If no edge is provided,
    // then the method assumes there is a node without routed output edges, and the outputs of
    // this node will be labelled as the output edges of the model
    pub fn route_output(&mut self, output_edges: Option<Vec<Edge>>) -> Result<()> {
        if let Some(output_edges) = output_edges {
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
        } else {
            // find the node with no output edges, which will be considered the output node
            let out_node = self.nodes.iter_mut().find(|(_id, node)| {
                node.outputs
                    .iter()
                    .all(|out| out.clone() == Default::default())
            });
            ensure!(out_node.is_some(), "No output node found for model");
            let node = out_node.unwrap().1;
            node.outputs.iter_mut().enumerate().for_each(|(i, out)| {
                out.edges = vec![Edge {
                    node: None,
                    index: i,
                }]
            });
        }

        Ok(())
    }

    /// Return the set of output nodes, that are nodes where at least one output
    /// tensor is an output of the model
    pub(crate) fn output_nodes(&self) -> Vec<(NodeId, &Node<N>)> {
        self.nodes
            .iter()
            .filter_map(|(id, node)| {
                if node
                    .outputs
                    .iter()
                    .all(|wire| wire.edges.iter().any(|edge| edge.node.is_none()))
                {
                    Some((*id, node))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Model<f32> {
    pub fn run_float(&self, input: &[Tensor<f32>]) -> anyhow::Result<Vec<Tensor<f32>>> {
        Ok(self
            .run::<GoldilocksExt2>(input)?
            .outputs()?
            .into_iter()
            .cloned()
            .collect())
    }
}

impl<N: Number> Model<N> {
    pub(crate) fn run_with_tracker<E>(
        &self,
        input: &[Tensor<N>],
        mut tracker: Option<&mut InferenceTracker>,
    ) -> anyhow::Result<InferenceTrace<'_, E, N>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
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
                        "Node {} requested output {} for node {}, which has only {} outputs",
                        node_id,
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
            // add output tensors to tracker, if any
            if let Some(tracker) = &mut tracker {
                for (i, out) in output.outputs().into_iter().enumerate() {
                    tracker.track(node_id, i, out.to_f32()?);
                }
            }
            let new_step = StepData {
                inputs: inputs.into_iter().cloned().collect(),
                outputs: output,
                unpadded_output_shapes: output_shapes,
            };
            trace.new_step(
                node_id,
                InferenceStep {
                    op: &node.operation,
                    step_data: new_step,
                },
            );
        }

        // compute the output tensor from the outputs of the output nodes
        let output_nodes = self.output_nodes();
        let mut outputs = BTreeMap::new();
        for (id, out_node) in output_nodes {
            let node_outputs = trace
                .get_step(&id)
                .ok_or(anyhow!("Output node {} not found in trace", id))?
                .outputs();
            ensure!(
                node_outputs.len() == out_node.outputs.len(),
                "Number of outputs found in trace ({}) for node {id} is different from number of expected outputs ({})",
                node_outputs.len(),
                out_node.outputs.len()
            );
            for (i, wire) in out_node.outputs.iter().enumerate() {
                if let Some(out_index) = wire.edges.iter().find_map(|edge| {
                    if edge.node.is_none() {
                        Some(edge.index)
                    } else {
                        None
                    }
                }) {
                    // if this output wire is an output of the model, insert in the collection of the
                    // model outputs, paired with the index among the outputs of the model
                    ensure!(
                        outputs.insert(out_index, node_outputs[i]).is_none(),
                        "Trying to insert twice an output value for the same index {out_index}"
                    );
                }
            }
        }
        // check that all outputs have been found
        ensure!(
            !outputs.is_empty(),
            "No outputs found for the model: {outputs:?}"
        );
        ensure!(
            *outputs.first_key_value().unwrap().0 == 0
                && *outputs.last_key_value().unwrap().0 == outputs.len() - 1
        );

        trace.output = outputs.into_values().cloned().collect();

        Ok(trace)
    }

    /// Run the inference of the model, producing the `InferenceTrace` necessary to
    /// later prove the model. The outputs of the model can be fetched from the returned
    /// trace
    pub fn run<E>(&self, input: &[Tensor<N>]) -> anyhow::Result<InferenceTrace<'_, E, N>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: ExtensionField + Serialize + DeserializeOwned,
        Layer<N>: Evaluate<N>,
    {
        self.run_with_tracker(input, None)
    }
}

/// Collection of the proving contexts of all the nodes in the model
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct ModelCtx<E: ExtensionField + DeserializeOwned>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    pub(crate) nodes: HashMap<NodeId, NodeCtx<E>>,
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        ScalingFactor, ScalingStrategy, init_test_logging, init_test_logging_default,
        layers::{
            Layer,
            activation::{Activation, Relu},
            convolution::{Convolution, SchoolBookConv},
            dense::Dense,
            matrix_mul::{MatMul, OperandMatrix},
            pooling::{MAXPOOL2D_KERNEL_SIZE, Maxpool2D, Pooling},
            provable::{Edge, Node, OpInfo, evaluate_layer},
            requant::Requant,
        },
        padding::{PaddingMode, pad_model},
        quantization::{self, InferenceObserver},
        tensor::Number,
        testing::{Pcs, random_bool_vector, random_vector},
    };
    use anyhow::Result;
    use ark_std::rand::{Rng, RngCore, thread_rng};
    use ff_ext::ExtensionField;
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::{
        mle::{IntoMLE, MultilinearExtension},
        virtual_poly::VirtualPolynomial,
    };
    use sumcheck::structs::{IOPProverState, IOPVerifierState};

    use crate::{Element, default_transcript, quantization::TensorFielder, tensor::Tensor};

    type F = GoldilocksExt2;
    const SELECTOR_DENSE: usize = 0;
    const SELECTOR_RELU: usize = 1;
    const SELECTOR_POOLING: usize = 2;
    const MOD_SELECTOR: usize = 2;

    impl Model<Element> {
        pub fn random(num_dense_layers: usize) -> Result<(Self, Vec<Tensor<Element>>)> {
            let mut rng = thread_rng();
            Self::random_with_rng(num_dense_layers, &mut rng)
        }
        /// Returns a random model with specified number of dense layers and a matching input.
        /// Note that currently everything is considered padded, e.g. unpadded_shape = padded_shape
        pub fn random_with_rng<R: RngCore>(
            num_dense_layers: usize,
            rng: &mut R,
        ) -> Result<(Self, Vec<Tensor<Element>>)> {
            let mut last_row: usize = rng.gen_range(3..15);
            let mut model = Self::new_from_input_shapes(
                vec![vec![last_row.next_power_of_two()]],
                PaddingMode::NoPadding,
            );
            let mut last_node_id = None;
            for selector in 0..num_dense_layers {
                if selector % MOD_SELECTOR == SELECTOR_DENSE {
                    // if true {
                    // last row becomes new column
                    let (nrows, ncols): (usize, usize) = (rng.gen_range(3..15), last_row);
                    last_row = nrows;
                    let dense =
                        Dense::random(vec![nrows.next_power_of_two(), ncols.next_power_of_two()]);
                    // Figure out the requant information such that output is still within range
                    let (min_output_range, max_output_range) =
                        dense.output_range(*quantization::MIN, *quantization::MAX);
                    let output_scaling_factor = ScalingFactor::from_scale(
                        ((max_output_range - min_output_range) as f64
                            / (*quantization::MAX - *quantization::MIN) as f64)
                            as f32,
                        None,
                    );
                    let input_scaling_factor = ScalingFactor::from_scale(1.0, None);
                    let max_model = dense.matrix.max_value().max(dense.bias.max_value()) as f32;
                    let model_scaling_factor = ScalingFactor::from_absolute_max(max_model, None);

                    let intermediate_bit_size = dense.output_bitsize();
                    let requant = Requant::from_scaling_factors(
                        input_scaling_factor,
                        model_scaling_factor,
                        output_scaling_factor,
                        intermediate_bit_size,
                    );

                    last_node_id =
                        Some(model.add_consecutive_layer(Layer::Dense(dense), last_node_id)?);
                    last_node_id =
                        Some(model.add_consecutive_layer(Layer::Requant(requant), last_node_id)?);
                } else if selector % MOD_SELECTOR == SELECTOR_RELU {
                    last_node_id = Some(model.add_consecutive_layer(
                        Layer::Activation(Activation::Relu(Relu::new())),
                        last_node_id,
                    )?);
                    // no need to change the `last_row` since RELU layer keeps the same shape
                    // of outputs
                } else if selector % MOD_SELECTOR == SELECTOR_POOLING {
                    // Currently unreachable until Model is updated to work with higher dimensional tensors
                    // TODO: Implement higher dimensional tensor functionality.
                    last_node_id = Some(model.add_consecutive_layer(
                        Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default())),
                        last_node_id,
                    )?);
                    last_row -= MAXPOOL2D_KERNEL_SIZE - 1;
                } else {
                    panic!("random selection shouldn't be in that case");
                }
            }
            model.route_output(None).unwrap();
            let inputs = model
                .input_shapes()
                .iter()
                .map(|shape| Tensor::random(shape))
                .collect();
            Ok((model, inputs))
        }

        /// Returns a model that only contains pooling and relu layers.
        /// The output [`Model`] will contain `num_layers` [`Maxpool2D`] layers and a [`Dense`] layer as well.
        pub fn random_pooling(num_layers: usize) -> Result<(Self, Vec<Tensor<Element>>)> {
            let mut rng = thread_rng();
            // Since Maxpool reduces the size of the output based on the kernel size and the stride we need to ensure that
            // Our starting input size is large enough for the number of layers.

            // If maxpool input matrix has dimensions w x h then output has width and height
            // out_w = (w - kernel_size) / stride + 1
            // out_h = (h - kenrel_size) / stride + 1
            // Hence to make sure we have a large enough tensor for the last step
            // we need to have that w_first > 2^{num_layers + 1} + 2^{num_layers}
            // and likewise for h_first.

            let minimum_initial_size = (1 << num_layers) * (3usize);

            let mut input_shape = (0..3)
                .map(|i| {
                    if i < 1 {
                        rng.gen_range(1..5usize).next_power_of_two()
                    } else {
                        (minimum_initial_size + rng.gen_range(1..4usize)).next_power_of_two()
                    }
                })
                .collect::<Vec<usize>>();

            let mut model =
                Model::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::NoPadding);

            let inputs = model
                .input_shapes()
                .iter()
                .map(|shape| Tensor::random(shape))
                .collect();

            let info = Maxpool2D::default();
            let mut last_node_id = None;
            for _ in 0..num_layers {
                input_shape
                    .iter_mut()
                    .skip(1)
                    .for_each(|dim| *dim = (*dim - info.kernel_size) / info.stride + 1);
                last_node_id = Some(model.add_consecutive_layer(
                    Layer::Pooling(Pooling::Maxpool2D(info)),
                    last_node_id,
                )?);
            }

            let (nrows, ncols): (usize, usize) =
                (rng.gen_range(3..15), input_shape.iter().product::<usize>());

            model.add_consecutive_layer(
                Layer::Dense(Dense::random(vec![
                    nrows.next_power_of_two(),
                    ncols.next_power_of_two(),
                ])),
                last_node_id,
            )?;

            model.route_output(None)?;

            Ok((model, inputs))
        }
    }

    #[test]
    fn test_model_long() {
        let (model, input) = Model::random(3).unwrap();
        model.run::<F>(&input).unwrap();
    }

    pub fn check_tensor_consistency_field<E: ExtensionField>(
        real_tensor: Tensor<E>,
        padded_tensor: Tensor<E>,
    ) {
        let n_x = padded_tensor.shape[1];
        for i in 0..real_tensor.shape[0] {
            for j in 0..real_tensor.shape[1] {
                for k in 0..real_tensor.shape[1] {
                    // if(real_tensor.data[i*real_tensor.shape[1]*real_tensor.shape[1]+j*real_tensor.shape[1]+k] > 0){
                    assert!(
                        real_tensor.data[i * real_tensor.shape[1] * real_tensor.shape[1]
                            + j * real_tensor.shape[1]
                            + k]
                            == padded_tensor.data[i * n_x * n_x + j * n_x + k],
                        "Error in tensor consistency"
                    );
                    //}else{
                    //   assert!(-E::from(-real_tensor.data[i*real_tensor.shape[1]*real_tensor.shape[1]+j*real_tensor.shape[1]+k] as u64) == E::from(padded_tensor.data[i*n_x*n_x + j*n_x + k] as u64) ,"Error in tensor consistency");
                    //}
                }

                // assert!(real_tensor.data[i*real_tensor.shape[1]*real_tensor.shape[1]+j ] == padded_tensor.data[i*n_x*n_x + j],"Error in tensor consistency");
            }
        }
    }

    fn random_vector_quant(n: usize) -> Vec<Element> {
        // vec![thread_rng().gen_range(-128..128); n]
        random_vector(n)
    }

    #[test]
    fn test_cnn() {
        let mut in_dimensions: Vec<Vec<usize>> =
            vec![vec![1, 32, 32], vec![16, 29, 29], vec![4, 26, 26]];

        for i in 0..in_dimensions.len() {
            for j in 0..in_dimensions[0].len() {
                in_dimensions[i][j] = (in_dimensions[i][j]).next_power_of_two();
            }
        }
        // println!("in_dimensions: {:?}", in_dimensions);
        let w1 = random_vector_quant(16 * 16);
        let w2 = random_vector_quant(16 * 4 * 16);
        let w3 = random_vector_quant(16 * 8);

        let shape1 = vec![1 << 4, 1 << 0, 1 << 2, 1 << 2]; // [16, 1, 4, 4]
        let shape2 = vec![1 << 2, 1 << 4, 1 << 2, 1 << 2]; // [4, 16, 4, 4]
        let shape3 = vec![1 << 1, 1 << 2, 1 << 2, 1 << 2]; // [2, 4, 4, 4]
        let bias1: Tensor<Element> = Tensor::zeros(vec![shape1[0]]);
        let bias2: Tensor<Element> = Tensor::zeros(vec![shape2[0]]);
        let bias3: Tensor<Element> = Tensor::zeros(vec![shape3[0]]);

        let trad_conv1: Tensor<Element> = Tensor::new(shape1.clone(), w1.clone());
        let trad_conv2: Tensor<i128> = Tensor::new(shape2.clone(), w2.clone());
        let trad_conv3: Tensor<i128> = Tensor::new(shape3.clone(), w3.clone());

        let input_shape = vec![1, 32, 32];

        let mut model =
            Model::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::Padding);
        let input = Tensor::random(&model.input_shapes()[0]);
        let first_id = model
            .add_consecutive_layer(
                Layer::Convolution(
                    Convolution::new(trad_conv1.clone(), bias1.clone())
                        .into_padded_and_ffted(&in_dimensions[0]),
                ),
                None,
            )
            .unwrap();
        let second_id = model
            .add_consecutive_layer(
                Layer::Convolution(
                    Convolution::new(trad_conv2.clone(), bias2.clone())
                        .into_padded_and_ffted(&in_dimensions[1]),
                ),
                Some(first_id),
            )
            .unwrap();
        let _third_id = model
            .add_consecutive_layer(
                Layer::Convolution(
                    Convolution::new(trad_conv3.clone(), bias3.clone())
                        .into_padded_and_ffted(&in_dimensions[2]),
                ),
                Some(second_id),
            )
            .unwrap();
        model.route_output(None).unwrap();

        // END TEST
        let trace = model.run::<F>(&vec![input.clone()]).unwrap();

        let mut model2 = Model::new_from_input_shapes(vec![input_shape], PaddingMode::NoPadding);
        let first_id = model2
            .add_consecutive_layer(
                Layer::SchoolBookConvolution(SchoolBookConv(Convolution::new(trad_conv1, bias1))),
                None,
            )
            .unwrap();
        let second_id = model2
            .add_consecutive_layer(
                Layer::SchoolBookConvolution(SchoolBookConv(Convolution::new(trad_conv2, bias2))),
                Some(first_id),
            )
            .unwrap();
        let _third_id = model2
            .add_consecutive_layer(
                Layer::SchoolBookConvolution(SchoolBookConv(Convolution::new(trad_conv3, bias3))),
                Some(second_id),
            )
            .unwrap();
        model2.route_output(None).unwrap();
        let trace2 = model.run::<F>(&vec![input]).unwrap();

        check_tensor_consistency_field::<GoldilocksExt2>(
            trace2.outputs().unwrap()[0].to_fields(),
            trace.outputs().unwrap()[0].to_fields(),
        );
    }

    #[test]
    fn test_conv_maxpool() {
        let input_shape = vec![3usize, 32, 32];
        let shape1 = vec![6, 3, 5, 5];
        let filter = Tensor::random(&shape1);
        let bias1 = Tensor::random(&vec![shape1[0]]);

        let mut model =
            Model::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::Padding);
        let conv_layer = model
            .add_consecutive_layer(
                Layer::Convolution(
                    Convolution::new(filter.clone(), bias1.clone())
                        .into_padded_and_ffted(&input_shape),
                ),
                None,
            )
            .unwrap();
        let _pool_layer = model
            .add_consecutive_layer(
                Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default())),
                Some(conv_layer),
            )
            .unwrap();
        model.route_output(None).unwrap();

        // TODO: have a "builder" for the model that automatically tracks the shape after each layer such that
        // we can just do model.prepare_input(&input).
        // Here is not possible since we didnt run through the onnx loader
        let input = Tensor::random(&input_shape);
        let input_padded = model.prepare_inputs(vec![input]).unwrap();
        let _ = model.run::<F>(&input_padded).unwrap();
    }

    #[test]
    fn test_model_manual_run() {
        let dense1 = Dense::<Element>::random(vec![
            10usize.next_power_of_two(),
            11usize.next_power_of_two(),
        ]);
        let dense2 = Dense::<Element>::random(vec![
            7usize.next_power_of_two(),
            dense1.ncols().next_power_of_two(),
        ]);
        let input_shape = vec![dense1.ncols()];
        let input = Tensor::<Element>::random(&input_shape);
        let output1 = evaluate_layer::<GoldilocksExt2, _, _>(&dense1, &vec![&input], None)
            .unwrap()
            .outputs()[0]
            .clone();
        let final_output = evaluate_layer::<GoldilocksExt2, _, _>(&dense2, &vec![&output1], None)
            .unwrap()
            .outputs()[0]
            .clone();

        let mut model =
            Model::<Element>::new_from_input_shapes(vec![input_shape], PaddingMode::NoPadding);
        let first_id = model
            .add_consecutive_layer(Layer::Dense(dense1.clone()), None)
            .unwrap();
        let second_id = model
            .add_consecutive_layer(Layer::Dense(dense2.clone()), Some(first_id))
            .unwrap();
        model.route_output(None).unwrap();

        let trace = model.run::<F>(&vec![input]).unwrap();
        assert_eq!(trace.steps.len(), 2);
        // Verify first step
        assert_eq!(*trace.get_step(&first_id).unwrap().outputs()[0], output1);

        // Verify second step
        assert_eq!(
            *trace.get_step(&second_id).unwrap().outputs()[0],
            final_output.clone()
        );
        let (nrow, _) = (dense2.nrows(), dense2.ncols());
        assert_eq!(final_output.get_data().len(), nrow);
    }

    use ff::Field;
    #[test]
    fn test_model_sequential() {
        let (model, input) = Model::random(1).unwrap();
        model.describe();
        let trace = model.run::<F>(&input).unwrap().to_field();
        let dense_layers = model
            .to_unstable_iterator()
            .flat_map(|(id, l)| match l.operation {
                Layer::Dense(ref dense) => Some((*id, dense.clone())),
                _ => None,
            })
            .collect_vec();
        let matrices_mle = dense_layers
            .iter()
            .map(|(id, d)| (*id, d.matrix.to_2d_mle::<F>()))
            .collect_vec();
        assert_eq!(dense_layers.len(), 1);
        let point1 = random_bool_vector(dense_layers[0].1.matrix.nrows_2d().ilog2() as usize);
        let computed_eval1 = trace
            .get_step(&dense_layers[0].0)
            .expect(format!("Node with id {} not found", dense_layers[0].0).as_str())
            .outputs()[0]
            .get_data()
            .to_vec()
            .into_mle()
            .evaluate(&point1);
        let flatten_mat1 = matrices_mle[0].1.fix_high_variables(&point1);
        let bias_eval = dense_layers[0]
            .1
            .bias
            .evals_flat::<F>()
            .into_mle()
            .evaluate(&point1);
        let computed_eval1_no_bias = computed_eval1 - bias_eval;
        let input_vector = trace.input[0].clone();
        // since y = SUM M(j,i) x(i) + B(j)
        // then
        // y(r) - B(r) = SUM_i m(r,i) x(i)
        let full_poly = vec![
            flatten_mat1.clone().into(),
            input_vector.get_data().to_vec().into_mle().into(),
        ];
        let mut vp = VirtualPolynomial::new(flatten_mat1.num_vars());
        vp.add_mle_list(full_poly, F::ONE);
        #[allow(deprecated)]
        let (proof, _state) =
            IOPProverState::<F>::prove_parallel(vp.clone(), &mut default_transcript());
        let (p2, _s2) =
            IOPProverState::prove_batch_polys(1, vec![vp.clone()], &mut default_transcript());
        let given_eval1 = proof.extract_sum();
        assert_eq!(p2.extract_sum(), proof.extract_sum());
        assert_eq!(computed_eval1_no_bias, given_eval1);

        let _subclaim = IOPVerifierState::<F>::verify(
            computed_eval1_no_bias,
            &proof,
            &vp.aux_info,
            &mut default_transcript(),
        );
    }

    use crate::{Context, Prover, verify};
    use transcript::BasicTranscript;

    use super::Model;

    #[test]
    #[ignore = "This test should be deleted since there is no requant and it is not testing much"]
    fn test_single_matvec_prover() {
        let w1 = random_vector_quant(1024 * 1024);
        let conv1 = Tensor::new(vec![1024, 1024], w1.clone());
        let w2 = random_vector_quant(1024);
        let conv2 = Tensor::new(vec![1024], w2.clone());
        let input_shape = vec![1024];

        let mut model = Model::new_from_input_shapes(vec![input_shape], PaddingMode::Padding);
        let input = Tensor::random(&model.input_shapes()[0]);
        model
            .add_consecutive_layer(Layer::Dense(Dense::new(conv1, conv2)), None)
            .unwrap();
        model.route_output(None).unwrap();
        model.describe();
        let trace = model.run::<F>(&vec![input]).unwrap();
        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"m2vec");
        let ctx = Context::<GoldilocksExt2, Pcs<GoldilocksExt2>>::generate(&model, None)
            .expect("Unable to generate context");
        let io = trace.to_verifier_io();
        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>, _> =
            Prover::new(&ctx, &mut tr);
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");
        verify::<_, _, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_single_matmul_prover() {
        // layer matrix shape
        let m_shape = vec![1000, 2000];
        let m = random_vector_quant(m_shape[0] * m_shape[1]);
        let tensor_m = Tensor::new(m_shape, m);
        let input_shape = vec![768, tensor_m.nrows_2d()];
        let mut model =
            Model::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::Padding);
        let matmul_layer = MatMul::new(
            OperandMatrix::Input,
            OperandMatrix::new_weight_matrix(tensor_m),
        )
        .unwrap();
        let padded_layer = matmul_layer.pad_next_power_of_two().unwrap();
        model
            .add_consecutive_layer(Layer::MatMul(padded_layer), None)
            .unwrap();
        model.route_output(None).unwrap();
        model.describe();

        let input = random_vector_quant(input_shape[0] * input_shape[1]);
        let input_tensor = model
            .prepare_inputs(vec![Tensor::new(input_shape, input)])
            .unwrap();

        let trace = model.run::<F>(&input_tensor).unwrap();
        let mut tr = BasicTranscript::<F>::new(b"matmul");
        let ctx = Context::<F, Pcs<F>>::generate(&model, None).expect("Unable to generate context");
        let io = trace.to_verifier_io();
        let prover = Prover::new(&ctx, &mut tr);
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript = BasicTranscript::<F>::new(b"matmul");
        verify::<_, _, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_single_cnn_prover() {
        let n_w = 1 << 2;
        let k_w = 1 << 4;
        let n_x = 1 << 5;
        let k_x = 1 << 1;

        let in_dimensions: Vec<Vec<usize>> =
            vec![vec![k_x, n_x, n_x], vec![16, 29, 29], vec![4, 26, 26]];

        let conv1 = Tensor::random(&vec![k_w, k_x, n_w, n_w]);
        let input_shape = vec![k_x, n_x, n_x];

        let mut model = Model::new_from_input_shapes(vec![input_shape], PaddingMode::Padding);
        let input = Tensor::random(&model.input_shapes()[0]);
        let _conv_layer = model
            .add_consecutive_layer(
                Layer::Convolution(
                    Convolution::new(conv1.clone(), Tensor::random(&vec![conv1.kw()]))
                        .into_padded_and_ffted(&in_dimensions[0]),
                ),
                None,
            )
            .unwrap();
        model.route_output(None).unwrap();
        model.describe();
        let trace = model.run::<F>(&vec![input]).unwrap();
        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"m2vec");
        let ctx = Context::<GoldilocksExt2, Pcs<GoldilocksExt2>>::generate(&model, None)
            .expect("Unable to generate context");
        let io = trace.to_verifier_io();

        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>, _> =
            Prover::new(&ctx, &mut tr);
        let proof = prover.prove(trace).expect("unable to generate proof");

        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");
        verify::<_, _, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_cnn_prover() {
        for i in 0..3 {
            for j in 2..5 {
                for l in 1..4 {
                    for n in 1..(j - 1) {
                        let n_w = 1 << n;
                        let k_w = 1 << l;
                        let n_x = 1 << j;
                        let k_x = 1 << i;

                        let in_dimensions: Vec<Vec<usize>> =
                            vec![vec![k_x, n_x, n_x], vec![16, 29, 29], vec![4, 26, 26]];
                        let input_shape = vec![k_x, n_x, n_x];
                        let conv1 = Tensor::random(&vec![k_w, k_x, n_w, n_w]);
                        let mut model = Model::<Element>::new_from_input_shapes(
                            vec![input_shape],
                            PaddingMode::Padding,
                        );
                        let input = Tensor::random(&model.input_shapes()[0]);
                        model
                            .add_consecutive_layer(
                                Layer::Convolution(
                                    Convolution::new(
                                        conv1.clone(),
                                        Tensor::random(&vec![conv1.kw()]),
                                    )
                                    .into_padded_and_ffted(&in_dimensions[0]),
                                ),
                                None,
                            )
                            .unwrap();
                        model.route_output(None).unwrap();
                        model.describe();
                        let trace = model.run::<F>(&vec![input]).unwrap();
                        let mut tr: BasicTranscript<GoldilocksExt2> =
                            BasicTranscript::new(b"m2vec");
                        let ctx =
                            Context::<GoldilocksExt2, Pcs<GoldilocksExt2>>::generate(&model, None)
                                .expect("Unable to generate context");
                        let io = trace.to_verifier_io();
                        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>, _> =
                            Prover::new(&ctx, &mut tr);
                        let proof = prover.prove(trace).expect("unable to generate proof");
                        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
                            BasicTranscript::new(b"m2vec");
                        verify::<_, _, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
                    }
                }
            }
        }
    }

    type E = GoldilocksExt2;
    type T = BasicTranscript<GoldilocksExt2>;
    type N = Element;

    fn build_test_model<N: Number, const INPUT_SIZE: usize>() -> Model<N> {
        let input_shape = vec![INPUT_SIZE];
        let mut model =
            Model::<N>::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::NoPadding);
        // add input dense layer
        // generate random dense matrix
        let ncols = input_shape[0];
        let nrows = 42;
        let dense = Dense::random(vec![nrows, ncols]);
        let dense_out_shape =
            &dense.output_shapes(&model.unpadded_input_shapes(), PaddingMode::NoPadding)[0];
        let input_node = model
            .add_consecutive_layer(
                Layer::Dense(dense),
                None, // it's connected to the inputs of the model
            )
            .unwrap();
        // add activation layer
        let relu = Activation::Relu(Relu::new());
        let relu_node = model
            .add_consecutive_layer(Layer::Activation(relu), Some(input_node))
            .unwrap();
        // add another dense layer as output
        let nrows = 37;
        let ncols = dense_out_shape[0]; // it's a vector, so it has only one dimension
        let dense = Dense::random(vec![nrows, ncols]);
        let output_node = model
            .add_consecutive_layer(Layer::Dense(dense), Some(relu_node))
            .unwrap();
        model.route_output(None).unwrap();

        assert_eq!(model.output_nodes()[0].0, output_node);

        model
    }

    #[test]
    fn test_model_inference() {
        const INPUT_SIZE: usize = 45;
        let model = build_test_model::<N, INPUT_SIZE>();
        let input_shape = model.input_shapes()[0].clone();

        let input = random_vector(input_shape.iter().product());
        let input_tensor = Tensor::new(input_shape, input);
        let trace = model.run::<E>(&[input_tensor]).unwrap();
        assert_eq!(trace.steps.len(), 3);
    }

    #[test]
    fn test_model_float_inference() {
        const INPUT_SIZE: usize = 45;
        let model = build_test_model::<f32, INPUT_SIZE>();
        let input_shape = model.input_shapes()[0].clone();

        let input_tensor = Tensor::random(&input_shape);
        let trace = model.run::<E>(&[input_tensor]).unwrap();
        assert_eq!(trace.steps.len(), 3);
    }

    pub(crate) fn prove_model(model: Model<f32>) -> anyhow::Result<()> {
        let float_inputs = model
            .input_shapes()
            .into_iter()
            .map(|shape| Tensor::random(&shape))
            .collect_vec();
        let (quantized_model, md) = InferenceObserver::new().quantize(model)?;
        let model = pad_model(quantized_model)?;

        model.describe();

        // quantize and pad input tensor
        let input_tensors = float_inputs
            .into_iter()
            .zip(&md.input)
            .map(|(tensor, s)| tensor.quantize(s).pad_next_power_of_two())
            .collect_vec();

        let trace = model.run(&input_tensors)?;
        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"model");
        let ctx = Context::<GoldilocksExt2, Pcs<GoldilocksExt2>>::generate(&model, None)
            .expect("Unable to generate context");
        let prover: Prover<'_, E, T, _> = Prover::new(&ctx, &mut tr);
        let io = trace.to_verifier_io();
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"model");
        verify::<_, _, _>(ctx, proof, io, &mut verifier_transcript)
    }

    #[test]
    fn test_model_proving() {
        init_test_logging_default();
        const INPUT_SIZE: usize = 57;
        let model = build_test_model::<f32, INPUT_SIZE>();
        prove_model(model).unwrap();
    }

    /// 2 relus connected. First relu receives two inputs and pass that to the second relu
    /// This test checks that when inserting a requant layer in between, the inputs and output edges
    /// are still correct.
    /// Relu is easy since in inference, it can support many inputs.
    /// Graph wise:
    ///      A
    ///     / \\  <-- double inputs for C
    ///    B   C
    /// should become:
    ///       A
    ///     /  \\
    ///    R1  R2  <-- distinct requant layers !
    ///    /    \\   
    ///   B      C
    #[test]
    fn test_model_insert_requant() {
        init_test_logging_default();
        const FIRST_INPUT_SIZE: usize = 27;
        const SECOND_INPUT_SIZE: usize = 49;
        let input_shapes = vec![vec![FIRST_INPUT_SIZE], vec![SECOND_INPUT_SIZE]];
        let mut model =
            Model::<Element>::new_from_input_shapes(input_shapes.clone(), PaddingMode::NoPadding);
        let relu1 = model
            .add_node(Node::new(
                vec![Edge::new_at_edge(0), Edge::new_at_edge(1)],
                Layer::Activation(Activation::Relu(Relu)),
            ))
            .unwrap();
        // here we take the first two outputs of relu1
        let relu2 = model
            .add_node(Node::new(
                vec![Edge::new(relu1, 0), Edge::new(relu1, 1)],
                Layer::Activation(Activation::Relu(Relu)),
            ))
            .unwrap();
        // here we only want to take the first output of relu1
        let relu3 = model
            .add_node(Node::new(
                vec![Edge::new(relu1, 0)],
                Layer::Activation(Activation::Relu(Relu)),
            ))
            .unwrap();
        let input_tensor = vec![
            Tensor::random(&input_shapes[0]),
            Tensor::random(&input_shapes[1]),
        ];
        let test_sf = ScalingFactor::from_scale(1.0, None);
        // 2 requants, one for each outgoing output wire (one for relu2 and one for relu3)
        let requants = vec![Requant::from_scaling_factors(test_sf, test_sf, test_sf, 10); 2];
        let requants_ids = model.add_requant_nodes(requants, relu1).unwrap();
        assert_eq!(requants_ids.len(), 2);
        model
            .route_output(Some(vec![
                Edge {
                    node: Some(relu2),
                    index: 0,
                },
                Edge {
                    node: Some(relu2),
                    index: 1,
                },
                Edge {
                    node: Some(relu3),
                    index: 0,
                },
            ]))
            .unwrap();
        model.run::<GoldilocksExt2>(&input_tensor).unwrap();
    }

    #[test]
    fn test_model_multiple_outputs() {
        init_test_logging("debug");
        const FIRST_INPUT_SIZE: usize = 27;
        const SECOND_INPUT_SIZE: usize = 49;
        let input_shapes = vec![vec![FIRST_INPUT_SIZE], vec![SECOND_INPUT_SIZE]];
        let mut model = Model::<f32>::new_from_input_shapes(input_shapes, PaddingMode::NoPadding);
        // add first dense layer
        // generate random dense matrix
        let ncols = FIRST_INPUT_SIZE;
        let nrows = 42;
        let dense = Dense::random(vec![nrows, ncols]);
        let first_dense_out_shape = &dense.output_shapes(
            &vec![model.unpadded_input_shapes()[0].clone()],
            PaddingMode::NoPadding,
        )[0];
        let first_input_dense = model
            .add_node(Node::new(
                vec![Edge {
                    node: None,
                    index: 0,
                }],
                Layer::Dense(dense),
            ))
            .unwrap();
        // add second input dense layer
        let ncols = SECOND_INPUT_SIZE;
        let nrows = 47;
        let dense = Dense::random(vec![nrows, ncols]);
        let second_dense_out_shape = &dense.output_shapes(
            &vec![model.unpadded_input_shapes()[1].clone()],
            PaddingMode::NoPadding,
        )[0];
        let second_input_dense = model
            .add_node(Node::new(
                vec![Edge {
                    node: None,
                    index: 1,
                }],
                Layer::Dense(dense),
            ))
            .unwrap();
        // add Relu nodes
        let relu = Activation::Relu(Relu::new());
        let first_relu_node = model
            .add_consecutive_layer(Layer::Activation(relu.clone()), Some(first_input_dense))
            .unwrap();
        let second_relu_node = model
            .add_consecutive_layer(Layer::Activation(relu), Some(second_input_dense))
            .unwrap();
        // add other dense nodes
        let nrows = 52;
        let ncols = second_dense_out_shape[0]; // it's a vector, so it has only one dimension
        let dense = Dense::random(vec![nrows, ncols]);
        let first_output_node = model
            .add_consecutive_layer(Layer::Dense(dense), Some(second_relu_node))
            .unwrap();
        let nrows = 17;
        let ncols = first_dense_out_shape[0];
        let dense = Dense::random(vec![nrows, ncols]);
        let second_output_node = model
            .add_consecutive_layer(Layer::Dense(dense), Some(first_relu_node))
            .unwrap();

        model
            .route_output(Some(vec![
                Edge {
                    node: Some(first_output_node),
                    index: 0,
                },
                Edge {
                    node: Some(second_output_node),
                    index: 0,
                },
            ]))
            .unwrap();

        let out_node_ids = model
            .output_nodes()
            .into_iter()
            .map(|(id, _)| id)
            .collect_vec();

        assert_eq!(out_node_ids.len(), 2);
        assert!(out_node_ids.contains(&first_output_node));
        assert!(out_node_ids.contains(&second_output_node));

        model.describe();

        prove_model(model).unwrap();
    }
}
