mod error;

use anyhow::{Result, anyhow, ensure};
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::{BTreeMap, HashMap},
    fmt::Debug,
};
use transcript::Transcript;

use crate::{
    Claim, Element, Prover, ScalingFactor, ScalingStrategy, Tensor,
    commit::precommit::PolyID,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    lookup::context::LookupWitnessGen,
    model::trace::StepData,
    padding::{PaddingMode, ShapeInfo},
    tensor::{ConvData, Number},
};

use super::{Layer, LayerCtx, LayerProof, flatten::Flatten, requant::Requant};

pub(crate) type NodeId = usize;

pub use error::ProvableOpError;

/// Represents a link between an input/output wire of a node with an input/output wire of
/// another node.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Edge {
    // Reference to the node linked to this wire, will be `None` if the wire is an input or
    // output of the model
    pub(crate) node: Option<NodeId>,
    // The index of the wire of `node` which is linked to this wire
    pub(crate) index: usize,
}

impl Edge {
    pub fn new(node: NodeId, index: usize) -> Self {
        Self {
            node: Some(node),
            index,
        }
    }

    /// Edge when the node is an input or an output of the model
    pub fn new_at_edge(index: usize) -> Self {
        Self { node: None, index }
    }
}

/// Represents all the edges that are connected to a node's output wire
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct OutputWire {
    // needs to be a vector because the output of a node can be used as input to multiple nodes
    pub(crate) edges: Vec<Edge>,
}

/// Represents a node in a model
#[derive(Clone, Debug)]
pub struct ProvableNode<N> {
    pub(crate) inputs: Vec<Edge>,
    pub(crate) outputs: Vec<OutputWire>,
    pub(crate) operation: Layer<N>,
}

pub trait NodeEgdes {
    // Get input edges for a node
    fn inputs(&self) -> &[Edge];
    // Get output edges of a node
    fn outputs(&self) -> &[OutputWire];
}

impl<N> NodeEgdes for ProvableNode<N> {
    fn inputs(&self) -> &[Edge] {
        &self.inputs
    }

    fn outputs(&self) -> &[OutputWire] {
        &self.outputs
    }
}

impl<E: ExtensionField + DeserializeOwned> NodeEgdes for NodeCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    fn inputs(&self) -> &[Edge] {
        &self.inputs
    }

    fn outputs(&self) -> &[OutputWire] {
        &self.outputs
    }
}

impl<N: Number> ProvableNode<N> {
    // Create a new node, from the set of inputs edges and the operation performed by the node
    pub fn new(inputs: Vec<Edge>, operation: Layer<N>) -> Self {
        let num_outputs = operation.num_outputs(inputs.len());
        Self::new_with_outputs(inputs, operation, vec![Default::default(); num_outputs])
    }

    pub(crate) fn new_with_outputs(
        inputs: Vec<Edge>,
        operation: Layer<N>,
        outputs: Vec<OutputWire>,
    ) -> Self {
        Self {
            inputs,
            outputs,
            operation,
        }
    }
}

/// Represents the output of the evaluation of a node operation
#[derive(Clone, Debug)]
pub struct LayerOut<T, E: ExtensionField> {
    pub(crate) outputs: Vec<Tensor<T>>,
    pub(crate) proving_data: Option<ConvData<E>>,
}

impl<T, E: ExtensionField> LayerOut<T, E> {
    pub(crate) fn from_vec(out: Vec<Tensor<T>>) -> Self {
        Self {
            outputs: out,
            proving_data: None,
        }
    }

    pub(crate) fn from_tensor(out: Tensor<T>) -> Self {
        Self::from_vec(vec![out])
    }

    pub fn outputs(&self) -> Vec<&Tensor<T>> {
        self.outputs.iter().collect()
    }
}
/// Represents the proving context for a given node, altogether with the input
/// and output edges of the node
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct NodeCtx<E>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    pub(crate) inputs: Vec<Edge>,
    pub(crate) outputs: Vec<OutputWire>,
    pub(crate) ctx: LayerCtx<E>,
}

impl<E> NodeCtx<E>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    /// Get the claims corresponding to the output edges of a node.
    /// Requires the input claims for the nodes of the model using the
    /// outputs of the current node, and the claims of the output
    /// tensors of the model
    pub(crate) fn get_claims_for_node<'a, 'b>(
        &self,
        claims_by_node: &'a HashMap<NodeId, Vec<Claim<E>>>,
        output_claims: &'b [Claim<E>],
    ) -> Result<Vec<&'a Claim<E>>>
    where
        'b: 'a,
    {
        self.outputs.iter().map(|out| {
            // For now, we support in proving only one edge per output wire,
            // as if an output is used as input in different nodes, we need
            // to batch claims about the same polynomial. ToDo: batch claims
            assert_eq!(out.edges.len(), 1);
            let edge = &out.edges[0];
            Ok(if let Some(id) = &edge.node {
                let claims_for_node = claims_by_node.get(id).ok_or(
                    anyhow!("No claims found for layer {}", id)
                )?;
                ensure!(edge.index < claims_for_node.len(),
                    "Not enough claims found for node {}: required claim for input {}, but {} claims found",
                    id,
                    edge.index,
                    claims_for_node.len()
                );
                &claims_for_node[edge.index]
            } else {
                // it's an output node, so we use directly the claim for the corresponding output
                ensure!(edge.index < output_claims.len(),
                 "Required claim for output {} of the model, but only {} output claims found",
                 edge.index,
                 output_claims.len(),
                );
                &output_claims[edge.index]
            })
        }).collect()
    }

    /// Get the claims corresponding to the input tensors of the model.
    /// Requires as inputs the contexts for all the nodes in the model
    /// and the set of claims for the input tensors of all the nodes of
    /// the model
    pub(crate) fn input_claims<'a, I: Iterator<Item = (&'a NodeId, &'a Self)>>(
        nodes: I,
        claims_by_node: &HashMap<NodeId, Vec<Claim<E>>>,
    ) -> Result<Vec<&Claim<E>>> {
        let mut claims = BTreeMap::new();
        for (node_id, ctx) in nodes {
            for (i, edge) in ctx.inputs.iter().enumerate() {
                if edge.node.is_none() {
                    let claims_for_node = claims_by_node
                        .get(node_id)
                        .ok_or(anyhow!("Claim not found for node {}", node_id))?;
                    claims.insert(edge.index, &claims_for_node[i]);
                }
            }
        }
        ensure!(
            claims.len() >= 1,
            "No input claims found for the set of nodes provided"
        );
        let min_index = claims.first_key_value().unwrap().0;
        let max_index = claims.last_key_value().unwrap().0;
        ensure!(
            *min_index == 0 && *max_index == claims.len() - 1,
            "Not all input claims were found"
        );

        Ok(claims.into_iter().map(|(_, claim)| claim).collect())
    }
}

pub trait OpInfo {
    /// Returns the shapes of the outputs (in the same order)
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>>;

    /// Compute the number of output tensors, given the number of input tensors
    /// `num_inputs`
    fn num_outputs(&self, num_inputs: usize) -> usize;

    /// Textual description of the operation
    fn describe(&self) -> String;

    /// Specify whether the operation needs to be proven or not
    fn is_provable(&self) -> bool;
}

pub trait Evaluate<T: Number> {
    /// Evaluates the operation given any inputs tensors and constant inputs.
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<T>],
        unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> Result<LayerOut<T, E>, ProvableOpError>;
}

/// Helper method employed to call `Evaluate::evaluate` when there are no `unpadded_input_shapes`
/// or when the `E` type cannot be inferred automatically by the compiler
pub fn evaluate_layer<E: ExtensionField, T: Number, O: Evaluate<T>>(
    layer: &O,
    inputs: &[&Tensor<T>],
    unpadded_input_shapes: Option<Vec<Vec<usize>>>,
) -> Result<LayerOut<T, E>, ProvableOpError> {
    layer.evaluate(inputs, unpadded_input_shapes.unwrap_or_default())
}

pub trait ProveInfo<E: ExtensionField>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    /// Compute the proving context for the operation
    fn step_info(
        &self,
        id: PolyID,
        aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux), ProvableOpError>;

    /// Compute the data necessary to commit to the constant polynomials
    /// associated to the operation. Returns `None` if there are no
    /// constant polynomials to be committed for the given operation
    fn commit_info(&self, _id: NodeId) -> Vec<Option<(PolyID, Vec<E>)>> {
        vec![None]
    }
}

/// Output of `QuantizeOp` method over a layer
pub struct QuantizeOutput<Op> {
    /// The actual layer after quantization
    pub(crate) quanzited_op: Op,
    /// The scaling factor of the output wires of the operation
    pub(crate) output_scalings: Vec<ScalingFactor>,
    /// The requant layer to be added to the model, if any
    pub(crate) requant_layer: Option<Requant>,
}

pub trait QuantizeOp<S: ScalingStrategy> {
    type QuantizedOp: Sized;

    /// Convert an operation into its quantized version
    fn quantize_op(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>>;
}

/// Helper method employed to call `QuantizeOp::quantize_op` when the `QuantizationStrategy` type
/// canno be inferred automatically
pub fn quantize_op<S: ScalingStrategy, O: QuantizeOp<S>>(
    op: O,
    data: &S::AuxData,
    node_id: NodeId,
    input_scaling: &[ScalingFactor],
) -> anyhow::Result<QuantizeOutput<O::QuantizedOp>> {
    op.quantize_op(data, node_id, input_scaling)
}

pub trait PadOp {
    // Pad the dimensions of the tensors in node `self`, updating the `ShapeInfo` with the output shapes
    // of the node
    fn pad_node(self, _si: &mut ShapeInfo) -> Result<Self, ProvableOpError>
    where
        Self: Sized,
    {
        Ok(self)
    }
}

pub trait ProvableOp<E>: OpInfo + PadOp
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    type Ctx: VerifiableCtx<E>;

    /// Produces a proof of correct execution for this operation.
    fn prove<T: Transcript<E>>(
        &self,
        _node_id: NodeId,
        _ctx: &Self::Ctx,
        _last_claims: Vec<&Claim<E>>,
        _step_data: &StepData<E, E>,
        _prover: &mut Prover<E, T>,
    ) -> Result<Vec<Claim<E>>, ProvableOpError> {
        // Default implementation, to avoid having to implement this method in case `is_provable` is false
        assert!(
            !self.is_provable(),
            "Running default prove implementation for a provable operation! Implement prove method"
        );
        Ok(vec![Claim::default()])
    }

    /// Generate witness for a node where a lookup table is employed in proving
    fn gen_lookup_witness(
        &self,
        _id: NodeId,
        _gen: &mut LookupWitnessGen<E>,
        _step_data: &StepData<Element, E>,
    ) -> Result<(), ProvableOpError> {
        // Default implementation for nodes that don't employ a lookup table
        Ok(())
    }
}

pub trait Op<E, N>: Evaluate<N> + ProveInfo<E> + Debug + ProvableOp<E>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    N: Number,
{
}

pub trait VerifiableCtx<E>: Debug
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    type Proof: Sized;

    /// Verify proof for the given operation
    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>, ProvableOpError>;

    /// Compute the shapes of the output tensors, given the shapes of the input tensors
    /// and the padding mode
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>>;
}

// Helper method to call `VerifiableCtx::output_shapes` when the type `E`
// cannot be inferred automatically by the compiler
pub(crate) fn output_shapes<E: ExtensionField, C: VerifiableCtx<E>>(
    ctx: &C,
    input_shapes: &[Vec<usize>],
    padding_mode: PaddingMode,
) -> Vec<Vec<usize>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    ctx.output_shapes(input_shapes, padding_mode)
}

impl<E: ExtensionField> VerifiableCtx<E> for LayerCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    type Proof = LayerProof<E>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &LayerProof<E>,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>, ProvableOpError> {
        match self {
            LayerCtx::Dense(dense_ctx) => {
                if let LayerProof::Dense(proof) = proof {
                    dense_ctx.verify(proof, last_claims, verifier, shape_step)
                } else {
                    Err(ProvableOpError::ParameterError(
                        "dense proof not found for dense layer".to_string(),
                    ))
                }
            }
            LayerCtx::Convolution(conv_ctx) => {
                if let LayerProof::Convolution(proof) = proof {
                    conv_ctx.verify(proof, last_claims, verifier, shape_step)
                } else {
                    Err(ProvableOpError::ParameterError(
                        "conv proof not found for convolution layer".to_string(),
                    ))
                }
            }
            LayerCtx::Activation(activation_ctx) => {
                if let LayerProof::Activation(proof) = proof {
                    activation_ctx.verify(proof, last_claims, verifier, shape_step)
                } else {
                    Err(ProvableOpError::ParameterError(
                        "activation proof not found for activation layer".to_string(),
                    ))
                }
            }
            LayerCtx::Requant(requant_ctx) => {
                if let LayerProof::Requant(proof) = proof {
                    requant_ctx.verify(proof, last_claims, verifier, shape_step)
                } else {
                    Err(ProvableOpError::ParameterError(
                        "requant proof not found for requant layer".to_string(),
                    ))
                }
            }
            LayerCtx::Pooling(pooling_ctx) => {
                if let LayerProof::Pooling(proof) = proof {
                    pooling_ctx.verify(proof, last_claims, verifier, shape_step)
                } else {
                    Err(ProvableOpError::ParameterError(
                        "pooling proof not found for pooling layer".to_string(),
                    ))
                }
            }
            _ => Err(ProvableOpError::NotProvableLayer(format!("{:?}", self))),
        }
    }

    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
        match self {
            LayerCtx::Dense(dense_ctx) => dense_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Convolution(conv_ctx) => conv_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Activation(activation_ctx) => {
                output_shapes::<E, _>(activation_ctx, input_shapes, padding_mode)
            }
            LayerCtx::Requant(requant_ctx) => {
                output_shapes::<E, _>(requant_ctx, input_shapes, padding_mode)
            }
            LayerCtx::Pooling(pooling_ctx) => {
                output_shapes::<E, _>(pooling_ctx, input_shapes, padding_mode)
            }
            LayerCtx::Flatten => {
                <Flatten as OpInfo>::output_shapes(&Flatten, input_shapes, padding_mode)
            }
            _ => unreachable!(),
        }
    }
}
