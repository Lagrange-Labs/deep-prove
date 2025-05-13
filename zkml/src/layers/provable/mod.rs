mod error;
mod model;

use anyhow::{Result, anyhow, ensure};
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::{BTreeMap, HashMap},
    fmt::{Debug, format},
};
use transcript::Transcript;

use crate::{
    Claim, Element, Prover, ScalingFactor, Tensor,
    commit::precommit::PolyID,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    lookup::context::LookupWitnessGen,
    padding::{PaddingMode, ShapeInfo},
    tensor::{ConvData, Number},
};

use super::{Layer, LayerCtx, LayerProof, requant::Requant, flatten::Flatten};

pub(crate) type NodeId = usize;

pub use error::ProvableOpError;
pub use model::{InferenceTrace, ModelCtx, ProvableModel, ToIterator};

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

#[derive(Clone, Debug)]
pub struct ProvableNode<N> {
    pub(crate) inputs: Vec<Edge>,
    pub(crate) outputs: Vec<OutputWire>,
    pub(crate) operation: Layer<N>,
}

pub(crate) trait NodeEgdes {
    fn inputs(&self) -> &[Edge];
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
    pub(crate) fn new(inputs: Vec<Edge>, operation: Layer<N>) -> Self {
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

// pub struct ProvableNode<E, T, D> {
// inputs: Vec<Edge>,
// input_shape: Vec<usize>,
// outputs: Vec<OutputWire>,
// output_shape: Vec<usize>,
// pub(crate) operation: Box<dyn ProvableOp<E, T, D>>,
// }

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

    pub fn outputs(&self) -> Vec<&Tensor<T>> {
        self.outputs.iter().collect()
    }
}
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
    pub(crate) fn get_claims_for_node(
        &self,
        claims_by_node: &HashMap<NodeId, Vec<Claim<E>>>,
        output_claims: &[Claim<E>],
    ) -> Result<Vec<Claim<E>>> {
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
                claims_for_node[edge.index].clone() // ToDo: avoid clone
            } else {
                // it's an output node, so we use directly the claim for the corresponding output
                ensure!(edge.index < output_claims.len(),
                 "Required claim for output {} of the model, but only {} output claims found",
                 edge.index,
                 output_claims.len(),
                );
                output_claims[edge.index].clone()
            })
        }).collect()
    }

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

    fn num_outputs(&self, num_inputs: usize) -> usize;

    fn describe(&self) -> String;

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

pub(crate) fn evaluate_layer<E: ExtensionField, T: Number, O: Evaluate<T>>(
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
    fn step_info(
        &self,
        id: PolyID,
        aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux), ProvableOpError>;

    fn commit_info(&self, _id: NodeId) -> Vec<Option<(PolyID, Vec<E>)>> {
        vec![None]
    }
}

pub trait QuantizationStrategy {
    type AuxData: Sized;
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

pub trait QuantizeOp<Q: QuantizationStrategy> {
    type QuantizedOp: Sized;

    fn quantize_op(
        self,
        data: &Q::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>>;
}

pub trait PadOp {
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
        node_id: NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T>,
    ) -> Result<Vec<Claim<E>>, ProvableOpError> {
        // Default implementation, to avoid having to implement this method in case `is_provable` is false
        assert!(
            !self.is_provable(),
            "Running default prove implementation for a provable operation! Implement prove method"
        );
        Ok(vec![Claim::default()])
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        gen: &mut LookupWitnessGen<E>,
        step_data: &StepData<Element, E>,
    ) -> Result<(), ProvableOpError> {
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

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[Claim<E>],
        verifier: &mut Verifier<E, T>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>, ProvableOpError>;

    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>>;
}

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
        last_claims: &[Claim<E>],
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

pub struct InferenceStep<'a, E: ExtensionField, N, D> {
    pub(crate) op: &'a Layer<N>,
    pub(crate) step_data: StepData<D, E>,
}

impl<'a, E: ExtensionField, N, D> InferenceStep<'a, E, N, D> {
    pub fn outputs(&self) -> Vec<&Tensor<D>> {
        self.step_data.outputs.outputs()
    }
}

pub struct StepData<D, E: ExtensionField> {
    pub(crate) inputs: Vec<Tensor<D>>,
    pub(crate) outputs: LayerOut<D, E>,
    pub(crate) unpadded_output_shapes: Vec<Vec<usize>>,
}
