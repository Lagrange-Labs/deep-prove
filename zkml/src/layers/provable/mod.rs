use anyhow::{Result, anyhow, bail, ensure};
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::mle::{IntoMLE, MultilinearExtension};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{collections::HashMap, fmt::Debug};
use transcript::Transcript;

use crate::{
    Claim, Context, Element, Prover, ScalingFactor, ScalingStrategy, Tensor,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        activation::ActivationCtx,
        matrix_mul::MatMulCtx,
        pooling::PoolingCtx,
        requant::RequantCtx,
        transformer::{embeddings::EmbeddingsCtx, qkv::QKVCtx},
    },
    lookup::context::LookupWitnessGen,
    model::trace::StepData,
    padding::{PaddingMode, ShapeInfo},
    tensor::{ConvData, Number, Shape},
};

use super::{
    Layer, LayerCtx, LayerProof, convolution::ConvCtx, dense::DenseCtx, flatten::Flatten,
    requant::Requant,
};

pub(crate) type NodeId = usize;

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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node<N> {
    pub(crate) inputs: Vec<Edge>,
    /// Vector of outgoing wires. The index in this vector means that the corresponding OutputWire
    /// is linking the i-th output of this node to the designated output node.
    pub(crate) outputs: Vec<OutputWire>,
    pub(crate) operation: Layer<N>,
}

pub trait NodeEdges {
    // Get input edges for a node
    fn inputs(&self) -> &[Edge];
    // Get output edges of a node
    fn outputs(&self) -> &[OutputWire];
}

impl<N> NodeEdges for Node<N> {
    fn inputs(&self) -> &[Edge] {
        &self.inputs
    }

    fn outputs(&self) -> &[OutputWire] {
        &self.outputs
    }
}

impl<E: ExtensionField + DeserializeOwned> NodeEdges for NodeCtx<E>
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

impl<N: Number> Node<N> {
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
    pub(crate) fn claims_for_node<'a, 'b>(
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
    #[allow(clippy::type_complexity)]
    pub(crate) fn input_claims<'a, I: Iterator<Item = (NodeId, &'a Self)>>(
        nodes: I,
        claims_by_node: &HashMap<NodeId, Vec<Claim<E>>>,
    ) -> Result<Vec<(NodeId, Vec<&Claim<E>>)>> {
        let mut claims = Vec::new();
        let (mut min_index, mut max_index) = (0, 0);
        let mut total = 0;
        for (node_id, ctx) in nodes {
            let mut node_claims = Vec::new();
            for (i, edge) in ctx.inputs.iter().enumerate() {
                if edge.node.is_none() {
                    let claims_for_node = claims_by_node
                        .get(&node_id)
                        .ok_or(anyhow!("Claim not found for node {}", node_id))?;
                    node_claims.push(&claims_for_node[i]);
                    min_index = min_index.min(edge.index);
                    max_index = max_index.max(edge.index);
                    total += 1;
                }
            }
            if !node_claims.is_empty() {
                claims.push((node_id, node_claims));
            }
        }
        ensure!(
            !claims.is_empty(),
            "No input claims found for the set of nodes provided"
        );
        ensure!(
            min_index == 0 && max_index == total - 1,
            "Not all input claims were found"
        );
        Ok(claims)
    }
}

pub trait OpInfo {
    /// Returns the shapes of the outputs (in the same order)
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape>;

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
        unpadded_input_shapes: Vec<Shape>,
    ) -> Result<LayerOut<T, E>>;
}

/// Helper method employed to call `Evaluate::evaluate` when there are no `unpadded_input_shapes`
/// or when the `E` type cannot be inferred automatically by the compiler
pub fn evaluate_layer<E: ExtensionField, T: Number, O: Evaluate<T>>(
    layer: &O,
    inputs: &[&Tensor<T>],
    unpadded_input_shapes: Option<Vec<Shape>>,
) -> Result<LayerOut<T, E>> {
    layer.evaluate(inputs, unpadded_input_shapes.unwrap_or_default())
}

pub trait ProveInfo<E: ExtensionField>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    /// Compute the proving context for the operation
    fn step_info(&self, id: NodeId, aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)>;
}

/// Output of `QuantizeOp` method over a layer
pub struct QuantizeOutput<Op> {
    /// The actual layer after quantization
    pub(crate) quantized_op: Op,
    /// The scaling factor of the output wires of the operation
    pub(crate) output_scalings: Vec<ScalingFactor>,
    /// The requant layer to be added to the model, if any
    pub(crate) requant_layer: Option<Vec<Requant>>,
}

impl<Op> QuantizeOutput<Op> {
    pub fn new(quantized_op: Op, output_scalings: Vec<ScalingFactor>) -> Self {
        Self {
            quantized_op,
            output_scalings,
            requant_layer: None,
        }
    }
    pub fn with_requant(self, requant: Requant) -> Self {
        assert!(
            self.output_scalings.len() == 1,
            "Number of output scalings must be 1"
        );
        Self::with_requants(self, vec![requant])
    }
    pub fn with_requants(self, requants: Vec<Requant>) -> Self {
        assert!(self.requant_layer.is_none(), "Requant layer already exists");
        assert!(
            self.output_scalings.len() == requants.len(),
            "Number of output scalings and requants must be the same"
        );
        Self {
            quantized_op: self.quantized_op,
            output_scalings: self.output_scalings,
            requant_layer: Some(requants),
        }
    }
    pub fn maybe_requants(self, requant: Option<Vec<Requant>>) -> Self {
        match requant {
            Some(requant) => self.with_requants(requant),
            None => self,
        }
    }
}

pub trait QuantizeOp {
    type QuantizedOp: Sized;

    /// Convert an operation into its quantized version
    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>>;
}

pub trait PadOp {
    // Pad the dimensions of the tensors in node `self`, updating the `ShapeInfo` with the output shapes
    // of the node
    fn pad_node(self, _si: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(self)
    }
}

pub trait ProvableOp<E, PCS>: OpInfo + PadOp + ProveInfo<E>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Ctx: VerifiableCtx<E, PCS>;

    /// Produces a proof of correct execution for this operation.
    fn prove<T: Transcript<E>>(
        &self,
        _node_id: NodeId,
        _ctx: &Self::Ctx,
        _last_claims: Vec<&Claim<E>>,
        _step_data: &StepData<E, E>,
        _prover: &mut Prover<E, T, PCS>,
    ) -> Result<Vec<Claim<E>>> {
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
        _gen: &mut LookupWitnessGen<E, PCS>,
        _ctx: &Context<E, PCS>,
        _step_data: &StepData<Element, E>,
    ) -> Result<()> {
        // Default implementation for nodes that don't employ a lookup table
        Ok(())
    }
}

pub trait VerifiableCtx<E, PCS>: Debug + OpInfo
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof: Sized;

    /// Verify proof for the given operation
    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>>;

    /// Verify the claim about the input of the model. Sometimes
    /// the input needs to be processed in a certain way before being evaluated.
    /// For example, Embeddings use one hot encoding of the input before
    /// running the matmul protocol.
    /// By default, it simply evaluates the input against the input claim.
    fn verify_input_claim(&self, inputs: &[Tensor<E>], claims: &[&Claim<E>]) -> anyhow::Result<()> {
        ensure!(
            inputs.len() == claims.len(),
            "number of input tensors and claims must be the same"
        );
        for (i, (input, claim)) in inputs.iter().zip(claims).enumerate() {
            let computed = input.get_data().into_mle().evaluate(&claim.point);
            ensure!(computed == claim.eval, "input claim {:?} is incorrect", i);
        }
        Ok(())
    }
}

impl<E: ExtensionField> OpInfo for LayerCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        match self {
            LayerCtx::Dense(dense_ctx) => dense_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Convolution(conv_ctx) => conv_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::MatMul(mat_ctx) => mat_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::QKV(qkv_ctx) => qkv_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::MhaQK => unimplemented!("MHA_QK layer not implemented"),
            LayerCtx::ConcatMatMul => unimplemented!("ConcatMatMul layer not implemented"),
            LayerCtx::LayerNorm => unimplemented!("LayerNorm layer not implemented"),
            LayerCtx::Softmax => unimplemented!("Softmax layer not implemented"),
            LayerCtx::Add => unimplemented!("Add layer not implemented"),
            LayerCtx::Logits => unimplemented!("Logits layer not implemented"),
            LayerCtx::Embeddings(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Positional => unimplemented!("Positional layer not implemented"),
            LayerCtx::Reshape => unimplemented!("Reshape layer not implemented"),
            LayerCtx::Activation(activation_ctx) => {
                activation_ctx.output_shapes(input_shapes, padding_mode)
            }
            LayerCtx::Requant(requant_ctx) => requant_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Pooling(pooling_ctx) => pooling_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Flatten => {
                <Flatten as OpInfo>::output_shapes(&Flatten, input_shapes, padding_mode)
            }
            LayerCtx::SchoolBookConvolution(_) | LayerCtx::Table(_) => unreachable!(),
        }
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        match self {
            LayerCtx::Dense(dense_ctx) => dense_ctx.num_outputs(num_inputs),
            LayerCtx::Convolution(conv_ctx) => conv_ctx.num_outputs(num_inputs),
            LayerCtx::MatMul(mat_ctx) => mat_ctx.num_outputs(num_inputs),
            LayerCtx::QKV(qkv_ctx) => qkv_ctx.num_outputs(num_inputs),
            LayerCtx::MhaQK => unimplemented!("MHA_QK layer not implemented"),
            LayerCtx::ConcatMatMul => unimplemented!("ConcatMatMul layer not implemented"),
            LayerCtx::LayerNorm => unimplemented!("LayerNorm layer not implemented"),
            LayerCtx::Softmax => unimplemented!("Softmax layer not implemented"),
            LayerCtx::Add => unimplemented!("Add layer not implemented"),
            LayerCtx::Logits => unimplemented!("Logits layer not implemented"),
            LayerCtx::Embeddings(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::Positional => unimplemented!("Positional layer not implemented"),
            LayerCtx::Reshape => unimplemented!("Reshape layer not implemented"),
            LayerCtx::Activation(activation_ctx) => activation_ctx.num_outputs(num_inputs),
            LayerCtx::Requant(requant_ctx) => requant_ctx.num_outputs(num_inputs),
            LayerCtx::Pooling(pooling_ctx) => pooling_ctx.num_outputs(num_inputs),
            LayerCtx::Flatten => <Flatten as OpInfo>::num_outputs(&Flatten, num_inputs),
            LayerCtx::SchoolBookConvolution(_) | LayerCtx::Table(_) => unreachable!(),
        }
    }

    fn describe(&self) -> String {
        match self {
            LayerCtx::Dense(dense_ctx) => dense_ctx.describe(),
            LayerCtx::Convolution(conv_ctx) => conv_ctx.describe(),
            LayerCtx::MatMul(mat_ctx) => mat_ctx.describe(),
            LayerCtx::QKV(qkv_ctx) => qkv_ctx.describe(),
            LayerCtx::MhaQK => unimplemented!("MHA_QK layer not implemented"),
            LayerCtx::ConcatMatMul => unimplemented!("ConcatMatMul layer not implemented"),
            LayerCtx::LayerNorm => unimplemented!("LayerNorm layer not implemented"),
            LayerCtx::Softmax => unimplemented!("Softmax layer not implemented"),
            LayerCtx::Add => unimplemented!("Add layer not implemented"),
            LayerCtx::Logits => unimplemented!("Logits layer not implemented"),
            LayerCtx::Embeddings(ctx) => ctx.describe(),
            LayerCtx::Positional => unimplemented!("Positional layer not implemented"),
            LayerCtx::Reshape => unimplemented!("Reshape layer not implemented"),
            LayerCtx::Activation(activation_ctx) => activation_ctx.describe(),
            LayerCtx::Requant(requant_ctx) => requant_ctx.describe(),
            LayerCtx::Pooling(pooling_ctx) => pooling_ctx.describe(),
            LayerCtx::Flatten => Flatten.describe(),
            LayerCtx::SchoolBookConvolution(_) | LayerCtx::Table(_) => unreachable!(),
        }
    }

    fn is_provable(&self) -> bool {
        match self {
            LayerCtx::Dense(dense_ctx) => dense_ctx.is_provable(),
            LayerCtx::Convolution(conv_ctx) => conv_ctx.is_provable(),
            LayerCtx::MatMul(mat_ctx) => mat_ctx.is_provable(),
            LayerCtx::QKV(qkv_ctx) => qkv_ctx.is_provable(),
            LayerCtx::MhaQK => unimplemented!("MHA_QK layer not implemented"),
            LayerCtx::ConcatMatMul => unimplemented!("ConcatMatMul layer not implemented"),
            LayerCtx::Activation(activation_ctx) => activation_ctx.is_provable(),
            LayerCtx::LayerNorm => unimplemented!("LayerNorm layer not implemented"),
            LayerCtx::Softmax => unimplemented!("Softmax layer not implemented"),
            LayerCtx::Add => unimplemented!("Add layer not implemented"),
            LayerCtx::Logits => unimplemented!("Logits layer not implemented"),
            LayerCtx::Embeddings(ctx) => ctx.is_provable(),
            LayerCtx::Positional => unimplemented!("Positional layer not implemented"),
            LayerCtx::Reshape => unimplemented!("Reshape layer not implemented"),
            LayerCtx::Requant(requant_ctx) => requant_ctx.is_provable(),
            LayerCtx::Pooling(pooling_ctx) => pooling_ctx.is_provable(),
            LayerCtx::Flatten => Flatten.is_provable(),
            LayerCtx::SchoolBookConvolution(_) | LayerCtx::Table(_) => unreachable!(),
        }
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> VerifiableCtx<E, PCS> for LayerCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    type Proof = LayerProof<E, PCS>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &LayerProof<E, PCS>,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        match (self, proof) {
            (LayerCtx::Dense(dense_ctx), LayerProof::Dense(proof)) => {
                <DenseCtx<E> as VerifiableCtx<E, PCS>>::verify(
                    dense_ctx,
                    proof,
                    last_claims,
                    verifier,
                    shape_step,
                )
            }
            (LayerCtx::Convolution(conv_ctx), LayerProof::Convolution(proof)) => {
                <ConvCtx<E> as VerifiableCtx<E, PCS>>::verify(
                    conv_ctx,
                    proof,
                    last_claims,
                    verifier,
                    shape_step,
                )
            }
            (LayerCtx::MatMul(matmul_ctx), LayerProof::MatMul(proof)) => {
                matmul_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::QKV(qkv_ctx), LayerProof::QKV(proof)) => {
                qkv_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::MhaQK, LayerProof::MhaQK) => {
                unimplemented!("MHA_QK layer not implemented")
            }
            (LayerCtx::Embeddings(ctx), LayerProof::Embeddings(proof)) => {
                ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::Positional, LayerProof::Positional) => {
                unimplemented!("Positional layer not implemented")
            }
            (LayerCtx::Add, LayerProof::Add) => {
                unimplemented!("Add layer not implemented")
            }
            (LayerCtx::Logits, LayerProof::Logits) => {
                unimplemented!("Logits layer not implemented")
            }
            (LayerCtx::Activation(activation_ctx), LayerProof::Activation(proof)) => {
                activation_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::LayerNorm, LayerProof::LayerNorm) => {
                unimplemented!("LayerNorm layer not implemented")
            }
            (LayerCtx::Requant(requant_ctx), LayerProof::Requant(proof)) => {
                requant_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::Pooling(pooling_ctx), LayerProof::Pooling(proof)) => {
                pooling_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::SchoolBookConvolution(_), _)
            | (LayerCtx::Table(_), _)
            | (LayerCtx::Flatten, _) => {
                unreachable!("Trying to verify a non-provable layer")
            }
            _ => bail!(
                "Incompatible layer {} and proof {} found",
                self.describe(),
                proof.variant_name()
            ),
        }
    }

    fn verify_input_claim(&self, inputs: &[Tensor<E>], claims: &[&Claim<E>]) -> anyhow::Result<()> {
        match self {
            LayerCtx::Dense(dense_ctx) => {
                <DenseCtx<E> as VerifiableCtx<E, PCS>>::verify_input_claim(
                    dense_ctx, inputs, claims,
                )
            }
            LayerCtx::Convolution(conv_ctx) => {
                <ConvCtx<E> as VerifiableCtx<E, PCS>>::verify_input_claim(conv_ctx, inputs, claims)
            }
            LayerCtx::MatMul(mat_ctx) => {
                <MatMulCtx<E> as VerifiableCtx<E, PCS>>::verify_input_claim(mat_ctx, inputs, claims)
            }
            LayerCtx::QKV(qkv_ctx) => {
                <QKVCtx<E> as VerifiableCtx<E, PCS>>::verify_input_claim(qkv_ctx, inputs, claims)
            }
            LayerCtx::MhaQK => unimplemented!("MHA_QK layer not implemented"),
            LayerCtx::ConcatMatMul => unimplemented!("ConcatMatMul layer not implemented"),
            LayerCtx::Activation(activation_ctx) => {
                <ActivationCtx as VerifiableCtx<E, PCS>>::verify_input_claim(
                    activation_ctx,
                    inputs,
                    claims,
                )
            }
            LayerCtx::LayerNorm => unimplemented!("LayerNorm layer not implemented"),
            LayerCtx::Softmax => unimplemented!("Softmax layer not implemented"),
            LayerCtx::Add => unimplemented!("Add layer not implemented"),
            LayerCtx::Logits => unimplemented!("Logits layer not implemented"),
            LayerCtx::Embeddings(ctx) => {
                <EmbeddingsCtx<E> as VerifiableCtx<E, PCS>>::verify_input_claim(ctx, inputs, claims)
            }
            LayerCtx::Positional => unimplemented!("Positional layer not implemented"),
            LayerCtx::Reshape => unimplemented!("Reshape layer not implemented"),
            LayerCtx::Requant(requant_ctx) => {
                <RequantCtx as VerifiableCtx<E, PCS>>::verify_input_claim(
                    requant_ctx,
                    inputs,
                    claims,
                )
            }
            LayerCtx::Pooling(pooling_ctx) => {
                <PoolingCtx as VerifiableCtx<E, PCS>>::verify_input_claim(
                    pooling_ctx,
                    inputs,
                    claims,
                )
            }
            LayerCtx::Flatten => unimplemented!("Flatten layer not implemented"),
            LayerCtx::SchoolBookConvolution(_) | LayerCtx::Table(_) => unreachable!(),
        }
    }
}
