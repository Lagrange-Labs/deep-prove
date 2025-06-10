pub mod activation;
pub mod add;
pub mod concat_matmul;
pub mod convolution;
pub mod dense;
pub mod flatten;
pub mod hadamard;
pub mod matrix_mul;
pub mod matvec;
pub mod mul;
pub mod permute;
pub mod pooling;
pub mod provable;
pub mod requant;
pub mod reshape;
pub mod transformer;

use std::fmt::Debug;

use anyhow::{Result, bail};
use ff_ext::ExtensionField;
use flatten::Flatten;
use mpcs::PolynomialCommitmentScheme;
use pooling::{PoolingCtx, PoolingProof};
use provable::{
    Evaluate, LayerOut, Node, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp,
    QuantizeOutput,
};
use requant::RequantCtx;
use transcript::Transcript;

use crate::{
    iop::context::{ContextAux, ShapeStep, TableCtx}, layers::{
        activation::{Activation, ActivationProof},
        convolution::Convolution,
        dense::Dense,
        pooling::Pooling,
        requant::{Requant, RequantProof},
        transformer::qkv::{QKVCtx, QKVProof, QKV},
    }, lookup::context::LookupWitnessGen, model::StepData, padding::{PaddingMode, ShapeInfo}, quantization::ScalingFactor, tensor::{Number, Tensor}, Context, Element, ScalingStrategy
};
use activation::ActivationCtx;
use convolution::{ConvCtx, ConvProof, SchoolBookConv, SchoolBookConvCtx};
use dense::{DenseCtx, DenseProof};
use matrix_mul::{MatMul, MatMulCtx, MatMulProof};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[derive(Clone, Debug)]
pub enum Layer<T> {
    Dense(Dense<T>),
    MatMul(MatMul<T>),
    // TODO: replace this with a Tensor based implementation
    Convolution(Convolution<T>),
    // Traditional convolution is used for debug purposes. That is because the actual convolution
    // we use relies on the FFT algorithm. This convolution does not have a snark implementation.
    SchoolBookConvolution(SchoolBookConv<T>),
    Activation(Activation),
    // this is the output quant info. Since we always do a requant layer after each dense,
    // then we assume the inputs requant info are default()
    Requant(Requant),
    Pooling(Pooling),
    // TODO: so far it's only flattening the input tensor, e.g. new_shape = vec![shape.iter().product()]
    Flatten(Flatten),
    QKV(QKV<T>),
}

/// Describes a steps wrt the polynomial to be proven/looked at. Verifier needs to know
/// the sequence of steps and the type of each step from the setup phase so it can make sure the prover is not
/// cheating on this.
/// NOTE: The context automatically appends a requant step after each dense layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub enum LayerCtx<E>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    Dense(DenseCtx<E>),
    MatMul(MatMulCtx<E>),
    Convolution(ConvCtx<E>),
    SchoolBookConvolution(SchoolBookConvCtx),
    Activation(ActivationCtx),
    Requant(RequantCtx),
    Pooling(PoolingCtx),
    Table(TableCtx<E>),
    QKV(QKVCtx<E>),
    Flatten,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum LayerProof<E, PCS>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    Dense(DenseProof<E>),
    MatMul(MatMulProof<E>),
    Convolution(ConvProof<E>),
    Activation(ActivationProof<E, PCS>),
    Requant(RequantProof<E, PCS>),
    Pooling(PoolingProof<E, PCS>),
    QKV(QKVProof<E>),
    Dummy, // To be used for non-provable layers
}

impl<E> LayerCtx<E>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    pub fn variant_name(&self) -> String {
        match self {
            Self::Dense(_) => "Dense".to_string(),
            Self::MatMul(_) => "Matrix Multiplication".to_string(),
            Self::QKV(_) => "QKV".to_string(),
            Self::SchoolBookConvolution(_) => "Traditional Convolution".to_string(),
            Self::Convolution(_) => "Convolution".to_string(),
            Self::Activation(_) => "Activation".to_string(),
            Self::Requant(_) => "Requant".to_string(),
            Self::Pooling(_) => "Pooling".to_string(),
            Self::Table(..) => "Table".to_string(),
            Self::Flatten => "Reshape".to_string(),
        }
    }

    pub fn has_proof(&self) -> bool {
        match self {
            Self::Flatten | Self::Table(_) | Self::SchoolBookConvolution(_) => false,
            _ => true,
        }
    }

    pub fn next_shape_step(&self, last_step: &ShapeStep) -> ShapeStep {
        let unpadded_output =
            self.output_shapes(&last_step.unpadded_output_shape, PaddingMode::NoPadding);
        let padded_output =
            self.output_shapes(&last_step.padded_output_shape, PaddingMode::Padding);
        ShapeStep::next_step(last_step, unpadded_output, padded_output)
    }
    pub fn shape_step(
        &self,
        unpadded_input: &[Vec<usize>],
        padded_input: &[Vec<usize>],
    ) -> ShapeStep {
        let unpadded_output = self.output_shapes(unpadded_input, PaddingMode::NoPadding);
        let padded_output = self.output_shapes(padded_input, PaddingMode::Padding);
        ShapeStep::new(
            unpadded_input.to_vec(),
            padded_input.to_vec(),
            unpadded_output,
            padded_output,
        )
    }
}

impl<N> Node<N>
where
    N: Number,
{
    pub fn describe(&self) -> String {
        self.operation.describe()
    }

    pub fn is_provable(&self) -> bool {
        self.operation.is_provable()
    }

    pub(crate) fn run<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> Result<LayerOut<N, E>>
    where
        N: Number,
        Layer<N>: Evaluate<N>,
    {
        self.operation.evaluate(inputs, unpadded_input_shapes)
    }

    pub(crate) fn step_info<E>(
        &self,
        id: NodeId,
        aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)>
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        Layer<N>: ProveInfo<E>,
    {
        self.operation.step_info(id, aux)
    }
}

impl Node<Element> {
    pub(crate) fn pad_node(self, si: &mut ShapeInfo) -> Result<Self> {
        Ok(Self {
            inputs: self.inputs,
            outputs: self.outputs,
            operation: self.operation.pad_node(si)?,
        })
    }
}

impl<N: Number> OpInfo for Layer<N> {
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
        match self {
            Layer::Dense(dense) => dense.output_shapes(input_shapes, padding_mode),
            Layer::Convolution(convolution) => {
                convolution.output_shapes(input_shapes, padding_mode)
            }
            Layer::MatMul(mat) => mat.output_shapes(input_shapes, padding_mode),
            Layer::QKV(qkv) => qkv.output_shapes(input_shapes, padding_mode),
            Layer::SchoolBookConvolution(convolution) => {
                convolution.output_shapes(input_shapes, padding_mode)
            }
            Layer::Activation(activation) => activation.output_shapes(input_shapes, padding_mode),
            Layer::Requant(requant) => requant.output_shapes(input_shapes, padding_mode),
            Layer::Pooling(pooling) => pooling.output_shapes(input_shapes, padding_mode),
            Layer::Flatten(reshape) => reshape.output_shapes(input_shapes, padding_mode),
        }
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        match self {
            Layer::Dense(dense) => dense.num_outputs(num_inputs),
            Layer::Convolution(convolution) => convolution.num_outputs(num_inputs),
            Layer::MatMul(mat) => mat.num_outputs(num_inputs),
            Layer::QKV(qkv) => qkv.num_outputs(num_inputs),
            Layer::SchoolBookConvolution(convolution) => convolution.num_outputs(num_inputs),
            Layer::Activation(activation) => activation.num_outputs(num_inputs),
            Layer::Requant(requant) => requant.num_outputs(num_inputs),
            Layer::Pooling(pooling) => pooling.num_outputs(num_inputs),
            Layer::Flatten(reshape) => reshape.num_outputs(num_inputs),
        }
    }

    fn describe(&self) -> String {
        match self {
            Layer::Dense(dense) => dense.describe(),
            Layer::Convolution(convolution) => convolution.describe(),
            Layer::MatMul(mat) => mat.describe(),
            Layer::QKV(qkv) => qkv.describe(),
            Layer::SchoolBookConvolution(convolution) => convolution.describe(),
            Layer::Activation(activation) => activation.describe(),
            Layer::Requant(requant) => requant.describe(),
            Layer::Pooling(pooling) => pooling.describe(),
            Layer::Flatten(reshape) => reshape.describe(),
        }
    }

    fn is_provable(&self) -> bool {
        match self {
            Layer::Dense(dense) => dense.is_provable(),
            Layer::Convolution(convolution) => convolution.is_provable(),
            Layer::MatMul(mat) => mat.is_provable(),
            Layer::QKV(qkv) => qkv.is_provable(),
            Layer::SchoolBookConvolution(school_book_conv) => school_book_conv.is_provable(),
            Layer::Activation(activation) => activation.is_provable(),
            Layer::Requant(requant) => requant.is_provable(),
            Layer::Pooling(pooling) => pooling.is_provable(),
            Layer::Flatten(reshape) => reshape.is_provable(),
        }
    }
}

impl Evaluate<f32> for Layer<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> Result<LayerOut<f32, E>> {
        match self {
            Layer::Dense(dense) => dense.evaluate(inputs, unpadded_input_shapes),
            Layer::Convolution(convolution) => convolution.evaluate(inputs, unpadded_input_shapes),
            Layer::MatMul(mat) => mat.evaluate(inputs, unpadded_input_shapes),
            Layer::QKV(qkv) => qkv.evaluate(inputs, unpadded_input_shapes),
            Layer::SchoolBookConvolution(school_book_conv) => {
                school_book_conv.evaluate(inputs, unpadded_input_shapes)
            }
            Layer::Activation(activation) => activation.evaluate(inputs, unpadded_input_shapes),
            Layer::Requant(_) => unreachable!("Requant layer found when evaluating over float"),
            Layer::Pooling(pooling) => pooling.evaluate(inputs, unpadded_input_shapes),
            Layer::Flatten(reshape) => reshape.evaluate(inputs, unpadded_input_shapes),
        }
    }
}

impl Evaluate<Element> for Layer<Element> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> Result<LayerOut<Element, E>> {
        match self {
            Layer::Dense(dense) => dense.evaluate(inputs, unpadded_input_shapes),
            Layer::Convolution(convolution) => convolution.evaluate(inputs, unpadded_input_shapes),
            Layer::MatMul(mat) => mat.evaluate(inputs, unpadded_input_shapes),
            Layer::QKV(qkv) => qkv.evaluate(inputs, unpadded_input_shapes),
            Layer::SchoolBookConvolution(school_book_conv) => {
                school_book_conv.evaluate(inputs, unpadded_input_shapes)
            }
            Layer::Activation(activation) => activation.evaluate(inputs, unpadded_input_shapes),
            Layer::Requant(requant) => requant.evaluate(inputs, unpadded_input_shapes),
            Layer::Pooling(pooling) => pooling.evaluate(inputs, unpadded_input_shapes),
            Layer::Flatten(reshape) => reshape.evaluate(inputs, unpadded_input_shapes),
        }
    }
}

impl<E: ExtensionField> ProveInfo<E> for Layer<Element>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(&self, id: NodeId, aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        match self {
            Layer::Dense(dense) => dense.step_info(id, aux),
            Layer::QKV(qkv) => qkv.step_info(id, aux),
            Layer::MatMul(mat) => mat.step_info(id, aux),
            Layer::Convolution(conv) => conv.step_info(id, aux),
            Layer::SchoolBookConvolution(conv) => conv.step_info(id, aux),
            Layer::Activation(activation) => activation.step_info(id, aux),
            Layer::Requant(requant) => requant.step_info(id, aux),
            Layer::Pooling(pooling) => pooling.step_info(id, aux),
            Layer::Flatten(reshape) => reshape.step_info(id, aux),
        }
    }
}

impl PadOp for Layer<Element> {
    fn pad_node(self, si: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(match self {
            Layer::Dense(dense) => Layer::Dense(dense.pad_node(si)?),
            Layer::Convolution(convolution) => Layer::Convolution(convolution.pad_node(si)?),
            Layer::QKV(qkv) => Layer::QKV(qkv.pad_node(si)?),
            Layer::MatMul(mat) => Layer::MatMul(mat.pad_node(si)?),
            Layer::SchoolBookConvolution(school_book_conv) => {
                Layer::SchoolBookConvolution(school_book_conv.pad_node(si)?)
            }
            Layer::Activation(activation) => Layer::Activation(activation.pad_node(si)?),
            Layer::Requant(requant) => Layer::Requant(requant.pad_node(si)?),
            Layer::Pooling(pooling) => Layer::Pooling(pooling.pad_node(si)?),
            Layer::Flatten(flatten) => Layer::Flatten(flatten.pad_node(si)?),
        })
    }
}

impl<E, PCS> ProvableOp<E, PCS> for Layer<Element>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Ctx = LayerCtx<E>;

    fn prove<T: Transcript<E>>(
        &self,
        node_id: provable::NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&crate::Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut crate::Prover<E, T, PCS>,
    ) -> Result<Vec<crate::Claim<E>>> {
        match (self, ctx) {
            (Layer::Dense(dense), LayerCtx::Dense(info)) => {
                dense.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::Convolution(convolution), LayerCtx::Convolution(info)) => {
                convolution.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::MatMul(m), LayerCtx::MatMul(info)) => {
                m.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::QKV(qkv), LayerCtx::QKV(info)) => {
                qkv.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::SchoolBookConvolution(_), LayerCtx::SchoolBookConvolution(_)) => {
                unreachable!("prove cannot be called for school book convolution")
            }
            (Layer::Activation(activation), LayerCtx::Activation(info)) => {
                activation.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::Requant(requant), LayerCtx::Requant(info)) => {
                requant.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::Pooling(pooling), LayerCtx::Pooling(info)) => {
                pooling.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::Flatten(_), LayerCtx::Flatten) => {
                unreachable!("prove cannot be called for reshape")
            }
            _ => bail!(
                "Incompatible layer {} and ctx {} found for node id {}",
                self.describe(),
                ctx.variant_name(),
                node_id
            ),
        }
    }

    fn gen_lookup_witness(
        &self,
        id: provable::NodeId,
        gen: &mut LookupWitnessGen<E, PCS>,
        ctx: &Context<E, PCS>,
        step_data: &StepData<Element, E>,
    ) -> Result<()> {
        match self {
            Layer::Dense(dense) => dense.gen_lookup_witness(id, gen, ctx, step_data),
            Layer::Convolution(convolution) => {
                convolution.gen_lookup_witness(id, gen, ctx, step_data)
            }
            Layer::MatMul(m) => m.gen_lookup_witness(id, gen, ctx, step_data),
            Layer::QKV(qkv) => qkv.gen_lookup_witness(id, gen, ctx, step_data),
            Layer::SchoolBookConvolution(school_book_conv) => {
                // check that the layer is not provable, so we don't need to call the method
                assert!(!school_book_conv.is_provable());
                Ok(())
            }
            Layer::Activation(activation) => activation.gen_lookup_witness(id, gen, ctx, step_data),
            Layer::Requant(requant) => requant.gen_lookup_witness(id, gen, ctx, step_data),
            Layer::Pooling(pooling) => pooling.gen_lookup_witness(id, gen, ctx, step_data),
            Layer::Flatten(reshape) => {
                // check that the layer is not provable, so we don't need to call the method
                assert!(!reshape.is_provable());
                Ok(())
            }
        }
    }
}

impl QuantizeOp for Layer<f32> {
    type QuantizedOp = Layer<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: provable::NodeId,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        Ok(match self {
            Layer::Dense(dense) => {
                let output = dense.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(Layer::Dense(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)
            }
            Layer::Convolution(convolution) => {
                let output = convolution.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(
                    Layer::Convolution(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)
            }
            Layer::MatMul(mat) => {
                let output = mat.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(Layer::MatMul(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)
            }
            Layer::QKV(qkv) => {
                let output = qkv.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(Layer::QKV(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)
            }
            Layer::SchoolBookConvolution(school_book_conv) => {
                let output = school_book_conv.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(
                    Layer::SchoolBookConvolution(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)
            }
            Layer::Activation(activation) => {
                QuantizeOutput::new(Layer::Activation(activation), input_scaling.to_vec())
            }
            Layer::Requant(requant) => {
                QuantizeOutput::new(Layer::Requant(requant), input_scaling.to_vec())
            }
            Layer::Pooling(pooling) => {
                QuantizeOutput::new(Layer::Pooling(pooling), input_scaling.to_vec())
            }
            Layer::Flatten(flatten) => {
                QuantizeOutput::new(Layer::Flatten(flatten), input_scaling.to_vec())
            }
        })
    }
}

impl<E, PCS> LayerProof<E, PCS>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub fn variant_name(&self) -> String {
        match self {
            Self::Dense(_) => "Dense".to_string(),
            Self::MatMul(_) => "Matmul".to_string(),
            Self::QKV(_) => "QKV".to_string(),
            Self::Convolution(_) => "Convolution".to_string(),
            Self::Activation(_) => "Activation".to_string(),
            Self::Requant(_) => "Requant".to_string(),
            Self::Pooling(_) => "Pooling".to_string(),
            Self::Dummy => "Dummy".to_string(),
        }
    }

    pub fn get_lookup_data(&self) -> Option<(Vec<E>, Vec<E>)> {
        match self {
            LayerProof::Dense(..) => None,
            LayerProof::MatMul(..) => None,
            LayerProof::QKV(..) => None,
            LayerProof::Convolution(..) => None,
            LayerProof::Dummy => None,
            LayerProof::Activation(ActivationProof { lookup, .. })
            | LayerProof::Pooling(PoolingProof { lookup, .. }) => Some(lookup.fractional_outputs()),
            LayerProof::Requant(RequantProof {
                clamping_lookup,
                shifted_lookup,
                ..
            }) => {
                let (clamp_nums, clamp_denoms) = clamping_lookup.fractional_outputs();
                let (shift_nums, shift_denoms) = shifted_lookup.fractional_outputs();
                Some((
                    [clamp_nums, shift_nums].concat(),
                    [clamp_denoms, shift_denoms].concat(),
                ))
            }
        }
    }
}
impl<T: Number> std::fmt::Display for Layer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.describe())
    }
}
