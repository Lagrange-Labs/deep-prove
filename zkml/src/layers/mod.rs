pub mod activation;
pub mod convolution;
pub mod dense;
pub mod flatten;
pub mod hadamard;
pub mod matvec;
pub mod matrix_mul;
pub mod pooling;
pub mod provable;
pub mod requant;

use std::fmt::Debug;

use anyhow::{Result, bail};
use ff_ext::ExtensionField;
use flatten::Flatten;
use pooling::{PoolingCtx, PoolingProof};
use provable::{
    Evaluate, LayerOut, Node, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp, QuantizeOutput,
};
use requant::RequantCtx;
use transcript::Transcript;

use crate::{
    Element, ScalingStrategy,
    commit::precommit::PolyID,
    iop::context::{ContextAux, ShapeStep, TableCtx},
    layers::{
        activation::{Activation, ActivationProof},
        convolution::Convolution,
        dense::Dense,
        pooling::Pooling,
        requant::{Requant, RequantProof},
    },
    lookup::context::LookupWitnessGen,
    model::StepData,
    padding::{PaddingMode, ShapeInfo},
    quantization::ScalingFactor,
    tensor::{Number, Tensor},
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
    Flatten,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum LayerProof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    Dense(DenseProof<E>),
    MatMul(MatMulProof<E>),
    Convolution(ConvProof<E>),
    Activation(ActivationProof<E>),
    Requant(RequantProof<E>),
    Pooling(PoolingProof<E>),
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
        let unpadded_output = self.output_shapes(&last_step.unpadded_output_shape, PaddingMode::NoPadding);
        let padded_output = self.output_shapes(&last_step.padded_output_shape, PaddingMode::Padding);
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
        id: PolyID,
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
    fn step_info(&self, id: PolyID, aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        match self {
            Layer::Dense(dense) => dense.step_info(id, aux),
            Layer::MatMul(mat) => mat.step_info(id, aux),
            Layer::Convolution(conv) => conv.step_info(id, aux),
            Layer::SchoolBookConvolution(conv) => conv.step_info(id, aux),
            Layer::Activation(activation) => activation.step_info(id, aux),
            Layer::Requant(requant) => requant.step_info(id, aux),
            Layer::Pooling(pooling) => pooling.step_info(id, aux),
            Layer::Flatten(reshape) => reshape.step_info(id, aux),
        }
    }

    fn commit_info(&self, id: provable::NodeId) -> Vec<Option<(PolyID, Vec<E>)>> {
        match self {
            Layer::Dense(dense) => dense.commit_info(id),
            Layer::Convolution(convolution) => convolution.commit_info(id),
            Layer::MatMul(mat) => mat.commit_info(id),
            Layer::SchoolBookConvolution(school_book_conv) => school_book_conv.commit_info(id),
            Layer::Activation(activation) => activation.commit_info(id),
            Layer::Requant(requant) => requant.commit_info(id),
            Layer::Pooling(pooling) => pooling.commit_info(id),
            Layer::Flatten(reshape) => reshape.commit_info(id),
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

impl<E: ExtensionField> ProvableOp<E> for Layer<Element>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    type Ctx = LayerCtx<E>;

    fn prove<T: Transcript<E>>(
        &self,
        node_id: provable::NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&crate::Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut crate::Prover<E, T>,
    ) -> Result<Vec<crate::Claim<E>>> {
        match self {
            Layer::Dense(dense) => {
                if let LayerCtx::Dense(info) = ctx {
                    dense.prove(node_id, info, last_claims, step_data, prover)
                } else {
                    bail!("No dense ctx found when proving dense layer")
                }
            }
            Layer::Convolution(convolution) => {
                if let LayerCtx::Convolution(info) = ctx {
                    convolution.prove(node_id, info, last_claims, step_data, prover)
                } else {
                    bail!("No convolution ctx found when proving convolution layer")
                }
            }
            Layer::MatMul(m) => {
                if let LayerCtx::MatMul(info) = ctx {
                    m.prove(node_id, info, last_claims, step_data, prover)
                } else {
                    bail!(
                        "No mat mul ctx found for when proving MatMul layer".to_string(),
                    )
                }
            }
            Layer::SchoolBookConvolution(_) => {
                unreachable!("prove cannot be called for school book convolution")
            }
            Layer::Activation(activation) => {
                if let LayerCtx::Activation(info) = ctx {
                    activation.prove(node_id, info, last_claims, step_data, prover)
                } else {
                    bail!("No activation ctx found when proving activation layer")
                }
            }
            Layer::Requant(requant) => {
                if let LayerCtx::Requant(info) = ctx {
                    requant.prove(node_id, info, last_claims, step_data, prover)
                } else {
                    bail!("No requant ctx found when proving requant layer")
                }
            }
            Layer::Pooling(pooling) => {
                if let LayerCtx::Pooling(info) = ctx {
                    pooling.prove(node_id, info, last_claims, step_data, prover)
                } else {
                    bail!("No pooling ctx found when proving pooling layer")
                }
            }
            Layer::Flatten(_) => unreachable!("prove cannot be called for reshape"),
        }
    }

    fn gen_lookup_witness(
        &self,
        id: provable::NodeId,
        gen: &mut LookupWitnessGen<E>,
        step_data: &StepData<Element, E>,
    ) -> Result<()> {
        match self {
            Layer::Dense(dense) => dense.gen_lookup_witness(id, gen, step_data),
            Layer::Convolution(convolution) => convolution.gen_lookup_witness(id, gen, step_data),
            Layer::MatMul(m) => m.gen_lookup_witness(id, gen, step_data),
            Layer::SchoolBookConvolution(school_book_conv) => {
                // check that the layer is not provable, so we don't need to call the method
                assert!(!school_book_conv.is_provable());
                Ok(())
            }
            Layer::Activation(activation) => activation.gen_lookup_witness(id, gen, step_data),
            Layer::Requant(requant) => requant.gen_lookup_witness(id, gen, step_data),
            Layer::Pooling(pooling) => pooling.gen_lookup_witness(id, gen, step_data),
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
                QuantizeOutput {
                    quanzited_op: Layer::Dense(output.quanzited_op),
                    output_scalings: output.output_scalings,
                    requant_layer: output.requant_layer,
                }
            }
            Layer::Convolution(convolution) => {
                let output = convolution.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput {
                    quanzited_op: Layer::Convolution(output.quanzited_op),
                    output_scalings: output.output_scalings,
                    requant_layer: output.requant_layer,
                }
            }
            Layer::MatMul(mat) => {
                let output = mat.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput {
                    quanzited_op: Layer::MatMul(output.quanzited_op),
                    output_scalings: output.output_scalings,
                    requant_layer: output.requant_layer,
                }
            }
            Layer::SchoolBookConvolution(school_book_conv) => {
                let output = school_book_conv.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput {
                    quanzited_op: Layer::SchoolBookConvolution(output.quanzited_op),
                    output_scalings: output.output_scalings,
                    requant_layer: output.requant_layer,
                }
            }
            Layer::Activation(activation) => QuantizeOutput {
                quanzited_op: Layer::Activation(activation),
                output_scalings: input_scaling.to_vec(),
                requant_layer: None,
            },
            Layer::Requant(requant) => QuantizeOutput {
                quanzited_op: Layer::Requant(requant),
                output_scalings: input_scaling.to_vec(),
                requant_layer: None,
            },
            Layer::Pooling(pooling) => QuantizeOutput {
                quanzited_op: Layer::Pooling(pooling),
                output_scalings: input_scaling.to_vec(),
                requant_layer: None,
            },
            Layer::Flatten(flatten) => QuantizeOutput {
                quanzited_op: Layer::Flatten(flatten),
                output_scalings: input_scaling.to_vec(),
                requant_layer: None,
            },
        })
    }
}

impl<E: ExtensionField> LayerProof<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    pub fn variant_name(&self) -> String {
        match self {
            Self::Dense(_) => "Dense".to_string(),
            Self::MatMul(_) => "Matmul".to_string(),
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
            LayerProof::Convolution(..) => None,
            LayerProof::Dummy => None,
            LayerProof::Activation(ActivationProof { lookup, .. })
            | LayerProof::Requant(RequantProof { lookup, .. })
            | LayerProof::Pooling(PoolingProof { lookup, .. }) => Some(lookup.fractional_outputs()),
        }
    }
}
impl<T: Number> std::fmt::Display for Layer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.describe())
    }
}
