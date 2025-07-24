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
    Context, Element, ScalingStrategy,
    iop::context::{ContextAux, ShapeStep, TableCtx},
    layers::{
        activation::{Activation, ActivationProof},
        add::{Add, AddCtx, AddProof},
        concat_matmul::{ConcatMatMul, ConcatMatMulCtx, ConcatMatMulProof},
        convolution::Convolution,
        dense::Dense,
        pooling::Pooling,
        requant::{Requant, RequantProof},
        reshape::{Reshape, ReshapeCtx},
        transformer::{
            embeddings::{Embeddings, EmbeddingsCtx, EmbeddingsProof},
            layernorm::{LayerNorm, LayerNormCtx, LayerNormProof},
            logits::{Logits, LogitsCtx, LogitsProof},
            mha::{Mha, MhaCtx, MhaProof},
            positional::{Positional, PositionalCtx, PositionalProof},
            qkv::{QKV, QKVCtx, QKVProof},
            softmax::{Softmax, SoftmaxCtx, SoftmaxProof},
        },
    },
    lookup::context::LookupWitnessGen,
    model::StepData,
    padding::{PaddingMode, ShapeInfo},
    quantization::ScalingFactor,
    tensor::{Number, Shape, Tensor},
};
use activation::ActivationCtx;
use convolution::{ConvCtx, ConvProof, SchoolBookConv, SchoolBookConvCtx};
use dense::{DenseCtx, DenseProof};
use matrix_mul::{MatMul, MatMulCtx, MatMulProof};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Layer<T> {
    Dense(Dense<T>),
    MatMul(MatMul<T>),
    // TODO: replace this with a Tensor based implementation
    Convolution(Convolution<T>),
    // Traditional convolution is used for debug purposes. That is because the actual convolution
    // we use relies on the FFT algorithm. This convolution does not have a snark implementation.
    SchoolBookConvolution(SchoolBookConv<T>),
    Activation(Activation<T>),
    // this is the output quant info. Since we always do a requant layer after each dense,
    // then we assume the inputs requant info are default()
    Requant(Requant),
    Pooling(Pooling),
    // TODO: so far it's only flattening the input tensor, e.g. new_shape = vec![shape.iter().product()]
    Flatten(Flatten),
    QKV(QKV<T>),
    Mha(Mha<T>),
    ConcatMatMul(ConcatMatMul),
    LayerNorm(LayerNorm<T>),
    Softmax(Softmax<T>),
    Add(Add<T>),
    Reshape(Reshape),
    Embeddings(Embeddings<T>),
    Positional(Positional<T>),
    Logits(Logits),
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
    Mha(MhaCtx<E>),
    ConcatMatMul(ConcatMatMulCtx<E>),
    LayerNorm(LayerNormCtx),
    Flatten,
    Add(AddCtx),
    Softmax(SoftmaxCtx),
    Reshape(ReshapeCtx),
    Embeddings(EmbeddingsCtx<E>),
    Positional(PositionalCtx),
    Logits(LogitsCtx<E>),
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub enum LayerProof<E, PCS>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    Dense(DenseProof<E>),
    MatMul(MatMulProof<E>),
    Convolution(Box<ConvProof<E>>),
    Activation(ActivationProof<E, PCS>),
    Requant(RequantProof<E, PCS>),
    Pooling(PoolingProof<E, PCS>),
    QKV(QKVProof<E>),
    Mha(MhaProof<E, PCS>),
    ConcatMatMul(ConcatMatMulProof<E>),
    Add(AddProof<E>),
    LayerNorm(LayerNormProof<E, PCS>),
    Softmax(SoftmaxProof<E, PCS>),
    Embeddings(EmbeddingsProof<E>),
    Logits(LogitsProof<E, PCS>),
    Positional(PositionalProof<E>),
    Dummy, // To be used for non-provable layers
}

impl<T> Layer<T> {
    /// Convert a layer to a string only containing its kind
    pub fn as_kind_str(&self) -> &'static str {
        match self {
            Layer::Dense(_) => "dense",
            Layer::Convolution(_) => "convolution",
            Layer::SchoolBookConvolution(_) => "school-book-convolution",
            Layer::Activation(_) => "activation",
            Layer::Requant(_) => "requant",
            Layer::Pooling(_) => "pooling",
            Layer::Flatten(_) => "flatten",
            Layer::QKV(_) => "qkv",
            Layer::Mha(_) => "mha-qk",
            Layer::MatMul(_) => "mat-mul",
            Layer::ConcatMatMul(_) => "concat-mat-mul",
            Layer::LayerNorm(_) => "layer-norm",
            Layer::Softmax(_) => "softmax",
            Layer::Add(_) => "add",
            Layer::Reshape(_) => "reshape",
            Layer::Embeddings(_) => "embeddings",
            Layer::Positional(_) => "positional",
            Layer::Logits(_) => "logits",
        }
    }
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
            Self::Mha(_) => "MHA".to_string(),
            Self::ConcatMatMul(_) => "ConcatMatMul".to_string(),
            Self::LayerNorm(_) => "LayerNorm".to_string(),
            Self::Softmax(_) => "Softmax".to_string(),
            Self::Add(_) => "Add".to_string(),
            Self::Logits(_) => "Logits".to_string(),
            Self::Reshape(_) => "Reshape".to_string(),
            Self::Positional(_) => "Positional".to_string(),
            Self::Embeddings(_) => "Embeddings".to_string(),
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
        !matches!(
            self,
            Self::Flatten | Self::Table(_) | Self::SchoolBookConvolution(_)
        )
    }

    pub fn next_shape_step(&self, last_step: &ShapeStep) -> ShapeStep {
        let unpadded_output =
            self.output_shapes(&last_step.unpadded_output_shape, PaddingMode::NoPadding);
        let padded_output =
            self.output_shapes(&last_step.padded_output_shape, PaddingMode::Padding);
        ShapeStep::next_step(last_step, unpadded_output, padded_output)
    }
    pub fn shape_step(&self, unpadded_input: &[Shape], padded_input: &[Shape]) -> ShapeStep {
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
        unpadded_input_shapes: Vec<Shape>,
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
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        match self {
            Layer::Dense(dense) => dense.output_shapes(input_shapes, padding_mode),
            Layer::Convolution(convolution) => {
                convolution.output_shapes(input_shapes, padding_mode)
            }
            Layer::MatMul(mat) => mat.output_shapes(input_shapes, padding_mode),
            Layer::Mha(mha) => mha.output_shapes(input_shapes, padding_mode),
            Layer::ConcatMatMul(concat_matmul) => {
                concat_matmul.output_shapes(input_shapes, padding_mode)
            }
            Layer::QKV(qkv) => qkv.output_shapes(input_shapes, padding_mode),
            Layer::Add(add) => add.output_shapes(input_shapes, padding_mode),
            Layer::Logits(logits) => logits.output_shapes(input_shapes, padding_mode),
            Layer::Positional(positional) => positional.output_shapes(input_shapes, padding_mode),
            Layer::LayerNorm(layernorm) => layernorm.output_shapes(input_shapes, padding_mode),
            Layer::Softmax(softmax) => softmax.output_shapes(input_shapes, padding_mode),
            Layer::Embeddings(embeddings) => embeddings.output_shapes(input_shapes, padding_mode),
            Layer::Reshape(reshape) => reshape.output_shapes(input_shapes, padding_mode),
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
            Layer::Mha(mha) => mha.num_outputs(num_inputs),
            Layer::ConcatMatMul(concat_matmul) => concat_matmul.num_outputs(num_inputs),
            Layer::LayerNorm(layernorm) => layernorm.num_outputs(num_inputs),
            Layer::Softmax(softmax) => softmax.num_outputs(num_inputs),
            Layer::Add(add) => add.num_outputs(num_inputs),
            Layer::Logits(logits) => logits.num_outputs(num_inputs),
            Layer::Reshape(reshape) => reshape.num_outputs(num_inputs),
            Layer::Positional(positional) => positional.num_outputs(num_inputs),
            Layer::Embeddings(embeddings) => embeddings.num_outputs(num_inputs),
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
            Layer::Mha(mha) => mha.describe(),
            Layer::ConcatMatMul(concat_matmul) => concat_matmul.describe(),
            Layer::LayerNorm(layernorm) => layernorm.describe(),
            Layer::Softmax(softmax) => softmax.describe(),
            Layer::Add(add) => add.describe(),
            Layer::Logits(logits) => logits.describe(),
            Layer::Positional(positional) => positional.describe(),
            Layer::Reshape(reshape) => reshape.describe(),
            Layer::Embeddings(embeddings) => embeddings.describe(),
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
            Layer::Mha(mha) => mha.is_provable(),
            Layer::ConcatMatMul(concat_matmul) => concat_matmul.is_provable(),
            Layer::LayerNorm(layernorm) => layernorm.is_provable(),
            Layer::Softmax(softmax) => softmax.is_provable(),
            Layer::Positional(positional) => positional.is_provable(),
            Layer::Add(add) => add.is_provable(),
            Layer::Logits(logits) => logits.is_provable(),
            Layer::Reshape(reshape) => reshape.is_provable(),
            Layer::Embeddings(embeddings) => embeddings.is_provable(),
            Layer::SchoolBookConvolution(school_book_conv) => !school_book_conv.is_provable(),
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
        unpadded_input_shapes: Vec<Shape>,
    ) -> Result<LayerOut<f32, E>> {
        match self {
            Layer::Dense(dense) => dense.evaluate(inputs, unpadded_input_shapes),
            Layer::Convolution(convolution) => convolution.evaluate(inputs, unpadded_input_shapes),
            Layer::MatMul(mat) => mat.evaluate(inputs, unpadded_input_shapes),
            Layer::QKV(qkv) => qkv.evaluate(inputs, unpadded_input_shapes),
            Layer::Mha(mha) => mha.evaluate(inputs, unpadded_input_shapes),
            Layer::ConcatMatMul(concat_matmul) => {
                concat_matmul.evaluate(inputs, unpadded_input_shapes)
            }
            Layer::LayerNorm(layernorm) => layernorm.evaluate(inputs, unpadded_input_shapes),
            Layer::Softmax(softmax) => softmax.evaluate(inputs, unpadded_input_shapes),
            Layer::Add(add) => add.evaluate(inputs, unpadded_input_shapes),
            Layer::Logits(logits) => logits.evaluate(inputs, unpadded_input_shapes),
            Layer::Positional(positional) => positional.evaluate(inputs, unpadded_input_shapes),
            Layer::Reshape(reshape) => reshape.evaluate(inputs, unpadded_input_shapes),
            Layer::Embeddings(embeddings) => embeddings.evaluate(inputs, unpadded_input_shapes),
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
        unpadded_input_shapes: Vec<Shape>,
    ) -> Result<LayerOut<Element, E>> {
        let output = match self {
            Layer::Dense(dense) => dense.evaluate(inputs, unpadded_input_shapes),
            Layer::Convolution(convolution) => convolution.evaluate(inputs, unpadded_input_shapes),
            Layer::MatMul(mat) => mat.evaluate(inputs, unpadded_input_shapes),
            Layer::QKV(qkv) => qkv.evaluate(inputs, unpadded_input_shapes),
            Layer::Mha(mha) => mha.evaluate(inputs, unpadded_input_shapes),
            Layer::ConcatMatMul(concat_matmul) => {
                concat_matmul.evaluate(inputs, unpadded_input_shapes)
            }
            Layer::LayerNorm(layernorm) => layernorm.evaluate(inputs, unpadded_input_shapes),
            Layer::Softmax(softmax) => softmax.evaluate(inputs, unpadded_input_shapes),
            Layer::Add(add) => add.evaluate(inputs, unpadded_input_shapes),
            Layer::Logits(logits) => logits.evaluate(inputs, unpadded_input_shapes),
            Layer::Positional(positional) => positional.evaluate(inputs, unpadded_input_shapes),
            Layer::Embeddings(embeddings) => embeddings.evaluate(inputs, unpadded_input_shapes),
            Layer::Reshape(reshape) => reshape.evaluate(inputs, unpadded_input_shapes),
            Layer::SchoolBookConvolution(school_book_conv) => {
                school_book_conv.evaluate(inputs, unpadded_input_shapes)
            }
            Layer::Activation(activation) => activation.evaluate(inputs, unpadded_input_shapes),
            Layer::Requant(requant) => requant.evaluate(inputs, unpadded_input_shapes),
            Layer::Pooling(pooling) => pooling.evaluate(inputs, unpadded_input_shapes),
            Layer::Flatten(reshape) => reshape.evaluate(inputs, unpadded_input_shapes),
        };

        #[cfg(feature = "capture-layers-quant")]
        {
            if let Ok(output) = output.as_ref() {
                let layer_kind = self.as_kind_str();
                let out_dir = std::path::PathBuf::from("layers-quant").join(layer_kind);
                crate::capture::store(&out_dir, &(self, inputs), &output.outputs);
            }
        }

        output
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
            Layer::Mha(mha) => mha.step_info(id, aux),
            Layer::ConcatMatMul(concat_matmul) => concat_matmul.step_info(id, aux),
            Layer::Add(add) => add.step_info(id, aux),
            Layer::LayerNorm(layernorm) => layernorm.step_info(id, aux),
            Layer::Softmax(softmax) => softmax.step_info(id, aux),
            Layer::Logits(logits) => logits.step_info(id, aux),
            Layer::Positional(positional) => positional.step_info(id, aux),
            Layer::Embeddings(embeddings) => embeddings.step_info(id, aux),
            Layer::Reshape(reshape) => reshape.step_info(id, aux),
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
            Layer::Mha(mha) => Layer::Mha(mha.pad_node(si)?),
            Layer::ConcatMatMul(concat_matmul) => Layer::ConcatMatMul(concat_matmul.pad_node(si)?),
            Layer::Add(add) => Layer::Add(add.pad_node(si)?),
            Layer::LayerNorm(layernorm) => Layer::LayerNorm(layernorm.pad_node(si)?),
            Layer::Softmax(softmax) => Layer::Softmax(softmax.pad_node(si)?),
            Layer::Logits(logits) => Layer::Logits(logits.pad_node(si)?),
            Layer::Positional(positional) => Layer::Positional(positional.pad_node(si)?),
            Layer::Embeddings(embeddings) => Layer::Embeddings(embeddings.pad_node(si)?),
            Layer::MatMul(mat) => Layer::MatMul(mat.pad_node(si)?),
            Layer::SchoolBookConvolution(school_book_conv) => {
                Layer::SchoolBookConvolution(school_book_conv.pad_node(si)?)
            }
            Layer::Activation(activation) => Layer::Activation(activation.pad_node(si)?),
            Layer::Requant(requant) => Layer::Requant(requant.pad_node(si)?),
            Layer::Pooling(pooling) => Layer::Pooling(pooling.pad_node(si)?),
            Layer::Flatten(flatten) => Layer::Flatten(flatten.pad_node(si)?),
            Layer::Reshape(reshape) => Layer::Reshape(reshape.pad_node(si)?),
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
            (Layer::Mha(mha), LayerCtx::Mha(info)) => {
                mha.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::ConcatMatMul(concat_matmul), LayerCtx::ConcatMatMul(info)) => {
                concat_matmul.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::Embeddings(embeddings), LayerCtx::Embeddings(ctx)) => {
                embeddings.prove(node_id, ctx, last_claims, step_data, prover)
            }
            (Layer::Positional(positional), LayerCtx::Positional(info)) => {
                positional.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::Add(add), LayerCtx::Add(info)) => {
                add.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::Logits(logits), LayerCtx::Logits(info)) => {
                logits.prove(node_id, info, last_claims, step_data, prover)
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
            (Layer::Softmax(softmax), LayerCtx::Softmax(info)) => {
                softmax.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::LayerNorm(layernorm), LayerCtx::LayerNorm(info)) => {
                layernorm.prove(node_id, info, last_claims, step_data, prover)
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
        ctx: &Context<E, PCS>,
        step_data: &StepData<Element, E>,
    ) -> Result<LookupWitnessGen<E, PCS>> {
        match self {
            Layer::Dense(dense) => dense.gen_lookup_witness(id, ctx, step_data),
            Layer::Convolution(convolution) => convolution.gen_lookup_witness(id, ctx, step_data),
            Layer::MatMul(m) => m.gen_lookup_witness(id, ctx, step_data),
            Layer::QKV(qkv) => qkv.gen_lookup_witness(id, ctx, step_data),
            Layer::Mha(mha) => mha.gen_lookup_witness(id, ctx, step_data),
            Layer::ConcatMatMul(concat_matmul) => {
                concat_matmul.gen_lookup_witness(id, ctx, step_data)
            }
            Layer::Add(add) => add.gen_lookup_witness(id, ctx, step_data),
            Layer::Softmax(softmax) => softmax.gen_lookup_witness(id, ctx, step_data),
            Layer::Logits(logits) => logits.gen_lookup_witness(id, ctx, step_data),
            Layer::LayerNorm(layernorm) => layernorm.gen_lookup_witness(id, ctx, step_data),
            Layer::Positional(positional) => positional.gen_lookup_witness(id, ctx, step_data),
            Layer::Embeddings(embeddings) => embeddings.gen_lookup_witness(id, ctx, step_data),
            Layer::SchoolBookConvolution(school_book_conv) => {
                // check that the layer is not provable, so we don't need to call the method
                assert!(!school_book_conv.is_provable());
                Ok(Default::default())
            }
            Layer::Activation(activation) => activation.gen_lookup_witness(id, ctx, step_data),
            Layer::Requant(requant) => requant.gen_lookup_witness(id, ctx, step_data),
            Layer::Pooling(pooling) => pooling.gen_lookup_witness(id, ctx, step_data),
            Layer::Reshape(r) => {
                assert!(!r.is_provable());
                Ok(Default::default())
            }
            Layer::Flatten(r) => {
                assert!(!r.is_provable());
                Ok(Default::default())
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
            Layer::Mha(mha) => {
                let output = mha.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(Layer::Mha(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)
            }
            Layer::ConcatMatMul(concat_matmul) => {
                let output = concat_matmul.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(
                    Layer::ConcatMatMul(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)
            }
            Layer::LayerNorm(layernorm) => {
                let output = layernorm.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(
                    Layer::LayerNorm(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)
            }
            Layer::Softmax(softmax) => {
                let output = softmax.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(Layer::Softmax(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)
            }
            Layer::Add(add) => {
                let output = add.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(Layer::Add(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)
            }
            Layer::Logits(logits) => {
                let output = logits.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(Layer::Logits(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)
            }
            Layer::Positional(positional) => {
                let output = positional.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(
                    Layer::Positional(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)
            }
            Layer::Embeddings(embeddings) => {
                let output = embeddings.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(
                    Layer::Embeddings(output.quantized_op),
                    output.output_scalings,
                )
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
                let output = activation.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(
                    Layer::Activation(output.quantized_op),
                    input_scaling.to_vec(),
                )
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
            Layer::Reshape(reshape) => {
                QuantizeOutput::new(Layer::Reshape(reshape), input_scaling.to_vec())
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
            Self::Mha(_) => "MHA".to_string(),
            Self::ConcatMatMul(..) => "ConcatMatMul".to_string(),
            Self::LayerNorm(_) => "LayerNorm".to_string(),
            Self::Softmax(_) => "Softmax".to_string(),
            Self::Logits(_) => "Logits".to_string(),
            Self::Positional(_) => "Positional".to_string(),
            Self::Add(_) => "Add".to_string(),
            Self::Embeddings(_) => "Embeddings".to_string(),
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
            LayerProof::Mha(proof) => Some(proof.get_lookup_data()),
            LayerProof::ConcatMatMul(..) => None,
            LayerProof::Add(_) => None,
            LayerProof::LayerNorm(proof) => Some(proof.get_lookup_data()),
            LayerProof::Softmax(proof) => Some(proof.get_lookup_data()),
            LayerProof::Logits(proof) => Some(proof.get_lookup_data()),
            LayerProof::Positional(_) => None,
            LayerProof::Embeddings(..) => None,
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
