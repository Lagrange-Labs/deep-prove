pub mod activation;
pub mod common;
pub mod convolution;
pub mod dense;
pub mod flatten;
pub mod hadamard;
pub mod matvec;
pub mod pooling;
pub mod provable;
pub mod requant;

use std::fmt::Debug;

use anyhow::Result;
use ff_ext::ExtensionField;
use flatten::Flatten;
use goldilocks::GoldilocksExt2;
use itertools::Itertools;
use pooling::{PoolingCtx, PoolingProof};
use provable::{
    Evaluate, LayerOut, Op, OpInfo, ProvableNode, ProvableOp, ProvableOpError, ProveInfo, StepData,
    evaluate_layer,
};
use requant::RequantCtx;
use statrs::statistics::{Data, Distribution};
use tracing::debug;
use transcript::Transcript;

use crate::{
    Element,
    commit::precommit::PolyID,
    iop::context::{ContextAux, ShapeStep, TableCtx},
    layers::{
        activation::{Activation, ActivationProof, Relu},
        convolution::Convolution,
        dense::Dense,
        pooling::Pooling,
        requant::{Requant, RequantProof},
    },
    lookup::context::LookupWitnessGen,
    padding::PaddingMode,
    quantization::ScalingFactor,
    tensor::{ConvData, Number, Tensor},
};
use activation::ActivationCtx;
use convolution::{ConvCtx, ConvProof, SchoolBookConv, SchoolBookConvCtx};
use dense::{DenseCtx, DenseProof};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[derive(Clone, Debug)]
pub enum Layer<T> {
    Dense(Dense<T>),
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
    Convolution(ConvProof<E>),
    Activation(ActivationProof<E>),
    Requant(RequantProof<E>),
    Pooling(PoolingProof<E>),
    Flatten,
}
#[derive(Clone, Debug)]
pub enum LayerOutput<F>
where
    F: ExtensionField,
{
    NormalOut(Tensor<Element>),
    ConvOut((Tensor<Element>, ConvData<F>)),
}

impl<E> LayerCtx<E>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    pub fn variant_name(&self) -> String {
        match self {
            Self::Dense(_) => "Dense".to_string(),
            Self::SchoolBookConvolution(_) => "Traditional Convolution".to_string(),
            Self::Convolution(_) => "Convolution".to_string(),
            Self::Activation(_) => "Activation".to_string(),
            Self::Requant(_) => "Requant".to_string(),
            Self::Pooling(_) => "Pooling".to_string(),
            Self::Table(..) => "Table".to_string(),
            Self::Flatten => "Reshape".to_string(),
        }
    }

    // TODO: is this used and correct ??
    pub fn requires_lookup(&self) -> bool {
        match self {
            Self::Dense(..) => false,
            Self::Flatten => false,
            _ => true,
        }
    }
    pub fn output_shape(&self, input_shape: &[usize], padding_mode: PaddingMode) -> Vec<usize> {
        match self {
            Self::Dense(ref dense) => dense.output_shape(input_shape, padding_mode),
            Self::Convolution(ref filter) => filter.output_shape(input_shape, padding_mode),
            Self::SchoolBookConvolution(ref _filter) => {
                panic!("SchoolBookConvolution should NOT be used in proving")
            }
            Self::Activation(..) => input_shape.to_vec(),
            Self::Requant(..) => input_shape.to_vec(),
            Self::Pooling(ref pooling) => pooling.output_shape(input_shape),
            Self::Flatten => <Flatten as OpInfo>::output_shapes(
                &Flatten,
                &vec![input_shape.to_vec()],
                padding_mode,
            )[0]
            .clone(),
            Self::Table(..) => panic!("Table should NOT be used in proving"),
        }
    }
    pub fn next_shape_step(&self, last_step: &ShapeStep) -> ShapeStep {
        let unpadded_output = last_step
            .unpadded_output_shape
            .iter()
            .map(|shape| self.output_shape(&shape, PaddingMode::NoPadding))
            .collect();
        let padded_output = last_step
            .padded_output_shape
            .iter()
            .map(|shape| self.output_shape(&shape, PaddingMode::Padding))
            .collect();
        ShapeStep::next_step(last_step, unpadded_output, padded_output)
    }
    pub fn shape_step(
        &self,
        unpadded_input: &[Vec<usize>],
        padded_input: &[Vec<usize>],
    ) -> ShapeStep {
        let unpadded_output = unpadded_input
            .iter()
            .map(|shape| self.output_shape(&shape, PaddingMode::NoPadding))
            .collect();
        let padded_output = padded_input
            .iter()
            .map(|shape| self.output_shape(&shape, PaddingMode::Padding))
            .collect();
        ShapeStep::new(
            unpadded_input.to_vec(),
            padded_input.to_vec(),
            unpadded_output,
            padded_output,
        )
    }
}

impl<N> ProvableNode<N>
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
    ) -> Result<LayerOut<N, E>, ProvableOpError>
    where
        N: Number,
        Layer<N>: Evaluate<N>,
    {
        self.operation.evaluate(inputs, unpadded_input_shapes)
    }

    pub(crate) fn step_info<E>(&self, id: PolyID, aux: ContextAux) -> (LayerCtx<E>, ContextAux)
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        Layer<N>: ProveInfo<E>,
    {
        self.operation.step_info(id, aux)
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
    ) -> Result<LayerOut<f32, E>, ProvableOpError> {
        match self {
            Layer::Dense(dense) => dense.evaluate(inputs, unpadded_input_shapes),
            Layer::Convolution(convolution) => convolution.evaluate(inputs, unpadded_input_shapes),
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
    ) -> Result<LayerOut<Element, E>, ProvableOpError> {
        match self {
            Layer::Dense(dense) => dense.evaluate(inputs, unpadded_input_shapes),
            Layer::Convolution(convolution) => convolution.evaluate(inputs, unpadded_input_shapes),
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
    fn step_info(&self, id: PolyID, aux: ContextAux) -> (LayerCtx<E>, ContextAux) {
        match self {
            Layer::Dense(dense) => dense.step_info(id, aux),
            Layer::Convolution(convolution) => convolution.step_info(id, aux),
            Layer::SchoolBookConvolution(convolution) => convolution.step_info(id, aux),
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
            Layer::SchoolBookConvolution(school_book_conv) => school_book_conv.commit_info(id),
            Layer::Activation(activation) => activation.commit_info(id),
            Layer::Requant(requant) => requant.commit_info(id),
            Layer::Pooling(pooling) => pooling.commit_info(id),
            Layer::Flatten(reshape) => reshape.commit_info(id),
        }
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
        last_claims: Vec<crate::Claim<E>>,
        step_data: &provable::StepData<E, E>,
        prover: &mut crate::Prover<E, T>,
    ) -> Result<Vec<crate::Claim<E>>, ProvableOpError> {
        match self {
            Layer::Dense(dense) => {
                if let LayerCtx::Dense(info) = ctx {
                    dense.prove(node_id, info, last_claims, step_data, prover)
                } else {
                    Err(ProvableOpError::ParameterError(
                        "No dense ctx found for dense layer".to_string(),
                    ))
                }
            }
            Layer::Convolution(convolution) => {
                if let LayerCtx::Convolution(info) = ctx {
                    convolution.prove(node_id, info, last_claims, step_data, prover)
                } else {
                    Err(ProvableOpError::ParameterError(
                        "No convolution ctx found for convolution layer".to_string(),
                    ))
                }
            }
            Layer::SchoolBookConvolution(_) => {
                unreachable!("prove cannot be called for school book convolution")
            }
            Layer::Activation(activation) => {
                if let LayerCtx::Activation(info) = ctx {
                    activation.prove(node_id, info, last_claims, step_data, prover)
                } else {
                    Err(ProvableOpError::ParameterError(
                        "No activation ctx found for activation layer".to_string(),
                    ))
                }
            }
            Layer::Requant(requant) => {
                if let LayerCtx::Requant(info) = ctx {
                    requant.prove(node_id, info, last_claims, step_data, prover)
                } else {
                    Err(ProvableOpError::ParameterError(
                        "No requant ctx found for requant layer".to_string(),
                    ))
                }
            }
            Layer::Pooling(pooling) => {
                if let LayerCtx::Pooling(info) = ctx {
                    pooling.prove(node_id, info, last_claims, step_data, prover)
                } else {
                    Err(ProvableOpError::ParameterError(
                        "No pooling ctx found for pooling layer".to_string(),
                    ))
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
    ) -> Result<(), ProvableOpError> {
        match self {
            Layer::Dense(dense) => dense.gen_lookup_witness(id, gen, step_data),
            Layer::Convolution(convolution) => convolution.gen_lookup_witness(id, gen, step_data),
            Layer::SchoolBookConvolution(school_book_conv) => {
                school_book_conv.gen_lookup_witness(id, gen, step_data)
            }
            Layer::Activation(activation) => activation.gen_lookup_witness(id, gen, step_data),
            Layer::Requant(requant) => requant.gen_lookup_witness(id, gen, step_data),
            Layer::Pooling(pooling) => pooling.gen_lookup_witness(id, gen, step_data),
            Layer::Flatten(reshape) => reshape.gen_lookup_witness(id, gen, step_data),
        }
    }
}

impl<E: ExtensionField> Op<E, Element> for Layer<Element>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
}

impl<T: Number> Layer<T> {
    pub fn output_shape(&self, input_shape: &[usize], padding_mode: PaddingMode) -> Vec<usize> {
        match self {
            Layer::Dense(ref dense) => dense.output_shape(input_shape, padding_mode),
            Layer::Convolution(ref filter) => filter.output_shape(input_shape, padding_mode),
            Layer::SchoolBookConvolution(ref filter) => {
                filter.0.output_shape(input_shape, padding_mode)
            }
            Layer::Activation(Activation::Relu(_)) => input_shape.to_vec(),
            Layer::Requant(_) => input_shape.to_vec(),
            Layer::Pooling(Pooling::Maxpool2D(info)) => info.output_shape(input_shape),
            Layer::Flatten(ref r) => {
                r.output_shapes(&vec![input_shape.to_vec()], padding_mode)[0].clone()
            }
        }
    }
    /// Returns the shape of the layer as used in the model. If the layer do NOT have a shape per se,
    /// e.g. RELU for example, it returns None.
    pub fn model_shape(&self) -> Option<Vec<usize>> {
        match &self {
            Layer::Dense(ref dense) => Some(dense.matrix.get_shape()),

            Layer::Convolution(ref filter) => Some(filter.get_shape()),
            Layer::SchoolBookConvolution(ref filter) => Some(filter.0.get_shape()),

            Layer::Activation(Activation::Relu(_)) => None,
            Layer::Requant(_) => None,
            Layer::Pooling(Pooling::Maxpool2D(info)) => {
                Some(vec![info.kernel_size, info.kernel_size])
            }
            Layer::Flatten(ref _reshape) => None,
        }
    }

    pub fn describe(&self) -> String {
        match &self {
            Layer::Dense(ref dense) => dense.describe(),
            Layer::Convolution(ref filter) => {
                format!(
                    "Conv: ({},{},{},{})",
                    filter.kw(),
                    filter.kx(),
                    filter.nw(),
                    filter.nw()
                )
            }
            Layer::SchoolBookConvolution(ref _filter) => {
                format!(
                    "Conv: Traditional convolution for debug purposes" /* matrix.fmt_integer() */
                )
            }
            Layer::Activation(Activation::Relu(_)) => {
                format!("RELU: {}", 1 << Relu::num_vars())
            }
            Layer::Requant(info) => {
                format!(
                    "Requant: shift: {}, offset: 2^{}",
                    info.right_shift,
                    (info.range << 1).ilog2() as usize,
                )
            }
            Layer::Pooling(Pooling::Maxpool2D(info)) => format!(
                "MaxPool2D{{ kernel size: {}, stride: {} }}",
                info.kernel_size, info.stride
            ),
            Layer::Flatten(ref reshape) => reshape.describe(),
        }
    }
    pub fn needs_requant(&self) -> bool {
        match self {
            Layer::Dense(..) | Layer::Convolution(..) => true,
            _ => false,
        }
    }
    pub fn is_provable(&self) -> bool {
        match self {
            Layer::Flatten(..) => false,
            _ => true,
        }
    }
}

impl Layer<f32> {
    pub fn quantize(self, s: &ScalingFactor, bias_s: Option<&ScalingFactor>) -> Layer<Element> {
        match self {
            Layer::Dense(dense) => {
                Layer::Dense(dense.quantize(s, bias_s.expect("bias_s is required for dense layer")))
            }
            Layer::Convolution(conv) => Layer::Convolution(conv.quantize(
                &s,
                bias_s.expect("bias_s is required for convolution layer"),
            )),
            Layer::SchoolBookConvolution(conv) => {
                Layer::SchoolBookConvolution(SchoolBookConv(conv.0.quantize(
                    &s,
                    bias_s.expect("bias_s is required for schoolbook convolution layer"),
                )))
            }
            Layer::Activation(activation) => Layer::Activation(activation),
            Layer::Requant(requant) => Layer::Requant(requant),
            Layer::Pooling(pooling) => Layer::Pooling(pooling),
            Layer::Flatten(_reshape) => Layer::Flatten(Flatten),
        }
    }
    /// TODO: limitation of enum is we can't have same names as in Element run
    pub(crate) fn run(&self, input: &Tensor<f32>) -> Tensor<f32> {
        match self {
            Layer::Dense(ref dense) => {
                evaluate_layer::<GoldilocksExt2, _, _>(dense, &vec![input], None)
                    .unwrap()
                    .outputs()[0]
                    .clone()
            }
            Layer::Activation(activation) => {
                evaluate_layer::<GoldilocksExt2, _, _>(activation, &vec![input], None)
                    .unwrap()
                    .outputs()[0]
                    .clone()
            }
            Layer::Convolution(ref conv_pair) => {
                input.conv2d(&conv_pair.filter, &conv_pair.bias, 1)
            }
            Layer::Pooling(info) => info.op(input),
            // Traditional convolution is used for debug purposes. That is because the actual convolution
            // we use relies on the FFT algorithm. This convolution does not have a snark implementation.
            Layer::SchoolBookConvolution(ref conv_pair) => {
                input.conv2d(&conv_pair.0.filter, &conv_pair.0.bias, 1)
            }
            Layer::Flatten(ref reshape) => {
                evaluate_layer::<GoldilocksExt2, _, _>(reshape, &vec![input], None)
                    .unwrap()
                    .outputs()[0]
                    .clone()
            }
            Layer::Requant(_) => {
                panic!(
                    "InferenceObserver: requantization layer found while observing inference on float !?"
                );
            }
        }
    }
}

impl Layer<Element> {
    /// Run the operation associated with that layer with the given input
    // TODO: move to tensor library : right now it works because we assume there is only Dense
    // layer which is matmul
    pub fn op<F: ExtensionField>(
        &self,
        input: &Tensor<Element>,
        unpadded_shape: &[usize],
    ) -> Result<LayerOutput<F>> {
        let output = match &self {
            Layer::Dense(ref dense) => Ok(LayerOutput::NormalOut(
                evaluate_layer::<GoldilocksExt2, _, _>(
                    dense,
                    &vec![input],
                    Some(vec![unpadded_shape.to_vec()]),
                )
                .unwrap()
                .outputs()[0]
                    .clone(),
            )),
            Layer::Activation(activation) => Ok(LayerOutput::NormalOut(
                evaluate_layer::<GoldilocksExt2, _, _>(
                    activation,
                    &vec![input],
                    Some(vec![unpadded_shape.to_vec()]),
                )
                .unwrap()
                .outputs()[0]
                    .clone(),
            )),
            Layer::Convolution(ref filter) => {
                Ok(LayerOutput::ConvOut(filter.op(input, unpadded_shape)))
            }
            // Traditional convolution is used for debug purposes. That is because the actual convolution
            // we use relies on the FFT algorithm. This convolution does not have a snark implementation.
            Layer::SchoolBookConvolution(ref conv_pair) => Ok(LayerOutput::NormalOut(
                input.conv2d(&conv_pair.0.filter, &conv_pair.0.bias, 1),
            )),

            Layer::Requant(info) => info.op(input).map(|r| LayerOutput::NormalOut(r)),
            Layer::Pooling(info) => Ok(LayerOutput::NormalOut(info.op(input))),
            Layer::Flatten(reshape) => Ok(LayerOutput::NormalOut(
                evaluate_layer::<GoldilocksExt2, _, _>(
                    reshape,
                    &vec![input],
                    Some(vec![unpadded_shape.to_vec()]),
                )
                .unwrap()
                .outputs()[0]
                    .clone(),
            )),
        }?;
        match output {
            LayerOutput::NormalOut(ref output) => {
                debug!(
                    "Layer::{:?}: shape {:?} op: {:?} - min {:?}, max {:?}",
                    self.describe(),
                    output.get_shape(),
                    &output.get_data()[..output.get_data().len().min(10)],
                    output.get_data().iter().min().unwrap(),
                    output.get_data().iter().max().unwrap()
                );
            }
            LayerOutput::ConvOut((ref output, _)) => {
                let d = Data::new(output.get_data().iter().map(|e| *e as f64).collect_vec());
                debug!(
                    "Layer::{:?}: shape {:?} op: {:?} - min {:?}, max {:?}, mean {:?}, std {:?}",
                    self.describe(),
                    output.get_shape(),
                    &output.get_data()[..output.get_data().len().min(10)],
                    output.get_data().iter().min().unwrap(),
                    output.get_data().iter().max().unwrap(),
                    d.mean().unwrap(),
                    d.std_dev().unwrap()
                );
            }
        }
        Ok(output)
    }
}

impl<E: ExtensionField> LayerProof<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    pub fn variant_name(&self) -> String {
        match self {
            Self::Dense(_) => "Dense".to_string(),
            Self::Convolution(_) => "Convolution".to_string(),
            Self::Activation(_) => "Activation".to_string(),
            Self::Requant(_) => "Requant".to_string(),
            Self::Pooling(_) => "Pooling".to_string(),
            Self::Flatten => "Reshape".to_string(),
        }
    }

    pub fn get_lookup_data(&self) -> Option<(Vec<E>, Vec<E>)> {
        match self {
            LayerProof::Dense(..) => None,
            LayerProof::Convolution(..) => None,
            LayerProof::Flatten => None,
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
