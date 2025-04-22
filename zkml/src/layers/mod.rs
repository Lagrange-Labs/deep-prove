pub mod activation;
pub mod common;
pub mod convolution;
pub mod dense;
pub mod hadamard;
pub mod pooling;
pub mod requant;
pub mod reshape;

use anyhow::Result;
use ff_ext::ExtensionField;
use itertools::Itertools;
use pooling::{PoolingCtx, PoolingProof};
use requant::{FullRequant, RequantCtx};
use reshape::Reshape;
use statrs::statistics::{Data, Distribution};

use crate::{
    Element,
    commit::precommit::PolyID,
    iop::context::{ContextAux, TableCtx},
    layers::{
        activation::{Activation, ActivationProof, Relu},
        convolution::Convolution,
        dense::Dense,
        pooling::Pooling,
        requant::{FullRequantProof, Requant, RequantProof},
    },
    quantization::ScalingFactor,
    tensor::{ConvData, Number, Tensor},
};
use activation::ActivationCtx;
use common::{Op, QuantizableOp};
use convolution::{ConvCtx, ConvProof, SchoolBookConvCtx};
use dense::{DenseCtx, DenseProof};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::Sha256;

#[derive(Clone, Debug)]
pub enum Layer<T> {
    Dense(Dense<T>),
    // TODO: replace this with a Tensor based implementation
    Convolution(Convolution<T>),
    // Traditional convolution is used for debug purposes. That is because the actual convolution
    // we use relies on the FFT algorithm. This convolution does not have a snark implementation.
    SchoolBookConvolution(Convolution<T>),
    Activation(Activation),
    // this is the output quant info. Since we always do a requant layer after each dense,
    // then we assume the inputs requant info are default()
    Requant(Requant),
    FullRequant(FullRequant),
    Pooling(Pooling),
    // TODO: so far it's only flattening the input tensor, e.g. new_shape = vec![shape.iter().product()]
    Reshape(Reshape),
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
    FullRequant(FullRequant),
    Pooling(PoolingCtx),
    Table(TableCtx<E>),
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
    FullRequant(FullRequantProof<E>),
    Pooling(PoolingProof<E>),
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
            Self::FullRequant(..) => "FullRequant".to_string(),
            Self::Pooling(_) => "Pooling".to_string(),
            Self::Table(..) => "Table".to_string(),
        }
    }

    pub fn requires_lookup(&self) -> bool {
        match self {
            Self::Dense(..) => false,
            _ => true,
        }
    }
}

impl<T: Number> Layer<T> {
    /// TODO: this method should be renamed to output_shape. THe internals of the layer should only be disclosed
    /// for some node but not all (e.g. reshape doesn't have an internal shape).
    pub fn shape(&self) -> Vec<usize> {
        match &self {
            Layer::Dense(ref dense) => vec![dense.matrix.nrows_2d(), dense.matrix.ncols_2d()],

            Layer::Convolution(ref filter) => filter.get_shape(),
            Layer::SchoolBookConvolution(ref filter) => filter.get_shape(),

            Layer::Activation(Activation::Relu(_)) => Relu::shape(),
            Layer::Requant(info) => info.shape(),

            Layer::Pooling(Pooling::Maxpool2D(info)) => vec![info.kernel_size, info.kernel_size],
            Layer::Reshape(ref _reshape) => vec![1],
            Layer::FullRequant(..) => vec![1],
        }
    }

    pub fn describe(&self) -> String {
        match &self {
            Layer::Dense(ref dense) => {
                format!(
                    "Dense: ({},{})",
                    dense.matrix.nrows_2d(),
                    dense.matrix.ncols_2d(),
                    // matrix.fmt_integer()
                )
            }
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
                    "Requant: shape: {}, shift: {}, offset: 2^{}",
                    info.shape()[1],
                    info.right_shift,
                    (info.range << 1).ilog2() as usize,
                )
            }
            Layer::FullRequant(info) => {
                format!("Full Requant: multiplier: {}", info.multiplier)
            }
            Layer::Pooling(Pooling::Maxpool2D(info)) => format!(
                "MaxPool2D{{ kernel size: {}, stride: {} }}",
                info.kernel_size, info.stride
            ),
            Layer::Reshape(ref reshape) => describe_op::<T, Reshape>(reshape),
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
            Layer::Reshape(..) => false,
            _ => true,
        }
    }
}

fn describe_op<N: Number, O: Op<N>>(op: &O) -> String {
    format!("{}: {:?}", op.describe(), op.output_shape())
}

impl Layer<f32> {
    pub fn quantize(self, s: &ScalingFactor, bias_s: Option<&ScalingFactor>) -> Layer<Element> {
        match self {
            Layer::Dense(dense) => Layer::Dense(dense.quantize(s, bias_s)),
            Layer::Convolution(conv) => Layer::Convolution(conv.quantize(&s, bias_s)),
            Layer::SchoolBookConvolution(conv) => {
                Layer::SchoolBookConvolution(conv.quantize(&s, bias_s))
            }
            Layer::Activation(activation) => Layer::Activation(activation),
            Layer::Requant(requant) => Layer::Requant(requant),
            Layer::FullRequant(full_requant) => Layer::FullRequant(full_requant),
            Layer::Pooling(pooling) => Layer::Pooling(pooling),
            Layer::Reshape(reshape) => reshape.quantize(s, bias_s),
        }
    }
    /// TODO: limitation of enum is we can't have same names as in Element run
    pub(crate) fn run(&self, input: &Tensor<f32>) -> Tensor<f32> {
        match self {
            Layer::Dense(ref dense) => dense.op(input),
            Layer::Activation(activation) => activation.op(input),
            Layer::Convolution(ref conv_pair) => {
                input.conv2d(&conv_pair.filter, &conv_pair.bias, 1)
            }
            Layer::Pooling(info) => info.op(input),
            // Traditional convolution is used for debug purposes. That is because the actual convolution
            // we use relies on the FFT algorithm. This convolution does not have a snark implementation.
            Layer::SchoolBookConvolution(ref conv_pair) => {
                input.conv2d(&conv_pair.filter, &conv_pair.bias, 1)
            }
            Layer::Reshape(ref reshape) => reshape.op(input),
            Layer::Requant(_) | Layer::FullRequant(..) => {
                panic!(
                    "InferenceObserver: requantization layer found while observing inference on float !?"
                );
            }
        }
    }
}

impl Layer<Element> {
    pub(crate) fn step_info<E>(&self, id: PolyID, aux: ContextAux) -> (LayerCtx<E>, ContextAux)
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        match self {
            Layer::Dense(dense) => dense.step_info(id, aux),
            Layer::Convolution(conv) => conv.step_info(id, aux),
            Layer::SchoolBookConvolution(_conv) => SchoolBookConvCtx.step_info(id, aux),
            Layer::Activation(activation) => activation.step_info(id, aux),
            Layer::Requant(requant) => requant.step_info(id, aux),
            Layer::FullRequant(full_requant) => full_requant.step_info(id, aux),
            Layer::Pooling(pooling) => pooling.step_info(id, aux),
            _ => panic!(
                "Layer::step_info: layer {} can not be proven",
                self.describe()
            ),
        }
    }

    /// Run the operation associated with that layer with the given input
    // TODO: move to tensor library : right now it works because we assume there is only Dense
    // layer which is matmul
    pub fn op<F: ExtensionField>(&self, input: &Tensor<Element>) -> Result<LayerOutput<F>> {
        let output = match &self {
            Layer::Dense(ref dense) => Ok(LayerOutput::NormalOut(dense.op(input))),
            Layer::Activation(activation) => Ok(LayerOutput::NormalOut(activation.op(input))),
            Layer::Convolution(ref filter) => Ok(LayerOutput::ConvOut(filter.op(input))),
            // Layer::Convolution(ref filter) => LayerOutput::NormalOut(input.conv2d(&filter.filter,&filter.bias,1)),
            // Traditional convolution is used for debug purposes. That is because the actual convolution
            // we use relies on the FFT algorithm. This convolution does not have a snark implementation.
            Layer::SchoolBookConvolution(ref conv_pair) => {
                // LayerOutput::NormalOut(filter.cnn_naive_convolution(input))
                Ok(LayerOutput::NormalOut(input.conv2d(
                    &conv_pair.filter,
                    &conv_pair.bias,
                    1,
                )))
            }

            Layer::Requant(info) => info.op(input).map(|r| LayerOutput::NormalOut(r)),
            Layer::FullRequant(info) => info.op(input).map(|r| LayerOutput::NormalOut(r)),
            Layer::Pooling(info) => Ok(LayerOutput::NormalOut(info.op(input))),
            Layer::Reshape(reshape) => Ok(LayerOutput::NormalOut(reshape.op(input))),
        }?;
        match output {
            LayerOutput::NormalOut(ref output) => {
                println!(
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
                println!(
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

use hex;
use sha2::Digest;
pub fn hashit(data: &[Element]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(
        data.iter()
            .map(|e| e.to_be_bytes())
            .flatten()
            .collect::<Vec<_>>(),
    );
    hex::encode(hasher.finalize().to_vec())
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
            Self::FullRequant(..) => "FullRequant".to_string(),
            Self::Pooling(_) => "Pooling".to_string(),
        }
    }

    pub fn get_lookup_data(&self) -> Option<(Vec<E>, Vec<E>)> {
        match self {
            LayerProof::Dense(..) => None,
            LayerProof::Convolution(..) => None,
            LayerProof::FullRequant(..) => None,
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
