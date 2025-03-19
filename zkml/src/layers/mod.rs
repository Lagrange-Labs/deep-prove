pub mod activation;
pub mod convolution;
pub mod dense;
pub mod pooling;
pub mod requant;

use crate::{
    Element,
    commit::precommit::PolyID,
    iop::context::{ContextAux, TableCtx},
    layers::{
        activation::{Activation, ActivationProof, Relu},
        convolution::Convolution,
        dense::Dense,
        pooling::Pooling,
        requant::{Requant, RequantProof},
    },
    tensors::{ConvData, Tensor},
};
use activation::ActivationCtx;
use convolution::{ConvCtx, ConvProof, SchoolBookConvCtx};
use dense::{DenseCtx, DenseProof};
use ff_ext::ExtensionField;
use pooling::{PoolingCtx, PoolingProof};
use requant::RequantCtx;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
#[derive(Clone, Debug)]
pub enum Layer {
    Dense(Dense),
    // TODO: replace this with a Tensor based implementation
    Convolution(Convolution),
    // Traditional convolution is used for debug purposes. That is because the actual convolution
    // we use relies on the FFT algorithm. This convolution does not have a snark implementation.
    SchoolBookConvolution(Convolution),
    Activation(Activation),
    // this is the output quant info. Since we always do a requant layer after each dense,
    // then we assume the inputs requant info are default()
    Requant(Requant),
    Pooling(Pooling),
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
}
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
        }
    }

    pub fn requires_lookup(&self) -> bool {
        match self {
            Self::Dense(..) => false,
            _ => true,
        }
    }
}
impl Layer {
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
            Layer::Pooling(pooling) => pooling.step_info(id, aux),
        }
    }
    /// Run the operation associated with that layer with the given input
    // TODO: move to tensor library : right now it works because we assume there is only Dense
    // layer which is matmul
    pub fn op<F: ExtensionField>(&self, input: &Tensor<Element>) -> LayerOutput<F> {
        match &self {
            Layer::Dense(ref dense) => LayerOutput::NormalOut(dense.op(input)),
            Layer::Activation(activation) => LayerOutput::NormalOut(activation.op(input)),

            Layer::Convolution(ref filter) => LayerOutput::ConvOut(filter.op(input)),
            // Traditional convolution is used for debug purposes. That is because the actual convolution
            // we use relies on the FFT algorithm. This convolution does not have a snark implementation.
            Layer::SchoolBookConvolution(ref conv_pair) => {
                // LayerOutput::NormalOut(filter.cnn_naive_convolution(input))
                LayerOutput::NormalOut(input.conv2d(&conv_pair.filter, &conv_pair.bias, 1))
            }

            Layer::Requant(info) => {
                // NOTE: we assume we have default quant structure as input
                LayerOutput::NormalOut(info.op(input))
            }
            Layer::Pooling(info) => LayerOutput::NormalOut(info.op(input)),
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match &self {
            Layer::Dense(ref dense) => vec![dense.matrix.nrows_2d(), dense.matrix.ncols_2d()],

            Layer::Convolution(ref filter) => filter.get_shape(),
            Layer::SchoolBookConvolution(ref filter) => filter.get_shape(),

            Layer::Activation(Activation::Relu(_)) => Relu::shape(),
            Layer::Requant(info) => info.shape(),
            Layer::Pooling(Pooling::Maxpool2D(info)) => vec![info.kernel_size, info.kernel_size],
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
                format!("Requant: {}", info.shape()[1])
            }
            Layer::Pooling(Pooling::Maxpool2D(info)) => format!(
                "MaxPool2D{{ kernel size: {}, stride: {} }}",
                info.kernel_size, info.stride
            ),
        }
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
        }
    }

    pub fn get_lookup_data(&self) -> Option<(Vec<E>, Vec<E>)> {
        match self {
            LayerProof::Dense(..) => None,
            LayerProof::Convolution(..) => None,
            LayerProof::Activation(ActivationProof { lookup, .. })
            | LayerProof::Requant(RequantProof { lookup, .. })
            | LayerProof::Pooling(PoolingProof { lookup, .. }) => Some(lookup.fractional_outputs()),
        }
    }
}
impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.describe())
    }
}
