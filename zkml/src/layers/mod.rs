pub mod activation;
pub mod convolution;
pub mod dense;
pub mod pooling;
use crate::{
    Element,
    layers::{
        activation::{Activation, Relu},
        convolution::Convolution,
        dense::Dense,
        pooling::Pooling,
    },
    quantization::Requant,
    tensor::{ConvData, Tensor},
};
use ff_ext::ExtensionField;
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

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.describe())
    }
}

pub enum LayerOutput<F>
where
    F: ExtensionField,
{
    NormalOut(Tensor<Element>),
    ConvOut((Tensor<Element>, ConvData<F>)),
}

impl Layer {
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
