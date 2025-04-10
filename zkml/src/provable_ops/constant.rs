//! Parent module for all code related to the [`Constant`] operation.

use crate::{quantization::QuantisationParams, tensor::{deep_tensor::Number, DeepTensor}};

mod op_impl;
mod tract_conversion;

#[derive(Debug, Clone)]
/// The [`Constant`] op, it just returns a [`DeepTensor`] that is used in other operations.
/// It also has quantisaiotn parameters (if applicable).
pub struct Constant<T: Number> {
    /// The constant tensor this node "outputs"
    tensor: DeepTensor<T>,
    /// Parameters for quantising the node.
    quant_params: Option<QuantisationParams>,
}
