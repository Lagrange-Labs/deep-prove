//! Definition and implementation of [`InferenceStep`]

use crate::{provable_ops::InferenceOp, tensor::DeepTensor};
use tract_onnx::prelude::*;

/// Info about a step in inference
pub struct InferenceStep<T> {
    /// The input tensors that depend on the run
    inputs: Vec<DeepTensor<T>>,
    /// Any constant inputs (together with their node info)
    const_inputs: Vec<(DeepTensor<T>, usize)>,
    /// Outputs of the operation
    outputs: Vec<DeepTensor<T>>,
    /// The operation performed
    op: Box<dyn InferenceOp<T>>,
}
