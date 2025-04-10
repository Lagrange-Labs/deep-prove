//! Module defining an [`InferenceNode`], an element of the [`InferenceModel`] graph.

use crate::{
    model::{Edge, OutputData},
    provable_ops::InferenceOp,
};

#[derive(Debug)]
pub struct InferenceNode<T> {
    /// The position of this node in the graphs storage
    id: usize,
    /// The inputs to this node, here `Edge` is just a type alias for `(usize, usize)`
    /// where the first `usize` is the index of the previous node and the second is its position among its outputs.
    inputs: Vec<Edge>,
    /// Where the tensors produced by this node will be used, each could have multiple
    /// locations so they are stored in a struct [`OutputData`].
    outputs: Vec<OutputData>,
    /// The type erased [`InferenceOp`] that this node relates to
    operation: Box<dyn InferenceOp<T>>,
}
