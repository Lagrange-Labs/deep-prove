//! This module contains the definition of [`InferenceModel`], the graph like structure we use to represent a deep
//! learning model. It can be specified to run with a number of different types and given some quantisation parameters
//! (if required) can be transformed into a [`ProvableModel`].

use super::node::InferenceNode;
use crate::{model::Edge, quantization::QuantisationParams};

#[derive(Debug)]
pub struct InferenceModel<T> {
    /// The list of all nodes in the graph
    nodes: Vec<InferenceNode<T>>,
    /// This contains the input nodes to this graph i.e. where we should start
    /// traversal.
    inputs: Vec<usize>,
    /// This contains all outputs of the graph i.e. nodes whose outputs are
    /// not used anywhere else
    outputs: Vec<Edge>,
    /// Contains quantisation parameters (if applicable), should always be
    /// intialised to `None` until calibration has been performed.
    quant_params: Option<QuantisationParams>,
    /// Contains the order to evaluate the nodes in `self.nodes`
    eval_order: Vec<usize>,
}
