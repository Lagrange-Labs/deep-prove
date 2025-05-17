use std::collections::{BTreeSet, HashMap};

use ff_ext::ExtensionField;
use serde::{Serialize, de::DeserializeOwned};

use crate::layers::provable::{NodeCtx, NodeEgdes, NodeId, ProvableNode};

use super::{ModelCtx, ProvableModel};

pub trait ToIterator<E: NodeEgdes> {
    /// Produces an iterator over a set of nodes in a model, starting from the inputs
    /// and yielding nodes in order according to whether their inputs all come from
    /// nodes already visited by the iterator  
    fn to_forward_iterator<'a>(&'a self) -> NodeIterator<'a, E, true>;

    /// Produces an iterator over a set of nodes in a model, starting from the outputs
    /// and yielding nodes in order according to whether their outputs all come from
    /// nodes already visited by the iterator.
    fn to_backward_iterator<'a>(&'a self) -> NodeIterator<'a, E, false>;

    /// Variant of `to_forward_iterator` which takes ownership of the set of nodes
    fn into_forward_iterator(self) -> IntoNodeIterator<E, true>
    where
        Self: Sized + NodeCollection<E>,
    {
        IntoNodeIterator::new(self)
    }

    /// Variant of `to_backward_iterator` which takes ownership of the set of nodes
    fn into_backward_iterator(self) -> IntoNodeIterator<E, false>
    where
        Self: Sized + NodeCollection<E>,
    {
        IntoNodeIterator::new(self)
    }
}

// Forward iterator for the nodes in a model. This is useful for traversing the model when
// evaluating it at interence time
pub type ModelForwardIterator<'a, N> = NodeIterator<'a, ProvableNode<N>, true>;
// Backward iterator for the nodes in a model. This is useful for traversing the model when
// proving
pub type ModelBackwardIterator<'a, N> = NodeIterator<'a, ProvableNode<N>, false>;

impl<E: ExtensionField + DeserializeOwned> ToIterator<NodeCtx<E>> for ModelCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    fn to_forward_iterator<'a>(&'a self) -> ModelCtxForwardIterator<'a, E> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().cloned().collect(),
            nodes: &self.nodes,
        }
    }

    fn to_backward_iterator<'a>(&'a self) -> ModelCtxBackwardIterator<'a, E> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().cloned().collect(),
            nodes: &self.nodes,
        }
    }
}

impl<N> ToIterator<ProvableNode<N>> for ProvableModel<N> {
    fn to_forward_iterator<'a>(&'a self) -> ModelForwardIterator<'a, N> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().cloned().collect(),
            nodes: &self.nodes,
        }
    }

    fn to_backward_iterator<'a>(&'a self) -> ModelBackwardIterator<'a, N> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().cloned().collect(),
            nodes: &self.nodes,
        }
    }
}

/// A trait representing a collections of nodes with their input/output edges
pub trait NodeCollection<E: NodeEgdes> {
    fn nodes(self) -> HashMap<NodeId, E>;
}

impl<N> NodeCollection<ProvableNode<N>> for ProvableModel<N> {
    fn nodes(self) -> HashMap<NodeId, ProvableNode<N>> {
        self.nodes
    }
}

/// Forward iterator for the proving contexts of nodes in a model
pub type ModelCtxForwardIterator<'a, E> = NodeIterator<'a, NodeCtx<E>, true>;
/// Backward iterator for the proving contexts of nodes in a model
pub type ModelCtxBackwardIterator<'a, E> = NodeIterator<'a, NodeCtx<E>, false>;

/// Structure employed to implemented forward and backward iterators
pub struct NodeIterator<'a, E: NodeEgdes, const FORWARD: bool> {
    pub(crate) unvisited_nodes: BTreeSet<NodeId>, /* Use BTreeSet to make the iterator deterministic */
    pub(crate) nodes: &'a HashMap<NodeId, E>,
}

/// Structure employed to implement forward and backward iterators that
/// take ownership of the set of nodes
pub struct IntoNodeIterator<E: NodeEgdes, const FORWARD: bool> {
    pub(crate) node_ids: Vec<NodeId>,
    pub(crate) nodes: HashMap<NodeId, E>,
}

impl<E: NodeEgdes, const FORWARD: bool> IntoNodeIterator<E, FORWARD> {
    fn new<I: ToIterator<E> + NodeCollection<E>>(iter: I) -> Self {
        let mut node_ids: Vec<_> = if FORWARD {
            iter.to_forward_iterator()
                .map(|(node_id, _)| node_id)
                .collect()
        } else {
            iter.to_backward_iterator()
                .map(|(node_id, _)| node_id)
                .collect()
        };
        node_ids.reverse(); // reverse since we will pop elements from the end in implementation
        // of Iterator
        Self {
            node_ids,
            nodes: iter.nodes(),
        }
    }
}

impl<'a, E: NodeEgdes, const FORWARD: bool> Iterator for NodeIterator<'a, E, FORWARD> {
    type Item = (NodeId, &'a E);

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.unvisited_nodes.iter().find_map(|node_id| {
            let node = self.nodes.get(node_id).unwrap(); // safe to unwrap since this should contain only nodes in the model
            let is_node_next = if FORWARD {
                node.inputs().iter().all(|edge| {
                    edge.node.is_none()
                        || !self.unvisited_nodes.contains(edge.node.as_ref().unwrap())
                })
            } else {
                node.outputs()
                    .iter()
                    .flat_map(|output| &output.edges)
                    .all(|edge| {
                        edge.node.is_none()
                            || !self.unvisited_nodes.contains(edge.node.as_ref().unwrap())
                    })
            };
            if is_node_next {
                Some((*node_id, node))
            } else {
                None
            }
        });
        if let Some((node_id, _)) = &node {
            // remove the node from the unvisited nodes set
            self.unvisited_nodes.remove(node_id);
        }
        node
    }
}

impl<E: NodeEgdes, const FORWARD: bool> Iterator for IntoNodeIterator<E, FORWARD> {
    type Item = (NodeId, E);

    fn next(&mut self) -> Option<Self::Item> {
        if self.node_ids.is_empty() {
            None
        } else {
            let node_id = self.node_ids.pop().unwrap();
            let node = self
                .nodes
                .remove(&node_id)
                .expect(format!("Node {node_id} not found").as_str());
            Some((node_id, node))
        }
    }
}
