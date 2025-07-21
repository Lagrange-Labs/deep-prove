use std::collections::{BTreeMap, BTreeSet, HashMap};

use ff_ext::ExtensionField;
use serde::{Serialize, de::DeserializeOwned};

use crate::layers::provable::{Node, NodeCtx, NodeEdges, NodeId};

use super::{Model, ModelCtx};

pub trait ToIterator<E: NodeEdges> {
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
pub type ModelForwardIterator<'a, N> = NodeIterator<'a, Node<N>, true>;
// Backward iterator for the nodes in a model. This is useful for traversing the model when
// proving
pub type ModelBackwardIterator<'a, N> = NodeIterator<'a, Node<N>, false>;

impl<E: ExtensionField + DeserializeOwned> ToIterator<NodeCtx<E>> for ModelCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    fn to_forward_iterator<'a>(&'a self) -> ModelCtxForwardIterator<'a, E> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().copied().collect(),
            nodes: BTreeMapNodeSet(&self.nodes).to_node_set(),
        }
    }

    fn to_backward_iterator<'a>(&'a self) -> ModelCtxBackwardIterator<'a, E> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().copied().collect(),
            nodes: BTreeMapNodeSet(&self.nodes).to_node_set(),
        }
    }
}

impl<N> ToIterator<Node<N>> for Model<N> {
    fn to_forward_iterator<'a>(&'a self) -> ModelForwardIterator<'a, N> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().copied().collect(),
            nodes: HashMapNodeSet(&self.nodes).to_node_set(),
        }
    }

    fn to_backward_iterator<'a>(&'a self) -> ModelBackwardIterator<'a, N> {
        NodeIterator {
            unvisited_nodes: self.nodes.keys().copied().collect(),
            nodes: HashMapNodeSet(&self.nodes).to_node_set(),
        }
    }
}

/// A trait representing a collections of nodes with their input/output edges
pub trait NodeCollection<E: NodeEdges> {
    fn nodes(self) -> HashMap<NodeId, E>;
}

impl<N> NodeCollection<Node<N>> for Model<N> {
    fn nodes(self) -> HashMap<NodeId, Node<N>> {
        self.nodes
    }
}

/// Forward iterator for the proving contexts of nodes in a model
pub type ModelCtxForwardIterator<'a, E> = NodeIterator<'a, NodeCtx<E>, true>;
/// Backward iterator for the proving contexts of nodes in a model
pub type ModelCtxBackwardIterator<'a, E> = NodeIterator<'a, NodeCtx<E>, false>;

pub trait ToNodeSet<'a, E: NodeEdges> {
    fn to_node_set(self) -> HashMap<NodeId, &'a E>;
}

struct HashMapNodeSet<'a, E>(&'a HashMap<NodeId, E>);

impl<'a, E: NodeEdges> ToNodeSet<'a, E> for HashMapNodeSet<'a, E> {
    fn to_node_set(self) -> HashMap<NodeId, &'a E> {
        self.0.iter().map(|(node_id, e)| (*node_id, e)).collect()
    }
}

struct BTreeMapNodeSet<'a, E>(&'a BTreeMap<NodeId, E>);

impl<'a, E: NodeEdges> ToNodeSet<'a, E> for BTreeMapNodeSet<'a, E> {
    fn to_node_set(self) -> HashMap<NodeId, &'a E> {
        self.0.iter().map(|(node_id, e)| (*node_id, e)).collect()
    }
}

/// Structure employed to implemented forward and backward iterators
pub struct NodeIterator<'a, E: NodeEdges, const FORWARD: bool> {
    pub(crate) unvisited_nodes: BTreeSet<NodeId>, /* Use BTreeSet to make the iterator deterministic */
    pub(crate) nodes: HashMap<NodeId, &'a E>,
}

/// Structure employed to implement forward and backward iterators that
/// take ownership of the set of nodes
pub struct IntoNodeIterator<E: NodeEdges, const FORWARD: bool> {
    pub(crate) node_ids: Vec<NodeId>,
    pub(crate) nodes: HashMap<NodeId, E>,
}

impl<E: NodeEdges, const FORWARD: bool> IntoNodeIterator<E, FORWARD> {
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

impl<'a, E: NodeEdges, const FORWARD: bool> Iterator for NodeIterator<'a, E, FORWARD> {
    type Item = (NodeId, &'a E);

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.unvisited_nodes.iter().find_map(|node_id| {
            let node = *self.nodes.get(node_id).unwrap(); // safe to unwrap since this should contain only nodes in the model
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

impl<E: NodeEdges, const FORWARD: bool> Iterator for IntoNodeIterator<E, FORWARD> {
    type Item = (NodeId, E);

    fn next(&mut self) -> Option<Self::Item> {
        if self.node_ids.is_empty() {
            None
        } else {
            let node_id = self.node_ids.pop().unwrap();
            let node = self
                .nodes
                .remove(&node_id)
                .unwrap_or_else(|| panic!("Node {node_id} not found"));
            Some((node_id, node))
        }
    }
}
