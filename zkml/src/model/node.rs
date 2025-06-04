use serde::{Deserialize, Serialize};

/// Represents a link between an input/output wire of a node with an input/output wire of
/// another node.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Edge {
    // Reference to the node linked to this wire, will be `None` if the wire is an input or
    // output of the model
    pub(crate) node: Option<NodeId>,
    // The index of the wire of `node` which is linked to this wire
    pub(crate) index: usize,
}

impl Edge {
    pub fn new(node: NodeId, index: usize) -> Self {
        Self {
            node: Some(node),
            index,
        }
    }

    /// Edge when the node is an input or an output of the model
    pub fn new_at_edge(index: usize) -> Self {
        Self { node: None, index }
    }
}

/// Represents all the edges that are connected to a node's output wire
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct OutputWire {
    // needs to be a vector because the output of a node can be used as input to multiple nodes
    pub(crate) edges: Vec<Edge>,
}

/// Represents a node in a model
#[derive(Clone, Debug)]
pub struct Node<N> {
    pub(crate) inputs: Vec<Edge>,
    pub(crate) outputs: Vec<OutputWire>,
    pub(crate) operation: Layer<N>,
}

pub trait NodeEgdes {
    // Get input edges for a node
    fn inputs(&self) -> &[Edge];
    // Get output edges of a node
    fn outputs(&self) -> &[OutputWire];
}

impl<N> NodeEgdes for Node<N> {
    fn inputs(&self) -> &[Edge] {
        &self.inputs
    }

    fn outputs(&self) -> &[OutputWire] {
        &self.outputs
    }
}

impl<E: ExtensionField + DeserializeOwned> NodeEgdes for NodeCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    fn inputs(&self) -> &[Edge] {
        &self.inputs
    }

    fn outputs(&self) -> &[OutputWire] {
        &self.outputs
    }
}

impl<N: Number> Node<N> {
    // Create a new node, from the set of inputs edges and the operation performed by the node
    pub fn new(inputs: Vec<Edge>, operation: Layer<N>) -> Self {
        let num_outputs = operation.num_outputs(inputs.len());
        Self::new_with_outputs(inputs, operation, vec![Default::default(); num_outputs])
    }

    pub(crate) fn new_with_outputs(
        inputs: Vec<Edge>,
        operation: Layer<N>,
        outputs: Vec<OutputWire>,
    ) -> Self {
        Self {
            inputs,
            outputs,
            operation,
        }
    }
}

/// Represents the proving context for a given node, altogether with the input
/// and output edges of the node
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct NodeCtx<E>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    pub(crate) inputs: Vec<Edge>,
    pub(crate) outputs: Vec<OutputWire>,
    pub(crate) ctx: LayerCtx<E>,
}

impl<E> NodeCtx<E>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    /// Get the claims corresponding to the output edges of a node.
    /// Requires the input claims for the nodes of the model using the
    /// outputs of the current node, and the claims of the output
    /// tensors of the model
    pub(crate) fn claims_for_node<'a, 'b>(
        &self,
        claims_by_node: &'a HashMap<NodeId, Vec<Claim<E>>>,
        output_claims: &'b [Claim<E>],
    ) -> Result<Vec<&'a Claim<E>>>
    where
        'b: 'a,
    {
        self.outputs.iter().map(|out| {
            // For now, we support in proving only one edge per output wire,
            // as if an output is used as input in different nodes, we need
            // to batch claims about the same polynomial. ToDo: batch claims
            assert_eq!(out.edges.len(), 1);
            let edge = &out.edges[0];
            Ok(if let Some(id) = &edge.node {
                let claims_for_node = claims_by_node.get(id).ok_or(
                    anyhow!("No claims found for layer {}", id)
                )?;
                ensure!(edge.index < claims_for_node.len(),
                    "Not enough claims found for node {}: required claim for input {}, but {} claims found",
                    id,
                    edge.index,
                    claims_for_node.len()
                );
                &claims_for_node[edge.index]
            } else {
                // it's an output node, so we use directly the claim for the corresponding output
                ensure!(edge.index < output_claims.len(),
                 "Required claim for output {} of the model, but only {} output claims found",
                 edge.index,
                 output_claims.len(),
                );
                &output_claims[edge.index]
            })
        }).collect()
    }

    /// Get the claims corresponding to the input tensors of the model.
    /// Requires as inputs the contexts for all the nodes in the model
    /// and the set of claims for the input tensors of all the nodes of
    /// the model
    pub(crate) fn input_claims<'a, I: Iterator<Item = (&'a NodeId, &'a Self)>>(
        nodes: I,
        claims_by_node: &HashMap<NodeId, Vec<Claim<E>>>,
    ) -> Result<Vec<&Claim<E>>> {
        let mut claims = BTreeMap::new();
        for (node_id, ctx) in nodes {
            for (i, edge) in ctx.inputs.iter().enumerate() {
                if edge.node.is_none() {
                    let claims_for_node = claims_by_node
                        .get(node_id)
                        .ok_or(anyhow!("Claim not found for node {}", node_id))?;
                    claims.insert(edge.index, &claims_for_node[i]);
                }
            }
        }
        ensure!(
            !claims.is_empty(),
            "No input claims found for the set of nodes provided"
        );
        let min_index = claims.first_key_value().unwrap().0;
        let max_index = claims.last_key_value().unwrap().0;
        ensure!(
            *min_index == 0 && *max_index == claims.len() - 1,
            "Not all input claims were found"
        );

        Ok(claims.into_values().collect())
    }
}
