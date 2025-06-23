//! Metadata related information for a model. These are the information derived from the
//! float based model weights and activations.
use std::collections::{BTreeMap, HashMap};

use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Serialize};

use crate::{
    Element,
    layers::provable::{Edge, Node, NodeId},
    model::Model,
};

use super::ScalingFactor;

/// Structure holding the scaling factors of the input and output of each layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub input: Vec<ScalingFactor>,
    pub(crate) input_layers_scaling: HashMap<NodeId, Vec<ScalingFactor>>,
    pub(crate) output_layers_scaling: HashMap<NodeId, Vec<ScalingFactor>>,
    pub(crate) output: Vec<ScalingFactor>,
    pub float_model: Option<Model<f32>>,
}

impl ModelMetadata {
    pub fn output_scaling_factor(&self) -> Vec<ScalingFactor> {
        self.output.clone()
    }

    pub fn layer_output_scaling_factor(&self, node_id: NodeId) -> &[ScalingFactor] {
        self.output_layers_scaling
            .get(&node_id)
            .unwrap_or_else(|| panic!("Node {node_id} not found"))
    }

    pub fn layer_input_scaling_factor(&self, node_id: NodeId) -> &[ScalingFactor] {
        self.input_layers_scaling
            .get(&node_id)
            .unwrap_or_else(|| panic!("Node {node_id} not found"))
    }
}

pub(crate) struct MetadataBuilder {
    pub(crate) input_scaling: Vec<ScalingFactor>,
    output_layers_scaling: HashMap<NodeId, Vec<ScalingFactor>>,
    input_layers_scaling: HashMap<NodeId, Vec<ScalingFactor>>,
}

impl MetadataBuilder {
    pub fn new(input_scaling: Vec<ScalingFactor>) -> Self {
        Self {
            input_scaling,
            output_layers_scaling: HashMap::new(),
            input_layers_scaling: HashMap::new(),
        }
    }

    pub fn set_layers_scaling(
        &mut self,
        node_id: NodeId,
        output_scaling: Vec<ScalingFactor>,
        input_scaling: Vec<ScalingFactor>,
    ) {
        self.output_layers_scaling.insert(node_id, output_scaling);
        self.input_layers_scaling.insert(node_id, input_scaling);
    }

    pub(crate) fn compute_input_scaling(&self, node_inputs: &[Edge]) -> Result<Vec<ScalingFactor>> {
        node_inputs.iter().map(|edge| {
                if let Some(n) = &edge.node {
                    let scalings = self.get_output_layer_scaling(n).ok_or(
                        anyhow!("Scaling factors for node {n} not found")
                    )?;
                    ensure!(edge.index < scalings.len(),
                        "Getting scaling factor {} for node {n}, but there are only {} scaling factors",
                        edge.index,
                        scalings.len(),
                    );
                    Ok(scalings[edge.index])
                } else {
                    ensure!(edge.index < self.input_scaling.len(),
                        "Getting scaling factor {} for model inputs, but there are only {} scaling factors",
                        edge.index,
                        self.input_scaling.len(),
                    );
                    Ok(self.input_scaling[edge.index])
                }
            }).collect()
    }

    pub(crate) fn get_output_layer_scaling(&self, node_id: &NodeId) -> Option<&[ScalingFactor]> {
        self.output_layers_scaling
            .get(node_id)
            .map(|s| s.as_slice())
    }

    pub fn build(self, output_nodes: Vec<(NodeId, &Node<Element>)>) -> Result<ModelMetadata> {
        let mut output_scalings = BTreeMap::new();
        for (id, node) in output_nodes.into_iter() {
            let scalings = self
                .get_output_layer_scaling(&id)
                .ok_or(anyhow!("Scaling factors not found for node {id}"))?;
            ensure!(
                scalings.len() == node.outputs.len(),
                "Number of scalings factors found for node {id} is different from
                the expected number of outputs of the node"
            );
            node.outputs.iter().enumerate().try_for_each(|(i, out)| {
                if let Some(out_index) = out.edges.iter().find_map(|edge| {
                    if edge.node.is_none() {
                        Some(edge.index)
                    } else {
                        None
                    }
                }) {
                    ensure!(
                        output_scalings.insert(out_index, scalings[i]).is_none(),
                        "Scaling factor for output {out_index} found twice"
                    );
                }
                Ok(())
            })?;
        }
        // check that all scaling factors have been found
        ensure!(
            !output_scalings.is_empty(),
            "No output scaling factors found"
        );
        ensure!(
            *output_scalings.first_key_value().unwrap().0 == 0
                && *output_scalings.last_key_value().unwrap().0 == output_scalings.len() - 1,
            "Not all output scaling factors found"
        );

        Ok(ModelMetadata {
            input: self.input_scaling,
            output_layers_scaling: self.output_layers_scaling,
            input_layers_scaling: self.input_layers_scaling,
            output: output_scalings.into_values().collect(),
            float_model: None,
        })
    }
}
