use goldilocks::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams};
use serde::{Deserialize, Serialize};

use super::{Element, Model, ModelMetadata, ProofG};

/// A type of the proof for the `v1` of the protocol
pub type Proof = ProofG<GoldilocksExt2, Basefold<GoldilocksExt2, BasefoldRSParams>>;

/// Inputs to the model
#[derive(Clone, Serialize, Deserialize)]
pub struct Input {
    input_data: Vec<Vec<f32>>,
    output_data: Vec<Vec<f32>>,
    pytorch_output: Vec<Vec<f32>>,
}

// TODO: this is a copypaste from `bench.rs`.
impl Input {
    pub fn filter(&self, indices: Option<&Vec<usize>>) -> Self {
        if let Some(indices) = indices {
            assert!(
                indices.iter().all(|i| *i < self.input_data.len()),
                "Index {} is out of range (max: {})",
                indices.iter().max().unwrap(),
                self.input_data.len() - 1
            );
            let input_data = indices
                .iter()
                .map(|i| self.input_data[*i].clone())
                .collect();
            let output_data = indices
                .iter()
                .map(|i| self.output_data[*i].clone())
                .collect();
            let pytorch_output = indices
                .iter()
                .map(|i| self.pytorch_output[*i].clone())
                .collect();
            Self {
                input_data,
                output_data,
                pytorch_output,
            }
        } else {
            self.clone()
        }
    }

    pub fn to_elements(self, md: &ModelMetadata) -> (Vec<Vec<Element>>, Vec<Vec<Element>>) {
        let input_sf = md.input.first().unwrap();
        let inputs = self
            .input_data
            .into_iter()
            .map(|input| input.into_iter().map(|e| input_sf.quantize(&e)).collect())
            .collect();
        let output_sf = *md.output_scaling_factor().first().unwrap();
        let outputs = self
            .output_data
            .into_iter()
            .map(|output| output.into_iter().map(|e| output_sf.quantize(&e)).collect())
            .collect();
        (inputs, outputs)
    }
}

/// The `v1` proving request
#[derive(Serialize, Deserialize)]
pub struct DeepProveRequest {
    /// The model
    pub model: Model<Element>,

    /// Model metadata
    pub model_metadata: ModelMetadata,

    /// An array of inputs to run proving for
    pub input: Input,

    /// Filter inputs for these indices only
    // TODO: is this needed?
    pub run_indices: Option<Vec<usize>>,
}

/// The `v1` proofs that have been computed by the worker
#[derive(Serialize, Deserialize)]
pub struct DeepProveResponse {
    pub proofs: Vec<Proof>,
}
