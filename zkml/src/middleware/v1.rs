use std::{io::BufReader, path::Path};

use anyhow::{Context, ensure};
use ff_ext::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams, Hasher};
use serde::{Deserialize, Serialize};

use crate::quantization::{QUANTIZATION_RANGE, ScalingStrategyKind};

use super::{Element, ModelMetadata, ProofG};

/// A type of the proof for the `v1` of the protocol
pub type Proof = ProofG<GoldilocksExt2, Basefold<GoldilocksExt2, BasefoldRSParams<Hasher>>>;

/// Inputs to the model
#[derive(Clone, Serialize, Deserialize)]
pub struct Input {
    input_data: Vec<Vec<f32>>,
}

impl Input {
    pub fn from_file<P: AsRef<Path>>(p: P) -> anyhow::Result<Self> {
        let inputs: Self = serde_json::from_reader(BufReader::new(
            std::fs::File::open(p.as_ref()).context("opening inputs file")?,
        ))
        .context("deserializing inputs")?;
        inputs.validate()?;

        Ok(inputs)
    }

    pub fn from_str<S: AsRef<str>>(s: S) -> anyhow::Result<Self> {
        let inputs: Self = serde_json::from_str(s.as_ref()).context("deserializing inputs")?;
        inputs.validate()?;

        Ok(inputs)
    }

    pub fn from_reader<R: std::io::Read>(r: R) -> anyhow::Result<Self> {
        let inputs: Self = serde_json::from_reader(r).context("deserializing inputs")?;
        inputs.validate()?;

        Ok(inputs)
    }

    fn validate(&self) -> anyhow::Result<()> {
        ensure!(self.input_data.len() > 0);
        ensure!(
            self.input_data
                .iter()
                .all(|v| v.iter().all(|&x| QUANTIZATION_RANGE.contains(&x))),
            "can only support real model so far (input at least)"
        );
        Ok(())
    }

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
            Self { input_data }
        } else {
            self.clone()
        }
    }

    pub fn to_elements(self, md: &ModelMetadata) -> Vec<Vec<Element>> {
        let input_sf = md.input.first().unwrap();

        self.input_data
            .into_iter()
            .map(|input| input.into_iter().map(|e| input_sf.quantize(&e)).collect())
            .collect()
    }
}

/// The `v1` proving request
#[derive(Serialize, Deserialize)]
pub struct DeepProveRequest {
    /// The model
    pub model: Vec<u8>,

    /// An array of inputs to run proving for
    pub input: Input,

    /// Model scaling strategy
    pub scaling_strategy: ScalingStrategyKind,

    /// A hash of model scaling strategy input, if any
    pub scaling_input_hash: Option<String>,
}

/// The `v1` proofs that have been computed by the worker
#[derive(Serialize, Deserialize)]
pub struct DeepProveResponse {
    pub proofs: Vec<Proof>,
}
