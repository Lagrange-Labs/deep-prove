use ff_ext::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams, Hasher};
use serde::{Deserialize, Serialize};
pub use zkml::inputs::Input;
use zkml::{Proof as ProofG, quantization::ScalingStrategyKind};

/// A type of the proof for the `v1` of the protocol
pub type Proof = ProofG<GoldilocksExt2, Basefold<GoldilocksExt2, BasefoldRSParams<Hasher>>>;

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
