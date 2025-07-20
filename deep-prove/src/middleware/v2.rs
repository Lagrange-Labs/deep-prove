use ff_ext::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams, Hasher};
use serde::{Deserialize, Serialize};
use zkml::{Proof as ZkmlProof, inputs::Input};

/// A type of the proof for the `v2` of the protocol
pub type Proof = ZkmlProof<GoldilocksExt2, Basefold<GoldilocksExt2, BasefoldRSParams<Hasher>>>;

/// The `v2` proving request
#[derive(Serialize, Deserialize)]
pub struct DeepProveRequest {
    /// The user-facing name of the submitted task.
    pub pretty_name: String,

    /// The ID of the model to use.
    pub model_id: i32,

    /// An array of inputs to run proving for
    pub input: Input,
}

/// The `v2` proofs that have been computed by the worker
#[derive(Serialize, Deserialize)]
pub struct DeepProveResponse {
    pub proofs: Vec<Proof>,
}
