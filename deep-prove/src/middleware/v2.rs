use anyhow::Context;
use base64::{Engine, prelude::BASE64_STANDARD};
use ff_ext::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams, Hasher};
use serde::{Deserialize, Serialize};
use zkml::{Proof as ZkmlProof, inputs::Input, quantization::ScalingStrategyKind};

/// A type of the proof for the `v2` of the protocol
pub type Proof = ZkmlProof<GoldilocksExt2, Basefold<GoldilocksExt2, BasefoldRSParams<Hasher>>>;

#[derive(Serialize, Deserialize)]
pub struct ClientToGw {
    /// The user-facing name of the submitted task.
    pub pretty_name: String,

    /// The ID of the model to use.
    pub model_id: i32,

    /// An array of inputs to run proving for
    pub input: Input,
}

#[derive(Serialize, Deserialize)]
pub struct GwToWorker {
    /// The job ID to use when communicating with the gateway.
    pub job_id: i64,

    /// The base64-encoded model - tests on random binary data show that base64
    /// encoding is 30% the size of classic array-of-bytes serde_json encoding.
    pub model: String,

    /// An array of inputs to run proving for
    pub input: Input,
}
impl TryFrom<GwToWorker> for super::v1::DeepProveRequest {
    type Error = anyhow::Error;

    fn try_from(r: GwToWorker) -> anyhow::Result<Self> {
        Ok(Self {
            model: BASE64_STANDARD
                .decode(r.model)
                .context("failed to base64-decode the model")?,
            input: r.input,
            scaling_strategy: ScalingStrategyKind::AbsoluteMax,
            scaling_input_hash: None,
        })
    }
}
