//! This module implements a prover instance that generates proofs completely
//! locally, in a one-shot manner. After a successful proof generation, they are
//! written to a local file.
use anyhow::Context;
use deep_prove::store::MemStore;
use memmap2::Mmap;
use std::io::Write;
use tracing::info;
use zkml::{ModelType, inputs::Input, quantization::ScalingStrategyKind};

use crate::RunMode;

/// Run the prover once, directly feeding it the required inputs. The proofs are
/// written to a file.
pub async fn run(args: RunMode) -> anyhow::Result<()> {
    let RunMode::Local { onnx, inputs } = args else {
        unreachable!()
    };

    crate::setup_logging(false);

    let input = Input::from_file(&inputs).context("loading input")?;
    let model_file = std::fs::File::open(&onnx).context("opening model file")?;
    let model = unsafe { Mmap::map(&model_file) }
        .context("mmap-ing model file")?
        .to_vec();

    let proto = {
        use prost_tract_compat::Message;
        tract_onnx::pb::ModelProto::decode(&*model).context("decoding ModelProto")?
    };
    let model_type = onnx
        .extension()
        .and_then(|ext| match ext.to_ascii_lowercase().to_str() {
            Some("cnn") => Some(ModelType::CNN),
            Some("mlp") => Some(ModelType::MLP),
            _ => None,
        });
    if let Some(model_type) = model_type {
        model_type.validate_proto(&proto)?;
    }

    let scaling_strategy = ScalingStrategyKind::AbsoluteMax;
    let scaling_input_hash = None;

    let request = crate::DeepProveRequestV1 {
        model,
        input,
        scaling_strategy,
        scaling_input_hash,
    };
    let proofs = crate::run_model_v1(request, &mut MemStore::default()).await?;

    // create a file to write the proofs to
    let mut file = tempfile::Builder::new()
        .prefix("proof-")
        .suffix(".json")
        .rand_bytes(10)
        .disable_cleanup(true)
        .tempfile_in(std::env::current_dir().unwrap_or("./".into()))?;
    file.write_all(serde_json::to_string_pretty(&proofs)?.as_bytes())?;

    info!("Successfully generated {} proofs", proofs.len());

    Ok(())
}
