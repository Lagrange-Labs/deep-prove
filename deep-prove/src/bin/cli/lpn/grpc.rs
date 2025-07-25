use std::{io::Write, str::FromStr};

use crate::{Command, Executor};
use alloy::signers::local::LocalSigner;
use anyhow::{Context, bail};
use deep_prove::middleware::{
    DeepProveRequest, DeepProveResponse,
    v1::{self, DeepProveRequest as DeepProveRequestV1, Input},
};
use lagrange::ProofChannelResponse;
use tokio::fs::File;
use tonic::{metadata::MetadataValue, transport::ClientTlsConfig};
use tracing::info;
use zkml::{ModelType, quantization::ScalingStrategyKind};

mod lagrange {
    tonic::include_proto!("lagrange");
}

pub async fn connect(gw_config: Executor) -> anyhow::Result<()> {
    let Executor::LpnGrpc {
        gw_url,
        private_key,
        max_message_size,
        timeout,
        command,
    } = gw_config
    else {
        unreachable!()
    };

    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");

    let channel = tonic::transport::Channel::builder(gw_url.as_str().parse()?)
        .tls_config(ClientTlsConfig::new().with_enabled_roots())?
        .connect()
        .await
        .with_context(|| format!("connecting to the GW at {gw_url}"))?;

    let wallet = LocalSigner::from_str(&private_key)?;
    let client_id: MetadataValue<_> = wallet
        .address()
        .to_string()
        .parse()
        .context("parsing client ID")?;
    let max_message_size = max_message_size * 1024 * 1024;
    let mut client = lagrange::clients_service_client::ClientsServiceClient::with_interceptor(
        channel,
        move |mut req: tonic::Request<()>| {
            req.metadata_mut().insert("client_id", client_id.clone());
            Ok(req)
        },
    )
    .max_encoding_message_size(max_message_size)
    .max_decoding_message_size(max_message_size);

    info!("Connection to Gateway established");

    match command {
        Command::Submit { onnx, inputs } => {
            let input = Input::from_file(&inputs).context("loading input")?;
            let model_file = File::open(&onnx).await.context("opening model file")?;
            let model = unsafe { memmap2::Mmap::map(&model_file) }
                .context("loading model file")?
                .to_vec();

            let proto = {
                use prost_tract_compat::Message;

                tract_onnx::pb::ModelProto::decode(&*model).context("decoding ModelProto")?
            };
            let model_type =
                onnx.extension()
                    .and_then(|ext| match ext.to_ascii_lowercase().to_str() {
                        Some("cnn") => Some(ModelType::CNN),
                        Some("mlp") => Some(ModelType::MLP),
                        _ => None,
                    });
            if let Some(model_type) = model_type {
                model_type.validate_proto(&proto)?;
            }

            // TODO Currently hard-coded in the ONNX loader. Adjust when choice is available
            let scaling_strategy = ScalingStrategyKind::AbsoluteMax;
            // TODO Currently hard-coded in the ONNX loader. Adjust when choice is available
            let scaling_input_hash = None;

            let task = tonic::Request::new(lagrange::SubmitTaskRequest {
                task_bytes: zstd::encode_all(
                    rmp_serde::to_vec(&DeepProveRequest::V1(DeepProveRequestV1 {
                        model,
                        input,
                        scaling_strategy,
                        scaling_input_hash,
                    }))
                    .context("serializing inference request")?
                    .as_slice(),
                    5,
                )
                .context("compressing payload")?,
                user_task_id: format!(
                    "{}-{}-{}",
                    onnx.with_extension("")
                        .file_name()
                        .and_then(|x| x.to_str())
                        .context("invalid ONNX file name")?,
                    inputs
                        .with_extension("")
                        .file_name()
                        .and_then(|x| x.to_str())
                        .context("invalid input file name")?,
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .expect("no time travel here")
                        .as_secs()
                ),
                timeout: Some(
                    prost_wkt_types::Duration::try_from(std::time::Duration::from_secs(timeout))
                        .unwrap(),
                ),
                price_requested: 12_u64.to_le_bytes().to_vec(), // TODO:
                stake_requested: vec![0u8; 32],                 // TODO:
                class: vec!["deep-prove".to_string()],          // TODO:
                priority: 0,
            });
            let response = client.submit_task(task).await?;
            info!("got the response {response:?}");
        }
        Command::Fetch { filename } => {
            let (proof_channel_tx, proof_channel_rx) = tokio::sync::mpsc::channel(1024);

            let proof_channel_rx = tokio_stream::wrappers::ReceiverStream::new(proof_channel_rx);
            let channel = client
                .proof_channel(tonic::Request::new(proof_channel_rx))
                .await
                .unwrap();
            let mut proof_response_stream = channel.into_inner();

            info!("Fetching ready proofs...");
            let mut acked_messages = Vec::new();
            while let Some(response) = proof_response_stream.message().await? {
                let ProofChannelResponse { response } = response;

                let lagrange::proof_channel_response::Response::Proof(v) = response.unwrap();

                let lagrange::ProofReady {
                    task_id,
                    task_output,
                } = v;

                let task_id = task_id.unwrap();
                let task_output: DeepProveResponse = rmp_serde::from_slice(&task_output)?;
                match task_output {
                    DeepProveResponse::V1(v1::DeepProveResponse { proofs }) => {
                        let uuid = uuid::Uuid::from_slice(&task_id.id).unwrap_or_default();
                        info!("Received proof for task {uuid}",);
                        for (i, proof) in proofs.iter().enumerate() {
                            std::fs::File::create(
                                filename
                                    .as_ref()
                                    .map(|f| format!("{f}-{uuid}-{i}."))
                                    .unwrap_or_else(|| format!("{uuid}-{i}.bin")),
                            )
                            .context("failed to create proof file")?
                            .write_all(serde_json::to_string(proof).unwrap().as_bytes())
                            .context("failed to write proof")?;
                        }
                    }
                }

                acked_messages.push(task_id);
            }

            proof_channel_tx
                .send(lagrange::ProofChannelRequest {
                    request: Some(lagrange::proof_channel_request::Request::AckedMessages(
                        lagrange::AckedMessages { acked_messages },
                    )),
                })
                .await?;
        }
        Command::Request { .. } => bail!("`request` is not supported"),
        Command::Cancel { .. } => bail!("`cancel` is not supported"),
    }

    Ok(())
}
