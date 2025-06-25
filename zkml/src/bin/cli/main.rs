use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};
use lagrange::ProofChannelResponse;
use tonic::{metadata::MetadataValue, transport::ClientTlsConfig};
use tracing::info;
use url::Url;
use zkml::{
    Element, FloatOnnxLoader,
    middleware::{DeepProveRequest, DeepProveResponse, v1::Input},
    model::Model,
    quantization::{AbsoluteMax, ModelMetadata},
};

mod lagrange {
    tonic::include_proto!("lagrange");
}

#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// The URL of the Gateway to the proving network to connect to
    #[clap(short, long, env)]
    gw_url: Url,

    /// The Client identity.
    #[clap(short, long, env)]
    client_id: String,

    /// Max message size passed through gRPC (in MBytes)
    #[arg(long, default_value = "100")]
    max_message_size: usize,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Submit a model and its input to prove inference.
    Submit {
        /// Path to the ONNX file of the model to prove.
        #[arg(short, long)]
        onnx: PathBuf,
        /// Path to the inputs to the model to prove inference for.
        #[arg(short, long)]
        inputs: PathBuf,
    },

    /// Fetch the generated proofs, if any.
    Fetch {},
}

fn parse_model<P: AsRef<Path>>(p: P) -> anyhow::Result<(Model<Element>, ModelMetadata)> {
    let strategy = AbsoluteMax::new();
    FloatOnnxLoader::new_with_scaling_strategy(
        p.as_ref()
            .as_os_str()
            .to_str()
            .context("failed to convert path to string")?,
        strategy,
    )
    .with_keep_float(true)
    .build()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");

    let channel = tonic::transport::Channel::builder(args.gw_url.as_str().parse()?)
        .tls_config(ClientTlsConfig::new().with_enabled_roots())?
        .connect()
        .await
        .with_context(|| format!("connecting to the GW at {}", args.gw_url))?;

    let client_id: MetadataValue<_> = args.client_id.parse().context("parsing client ID")?;
    let max_message_size = args.max_message_size * 1024 * 1024;
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

    match args.command {
        Command::Submit { onnx, inputs } => {
            let input = Input::from_file(&inputs).context("loading input:")?;
            let (model, model_metadata) = parse_model(&onnx).context("parsing ONNX file")?;
            let task = tonic::Request::new(lagrange::SubmitTaskRequest {
                task_bytes: zstd::encode_all(
                    rmp_serde::to_vec(&DeepProveRequest::V1(
                        zkml::middleware::v1::DeepProveRequest {
                            model,
                            model_metadata,
                            input,
                        },
                    ))
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
                    prost_wkt_types::Duration::try_from(std::time::Duration::from_secs(60 * 15))
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

        Command::Fetch {} => {
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
                    DeepProveResponse::V1(_) => {
                        info!(
                            "Received proof for task {}",
                            uuid::Uuid::from_slice(&task_id.id).unwrap_or_default()
                        );
                        // TODO: write to file or whatever
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
    }
    Ok(())
}
