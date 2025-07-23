use std::{io::Write, path::PathBuf, str::FromStr};

use alloy::signers::local::LocalSigner;
use anyhow::Context;
use clap::{Parser, Subcommand};
use deep_prove::middleware::{
    DeepProveRequest, DeepProveResponse,
    v1::{DeepProveRequest as DeepProveRequestV1, Input},
};
use lagrange::ProofChannelResponse;
use memmap2::Mmap;
use tokio::fs::File;
use tonic::{metadata::MetadataValue, transport::ClientTlsConfig};
use tracing::{error, info, level_filters::LevelFilter};
use tracing_subscriber::EnvFilter;
use ureq::http::status::StatusCode;
use url::Url;
use zkml::{ModelType, quantization::ScalingStrategyKind};

mod lagrange {
    tonic::include_proto!("lagrange");
}

#[derive(Parser)]
#[command(version, about)]
struct Args {
    #[command(subcommand)]
    executor: Executor,
}

#[derive(Subcommand)]
enum Executor {
    /// Interact with a LPN gateway.
    Lpn {
        /// The URL of the LPN gateway.
        #[clap(short, long, env)]
        gw_url: Url,

        /// The client ETH private key.
        #[clap(short, long, env)]
        private_key: String,

        /// Max message size passed through gRPC (in MBytes).
        #[arg(long, default_value = "100")]
        max_message_size: usize,

        /// Timeout for the task in seconds.
        #[arg(long, default_value = "3600")]
        timeout: u64,

        #[command(subcommand)]
        command: Command,
    },

    /// Interact with the API exposed by a prover.
    LocalApi {
        /// The root URL of the worker
        #[arg(short, long, env, default_value = "http://localhost:8080")]
        worker_url: String,

        #[command(subcommand)]
        command: Command,
    },
}

#[derive(Subcommand)]
enum Command {
    /// Submit a model and its input to prove inference.
    Submit {
        /// Path to the ONNX file of the model to prove.
        #[arg(short = 'm', long)]
        onnx: PathBuf,

        /// Path to the inputs to the model to prove inference for.
        #[arg(short, long)]
        inputs: PathBuf,
    },

    /// Fetch a generated proof, if any are available.
    Fetch {},
}

async fn connect_to_lpn(gw_config: Executor) -> anyhow::Result<()> {
    let Executor::Lpn {
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
            let model = unsafe { Mmap::map(&model_file) }
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

async fn connect_to_prover(executor: Executor) -> anyhow::Result<()> {
    let Executor::LocalApi {
        worker_url,
        command,
    } = executor
    else {
        unreachable!()
    };

    let root_url = Url::parse(&worker_url).context("parsing worker URL")?;

    match command {
        Command::Submit { onnx, inputs } => {
            let input = Input::from_file(&inputs).context("loading input")?;
            let model_file = std::fs::File::open(&onnx).context("opening model file")?;
            let model = unsafe { Mmap::map(&model_file) }
                .context("mmap-ing model file")?
                .to_vec();
            let proto_model = {
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
                model_type.validate_proto(&proto_model)?;
            }
            let scaling_strategy = ScalingStrategyKind::AbsoluteMax;
            let scaling_input_hash = None;

            let request = DeepProveRequestV1 {
                model,
                input,
                scaling_strategy,
                scaling_input_hash,
            };

            // build the API endpoint and send the whole thing
            let mut resp = ureq::post(root_url.join("/proofs")?.as_str())
                .send_json(request)
                .context("sending proof request to the worker")?;
            match resp.status() {
                StatusCode::CREATED => {
                    info!("{}", resp.body_mut().read_to_string()?);
                }
                c => {
                    error!(
                        "failed to send request: [{}] {}",
                        c.as_str(),
                        resp.body_mut().read_to_string()?
                    );
                }
            }
        }
        Command::Fetch {} => {
            // Build the endpoint URL
            let mut resp = ureq::get(root_url.join("/proofs")?.as_str()).call()?;

            match resp.status() {
                StatusCode::OK => {
                    // create a file to write the proofs to
                    let mut file = tempfile::Builder::new()
                        .prefix("proof-")
                        .suffix(".json")
                        .rand_bytes(10)
                        .disable_cleanup(true)
                        .tempfile_in(std::env::current_dir().unwrap_or("./".into()))?;

                    // save the list of proofs
                    let body = resp
                        .body_mut()
                        .with_config()
                        .limit(1000 * 1024 * 1024)
                        .read_to_vec()?;
                    file.write_all(&body)?;
                    info!("proof received, saved to {}", file.path().display());
                }
                StatusCode::NO_CONTENT => {
                    info!("no proof ready yet");
                }
                c => {
                    // these status codes should never be produced by the worker
                    error!("unknown status: {}", c.as_str())
                }
            }
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let subscriber = tracing_subscriber::fmt()
        .pretty()
        .compact()
        .with_level(true)
        .with_file(false)
        .with_line_number(false)
        .with_target(false)
        .without_time()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .finish();
    tracing::subscriber::set_global_default(subscriber).context("Setting up logging failed")?;

    let args = Args::parse();

    match args.executor {
        gw_config @ Executor::Lpn { .. } => connect_to_lpn(gw_config).await,
        local_config @ Executor::LocalApi { .. } => connect_to_prover(local_config).await,
    }
}
