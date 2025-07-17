use alloy::signers::local::LocalSigner;
use anyhow::{Context as _, Result};
use axum::{Json, Router, routing::get};
use clap::{Parser, Subcommand};
use deep_prove::{
    middleware::{
        DeepProveRequest, DeepProveResponse,
        v1::{
            DeepProveRequest as DeepProveRequestV1, DeepProveResponse as DeepProveResponseV1,
            Input, Proof as ProofV1,
        },
    },
    store::{self, MemStore, S3Store, Store},
};
use ff_ext::GoldilocksExt2;
use futures::{FutureExt, StreamExt};
use lagrange::{WorkerToGwRequest, worker_to_gw_request::Request};
use memmap2::Mmap;
use mpcs::{Basefold, BasefoldRSParams, Hasher};
use reqwest::StatusCode;
use std::{net::SocketAddr, path::PathBuf, str::FromStr};
use tonic::{metadata::MetadataValue, transport::ClientTlsConfig};
use tracing::{debug, error, info, warn};
use tracing_subscriber::{EnvFilter, filter::LevelFilter, fmt::format::FmtSpan};
use zkml::{
    Context, Element, FloatOnnxLoader, ModelType, Prover, default_transcript,
    model::Model,
    quantization::{AbsoluteMax, ModelMetadata, ScalingStrategyKind},
};

use crate::lagrange::WorkerToGwResponse;

mod lagrange {
    tonic::include_proto!("lagrange");
}

type F = GoldilocksExt2;
type Pcs<E> = Basefold<E, BasefoldRSParams<Hasher>>;

async fn run_model_v1(model: DeepProveRequestV1, mut store: impl Store) -> Result<Vec<ProofV1>> {
    info!("Proving inference");
    let DeepProveRequestV1 {
        model,
        input,
        scaling_strategy,
        scaling_input_hash,
    } = model;

    let model_file_hash = {
        let hash = <sha2::Sha256 as sha2::Digest>::digest(&model);
        format!("{hash:X}")
    };

    let params_key = store::ParamsKey {
        model_file_hash: &model_file_hash,
    };
    let model_key = store::ModelKey {
        model_file_hash: &model_file_hash,
        scaling_strategy,
        scaling_input_hash: scaling_input_hash.as_deref(),
    };

    let params = store.get_params(params_key).await.context("fetching PPs")?;
    let is_stored_params = params.is_some();

    let store::ScaledModel {
        model,
        model_metadata,
    } = store
        .get_or_init_model_with(model_key, async move || {
            let (model, model_metadata) = tokio::task::spawn_blocking(move || parse_model(&model))
                .await
                .context("running parsing model task")?
                .context("parsing model")?;
            Ok(store::ScaledModel {
                model,
                model_metadata,
            })
        })
        .await
        .context("initializing model")?;

    let inputs = input.to_elements(&model_metadata);

    let mut failed_inputs = vec![];
    let (ctx, model) = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let ctx = Context::<F, Pcs<F>>::generate(
            &model,
            None,
            params.map(|store::Params { prover, verifier }| (prover, verifier)),
        )
        .context("generating model")?;
        Ok((ctx, model))
    })
    .await
    .context("running context generation task")?
    .context("generating context")?;

    if !is_stored_params {
        store
            .insert_params(
                params_key,
                store::Params {
                    prover: ctx.commitment_ctx.prover_params().clone(),
                    verifier: ctx.commitment_ctx.verifier_params().clone(),
                },
            )
            .await
            .context("storing PPs")?;
    }

    let proofs = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let mut proofs = vec![];
        for (i, input) in inputs.into_iter().enumerate() {
            debug!("Running input #{i}");
            let input_tensor = model
                .load_input_flat(vec![input])
                .context("loading flat inputs")?;

            let trace_result = model.run(&input_tensor);
            // If model.run fails, print the error and continue to the next input
            let trace = match trace_result {
                Ok(trace) => trace,
                Err(e) => {
                    error!(
                        "[!] Error running inference for input {}/{}: {}",
                        i + 1,
                        0, // num_samples,
                        e
                    );
                    failed_inputs.push(i);
                    continue; // Skip to the next input without writing to CSV
                }
            };
            let mut prover_transcript = default_transcript();
            let prover = Prover::<_, _, _>::new(&ctx, &mut prover_transcript);
            let proof = prover
                .prove(&trace)
                .with_context(|| "unable to generate proof for {i}th input")?;

            proofs.push(proof);
        }
        Ok(proofs)
    })
    .await
    .context("generating proof")?
    .context("running proof generation task")?;

    info!("Proving done.");
    Ok(proofs)
}

fn parse_model(bytes: &[u8]) -> anyhow::Result<(Model<Element>, ModelMetadata)> {
    let strategy = AbsoluteMax::new();
    FloatOnnxLoader::from_bytes_with_scaling_strategy(bytes, strategy)
        .with_keep_float(true)
        .build()
}

fn setup_logging(json: bool) {
    if json {
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_level(true)
            .with_file(true)
            .with_line_number(true)
            .with_target(true)
            .with_env_filter(
                EnvFilter::builder()
                    .with_default_directive(LevelFilter::INFO.into())
                    .from_env_lossy(),
            )
            .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
            .finish();
        tracing::subscriber::set_global_default(subscriber).expect("Setting up logging failed");
    } else {
        let subscriber = tracing_subscriber::fmt()
            .pretty()
            .compact()
            .with_level(true)
            .with_file(true)
            .with_line_number(true)
            .with_target(true)
            .with_env_filter(
                EnvFilter::builder()
                    .with_default_directive(LevelFilter::INFO.into())
                    .from_env_lossy(),
            )
            .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
            .finish();
        tracing::subscriber::set_global_default(subscriber).expect("Setting up logging failed");
    };
}

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    run_mode: RunMode,
}

#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum RunMode {
    /// Connect to a LPN gateway to receive inference tasks
    #[command(group(clap::ArgGroup::new("s3_store").multiple(true).args(&["s3_region", "s3_bucket", "s3_endpoint", "s3_access_key_id", "s3_secret_access_key"])))]
    Remote {
        #[arg(long, env, default_value = "http://localhost:10000")]
        gw_url: String,

        /// An address of the `/health` probe.
        #[arg(long, env, default_value = "127.0.0.1:8080")]
        healthcheck_addr: SocketAddr,

        #[arg(long, env, default_value = "deep-prove-1")]
        worker_class: String,

        #[arg(long, env, default_value = "Lagrange Labs")]
        operator_name: String,

        #[arg(long, env)]
        operator_priv_key: String,

        /// Max message size passed through gRPC (in MBytes)
        #[arg(long, env, default_value = "100")]
        max_message_size: usize,

        /// Should the logs be printed in json format or not
        #[arg(long, env)]
        json: bool,

        #[arg(long, env, default_value = "us-east-2", requires_all = &["s3_store"])]
        s3_region: Option<String>,
        #[arg(long, env, requires_all = &["s3_store"])]
        s3_bucket: Option<String>,
        #[arg(long, env, requires_all = &["s3_store"])]
        s3_endpoint: Option<String>,
        #[arg(long, env, default_value = "1000", requires_all = &["s3_store"])]
        s3_timeout_secs: Option<u64>,
        #[arg(env, requires_all = &["s3_store"])]
        s3_access_key_id: Option<String>,
        #[arg(env, requires_all = &["s3_store"])]
        s3_secret_access_key: Option<String>,
    },
    /// Prove inference on local files
    Local {
        /// The model to prove inference on
        #[arg(short = 'm', long)]
        onnx: PathBuf,

        /// The inputs to prove inference for
        #[arg(short = 'i', long)]
        inputs: PathBuf,
    },
}

async fn process_message_from_gw(
    msg: WorkerToGwResponse,
    outbound_tx: &tokio::sync::mpsc::Sender<WorkerToGwRequest>,
    store: StoreKind,
) -> anyhow::Result<()> {
    let task: DeepProveRequest = rmp_serde::from_slice(
        zstd::decode_all(msg.task.as_slice())
            .context("decompressing task payload")?
            .as_slice(),
    )
    .context("deserializing task")?;

    let result = match task {
        DeepProveRequest::V1(deep_prove_request_v1) => match store {
            StoreKind::S3(store) => run_model_v1(deep_prove_request_v1, store).await,
            StoreKind::Mem(store) => run_model_v1(deep_prove_request_v1, store).await,
        },
    };

    let reply = match result {
        Ok(result) => lagrange::worker_done::Reply::TaskOutput(
            rmp_serde::to_vec(&DeepProveResponse::V1(DeepProveResponseV1 {
                proofs: result,
            }))
            .unwrap(),
        ),
        Err(err) => {
            error!("failed to run model: {err:?}");
            lagrange::worker_done::Reply::WorkerError(err.to_string())
        }
    };

    let reply = Request::WorkerDone(lagrange::WorkerDone {
        task_id: msg.task_id.clone(),
        reply: Some(reply),
    });
    outbound_tx
        .send(WorkerToGwRequest {
            request: Some(reply),
        })
        .await
        .context("sending response to gateway")?;

    Ok(())
}

async fn health_check() -> (StatusCode, Json<()>) {
    (StatusCode::OK, Json(()))
}

async fn serve_health_check(addr: SocketAddr) -> anyhow::Result<()> {
    let app = Router::new().route("/health", get(health_check));
    let listener = tokio::net::TcpListener::bind(addr).await?;

    axum::serve(listener, app).await?;

    Ok(())
}

async fn run_against_gw(args: RunMode) -> anyhow::Result<()> {
    let RunMode::Remote {
        gw_url,
        healthcheck_addr,
        worker_class,
        operator_name,
        operator_priv_key,
        max_message_size,
        json,
        s3_region,
        s3_bucket,
        s3_endpoint,
        s3_timeout_secs,
        s3_access_key_id,
        s3_secret_access_key,
    } = args
    else {
        unreachable!()
    };
    setup_logging(json);

    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");

    let channel =
        tonic::transport::Channel::builder(gw_url.parse().context("parsing gateway URL")?)
            .tls_config(ClientTlsConfig::new().with_enabled_roots())
            .context("setting up TLS configuration")?
            .connect()
            .await?;

    let (outbound_tx, outbound_rx) = tokio::sync::mpsc::channel(1024);

    let wallet =
        LocalSigner::from_str(&operator_priv_key).context("parsing operator private key")?;

    let claims = grpc_worker::auth::jwt::get_claims(
        operator_name.to_string(),
        env!("CARGO_PKG_VERSION").to_string(),
        "deep-prove-1".to_string(),
        worker_class.clone(),
    )
    .context("generating gRPC token claims")?;

    let token = grpc_worker::auth::jwt::JWTAuth::new(claims, &wallet)?.encode()?;
    let token: MetadataValue<_> = format!("Bearer {token}").parse()?;

    let max_message_size = max_message_size * 1024 * 1024;
    let mut client = lagrange::workers_service_client::WorkersServiceClient::with_interceptor(
        channel,
        move |mut req: tonic::Request<()>| {
            req.metadata_mut().insert("authorization", token.clone());
            Ok(req)
        },
    )
    .max_encoding_message_size(max_message_size)
    .max_decoding_message_size(max_message_size);

    let outbound_rx = tokio_stream::wrappers::ReceiverStream::new(outbound_rx);

    outbound_tx
        .send(WorkerToGwRequest {
            request: Some(Request::WorkerReady(lagrange::WorkerReady {
                version: env!("CARGO_PKG_VERSION").to_string(),
                worker_class,
            })),
        })
        .await?;

    let response = client
        .worker_to_gw(tonic::Request::new(outbound_rx))
        .await?;

    let mut inbound = response.into_inner();

    let healthcheck_handler = tokio::spawn(serve_health_check(healthcheck_addr));
    let mut healthcheck_handler = healthcheck_handler.fuse();

    let store = if s3_region.is_some() {
        info!("Running with S3 store");
        let region = s3_region.context("gathering S3 config arguments")?;
        let timeout = std::time::Duration::from_secs(s3_timeout_secs.unwrap());
        let s3: store::AmazonS3 = store::AmazonS3Builder::new()
            .with_region(region)
            .with_bucket_name(s3_bucket.unwrap())
            .with_access_key_id(s3_access_key_id.unwrap())
            .with_secret_access_key(s3_secret_access_key.unwrap())
            .with_endpoint(s3_endpoint.unwrap())
            .with_client_options(
                store::ClientOptions::default()
                    .with_timeout(timeout)
                    .with_allow_http(true),
            )
            .build()
            .context("AWS S3 builder")?;
        StoreKind::S3(S3Store::from(s3))
    } else {
        warn!("Running with in-memory store. Specify S3 args to use S3 instead");
        StoreKind::Mem(MemStore::default())
    };

    loop {
        info!("Waiting for message...");
        tokio::select! {
            Some(inbound_message) = inbound.next() => {
                info!("Message received");
                let msg = match inbound_message {
                    Ok(msg) => msg,
                    Err(e) => {
                        error!("connection to the gateway ended with status: {e}");
                        break;
                    }
                };
                process_message_from_gw(msg, &outbound_tx, store.clone()).await?;
            }
            h = &mut healthcheck_handler => {
                if let Err(e) = h {
                    error!("healthcheck handler has shut down with error {e:?}, shutting down");
                } else {
                    info!("healthcheck handler exited, shutting down");
                }
                break
            }
        }
    }

    Ok(())
}

async fn run_locally(args: RunMode) -> anyhow::Result<()> {
    let RunMode::Local { onnx, inputs } = args else {
        unreachable!()
    };

    setup_logging(false);

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

    let request = DeepProveRequestV1 {
        model,
        input,
        scaling_strategy,
        scaling_input_hash,
    };
    let proofs = run_model_v1(request, MemStore::default()).await?;
    info!("Successfully generated {} proofs", proofs.len());

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.run_mode {
        remote_args @ RunMode::Remote { .. } => run_against_gw(remote_args).await,
        local_args @ RunMode::Local { .. } => run_locally(local_args).await,
    }
}

#[derive(Debug, Clone)]
enum StoreKind {
    S3(S3Store),
    Mem(MemStore),
}
