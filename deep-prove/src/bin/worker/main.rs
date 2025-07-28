use anyhow::{Context as _, Result};
use clap::{ArgGroup, Parser, Subcommand};
use deep_prove::{
    middleware::v1::{DeepProveRequest as DeepProveRequestV1, Proof as ProofV1},
    store::{self, MemStore, S3Store, Store},
};
use ff_ext::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams, Hasher};
use std::{net::SocketAddr, path::PathBuf};
use tracing::{debug, error, info};
use tracing_subscriber::{EnvFilter, filter::LevelFilter, fmt::format::FmtSpan};
use url::Url;
use zkml::{
    Context, Element, FloatOnnxLoader, Prover, default_transcript,
    model::Model,
    quantization::{AbsoluteMax, ModelMetadata},
};

mod api;
mod immediate;
mod lpn;

mod lagrange {
    tonic::include_proto!("lagrange");
}

type F = GoldilocksExt2;
type Pcs<E> = Basefold<E, BasefoldRSParams<Hasher>>;

/// From a proof request wrapped in a [`DeepProveRequestV1`] and a [`Store`]
/// implementation (to interact with the PPs), attempt to generate proofs for a
/// list of inputs.
async fn run_model_v1<S: Store>(model: DeepProveRequestV1, store: &mut S) -> Result<Vec<ProofV1>> {
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
                .prove(trace)
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

#[derive(clap::Args)]
struct S3Args {
    #[arg(long, env, default_value = "us-east-2", requires = "s3_store")]
    s3_region: Option<String>,
    #[arg(long, env, requires = "s3_store")]
    s3_bucket: Option<String>,
    #[arg(long, env, requires = "s3_store")]
    s3_endpoint: Option<String>,
    #[arg(long, env, default_value = "1000", requires = "s3_store")]
    s3_timeout_secs: Option<u64>,
    #[arg(env, requires = "s3_store")]
    s3_access_key_id: Option<String>,
    #[arg(env, requires = "s3_store")]
    s3_secret_access_key: Option<String>,
    /// Enable local file-system cache for S3 data
    #[arg(long, env, requires = "s3_store")]
    fs_cache: bool,
    /// Set the path of the S3 store local cache.
    #[arg(long, env, requires = "s3_store", default_value = "/var/cache")]
    fs_cache_dir: PathBuf,
}

#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum RunMode {
    /// Connect to a LPN gateway to receive inference tasks
    #[command(
        name = "lpn",
        group(ArgGroup::new("s3_store")
        .multiple(true)
        .args(&["s3_region", "s3_bucket", "s3_endpoint", "s3_access_key_id", "s3_secret_access_key"])
        .requires_all(&["s3_region", "s3_bucket", "s3_endpoint", "s3_access_key_id", "s3_secret_access_key"])
    ))]
    Grpc {
        #[arg(long, env, default_value = "http://localhost:10000")]
        gw_url: String,

        /// An address of the `/health` probe.
        #[arg(long, env, default_value = "127.0.0.1:8080")]
        healthcheck_addr: SocketAddr,

        #[arg(long, env, default_value = "deep-prove-1")]
        worker_class: String,

        /// The operator name.
        #[arg(long, env, default_value = "Lagrange Labs")]
        operator_name: String,

        /// The operator private key.
        #[arg(long, env)]
        private_key: String,

        /// Max message size passed through gRPC (in MBytes)
        #[arg(long, env, default_value = "100")]
        max_message_size: usize,

        /// Print the logs in JSON format
        #[arg(long, env)]
        json: bool,

        /// If set, use S3 to store & fetch PPs, otherwise use memory.
        #[command(flatten)]
        s3_args: S3Args,
    },
    /// Connect to a LPN gateway to receive inference tasks.
    #[command(
        group(ArgGroup::new("s3_store")
        .multiple(true)
        .args(&["s3_region", "s3_bucket", "s3_endpoint", "s3_access_key_id", "s3_secret_access_key"])
        .requires_all(&["s3_region", "s3_bucket", "s3_endpoint", "s3_access_key_id", "s3_secret_access_key"])
    ))]
    Http {
        #[arg(long, env, default_value = "http://localhost:4000")]
        gw_url: Url,

        /// This worker unique name. If not set, a UID will be tentatively built
        /// from the machine ID.
        #[arg(short, long)]
        worker_name: Option<String>,

        /// The operator ETH address.
        #[arg(long, env)]
        address: String,

        /// Print the logs in JSON format.
        #[arg(long, env)]
        json: bool,

        /// If set, use S3 to store & fetch PPs, otherwise use memory.
        #[command(flatten)]
        s3_args: S3Args,
    },
    /// Prove inference on local files
    Local {
        /// The model to prove inference on.
        #[arg(short = 'm', long)]
        onnx: PathBuf,

        /// The inputs to prove inference for.
        #[arg(short = 'i', long)]
        inputs: PathBuf,
    },
    /// Run a HTTP server and process requests
    LocalApi {
        /// Listening port
        #[arg(short, long, env, default_value_t = 8080)]
        port: u16,

        /// Print the logs in JSON format
        #[arg(long, env)]
        json: bool,

        /// The maximal proof request size to accept (in MB)
        #[arg(long, env, default_value_t = 200)]
        max_body_size: usize,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.run_mode {
        grpc_args @ RunMode::Grpc { .. } => lpn::grpc::run(grpc_args).await,
        local_args @ RunMode::Local { .. } => immediate::run(local_args).await,
        api_args @ RunMode::LocalApi { .. } => api::serve(api_args).await,
        http_args @ RunMode::Http { .. } => lpn::http::run(http_args).await,
    }
}

#[derive(Clone)]
enum StoreKind {
    S3(S3Store),
    Mem(MemStore),
}
