use std::str::FromStr;

use alloy::signers::local::LocalSigner;
use anyhow::{Context as _, Result};
use clap::Parser;
use futures::StreamExt;
use goldilocks::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams};
use tonic::{metadata::MetadataValue, transport::ClientTlsConfig};

use lagrange::{WorkerToGwRequest, worker_to_gw_request::Request};
use zkml::{
    Context, Prover, default_transcript,
    middleware::{
        DeepProveRequest, DeepProveResponse,
        v1::{
            DeepProveRequest as DeepProveRequestV1, DeepProveResponse as DeepProveResponseV1,
            Proof as ProofV1,
        },
    },
};

use crate::lagrange::WorkerToGwResponse;

mod lagrange {
    tonic::include_proto!("lagrange");
}

type F = GoldilocksExt2;
type Pcs<E> = Basefold<E, BasefoldRSParams>;

fn run_model_v1(model: DeepProveRequestV1) -> Result<Vec<ProofV1>> {
    let DeepProveRequestV1 {
        model,
        model_metadata,
        input,
    } = model;

    let (inputs, given_outputs) = input.to_elements(&model_metadata);

    let mut failed_inputs = vec![];
    let ctx =
        Some(Context::<F, Pcs<F>>::generate(&model, None).context("unable to generate context")?);

    let mut proofs = vec![];
    for (i, input) in inputs.into_iter().enumerate() {
        let input_tensor = model
            .load_input_flat(vec![input])
            .context("failed to call load_input_flat on the model")?;

        let trace_result = model.run(&input_tensor);
        // If model.run fails, print the error and continue to the next input
        let trace = match trace_result {
            Ok(trace) => trace,
            Err(e) => {
                tracing::error!(
                    "[!] Error running inference for input {}/{}: {}",
                    i + 1,
                    0, // args.num_samples,
                    e
                );
                failed_inputs.push(i);
                continue; // Skip to the next input without writing to CSV
            }
        };
        let mut prover_transcript = default_transcript();
        let prover = Prover::<_, _, _>::new(ctx.as_ref().unwrap(), &mut prover_transcript);
        let proof = prover.prove(trace).context("unable to generate proof")?;

        proofs.push(proof);
    }

    Ok(proofs)
}

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "http://localhost:10000")]
    gw_url: String,

    #[arg(long, default_value = "deep-prove-1")]
    worker_class: String,

    #[arg(long, default_value = "Lagrange Labs")]
    operator_name: String,

    #[arg(long)]
    operator_priv_key: String,
}

async fn process_message_from_gw(
    msg: WorkerToGwResponse,
    outbound_tx: &tokio::sync::mpsc::Sender<WorkerToGwRequest>,
) -> anyhow::Result<()> {
    let task: DeepProveRequest = rmp_serde::from_slice(&msg.task)?;

    let result = match task {
        DeepProveRequest::V1(deep_prove_request_v1) => run_model_v1(deep_prove_request_v1),
    };

    let reply = match result {
        Ok(result) => lagrange::worker_done::Reply::TaskOutput(
            rmp_serde::to_vec(&DeepProveResponse::V1(DeepProveResponseV1 {
                proofs: result,
            }))
            .unwrap(),
        ),
        Err(err) => lagrange::worker_done::Reply::WorkerError(err.to_string()),
    };

    let reply = Request::WorkerDone(lagrange::WorkerDone {
        task_id: msg.task_id.clone(),
        reply: Some(reply),
    });
    outbound_tx
        .send(WorkerToGwRequest {
            request: Some(reply),
        })
        .await?;

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");

    let channel = tonic::transport::Channel::builder(args.gw_url.parse()?)
        .tls_config(ClientTlsConfig::new().with_enabled_roots())?
        .connect()
        .await?;

    let (outbound_tx, outbound_rx) = tokio::sync::mpsc::channel(1024);

    let operator_name = args.operator_name;
    let worker_priv_key = args.operator_priv_key;
    let wallet = LocalSigner::from_str(&worker_priv_key)?;

    let claims = grpc_worker::auth::jwt::get_claims(
        operator_name.to_string(),
        env!("CARGO_PKG_VERSION").to_string(),
        "deep-prove-1".to_string(),
        args.worker_class.clone(),
    )?;

    let token = grpc_worker::auth::jwt::JWTAuth::new(claims, &wallet)?.encode()?;
    let token: MetadataValue<_> = format!("Bearer {token}").parse()?;

    let mut client = lagrange::workers_service_client::WorkersServiceClient::with_interceptor(
        channel,
        move |mut req: tonic::Request<()>| {
            req.metadata_mut().insert("authorization", token.clone());
            Ok(req)
        },
    );

    let outbound_rx = tokio_stream::wrappers::ReceiverStream::new(outbound_rx);

    outbound_tx
        .send(WorkerToGwRequest {
            request: Some(Request::WorkerReady(lagrange::WorkerReady {
                version: env!("CARGO_PKG_VERSION").to_string(),
                worker_class: args.worker_class,
            })),
        })
        .await?;

    let response = client
        .worker_to_gw(tonic::Request::new(outbound_rx))
        .await?;

    let mut inbound = response.into_inner();

    loop {
        tokio::select! {
            Some(inbound_message) = inbound.next() => {

                let msg = match inbound_message {
                    Ok(msg) => msg,
                    Err(e) => {
                        println!("connection to the gateway ended with status: {e}");
                        break;
                    }
                };
                process_message_from_gw(msg, &outbound_tx).await?;
            }
        }
    }

    Ok(())
}
