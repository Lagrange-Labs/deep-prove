//! This module implements a prover instance that connects to a LPN gateway with
//! the Lagrange gRPC protocol to receive work. Proofs are executed locally,
//! then sent back to the gateway, that is in turn tasked with propagating them
//! back to the original customer.
use crate::StoreKind;
use alloy::signers::local::LocalSigner;
use anyhow::Context;
use axum::{Json, Router, http::StatusCode, routing::get};
use deep_prove::middleware::{
    DeepProveRequest, DeepProveResponse, v1::DeepProveResponse as DeepProveResponseV1,
};
use futures::{FutureExt, StreamExt};
use lagrange::{WorkerToGwRequest, WorkerToGwResponse, worker_to_gw_request::Request};
use std::{net::SocketAddr, str::FromStr};
use tonic::{metadata::MetadataValue, transport::ClientTlsConfig};
use tracing::{error, info};

use super::instantiate_store;

mod lagrange {
    tonic::include_proto!("lagrange");
}

async fn serve_health_check(addr: SocketAddr) -> anyhow::Result<()> {
    async fn health_check() -> (StatusCode, Json<()>) {
        (StatusCode::OK, Json(()))
    }
    let app = Router::new().route("/health", get(health_check));
    let listener = tokio::net::TcpListener::bind(addr).await?;

    axum::serve(listener, app).await?;

    Ok(())
}

async fn process_message_from_gw(
    msg: WorkerToGwResponse,
    outbound_tx: &tokio::sync::mpsc::Sender<WorkerToGwRequest>,
    store: &mut StoreKind,
) -> anyhow::Result<()> {
    let task: DeepProveRequest = rmp_serde::from_slice(
        zstd::decode_all(msg.task.as_slice())
            .context("decompressing task payload")?
            .as_slice(),
    )
    .context("deserializing task")?;

    let result = match task {
        DeepProveRequest::V1(deep_prove_request_v1) => match store {
            StoreKind::S3(store) => crate::run_model_v1(deep_prove_request_v1, store).await,
            StoreKind::Mem(store) => crate::run_model_v1(deep_prove_request_v1, store).await,
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

pub async fn run(args: crate::RunMode) -> anyhow::Result<()> {
    let crate::RunMode::Grpc {
        gw_url,
        healthcheck_addr,
        worker_class,
        operator_name,
        private_key,
        max_message_size,
        json,
        s3_args,
    } = args
    else {
        unreachable!()
    };
    crate::setup_logging(json);

    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");

    let mut store = instantiate_store(s3_args).context("instantiating PPs store")?;

    let channel =
        tonic::transport::Channel::builder(gw_url.parse().context("parsing gateway URL")?)
            .tls_config(ClientTlsConfig::new().with_enabled_roots())
            .context("setting up TLS configuration")?
            .connect()
            .await?;

    let (outbound_tx, outbound_rx) = tokio::sync::mpsc::channel(1024);

    let wallet = LocalSigner::from_str(&private_key).context("parsing operator private key")?;

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
                process_message_from_gw(msg, &outbound_tx, &mut store).await?;
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
