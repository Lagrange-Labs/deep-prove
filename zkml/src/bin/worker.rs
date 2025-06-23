use std::str::FromStr;

use alloy::signers::local::LocalSigner;
use clap::Parser;
use futures::StreamExt;
use lagrange::{WorkerToGwRequest, worker_to_gw_request::Request};
use tonic::{metadata::MetadataValue, transport::ClientTlsConfig};

pub mod lagrange {
    tonic::include_proto!("lagrange");

    pub const FILE_DESCRIPTOR_SET: &[u8] =
        tonic::include_file_descriptor_set!("lagrange_descriptor");
}

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "http://localhost:10000")]
    gw_url: String,

    #[arg(short, long, default_value_t = 30)]
    sleep_time: u64,

    #[arg(short, long, default_value = "mock_worker")]
    worker_id: String,

    #[arg(long, default_value = "testingify")]
    worker_class: String,
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

    let operator_name = "grpc_operator";
    let worker_priv_key = "779ff5fe168de6560e95dff8c91d3af4c45ad1b261d03d22e2e1558fb27ea450";
    let wallet = LocalSigner::from_str(worker_priv_key)?;

    let claims = grpc_worker::auth::jwt::get_claims(
        operator_name.to_string(),
        env!("CARGO_PKG_VERSION").to_string(),
        args.worker_id.to_string(),
        args.worker_class.clone(),
    )?;

    println!("claims {claims:?}");
    let token = grpc_worker::auth::jwt::JWTAuth::new(claims, &wallet)?.encode()?;
    let token: MetadataValue<_> = format!("Bearer {token}").parse()?;

    println!("Inserting interceptor");
    let mut client = lagrange::workers_service_client::WorkersServiceClient::with_interceptor(
        channel,
        move |mut req: tonic::Request<()>| {
            req.metadata_mut().insert("authorization", token.clone());
            Ok(req)
        },
    );

    print!("opening outbound connection...");
    let outbound_rx = tokio_stream::wrappers::ReceiverStream::new(outbound_rx);
    println!("done.");

    eprint!("sending initial message...");

    outbound_tx
        .send(WorkerToGwRequest {
            request: Some(Request::WorkerReady(lagrange::WorkerReady {
                version: env!("CARGO_PKG_VERSION").to_string(),
                worker_class: args.worker_class,
            })),
        })
        .await?;

    eprintln!("sent.");

    eprint!("calling worker_to_gw...");
    let response = client
        .worker_to_gw(tonic::Request::new(outbound_rx))
        .await?;
    eprintln!("done.");

    eprintln!("opening inbound connection");
    let mut inbound = response.into_inner();

    loop {
        tokio::select! {
            Some(inbound_message) = inbound.next() => {
                let msg = match inbound_message {
                    Ok(ref msg) => msg,
                    Err(e) => {
                        println!("connection to the gateway ended with status: {e}");
                        break;
                    }
                };
                println!("got inbound message from gateway: {msg:?}");

                tokio::time::sleep(std::time::Duration::from_secs(args.sleep_time)).await;
                outbound_tx.send(WorkerToGwRequest {
                    request: Some(Request::WorkerDone(lagrange::WorkerDone {
                        task_id: msg.task_id.clone(),
                        reply: Some(lagrange::worker_done::Reply::TaskOutput(vec![1, 2, 3]))
                    }))
                }).await?;
            }
            else => break,
        }
    }

    Ok(())
}
