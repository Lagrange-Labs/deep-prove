use clap::{Parser, Subcommand};
use lagrange::ProofChannelResponse;
use tonic::{metadata::MetadataValue, transport::ClientTlsConfig};
use url::Url;

mod lagrange {
    tonic::include_proto!("lagrange");

    pub const FILE_DESCRIPTOR_SET: &[u8] =
        tonic::include_file_descriptor_set!("lagrange_descriptor");
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

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Submit a model and its input to prove inference.
    Submit {
        /// Path to the ONNX file of the model to prove.
        #[arg(short, long)]
        onnx: String,
        /// Path to the inputs to the model to prove inference for.
        #[arg(short, long)]
        inputs: String,
    },

    /// Fetch the generated proofs, if any.
    Fetch {},
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
        .await?;

    let client_id: MetadataValue<_> = args.client_id.parse()?;
    let mut client = lagrange::clients_service_client::ClientsServiceClient::with_interceptor(
        channel,
        move |mut req: tonic::Request<()>| {
            req.metadata_mut().insert("client_id", client_id.clone());
            Ok(req)
        },
    );

    match args.command {
        Command::Submit { onnx, inputs } => {
            let task = tonic::Request::new(lagrange::SubmitTaskRequest {
                task_bytes: vec![0xde, 0xad, 0xbe, 0xef],
                user_task_id: format!(),
                timeout: Some(
                    prost_wkt_types::Duration::try_from(std::time::Duration::from_secs(300))
                        .unwrap(),
                ),
                price_requested: 12_u64.to_le_bytes().to_vec(),
                stake_requested: vec![0u8; 32],
                class: vec!["deep-prove".to_string()],
                priority: 0,
            });
            let response = client.submit_task().await?;
            println!("got the response {response:?}");
        }
        Command::Fetch {} => {
            let (proof_channel_tx, proof_channel_rx) = tokio::sync::mpsc::channel(1024);

            let proof_channel_rx = tokio_stream::wrappers::ReceiverStream::new(proof_channel_rx);
            let channel = client
                .proof_channel(tonic::Request::new(proof_channel_rx))
                .await
                .unwrap();
            let mut proof_response_stream = channel.into_inner();

            println!("Fetching ready proofs...");
            while let Some(response) = proof_response_stream.message().await? {
                let ProofChannelResponse { response } = response;

                let lagrange::proof_channel_response::Response::Proof(v) = response.unwrap();

                let lagrange::ProofReady {
                    task_id,
                    task_output,
                } = v;

                let task_id = task_id.unwrap();

                println!("Received proof for task {:?}: {task_output:?}", task_id.id);

                proof_channel_tx
                    .send(lagrange::ProofChannelRequest {
                        request: Some(lagrange::proof_channel_request::Request::AckedMessages(
                            lagrange::AckedMessages {
                                acked_messages: vec![task_id],
                            },
                        )),
                    })
                    .await?;
            }
        }
    }
    Ok(())
}
