use std::str::FromStr;

use alloy::signers::local::LocalSigner;
use clap::Parser;
use futures::StreamExt;
use goldilocks::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams};
use serde::{Deserialize, Serialize};
use tonic::{metadata::MetadataValue, transport::ClientTlsConfig};

use lagrange::{WorkerToGwRequest, worker_to_gw_request::Request};
use zkml::{
    Context, Element, Proof, Prover, default_transcript, model::Model, quantization::ModelMetadata,
};

pub mod lagrange {
    tonic::include_proto!("lagrange");

    pub const FILE_DESCRIPTOR_SET: &[u8] =
        tonic::include_file_descriptor_set!("lagrange_descriptor");
}

#[derive(Clone, Serialize, Deserialize)]
struct Input {
    input_data: Vec<Vec<f32>>,
    output_data: Vec<Vec<f32>>,
    pytorch_output: Vec<Vec<f32>>,
}

// TODO: this is a copypaste from `bench.rs`.
impl Input {
    fn filter(&self, indices: Option<&Vec<usize>>) -> Self {
        if let Some(indices) = indices {
            assert!(
                indices.iter().all(|i| *i < self.input_data.len()),
                "Index {} is out of range (max: {})",
                indices.iter().max().unwrap(),
                self.input_data.len() - 1
            );
            let input_data = indices
                .iter()
                .map(|i| self.input_data[*i].clone())
                .collect();
            let output_data = indices
                .iter()
                .map(|i| self.output_data[*i].clone())
                .collect();
            let pytorch_output = indices
                .iter()
                .map(|i| self.pytorch_output[*i].clone())
                .collect();
            Self {
                input_data,
                output_data,
                pytorch_output,
            }
        } else {
            self.clone()
        }
    }

    fn to_elements(self, md: &ModelMetadata) -> (Vec<Vec<Element>>, Vec<Vec<Element>>) {
        let input_sf = md.input.first().unwrap();
        let inputs = self
            .input_data
            .into_iter()
            .map(|input| input.into_iter().map(|e| input_sf.quantize(&e)).collect())
            .collect();
        let output_sf = *md.output_scaling_factor().first().unwrap();
        let outputs = self
            .output_data
            .into_iter()
            .map(|output| output.into_iter().map(|e| output_sf.quantize(&e)).collect())
            .collect();
        (inputs, outputs)
    }
}

#[derive(Serialize, Deserialize)]
struct DeepProveRequestV1 {
    model: Model<Element>,
    model_metadata: ModelMetadata,
    input: Input,
    run_indices: Option<Vec<usize>>,
}

#[derive(Serialize, Deserialize)]
struct DeepProveResponseV1 {
    proofs: Vec<ProofV1>,
}

#[derive(Serialize, Deserialize)]
enum DeepProveRequest {
    V1(DeepProveRequestV1),
}

#[derive(Serialize, Deserialize)]
enum DeepProveResponse {
    V1(DeepProveResponseV1),
}

type F = GoldilocksExt2;
type Pcs<E> = Basefold<E, BasefoldRSParams>;
type ProofV1 = Proof<GoldilocksExt2, Basefold<GoldilocksExt2, BasefoldRSParams>>;

fn run_model_v1(model: DeepProveRequestV1) -> Result<Vec<ProofV1>, ()> {
    let DeepProveRequestV1 {
        model,
        model_metadata,
        input,
        run_indices,
    } = model;

    let run_inputs = input.filter(run_indices.as_ref());
    let (inputs, given_outputs) = run_inputs.to_elements(&model_metadata);

    let input_iter = inputs.into_iter().zip(given_outputs).enumerate();
    let mut failed_inputs = vec![];
    let ctx =
        Some(Context::<F, Pcs<F>>::generate(&model, None).expect("unable to generate context"));

    let mut proofs = vec![];
    for (i, (input, _given_output)) in input_iter {
        let input_tensor = model.load_input_flat(vec![input]).unwrap();

        let trace_result = model.run(&input_tensor);
        // If model.run fails, print the error and continue to the next input
        let trace = match trace_result {
            Ok(trace) => trace,
            Err(e) => {
                tracing::info!(
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
        let proof = prover.prove(trace).expect("unable to generate proof");

        proofs.push(proof);
    }

    Ok(proofs)
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
                    Ok(ref msg) => msg,
                    Err(e) => {
                        println!("connection to the gateway ended with status: {e}");
                        break;
                    }
                };

                let task: DeepProveRequest = rmp_serde::from_slice(&msg.task)?;

                let result = match task {
                    DeepProveRequest::V1(deep_prove_request_v1) => run_model_v1(deep_prove_request_v1).unwrap(),
                };
                outbound_tx.send(WorkerToGwRequest {
                    request: Some(Request::WorkerDone(lagrange::WorkerDone {
                        task_id: msg.task_id.clone(),
                        reply: Some(lagrange::worker_done::Reply::TaskOutput(
                            rmp_serde::to_vec(&DeepProveResponse::V1(
                                DeepProveResponseV1 {
                                    proofs: result
                                }
                            )
                        ).unwrap()))
                    }))
                }).await?;
            }
            else => break,
        }
    }

    Ok(())
}
