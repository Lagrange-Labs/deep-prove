use crate::{RunMode, StoreKind};
use anyhow::{Context, anyhow};
use base64::{Engine, prelude::BASE64_STANDARD};
use deep_prove::{middleware::v2, store::MemStore};
use exponential_backoff::Backoff;
use serde_json::json;
use tracing::{debug, error, info, warn};
use url::Url;

const ATTEMPTS: u32 = 5;
const MIN_WAIT_MS: u64 = 1000;
const MAX_WAIT_MS: u64 = 100000;

pub fn retry_operation<F, T, E: std::fmt::Debug>(func: F, log: impl Fn() -> String) -> Result<T, E>
where
    F: Fn() -> Result<T, E>,
{
    for duration in Backoff::new(
        ATTEMPTS,
        std::time::Duration::from_millis(MIN_WAIT_MS),
        std::time::Duration::from_millis(MAX_WAIT_MS),
    ) {
        let result = func();
        match &result {
            Ok(_) => {
                return result;
            }
            Err(e) => match duration {
                Some(duration) => {
                    warn!(
                        "failed to execute operation. operation: {} retry_secs: {} err: {:?}",
                        log(),
                        duration.as_secs(),
                        &e
                    );
                    std::thread::sleep(duration);
                }
                None => {
                    error!("eventually failed to execute operation {}", log());
                    return result;
                }
            },
        }
    }

    unreachable!()
}

fn request_job(gw_url: &Url, worker_name: &str) -> anyhow::Result<v2::GwToWorker> {
    ureq::get(
        gw_url
            .join(&format!("api/v1/jobs/{worker_name}"))
            .unwrap()
            .as_str(),
    )
    .call()
    .context("fetching job from gateway")
    .and_then(|mut r| {
        serde_json::from_reader::<_, v2::GwToWorker>(r.body_mut().as_reader())
            .context("deserializing job from gateway")
    })
}
fn ack_job(gw_url: &Url, worker_name: &str, job_id: i64) -> anyhow::Result<()> {
    retry_operation(
        || {
            ureq::get(
                gw_url
                    .join(&format!("/api/v1/jobs/{worker_name}/{job_id}/ack"))
                    .unwrap()
                    .as_str(),
            )
            .call()
        },
        || format!("ACK-ing job #{job_id}"),
    )?;

    Ok(())
}

fn submit_proof(gw_url: &Url, worker_name: &str, job_id: i64, proof: &[u8]) -> anyhow::Result<()> {
    let encoded_proof = BASE64_STANDARD.encode(proof);
    info!(
        "submitting a {} proof",
        humansize::format_size(encoded_proof.len(), humansize::DECIMAL)
    );
    retry_operation(
        || {
            ureq::put(
                gw_url
                    .join(&format!("/api/v1/jobs/{worker_name}/{job_id}/proof"))
                    .unwrap()
                    .as_str(),
            )
            .send_json(json!({
                "proof": BASE64_STANDARD.encode(proof),
            }))
        },
        || format!("sending proof for job #{job_id} to the gateway"),
    )?;

    Ok(())
}

fn submit_error(gw_url: &Url, worker_name: &str, job_id: i64, err_msg: &str) -> anyhow::Result<()> {
    retry_operation(
        || {
            ureq::put(
                gw_url
                    .join(&format!("/api/v1/jobs/{worker_name}/{job_id}/error"))
                    .unwrap()
                    .as_str(),
            )
            .send_json(json!({
                "error": err_msg,
            }))
        },
        || format!("sending error for job #{job_id} to the gateway"),
    )?;

    Ok(())
}

async fn process_job(job: v2::GwToWorker, store: &mut StoreKind) -> Result<Vec<u8>, String> {
    let result = match store {
        StoreKind::S3(store) => crate::run_model_v1(job.into(), store).await,
        StoreKind::Mem(store) => crate::run_model_v1(job.into(), store).await,
    };

    match result {
        Ok(proofs) => Ok(rmp_serde::to_vec(&proofs).unwrap()),

        Err(err) => {
            error!("failed to run model: {err:?}");

            Err(err.to_string())
        }
    }
}

pub async fn run(args: crate::RunMode) -> anyhow::Result<()> {
    let RunMode::Http {
        gw_url,
        address,
        json,
        worker_name,
    } = args
    else {
        unreachable!()
    };
    crate::setup_logging(json);

    let worker_name = worker_name
        .ok_or(anyhow!("no worker name set"))
        .or_else(|_| machine_uid::get())
        .map_err(|_| anyhow!("failed to build a unique worker name"))?;
    info!("gateway URL: {gw_url}");
    info!("operator address: {address}");
    info!("worker unique name: {worker_name}");

    // TODO: add S3
    // TODO:
    // let handshake = get_token(&gw_url, &worker_name).context("authenticating to the Gateway")?;
    // let secret = get_token(&gw_url, &worker_name).context("authenticating to the Gateway")?;
    let mut store = StoreKind::Mem(MemStore::default());

    loop {
        // 1. Request job to the GW
        debug!("waiting for task from gateway");
        let job = request_job(&gw_url, &worker_name).context("fetching job from LPM gateway")?;
        let job_id = job.job_id;
        info!("received job #{job_id} to execute");

        // 2. ACK job
        match ack_job(&gw_url, &worker_name, job_id) {
            Ok(_) => debug!("ACK-ed job #{job_id}"),
            Err(err) => error!("failed to ACK job: {err:?}"),
        }
        // 3. Process job & submit proof
        match process_job(job, &mut store).await {
            Ok(proof) => {
                submit_proof(&gw_url, &worker_name, job_id, &proof)
                    .context("submitting proofs to gateway")?;
                info!("submitted proof for job #{job_id}");
            }
            Err(err_msg) => {
                submit_error(&gw_url, &worker_name, job_id, &err_msg)
                    .context("submitting error to gateway")?;
                info!("submitted error for job #{job_id}");
            }
        }
    }
}
