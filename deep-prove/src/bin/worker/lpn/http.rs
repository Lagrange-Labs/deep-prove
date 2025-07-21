use crate::{RunMode, StoreKind};
use anyhow::{Context, bail};
use deep_prove::{
    middleware::{v1, v2},
    store::MemStore,
};
use exponential_backoff::Backoff;
use serde_json::json;
use tracing::{error, trace, warn};
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

fn request_job(gw_url: &Url, worker_name: &str) -> anyhow::Result<(i64, v1::DeepProveRequest)> {
    // ureq::get(gw_url.join(format!("api/v1/jobs/{worker_name}"))).call().and_then(|r| )
    todo!()
}
fn ack_job(gw_url: &Url, worker_name: &str, job_id: i64) -> anyhow::Result<()> {
    retry_operation(
        || {
            ureq::put(
                gw_url
                    .join(&format!("/api/v1/jobs/{worker_name}/{job_id}/ack"))
                    .unwrap()
                    .as_str(),
            )
            .send_empty()
        },
        || format!("ACK-ing job #{job_id}"),
    )?;

    Ok(())
}

fn submit_proof(gw_url: &Url, worker_name: &str, job_id: i64, proof: &[u8]) -> anyhow::Result<()> {
    retry_operation(
        || {
            ureq::put(
                gw_url
                    .join(&format!("/api/v1/jobs/{worker_name}/{job_id}/proof"))
                    .unwrap()
                    .as_str(),
            )
            .send_json(json!({
                "proof": proof,
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

async fn process_job(job: v1::DeepProveRequest, store: &mut StoreKind) -> Result<Vec<u8>, String> {
    let result = match store {
        StoreKind::S3(store) => crate::run_model_v1(job, store).await,
        StoreKind::Mem(store) => crate::run_model_v1(job, store).await,
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

    // TODO: add S3
    // TODO:
    // let handshake = get_token(&gw_url, &worker_name).context("authenticating to the Gateway")?;
    // let secret = get_token(&gw_url, &worker_name).context("authenticating to the Gateway")?;
    let mut store = StoreKind::Mem(MemStore::default());

    loop {
        // 1. Request job to the GW
        let (job_id, job) =
            request_job(&gw_url, &worker_name).context("fetching job from LPM gateway")?;

        // 2. ACK job
        match ack_job(&gw_url, &worker_name, job_id) {
            Ok(_) => trace!("ACK-ed job"),
            Err(err) => error!("failed to ACK job: {err:?}"),
        }
        // 3. Process job & submit proof
        match process_job(job, &mut store).await {
            Ok(proof) => submit_proof(&gw_url, &worker_name, job_id, &proof)
                .context("submitting proofs to gateway")?,
            Err(err_msg) => submit_error(&gw_url, &worker_name, job_id, &err_msg)
                .context("submitting error to gateway")?,
        }
    }
}
