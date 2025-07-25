use super::instantiate_store;
use crate::{RunMode, StoreKind};
use anyhow::{Context, anyhow, bail};
use base64::{Engine, prelude::BASE64_STANDARD};
use deep_prove::middleware::v2;
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

/// A wrapper for the connection settings, as well as helper functions to
/// interact with a gateway.
struct ConnContext {
    gw_url: Url,
    worker_name: String,
    address: String,
}
impl ConnContext {
    fn new(gw_url: Url, worker_name: String, address: String) -> Self {
        let address = address.trim_start_matches("0x").to_string();
        Self {
            gw_url,
            worker_name,
            address,
        }
    }

    /// Request a new job from the gateway.
    ///
    ///  - Will fail if the connection settings are not valid.
    ///  - Will fail after retries if the connection can not be established.
    fn request_job(&self) -> anyhow::Result<v2::GwToWorker> {
        ureq::get(
            self.gw_url
                .join(&format!("api/v1/jobs/{}", self.worker_name))
                .unwrap()
                .as_str(),
        )
        .header("authorization", &self.address)
        .call()
        .context("connecting to gateway")
        .and_then(|mut r| {
            serde_json::from_reader::<_, v2::GwToWorker>(r.body_mut().as_reader())
                .context("deserializing job from gateway")
        })
    }

    /// Confirm to the GW that we successfully received the job.
    ///
    ///  - Will fail if the connection settings are not valid.
    ///  - Will fail after retries if the connection can not be established.
    fn ack_job(&self, job_id: i64) -> anyhow::Result<()> {
        retry_operation(
            || {
                ureq::get(
                    self.gw_url
                        .join(&format!("/api/v1/jobs/{}/{job_id}/ack", self.worker_name))
                        .unwrap()
                        .as_str(),
                )
                .header("authorization", &self.address)
                .call()
            },
            || format!("ACK-ing job #{job_id}"),
        )?;

        Ok(())
    }

    /// Submit the proof for the given `job_id`.
    ///
    ///  - Will fail if the connection settings are not valid.
    ///  - Will fail after retries if the connection can not be established.
    fn submit_proof(&self, job_id: i64, proof: &[u8]) -> anyhow::Result<()> {
        let encoded_proof = BASE64_STANDARD.encode(proof);
        info!(
            "submitting a {} proof",
            humansize::format_size(encoded_proof.len(), humansize::DECIMAL)
        );
        retry_operation(
            || {
                ureq::put(
                    self.gw_url
                        .join(&format!("/api/v1/jobs/{}/{job_id}/proof", self.worker_name))
                        .unwrap()
                        .as_str(),
                )
                .header("authorization", &self.address)
                .send_json(json!({
                    "proof": BASE64_STANDARD.encode(proof),
                }))
            },
            || format!("sending proof for job #{job_id} to the gateway"),
        )?;

        Ok(())
    }

    /// Submit a failure message for the given `job_id`.
    ///
    ///  - Will fail if the connection settings are not valid.
    ///  - Will fail after retries if the connection can not be established.
    fn submit_error(&self, job_id: i64, err_msg: &str) -> anyhow::Result<()> {
        retry_operation(
            || {
                ureq::put(
                    self.gw_url
                        .join(&format!("/api/v1/jobs/{}/{job_id}/error", self.worker_name))
                        .unwrap()
                        .as_str(),
                )
                .header("authorization", &self.address)
                .send_json(json!({
                    "error": err_msg,
                }))
            },
            || format!("sending error for job #{job_id} to the gateway"),
        )?;

        Ok(())
    }
}

async fn process_job(job: v2::GwToWorker, store: &mut StoreKind) -> anyhow::Result<Vec<u8>> {
    let result = match store {
        StoreKind::S3(store) => {
            crate::run_model_v1(job.try_into().context("parsing job")?, store).await
        }
        StoreKind::Mem(store) => {
            crate::run_model_v1(job.try_into().context("parsing job")?, store).await
        }
    };

    match result {
        Ok(proofs) => Ok(rmp_serde::to_vec(&proofs).unwrap()),

        Err(err) => {
            bail!("failed to run model: {err:?}");
        }
    }
}

pub async fn run(args: crate::RunMode) -> anyhow::Result<()> {
    let RunMode::Http {
        gw_url,
        address,
        json,
        worker_name,
        s3_args,
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

    let mut store = instantiate_store(s3_args).context("instantiating PPs store")?;
    let conn = ConnContext::new(gw_url, worker_name, address);

    loop {
        // 1. Request job to the GW
        debug!("waiting for task from gateway");
        let job = conn.request_job().context("fetching job from gateway")?;
        let job_id = job.job_id;
        info!("received job #{job_id} to execute");

        // 2. ACK job
        match conn.ack_job(job_id) {
            Ok(_) => debug!("ACK-ed job #{job_id}"),
            Err(err) => error!("failed to ACK job: {err:?}"),
        }
        // 3. Process job & submit proof
        match process_job(job, &mut store).await {
            Ok(proof) => {
                conn.submit_proof(job_id, &proof)
                    .context("submitting proofs to gateway")?;
                info!("submitted proof for job #{job_id}");
            }
            Err(err) => {
                conn.submit_error(job_id, &format!("{err:?}"))
                    .context("submitting error to gateway")?;
                info!("submitted error for job #{job_id}");
            }
        }
    }
}
