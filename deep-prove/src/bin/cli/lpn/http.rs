use anyhow::{Context, bail};
use axum::http::StatusCode;
use deep_prove::middleware::v2::ClientToGw;
use std::io::Write;
use tracing::{error, info};
use zkml::inputs::Input;

use crate::{Command, Executor};

/// Maximum proof size (1GiB)
const MAX_PROOF_SIZE: u64 = 1000 * 1024 * 1024;

pub async fn connect(executor: Executor) -> anyhow::Result<()> {
    let Executor::LpnHttp {
        gw_url,
        private_key: _,
        command,
    } = executor
    else {
        unreachable!()
    };
    let api_url = gw_url.join("api/v1/").context("building API URL")?;

    match command {
        Command::Submit { .. } => bail!("`submit` is not supported"),
        Command::Request {
            pretty_name,
            model_id,
            inputs,
        } => {
            let input = Input::from_file(&inputs).context("loading input")?;

            let request = ClientToGw {
                pretty_name: pretty_name.unwrap_or_else(|| {
                    format!(
                        "{model_id}-{}",
                        std::time::SystemTime::now()
                            .duration_since(std::time::SystemTime::UNIX_EPOCH)
                            .expect("you're not Dr. Who -- come back to the forward-flowing time")
                            .as_secs()
                    )
                }),
                model_id: model_id.try_into().context("`model_id` is too large")?,
                input,
            };

            // build the API endpoint request and send the whole thing
            let mut resp = ureq::post(api_url.join("tasks")?.as_str())
                .send_json(request)
                .context("calling API")?;
            match resp.status() {
                StatusCode::CREATED => {
                    info!("[CREATED] {}", resp.body_mut().read_to_string()?);
                }
                c => {
                    error!(
                        "failed to send request: [{}] {}",
                        c.as_str(),
                        resp.body_mut().read_to_string()?
                    );
                }
            }
        }
        Command::Fetch { filename } => {
            let mut resp = ureq::get(api_url.join("proof")?.as_str())
                .call()
                .context("calling API")?;

            match resp.status() {
                StatusCode::OK => {
                    let filename = filename.unwrap_or_else(|| {
                        format!(
                            "{}.bin",
                            std::time::SystemTime::now()
                                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                                .unwrap()
                                .as_secs()
                        )
                    });
                    std::fs::File::create(&filename)
                        .context("failed to create proof file")?
                        .write_all(
                            resp.body_mut()
                                .with_config()
                                .limit(MAX_PROOF_SIZE)
                                .read_to_vec()?
                                .as_slice(),
                        )
                        .context("failed to write proof")?;
                    info!("proof written to {filename}");
                }
                StatusCode::NO_CONTENT => {
                    info!("no proof ready");
                }
                c => {
                    error!(
                        "failed to fetch proof: [{}] {}",
                        c.as_str(),
                        resp.body_mut().read_to_string()?
                    );
                }
            }
        }
        Command::Cancel { task_id } => {
            // build the API endpoint request and send the whole thing
            let mut resp =
                ureq::delete(api_url.join(format!("tasks/{task_id}").as_str())?.as_str())
                    .call()
                    .context("calling API")?;
            match resp.status() {
                StatusCode::NO_CONTENT => {
                    info!("task successfully cancelled");
                }
                c => {
                    error!(
                        "failed to send request: [{}] {}",
                        c.as_str(),
                        resp.body_mut().read_to_string()?
                    );
                }
            }
        }
    }

    Ok(())
}
