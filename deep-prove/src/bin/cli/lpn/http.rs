use anyhow::bail;

use crate::{Command, Executor};

pub async fn connect(executor: Executor) -> anyhow::Result<()> {
    let Executor::LpnHttp {
        gw_url,
        private_key,
        command,
    } = executor
    else {
        unreachable!()
    };

    match command {
        Command::Submit { .. } => bail!("`submit` is not supported"),
        Command::Request { model_id, inputs } => todo!(),
        Command::Fetch {} => todo!(),
    }

    Ok(())
}
