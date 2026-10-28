//! Separate utility entrypoint. Binding is explicit until installer support lands.
use anyhow::Result;
#[cfg(unix)]
use anyhow::{Context, ensure};
use std::io::Write;

#[cfg(unix)]
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Launch {
    payment: mesh_llm_pay_lightning::Config,
    wallet_plugin: String,
    wallet_executable: std::path::PathBuf,
    wallet_directory: std::path::PathBuf,
    #[serde(default)]
    allow_wallet_provisioning: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::args().any(|arg| arg == "--manifest") {
        writeln!(
            mesh_llm_events::machine_out(),
            "{}",
            mesh_llm_pay_lightning::manifest_json("pay-lightning")?
        )?;
        return Ok(());
    }
    launch().await
}

#[cfg(unix)]
async fn launch() -> Result<()> {
    let path = std::env::var_os("MESH_LLM_PLUGIN_CONFIG")
        .context("explicit utility binding config required")?;
    let config: Launch = serde_json::from_slice(&std::fs::read(path)?)?;
    config.payment.validate()?;
    ensure!(
        config.wallet_executable.is_absolute() && config.wallet_directory.is_absolute(),
        "wallet executable and state paths must be absolute"
    );
    let pin = mesh_llm_wallet::provisioning::WalletPin::load(&config.wallet_directory)?;
    ensure!(
        pin.is_some() || config.allow_wallet_provisioning,
        "wallet not provisioned; explicit provisioning approval required"
    );
    mesh_llm_pay_lightning::run(
        "pay-lightning".into(),
        env!("CARGO_PKG_VERSION"),
        mesh_llm_pay_lightning::wallet_child::ChildWalletFactory {
            executable: config.wallet_executable,
            plugin: config.wallet_plugin,
            directory: config.wallet_directory,
        },
        config.payment,
    )
    .await
}

#[cfg(not(unix))]
async fn launch() -> Result<()> {
    anyhow::bail!("wallet child transport is not yet supported on this platform")
}
