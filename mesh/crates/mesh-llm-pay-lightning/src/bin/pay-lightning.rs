//! Separate utility entrypoint using an operator-selected host binding.
use anyhow::{Context, Result, ensure};
use std::{io::Write, path::PathBuf};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Launch {
    config_path: PathBuf,
    wallet_binding: String,
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
    let settings = std::env::var("MESH_LLM_PLUGIN_SETTINGS_JSON")
        .context("explicit utility settings required")?;
    let launch: Launch = serde_json::from_str(&settings)?;
    ensure!(
        launch.config_path.is_absolute(),
        "config path must be absolute"
    );
    ensure!(!launch.wallet_binding.is_empty(), "wallet binding required");
    let config = mesh_llm_pay_lightning::Config::load(&launch.config_path)?;
    mesh_llm_pay_lightning::run_bound(
        "pay-lightning".into(),
        env!("CARGO_PKG_VERSION"),
        launch.wallet_binding,
        launch.allow_wallet_provisioning,
        config,
    )
    .await
}
