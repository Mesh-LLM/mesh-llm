//! Standalone OUSD provider. Local management commands and Mesh runtime are separate entry modes.
mod journal;
mod keys;
mod receipt;
mod rpc;
#[cfg(test)]
mod rpc_fixture_tests;
#[cfg(test)]
mod tests;
mod wallet;

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
};
use wallet::{Settings, Wallet};
use wallet_contract::{LookupRequest, PayRequest, PaymentRequest, VerifyRequest, ops};

#[derive(Parser)]
#[command(version, about = "Non-custodial OUSD wallet on Tempo mainnet (4217)")]
struct Cli {
    #[arg(long, global = true)]
    state_dir: Option<PathBuf>,
    #[arg(long, global = true)]
    rpc_url: Option<String>,
    #[arg(long, global = true)]
    password_file: Option<PathBuf>,
    #[arg(long, global = true)]
    max_payment: Option<u64>,
    #[arg(long, global = true)]
    max_fee: Option<u64>,
    #[command(subcommand)]
    command: Option<Command>,
}
#[derive(Subcommand)]
enum Command {
    /// Explicitly create a local encrypted keystore. No network transaction.
    Init,
    /// Print the persisted mainnet receiving address, without unlocking.
    Address,
    Info,
    Balance,
    Estimate {
        #[arg(long)]
        request: PathBuf,
    },
    /// Broadcast a real mainnet payment only with the explicit spending flag.
    Pay {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        confirm_mainnet: bool,
    },
    Lookup {
        #[arg(long)]
        payment_id: String,
    },
    Verify {
        #[arg(long)]
        request: PathBuf,
    },
    Attempts,
    /// Run the host operation protocol (also the default when no command is given).
    Serve,
}

fn output(value: impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}
fn read<T: serde::de::DeserializeOwned>(path: PathBuf) -> Result<T> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::args().any(|a| a == "--manifest") {
        println!(
            "{}",
            mesh_llm_plugin::package_manifest_json(&server::manifest())?
        );
        return Ok(());
    }
    let cli = Cli::parse();
    let mut settings = Settings::default();
    if let Some(value) = cli.state_dir {
        settings.data_dir = value;
    }
    if let Some(value) = cli.rpc_url {
        settings.rpc_url = value;
    }
    settings.password_file = cli.password_file;
    if let Some(value) = cli.max_payment {
        settings.max_payment = value;
    }
    if let Some(value) = cli.max_fee {
        settings.max_fee = value;
    }
    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => server::run().await,
        Command::Init => {
            ensure!(
                settings.data_dir.is_absolute(),
                "--state-dir must be an absolute owner-only directory"
            );
            output(keys::initialize(
                &settings.data_dir,
                settings
                    .password_file
                    .as_deref()
                    .context("init requires --password-file")?,
            )?)
        }
        Command::Address => output(keys::load_profile(&settings.data_dir)?),
        command => {
            let wallet = Wallet::load(settings)?;
            match command {
                Command::Info => output(wallet.info().await?),
                Command::Balance => output(wallet.balance().await?),
                Command::Estimate { request } => {
                    output(wallet.estimate(read::<PaymentRequest>(request)?).await?)
                }
                Command::Pay {
                    request,
                    confirm_mainnet,
                } => {
                    ensure!(
                        confirm_mainnet,
                        "real mainnet spending requires --confirm-mainnet"
                    );
                    output(wallet.pay(read::<PayRequest>(request)?).await?)
                }
                Command::Lookup { payment_id } => output(wallet.lookup(&payment_id).await?),
                Command::Verify { request } => {
                    output(wallet.verify(read::<VerifyRequest>(request)?).await?)
                }
                Command::Attempts => output(wallet.journal.list()?),
                Command::Serve | Command::Init | Command::Address => unreachable!(),
            }
        }
    }
}

mod server {
    use super::*;
    use mesh_llm_plugin::{
        OperationRouter, PluginError, PluginMetadata, PluginRuntime, SimplePlugin, capability,
        config_path, config_schema, config_setting, config_string, config_url,
        operation_with_schema, plugin_server_info, proto,
    };
    use serde_json::Value;

    pub fn manifest() -> proto::PluginManifest {
        // Binding-only provider: publishing operations here also exposes them
        // through the host's HTTP/MCP tools, bypassing the engine's policy.
        mesh_llm_plugin::plugin_manifest![
            capability(wallet_contract::CAPABILITY),
            config_schema("wallet-ousd-tempo")
              .setting(config_setting("data_dir",config_path()).required(true).description("Absolute, initialized wallet state directory; owner-only access."))
              .setting(config_setting("rpc_url",config_url()).default_value(&"https://rpc.tempo.xyz").description("Tempo mainnet HTTPS RPC endpoint."))
              .setting(config_setting("password_file",config_path()).description("Optional absolute owner-only credential file path. Contains the keystore password; never published."))
              .setting(config_setting("max_payment",config_string()).default_value(&"1000000").description("Local per-payment OUSD limit in raw six-decimal units."))
              .setting(config_setting("max_fee",config_string()).default_value(&"10000").description("Local maximum gas charge in raw OUSD units."))
        ]
    }
    fn descriptions() -> [(&'static str, &'static str); 7] {
        [
            (
                ops::OPEN,
                "Open the operator-configured OUSD wallet and report pinned identity; never creates keys.",
            ),
            (
                ops::BALANCE,
                "Read finalized OUSD balance and local signed-payment reservations.",
            ),
            (
                ops::ESTIMATE,
                "Estimate a direct OUSD payment and OUSD gas reserve.",
            ),
            (
                ops::PAY,
                "Idempotently sign/journal/submit one authorized OUSD payment.",
            ),
            (
                ops::LOOKUP,
                "Reconcile a previously journaled payment without creating another transfer.",
            ),
            (
                ops::VERIFY,
                "Independently verify finalized OUSD delivery and its obligation memo.",
            ),
            (
                ops::ATTEMPTS,
                "List local payment attempts; no secrets or raw signed transactions returned.",
            ),
        ]
    }
    pub async fn run() -> Result<()> {
        let state: Arc<OnceLock<Arc<Wallet>>> = Arc::new(OnceLock::new());
        let initialized = state.clone();
        let mut router = OperationRouter::new();
        for (name, description) in descriptions() {
            let state = state.clone();
            router.add_raw(operation_with_schema(name,description,serde_json::Map::new()),move |request,_context| {
                let state=state.clone();
                Box::pin(async move {
                    let wallet=state.get().cloned().ok_or_else(|| PluginError::internal("wallet initialization is incomplete"))?;
                    let result:Result<Value>=async {
                        match name {
                            ops::OPEN => {ensure!(request.arguments.as_object().is_some_and(|o| o.is_empty()),"wallet_open accepts an empty object; data_dir belongs to provider settings");Ok(serde_json::to_value(wallet.info().await?)?)},
                            ops::BALANCE => Ok(serde_json::to_value(wallet.balance().await?)?),
                            ops::ESTIMATE => Ok(serde_json::to_value(wallet.estimate(serde_json::from_value(request.arguments)?).await?)?),
                            ops::PAY => Ok(serde_json::to_value(wallet.pay(serde_json::from_value(request.arguments)?).await?)?),
                            ops::LOOKUP => {let request:LookupRequest=serde_json::from_value(request.arguments)?;Ok(serde_json::to_value(wallet.lookup(&request.payment_id).await?)?)},
                            ops::VERIFY => Ok(serde_json::to_value(wallet.verify(serde_json::from_value(request.arguments)?).await?)?),
                            ops::ATTEMPTS => Ok(serde_json::to_value(wallet.journal.list()?)?),
                            _ => anyhow::bail!("unknown wallet operation"),
                        }
                    }.await;
                    match result {
                        Ok(value) => mesh_llm_plugin::structured_tool_result(value),
                        Err(error) => Ok(mesh_llm_plugin::operation_error(error.to_string())),
                    }
                })
            });
        }
        let name =
            std::env::var("MESH_LLM_PLUGIN_NAME").unwrap_or_else(|_| "wallet-ousd-tempo".into());
        let plugin = SimplePlugin::new(PluginMetadata::new(
            name,
            env!("CARGO_PKG_VERSION"),
            plugin_server_info(
                "wallet-ousd-tempo",
                env!("CARGO_PKG_VERSION"),
                "OUSD Tempo wallet",
                "Local non-custodial mainnet OUSD wallet provider",
                None::<String>,
            ),
        ))
        .with_capabilities(vec![wallet_contract::CAPABILITY.into()])
        .with_manifest(manifest())
        .with_operation_router(router)
        .on_initialize(move |request, _context| {
            let state = initialized.clone();
            Box::pin(async move {
                let settings: Settings = serde_json::from_str(&request.settings_json)
                    .map_err(|e| PluginError::invalid_params(e.to_string()))?;
                if !settings.data_dir.is_absolute() {
                    return Err(PluginError::invalid_params(
                        "data_dir must be an absolute initialized directory",
                    ));
                }
                let wallet =
                    Wallet::load(settings).map_err(|e| PluginError::internal(e.to_string()))?;
                state
                    .set(wallet)
                    .map_err(|_| PluginError::internal("wallet already initialized"))?;
                Ok(())
            })
        });
        PluginRuntime::run(plugin).await
    }
}
