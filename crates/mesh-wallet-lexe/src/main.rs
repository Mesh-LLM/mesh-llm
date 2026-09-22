//! `mesh-wallet-lexe`: the Lexe Lightning wallet as a mesh-llm plugin.
//!
//! Launched by the host under the ordinary plugin launch contract
//! (`MESH_LLM_PLUGIN_ENDPOINT` / `MESH_LLM_PLUGIN_NAME`). Serves the `wallet.v1`
//! capability; the host keeps the ledger, metering and gates in-process and
//! only asks this plugin to create invoices, pay, look up and observe payments.

mod lexe;

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use mesh_llm_plugin::PluginRuntime;
use mesh_llm_wallet::backend::{OpenedWallet, WalletBackend};
use mesh_llm_wallet::plugin_server::wallet_plugin;

/// Name the host launches this plugin under when bundled next to `mesh-llm`.
pub const PLUGIN_NAME: &str = "wallet-lexe";
const VERSION: &str = env!("CARGO_PKG_VERSION");

struct LexeBackend;

#[async_trait]
impl WalletBackend for LexeBackend {
    fn provider_name(&self) -> &'static str {
        "lexe"
    }

    fn is_provisioned(&self, directory: &Path) -> bool {
        lexe::is_provisioned(directory)
    }

    async fn open(&self, directory: &Path) -> Result<OpenedWallet> {
        let opened = lexe::LexeProvider::open(directory).await?;
        Ok(OpenedWallet {
            identity: opened.identity,
            provider: Arc::new(opened.provider),
            created: opened.created,
        })
    }
}

fn init_tracing() {
    // The host inherits our stderr and folds it into its own log stream.
    // Never log invoices, hashes or seeds at any level.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .json()
        .try_init();
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let plugin_name = std::env::var("MESH_LLM_PLUGIN_NAME").unwrap_or_else(|_| PLUGIN_NAME.into());
    PluginRuntime::run(wallet_plugin(plugin_name, VERSION, LexeBackend)).await
}
