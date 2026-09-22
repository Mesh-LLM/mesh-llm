//! The Lexe Lightning wallet as a built-in mesh-llm plugin.
//!
//! Like `blobstore`, this runs as a separate process launched by the host as
//! `mesh-llm --plugin wallet-lexe` under the ordinary plugin launch contract
//! (`MESH_LLM_PLUGIN_ENDPOINT` / `MESH_LLM_PLUGIN_NAME`). It serves the
//! `wallet.v1` capability; the host keeps the ledger, metering and gates
//! in-process and only asks this plugin to create invoices, pay, look up and
//! observe payments.
//!
//! This is the only crate in the workspace that links the Lexe SDK. The
//! shipped `mesh-llm` binary enables it through the host-runtime `wallet-lexe`
//! feature; SDK consumers that do not enable that feature never compile it.
#![forbid(unsafe_code)]

mod lexe;

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use mesh_llm_plugin::PluginRuntime;
use mesh_llm_wallet::backend::{OpenedWallet, WalletBackend};
use mesh_llm_wallet::plugin_server::wallet_plugin;

/// Plugin name the host launches this implementation under.
pub const PLUGIN_NAME: &str = "wallet-lexe";
const VERSION: &str = env!("CARGO_PKG_VERSION");

struct LexeBackend;

#[async_trait]
impl WalletBackend for LexeBackend {
    fn provider_name(&self) -> &'static str {
        "lexe"
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

/// Serve the `wallet.v1` capability on the host-supplied plugin endpoint until
/// the host closes the connection. `name` is the plugin name the host launched
/// this process under. Logging is owned by the launching binary.
pub async fn run_plugin(name: String) -> Result<()> {
    PluginRuntime::run(wallet_plugin(name, VERSION, LexeBackend)).await
}
