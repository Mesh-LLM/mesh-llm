//! Lazy wallet construction and persisted-wallet discovery belong to adapters,
//! not routing or settlement. Discovery must never provision or contact a wallet.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use crate::wallet::WalletProvider;

#[async_trait]
pub trait WalletFactory: Send + Sync {
    /// Inspect local provisioning state without network calls or side effects.
    fn is_provisioned(&self, payment_directory: &Path) -> bool;
    /// Open or provision on an explicit wallet operation only.
    async fn open(&self, payment_directory: &Path) -> Result<Arc<dyn WalletProvider>>;
}

/// The shipped default. Alternative embedders inject their factory into the
/// payment service; no Lexe SDK type crosses this boundary.
pub struct DefaultWalletFactory;

#[async_trait]
impl WalletFactory for DefaultWalletFactory {
    fn is_provisioned(&self, directory: &Path) -> bool {
        #[cfg(feature = "lexe")]
        {
            crate::lexe::is_provisioned(directory)
        }
        #[cfg(not(feature = "lexe"))]
        {
            let _ = directory;
            false
        }
    }

    async fn open(&self, directory: &Path) -> Result<Arc<dyn WalletProvider>> {
        #[cfg(feature = "lexe")]
        {
            crate::open_wallet(&directory.join("lexe")).await
        }
        #[cfg(not(feature = "lexe"))]
        {
            let _ = directory;
            anyhow::bail!("no wallet provider configured")
        }
    }
}

pub fn has_persisted_wallet(directory: &Path) -> bool {
    DefaultWalletFactory.is_provisioned(directory)
}
