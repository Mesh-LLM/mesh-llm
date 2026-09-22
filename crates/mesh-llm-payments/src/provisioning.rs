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

/// A factory with no wallet behind it. `PaymentService::open` uses this so
/// ledger-only operations (policy, pricing, pending) work without any wallet
/// plugin; every wallet operation fails with a clear error. Embedders inject a
/// real factory with `PaymentService::with_factory`.
pub struct NoWalletFactory;

#[async_trait]
impl WalletFactory for NoWalletFactory {
    fn is_provisioned(&self, directory: &Path) -> bool {
        WalletPin::load(directory).is_some()
    }

    async fn open(&self, _directory: &Path) -> Result<Arc<dyn WalletProvider>> {
        anyhow::bail!("no wallet provider available; a wallet plugin must be running")
    }
}

/// Host-owned record of which wallet backs this payment directory.
///
/// Written after the first successful plugin open; read on every later open.
/// Outstanding reservations and receivables in the ledger are only meaningful
/// against this exact wallet, so a different plugin or a different wallet
/// identity is refused rather than silently adopted.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WalletPin {
    pub plugin: String,
    pub wallet_id: String,
    pub provider: String,
    pub network: String,
}

impl WalletPin {
    pub const FILE_NAME: &'static str = "wallet-provider.json";

    pub fn path(directory: &Path) -> std::path::PathBuf {
        directory.join(Self::FILE_NAME)
    }

    /// Side-effect free: absent or unreadable pins read as `None`.
    pub fn load(directory: &Path) -> Option<Self> {
        let raw = std::fs::read(Self::path(directory)).ok()?;
        serde_json::from_slice(&raw).ok()
    }

    /// Persist durably before anything that depends on the pin.
    pub fn store(&self, directory: &Path) -> Result<()> {
        std::fs::create_dir_all(directory)?;
        let path = Self::path(directory);
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?)?;
        std::fs::File::open(&tmp)?.sync_all()?;
        std::fs::rename(&tmp, &path)?;
        #[cfg(unix)]
        std::fs::File::open(directory)?.sync_all()?;
        Ok(())
    }
}

/// Persisted wallet state exists for this payment directory. Side-effect free.
pub fn has_persisted_wallet(directory: &Path) -> bool {
    NoWalletFactory.is_provisioned(directory)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pin_round_trips_and_is_absent_by_default() {
        let dir = tempfile::tempdir().unwrap();
        assert!(WalletPin::load(dir.path()).is_none());
        assert!(!has_persisted_wallet(dir.path()));
        let pin = WalletPin {
            plugin: "wallet-lexe".into(),
            wallet_id: "abc".into(),
            provider: "lexe".into(),
            network: "mainnet".into(),
        };
        pin.store(dir.path()).unwrap();
        assert_eq!(WalletPin::load(dir.path()), Some(pin));
        assert!(has_persisted_wallet(dir.path()));
        assert!(!dir.path().join("wallet-provider.json.tmp").exists());
    }
}
