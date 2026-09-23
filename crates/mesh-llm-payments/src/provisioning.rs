//! Lazy wallet construction and persisted-wallet discovery belong to adapters,
//! not routing or settlement. Discovery must never provision or contact a wallet.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
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
        !matches!(WalletPin::load(directory), Ok(None))
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

    /// Side-effect free. Absent reads as `Ok(None)`; a pin that exists but
    /// cannot be parsed is an error, never `None`, so a damaged file can not
    /// be silently replaced by whatever wallet opens next.
    pub fn load(directory: &Path) -> Result<Option<Self>> {
        let path = Self::path(directory);
        let raw = match std::fs::read(&path) {
            Ok(raw) => raw,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(anyhow::Error::new(error)
                    .context(format!("read wallet pin {}", path.display())));
            }
        };
        serde_json::from_slice(&raw)
            .map(Some)
            .with_context(|| {
                format!(
                    "wallet pin {} is unreadable; refusing to adopt a wallet until it is repaired or removed",
                    path.display()
                )
            })
    }

    /// Persist durably before anything that depends on the pin. The temp
    /// name is per-process so concurrent writers in different processes
    /// cannot truncate each other's staging file.
    pub fn store(&self, directory: &Path) -> Result<()> {
        std::fs::create_dir_all(directory)?;
        let path = Self::path(directory);
        let tmp = directory.join(format!("{}.{}.tmp", Self::FILE_NAME, std::process::id()));
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?)?;
        // Windows requires write access for FlushFileBuffers (sync_all).
        std::fs::OpenOptions::new()
            .write(true)
            .open(&tmp)?
            .sync_all()?;
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
        assert!(WalletPin::load(dir.path()).unwrap().is_none());
        assert!(!has_persisted_wallet(dir.path()));
        let pin = WalletPin {
            plugin: "wallet-lexe".into(),
            wallet_id: "abc".into(),
            provider: "lexe".into(),
            network: "mainnet".into(),
        };
        pin.store(dir.path()).unwrap();
        assert_eq!(WalletPin::load(dir.path()).unwrap(), Some(pin.clone()));
        assert!(has_persisted_wallet(dir.path()));
        // Replacing an existing pin must also flush and rename successfully.
        let replacement = WalletPin {
            wallet_id: "replacement".into(),
            ..pin
        };
        replacement.store(dir.path()).unwrap();
        assert_eq!(WalletPin::load(dir.path()).unwrap(), Some(replacement));
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty());
    }

    #[test]
    fn corrupt_pin_is_an_error_not_absent() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(WalletPin::path(dir.path()), b"{not json").unwrap();
        let error = WalletPin::load(dir.path()).unwrap_err().to_string();
        assert!(error.contains("unreadable"), "{error}");
        // Still counts as "a wallet exists" so callers stay conservative.
        assert!(has_persisted_wallet(dir.path()));
    }
}
