//! Provider-neutral wallets and crash-recoverable inference payments.
#![forbid(unsafe_code)]

pub mod control;
pub mod intent;
pub mod invoice;
pub mod ledger;
pub mod lifecycle;
pub mod pricing;
pub mod provisioning;
pub mod service;
pub mod wallet;
pub mod wire;

#[cfg(feature = "lexe")]
mod lexe;

/// Open the embedded mainnet wallet without exposing provider SDK types.
#[cfg(feature = "lexe")]
pub async fn open_wallet(
    directory: &std::path::Path,
) -> anyhow::Result<std::sync::Arc<dyn wallet::WalletProvider>> {
    Ok(std::sync::Arc::new(
        lexe::LexeProvider::open(directory).await?,
    ))
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests;
