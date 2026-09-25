//! Provider-neutral Lightning wallet abstraction for mesh-llm.
//!
//! This crate owns three things and nothing else:
//!
//! - [`provider::WalletProvider`], the Rust trait the payment ledger drives.
//! - [`invoice::Invoice`], BOLT11 parsing and validation at trust boundaries.
//! - [`contract`], the versioned `wallet.v1` plugin capability: operation
//!   names plus the JSON request/response/error shapes that cross the plugin
//!   IPC boundary.
//!
//! No wallet SDK (Lexe or otherwise) is linked here. Concrete wallets are
//! plugin processes that implement [`backend::WalletBackend`] and are
//! projected onto the plugin runtime by the `plugin-server` feature; the
//! shipped one (`mesh-wallet-lexe`) is served from the mesh-llm executable as
//! `--plugin wallet-lexe`, external ones from their own.
#![forbid(unsafe_code)]

pub mod contract;
pub mod invoice;
pub mod provider;
pub mod provisioning;

#[cfg(feature = "plugin-server")]
pub mod backend;
#[cfg(feature = "plugin-server")]
pub mod plugin_server;

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
