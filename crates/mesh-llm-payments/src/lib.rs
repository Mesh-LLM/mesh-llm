//! Crash-recoverable inference payments over a provider-neutral wallet.
//!
//! The wallet itself lives behind [`mesh_llm_wallet::provider::WalletProvider`]
//! and is supplied by a plugin through [`provisioning::WalletFactory`]. This
//! crate owns everything the host must stay authoritative for: the ledger,
//! pricing, budgets, payment intents and settlement orchestration.
#![forbid(unsafe_code)]

pub mod control;
pub mod intent;
pub mod ledger;
#[cfg(feature = "plugin-server")]
pub mod plugin_server;
pub mod provisioning;
pub mod service;

// Pure data types live in `mesh-llm-payments-types` so mesh core can depend on
// them without linking the ledger; re-exported here under their old paths.
pub use mesh_llm_payments_types::{lifetimes, pricing, wire};

/// Wallet types re-exported under their historical paths.
pub mod invoice {
    pub use mesh_llm_wallet::invoice::Invoice;
}

/// Wallet types re-exported under their historical paths.
pub mod wallet {
    pub use mesh_llm_wallet::provider::{
        Balance, PayError, PaymentStatus, Transaction, WalletProvider,
    };
}

pub use mesh_llm_wallet::now_ms;

#[cfg(test)]
mod tests;
