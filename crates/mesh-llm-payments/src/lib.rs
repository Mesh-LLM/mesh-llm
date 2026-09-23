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
pub mod lifetimes;
pub mod pricing;
pub mod provisioning;
pub mod service;
pub mod wire;

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
