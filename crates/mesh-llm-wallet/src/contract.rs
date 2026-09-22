//! The `wallet.v1` plugin capability contract.
//!
//! A wallet plugin advertises [`CAPABILITY`] in its manifest and serves the
//! operations below over the ordinary plugin operation transport. The host
//! resolves the provider by capability, never by plugin name, so any plugin
//! that speaks this contract can back paid inference.
//!
//! Every shape here is additive JSON: unknown fields are ignored on both
//! sides, and new optional fields may be added without a version bump.
//! Renaming or removing a field, or changing an operation's semantics, is a
//! `wallet.v2`.
//!
//! The host, not the plugin, owns token metering, output gating, the ledger,
//! budgets and settlement bookkeeping. The plugin only turns wallet intents
//! into wallet facts: invoices, payments, balances and settlement observation.

use serde::{Deserialize, Serialize};

use crate::invoice::Invoice;
use crate::provider::{Balance, PayError, Transaction};

/// Capability name a wallet plugin declares in its manifest.
pub const CAPABILITY: &str = "wallet.v1";

/// Operation names. These are plugin *operations* (host-invoked services),
/// projected onto MCP tools only if the plugin chooses to; a wallet plugin
/// should not expose them as agent-callable tools by default.
pub mod ops {
    /// Open (or provision) the wallet under the host-supplied directory.
    pub const OPEN: &str = "wallet_open";
    /// Report whether persisted wallet state exists. No network, no side effects.
    pub const STATUS: &str = "wallet_status";
    pub const BALANCE: &str = "wallet_balance";
    pub const TRANSACTIONS: &str = "wallet_transactions";
    pub const CREATE_INVOICE: &str = "wallet_create_invoice";
    pub const PAY: &str = "wallet_pay";
    pub const LOOKUP: &str = "wallet_lookup";
    /// Long-poll until a payment reaches a terminal state.
    pub const WAIT_FOR_PAYMENT: &str = "wallet_wait_for_payment";
    /// Long-poll until receiver-side arrival evidence or a terminal state.
    pub const WAIT_FOR_ARRIVAL: &str = "wallet_wait_for_arrival";

    pub const ALL: &[&str] = &[
        OPEN,
        STATUS,
        BALANCE,
        TRANSACTIONS,
        CREATE_INVOICE,
        PAY,
        LOOKUP,
        WAIT_FOR_PAYMENT,
        WAIT_FOR_ARRIVAL,
    ];
}

/// Stable identity of a specific wallet instance.
///
/// The host pins this after the first successful open and refuses to settle
/// outstanding payments against a different wallet, so a swapped seed or a
/// swapped plugin cannot silently orphan reservations.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct WalletIdentity {
    /// Plugin-defined, stable across restarts for the same wallet. For a
    /// Lightning node this is typically the node or user public key.
    pub wallet_id: String,
    /// Human-readable provider label, e.g. `"lexe"`.
    pub provider: String,
    /// `"mainnet"`, `"testnet"`, `"signet"`, `"regtest"`. The host currently
    /// requires `"mainnet"` for paid inference.
    pub network: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "plugin-server", derive(schemars::JsonSchema))]
pub struct OpenRequest {
    /// Host-owned wallet data directory. The plugin stores everything under
    /// this path and must not write elsewhere.
    pub directory: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenResponse {
    pub identity: WalletIdentity,
    /// True if this call created a brand-new wallet rather than loading one.
    pub created: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "plugin-server", derive(schemars::JsonSchema))]
pub struct StatusRequest {
    pub directory: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StatusResponse {
    /// Persisted wallet state exists under `directory`.
    pub provisioned: bool,
    /// The plugin currently holds this wallet open.
    pub open: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity: Option<WalletIdentity>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "plugin-server", derive(schemars::JsonSchema))]
pub struct Empty {}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "plugin-server", derive(schemars::JsonSchema))]
pub struct TransactionsRequest {
    pub limit: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "plugin-server", derive(schemars::JsonSchema))]
pub struct CreateInvoiceRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub amount_msat: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "plugin-server", derive(schemars::JsonSchema))]
pub struct PayRequest {
    pub invoice: Invoice,
    pub amount_msat: u64,
    pub max_total_msat: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "plugin-server", derive(schemars::JsonSchema))]
pub struct PaymentHashRequest {
    pub payment_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LookupResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transaction: Option<Transaction>,
}

/// Why a wallet operation failed. The host maps these back onto
/// [`PayError`] and keeps the "did money possibly move?" distinction intact
/// across the IPC boundary.
///
/// **Contract requirement for plugin authors:** `NotOpen`, `InvalidRequest`
/// and `NotSubmitted` may only be returned from `wallet_pay` if the plugin
/// can guarantee no payment was handed to the network. Once a payment may
/// have been submitted, every failure must be `Uncertain` (or `Failed`,
/// which the host treats identically). Getting this wrong lets the host
/// release a reservation for a payment that later settles.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WalletErrorKind {
    /// The wallet has not been opened in this plugin process. The host should
    /// call `wallet_open` and retry exactly once.
    NotOpen,
    /// Bad request from the host's point of view; nothing was attempted.
    InvalidRequest,
    /// The operation was rejected before any payment was submitted.
    NotSubmitted,
    /// A payment may have been submitted; the caller must reconcile by hash.
    Uncertain,
    /// Any other provider failure. Treated as `Uncertain` for `pay`.
    Failed,
}

/// Structured error payload carried in the operation result when
/// `is_error` is set. Plain-text errors from a plugin that does not use this
/// shape are treated as [`WalletErrorKind::Failed`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct WalletError {
    pub kind: WalletErrorKind,
    pub message: String,
}

impl WalletError {
    pub fn new(kind: WalletErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn not_open() -> Self {
        Self::new(WalletErrorKind::NotOpen, "wallet is not open")
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(WalletErrorKind::InvalidRequest, message)
    }

    pub fn failed(message: impl Into<String>) -> Self {
        Self::new(WalletErrorKind::Failed, message)
    }

    /// Best-effort decode of an operation error body. Falls back to
    /// `Failed` with the raw text so nothing is lost.
    pub fn decode(content_json: &str) -> Self {
        serde_json::from_str::<Self>(content_json).unwrap_or_else(|_| {
            // MCP-style error results wrap text in a content array.
            #[derive(Deserialize)]
            struct TextBlock {
                text: String,
            }
            let text = serde_json::from_str::<Vec<TextBlock>>(content_json)
                .ok()
                .and_then(|blocks| blocks.into_iter().next())
                .map(|block| block.text)
                .unwrap_or_else(|| content_json.to_owned());
            Self::failed(text)
        })
    }

    /// Map onto the provider-level pay error. Only `NotSubmitted` and
    /// `InvalidRequest` are safe to classify as not submitted; everything else,
    /// including transport-level ambiguity, stays uncertain.
    pub fn into_pay_error(self) -> PayError {
        let error = anyhow::anyhow!("{}", self.message);
        match self.kind {
            WalletErrorKind::NotSubmitted | WalletErrorKind::InvalidRequest => {
                PayError::NotSubmitted(error)
            }
            WalletErrorKind::NotOpen | WalletErrorKind::Uncertain | WalletErrorKind::Failed => {
                PayError::Uncertain(error)
            }
        }
    }
}

impl std::fmt::Display for WalletError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for WalletError {}

impl From<PayError> for WalletError {
    fn from(error: PayError) -> Self {
        match error {
            PayError::NotSubmitted(error) => {
                Self::new(WalletErrorKind::NotSubmitted, error.to_string())
            }
            PayError::Uncertain(error) => Self::new(WalletErrorKind::Uncertain, error.to_string()),
        }
    }
}

// `Balance` is re-exported so contract consumers need only this module.
pub type BalanceResponse = Balance;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_round_trips_as_json() {
        let error = WalletError::new(WalletErrorKind::Uncertain, "socket closed");
        let json = serde_json::to_string(&error).unwrap();
        assert_eq!(WalletError::decode(&json), error);
    }

    #[test]
    fn plain_text_error_falls_back_to_failed() {
        let decoded = WalletError::decode("boom");
        assert_eq!(decoded.kind, WalletErrorKind::Failed);
        assert_eq!(decoded.message, "boom");
    }

    #[test]
    fn mcp_text_block_error_is_unwrapped() {
        let decoded = WalletError::decode(r#"[{"type":"text","text":"nope"}]"#);
        assert_eq!(decoded.kind, WalletErrorKind::Failed);
        assert_eq!(decoded.message, "nope");
    }

    #[test]
    fn only_explicit_rejections_map_to_not_submitted() {
        for kind in [
            WalletErrorKind::NotSubmitted,
            WalletErrorKind::InvalidRequest,
        ] {
            assert!(matches!(
                WalletError::new(kind, "x").into_pay_error(),
                PayError::NotSubmitted(_)
            ));
        }
        for kind in [
            WalletErrorKind::NotOpen,
            WalletErrorKind::Uncertain,
            WalletErrorKind::Failed,
        ] {
            assert!(matches!(
                WalletError::new(kind, "x").into_pay_error(),
                PayError::Uncertain(_)
            ));
        }
    }

    #[test]
    fn op_names_are_unique_and_prefixed() {
        let mut seen = std::collections::BTreeSet::new();
        for op in ops::ALL {
            assert!(op.starts_with("wallet_"), "{op}");
            assert!(seen.insert(*op), "duplicate op {op}");
        }
    }
}
