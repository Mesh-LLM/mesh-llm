use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::invoice::Invoice;

/// Whether a failed call could have submitted a payment. Transport errors
/// after submission must remain uncertain, even when lookup finds no record yet.
#[derive(Debug)]
pub enum PayError {
    NotSubmitted(anyhow::Error),
    Uncertain(anyhow::Error),
}

impl std::fmt::Display for PayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSubmitted(error) => write!(f, "payment not submitted: {error}"),
            Self::Uncertain(error) => write!(f, "payment outcome uncertain: {error}"),
        }
    }
}

impl std::error::Error for PayError {}

// Unclassified provider errors must fail conservatively.
impl From<anyhow::Error> for PayError {
    fn from(error: anyhow::Error) -> Self {
        Self::Uncertain(error)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Balance {
    pub spendable_msat: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PaymentStatus {
    Pending,
    Succeeded,
    Failed,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Transaction {
    pub id: String,
    pub payment_hash: Option<String>,
    pub inbound: bool,
    pub amount_msat: u64,
    pub fee_msat: u64,
    pub status: PaymentStatus,
    /// Normalized receiver-side arrival evidence for an inbound payment that
    /// is still `Pending`: the HTLC has been irrevocably committed to this
    /// node and it is claiming the funds. Providers set this only when their
    /// API exposes such a state; `false` means "not observed", never "not
    /// arrived". Meaningless for outbound or terminal payments, and no
    /// substitute for `Succeeded`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub claiming: bool,
    /// Provider-specific detailed status for display and diagnostics only.
    /// Nothing in the payment path may branch on its contents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status_msg: Option<String>,
    pub created_at_ms: u64,
    pub settled_at_ms: Option<u64>,
    /// The payment's preimage as 64 hex digits, for a `Succeeded` payment
    /// whose provider reports one. A payer holds it only once the payee
    /// released it, so it proves the payment completed. Providers leave it
    /// `None` for pending and failed payments, and when their API has none.
    /// Besides the host's settlement events, the operator sees it through the
    /// wallet's local control API.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preimage: Option<String>,
}

/// Like a derived `Debug`, but `preimage` prints as `<redacted>`: it is bearer
/// proof of payment and must not reach logs.
impl std::fmt::Debug for Transaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transaction")
            .field("id", &self.id)
            .field("payment_hash", &self.payment_hash)
            .field("inbound", &self.inbound)
            .field("amount_msat", &self.amount_msat)
            .field("fee_msat", &self.fee_msat)
            .field("status", &self.status)
            .field("claiming", &self.claiming)
            .field("status_msg", &self.status_msg)
            .field("created_at_ms", &self.created_at_ms)
            .field("settled_at_ms", &self.settled_at_ms)
            .field("preimage", &self.preimage.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

impl Transaction {
    /// Receiver-side evidence that the HTLC arrived and the node is claiming
    /// it. Advisory only: it is a transient state and may be missed entirely.
    pub fn is_claiming(&self) -> bool {
        self.inbound && self.status == PaymentStatus::Pending && self.claiming
    }
}

#[async_trait]
pub trait WalletProvider: Send + Sync {
    async fn balance(&self) -> Result<Balance>;
    async fn transactions(&self, limit: usize) -> Result<Vec<Transaction>>;
    /// Create a BOLT11 invoice that expires `expiry_secs` from now.
    ///
    /// The caller decides how long the invoice stays payable; a provider must
    /// not substitute its own default. Implementations may clamp to the
    /// provider's supported range but must not silently lengthen a short
    /// expiry: the host relies on expiry to bound how long an unpaid inference
    /// invoice pins state and to make a late payment fail at the payee rather
    /// than land unexpectedly.
    async fn create_invoice(&self, amount_msat: Option<u64>, expiry_secs: u32) -> Result<Invoice>;

    /// Start or recover payment of this invoice. An implementation that can
    /// bound routing fees must not initiate a payment whose amount plus fees
    /// exceeds `max_total_msat`, and refuses it as `NotSubmitted`. One that
    /// cannot must pass the headroom to its wallet where it can and report
    /// the fee actually paid: the host records that fee against the daily
    /// budget either way.
    /// `NotSubmitted` guarantees this call did not submit a payment. All other
    /// errors are uncertain and callers must reconcile by hash. The service
    /// may call `pay` again for the same invoice after an uncertain outcome;
    /// providers must treat a repeat as a lookup of the existing payment, not
    /// a new payment (Lexe does, from SDK 0.1.24).
    async fn pay(
        &self,
        invoice: &Invoice,
        amount_msat: u64,
        max_total_msat: u64,
    ) -> Result<Transaction, PayError>;

    /// Query the authoritative wallet, not only a potentially stale local cache.
    async fn lookup(&self, payment_hash: &str) -> Result<Option<Transaction>>;

    /// Wait for this incoming or outgoing payment to succeed or fail.
    ///
    /// Return an authoritative terminal transaction for the requested hash,
    /// including when settlement preceded this call. Event-backed providers must
    /// subscribe before checking current state (or use a replayable subscription)
    /// so settlement cannot be lost between lookup and subscription. Providers
    /// without notifications may poll internally.
    ///
    /// Multiple waiters must be supported. Dropping this future only stops
    /// observation; it must not cancel or retry a payment. Errors or subscription
    /// loss leave the outcome uncertain and callers retain durable reservations.
    /// Invoice expiry alone must not terminate an outgoing in-flight payment;
    /// callers own any deadline for waiting on an unpaid incoming invoice.
    async fn wait_for_payment(&self, payment_hash: &str) -> Result<Transaction>;

    /// Wait for the earliest receiver-side evidence that this incoming payment
    /// has arrived: either a transient claiming state or a terminal status.
    ///
    /// This is a latency optimization for opening a work gate, never a
    /// settlement record. Callers must still await `wait_for_payment` before
    /// recording the payment as received. The claiming state is transient and
    /// may be missed; the default implementation simply waits for a terminal
    /// status, which is always a correct (if slower) answer.
    async fn wait_for_arrival(&self, payment_hash: &str) -> Result<Transaction> {
        self.wait_for_payment(payment_hash).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wallet plugin built before `preimage` existed still decodes, and a
    /// transaction without one encodes exactly as before.
    #[test]
    fn a_transaction_without_a_preimage_is_unchanged_on_the_wire() {
        let json = r#"{"id":"t","payment_hash":"h","inbound":false,"amount_msat":1,"fee_msat":0,"status":"succeeded","created_at_ms":0,"settled_at_ms":1}"#;
        let transaction: Transaction = serde_json::from_str(json).unwrap();
        assert_eq!(transaction.preimage, None);
        assert_eq!(serde_json::to_string(&transaction).unwrap(), json);
    }

    #[test]
    fn debug_never_prints_a_preimage() {
        let json = r#"{"id":"t","payment_hash":"h","inbound":false,"amount_msat":1,"fee_msat":0,"status":"succeeded","created_at_ms":0,"settled_at_ms":1,"preimage":"secret-preimage"}"#;
        let transaction: Transaction = serde_json::from_str(json).unwrap();
        let debug = format!("{transaction:?}");
        assert!(!debug.contains("secret-preimage"));
        assert!(debug.contains("<redacted>") && debug.contains("payment_hash"));
    }
}
