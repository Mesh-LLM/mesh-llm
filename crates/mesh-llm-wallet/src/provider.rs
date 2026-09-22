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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Transaction {
    pub id: String,
    pub payment_hash: Option<String>,
    pub inbound: bool,
    pub amount_msat: u64,
    pub fee_msat: u64,
    pub status: PaymentStatus,
    /// Provider-specific detailed status, preserved verbatim. The coarse
    /// `status` collapses distinct provider states (Lexe maps both
    /// `InvoiceGenerated` and `Claiming` to `Pending`), so this field is the
    /// only way to observe receiver-side HTLC arrival.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status_msg: Option<String>,
    pub created_at_ms: u64,
    pub settled_at_ms: Option<u64>,
}

impl Transaction {
    /// Receiver-side evidence that the HTLC arrived and the node is claiming
    /// it. Advisory only: it is a transient state and may be missed entirely.
    pub fn is_claiming(&self) -> bool {
        self.inbound
            && self.status == PaymentStatus::Pending
            && self.status_msg.as_deref() == Some("claiming")
    }
}

#[async_trait]
pub trait WalletProvider: Send + Sync {
    async fn balance(&self) -> Result<Balance>;
    async fn transactions(&self, limit: usize) -> Result<Vec<Transaction>>;
    async fn create_invoice(&self, amount_msat: Option<u64>) -> Result<Invoice>;

    /// Start or recover payment of this invoice. The implementation must not
    /// initiate a payment whose amount plus fees exceeds `max_total_msat`.
    /// `NotSubmitted` guarantees this call did not submit a payment. All other
    /// errors are uncertain and callers must reconcile by hash. The service
    /// never calls this again for an attempt with an uncertain outcome.
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
