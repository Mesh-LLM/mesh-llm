//! The `payments.v1` plugin contract: capability, operation names and their
//! request/response shapes. The host speaks only this; any provider of the
//! capability (the in-process builtin or an external plugin) implements it.

use serde::{Deserialize, Serialize};

use crate::intent::PaymentIntent;
use crate::terms::RequestTerms;

/// Capability name the host resolves the payments engine by.
pub const CAPABILITY: &str = "payments.v1";

/// Operation names.
pub mod ops {
    /// Run one local operator command (the `/api/wallet` surface).
    pub const CONTROL: &str = "control";
    /// Effective intent and spendable budget for pay-first routing.
    pub const ROUTING_BUDGET: &str = "routing_budget";
    /// Reconcile uncertain charges and recover output debt; returns the
    /// approved requests the host must re-contact their sellers about.
    pub const RECONCILE: &str = "reconcile";
    /// Validate and pay a seller's output invoice for a request.
    pub const SETTLE_OUTPUT: &str = "settle_output";
    /// Mark a request finished.
    pub const FINISH: &str = "finish";
    /// The operator's profile payment intent (no wallet I/O).
    pub const PAYMENT_INTENT: &str = "payment_intent";
    /// Start reading the balance for a request so it overlaps seller prefill.
    pub const PREFETCH: &str = "prefetch";
    /// Durably propose and approve request terms against policy and balance.
    pub const AUTHORIZE: &str = "authorize";
    /// Release a request that did not (or can no longer) start paying.
    pub const CANCEL: &str = "cancel";
    /// Pay the seller's input invoice for an authorized request.
    pub const PAY_INPUT: &str = "pay_input";

    // Seller (serving) side.
    /// Check prices, wait out the peer's prior debt, and open serving.
    pub const SERVE_BEGIN: &str = "serve_begin";
    /// Fix the output allowance and issue the input invoice.
    pub const SERVE_INPUT_INVOICE: &str = "serve_input_invoice";
    /// Wait for the earliest receiver-side evidence of an invoice payment.
    pub const AWAIT_ARRIVAL: &str = "await_arrival";
    /// Wait for an invoice to settle and record it received.
    pub const SETTLE_RECEIVED: &str = "settle_received";
    /// Raise the delivered-token watermark of a serving request.
    pub const RECORD_DELIVERED: &str = "record_delivered";
    /// Close a serving request's accounting.
    pub const SERVE_FINISH: &str = "serve_finish";
    /// Issue (or return) the output invoice of a finished serving request.
    pub const OUTPUT_RECEIVABLE: &str = "output_receivable";
    /// Answer a payer's recovery probe for a serving request.
    pub const SERVE_RECOVER: &str = "serve_recover";
}

/// Error body of a failed operation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpError {
    pub message: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RoutingBudgetRequest {
    /// The caller's raw `mesh_payment` request field, if present. The provider
    /// validates it; an invalid value restricts the request to free only.
    #[serde(default)]
    pub request_intent: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RoutingBudgetResponse {
    /// Profile intent restricted by the request intent.
    pub intent: PaymentIntent,
    /// Spendable budget now; zero when free only or no wallet is provisioned.
    pub available_msat: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReconcileResponse {
    /// Approved requests (excluding local wallet sends) still owing output.
    pub approved: Vec<RequestTerms>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SettleOutputRequest {
    pub terms: RequestTerms,
    pub tokens: u64,
    pub invoice: mesh_llm_wallet::invoice::Invoice,
}

/// Request for [`ops::FINISH`].
pub type FinishRequest = IdRequest;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Empty {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdRequest {
    pub id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthorizeRequest {
    pub terms: RequestTerms,
}

/// Which release a [`ops::CANCEL`] performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelStage {
    /// Before approval: reject the request if no charge exists.
    Unstarted,
    /// After approval: fail the authorization if no charge is in flight.
    Authorization,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CancelRequest {
    pub id: String,
    pub stage: CancelStage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PayInputRequest {
    pub terms: RequestTerms,
    pub invoice: mesh_llm_wallet::invoice::Invoice,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServeBeginRequest {
    pub id: String,
    pub peer: String,
    pub model: String,
    pub pricing: crate::pricing::Pricing,
    pub max_output: u64,
    /// How long to wait for the peer's prior debt before refusing.
    pub prior_settlement_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServeInputInvoiceRequest {
    pub id: String,
    pub peer: String,
    pub model: String,
    pub pricing: crate::pricing::Pricing,
    pub input_tokens: u64,
    pub max_output_tokens: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InvoiceRequest {
    pub invoice: mesh_llm_wallet::invoice::Invoice,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArrivalResponse {
    /// True when a transient claiming state was seen; false when the wait
    /// ended on a terminal status. Diagnostic only.
    pub claiming: bool,
}

/// Request for [`ops::SERVE_FINISH`]: raise the delivered-token watermark to
/// `tokens` and close serving accounting in one atomic step, so a close can
/// never freeze a watermark lower than what was delivered.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServeFinishRequest {
    pub id: String,
    pub tokens: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordDeliveredRequest {
    pub id: String,
    pub tokens: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputInvoice {
    pub tokens: u64,
    pub invoice: mesh_llm_wallet::invoice::Invoice,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutputReceivableResponse {
    pub output: Option<OutputInvoice>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ServeRecoverResponse {
    Pending,
    Complete { output: Option<OutputInvoice> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputInvoiceResponse {
    pub terms: RequestTerms,
    pub invoice: mesh_llm_wallet::invoice::Invoice,
}
