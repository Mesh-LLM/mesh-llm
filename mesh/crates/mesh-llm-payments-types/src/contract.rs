//! The `payments.v1` plugin contract: capability, operation names and their
//! request/response shapes. The host speaks only this; any provider of the
//! capability (the in-process builtin or an external plugin) implements it.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::intent::PaymentIntent;
use crate::terms::RequestTerms;

/// Capability name the host resolves the payments engine by.
pub const CAPABILITY: &str = "payments.v1";

/// Billing semantics explicitly selected by the host for a new admission.
/// No default: old hosts and unknown revisions must fail before opening the
/// provider or creating authorization/invoices. Contract negotiation separately
/// protects new hosts from old providers that ignore unknown request fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BillingRevision {
    /// Cached-input discounts and the positive final-remainder floor.
    #[serde(rename = "v3")]
    V3,
}

/// Ledger-free declaration of the billing contract implemented by a provider.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BillingContract {
    pub revision: BillingRevision,
}

/// Operation names.
pub mod ops {
    /// Ledger-free billing contract check; incompatible providers fail closed.
    pub const BILLING_CONTRACT: &str = "billing_contract";
    /// Run one local operator command (the `/api/wallet` surface).
    pub const CONTROL: &str = "control";
    /// Advertised seller prices, and whether the provider has payments state.
    pub const PRICING: &str = "pricing";
    /// Effective intent and spendable budget for pay-first routing.
    pub const ROUTING_BUDGET: &str = "routing_budget";
    /// Reconcile uncertain charges and recover output debt; returns the
    /// approved requests the host must re-contact their sellers about.
    pub const RECONCILE: &str = "reconcile";
    /// Close a metered request the seller completed: retire an unsubmitted
    /// deposit and mark the request finished.
    pub const FINISH: &str = "finish";
    /// The operator's profile payment intent (no wallet I/O).
    pub const PAYMENT_INTENT: &str = "payment_intent";
    /// Durably approve a seller's admission: the local spending cap against
    /// policy and the daily budget, and the admitted terms every later payment of the
    /// request is checked against.
    pub const ACCEPT_ADMISSION: &str = "accept_admission";
    /// Release a request that did not (or can no longer) start paying.
    pub const CANCEL: &str = "cancel";
    /// Pay the seller's deposit invoice for an accepted admission.
    pub const PAY_DEPOSIT: &str = "pay_deposit";
    /// Check a seller's settlement against the admission and pay the final
    /// invoice; an abandoned settlement creates no charge.
    pub const SETTLE_FINAL: &str = "settle_final";

    // Seller (serving) side: token-metered billing.
    /// Admit a buyer's request at the seller's current prices: refuse a buyer
    /// at the unpaid-final-invoice strike limit, otherwise state the deposit
    /// (a flat 1 sat, applied to the final bill) or admit without one.
    pub const SERVE_ADMIT: &str = "serve_admit";
    /// Wait for the admitted deposit to settle and claim the once-only
    /// backend dispatch. Nothing runs before this returns.
    pub const SERVE_DEPOSIT_CONFIRMED: &str = "serve_deposit_confirmed";
    /// Price the backend-reported usage at the admitted rates, apply the paid
    /// deposit and issue (or return) the final invoice for the remainder.
    pub const SERVE_SETTLE: &str = "serve_settle";
    /// Close a request without usage: no bill, no invoice, no debt.
    pub const SERVE_ABANDON: &str = "serve_abandon";
    /// Answer a buyer's recovery probe: the stored bill and final invoice,
    /// never a new dispatch.
    pub const SERVE_RECOVER: &str = "serve_recover";
    /// Wait for the earliest receiver-side evidence of an invoice payment.
    pub const AWAIT_ARRIVAL: &str = "await_arrival";
    /// Wait for an invoice to settle and record it received.
    pub const SETTLE_RECEIVED: &str = "settle_received";
}

/// Bound for the ledger-only operations listed by [`deadline`].
pub const BOOKKEEPING_DEADLINE: Duration = Duration::from_secs(10);

/// How long the host waits for one `payments.v1` operation, or `None` when the
/// operation legitimately waits on the wallet and carries its own deadline.
///
/// Bookkeeping operations touch only the ledger, or start background work and
/// return: they must answer promptly. Without a bound, a stalled engine holds a
/// response, a routing decision or the serving close open forever. Operations
/// that create invoices, observe settlement, wait out a peer's prior debt or pay
/// are deliberately unbounded here — the provider bounds them (invoice expiry,
/// arrival wait, prior-settlement deadline) and the caller owns cancellation.
/// `serve_admit`, `serve_settle` and `serve_recover` may each create one
/// invoice in the wallet, which has been observed to take 30-45s on a freshly
/// started wallet; bounding them here cancelled the ledger write after the
/// wallet had already issued the invoice, leaving an orphan pending invoice
/// and a 503 for the buyer. The buyer's admission read is the bound instead.
///
/// An operation that is not listed is unbounded: a new operation must be
/// classified here, rather than silently inheriting a housekeeping deadline.
pub fn deadline(operation: &str) -> Option<Duration> {
    matches!(
        operation,
        ops::BILLING_CONTRACT
            | ops::PRICING
            | ops::FINISH
            | ops::PAYMENT_INTENT
            | ops::CANCEL
            | ops::SERVE_ABANDON
    )
    .then_some(BOOKKEEPING_DEADLINE)
}

/// Advertised seller prices, and whether the provider has any payments state.
///
/// The provider must answer without provisioning: `configured` is false for a
/// node that has never had a ledger, so the host can skip periodic recovery
/// instead of opening one to find nothing.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AdvertisedPricing {
    pub configured: bool,
    pub prices: BTreeMap<String, crate::pricing::Pricing>,
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
    /// Accepted admissions (excluding local wallet sends) whose settlement
    /// has not been completed; the host refetches each from its seller.
    pub approved: Vec<RequestTerms>,
}

/// Request for [`ops::SETTLE_FINAL`]: the seller's settlement frame, as
/// received. The provider recomputes the bill from the stored admission and
/// pays only an invoice that matches the remainder.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SettleFinalRequest {
    pub request_id: String,
    /// `None` means the seller abandoned the request: no bill, no charge.
    pub usage: Option<crate::metering::MeteredUsage>,
    pub bill_msat: Option<u64>,
    pub deposit_applied_msat: u64,
    pub invoice: Option<mesh_llm_wallet::invoice::Invoice>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SettleFinalResponse {
    /// The bill the provider verified; zero for an abandoned request.
    pub bill_msat: u64,
    /// The final payment, when a remainder was owed and paid.
    pub payment: Option<mesh_llm_wallet::provider::Transaction>,
}

/// Request for [`ops::FINISH`].
pub type FinishRequest = IdRequest;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Empty {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdRequest {
    pub id: String,
}

/// Request for [`ops::ACCEPT_ADMISSION`]. `terms` carry the buyer's local
/// cap (`max_total_msat`) and never leave this node; `admission` is what the
/// seller stated and what every payment of the request is checked against.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcceptAdmissionRequest {
    pub billing_revision: BillingRevision,
    pub terms: RequestTerms,
    pub admission: crate::metered_wire::AdmissionTerms,
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

/// Request for [`ops::PAY_DEPOSIT`]; the amount comes from the admission.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PayDepositRequest {
    pub request_id: String,
    pub invoice: mesh_llm_wallet::invoice::Invoice,
}

/// Request for [`ops::SERVE_ADMIT`]. `request_id` is the buyer's unguessable
/// request identifier; it doubles as the bearer recovery capability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServeAdmitRequest {
    pub billing_revision: BillingRevision,
    pub request_id: String,
    /// Authenticated mesh endpoint ID of the buyer.
    pub buyer: String,
    pub model: String,
    pub pricing: crate::pricing::Pricing,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ServeAdmitResponse {
    /// Pay this deposit, then call [`ops::SERVE_DEPOSIT_CONFIRMED`] before
    /// any backend work. The amount is applied against the final bill.
    Deposit {
        request_id: String,
        invoice: mesh_llm_wallet::invoice::Invoice,
        /// The seller's minimum payment, carried into the admission terms.
        #[serde(default = "crate::contract::default_payment_floor")]
        min_payment_msat: u64,
    },
    /// Buyer in good standing: dispatch now, no deposit.
    Admitted {
        request_id: String,
        #[serde(default = "crate::contract::default_payment_floor")]
        min_payment_msat: u64,
    },
    /// Refused: the buyer has too many issued and unpaid final invoices.
    Blocked { reason: String },
    /// This seller already holds the request: answer with
    /// [`ops::SERVE_RECOVER`], never a second dispatch.
    Replayed { request_id: String },
}

/// Request for [`ops::SERVE_DEPOSIT_CONFIRMED`], [`ops::SERVE_ABANDON`] and
/// [`ops::SERVE_RECOVER`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestIdRequest {
    pub request_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServeSettleRequest {
    pub request_id: String,
    pub usage: crate::metering::MeteredUsage,
}

/// The bill of a settled request. `bill = input × input_rate + output ×
/// output_rate`; `remaining = max(bill − deposit, 0)`. A deposit above the
/// bill is kept, never refunded. `invoice` is `None` when nothing remains.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalBill {
    pub usage: crate::metering::MeteredUsage,
    pub bill_msat: u64,
    /// Deposit actually received and applied.
    pub deposit_msat: u64,
    pub remaining_msat: u64,
    /// True once nothing is owed: the deposit covered the bill or the final
    /// invoice has been paid.
    pub settled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invoice: Option<mesh_llm_wallet::invoice::Invoice>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ServeRecoverResponse {
    /// Admitted but neither settled nor abandoned yet.
    Pending,
    /// Closed without usage: nothing owed, nothing to recover.
    Abandoned,
    /// The stored bill, with the issued final invoice when a remainder is owed.
    Billed { bill: FinalBill },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InvoiceRequest {
    pub invoice: mesh_llm_wallet::invoice::Invoice,
}

/// `settle_received`'s answer: the receiving wallet's own record of the
/// payment, as its `Transaction` reports it. Both members are optional, so a
/// provider that answers `{}` still conforms; the host then knows only the
/// invoice amount. Values are what the wallet reported, not checked against
/// the invoice (an overpayment shows as a credit above the invoice amount).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SettledReceived {
    /// The inbound transaction's `amount_msat`: what the wallet reports it
    /// credited. Lexe reports it net of the receive fee (995 for a 1,000 msat
    /// invoice with a 5 msat fee). Absent when the wallet reported no amount.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credited_msat: Option<u64>,
    /// The inbound transaction's `fee_msat`: the fee the wallet reports it
    /// deducted on receipt (0 when it reported none).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fee_msat: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArrivalResponse {
    /// True when a transient claiming state was seen; false when the wait
    /// ended on a terminal status. Diagnostic only.
    pub claiming: bool,
}

pub(crate) fn default_payment_floor() -> u64 {
    crate::metering::PAYMENT_FLOOR_MSAT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settled_received_accepts_a_provider_that_answers_empty() {
        let settled: SettledReceived = serde_json::from_str("{}").unwrap();
        assert_eq!((settled.credited_msat, settled.fee_msat), (None, None));
        assert_eq!(serde_json::to_string(&settled).unwrap(), "{}");
        let full = SettledReceived {
            credited_msat: Some(995),
            fee_msat: Some(5),
        };
        assert_eq!(
            serde_json::to_string(&full).unwrap(),
            r#"{"credited_msat":995,"fee_msat":5}"#
        );
    }

    /// Every operation the host may invoke, so a new one cannot ship without
    /// being classified as bounded or as a wallet wait.
    const ALL_OPS: &[&str] = &[
        ops::BILLING_CONTRACT,
        ops::CONTROL,
        ops::PRICING,
        ops::ROUTING_BUDGET,
        ops::RECONCILE,
        ops::FINISH,
        ops::PAYMENT_INTENT,
        ops::ACCEPT_ADMISSION,
        ops::CANCEL,
        ops::PAY_DEPOSIT,
        ops::SETTLE_FINAL,
        ops::SERVE_ADMIT,
        ops::SERVE_DEPOSIT_CONFIRMED,
        ops::SERVE_SETTLE,
        ops::SERVE_ABANDON,
        ops::SERVE_RECOVER,
        ops::AWAIT_ARRIVAL,
        ops::SETTLE_RECEIVED,
    ];

    #[test]
    fn only_ledger_bookkeeping_operations_are_bounded() {
        let bounded: Vec<&str> = ALL_OPS
            .iter()
            .copied()
            .filter(|operation| deadline(operation).is_some())
            .collect();
        assert_eq!(
            bounded,
            [
                ops::BILLING_CONTRACT,
                ops::PRICING,
                ops::FINISH,
                ops::PAYMENT_INTENT,
                ops::CANCEL,
                ops::SERVE_ABANDON,
            ]
        );
    }

    #[test]
    fn an_unclassified_operation_is_unbounded() {
        // A new operation must be classified here rather than silently
        // inheriting a housekeeping deadline.
        assert!(deadline("a_future_operation").is_none());
    }
}
