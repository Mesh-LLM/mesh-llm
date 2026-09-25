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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinishRequest {
    pub id: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Empty {}
