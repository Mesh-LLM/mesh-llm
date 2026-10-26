//! Coarse per-request operations behind the `payments.v1` contract. The host
//! calls these through the plugin; they compose the ledger and wallet so no
//! fine-grained ledger call crosses the capability boundary.

use anyhow::Result;

use mesh_llm_payments_types::RequestTerms;
use mesh_llm_payments_types::contract::{
    CancelRequest, CancelStage, ReconcileResponse, RoutingBudgetRequest, RoutingBudgetResponse,
};

use crate::intent::PaymentIntent;
use crate::service::PaymentService;

impl PaymentService {
    /// Effective intent for one request and the budget it may spend. Never
    /// provisions a wallet or reads its balance.
    pub async fn routing_budget(&self, request: RoutingBudgetRequest) -> RoutingBudgetResponse {
        let profile = self.ledger.payment_intent().unwrap_or_default();
        let intent = match request.request_intent {
            Some(value) => match serde_json::from_value::<PaymentIntent>(value) {
                Ok(request) if request.validate().is_ok() => profile.restrict(&request),
                _ => PaymentIntent::FreeOnly,
            },
            None => profile,
        };
        let available_msat = if matches!(intent, PaymentIntent::FreeOnly) || !self.has_wallet() {
            0
        } else {
            self.ledger.available_budget(crate::now_ms()).unwrap_or(0)
        };
        RoutingBudgetResponse {
            intent,
            available_msat,
        }
    }

    /// Reconcile financial state only. One uncertain charge or unreachable
    /// wallet lookup must not block other debts, so those failures are ignored.
    pub async fn reconcile(&self) -> Result<ReconcileResponse> {
        let _ = self.reconcile_pending().await;
        let _ = self.metered_refresh_all().await;
        let approved = self
            .ledger
            .requests()?
            .into_iter()
            .filter(|request| request.state == "approved" && request.terms.peer != "wallet-send")
            .map(|request| request.terms)
            .collect();
        Ok(ReconcileResponse { approved })
    }

    /// Propose and approve `terms` against policy and the daily budget.
    pub(crate) async fn authorize_terms(&self, terms: &RequestTerms) -> Result<()> {
        self.await_authorization(terms).await
    }

    pub fn cancel(&self, request: CancelRequest) -> Result<()> {
        match request.stage {
            CancelStage::Unstarted => self.ledger.cancel_unstarted(&request.id),
            CancelStage::Authorization => self.ledger.fail_authorization_if_idle(&request.id),
        }
    }
}
