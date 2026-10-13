//! Coarse per-request operations behind the `payments.v1` contract. The host
//! calls these through the plugin; they compose the ledger and wallet so no
//! fine-grained ledger call crosses the capability boundary.

use anyhow::Result;
use std::sync::Arc;

use mesh_llm_payments_types::RequestTerms;
use mesh_llm_payments_types::contract::{
    CancelRequest, CancelStage, ReconcileResponse, RoutingBudgetRequest, RoutingBudgetResponse,
};

use crate::intent::PaymentIntent;
use crate::service::PaymentService;

impl PaymentService {
    /// Effective intent for one request and the budget it may spend. Never
    /// provisions a wallet, and only reads the balance when paid is permitted.
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
            self.spendable_budget().await
        };
        RoutingBudgetResponse {
            intent,
            available_msat,
        }
    }

    async fn spendable_budget(&self) -> u64 {
        let Ok(wallet) = self.wallet().await else {
            return 0;
        };
        let Ok(balance) = wallet.balance().await else {
            return 0;
        };
        self.ledger
            .available_budget(balance.spendable_msat, crate::now_ms())
            .unwrap_or(0)
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

    /// Start the balance read for request `id`. The table is bounded: when
    /// full, completed reads are dropped, and if still full the read is
    /// skipped and `authorize` reads the balance itself.
    pub fn prefetch(self: &Arc<Self>, id: String) -> Result<()> {
        const MAX_PREFETCHED: usize = 256;
        let handle = self.prefetch_balance();
        let mut prefetched = self
            .prefetched
            .lock()
            .map_err(|_| anyhow::anyhow!("prefetch table poisoned"))?;
        if prefetched.len() >= MAX_PREFETCHED {
            prefetched.retain(|_, handle| !handle.is_finished());
        }
        if prefetched.len() < MAX_PREFETCHED {
            prefetched.insert(id, handle);
        }
        Ok(())
    }

    fn take_prefetched(
        &self,
        id: &str,
    ) -> Option<tokio::task::JoinHandle<Result<crate::wallet::Balance>>> {
        self.prefetched.lock().ok()?.remove(id)
    }

    /// Propose and approve `terms`, using the prefetched balance when present.
    pub(crate) async fn authorize_terms(&self, terms: &RequestTerms) -> Result<()> {
        let prefetched = self.take_prefetched(&terms.id);
        self.await_authorization_with(terms, async {
            match prefetched {
                Some(handle) => handle.await?,
                None => self.wallet().await?.balance().await,
            }
        })
        .await
    }

    pub fn cancel(&self, request: CancelRequest) -> Result<()> {
        drop(self.take_prefetched(&request.id));
        match request.stage {
            CancelStage::Unstarted => self.ledger.cancel_unstarted(&request.id),
            CancelStage::Authorization => self.ledger.fail_authorization_if_idle(&request.id),
        }
    }
}
