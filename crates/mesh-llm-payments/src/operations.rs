//! Coarse per-request operations behind the `payments.v1` contract. The host
//! calls these through the plugin; they compose the ledger and wallet so no
//! fine-grained ledger call crosses the capability boundary.

use anyhow::{Result, ensure};
use mesh_llm_payments_types::contract::{
    ReconcileResponse, RoutingBudgetRequest, RoutingBudgetResponse, SettleOutputRequest,
};

use crate::intent::PaymentIntent;
use crate::ledger::Charge;
use crate::pricing::payment_cap_msat;
use crate::service::PaymentService;
use crate::wallet::Transaction;

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

    /// Reconcile financial state only. One uncertain charge or unavailable
    /// invoice must not block other debts, so those failures are ignored.
    pub async fn reconcile(&self) -> Result<ReconcileResponse> {
        let _ = self.reconcile_pending().await;
        let _ = self.recover_output_debt().await;
        let approved = self
            .ledger
            .requests()?
            .into_iter()
            .filter(|request| request.state == "approved" && request.terms.peer != "wallet-send")
            .map(|request| request.terms)
            .collect();
        Ok(ReconcileResponse { approved })
    }

    /// Validate a seller's output invoice against the request terms and pay it.
    pub async fn settle_output(&self, request: SettleOutputRequest) -> Result<Transaction> {
        let SettleOutputRequest {
            terms,
            tokens,
            invoice,
        } = request;
        ensure!(
            tokens > 0 && tokens <= terms.max_output_tokens,
            "output token allowance exceeded"
        );
        ensure!(
            terms.payee.as_deref() == Some(invoice.payee.as_str()),
            "output invoice changed receiving wallet"
        );
        let amount_msat = terms.pricing.output_charge(tokens)?;
        ensure!(
            invoice.amount_msat == Some(amount_msat),
            "output invoice amount mismatch"
        );
        self.pay_charge(&Charge {
            request_id: terms.id.clone(),
            segment: 1,
            invoice,
            amount_msat,
            max_total_msat: payment_cap_msat(amount_msat)?,
        })
        .await
    }
}
