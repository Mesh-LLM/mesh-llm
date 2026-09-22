//! Best-effort payer observations; never a ledger, replay service or payment authority.
use mesh_llm_payments::{invoice::Invoice, ledger::RequestTerms, wallet::Transaction};
use serde::Serialize;
use tokio::sync::mpsc;

use crate::{mesh::Node, plugin::openai_exchange::request_body_digest};

const CHANNEL: &str = "payment.lifecycle.v1";

#[derive(Clone, Serialize)]
struct Event {
    exchange_id: String,
    event_ref: String,
    terms_digest: String,
    phase: &'static str,
    source: &'static str,
    settlement: Option<&'static str>,
    segment: Option<u32>,
    payment_hash: Option<String>,
    amount_msat: u64,
}

/// One bounded queue per observed paid exchange. No task or hashing without a subscriber.
#[derive(Default)]
pub(super) struct Observations {
    sender: Option<mpsc::Sender<Event>>,
    exchange_id: String,
    terms_digest: String,
}

impl Observations {
    pub(super) async fn for_exchange(
        evidence: Option<&(Node, String)>,
        terms: &RequestTerms,
    ) -> Self {
        match evidence {
            Some((node, _)) => Self::new(node, terms).await.unwrap_or_default(),
            None => Self::default(),
        }
    }

    async fn new(node: &Node, terms: &RequestTerms) -> Option<Self> {
        let exchange_id = terms.exchange_id.clone()?;
        let manager = node.plugin_manager().await?;
        if !manager.any_plugin_declares_mesh_channel(CHANNEL).await {
            return None;
        }
        let terms_digest = terms_digest(terms)?;
        let (sender, mut receiver) = mpsc::channel::<Event>(8);
        tokio::spawn(async move {
            while let Some(event) = receiver.recv().await {
                let Ok(body) = serde_json::to_vec(&event) else {
                    continue;
                };
                let _ = tokio::time::timeout(
                    std::time::Duration::from_secs(1),
                    manager.broadcast_channel_message(
                        CHANNEL,
                        "application/json",
                        body,
                        &event.exchange_id,
                    ),
                )
                .await;
            }
        });
        Some(Self {
            sender: Some(sender),
            exchange_id,
            terms_digest,
        })
    }

    pub(super) fn accepted(&self, cap: u64) {
        self.emit("terms_accepted", "payer_asserted", None, None, cap);
    }

    pub(super) fn invoice(&self, segment: u32, invoice: &Invoice) {
        self.emit(
            if segment == 0 {
                "input_invoice_issued"
            } else {
                "output_invoice_issued"
            },
            "provider_asserted",
            Some(segment),
            Some(invoice.payment_hash.clone()),
            invoice.amount_msat.unwrap_or(0),
        );
    }

    pub(super) fn settled(&self, segment: u32, transaction: &Transaction) {
        if transaction.status != mesh_llm_payments::wallet::PaymentStatus::Succeeded {
            return;
        }
        self.emit(
            if segment == 0 {
                "input_settlement_observed"
            } else {
                "output_settlement_observed"
            },
            "wallet_reported",
            Some(segment),
            transaction.payment_hash.clone(),
            transaction.amount_msat,
        );
    }

    pub(super) fn final_amount(&self, amount: u64) {
        self.emit("final_accounted", "payer_asserted", None, None, amount);
    }

    fn emit(
        &self,
        phase: &'static str,
        source: &'static str,
        segment: Option<u32>,
        payment_hash: Option<String>,
        amount_msat: u64,
    ) {
        let Some(sender) = &self.sender else { return };
        let mut event = Event {
            exchange_id: self.exchange_id.clone(),
            event_ref: String::new(),
            terms_digest: self.terms_digest.clone(),
            phase,
            source,
            settlement: (source == "wallet_reported").then_some("terminal"),
            segment,
            payment_hash,
            amount_msat,
        };
        let Ok(value) = serde_json::to_value(&event) else {
            return;
        };
        let Some(reference) = request_body_digest(&value, None) else {
            return;
        };
        event.event_ref = reference;
        // A full queue means incomplete evidence, not failed inference.
        let _ = sender.try_send(event);
    }
}

fn terms_digest(terms: &RequestTerms) -> Option<String> {
    // Public terms only: never hash/serialize the bearer recovery ID or local peer field.
    request_body_digest(
        &serde_json::json!({
            "exchange_id": terms.exchange_id, "payee": terms.payee, "model": terms.model,
            "pricing": terms.pricing, "input_tokens": terms.input_tokens,
            "max_output_tokens": terms.max_output_tokens, "max_total_msat": terms.max_total_msat,
            "expires_at_ms": terms.expires_at_ms,
        }),
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms() -> RequestTerms {
        RequestTerms {
            id: "private-recovery".into(),
            exchange_id: Some("exchange-one".into()),
            peer: "private-peer".into(),
            payee: Some("payee".into()),
            model: "model".into(),
            pricing: mesh_llm_payments::pricing::Pricing {
                input_msat_per_million: 1000,
                output_msat_per_million: 2000,
                minimum_invoice_msat: 1,
            },
            input_tokens: 10,
            max_output_tokens: 20,
            max_total_msat: 100,
            expires_at_ms: 1000,
        }
    }

    #[test]
    fn digest_is_checked_and_excludes_private_capability() {
        let mut request = terms();
        let digest = terms_digest(&request).unwrap();
        request.id = "different-secret".into();
        request.peer = "different-local-peer".into();
        assert_eq!(terms_digest(&request).unwrap(), digest);
        request.max_total_msat = u64::MAX;
        assert!(terms_digest(&request).is_none());
    }

    #[test]
    fn event_contract_is_private_stable_and_bounded() {
        let (sender, mut receiver) = mpsc::channel(8);
        let observations = Observations {
            sender: Some(sender),
            exchange_id: "exchange-one".into(),
            terms_digest: terms_digest(&terms()).unwrap(),
        };
        observations.accepted(100);
        observations.accepted(100);
        let first = receiver.try_recv().unwrap();
        let second = receiver.try_recv().unwrap();
        assert_eq!(first.event_ref, second.event_ref);
        assert_eq!(first.exchange_id, "exchange-one");
        assert_eq!(first.source, "payer_asserted");
        let json = serde_json::to_string(&first).unwrap();
        for secret in [
            "private-recovery",
            "private-peer",
            "preimage",
            "bolt11",
            "prompt",
        ] {
            assert!(!json.contains(secret));
        }
        for _ in 0..20 {
            observations.accepted(100);
        }
        assert_eq!(receiver.len(), 8);
    }

    #[test]
    fn pending_and_failed_never_emit_settlement() {
        let (sender, mut receiver) = mpsc::channel(8);
        let observations = Observations {
            sender: Some(sender),
            exchange_id: "exchange-two".into(),
            terms_digest: "digest".into(),
        };
        let mut payment = Transaction {
            id: "wallet-private".into(),
            payment_hash: Some("public-hash".into()),
            inbound: false,
            amount_msat: 10,
            fee_msat: 1,
            status: mesh_llm_payments::wallet::PaymentStatus::Pending,
            claiming: false,
            status_msg: None,
            created_at_ms: 0,
            settled_at_ms: None,
        };
        observations.settled(0, &payment);
        payment.status = mesh_llm_payments::wallet::PaymentStatus::Failed;
        observations.settled(0, &payment);
        assert!(receiver.try_recv().is_err());
        payment.status = mesh_llm_payments::wallet::PaymentStatus::Succeeded;
        observations.settled(0, &payment);
        let event = receiver.try_recv().unwrap();
        assert_eq!(event.phase, "input_settlement_observed");
        assert_eq!(event.settlement, Some("terminal"));
        assert_eq!(event.exchange_id, "exchange-two");
        assert_eq!(event.payment_hash.as_deref(), Some("public-hash"));
    }
}
