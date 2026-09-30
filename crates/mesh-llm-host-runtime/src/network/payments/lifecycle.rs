//! Best-effort payment lifecycle observations, payer and provider side; never
//! a ledger, replay service or payment authority.
use mesh_llm_payments_types::RequestTerms;
use mesh_llm_wallet::{invoice::Invoice, provider::Transaction};
use serde::Serialize;
use tokio::sync::mpsc;

use crate::{mesh::Node, plugin::PluginManager, plugin::openai_exchange::request_body_digest};

const CHANNEL: &str = "payment.lifecycle.v1";

/// Which side of a paid exchange observed an event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Role {
    Payer,
    Provider,
}

impl Role {
    /// The source a side's own assertion is labeled with.
    fn asserted(self) -> &'static str {
        match self {
            Self::Payer => "payer_asserted",
            Self::Provider => "provider_asserted",
        }
    }
}

#[derive(Clone, Serialize)]
struct Event {
    exchange_id: String,
    event_ref: String,
    terms_digest: String,
    role: Role,
    phase: &'static str,
    source: &'static str,
    settlement: Option<&'static str>,
    segment: Option<u32>,
    payment_hash: Option<String>,
    amount_msat: u64,
    tokens: Option<u64>,
}

/// One bounded queue per observed paid exchange. No task or hashing without a subscriber.
#[derive(Clone, Default)]
pub(crate) struct Observations {
    sender: Option<mpsc::Sender<Event>>,
    exchange_id: String,
    terms_digest: String,
    role: Option<Role>,
}

impl Observations {
    /// Payer side: joins on the host's OpenAI exchange ID carried in `terms`.
    pub(crate) async fn for_exchange(
        evidence: Option<&(Node, String)>,
        terms: &RequestTerms,
    ) -> Self {
        let Some((node, _)) = evidence else {
            return Self::default();
        };
        let Some(exchange_id) = terms.exchange_id.clone() else {
            return Self::default();
        };
        let Some(manager) = node.plugin_manager().await else {
            return Self::default();
        };
        if !manager.any_plugin_declares_mesh_channel(CHANNEL).await {
            return Self::default();
        }
        Self::start(manager, Role::Payer, exchange_id, terms).unwrap_or_default()
    }

    fn start(
        manager: PluginManager,
        role: Role,
        exchange_id: String,
        terms: &RequestTerms,
    ) -> Option<Self> {
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
            role: Some(role),
        })
    }

    /// Terms accepted by this side; `cap` is the terms' `max_total_msat`.
    pub(crate) fn accepted(&self, cap: u64) {
        let Some(role) = self.role else { return };
        self.emit("terms_accepted", role.asserted(), None, None, cap, None);
    }

    /// An invoice for `segment` (0 = input, 1 = output). The provider issues
    /// every invoice, so this is `provider_asserted` on both sides.
    pub(crate) fn invoice(&self, segment: u32, invoice: &Invoice) {
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
            None,
        );
    }

    /// Payer side: the wallet's own record of an outgoing payment. Only a
    /// succeeded payment is a settlement.
    pub(crate) fn settled(&self, segment: u32, transaction: &Transaction) {
        if transaction.status != mesh_llm_wallet::provider::PaymentStatus::Succeeded {
            return;
        }
        self.settlement(
            segment,
            transaction.payment_hash.clone(),
            transaction.amount_msat,
        );
    }

    /// Provider side: the receiving wallet reported `invoice` settled and the
    /// provider recorded it received.
    pub(crate) fn received(&self, segment: u32, invoice: &Invoice) {
        self.settlement(
            segment,
            Some(invoice.payment_hash.clone()),
            invoice.amount_msat.unwrap_or(0),
        );
    }

    /// Provider side: the delivered-token watermark written when serving closed.
    pub(crate) fn delivered(&self, tokens: u64) {
        self.emit(
            "delivered",
            "provider_asserted",
            None,
            None,
            0,
            Some(tokens),
        );
    }

    pub(crate) fn final_amount(&self, amount: u64) {
        self.emit(
            "final_accounted",
            "payer_asserted",
            None,
            None,
            amount,
            None,
        );
    }

    fn settlement(&self, segment: u32, payment_hash: Option<String>, amount_msat: u64) {
        self.emit(
            if segment == 0 {
                "input_settlement_observed"
            } else {
                "output_settlement_observed"
            },
            "wallet_reported",
            Some(segment),
            payment_hash,
            amount_msat,
            None,
        );
    }

    fn emit(
        &self,
        phase: &'static str,
        source: &'static str,
        segment: Option<u32>,
        payment_hash: Option<String>,
        amount_msat: u64,
        tokens: Option<u64>,
    ) {
        let (Some(sender), Some(role)) = (&self.sender, self.role) else {
            return;
        };
        let mut event = Event {
            exchange_id: self.exchange_id.clone(),
            event_ref: String::new(),
            terms_digest: self.terms_digest.clone(),
            role,
            phase,
            source,
            settlement: (source == "wallet_reported").then_some("terminal"),
            segment,
            payment_hash,
            amount_msat,
            tokens,
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

/// Provider side, resolved once per serving request before anything is
/// priced: holds the plugin manager only when a loaded plugin declares the
/// channel, so a node with no subscriber never generates an ID or hashes.
#[derive(Clone, Default)]
pub(crate) struct ProviderLifecycle(Option<PluginManager>);

impl ProviderLifecycle {
    pub(crate) async fn for_node(node: &Node) -> Self {
        let Some(manager) = node.plugin_manager().await else {
            return Self::default();
        };
        if !manager.any_plugin_declares_mesh_channel(CHANNEL).await {
            return Self::default();
        }
        Self(Some(manager))
    }

    /// Start observing a serving request once its input invoice exists. The
    /// provider has no payer-side exchange to join: it names the exchange
    /// with its own fresh ID, never the private request (recovery) ID.
    pub(crate) fn observe(&self, terms: &RequestTerms) -> Observations {
        let Some(manager) = self.0.clone() else {
            return Observations::default();
        };
        let exchange_id = uuid::Uuid::new_v4().to_string();
        let mut terms = terms.clone();
        terms.exchange_id = Some(exchange_id.clone());
        Observations::start(manager, Role::Provider, exchange_id, &terms).unwrap_or_default()
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
            pricing: mesh_llm_payments_types::pricing::Pricing {
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

    fn observing(role: Role, exchange_id: &str) -> (Observations, mpsc::Receiver<Event>) {
        let (sender, receiver) = mpsc::channel(8);
        (
            Observations {
                sender: Some(sender),
                exchange_id: exchange_id.into(),
                terms_digest: terms_digest(&terms()).unwrap(),
                role: Some(role),
            },
            receiver,
        )
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
        let (observations, mut receiver) = observing(Role::Payer, "exchange-one");
        observations.accepted(100);
        observations.accepted(100);
        let first = receiver.try_recv().unwrap();
        let second = receiver.try_recv().unwrap();
        assert_eq!(first.event_ref, second.event_ref);
        assert_eq!(first.exchange_id, "exchange-one");
        assert_eq!(first.source, "payer_asserted");
        assert_eq!(first.role, Role::Payer);
        let json = serde_json::to_string(&first).unwrap();
        assert!(json.contains(r#""role":"payer""#));
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
        let (observations, mut receiver) = observing(Role::Payer, "exchange-two");
        let mut payment = Transaction {
            id: "wallet-private".into(),
            payment_hash: Some("public-hash".into()),
            inbound: false,
            amount_msat: 10,
            fee_msat: 1,
            status: mesh_llm_wallet::provider::PaymentStatus::Pending,
            claiming: false,
            status_msg: None,
            created_at_ms: 0,
            settled_at_ms: None,
        };
        observations.settled(0, &payment);
        payment.status = mesh_llm_wallet::provider::PaymentStatus::Failed;
        observations.settled(0, &payment);
        assert!(receiver.try_recv().is_err());
        payment.status = mesh_llm_wallet::provider::PaymentStatus::Succeeded;
        observations.settled(0, &payment);
        let event = receiver.try_recv().unwrap();
        assert_eq!(event.phase, "input_settlement_observed");
        assert_eq!(event.settlement, Some("terminal"));
        assert_eq!(event.exchange_id, "exchange-two");
        assert_eq!(event.payment_hash.as_deref(), Some("public-hash"));
    }

    #[test]
    fn provider_asserts_its_own_acceptance_and_delivered_watermark() {
        let (observations, mut receiver) = observing(Role::Provider, "provider-exchange");
        observations.accepted(100);
        observations.delivered(42);
        let accepted = receiver.try_recv().unwrap();
        assert_eq!(accepted.phase, "terms_accepted");
        assert_eq!(accepted.source, "provider_asserted");
        assert_eq!(accepted.role, Role::Provider);
        let delivered = receiver.try_recv().unwrap();
        assert_eq!(delivered.phase, "delivered");
        assert_eq!(delivered.tokens, Some(42));
        assert_eq!(delivered.amount_msat, 0);
        assert_eq!(delivered.settlement, None);
    }

    #[test]
    fn unobserved_exchange_emits_nothing() {
        let observations = Observations::default();
        observations.accepted(1);
        observations.delivered(1);
        observations.final_amount(1);
        assert!(observations.sender.is_none());
    }

    #[tokio::test]
    async fn provider_without_a_subscriber_never_starts_observing() {
        let observations = ProviderLifecycle::default().observe(&terms());
        assert!(observations.sender.is_none());
    }
}
