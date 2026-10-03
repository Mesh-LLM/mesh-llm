//! Best-effort payment lifecycle observations, payer and provider side; never
//! a ledger, replay service or payment authority.
use mesh_llm_payments_types::RequestTerms;
use mesh_llm_payments_types::contract::SettledReceived;
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
    /// A settlement as the wallet recorded it: what it credited (provider side)
    /// and the fee it charged (payer: paid on top; provider: deducted). Absent
    /// when the wallet did not say, so such events are unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    credited_msat: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fee_msat: Option<u64>,
}

/// An event's optional members: the delivered watermark, and a settlement's
/// amounts as the wallet reported them. All absent unless the phase has them.
#[derive(Clone, Copy, Default)]
struct Optional {
    tokens: Option<u64>,
    credited_msat: Option<u64>,
    fee_msat: Option<u64>,
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
            None,
            Some(transaction.fee_msat),
        );
    }

    /// Provider side: the receiving wallet reported `invoice` settled and the
    /// provider recorded it received. `amount_msat` stays the invoice amount;
    /// what the wallet credited and deducted ride alongside when it said.
    pub(crate) fn received(&self, segment: u32, invoice: &Invoice, settled: &SettledReceived) {
        self.settlement(
            segment,
            Some(invoice.payment_hash.clone()),
            invoice.amount_msat.unwrap_or(0),
            settled.credited_msat,
            settled.fee_msat,
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

    fn settlement(
        &self,
        segment: u32,
        payment_hash: Option<String>,
        amount_msat: u64,
        credited_msat: Option<u64>,
        fee_msat: Option<u64>,
    ) {
        self.emit_with(
            if segment == 0 {
                "input_settlement_observed"
            } else {
                "output_settlement_observed"
            },
            "wallet_reported",
            Some(segment),
            payment_hash,
            amount_msat,
            Optional {
                tokens: None,
                credited_msat,
                fee_msat,
            },
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
        self.emit_with(
            phase,
            source,
            segment,
            payment_hash,
            amount_msat,
            Optional {
                tokens,
                ..Optional::default()
            },
        );
    }

    fn emit_with(
        &self,
        phase: &'static str,
        source: &'static str,
        segment: Option<u32>,
        payment_hash: Option<String>,
        amount_msat: u64,
        optional: Optional,
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
            tokens: optional.tokens,
            credited_msat: optional.credited_msat,
            fee_msat: optional.fee_msat,
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
/// channel, so a node with no subscriber never hashes.
#[derive(Clone, Default)]
pub(crate) struct ProviderLifecycle {
    manager: Option<PluginManager>,
    exchange_id: Option<String>,
}

impl ProviderLifecycle {
    pub(crate) async fn for_node(node: &Node) -> Self {
        let Some(manager) = node.plugin_manager().await else {
            return Self::default();
        };
        if !manager.any_plugin_declares_mesh_channel(CHANNEL).await {
            return Self::default();
        }
        Self {
            manager: Some(manager),
            exchange_id: None,
        }
    }

    /// Whether a loaded plugin subscribes to this channel.
    pub(crate) fn is_subscribed(&self) -> bool {
        self.manager.is_some()
    }

    /// Name the serving request's exchange with `exchange_id`, the id its
    /// `openai.exchange.v1` events carry, so the provider's two channels join.
    pub(crate) fn named(self, exchange_id: Option<String>) -> Self {
        Self {
            exchange_id,
            ..self
        }
    }

    /// Start observing a serving request once its input invoice exists. The
    /// provider has no payer-side exchange to join: the serving path names
    /// the exchange with its own fresh ID ([`Self::named`]), never the private
    /// request (recovery) ID. Nothing is observed for an unnamed request.
    pub(crate) fn observe(&self, terms: &RequestTerms) -> Observations {
        let (Some(manager), Some(exchange_id)) = (self.manager.clone(), self.exchange_id.clone())
        else {
            return Observations::default();
        };
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

    fn invoice(amount_msat: u64) -> Invoice {
        Invoice {
            bolt11: "lnbc-private".into(),
            payment_hash: "public-hash".into(),
            payee: "payee".into(),
            amount_msat: Some(amount_msat),
            expires_at_ms: 1000,
        }
    }

    #[test]
    fn payer_settlement_carries_the_wallets_fee() {
        let (observations, mut receiver) = observing(Role::Payer, "exchange-fee");
        let payment = Transaction {
            id: "wallet-private".into(),
            payment_hash: Some("public-hash".into()),
            inbound: false,
            amount_msat: 1000,
            fee_msat: 3,
            status: mesh_llm_wallet::provider::PaymentStatus::Succeeded,
            claiming: false,
            status_msg: None,
            created_at_ms: 0,
            settled_at_ms: Some(1),
        };
        observations.settled(1, &payment);
        let event = receiver.try_recv().unwrap();
        assert_eq!(
            (event.amount_msat, event.fee_msat, event.credited_msat),
            (1000, Some(3), None)
        );
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""fee_msat":3"#) && !json.contains("credited_msat"));
    }

    #[test]
    fn provider_settlement_carries_what_the_wallet_credited_and_deducted() {
        let (observations, mut receiver) = observing(Role::Provider, "provider-fee");
        let settled = SettledReceived {
            credited_msat: Some(995),
            fee_msat: Some(5),
        };
        observations.received(0, &invoice(1000), &settled);
        let event = receiver.try_recv().unwrap();
        assert_eq!(event.phase, "input_settlement_observed");
        assert_eq!(
            event.amount_msat, 1000,
            "amount_msat stays the invoice amount"
        );
        assert_eq!((event.credited_msat, event.fee_msat), (Some(995), Some(5)));
    }

    #[test]
    fn a_settlement_the_wallet_did_not_describe_is_unchanged_on_the_wire() {
        let (observations, mut receiver) = observing(Role::Provider, "provider-plain");
        observations.received(1, &invoice(1000), &SettledReceived::default());
        let event = receiver.try_recv().unwrap();
        let json = serde_json::to_value(&event).unwrap();
        let mut keys: Vec<&str> = json
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        let mut before = vec![
            "exchange_id",
            "event_ref",
            "terms_digest",
            "role",
            "phase",
            "source",
            "settlement",
            "segment",
            "payment_hash",
            "amount_msat",
            "tokens",
        ];
        before.sort_unstable();
        assert_eq!(
            keys, before,
            "exactly the members events had before wallet amounts"
        );
        let mut blank = json.clone();
        blank["event_ref"] = serde_json::Value::String(String::new());
        assert_eq!(request_body_digest(&blank, None).unwrap(), event.event_ref);
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
