//! The host's `payment.lifecycle.v1` mesh channel: payer-side payment
//! lifecycle events the host broadcasts for a paid exchange, parsed and
//! checked here before the plugin seals one settlement record per event
//! (`CapsuleState::emit_settlement_record`).
//!
//! A settlement record is the payer node's own sealed observation of what its
//! host broadcast. The event's `source` says who asserted each value: the
//! payer (`payer_asserted`), the provider's invoice (`provider_asserted`), or
//! the payer's wallet (`wallet_reported`). A record is not a claim that money
//! moved beyond what a `wallet_reported` event says, and never a claim about
//! the provider's books. No records for an exchange means "no payment
//! lifecycle observed" (a free exchange, payments off, or a failure before
//! authorization), never "unpaid".
//!
//! Hosts since `role` and `tokens` were added to the event emit it on both
//! sides of a paid exchange, and this plugin seals both: each record names the
//! side that observed it (`observed_by`: `payer` or `provider`), so a
//! provider's records are never read as the payer's. An event with no `role`
//! comes from a host that emitted the payer side only, and is read as the
//! payer's. The provider side adds its own `terms_accepted`
//! (`provider_asserted`) and a `delivered` phase carrying the delivered-token
//! watermark (`tokens`).
//!
//! [`parse_and_check`] refuses an event whose own `event_ref` does not
//! recompute, and any phase/source/segment/hash combination the host's
//! emitter cannot produce. The combination rules mirror the emitter exactly
//! and are no stricter: in particular a settlement event may carry a `null`
//! `payment_hash`, because the emitter copies the wallet transaction's
//! optional hash as-is.

use crate::producer::capsule::{SettlementObservation, SETTLEMENT_CHANNEL};
use crate::producer::jcs;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The channel name this plugin declares and dispatches on.
pub const PAYMENT_LIFECYCLE_CHANNEL: &str = SETTLEMENT_CHANNEL;

/// Lifecycle phase, with the host's exact wire strings. An unknown phase
/// fails deserialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    TermsAccepted,
    InputInvoiceIssued,
    OutputInvoiceIssued,
    InputSettlementObserved,
    OutputSettlementObserved,
    FinalAccounted,
    /// Provider side: the delivered-token watermark written when serving
    /// closed.
    Delivered,
}

impl Phase {
    pub fn wire(self) -> &'static str {
        match self {
            Phase::TermsAccepted => "terms_accepted",
            Phase::InputInvoiceIssued => "input_invoice_issued",
            Phase::OutputInvoiceIssued => "output_invoice_issued",
            Phase::InputSettlementObserved => "input_settlement_observed",
            Phase::OutputSettlementObserved => "output_settlement_observed",
            Phase::FinalAccounted => "final_accounted",
            Phase::Delivered => "delivered",
        }
    }
}

/// Who asserted the event's values, with the host's exact wire strings. An
/// unknown source fails deserialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    PayerAsserted,
    ProviderAsserted,
    WalletReported,
}

impl Source {
    pub fn wire(self) -> &'static str {
        match self {
            Source::PayerAsserted => "payer_asserted",
            Source::ProviderAsserted => "provider_asserted",
            Source::WalletReported => "wallet_reported",
        }
    }
}

/// The only `settlement` value the host emits (for a wallet-reported event).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Settlement {
    Terminal,
}

impl Settlement {
    pub fn wire(self) -> &'static str {
        match self {
            Settlement::Terminal => "terminal",
        }
    }
}

/// Which side of a paid exchange emitted an event (`role`). Absent on hosts
/// that emitted the payer side only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Payer,
    Provider,
}

impl Role {
    pub fn wire(self) -> &'static str {
        match self {
            Role::Payer => "payer",
            Role::Provider => "provider",
        }
    }
}

/// One `payment.lifecycle.v1` event. Unknown extra fields are tolerated (a
/// newer host may add some); every listed field is required, including the
/// nullable ones -- `deserialize_with = "Option::deserialize"` turns off
/// serde's missing-`Option`-means-`None` default, so an absent key is an
/// error while an explicit `null` is `None`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct PaymentLifecycleEvent {
    pub exchange_id: String,
    pub event_ref: String,
    pub terms_digest: String,
    pub phase: Phase,
    pub source: Source,
    #[serde(deserialize_with = "Option::deserialize")]
    pub settlement: Option<Settlement>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub segment: Option<u32>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub payment_hash: Option<String>,
    pub amount_msat: u64,
    /// `payer` or `provider`; absent from a host that emitted the payer side
    /// only.
    #[serde(default)]
    pub role: Option<Role>,
    /// The provider's delivered-token watermark, on a `delivered` event;
    /// `null` or absent on every other.
    #[serde(default)]
    pub tokens: Option<u64>,
    /// A settlement as the provider's receiving wallet reported it: what it
    /// credited. Absent from hosts that do not pass the wallet's numbers.
    #[serde(default)]
    pub credited_msat: Option<u64>,
    /// A settlement's fee as the wallet reported it (payer: paid on top;
    /// provider: deducted on receipt). Absent from hosts that do not pass it.
    #[serde(default)]
    pub fee_msat: Option<u64>,
}

impl PaymentLifecycleEvent {
    /// Borrow this event as the observation `seal_settlement_record` copies
    /// verbatim.
    pub fn observation(&self) -> SettlementObservation<'_> {
        SettlementObservation {
            exchange_id: &self.exchange_id,
            event_ref: &self.event_ref,
            terms_digest: &self.terms_digest,
            phase: self.phase.wire(),
            source: self.source.wire(),
            settlement: self.settlement.map(Settlement::wire),
            segment: self.segment,
            payment_hash: self.payment_hash.as_deref(),
            amount_msat: self.amount_msat,
            observed_by: self.role.unwrap_or(Role::Payer).wire(),
            tokens: self.tokens,
            credited_msat: self.credited_msat,
            fee_msat: self.fee_msat,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SettlementEventError {
    #[error("not a JSON object: {0}")]
    NotJson(String),
    #[error("event shape: {0}")]
    Shape(#[from] serde_json::Error),
    #[error("empty exchange_id")]
    EmptyExchangeId,
    #[error("impossible event: {0}")]
    Impossible(&'static str),
    #[error("event_ref cannot be recomputed: {0}")]
    Digest(#[from] jcs::JcsError),
    #[error("event_ref {claimed} does not recompute (got {recomputed})")]
    EventRefMismatch { claimed: String, recomputed: String },
}

/// Recompute `event_ref` over the received object (extra fields included, as
/// the host digested them) with `event_ref` set to `""`.
fn check_event_ref(
    mut object: serde_json::Map<String, Value>,
    claimed: &str,
) -> Result<(), SettlementEventError> {
    object.insert("event_ref".into(), Value::String(String::new()));
    let recomputed = jcs::json_digest(&Value::Object(object))?;
    if recomputed != claimed {
        return Err(SettlementEventError::EventRefMismatch {
            claimed: claimed.to_string(),
            recomputed,
        });
    }
    Ok(())
}

/// Parse one channel body and check it before anything is sealed:
/// 1. JSON object, typed into [`PaymentLifecycleEvent`] (unknown phase/source
///    refused, missing field refused, extra fields tolerated);
/// 2. non-empty `exchange_id`;
/// 3. the phase/source/settlement/segment/payment_hash combination is one the
///    host's emitter produces (see [`check_combination`]);
/// 4. `event_ref` recomputes: lowercase-hex SHA-256 of the plugin's own JCS
///    over the received object (extra fields included, as the host digested
///    them) with `event_ref` set to `""`. A float anywhere, or an integer
///    above 2^53-1, makes the JCS refuse and the event is refused.
///
/// Duplicate JSON keys are not detected: `serde_json` keeps the last one.
pub fn parse_and_check(bytes: &[u8]) -> Result<PaymentLifecycleEvent, SettlementEventError> {
    let value: Value = serde_json::from_slice(bytes)?;
    let Value::Object(object) = value else {
        return Err(SettlementEventError::NotJson(
            "channel body is not a JSON object".into(),
        ));
    };
    let event = PaymentLifecycleEvent::deserialize(Value::Object(object.clone()))?;
    if event.exchange_id.is_empty() {
        return Err(SettlementEventError::EmptyExchangeId);
    }
    check_combination(&event)?;
    check_event_ref(object, &event.event_ref)?;
    Ok(event)
}

/// The combinations the host's emitter produces (`paid_events.rs`, and its
/// lifecycle observer on both sides), and no stricter:
/// - `settlement` is `"terminal"` exactly when `source` is `wallet_reported`;
/// - `terms_accepted`: asserted by the side that accepted them
///   (`payer_asserted` from the payer, `provider_asserted` from the provider),
///   no segment, no payment hash;
/// - `final_accounted`: payer side only, `payer_asserted`, no segment, no
///   payment hash;
/// - `delivered`: provider side only, `provider_asserted`, no segment, no
///   payment hash, and a `tokens` watermark;
/// - `*_invoice_issued`: `provider_asserted` with a segment and a payment hash,
///   on either side (the provider issues every invoice);
/// - `*_settlement_observed`: `wallet_reported` with a segment; the payment
///   hash may be `null`;
/// - `input_*` phases are segment 0, `output_*` phases a non-zero segment;
/// - `tokens` appears on `delivered` only.
///
/// An event with no `role` is the payer's.
fn check_combination(event: &PaymentLifecycleEvent) -> Result<(), SettlementEventError> {
    use Phase::*;
    let impossible = |why| Err(SettlementEventError::Impossible(why));
    let role = event.role.unwrap_or(Role::Payer);
    if (event.source == Source::WalletReported) != (event.settlement == Some(Settlement::Terminal))
    {
        return impossible("settlement is \"terminal\" exactly when source is wallet_reported");
    }
    let expected_source = match (event.phase, role) {
        (TermsAccepted, Role::Payer) | (FinalAccounted, Role::Payer) => Source::PayerAsserted,
        (TermsAccepted, Role::Provider) | (Delivered, Role::Provider) => Source::ProviderAsserted,
        (FinalAccounted, Role::Provider) => {
            return impossible("final_accounted is the payer side's");
        }
        (Delivered, Role::Payer) => return impossible("delivered is the provider side's"),
        (InputInvoiceIssued | OutputInvoiceIssued, _) => Source::ProviderAsserted,
        (InputSettlementObserved | OutputSettlementObserved, _) => Source::WalletReported,
    };
    if event.source != expected_source {
        return impossible("source does not match phase");
    }
    if (event.phase == Delivered) != event.tokens.is_some() {
        return impossible("tokens is carried by a delivered event, and only by one");
    }
    let settlement_phase = matches!(
        event.phase,
        InputSettlementObserved | OutputSettlementObserved
    );
    if !settlement_phase && (event.credited_msat.is_some() || event.fee_msat.is_some()) {
        return impossible("wallet amounts are carried by settlement events only");
    }
    if role == Role::Payer && event.credited_msat.is_some() {
        return impossible("credited_msat is the receiving (provider) wallet's");
    }
    match event.phase {
        TermsAccepted | FinalAccounted | Delivered => {
            if event.segment.is_some() || event.payment_hash.is_some() {
                return impossible(
                    "a terms, final or delivered event carries no segment or payment_hash",
                );
            }
        }
        InputInvoiceIssued
        | OutputInvoiceIssued
        | InputSettlementObserved
        | OutputSettlementObserved => {
            let Some(segment) = event.segment else {
                return impossible("invoice/settlement phase requires a segment");
            };
            let is_input = matches!(event.phase, InputInvoiceIssued | InputSettlementObserved);
            if is_input != (segment == 0) {
                return impossible("input phases are segment 0, output phases non-zero");
            }
            if matches!(event.phase, InputInvoiceIssued | OutputInvoiceIssued)
                && event.payment_hash.is_none()
            {
                return impossible("invoice phase requires a payment_hash");
            }
        }
    }
    Ok(())
}

/// Build a checked event body the way the host does: blank `event_ref`,
/// digest the JCS, fill it in. Shared by this module's tests and the
/// capsule_emit tests/fixture generator.
/// Whether a channel message is the host's own local broadcast, never a frame
/// relayed from a mesh peer.
///
/// The host's local broadcast (`PluginManager::broadcast_channel_message`,
/// `plugin/channel_broadcast.rs`) sends with an empty `source_peer_id` and
/// `target_peer_id` set to the receiving plugin's name. A frame from a peer is
/// delivered to local plugins only when its `target_peer_id` is empty or this
/// node's own peer id, a 64-hex endpoint id (`handle_plugin_channel_stream`,
/// `mesh/plugin_mesh.rs`), and its `source_peer_id` is whatever the sender
/// wrote. So a message with an empty source and a non-empty target that is not
/// a peer id can only be the local broadcast. Payment lifecycle events are
/// this node's own observations; anything else is refused before parsing.
pub(crate) fn is_local_host_broadcast(source_peer_id: &str, target_peer_id: &str) -> bool {
    let target_is_peer_id =
        target_peer_id.len() == 64 && target_peer_id.bytes().all(|b| b.is_ascii_hexdigit());
    source_peer_id.is_empty() && !target_peer_id.is_empty() && !target_is_peer_id
}

#[cfg(test)]
pub(crate) fn host_shaped_event(
    exchange_id: &str,
    phase: Phase,
    segment: Option<u32>,
    payment_hash: Option<&str>,
    amount_msat: u64,
) -> Value {
    let source = match phase {
        Phase::TermsAccepted | Phase::FinalAccounted => Source::PayerAsserted,
        Phase::InputInvoiceIssued | Phase::OutputInvoiceIssued => Source::ProviderAsserted,
        _ => Source::WalletReported,
    };
    let mut event = serde_json::json!({
        "exchange_id": exchange_id,
        "event_ref": "",
        "terms_digest": "7d".repeat(32),
        "phase": phase,
        "source": source,
        "settlement": (source == Source::WalletReported).then_some("terminal"),
        "segment": segment,
        "payment_hash": payment_hash,
        "amount_msat": amount_msat,
    });
    event["event_ref"] = Value::String(jcs::json_digest(&event).unwrap());
    event
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn only_the_hosts_local_broadcast_is_accepted() {
        // The host's local broadcast: empty source, target = plugin name.
        assert!(is_local_host_broadcast("", "capsules"));
        assert!(is_local_host_broadcast("", "capsules"));
        // A peer's frame delivered here: target empty or our own peer id.
        let our_peer_id = "ab".repeat(32);
        assert!(!is_local_host_broadcast("", ""));
        assert!(!is_local_host_broadcast("", &our_peer_id));
        assert!(!is_local_host_broadcast("", &our_peer_id.to_uppercase()));
        // A named sender is never the local broadcast.
        assert!(!is_local_host_broadcast(&"cd".repeat(32), "capsules"));
    }
    use serde_json::json;

    fn bytes(v: &Value) -> Vec<u8> {
        serde_json::to_vec(v).unwrap()
    }

    fn settled_input() -> Value {
        host_shaped_event(
            "ex-1",
            Phase::InputSettlementObserved,
            Some(0),
            Some(&"aa".repeat(32)),
            123457,
        )
    }

    /// Re-digest after a test mutates a field, so only the guard under test
    /// can refuse the body.
    fn redigest(mut v: Value) -> Value {
        v["event_ref"] = json!("");
        v["event_ref"] = Value::String(jcs::json_digest(&v).unwrap());
        v
    }

    #[test]
    fn channel_name_is_the_hosts() {
        assert_eq!(PAYMENT_LIFECYCLE_CHANNEL, "payment.lifecycle.v1");
    }

    /// Every phase, with the segment/hash the emitter gives it, passes. The
    /// upstream `Event` serializes every field with `null`s present, as
    /// `host_shaped_event` does; JCS sorts keys, so field order is irrelevant.
    #[test]
    fn every_phase_the_host_emits_is_accepted() {
        let hash = "aa".repeat(32);
        for (phase, segment, payment_hash) in [
            (Phase::TermsAccepted, None, None),
            (Phase::InputInvoiceIssued, Some(0), Some(hash.as_str())),
            (Phase::InputSettlementObserved, Some(0), Some(hash.as_str())),
            (Phase::OutputInvoiceIssued, Some(1), Some(hash.as_str())),
            (
                Phase::OutputSettlementObserved,
                Some(1),
                Some(hash.as_str()),
            ),
            (Phase::FinalAccounted, None, None),
        ] {
            let body = host_shaped_event("ex-1", phase, segment, payment_hash, 5);
            let event = parse_and_check(&bytes(&body)).unwrap_or_else(|e| panic!("{phase:?}: {e}"));
            assert_eq!(event.phase, phase);
        }
    }

    /// Pinned vector: the upstream event shape, serialized with its `null`s
    /// as the host's `Event` does, with the `event_ref` an independent
    /// computation gives (Python `hashlib.sha256` over `json.dumps(e,
    /// sort_keys=True, separators=(",", ":"))`, which equals JCS for this
    /// ASCII-only, integer-only body).
    #[test]
    fn event_ref_matches_an_independent_digest() {
        let body = br#"{"exchange_id":"exchange-one","event_ref":"f7c6847a6fbfcb6bc6ccc26bdb748b333e22946c70a4aa5684a60ba1c882b634","terms_digest":"7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d","phase":"terms_accepted","source":"payer_asserted","settlement":null,"segment":null,"payment_hash":null,"amount_msat":100}"#;
        let event = parse_and_check(body).expect("independently digested event checks");
        assert_eq!(event.amount_msat, 100);
    }

    #[test]
    fn amount_and_fields_parse_verbatim() {
        let event = parse_and_check(&bytes(&settled_input())).unwrap();
        assert_eq!(event.amount_msat, 123457);
        assert_eq!(
            event.payment_hash.as_deref(),
            Some("aa".repeat(32).as_str())
        );
        assert_eq!(event.segment, Some(0));
        assert_eq!(event.settlement, Some(Settlement::Terminal));
        let obs = event.observation();
        assert_eq!(obs.phase, "input_settlement_observed");
        assert_eq!(obs.source, "wallet_reported");
        assert_eq!(obs.settlement, Some("terminal"));
    }

    #[test]
    fn extra_unknown_field_is_tolerated() {
        let mut body = settled_input();
        body["newer_host_field"] = json!("x");
        let body = redigest(body);
        assert!(parse_and_check(&bytes(&body)).is_ok());
    }

    /// Follows upstream: the emitter copies the wallet transaction's optional
    /// payment hash, so a settlement event with a `null` hash is one it can
    /// produce and is accepted.
    #[test]
    fn settlement_observed_with_null_payment_hash_is_accepted() {
        let mut body = settled_input();
        body["payment_hash"] = Value::Null;
        let body = redigest(body);
        assert!(parse_and_check(&bytes(&body)).is_ok());
    }

    #[test]
    fn event_ref_mismatch_is_refused() {
        let mut body = settled_input();
        body["amount_msat"] = json!(999);
        assert!(matches!(
            parse_and_check(&bytes(&body)),
            Err(SettlementEventError::EventRefMismatch { .. })
        ));
    }

    #[test]
    fn unknown_phase_is_refused() {
        let mut body = settled_input();
        body["phase"] = json!("refund_issued");
        let body = redigest(body);
        assert!(matches!(
            parse_and_check(&bytes(&body)),
            Err(SettlementEventError::Shape(_))
        ));
    }

    #[test]
    fn unknown_source_is_refused() {
        let mut body = settled_input();
        body["source"] = json!("oracle_asserted");
        let body = redigest(body);
        assert!(matches!(
            parse_and_check(&bytes(&body)),
            Err(SettlementEventError::Shape(_))
        ));
    }

    #[test]
    fn missing_nullable_field_is_refused() {
        let mut body = settled_input();
        body.as_object_mut().unwrap().remove("payment_hash");
        let body = redigest(body);
        assert!(matches!(
            parse_and_check(&bytes(&body)),
            Err(SettlementEventError::Shape(_))
        ));
    }

    #[test]
    fn empty_exchange_id_is_refused() {
        let mut body = settled_input();
        body["exchange_id"] = json!("");
        let body = redigest(body);
        assert!(matches!(
            parse_and_check(&bytes(&body)),
            Err(SettlementEventError::EmptyExchangeId)
        ));
    }

    #[test]
    fn invoice_without_payment_hash_is_refused() {
        let body = host_shaped_event("ex-1", Phase::InputInvoiceIssued, Some(0), None, 5);
        assert!(matches!(
            parse_and_check(&bytes(&body)),
            Err(SettlementEventError::Impossible(_))
        ));
    }

    #[test]
    fn settlement_observed_without_segment_is_refused() {
        let hash = "aa".repeat(32);
        let body = host_shaped_event("ex-1", Phase::InputSettlementObserved, None, Some(&hash), 5);
        assert!(matches!(
            parse_and_check(&bytes(&body)),
            Err(SettlementEventError::Impossible(_))
        ));
    }

    #[test]
    fn output_phase_on_segment_zero_is_refused() {
        let hash = "bb".repeat(32);
        let body = host_shaped_event(
            "ex-1",
            Phase::OutputSettlementObserved,
            Some(0),
            Some(&hash),
            5,
        );
        assert!(matches!(
            parse_and_check(&bytes(&body)),
            Err(SettlementEventError::Impossible(_))
        ));
    }

    #[test]
    fn source_not_matching_phase_is_refused() {
        let mut body = host_shaped_event("ex-1", Phase::TermsAccepted, None, None, 5);
        body["source"] = json!("provider_asserted");
        let body = redigest(body);
        assert!(matches!(
            parse_and_check(&bytes(&body)),
            Err(SettlementEventError::Impossible(_))
        ));
    }

    #[test]
    fn wallet_reported_without_terminal_settlement_is_refused() {
        let mut body = settled_input();
        body["settlement"] = Value::Null;
        let body = redigest(body);
        assert!(matches!(
            parse_and_check(&bytes(&body)),
            Err(SettlementEventError::Impossible(_))
        ));
    }

    #[test]
    fn payer_phase_with_payment_hash_is_refused() {
        let hash = "aa".repeat(32);
        let body = host_shaped_event("ex-1", Phase::FinalAccounted, None, Some(&hash), 5);
        assert!(matches!(
            parse_and_check(&bytes(&body)),
            Err(SettlementEventError::Impossible(_))
        ));
    }

    /// Above 2^53-1 the host drops the event; if one arrives anyway the JCS
    /// recompute refuses it.
    #[test]
    fn unsafe_amount_is_refused() {
        let mut body = settled_input();
        body["amount_msat"] = json!(1u64 << 53);
        assert!(matches!(
            parse_and_check(&bytes(&body)),
            Err(SettlementEventError::Digest(_))
        ));
    }

    // ---- the event shape with `role` and `tokens` -------------------------
    // Hosts that emit both sides of a paid exchange add `role` and `tokens`,
    // and digest them into `event_ref` like every other member. The two bodies
    // below are in the host's member order, with `event_ref` computed by the
    // Agent Action Capsule reference implementation's JSON-DIGEST
    // (`agent_action_capsule.canonical.json_digest`), not by this crate.

    const MERGED_PAYER_TERMS: &[u8] = br#"{"exchange_id":"host-exchange-7","event_ref":"acbe0fedd8dab68c42610adecd964c31b780f85f752ceda75c168fb9740ac089","terms_digest":"7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d","role":"payer","phase":"terms_accepted","source":"payer_asserted","settlement":null,"segment":null,"payment_hash":null,"amount_msat":100,"tokens":null}"#;

    const MERGED_PROVIDER_DELIVERED: &[u8] = br#"{"exchange_id":"3f0c9a2e-6b1d-4c8e-9a57-1d2e3f4a5b6c","event_ref":"272c8c1799e8649427106f7b12a2c3ae571f8b4253017b32a20bb0621ddbd78b","terms_digest":"8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e","role":"provider","phase":"delivered","source":"provider_asserted","settlement":null,"segment":null,"payment_hash":null,"amount_msat":0,"tokens":42}"#;

    /// `event` with the members a both-sides host adds, re-digested.
    fn with_role(mut event: Value, role: &str, tokens: Option<u64>) -> Value {
        event["role"] = json!(role);
        event["tokens"] = json!(tokens);
        redigest(event)
    }

    #[test]
    fn a_payer_event_with_role_and_tokens_matches_an_independent_digest() {
        let event = parse_and_check(MERGED_PAYER_TERMS).unwrap();
        assert_eq!(event.role, Some(Role::Payer));
        assert_eq!(event.tokens, None);
        assert_eq!(event.phase, Phase::TermsAccepted);
        assert_eq!(event.exchange_id, "host-exchange-7");
    }

    #[test]
    fn every_payer_phase_is_accepted_with_role_and_tokens() {
        let hash = "aa".repeat(32);
        for (phase, segment, payment_hash) in [
            (Phase::TermsAccepted, None, None),
            (Phase::InputInvoiceIssued, Some(0), Some(hash.as_str())),
            (Phase::InputSettlementObserved, Some(0), Some(hash.as_str())),
            (Phase::OutputInvoiceIssued, Some(1), Some(hash.as_str())),
            (
                Phase::OutputSettlementObserved,
                Some(1),
                Some(hash.as_str()),
            ),
            (Phase::FinalAccounted, None, None),
        ] {
            let body = with_role(
                host_shaped_event("ex-9", phase, segment, payment_hash, 500),
                "payer",
                None,
            );
            let event = parse_and_check(&bytes(&body)).unwrap_or_else(|e| panic!("{phase:?}: {e}"));
            assert_eq!(event.role, Some(Role::Payer));
        }
    }

    #[test]
    fn an_event_without_role_is_still_read_as_the_payers() {
        let event = parse_and_check(&bytes(&settled_input())).unwrap();
        assert_eq!(event.role, None);
        assert_eq!(event.tokens, None);
    }

    /// A provider-side event of the merged shape, built the way the host's
    /// lifecycle observer emits it, and re-digested.
    fn provider(
        phase: &str,
        source: &str,
        segment: Option<u32>,
        hash: Option<&str>,
        tokens: Option<u64>,
    ) -> Value {
        redigest(json!({
            "exchange_id": "3f0c9a2e-6b1d-4c8e-9a57-1d2e3f4a5b6c",
            "event_ref": "",
            "terms_digest": "8e".repeat(32),
            "role": "provider",
            "phase": phase,
            "source": source,
            "settlement": (source == "wallet_reported").then_some("terminal"),
            "segment": segment,
            "payment_hash": hash,
            "amount_msat": 0,
            "tokens": tokens,
        }))
    }

    #[test]
    fn a_provider_delivered_event_matches_an_independent_digest() {
        let event = parse_and_check(MERGED_PROVIDER_DELIVERED).unwrap();
        assert_eq!(event.role, Some(Role::Provider));
        assert_eq!(event.phase, Phase::Delivered);
        assert_eq!(event.tokens, Some(42));
        let observation = event.observation();
        assert_eq!(observation.observed_by, "provider");
        assert_eq!(observation.phase, "delivered");
        assert_eq!(observation.tokens, Some(42));
    }

    #[test]
    fn every_phase_the_provider_side_emits_is_accepted() {
        let hash = "bb".repeat(32);
        for body in [
            provider("terms_accepted", "provider_asserted", None, None, None),
            provider(
                "input_invoice_issued",
                "provider_asserted",
                Some(0),
                Some(&hash),
                None,
            ),
            provider(
                "input_settlement_observed",
                "wallet_reported",
                Some(0),
                Some(&hash),
                None,
            ),
            provider(
                "output_invoice_issued",
                "provider_asserted",
                Some(1),
                Some(&hash),
                None,
            ),
            provider(
                "output_settlement_observed",
                "wallet_reported",
                Some(1),
                Some(&hash),
                None,
            ),
            provider("delivered", "provider_asserted", None, None, Some(7)),
        ] {
            let event =
                parse_and_check(&bytes(&body)).unwrap_or_else(|e| panic!("{}: {e}", body["phase"]));
            assert_eq!(event.role, Some(Role::Provider));
            assert_eq!(event.observation().observed_by, "provider");
        }
    }

    #[test]
    fn the_role_rules_refuse_what_neither_side_emits() {
        let refused = |body: Value, why: &str| {
            assert!(
                matches!(
                    parse_and_check(&bytes(&body)),
                    Err(SettlementEventError::Impossible(_))
                ),
                "{why}"
            );
        };
        refused(
            provider("terms_accepted", "payer_asserted", None, None, None),
            "the provider asserts its own acceptance",
        );
        refused(
            provider("final_accounted", "payer_asserted", None, None, None),
            "final_accounted is the payer side's",
        );
        refused(
            provider("delivered", "provider_asserted", None, None, None),
            "delivered carries a tokens watermark",
        );
        refused(
            provider("delivered", "provider_asserted", Some(1), None, Some(7)),
            "delivered carries no segment",
        );
        refused(
            with_role(
                host_shaped_event("ex-9", Phase::TermsAccepted, None, None, 500),
                "payer",
                Some(3),
            ),
            "tokens only on delivered",
        );
        let mut payer_delivered = with_role(
            host_shaped_event("ex-9", Phase::TermsAccepted, None, None, 0),
            "payer",
            Some(3),
        );
        payer_delivered["phase"] = json!("delivered");
        payer_delivered["source"] = json!("provider_asserted");
        refused(
            redigest(payer_delivered),
            "delivered is the provider side's",
        );
    }

    #[test]
    fn a_provider_event_whose_event_ref_does_not_recompute_is_refused() {
        let mut body: Value = serde_json::from_slice(MERGED_PROVIDER_DELIVERED).unwrap();
        body["tokens"] = json!(43);
        assert!(matches!(
            parse_and_check(&bytes(&body)),
            Err(SettlementEventError::EventRefMismatch { .. })
        ));
    }

    #[test]
    fn an_unknown_role_is_refused() {
        let body = with_role(settled_input(), "broker", None);
        assert!(matches!(
            parse_and_check(&bytes(&body)),
            Err(SettlementEventError::Shape(_))
        ));
    }

    // ---- one subscriber to both channels ----------------------------------
    // How a lifecycle event lines up with the `openai.exchange.v1` events for
    // the same paid exchange, with both parsed by this plugin's own parsers.

    fn exchange_envelope(exchange_id: &str, dispatch_path: &str, request_digest: &str) -> Value {
        json!({
            "exchange_id": exchange_id,
            "dispatch_path": dispatch_path,
            "phase": "terminal",
            "model": "model-x",
            "status": 200,
            "request_digest": request_digest,
        })
    }

    fn parse_envelope(v: &Value) -> crate::lifecycle_channel::OpenAiExchangeEnvelope {
        serde_json::from_slice(&bytes(v)).unwrap()
    }

    /// The payer's host names its lifecycle events with its own OpenAI
    /// exchange id, so they join the payer's exchange events exactly: the
    /// settlement book keys on the same `exchange_id` the exchange rows carry.
    #[test]
    fn a_subscriber_joins_the_payers_two_channels_on_exchange_id() {
        let exchange_id = "payer-exchange-1";
        let envelope = parse_envelope(&exchange_envelope(
            exchange_id,
            "remote_mesh",
            &"1a".repeat(32),
        ));
        let hash = "cc".repeat(32);
        for body in [
            with_role(
                host_shaped_event(exchange_id, Phase::TermsAccepted, None, None, 500),
                "payer",
                None,
            ),
            with_role(
                host_shaped_event(
                    exchange_id,
                    Phase::InputInvoiceIssued,
                    Some(0),
                    Some(&hash),
                    200,
                ),
                "payer",
                None,
            ),
        ] {
            let event = parse_and_check(&bytes(&body)).unwrap();
            assert_eq!(
                Some(event.exchange_id.as_str()),
                envelope.exchange_id.as_deref()
            );
        }
    }

    /// On the provider, the lifecycle events and the paid path's exchange
    /// events each carry an id their host minted separately, and no other
    /// member in common: the lifecycle events carry no request digest or
    /// model, the exchange events no payment hash or terms digest. Nothing a
    /// subscriber receives joins them; only timing and token counts come
    /// close, and those are ambiguous when the same model serves two paid
    /// requests at once. This is the host's current state, not a goal: once
    /// the host mints one id per paid serving request and hands it to both
    /// observers (proposed upstream), this test is replaced by one that joins
    /// them, as the payer's already do.
    #[test]
    fn a_subscriber_cannot_join_the_providers_two_channels() {
        let lifecycle: Value = serde_json::from_slice(MERGED_PROVIDER_DELIVERED).unwrap();
        let exchange = exchange_envelope(
            "5a1f2e3d-0000-4000-8000-00000000abcd",
            "raw_proxy",
            &"1a".repeat(32),
        );
        let envelope = parse_envelope(&exchange);
        assert_ne!(
            lifecycle["exchange_id"].as_str(),
            envelope.exchange_id.as_deref()
        );
        let lifecycle_keys: BTreeSet<&str> = lifecycle
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        let exchange_keys: BTreeSet<&str> = exchange
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        let join_keys = ["request_digest", "model", "payment_hash", "terms_digest"];
        for key in join_keys {
            assert!(
                !(lifecycle_keys.contains(key) && exchange_keys.contains(key)),
                "{key} would be a join key"
            );
        }
    }

    /// Across the two nodes, the invoice is the one thing both sides see: the
    /// provider issues it and the payer receives it, so an invoice event
    /// carries the same `payment_hash` on each side, segment by segment.
    #[test]
    fn the_payer_and_provider_share_each_invoices_payment_hash() {
        let hash = "dd".repeat(32);
        let payer = with_role(
            host_shaped_event(
                "payer-exchange-1",
                Phase::OutputInvoiceIssued,
                Some(1),
                Some(&hash),
                300,
            ),
            "payer",
            None,
        );
        let provider = with_role(
            host_shaped_event(
                "3f0c9a2e-6b1d-4c8e-9a57-1d2e3f4a5b6c",
                Phase::OutputInvoiceIssued,
                Some(1),
                Some(&hash),
                300,
            ),
            "provider",
            None,
        );
        let payer_event = parse_and_check(&bytes(&payer)).unwrap();
        let provider_event = parse_and_check(&bytes(&provider)).unwrap();
        assert_eq!(provider_event.observation().observed_by, "provider");
        assert_eq!(payer_event.observation().observed_by, "payer");
        assert_eq!(
            payer_event.payment_hash.as_deref(),
            provider["payment_hash"].as_str()
        );
        assert_eq!(
            payer_event.segment,
            provider["segment"].as_u64().map(|s| s as u32)
        );
        assert_ne!(
            payer_event.exchange_id,
            provider["exchange_id"].as_str().unwrap()
        );
    }
}
