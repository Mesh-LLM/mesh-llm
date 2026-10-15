//! Settlement-records legs (draft-mih-agent-settlement-records-00) for paid
//! exchanges, beside the `payment.lifecycle.v1` trail
//! ([`crate::settlement_channel`]), which seals what it did before.
//!
//! One terms leg per invoice. Mesh pays per invoice (an input and an output
//! invoice per exchange, each with its own payment hash), and -00 joins one
//! payer-observed and one payee-observed leg per terms leg.
//!
//! **Who seals what.**
//! - The payer seals the terms leg when it sees an invoice issued (the
//!   invoice amount, `payment_ref` = `ln.payment_hash`) and pushes it to the
//!   provider over `record-push/1`. -00 lets either party seal the terms leg
//!   (§3 item 4); the payer does here because only it can address the other
//!   side: its lifecycle events carry the host exchange id, which joins to the
//!   exchange event naming the serving peer, while a provider's lifecycle
//!   events name no requester.
//! - The payer seals `payer_observed` when its wallet reports the payment
//!   settled: `amount` is the wallet transaction's own amount (the host's
//!   payer settlement event carries it), and `routing_fee` is the wallet's fee
//!   when the host passes it.
//! - The provider seals `payee_observed` citing the payer's terms leg, when
//!   its wallet reported what it credited and deducted (`received`,
//!   `receive_fee`). A host that does not pass those numbers gets no payee
//!   leg: the invoice amount is not what the wallet credited, and a missing
//!   fee never reconciles. The provider cites only the first pushed terms leg
//!   for one of its own invoices whose amount is that invoice's amount; any
//!   other pushed terms leg is ignored.
//! - The payer seals `delivered` (`received`) with the exchange's response
//!   digest, citing the output invoice's terms leg. The provider's own
//!   delivered leg needs its exchange event joined to its lifecycle events,
//!   which the host does not do yet, so it is not sealed.
//!
//! `observed_at` is when this node received the host's event, coarsened to
//! the minute like every committed time here; the host's events carry no
//! wallet time.
//!
//! **Bounded and best-effort.** What waits (a terms leg not yet pushed, a
//! payee observation waiting for its terms leg, this node's invoices) is held
//! in memory, bounded by count (at most [`MAX_HELD`] entries per kind, oldest
//! dropped first) and by time (an entry older than [`MAX_AGE`] is dropped,
//! with a log line, checked on each event), and lost on restart. A push that
//! fails is queued again and retried with the next settlement event while it
//! is held. A leg that cannot be sealed is missing evidence, never a failed
//! exchange.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use capsule_emit_lib::settlement::{amount, build_leg};
use serde_json::{json, Map, Value};

use crate::capsule_emit::{CapsuleState, EmittedCapsule};
use crate::settlement_channel::{PaymentLifecycleEvent, Phase, Role};

/// How many entries each waiting kind holds.
pub const MAX_HELD: usize = 512;

/// How long an entry is held. A paid exchange's events arrive within
/// seconds to minutes of each other; anything older is not coming.
pub const MAX_AGE: Duration = Duration::from_secs(15 * 60);

/// A sealed terms leg to push to `peer`.
#[derive(Debug, Clone)]
pub struct Push {
    pub peer: String,
    pub capsule: Value,
    exchange_id: String,
    hash: String,
}

/// A map that keeps at most [`MAX_HELD`] entries, dropping the oldest, each
/// stamped with when it was inserted (for [`MAX_AGE`]).
struct Bounded<V> {
    map: HashMap<String, (Instant, V)>,
    order: VecDeque<String>,
}

impl<V> Default for Bounded<V> {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }
}

impl<V> Bounded<V> {
    fn insert(&mut self, key: String, value: V) {
        if self
            .map
            .insert(key.clone(), (Instant::now(), value))
            .is_none()
        {
            self.order.push_back(key);
            while self.order.len() > MAX_HELD {
                if let Some(old) = self.order.pop_front() {
                    self.map.remove(&old);
                }
            }
        }
    }
    fn get(&self, key: &str) -> Option<&V> {
        self.map.get(key).map(|(_, value)| value)
    }
    fn contains(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }
    fn remove(&mut self, key: &str) -> Option<V> {
        let (_, value) = self.map.remove(key)?;
        self.order.retain(|k| k != key);
        Some(value)
    }
    /// Drop the entries inserted before `cutoff`; returns how many.
    fn expire(&mut self, cutoff: Instant) -> usize {
        let before = self.map.len();
        self.map.retain(|_, (at, _)| *at >= cutoff);
        let map = &self.map;
        self.order.retain(|k| map.contains_key(k));
        before - self.map.len()
    }
    #[cfg(test)]
    fn len(&self) -> usize {
        self.map.len()
    }
}

/// A terms leg this node (as payer) sealed.
#[derive(Clone)]
struct OwnTerms {
    capsule: Value,
    capsule_id: String,
}

#[derive(Default)]
struct Inner {
    /// Payer: its terms legs, by payment hash.
    own_terms: Bounded<OwnTerms>,
    /// Payer: the output invoice's payment hash, by exchange id.
    output_hash: Bounded<String>,
    /// Payer: the serving peer, by exchange id.
    peer: Bounded<String>,
    /// Payer: payment hashes whose terms leg is not pushed yet, by exchange id.
    unpushed: Bounded<Vec<String>>,
    /// Payer: the response digest, by exchange id.
    response: Bounded<String>,
    /// Provider: its own invoices' amounts (msat), by payment hash.
    invoices: Bounded<u64>,
    /// Provider: the terms leg it cites, by payment hash (first valid wins).
    received_terms: Bounded<String>,
    /// Provider: payee observations waiting for their terms leg.
    waiting_payee: Bounded<PaymentLifecycleEvent>,
}

/// This node's settlement-records state (held by [`CapsuleState`]).
#[derive(Default)]
pub struct Legs {
    inner: Mutex<Inner>,
}

fn msat(value: u64) -> Value {
    amount(u128::from(value), "BTC", 11)
}

fn ln(hash: &str) -> Value {
    json!({"type": "ln.payment_hash", "value": hash})
}

fn now() -> String {
    capsule_emit_lib::timestamp::utc_now_minute()
}

fn built(leg: &str, role: &str, pairs: Vec<(&str, Value)>) -> anyhow::Result<Value> {
    let members: Map<String, Value> = pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    build_leg(leg, role, members).map_err(|e| anyhow::anyhow!("{e}"))
}

fn seal(
    capsules: &CapsuleState,
    action: &str,
    member: anyhow::Result<Value>,
) -> Option<EmittedCapsule> {
    let member = match member {
        Ok(member) => member,
        Err(error) => {
            tracing::warn!(%error, %action, "settlement leg not built");
            return None;
        }
    };
    match capsules.emit_settlement_leg(format!("mesh/settlement/{action}"), member, None) {
        Ok(Some(sealed)) => {
            tracing::debug!(capsule_id = %sealed.capsule_id, %action, "sealed settlement leg");
            Some(sealed)
        }
        Ok(None) => None,
        Err(error) => {
            tracing::warn!(%error, %action, "settlement leg not sealed");
            None
        }
    }
}

impl Legs {
    fn inner(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Drop every entry held longer than [`MAX_AGE`] as of `now`, with one log
    /// line per kind that lost entries. Run on each event.
    fn expire(&self, now: Instant) {
        let Some(cutoff) = now.checked_sub(MAX_AGE) else {
            return;
        };
        let mut inner = self.inner();
        let dropped = [
            ("terms legs", inner.own_terms.expire(cutoff)),
            ("output invoices", inner.output_hash.expire(cutoff)),
            ("serving peers", inner.peer.expire(cutoff)),
            ("unpushed terms legs", inner.unpushed.expire(cutoff)),
            ("response digests", inner.response.expire(cutoff)),
            ("own invoices", inner.invoices.expire(cutoff)),
            ("received terms legs", inner.received_terms.expire(cutoff)),
            (
                "waiting payee observations",
                inner.waiting_payee.expire(cutoff),
            ),
        ];
        for (kind, count) in dropped {
            if count > 0 {
                tracing::info!(
                    count,
                    kind,
                    "settlement legs: dropped entries held longer than the time bound"
                );
            }
        }
    }

    /// A checked lifecycle event. Returns the terms legs to push now.
    pub fn on_lifecycle(
        &self,
        capsules: &CapsuleState,
        event: &PaymentLifecycleEvent,
    ) -> Vec<Push> {
        self.expire(Instant::now());
        let role = event.role.unwrap_or(Role::Payer);
        let Some(hash) = event.payment_hash.as_deref() else {
            return Vec::new();
        };
        let invoice = matches!(
            event.phase,
            Phase::InputInvoiceIssued | Phase::OutputInvoiceIssued
        );
        let settled = matches!(
            event.phase,
            Phase::InputSettlementObserved | Phase::OutputSettlementObserved
        );
        match role {
            Role::Payer if invoice => return self.payer_terms(capsules, event, hash),
            Role::Payer if settled => self.payer_observed(capsules, event, hash),
            Role::Provider if invoice => self
                .inner()
                .invoices
                .insert(hash.to_string(), event.amount_msat),
            Role::Provider if settled => {
                if event.credited_msat.is_none() || event.fee_msat.is_none() {
                    tracing::debug!("host passes no wallet amounts; no payee leg");
                    return Vec::new();
                }
                // One lock: a terms leg arriving between the check and the
                // insert would otherwise leave this observation waiting.
                let terms = {
                    let mut inner = self.inner();
                    let terms = inner.received_terms.get(hash).cloned();
                    if terms.is_none() {
                        inner.waiting_payee.insert(hash.to_string(), event.clone());
                    }
                    terms
                };
                if let Some(terms_ref) = terms {
                    self.payee_observed(capsules, event, hash, &terms_ref);
                }
            }
            _ => {}
        }
        Vec::new()
    }

    fn payer_terms(
        &self,
        capsules: &CapsuleState,
        event: &PaymentLifecycleEvent,
        hash: &str,
    ) -> Vec<Push> {
        let leg = built(
            "terms",
            "payer",
            vec![
                ("amount", msat(event.amount_msat)),
                ("payment_ref", ln(hash)),
            ],
        );
        let Some(sealed) = seal(capsules, "terms", leg) else {
            return Vec::new();
        };
        let exchange_id = event.exchange_id.clone();
        let (push, delivered) = {
            let mut inner = self.inner();
            inner.own_terms.insert(
                hash.to_string(),
                OwnTerms {
                    capsule: sealed.capsule.clone(),
                    capsule_id: sealed.capsule_id.clone(),
                },
            );
            if event.phase == Phase::OutputInvoiceIssued {
                inner
                    .output_hash
                    .insert(exchange_id.clone(), hash.to_string());
            }
            let push = match inner.peer.get(&exchange_id).cloned() {
                Some(peer) => Some(Push {
                    peer,
                    capsule: sealed.capsule,
                    exchange_id: exchange_id.clone(),
                    hash: hash.to_string(),
                }),
                None => {
                    let mut waiting = inner.unpushed.remove(&exchange_id).unwrap_or_default();
                    waiting.push(hash.to_string());
                    inner.unpushed.insert(exchange_id.clone(), waiting);
                    None
                }
            };
            (push, Self::delivered_due(&mut inner, &exchange_id))
        };
        if let Some((terms_ref, digest)) = delivered {
            self.seal_payer_delivered(capsules, &terms_ref, &digest);
        }
        push.into_iter().collect()
    }

    fn payer_observed(&self, capsules: &CapsuleState, event: &PaymentLifecycleEvent, hash: &str) {
        let Some(terms) = self.inner().own_terms.get(hash).cloned() else {
            tracing::debug!(%hash, "payer settlement before its terms leg; no payer leg");
            return;
        };
        let mut pairs = vec![
            ("terms_ref", json!(terms.capsule_id)),
            ("amount", msat(event.amount_msat)),
            ("payment_ref", ln(hash)),
            ("status", json!("settled")),
            ("observed_at", json!(now())),
        ];
        if let Some(fee) = event.fee_msat {
            pairs.push(("routing_fee", msat(fee)));
        }
        seal(
            capsules,
            "payer_observed",
            built("payer_observed", "payer", pairs),
        );
    }

    fn payee_observed(
        &self,
        capsules: &CapsuleState,
        event: &PaymentLifecycleEvent,
        hash: &str,
        terms_ref: &str,
    ) {
        let (Some(credited), Some(fee)) = (event.credited_msat, event.fee_msat) else {
            return;
        };
        let leg = built(
            "payee_observed",
            "payee",
            vec![
                ("terms_ref", json!(terms_ref)),
                ("received", msat(credited)),
                ("receive_fee", msat(fee)),
                ("payment_ref", ln(hash)),
                ("status", json!("settled")),
                ("observed_at", json!(now())),
            ],
        );
        seal(capsules, "payee_observed", leg);
    }

    /// Payer: the exchange event for `exchange_id` named its serving `peer`
    /// and, on a terminal event, the `response_digest`. Returns the terms
    /// legs that were waiting for the peer.
    pub fn on_exchange(
        &self,
        capsules: &CapsuleState,
        exchange_id: &str,
        peer: Option<&str>,
        response_digest: Option<&str>,
    ) -> Vec<Push> {
        self.expire(Instant::now());
        let (pushes, delivered) = {
            let mut inner = self.inner();
            if let Some(peer) = peer {
                inner.peer.insert(exchange_id.to_string(), peer.to_string());
            }
            let pushes = Self::take_ready(&mut inner, exchange_id);
            if let Some(digest) = response_digest {
                inner
                    .response
                    .insert(exchange_id.to_string(), digest.to_string());
            }
            (pushes, Self::delivered_due(&mut inner, exchange_id))
        };
        if let Some((terms_ref, digest)) = delivered {
            self.seal_payer_delivered(capsules, &terms_ref, &digest);
        }
        pushes
    }

    /// The terms legs not yet pushed whose serving peer is known: retried
    /// after a push failed or was skipped.
    pub fn retry(&self) -> Vec<Push> {
        self.expire(Instant::now());
        let mut inner = self.inner();
        let ready: Vec<String> = inner
            .unpushed
            .order
            .iter()
            .filter(|ex| inner.peer.contains(ex))
            .cloned()
            .collect();
        ready
            .iter()
            .flat_map(|ex| Self::take_ready(&mut inner, ex))
            .collect()
    }

    fn take_ready(inner: &mut Inner, exchange_id: &str) -> Vec<Push> {
        let Some(peer) = inner.peer.get(exchange_id).cloned() else {
            return Vec::new();
        };
        let hashes = inner.unpushed.remove(exchange_id).unwrap_or_default();
        hashes
            .into_iter()
            .filter_map(|hash| {
                let terms = inner.own_terms.get(&hash)?;
                Some(Push {
                    peer: peer.clone(),
                    capsule: terms.capsule.clone(),
                    exchange_id: exchange_id.to_string(),
                    hash,
                })
            })
            .collect()
    }

    /// A push that failed or was skipped: queue its terms leg again.
    pub fn requeue(&self, push: &Push) {
        let mut inner = self.inner();
        let mut waiting = inner.unpushed.remove(&push.exchange_id).unwrap_or_default();
        if !waiting.contains(&push.hash) {
            waiting.push(push.hash.clone());
        }
        inner.unpushed.insert(push.exchange_id.clone(), waiting);
        inner
            .peer
            .insert(push.exchange_id.clone(), push.peer.clone());
    }

    /// The payer's delivered leg is due once both the output invoice's terms
    /// leg and the response digest are known; the exchange's entries are then
    /// cleared, so a later event for the same id starts afresh.
    fn delivered_due(inner: &mut Inner, exchange_id: &str) -> Option<(String, String)> {
        let hash = inner.output_hash.get(exchange_id)?.clone();
        let digest = inner.response.get(exchange_id)?.clone();
        let terms = inner.own_terms.get(&hash)?.capsule_id.clone();
        inner.output_hash.remove(exchange_id);
        inner.response.remove(exchange_id);
        if !inner.unpushed.contains(exchange_id) {
            inner.peer.remove(exchange_id);
        }
        Some((terms, digest))
    }

    fn seal_payer_delivered(&self, capsules: &CapsuleState, terms_ref: &str, digest: &str) {
        let leg = built(
            "delivered",
            "payer",
            vec![
                ("terms_ref", json!(terms_ref)),
                ("observed_at", json!(now())),
                (
                    "delivery",
                    json!({"direction": "received", "content_digest": digest}),
                ),
            ],
        );
        seal(capsules, "delivered", leg);
    }

    /// Provider: a peer's pushed record was received and verified. It is held
    /// as the terms this node cites only when it is a payer's terms leg for
    /// one of this node's own invoices, at that invoice's amount, and no
    /// terms leg is held for that invoice yet. A waiting payee leg is sealed.
    pub fn on_received(&self, capsules: &CapsuleState, capsule: &Value) {
        let Some(member) = capsule.get("settlement") else {
            return;
        };
        let text = |pointer: &str| member.pointer(pointer).and_then(Value::as_str);
        if text("/leg") != Some("terms")
            || text("/sealer_role") != Some("payer")
            || text("/payment_ref/type") != Some("ln.payment_hash")
        {
            return;
        }
        let (Some(hash), Some(capsule_id)) = (
            text("/payment_ref/value"),
            capsule.get("capsule_id").and_then(Value::as_str),
        ) else {
            return;
        };
        let waiting = {
            let mut inner = self.inner();
            let Some(invoiced) = inner.invoices.get(hash).copied() else {
                tracing::debug!(%hash, "pushed terms leg for an invoice this node did not issue; ignored");
                return;
            };
            if member.get("amount") != Some(&msat(invoiced)) {
                tracing::warn!(%hash, "pushed terms leg does not state this invoice's amount; ignored");
                return;
            }
            if inner.received_terms.contains(hash) {
                return;
            }
            inner
                .received_terms
                .insert(hash.to_string(), capsule_id.to_string());
            inner.waiting_payee.remove(hash)
        };
        if let Some(event) = waiting {
            self.payee_observed(capsules, &event, hash, capsule_id);
        }
    }

    /// How many entries each kind holds, in field order.
    #[cfg(test)]
    fn held(&self) -> [usize; 8] {
        let inner = self.inner();
        [
            inner.own_terms.len(),
            inner.output_hash.len(),
            inner.peer.len(),
            inner.unpushed.len(),
            inner.response.len(),
            inner.invoices.len(),
            inner.received_terms.len(),
            inner.waiting_payee.len(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settlement_channel::{host_shaped_event, parse_and_check};
    use serde_json::Value;
    use std::path::Path;

    const INPUT: &str = "8437d5a5e0c5aa3304c0e0d17288cd504593ca7e560e3691a30739f1725ee7c5";
    const OUTPUT: &str = "59c0c4d2859c994004ad65ce13831b9cdaf688828d6c9f08cdb06f1bda9f15dd";
    const RESPONSE: &str = "33e3aff1f39db84ca6bc2e4a8d19703c57566b215e2fa5282dcfefc983bf4ebf";
    const PROVIDER_PEER: &str = "284a8bb62c0491b1a27f57425ce585626f7dc9a1495de4519a869a8c11d9f478";
    const WALLET: &[(&str, u64)] = &[("credited_msat", 995), ("fee_msat", 5)];

    /// A host-shaped event, with role and the wallet's numbers as given.
    fn event(
        role: &str,
        exchange_id: &str,
        phase: Phase,
        segment: u32,
        hash: &str,
        amount_msat: u64,
        wallet: &[(&str, u64)],
    ) -> PaymentLifecycleEvent {
        let mut body =
            host_shaped_event(exchange_id, phase, Some(segment), Some(hash), amount_msat);
        body["role"] = json!(role);
        if role == "provider"
            && matches!(
                phase,
                Phase::InputInvoiceIssued | Phase::OutputInvoiceIssued
            )
        {
            body["source"] = json!("provider_asserted");
        }
        for (k, v) in wallet {
            body[*k] = json!(v);
        }
        body["event_ref"] = json!("");
        body["event_ref"] = json!(crate::producer::jcs::json_digest(&body).unwrap());
        parse_and_check(&serde_json::to_vec(&body).unwrap()).expect("a host-shaped event")
    }

    fn legs_of(dir: &Path) -> Vec<Value> {
        std::fs::read_to_string(dir.join("ledger").join("capsules.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .filter(|c| c.get("settlement").is_some())
            .collect()
    }

    fn kinds(legs: &[Value]) -> Vec<&str> {
        legs.iter()
            .map(|c| c["settlement"]["leg"].as_str().unwrap())
            .collect()
    }

    struct Pair {
        payer: CapsuleState,
        provider: CapsuleState,
    }

    impl Pair {
        fn open(a: &Path, b: &Path) -> Self {
            Pair {
                payer: CapsuleState::open(a, "payer-node").unwrap(),
                provider: CapsuleState::open(b, "provider-node").unwrap(),
            }
        }
        fn payer_event(
            &self,
            phase: Phase,
            seg: u32,
            hash: &str,
            wallet: &[(&str, u64)],
        ) -> Vec<Push> {
            self.payer.settlement_legs().on_lifecycle(
                &self.payer,
                &event("payer", "ex", phase, seg, hash, 1000, wallet),
            )
        }
        fn provider_event(&self, phase: Phase, seg: u32, hash: &str, wallet: &[(&str, u64)]) {
            self.provider.settlement_legs().on_lifecycle(
                &self.provider,
                &event("provider", "p-ex", phase, seg, hash, 1000, wallet),
            );
        }
        fn deliver(&self, pushes: &[Push]) {
            for push in pushes {
                assert_eq!(push.peer, PROVIDER_PEER);
                self.provider
                    .settlement_legs()
                    .on_received(&self.provider, &push.capsule);
            }
        }
    }

    /// Both nodes of one paid exchange, in the order a live run sees it:
    /// invoices, then each wallet's settlement, then the payer's exchange
    /// event naming the provider (when the terms legs are pushed).
    fn two_nodes(a: &Path, b: &Path, provider_wallet: &[(&str, u64)]) -> (Vec<Value>, Vec<Value>) {
        let pair = Pair::open(a, b);
        for (phase, seg, hash) in [
            (Phase::InputInvoiceIssued, 0, INPUT),
            (Phase::OutputInvoiceIssued, 1, OUTPUT),
        ] {
            pair.provider_event(phase, seg, hash, &[]);
            assert!(
                pair.payer_event(phase, seg, hash, &[]).is_empty(),
                "no peer known yet"
            );
        }
        for (phase, seg, hash) in [
            (Phase::InputSettlementObserved, 0, INPUT),
            (Phase::OutputSettlementObserved, 1, OUTPUT),
        ] {
            pair.provider_event(phase, seg, hash, provider_wallet);
            pair.payer_event(phase, seg, hash, &[("fee_msat", 0)]);
        }
        let pushes = pair.payer.settlement_legs().on_exchange(
            &pair.payer,
            "ex",
            Some(PROVIDER_PEER),
            Some(RESPONSE),
        );
        assert_eq!(pushes.len(), 2);
        pair.deliver(&pushes);
        (legs_of(a), legs_of(b))
    }

    fn dirs() -> (tempfile::TempDir, tempfile::TempDir) {
        (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap())
    }

    #[test]
    fn a_paid_exchange_yields_both_sides_legs() {
        let (a, b) = dirs();
        let (payer, provider) = two_nodes(a.path(), b.path(), WALLET);
        assert_eq!(
            kinds(&payer),
            [
                "terms",
                "terms",
                "payer_observed",
                "payer_observed",
                "delivered"
            ]
        );
        assert_eq!(kinds(&provider), ["payee_observed", "payee_observed"]);
        let terms: Vec<&str> = payer[..2]
            .iter()
            .map(|c| c["capsule_id"].as_str().unwrap())
            .collect();
        for leg in provider.iter().chain(&payer[2..4]) {
            assert!(terms.contains(&leg["settlement"]["terms_ref"].as_str().unwrap()));
        }
        assert_eq!(provider[0]["settlement"]["received"]["value"], json!("995"));
        assert_eq!(
            provider[0]["settlement"]["receive_fee"]["value"],
            json!("5")
        );
        assert_eq!(payer[2]["settlement"]["routing_fee"]["value"], json!("0"));
        assert_eq!(
            payer[4]["settlement"]["terms_ref"],
            json!(terms[1]),
            "delivery cites the output invoice"
        );
        assert_eq!(
            payer[4]["settlement"]["delivery"]["content_digest"],
            json!(RESPONSE)
        );
        assert_eq!(payer[0]["settlement"]["sealer_role"], json!("payer"));
    }

    #[test]
    fn a_host_without_wallet_amounts_gets_no_payee_leg() {
        let (a, b) = dirs();
        let (payer, provider) = two_nodes(a.path(), b.path(), &[]);
        assert!(provider.is_empty());
        assert_eq!(payer.len(), 5);
    }

    #[test]
    fn a_terms_leg_that_arrives_first_is_cited_when_the_wallet_reports() {
        let (a, b) = dirs();
        let pair = Pair::open(a.path(), b.path());
        pair.provider_event(Phase::InputInvoiceIssued, 0, INPUT, &[]);
        pair.payer
            .settlement_legs()
            .on_exchange(&pair.payer, "ex", Some(PROVIDER_PEER), None);
        let pushes = pair.payer_event(Phase::InputInvoiceIssued, 0, INPUT, &[]);
        assert_eq!(pushes.len(), 1, "the peer was known: pushed at once");
        pair.deliver(&pushes);
        assert!(legs_of(b.path()).is_empty());
        pair.provider_event(Phase::InputSettlementObserved, 0, INPUT, WALLET);
        let provider = legs_of(b.path());
        assert_eq!(kinds(&provider), ["payee_observed"]);
        assert_eq!(
            provider[0]["settlement"]["terms_ref"],
            pushes[0].capsule["capsule_id"]
        );
    }

    #[test]
    fn the_provider_cites_only_a_payers_terms_leg_for_its_own_invoice_at_its_amount() {
        let (a, b) = dirs();
        let pair = Pair::open(a.path(), b.path());
        pair.provider_event(Phase::InputInvoiceIssued, 0, INPUT, &[]);
        pair.provider_event(Phase::InputSettlementObserved, 0, INPUT, WALLET);
        let forged = |amount_msat: u64, role: &str, hash: &str| {
            let member = build_leg(
                "terms",
                role,
                [("amount", msat(amount_msat)), ("payment_ref", ln(hash))]
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            )
            .unwrap();
            // signed by a fresh key each time, like any peer could
            let key = ed25519_dalek::SigningKey::from_bytes(
                &[amount_msat as u8 ^ hash.as_bytes()[0]; 32],
            );
            crate::producer::capsule::seal_settlement_leg(
                format!("t/{amount_msat}/{role}/{hash}"),
                member,
                None,
                None,
                &key,
            )
            .unwrap()
        };
        let legs = pair.provider.settlement_legs();
        legs.on_received(&pair.provider, &forged(1, "payer", INPUT));
        legs.on_received(&pair.provider, &forged(1000, "payer", OUTPUT));
        assert!(
            legs_of(b.path()).is_empty(),
            "a wrong amount, or an invoice not issued here, is never cited"
        );
        let genuine = forged(1000, "payer", INPUT);
        legs.on_received(&pair.provider, &genuine);
        let provider = legs_of(b.path());
        assert_eq!(provider.len(), 1);
        assert_eq!(
            provider[0]["settlement"]["terms_ref"],
            genuine["capsule_id"]
        );
        // a later, different terms leg for the same invoice changes nothing
        let mut later = genuine.clone();
        later["capsule_id"] = json!("f".repeat(64));
        legs.on_received(&pair.provider, &later);
        assert_eq!(
            legs.inner().received_terms.get(INPUT),
            genuine["capsule_id"].as_str().map(String::from).as_ref()
        );
    }

    #[test]
    fn a_push_that_failed_is_retried() {
        let (a, b) = dirs();
        let pair = Pair::open(a.path(), b.path());
        pair.payer
            .settlement_legs()
            .on_exchange(&pair.payer, "ex", Some(PROVIDER_PEER), None);
        let pushes = pair.payer_event(Phase::InputInvoiceIssued, 0, INPUT, &[]);
        assert!(pair.payer.settlement_legs().retry().is_empty());
        pair.payer.settlement_legs().requeue(&pushes[0]);
        let again = pair.payer.settlement_legs().retry();
        assert_eq!(again.len(), 1);
        assert_eq!(again[0].capsule, pushes[0].capsule);
        assert!(
            pair.payer.settlement_legs().retry().is_empty(),
            "taken once"
        );
    }

    #[test]
    fn an_exchanges_entries_are_cleared_once_its_delivery_is_sealed() {
        let (a, b) = dirs();
        let pair = Pair::open(a.path(), b.path());
        let legs = pair.payer.settlement_legs();
        pair.payer_event(Phase::OutputInvoiceIssued, 1, OUTPUT, &[]);
        legs.on_exchange(&pair.payer, "ex", Some(PROVIDER_PEER), Some(RESPONSE));
        assert_eq!(kinds(&legs_of(a.path())), ["terms", "delivered"]);
        let [_, output_hash, peer, unpushed, response, ..] = legs.held();
        assert_eq!((output_hash, peer, unpushed, response), (0, 0, 0, 0));
    }

    #[test]
    fn entries_older_than_the_time_bound_are_dropped() {
        let (a, b) = dirs();
        let pair = Pair::open(a.path(), b.path());
        pair.payer_event(Phase::InputInvoiceIssued, 0, INPUT, &[]);
        pair.provider_event(Phase::InputInvoiceIssued, 0, INPUT, &[]);
        pair.provider_event(Phase::InputSettlementObserved, 0, INPUT, WALLET);
        let (payer, provider) = (
            pair.payer.settlement_legs(),
            pair.provider.settlement_legs(),
        );
        payer.expire(Instant::now());
        assert_ne!(payer.held(), [0; 8], "fresh entries stay");
        let later = Instant::now() + MAX_AGE + Duration::from_secs(1);
        payer.expire(later);
        provider.expire(later);
        assert_eq!(payer.held(), [0; 8]);
        assert_eq!(provider.held(), [0; 8]);
        assert!(payer.retry().is_empty());
    }

    #[test]
    fn a_repeated_event_seals_nothing_more() {
        let dir = tempfile::tempdir().unwrap();
        let state = CapsuleState::open(dir.path(), "payer-node").unwrap();
        let invoice = event("payer", "x", Phase::InputInvoiceIssued, 0, INPUT, 1000, &[]);
        state.settlement_legs().on_lifecycle(&state, &invoice);
        state.settlement_legs().on_lifecycle(&state, &invoice);
        assert_eq!(legs_of(dir.path()).len(), 1);
    }

    #[test]
    fn waiting_entries_are_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let state = CapsuleState::open(dir.path(), "provider-node").unwrap();
        let legs = Legs::default();
        for i in 0..(MAX_HELD + 10) {
            let hash = format!("{i:064x}");
            legs.on_lifecycle(
                &state,
                &event(
                    "provider",
                    "p",
                    Phase::InputInvoiceIssued,
                    0,
                    &hash,
                    1000,
                    &[],
                ),
            );
            legs.on_lifecycle(
                &state,
                &event(
                    "provider",
                    "p",
                    Phase::InputSettlementObserved,
                    0,
                    &hash,
                    1000,
                    WALLET,
                ),
            );
        }
        let [.., invoices, _, waiting] = legs.held();
        assert_eq!((invoices, waiting), (MAX_HELD, MAX_HELD));
    }

    /// Both nodes' legs, checked by the Python reference verifier:
    /// AAC_PYTHON=python3 AAC_SETTLEMENT_VERIFY_SCRIPT=<capsule-emit>/rust/capsule-emit/tests/scripts/verify_rust_settlement.py
    #[test]
    #[ignore]
    fn the_reference_derives_agreed_for_both_invoices() {
        let (Ok(python), Ok(script)) = (
            std::env::var("AAC_PYTHON"),
            std::env::var("AAC_SETTLEMENT_VERIFY_SCRIPT"),
        ) else {
            panic!("AAC_PYTHON / AAC_SETTLEMENT_VERIFY_SCRIPT not set (this test is run on purpose, so fail loudly)");
        };
        let (a, b) = dirs();
        let (payer, provider) = two_nodes(a.path(), b.path(), WALLET);
        let all: Vec<Value> = payer.into_iter().chain(provider).collect();
        let path = a.path().join("legs.json");
        std::fs::write(&path, serde_json::to_vec(&all).unwrap()).unwrap();
        let out = std::process::Command::new(python)
            .arg(script)
            .arg(&path)
            .output()
            .unwrap();
        let report: Value = serde_json::from_slice(&out.stdout).expect("one JSON line");
        eprintln!("{report}");
        assert_eq!(report["conforming"], json!(true));
        let field = |name: &str| -> Vec<String> {
            report["settlements"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s[name].as_str().unwrap().to_string())
                .collect()
        };
        assert_eq!(field("payment_state"), ["agreed", "agreed"]);
        assert_eq!(
            field("delivery_state"),
            ["none", "stated"],
            "only the payer's delivered leg is sealed yet"
        );
    }
}
