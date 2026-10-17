//! The opt-in operator rule "stop routing to a peer after N referee-signed
//! contradictions within D days". Off unless the operator sets N.
//!
//! The rule never routes anything itself. When it fires it asks the host, over
//! the plugin protocol (`PeerBlockRequest`), to stop routing to that peer
//! until the operator undoes it. The host refuses unless its operator set
//! `allow_peer_blocks = true` for this plugin, and records the block as
//! requested by this plugin, never as the operator. A host without that path
//! (it does not list `peer_blocks.v1`) is never asked. The plugin calls no
//! operator route of the host.
//!
//! When the host blocks, this plugin seals the record itself, from the
//! choice the host answered: it names the rule and cites the verdicts that met
//! it (by commitment; see `crate::producer::capsule::RoutingRuleCitation`).
//! Undo is the operator's unblock, exactly as for a manual block.
//!
//! Inputs: only [`crate::verdict_counts::fold`] over this node's own chain:
//! verdicts whose referee signature was verified before they were recorded, from a referee
//! this node itself asked about that pair of halves, once per (referee, pair).
//! On top of that, one referee alone never fires the rule when N is 2 or more:
//! it contributes at most N - 1 contradictions ([`per_referee_cap`]).
//!
//! A verdict that met the rule is written to [`CITED_FILENAME`] whenever the
//! host holds a block of that peer afterwards (this rule's, or one already in
//! place, as `crate::peer_blocks_seen` knows), and never counts again. So
//! undoing a block holds: the rule fires again only on new contradictions. A
//! refused request cites nothing, and that peer is not asked again for an hour.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use chrono::{DateTime, Duration, Utc};
use serde_json::{json, Value};

use crate::capsule_emit::CapsuleState;
use crate::verdict_counts::{read_requested, PeerVerdict, CONTRADICTED};

pub const RULE_NAME: &str = "stop_routing_after_contradictions";
/// N. Unset, empty, `0` or not a number: the rule is off.
pub const ENV_AFTER: &str = "CAPSULES_STOP_ROUTING_AFTER_CONTRADICTIONS";
/// D, in days. Unset: [`DEFAULT_WINDOW_DAYS`]. `0` or not a number: off.
pub const ENV_WINDOW_DAYS: &str = "CAPSULES_STOP_ROUTING_WINDOW_DAYS";
pub const DEFAULT_WINDOW_DAYS: u32 = 30;
/// After the host refuses a block for a peer, it is not asked again for that
/// peer for this long.
const REFUSED_RETRY: std::time::Duration = std::time::Duration::from_secs(3600);
/// Beside the ledger: one line per rule block, the verdicts it cited, in
/// clear. Local; the sealed record carries only their commitments.
pub const CITED_FILENAME: &str = "routing-rule-cited.jsonl";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rule {
    pub after: u32,
    pub window_days: u32,
}

impl Rule {
    pub fn from_env() -> Option<Self> {
        rule_for(
            crate::settings::var(ENV_AFTER).ok().as_deref(),
            crate::settings::var(ENV_WINDOW_DAYS).ok().as_deref(),
        )
    }
}

/// A value that isn't a positive whole number turns the rule off rather than
/// being guessed at.
pub(crate) fn rule_for(after: Option<&str>, window_days: Option<&str>) -> Option<Rule> {
    let positive = |raw: &str| {
        let raw = raw.trim();
        (!raw.is_empty() && raw.bytes().all(|b| b.is_ascii_digit()))
            .then(|| raw.parse::<u32>().ok())
            .flatten()
            .filter(|n| *n > 0)
    };
    let after = positive(after?)?;
    let window_days = match window_days {
        None => DEFAULT_WINDOW_DAYS,
        Some(raw) => positive(raw)?,
    };
    Some(Rule { after, window_days })
}

/// The rule met for one peer: the verdicts that met it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Firing {
    pub peer_id: String,
    pub verdict_capsule_ids: Vec<String>,
}

/// The most contradictions one referee can contribute toward N: N - 1, so
/// that with N of 2 or more no single referee blocks a peer alone. N = 1 is
/// the operator's explicit choice to act on one referee's word.
pub fn per_referee_cap(rule: Rule) -> usize {
    (rule.after as usize).saturating_sub(1).max(1)
}

/// Every peer with at least `rule.after` contradictions recorded within the
/// last `rule.window_days` days, not counting verdicts an earlier rule block
/// already cited, and at most [`per_referee_cap`] from any one referee. A
/// verdict with no readable time counts nowhere.
pub fn due(
    rule: Rule,
    by_peer: &BTreeMap<String, Vec<PeerVerdict>>,
    cited: &HashSet<String>,
    now: DateTime<Utc>,
) -> Vec<Firing> {
    let since = now - Duration::days(i64::from(rule.window_days));
    by_peer
        .iter()
        .filter_map(|(peer, verdicts)| {
            let mut per_referee: HashMap<&str, usize> = HashMap::new();
            let ids: Vec<String> = verdicts
                .iter()
                .filter(|v| v.bucket == CONTRADICTED && !cited.contains(&v.verdict_capsule_id))
                .filter(|v| {
                    v.recorded_at
                        .as_deref()
                        .and_then(|at| DateTime::parse_from_rfc3339(at).ok())
                        .is_some_and(|at| at >= since && at <= now)
                })
                .filter(|v| {
                    let taken = per_referee.entry(v.referee_node_id.as_str()).or_default();
                    *taken += 1;
                    *taken <= per_referee_cap(rule)
                })
                .map(|v| v.verdict_capsule_id.clone())
                .collect();
            (ids.len() >= rule.after as usize).then(|| Firing {
                peer_id: peer.clone(),
                verdict_capsule_ids: ids,
            })
        })
        .collect()
}

/// What the rule asks of the host for the firings of one evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Planned {
    /// This host has no plugin peer-block path: nothing is asked.
    NoHostHook,
    /// The peer is already blocked: nothing is asked, and its verdicts are
    /// spent.
    AlreadyBlocked(Firing),
    /// Ask the host to block the peer.
    Block(Firing),
}

/// The steps for `firings`, given whether the host already blocks a peer
/// (`None`: the host has no peer-block path). Pure: the decision the parity
/// corpus checks.
pub fn plan(firings: &[Firing], blocked: Option<&dyn Fn(&str) -> bool>) -> Vec<Planned> {
    if firings.is_empty() {
        return Vec::new();
    }
    let Some(blocked) = blocked else {
        return vec![Planned::NoHostHook];
    };
    firings
        .iter()
        .map(|firing| {
            if blocked(&firing.peer_id) {
                Planned::AlreadyBlocked(firing.clone())
            } else {
                Planned::Block(firing.clone())
            }
        })
        .collect()
}

/// Every verdict id a rule block has cited.
pub fn read_cited(ledger_dir: &Path) -> HashSet<String> {
    let Ok(text) = std::fs::read_to_string(ledger_dir.join(CITED_FILENAME)) else {
        return HashSet::new();
    };
    text.lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter_map(|line| {
            line.get("verdict_capsule_ids")
                .and_then(Value::as_array)
                .cloned()
        })
        .flatten()
        .filter_map(|id| id.as_str().map(str::to_string))
        .collect()
}

/// Note `verdict_capsule_ids` as acted on, with the sealed record's id (and
/// the salt of its peer commitment, which only this node keeps) when the rule
/// sealed one.
pub fn record_cited(
    ledger_dir: &Path,
    routing_choice_capsule_id: Option<&str>,
    salt_hex: Option<&str>,
    verdict_capsule_ids: &[String],
) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(ledger_dir.join(CITED_FILENAME))?;
    let line = json!({
        "routing_choice_capsule_id": routing_choice_capsule_id,
        "peer_commitment_salt": salt_hex,
        "verdict_capsule_ids": verdict_capsule_ids,
    });
    writeln!(file, "{line}")
}

/// Why the host did not block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The host has no plugin peer-block path (it does not list
    /// `peer_blocks.v1`); nothing was sent.
    NoHostPath,
    /// The host answered with an error: for one, its operator has not set
    /// `allow_peer_blocks = true` for this plugin.
    Refused(String),
}

/// The host's side of a block request. [`ContextHost`] is the real one.
pub trait BlockHost {
    /// Ask the host to stop routing to `peer_id` until undone. `Ok` carries
    /// the applied choice, as the host publishes it on `routing.choice.v1`.
    fn block(
        &mut self,
        peer_id: &str,
        reason_json: String,
    ) -> impl std::future::Future<Output = Result<String, Refusal>> + Send;
}

/// The host this plugin runs in, through its plugin protocol.
pub struct ContextHost<'a, 'b>(pub &'a mut mesh_llm_plugin::PluginContext<'b>);

impl BlockHost for ContextHost<'_, '_> {
    async fn block(&mut self, peer_id: &str, reason_json: String) -> Result<String, Refusal> {
        use mesh_llm_plugin::proto::{
            self,
            peer_block_request::{Change, Length},
        };
        if !self
            .0
            .host_supports(mesh_llm_plugin::host_capabilities::PEER_BLOCKS)
        {
            return Err(Refusal::NoHostPath);
        }
        let request = proto::PeerBlockRequest {
            change: Change::Block as i32,
            peer_id: peer_id.to_string(),
            length: Length::UntilUndone as i32,
            reason_json: Some(reason_json),
        };
        match self.0.request_peer_block(request).await {
            Ok(response) => Ok(response.choice_json),
            Err(error) => Err(Refusal::Refused(error.to_string())),
        }
    }
}

/// What one evaluation did, per peer, for the log and the tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The host blocked the peer; `sealed`: this node sealed the record.
    Blocked { peer_id: String, sealed: bool },
    /// The peer was already blocked; nothing asked.
    AlreadyBlocked { peer_id: String },
    /// This host has no plugin peer-block path; nothing asked.
    NoHostPath,
    /// The host refused (for one, `allow_peer_blocks` is off) or failed.
    Refused { peer_id: String, error: String },
}

fn refused_recently() -> &'static Mutex<HashMap<String, Instant>> {
    static REFUSED: std::sync::OnceLock<Mutex<HashMap<String, Instant>>> =
        std::sync::OnceLock::new();
    REFUSED.get_or_init(Mutex::default)
}

/// Seal this node's record of a block the rule asked for, from the host's
/// applied `choice_json`. Returns the record id and the commitment's salt.
fn seal_rule_block(
    capsules: &CapsuleState,
    rule: Rule,
    firing: &Firing,
    choice_json: &str,
) -> anyhow::Result<(String, String)> {
    let choice: Value = serde_json::from_str(choice_json)?;
    let until = choice
        .get("until_ms")
        .and_then(Value::as_u64)
        .and_then(|ms| DateTime::<Utc>::from_timestamp_millis(ms as i64))
        .map(|at| at.to_rfc3339());
    let salt_hex = crate::producer::capsule::fresh_store_nonce();
    let salt: [u8; 32] = hex::decode(&salt_hex)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("a store nonce is 32 bytes"))?;
    let citation = crate::producer::capsule::RoutingRuleCitation {
        rule: RULE_NAME,
        after: rule.after,
        window_days: rule.window_days,
        verdict_capsule_ids: &firing.verdict_capsule_ids,
    };
    let emitted = capsules.emit_local_routing_choice(
        crate::producer::capsule::RoutingChoiceChange::Block,
        &firing.peer_id,
        until.as_deref(),
        &salt,
        Some(&citation),
    )?;
    Ok((emitted.capsule_id, salt_hex))
}

/// Run the rule once against this node's chain, asking `host` for any block.
pub async fn evaluate<H: BlockHost>(
    capsules: &Arc<CapsuleState>,
    rule: Rule,
    host: &mut H,
    now: DateTime<Utc>,
) -> Vec<Outcome> {
    let ledger_dir = capsules.ledger_dir().to_path_buf();
    let (by_peer, cited, blocked) = match tokio::task::spawn_blocking(move || {
        let records = crate::evidence_panes::read_capsule_records(&ledger_dir);
        (
            crate::verdict_counts::fold(&records, &read_requested(&ledger_dir)),
            read_cited(&ledger_dir),
            crate::peer_blocks_seen::blocked_now(&ledger_dir),
        )
    })
    .await
    {
        Ok(read) => read,
        Err(error) => {
            return vec![Outcome::Refused {
                peer_id: String::new(),
                error: error.to_string(),
            }];
        }
    };
    let mut outcomes = Vec::new();
    let firings = due(rule, &by_peer, &cited, now);
    let holds = |peer: &str| blocked.iter().any(|b| b == peer);
    for planned in plan(&firings, Some(&holds)) {
        let firing = match planned {
            Planned::NoHostHook => {
                outcomes.push(Outcome::NoHostPath);
                break;
            }
            Planned::AlreadyBlocked(firing) => {
                // Already blocked: these verdicts are spent, nothing is asked.
                note_cited(capsules, None, None, &firing);
                outcomes.push(Outcome::AlreadyBlocked {
                    peer_id: firing.peer_id,
                });
                continue;
            }
            Planned::Block(firing) => firing,
        };
        {
            let refused = refused_recently()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if refused
                .get(&firing.peer_id)
                .is_some_and(|at| at.elapsed() < REFUSED_RETRY)
            {
                continue;
            }
        }
        let reason = json!({
            "rule": RULE_NAME,
            "after": rule.after,
            "window_days": rule.window_days,
            "contradictions": firing.verdict_capsule_ids.len(),
        })
        .to_string();
        match host.block(&firing.peer_id, reason).await {
            Ok(choice_json) => {
                if let Err(error) =
                    crate::peer_blocks_seen::record(capsules.ledger_dir(), choice_json.as_bytes())
                {
                    tracing::warn!(%error, "the host's routing choice was not kept");
                }
                let sealed = {
                    let (capsules, firing, choice_json) =
                        (capsules.clone(), firing.clone(), choice_json.clone());
                    tokio::task::spawn_blocking(move || {
                        seal_rule_block(&capsules, rule, &firing, &choice_json)
                    })
                    .await
                };
                let (record, salt) = match sealed {
                    Ok(Ok((id, salt))) => (Some(id), Some(salt)),
                    Ok(Err(error)) => {
                        tracing::warn!(%error, peer_id = %firing.peer_id, "the peer is blocked, but its record did not seal");
                        (None, None)
                    }
                    Err(error) => {
                        tracing::warn!(%error, peer_id = %firing.peer_id, "the peer is blocked, but its record did not seal");
                        (None, None)
                    }
                };
                note_cited(capsules, record.as_deref(), salt.as_deref(), &firing);
                outcomes.push(Outcome::Blocked {
                    peer_id: firing.peer_id,
                    sealed: record.is_some(),
                });
            }
            Err(Refusal::NoHostPath) => {
                outcomes.push(Outcome::NoHostPath);
                break;
            }
            Err(Refusal::Refused(error)) => {
                refused_recently()
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(firing.peer_id.clone(), Instant::now());
                outcomes.push(Outcome::Refused {
                    peer_id: firing.peer_id,
                    error,
                });
            }
        }
    }
    outcomes
}

/// Whenever the host holds a block of this peer, these verdicts have been
/// acted on: note them, so an undo is not undone by the same verdicts later.
fn note_cited(capsules: &CapsuleState, record: Option<&str>, salt: Option<&str>, firing: &Firing) {
    if let Err(error) = record_cited(
        capsules.ledger_dir(),
        record,
        salt,
        &firing.verdict_capsule_ids,
    ) {
        tracing::warn!(%error, peer_id = %firing.peer_id, "rule verdicts not noted as cited; an undo could re-fire on them");
    }
}

/// Set when a verdict lands on the chain (and at start): the rule is
/// evaluated at the next point the plugin can reach its host.
static DUE: AtomicBool = AtomicBool::new(true);

/// A verdict landed: evaluate the rule at the next chance.
pub fn mark_due() {
    DUE.store(true, Ordering::SeqCst);
}

/// Evaluate the rule now if it is on and a verdict landed since the last
/// evaluation. Called from the plugin's handlers, which hold the host's
/// context; one evaluation at a time.
pub async fn evaluate_if_due(
    context: &mut mesh_llm_plugin::PluginContext<'_>,
    capsules: &Arc<CapsuleState>,
) {
    let Some(rule) = Rule::from_env() else {
        return;
    };
    if !DUE.swap(false, Ordering::SeqCst) {
        return;
    }
    static ONE_AT_A_TIME: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    let _one = ONE_AT_A_TIME.lock().await;
    for outcome in evaluate(capsules, rule, &mut ContextHost(context), Utc::now()).await {
        match outcome {
            Outcome::Blocked { peer_id, sealed } => {
                tracing::info!(%peer_id, sealed, rule = RULE_NAME, "operator rule stopped routing to a peer")
            }
            Outcome::AlreadyBlocked { peer_id } => {
                tracing::info!(%peer_id, rule = RULE_NAME, "rule met for a peer already blocked; nothing asked")
            }
            Outcome::NoHostPath => {
                tracing::warn!(rule = RULE_NAME, "rule met, but this host has no plugin peer-block path (peer_blocks.v1); nothing blocked")
            }
            Outcome::Refused { peer_id, error } => {
                tracing::warn!(%peer_id, %error, rule = RULE_NAME, "rule met, but the host did not block (it needs allow_peer_blocks = true for this plugin)")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verdict_counts::{CORROBORATED, INCONCLUSIVE, REQUESTED_FILENAME};

    const NOW: &str = "2026-09-28T16:00:00Z";
    const RULE: Rule = Rule {
        after: 3,
        window_days: 7,
    };

    fn now() -> DateTime<Utc> {
        NOW.parse().unwrap()
    }

    /// A counted verdict about peer `p`, from a referee of its own.
    fn verdict(id: &str, bucket: &'static str, days_ago: i64) -> PeerVerdict {
        by_referee(id, &format!("referee-{id}"), bucket, days_ago)
    }

    fn by_referee(id: &str, referee: &str, bucket: &'static str, days_ago: i64) -> PeerVerdict {
        PeerVerdict {
            verdict_capsule_id: id.to_string(),
            referee_node_id: referee.to_string(),
            bucket,
            recorded_at: Some((now() - Duration::days(days_ago)).to_rfc3339()),
            model_hash: None,
        }
    }

    fn peer(verdicts: Vec<PeerVerdict>) -> BTreeMap<String, Vec<PeerVerdict>> {
        BTreeMap::from([("p".to_string(), verdicts)])
    }

    #[test]
    fn off_unless_the_operator_sets_a_positive_n() {
        assert_eq!(rule_for(None, None), None);
        assert_eq!(rule_for(Some(""), None), None);
        assert_eq!(rule_for(Some("0"), None), None);
        assert_eq!(rule_for(Some("three"), Some("7")), None);
        assert_eq!(
            rule_for(Some("3"), Some("0")),
            None,
            "a zero window is off, not 'forever'"
        );
        assert_eq!(rule_for(Some("3"), Some("x")), None);
        assert_eq!(
            rule_for(Some("3"), None),
            Some(Rule {
                after: 3,
                window_days: DEFAULT_WINDOW_DAYS
            })
        );
        assert_eq!(rule_for(Some(" 3 "), Some("7")), Some(RULE));
    }

    #[test]
    fn fires_at_exactly_n_within_d() {
        let by_peer = peer(vec![
            verdict("a", CONTRADICTED, 0),
            verdict("b", CONTRADICTED, 3),
            verdict("c", CONTRADICTED, 7),
        ]);
        assert_eq!(
            due(RULE, &by_peer, &HashSet::new(), now()),
            vec![Firing {
                peer_id: "p".into(),
                verdict_capsule_ids: vec!["a".into(), "b".into(), "c".into()]
            }]
        );
    }

    #[test]
    fn does_not_fire_at_n_minus_one() {
        let by_peer = peer(vec![
            verdict("a", CONTRADICTED, 0),
            verdict("b", CONTRADICTED, 1),
            // Not contradictions: they never count toward N.
            verdict("c", CORROBORATED, 1),
            verdict("d", INCONCLUSIVE, 1),
        ]);
        assert!(due(RULE, &by_peer, &HashSet::new(), now()).is_empty());
    }

    #[test]
    fn a_contradiction_outside_the_window_does_not_count() {
        let by_peer = peer(vec![
            verdict("a", CONTRADICTED, 0),
            verdict("b", CONTRADICTED, 6),
            verdict("c", CONTRADICTED, 8),
        ]);
        assert!(due(RULE, &by_peer, &HashSet::new(), now()).is_empty());
    }

    #[test]
    fn a_contradiction_without_a_time_does_not_count() {
        let mut undated = verdict("c", CONTRADICTED, 0);
        undated.recorded_at = None;
        let by_peer = peer(vec![
            verdict("a", CONTRADICTED, 0),
            verdict("b", CONTRADICTED, 1),
            undated,
        ]);
        assert!(due(RULE, &by_peer, &HashSet::new(), now()).is_empty());
    }

    #[test]
    fn verdicts_an_earlier_rule_block_cited_never_count_again() {
        let by_peer = peer(vec![
            verdict("a", CONTRADICTED, 0),
            verdict("b", CONTRADICTED, 1),
            verdict("c", CONTRADICTED, 2),
        ]);
        let cited: HashSet<String> = ["a".to_string()].into();
        assert!(due(RULE, &by_peer, &cited, now()).is_empty());
    }

    /// With N of 2 or more, one referee alone never fires the rule: it
    /// contributes at most N - 1.
    #[test]
    fn one_referee_alone_never_reaches_n() {
        assert_eq!(
            per_referee_cap(Rule {
                after: 1,
                window_days: 7
            }),
            1
        );
        assert_eq!(
            per_referee_cap(Rule {
                after: 2,
                window_days: 7
            }),
            1
        );
        assert_eq!(per_referee_cap(RULE), 2);
        let one_referee: Vec<PeerVerdict> = ["a", "b", "c", "d"]
            .iter()
            .map(|id| by_referee(id, "r1", CONTRADICTED, 0))
            .collect();
        assert!(due(RULE, &peer(one_referee.clone()), &HashSet::new(), now()).is_empty());
        let mut two_referees = one_referee;
        two_referees.push(by_referee("e", "r2", CONTRADICTED, 0));
        assert_eq!(
            due(RULE, &peer(two_referees), &HashSet::new(), now()),
            vec![Firing {
                peer_id: "p".into(),
                verdict_capsule_ids: vec!["a".into(), "b".into(), "e".into()]
            }]
        );
        let n_one = Rule {
            after: 1,
            window_days: 7,
        };
        assert_eq!(
            due(
                n_one,
                &peer(vec![by_referee("a", "r1", CONTRADICTED, 0)]),
                &HashSet::new(),
                now()
            )
            .len(),
            1
        );
    }

    /// "Not for unsigned verdicts": the rule reads only the fold, and the fold
    /// reads only the records sealed after a referee's signature verified. A bare
    /// adjudication block claiming three contradictions fires nothing.
    #[test]
    fn unsigned_verdicts_never_fire_the_rule() {
        let bare = |id: &str| {
            json!({ "model_attestation": { "compute_attestation": { "adjudication": {
                "verdict": "contradicted:p", "verdict_capsule_id": id, "referee_node_id": format!("r-{id}"),
                "halves": [format!("{id}-a"), format!("{id}-b")],
                "half_node_ids": ["p", "q"], "received_at": NOW,
            }}}})
        };
        let by_peer =
            crate::verdict_counts::fold(&[bare("a"), bare("b"), bare("c")], &HashSet::new());
        assert!(due(RULE, &by_peer, &HashSet::new(), now()).is_empty());
    }

    /// A stand-in for the host's plugin peer-block path.
    #[derive(Default)]
    struct FakeHost {
        /// The host does not list `peer_blocks.v1`.
        no_path: bool,
        /// The host refuses every request with this message.
        refuse: Option<String>,
        /// The peers asked to block, in order.
        asked: Vec<String>,
    }

    impl BlockHost for FakeHost {
        async fn block(&mut self, peer_id: &str, reason_json: String) -> Result<String, Refusal> {
            if self.no_path {
                return Err(Refusal::NoHostPath);
            }
            self.asked.push(peer_id.to_string());
            let reason: Value = serde_json::from_str(&reason_json).unwrap();
            assert_eq!(reason["rule"], json!(RULE_NAME));
            if let Some(message) = &self.refuse {
                return Err(Refusal::Refused(message.clone()));
            }
            Ok(json!({
                "change": "block", "peer": peer_id, "at_ms": 1000,
                "requested_by": "plugin:capsules", "reason": reason,
            })
            .to_string())
        }
    }

    /// The operator's own unblock (or block), as the host publishes it.
    fn host_publishes(capsules: &CapsuleState, change: &str, peer: &str, at_ms: u64) {
        let choice =
            json!({"change": change, "peer": peer, "at_ms": at_ms, "requested_by": "operator"});
        crate::peer_blocks_seen::record(capsules.ledger_dir(), choice.to_string().as_bytes())
            .unwrap();
    }

    /// The record ids the rule's blocks sealed, in order.
    fn sealed_ids(capsules: &CapsuleState) -> Vec<String> {
        std::fs::read_to_string(capsules.ledger_dir().join(CITED_FILENAME))
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter_map(|l| l["routing_choice_capsule_id"].as_str().map(str::to_string))
            .collect()
    }

    /// A verdict from `referee` contradicting `peer` (against its twin `q`),
    /// about a pair of halves of its own, recorded on the chain the way the
    /// verified-verdict path records one. With `asked`, this node's own request
    /// to that referee about that pair is noted too. Each test uses its own
    /// `peer`: the pending citations are one map per process.
    fn contradiction(
        capsules: &CapsuleState,
        peer: &str,
        id: &str,
        referee: &str,
        asked: bool,
        at: DateTime<Utc>,
    ) {
        let verdict = format!("contradicted:{peer}");
        let halves = [format!("{id}-a"), format!("{id}-b")];
        let facts = crate::capsule_emit::VerdictFacts {
            verdict: &verdict,
            verdict_capsule_id: id,
            referee_node_id: referee,
            halves: [&halves[0], &halves[1]],
            half_node_ids: [peer, "q"],
            twin_bracket_id: None,
            model_hash: None,
        };
        capsules
            .emit_adjudication_received(&facts, &halves[1], "q", &at.to_rfc3339())
            .unwrap()
            .expect("a new verdict seals a record");
        if asked {
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(capsules.ledger_dir().join(REQUESTED_FILENAME))
                .unwrap();
            writeln!(file, "{}", json!({ "referee": referee, "halves": halves })).unwrap();
        }
    }

    fn rule_record(capsules: &CapsuleState, capsule_id: &str) -> Value {
        crate::evidence_panes::read_capsule_records(capsules.ledger_dir())
            .into_iter()
            .find(|r| r["capsule_id"] == json!(capsule_id))
            .expect("the sealed record is on the chain")["model_attestation"]["compute_attestation"]
            ["local_routing_choice"]
            .clone()
    }

    fn open(dir: &tempfile::TempDir) -> Arc<CapsuleState> {
        Arc::new(CapsuleState::open(dir.path(), "node-under-test").unwrap())
    }

    const N2: Rule = Rule {
        after: 2,
        window_days: 7,
    };

    #[tokio::test]
    async fn fires_through_the_plugin_peer_block_path_and_undo_works() {
        let dir = tempfile::tempdir().unwrap();
        let capsules = open(&dir);
        let mut host = FakeHost::default();
        let now = Utc::now();
        let (v1, v2, v3, v4) = (
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            "4".repeat(64),
        );

        // N-1: nothing asked of the host.
        contradiction(&capsules, "p", &v1, "r1", true, now);
        assert!(evaluate(&capsules, N2, &mut host, now).await.is_empty());
        assert!(host.asked.is_empty());

        // N within D, from two referees this node asked: the host blocks, and
        // this node seals a record naming the rule and citing both verdicts.
        contradiction(&capsules, "p", &v2, "r2", true, now);
        assert_eq!(
            evaluate(&capsules, N2, &mut host, now).await,
            vec![Outcome::Blocked {
                peer_id: "p".into(),
                sealed: true
            }]
        );
        assert_eq!(host.asked, ["p"]);
        let fact = rule_record(&capsules, &sealed_ids(&capsules)[0]);
        assert_eq!(fact["change"], json!("block"));
        assert_eq!(fact["rule"]["rule"], json!(RULE_NAME));
        assert_eq!(
            fact["rule"]["verdict_commitments"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            read_cited(capsules.ledger_dir()),
            [v1.clone(), v2.clone()].into()
        );
        assert_eq!(
            crate::peer_blocks_seen::blocked_at(capsules.ledger_dir(), 2000),
            ["p"]
        );

        // The operator undoes it.
        host_publishes(&capsules, "unblock", "p", 5000);
        // The verdicts that already met the rule don't fire it again...
        assert!(evaluate(&capsules, N2, &mut host, now).await.is_empty());
        // ...one new contradiction is N-1 again...
        contradiction(&capsules, "p", &v3, "r1", true, now);
        assert!(evaluate(&capsules, N2, &mut host, now).await.is_empty());
        // ...and N new ones fire it.
        contradiction(&capsules, "p", &v4, "r3", true, now);
        assert_eq!(
            evaluate(&capsules, N2, &mut host, now).await,
            vec![Outcome::Blocked {
                peer_id: "p".into(),
                sealed: true
            }]
        );
        assert_eq!(host.asked, ["p", "p"]);
    }

    /// The adversarial case end to end: a peer with an announced key signs
    /// many contradictions of `f`, each with its own id and a pair this node
    /// never asked it about. The verified records exist, but nothing is
    /// asked of the host.
    #[tokio::test]
    async fn verdicts_this_node_never_asked_for_block_nobody() {
        let dir = tempfile::tempdir().unwrap();
        let capsules = open(&dir);
        let mut host = FakeHost::default();
        let now = Utc::now();
        for i in 0..6 {
            contradiction(
                &capsules,
                "f",
                &format!("{i}").repeat(64),
                &format!("x{}", i % 2),
                false,
                now,
            );
        }
        assert!(evaluate(&capsules, N2, &mut host, now).await.is_empty());
        assert!(host.asked.is_empty());
    }

    /// The host's own gate: with `allow_peer_blocks` off for this plugin, the
    /// host refuses, and nothing is blocked, sealed or used up. The same peer
    /// is not asked again within the hour.
    #[tokio::test]
    async fn with_allow_peer_blocks_off_nothing_is_blocked() {
        let dir = tempfile::tempdir().unwrap();
        let capsules = open(&dir);
        let mut host = FakeHost {
            refuse: Some(
                "plugin capsules may not request peer blocks: allow_peer_blocks is off".into(),
            ),
            ..FakeHost::default()
        };
        let now = Utc::now();
        contradiction(&capsules, "g", &"1".repeat(64), "r1", true, now);
        contradiction(&capsules, "g", &"2".repeat(64), "r2", true, now);
        let outcomes = evaluate(&capsules, N2, &mut host, now).await;
        assert!(
            matches!(&outcomes[..], [Outcome::Refused { peer_id, error }]
            if peer_id == "g" && error.contains("allow_peer_blocks"))
        );
        assert!(
            read_cited(capsules.ledger_dir()).is_empty(),
            "nothing used up"
        );
        assert!(sealed_ids(&capsules).is_empty(), "nothing sealed");
        assert!(
            crate::peer_blocks_seen::blocked_now(capsules.ledger_dir()).is_empty(),
            "nothing blocked"
        );
        assert!(
            evaluate(&capsules, N2, &mut host, now).await.is_empty(),
            "not asked again within the hour"
        );
        assert_eq!(host.asked, ["g"]);
    }

    /// A host without the plugin peer-block path (it does not list
    /// `peer_blocks.v1`, e.g. mesh-llm 0.77; 0.78 has it) is never asked.
    #[tokio::test]
    async fn a_host_without_the_path_blocks_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let capsules = open(&dir);
        let mut host = FakeHost {
            no_path: true,
            ..FakeHost::default()
        };
        let now = Utc::now();
        contradiction(&capsules, "s", &"1".repeat(64), "r1", true, now);
        assert_eq!(
            evaluate(
                &capsules,
                Rule {
                    after: 1,
                    window_days: 7
                },
                &mut host,
                now
            )
            .await,
            vec![Outcome::NoHostPath]
        );
        assert!(
            read_cited(capsules.ledger_dir()).is_empty(),
            "nothing blocked, so nothing is used up"
        );
        assert!(sealed_ids(&capsules).is_empty());
    }

    /// The operator had already blocked the peer when the rule was met: nothing
    /// is asked, the verdicts are spent, and undoing that block holds.
    #[tokio::test]
    async fn undo_holds_when_the_peer_was_already_blocked() {
        let dir = tempfile::tempdir().unwrap();
        let capsules = open(&dir);
        let mut host = FakeHost::default();
        let now = Utc::now();
        host_publishes(&capsules, "block", "w", 1000);
        contradiction(&capsules, "w", &"1".repeat(64), "r1", true, now);
        contradiction(&capsules, "w", &"2".repeat(64), "r2", true, now);
        assert_eq!(
            evaluate(&capsules, N2, &mut host, now).await,
            vec![Outcome::AlreadyBlocked {
                peer_id: "w".into()
            }]
        );
        assert!(host.asked.is_empty());
        host_publishes(&capsules, "unblock", "w", 2000);
        assert!(
            evaluate(&capsules, N2, &mut host, now).await.is_empty(),
            "the undo is not undone"
        );
    }

    #[test]
    fn the_cited_file_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_cited(dir.path()).is_empty());
        record_cited(
            dir.path(),
            Some("r1"),
            Some("ab"),
            &["a".into(), "b".into()],
        )
        .unwrap();
        record_cited(dir.path(), None, None, &["c".into()]).unwrap();
        assert_eq!(
            read_cited(dir.path()),
            ["a", "b", "c"].map(String::from).into()
        );
    }
}
