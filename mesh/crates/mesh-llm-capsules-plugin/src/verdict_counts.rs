//! Referee-signed verdicts about each peer, counted from this node's own
//! chain: the Peers drill's "Verdicts about them" counts, and the input the
//! opt-in stop-routing rule reads (`routing_rule`).
//!
//! Two gates, both required:
//!
//! 1. **Verified.** Only `adjudication_received` / `adjudication_issued` on this
//!    node's own chain count. Those are sealed only after the referee's
//!    signature has been checked under the key that referee announced; on the
//!    referee, it signed them itself.
//!    A verdict string in any other record counts nowhere.
//! 2. **Asked for by this node.** A signature only says which referee signed,
//!    not that anyone asked it: any peer with an announced key could mint as
//!    many verdicts about a peer as it likes. So a received verdict counts only
//!    when this node itself asked that referee about that exact pair of halves
//!    ([`REQUESTED_FILENAME`], written when this node sends an adjudicate
//!    request). A verdict delivered by anyone else, about a pair this node
//!    never put to that referee, counts nowhere. A verdict this node issued as
//!    referee is its own judgment and counts.
//!
//! Then each (referee, pair of halves) counts once, whatever number of
//! differently-signed verdicts a referee issues about it.
//!
//! Four counts per peer, each a list of the verdicts' capsule ids so the page
//! can open every one. No computed figure about a peer, nothing blended into
//! one number; the counts are this node's and are never sent to anyone.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::Path;

use crate::producer::capsule::{ADJUDICATION_ISSUED_BLOCK, ADJUDICATION_RECEIVED_BLOCK};
use serde_json::{json, Value};

pub const CORROBORATED: &str = "corroborated";
pub const INCONCLUSIVE: &str = "inconclusive";
pub const NOT_COMPARABLE: &str = "not_comparable";
pub const CONTRADICTED: &str = "contradicted";
const CONTRADICTED_PREFIX: &str = "contradicted:";

/// The four buckets, in the order the page shows them.
pub const BUCKETS: [&str; 4] = [CORROBORATED, CONTRADICTED, INCONCLUSIVE, NOT_COMPARABLE];

/// Beside the ledger: one line per adjudicate request this node sent,
/// `{referee, halves: [id, id], twin_bracket_id, asked_at}`.
pub const REQUESTED_FILENAME: &str = "requested-adjudications.jsonl";

/// `(referee, the pair of halves in sorted order)`.
type PairKey = (String, [String; 2]);

fn pair_key(referee: &str, a: &str, b: &str) -> PairKey {
    let mut halves = [a.to_string(), b.to_string()];
    halves.sort();
    (referee.to_string(), halves)
}

/// The (referee, pair) keys this node asked about. A missing or unreadable
/// file is "asked nothing": no received verdict counts.
pub fn read_requested(ledger_dir: &Path) -> HashSet<PairKey> {
    let Ok(text) = std::fs::read_to_string(ledger_dir.join(REQUESTED_FILENAME)) else {
        return HashSet::new();
    };
    text.lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter_map(|line| {
            let referee = line.get("referee")?.as_str()?.to_string();
            let halves = line.get("halves")?.as_array()?;
            match halves.as_slice() {
                [a, b] => Some(pair_key(&referee, a.as_str()?, b.as_str()?)),
                _ => None,
            }
        })
        .collect()
}

/// One verified verdict, as it bears on one judged peer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerVerdict {
    pub verdict_capsule_id: String,
    /// Who signed it: the rule caps how much one referee can count.
    pub referee_node_id: String,
    /// One of [`BUCKETS`].
    pub bucket: &'static str,
    /// When this node recorded the verdict (`received_at` or `issued_at`),
    /// minute-granular as sealed. `None` when the record carries neither.
    pub recorded_at: Option<String>,
    /// The model the twins served, as the verdict seals it: the counts, the
    /// bar window and the tiers are all per model.
    pub model_hash: Option<String>,
}

/// Which bucket `verdict` puts `node_id` in, when `node_id` is one of the two
/// judged twins. A contradiction naming the OTHER twin means the referee's
/// answer matched this one's (`crate::referee::verdict::referee_verdict`), so it is
/// corroborated for this one. An unknown ruling is in no bucket.
pub fn ruling_about(
    verdict: &str,
    half_node_ids: [&str; 2],
    node_id: &str,
) -> Option<&'static str> {
    if !half_node_ids.contains(&node_id) {
        return None;
    }
    if let Some(named) = verdict.strip_prefix(CONTRADICTED_PREFIX) {
        return if named == node_id {
            Some(CONTRADICTED)
        } else if half_node_ids.contains(&named) {
            Some(CORROBORATED)
        } else {
            None
        };
    }
    match verdict {
        CORROBORATED => Some(CORROBORATED),
        INCONCLUSIVE => Some(INCONCLUSIVE),
        NOT_COMPARABLE => Some(NOT_COMPARABLE),
        _ => None,
    }
}

/// The verdict block, its time key, and whether this node issued it.
fn verdict_block(record: &Value) -> Option<(&Value, &'static str, bool)> {
    let attestation = record.pointer("/model_attestation/compute_attestation")?;
    [
        (ADJUDICATION_RECEIVED_BLOCK, "received_at", false),
        (ADJUDICATION_ISSUED_BLOCK, "issued_at", true),
    ]
    .into_iter()
    .find_map(|(name, at, issued)| attestation.get(name).map(|block| (block, at, issued)))
}

/// Every counted verdict on `records` (this node's own chain), per judged
/// peer id, given the pairs this node asked about (`requested`, see
/// [`read_requested`]). Each (referee, pair of halves) counts once.
pub fn fold(records: &[Value], requested: &HashSet<PairKey>) -> BTreeMap<String, Vec<PeerVerdict>> {
    let mut seen_pairs: BTreeSet<PairKey> = BTreeSet::new();
    let mut by_peer: BTreeMap<String, Vec<PeerVerdict>> = BTreeMap::new();
    for record in records {
        let Some((block, at_key, issued)) = verdict_block(record) else {
            continue;
        };
        let text = |key: &str| {
            block
                .get(key)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
        };
        let (Some(verdict), Some(verdict_capsule_id), Some(referee)) = (
            text("verdict"),
            text("verdict_capsule_id"),
            text("referee_node_id"),
        ) else {
            continue;
        };
        let [Some(a), Some(b)] = pair(block, "half_node_ids") else {
            continue;
        };
        let [Some(half_a), Some(half_b)] = pair(block, "halves") else {
            continue;
        };
        if a == b || half_a == half_b || referee == a || referee == b {
            continue;
        }
        let key = pair_key(referee, half_a, half_b);
        if !issued
            && !requested.contains(&key)
            && !cfg!(feature = "mutant-referee-counts-skip-asked-gate")
        {
            continue;
        }
        if !seen_pairs.insert(key) {
            continue;
        }
        for peer in [a, b] {
            let Some(bucket) = ruling_about(verdict, [a, b], peer) else {
                continue;
            };
            by_peer
                .entry(peer.to_string())
                .or_default()
                .push(PeerVerdict {
                    verdict_capsule_id: verdict_capsule_id.to_string(),
                    referee_node_id: referee.to_string(),
                    bucket,
                    recorded_at: text(at_key).map(str::to_string),
                    model_hash: text("model_hash").map(str::to_string),
                });
        }
    }
    by_peer
}

/// A two-string array member, or `[None, None]` when it is not exactly that.
fn pair<'a>(block: &'a Value, key: &str) -> [Option<&'a str>; 2] {
    match block.get(key).and_then(Value::as_array).map(Vec::as_slice) {
        Some([a, b]) => [a.as_str(), b.as_str()],
        _ => [None, None],
    }
}

/// The page's shape: every bucket present, zero included, each with the ids
/// that make up its count.
pub fn counts_json(verdicts: &[PeerVerdict]) -> Value {
    let mut out = serde_json::Map::new();
    for bucket in BUCKETS {
        let ids: Vec<&str> = verdicts
            .iter()
            .filter(|v| v.bucket == bucket)
            .map(|v| v.verdict_capsule_id.as_str())
            .collect();
        out.insert(
            bucket.to_string(),
            json!({ "count": ids.len(), "verdict_capsule_ids": ids }),
        );
    }
    Value::Object(out)
}

/// [`counts_json`] per model hash: `{model_hash: {bucket: {count, ids}}}`.
/// A verdict that seals no model hash is under `"null"`.
pub fn counts_by_model(verdicts: &[PeerVerdict]) -> Value {
    let models: BTreeSet<Option<&str>> = verdicts.iter().map(|v| v.model_hash.as_deref()).collect();
    let mut out = serde_json::Map::new();
    for model in models {
        let these: Vec<PeerVerdict> = verdicts
            .iter()
            .filter(|v| v.model_hash.as_deref() == model)
            .cloned()
            .collect();
        out.insert(model.unwrap_or("null").to_string(), counts_json(&these));
    }
    Value::Object(out)
}

/// The five counts of the verdicts a node's references hold about it
/// (`references_tally` on the page). This plugin does not gather the
/// references' records yet; the rule is here, and judged by the parity
/// corpus, for when it does.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReferenceTally {
    pub corroborated: usize,
    pub contradicted: usize,
    pub inconclusive: usize,
    pub not_comparable: usize,
    pub ack_refusals: usize,
}

#[cfg_attr(not(test), allow(dead_code))]
impl ReferenceTally {
    pub fn to_json(&self) -> Value {
        json!({
            "corroborated": self.corroborated,
            "contradicted": self.contradicted,
            "inconclusive": self.inconclusive,
            "not_comparable": self.not_comparable,
            "ack_refusals": self.ack_refusals,
        })
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn dicts_under<'a>(value: &'a Value, key: &str, out: &mut Vec<&'a serde_json::Map<String, Value>>) {
    match value {
        Value::Object(map) => {
            for (k, v) in map {
                if k == key {
                    if let Value::Object(found) = v {
                        out.push(found);
                    }
                }
                dicts_under(v, key, out);
            }
        }
        Value::Array(items) => items.iter().for_each(|v| dicts_under(v, key, out)),
        _ => {}
    }
}

/// The verdicts about `x` among records `x`'s references hold. Only a record
/// that IS a referee's verdict, signed with the key that referee announced
/// (`peer_keys`), counts: a reference's own signature says it holds the
/// record, not that a referee signed it. Each verdict counts once, and each
/// referee once per pair. A refusal counts only when it cites a verdict
/// verified here that contradicts `x`; its own copy of the ruling is never
/// read.
#[cfg_attr(not(test), allow(dead_code))]
pub fn classify_reference_receipts(
    receipts: &[Value],
    x: &str,
    peer_keys: Option<&str>,
) -> ReferenceTally {
    let mut verified: Vec<crate::referee::hold::Facts> = Vec::new();
    let mut pairs: BTreeSet<PairKey> = BTreeSet::new();
    for receipt in receipts {
        let Some(facts) = crate::referee::hold::verdict_facts(receipt, peer_keys) else {
            continue;
        };
        let key = pair_key(&facts.referee_node_id, &facts.halves[0], &facts.halves[1]);
        if verified
            .iter()
            .any(|v| v.verdict_capsule_id == facts.verdict_capsule_id)
            || !pairs.insert(key)
        {
            continue;
        }
        verified.push(facts);
    }
    let mut tally = ReferenceTally::default();
    for facts in &verified {
        let nodes = [
            facts.half_node_ids[0].as_str(),
            facts.half_node_ids[1].as_str(),
        ];
        match ruling_about(&facts.verdict, nodes, x) {
            Some(CORROBORATED) => tally.corroborated += 1,
            Some(CONTRADICTED) => tally.contradicted += 1,
            Some(INCONCLUSIVE) => tally.inconclusive += 1,
            Some(NOT_COMPARABLE) => tally.not_comparable += 1,
            _ => {}
        }
    }
    let contradicts_x = format!("{CONTRADICTED_PREFIX}{x}");
    let mut refused: BTreeSet<&str> = BTreeSet::new();
    for receipt in receipts {
        let mut found = Vec::new();
        dicts_under(receipt, "adjudication_ack_refused", &mut found);
        for block in found {
            let Some(cited) = block.get("adjudication_capsule_id").and_then(Value::as_str) else {
                continue;
            };
            if let Some(facts) = verified.iter().find(|v| v.verdict_capsule_id == cited) {
                if facts.verdict == contradicts_x {
                    refused.insert(facts.verdict_capsule_id.as_str());
                }
            }
        }
    }
    tally.ack_refusals = refused.len();
    tally
}

/// The ids a Pane B row can be matched on: its identity's endpoint and node
/// ids, and its label without the `node:` / `endpoint:` prefix.
fn row_peer_ids(row: &Value) -> Vec<&str> {
    let mut ids: Vec<&str> = ["endpoint_id", "node_id"]
        .into_iter()
        .filter_map(|key| {
            row.pointer(&format!("/identity/{key}"))
                .and_then(Value::as_str)
        })
        .collect();
    if let Some(label) = row.get("peer_id").and_then(Value::as_str) {
        ids.extend(
            ["node:", "endpoint:"]
                .into_iter()
                .filter_map(|p| label.strip_prefix(p)),
        );
    }
    ids.retain(|id| !id.is_empty());
    ids
}

/// Put `referee_verdicts` on every attributed Pane B row, from this node's own
/// chain (`our_records`) and the pairs it asked about (in `ledger_dir`). A
/// row whose ids name no judged peer gets four zeros: zero is a fact here,
/// since the chain was read.
pub fn attach(pane_b: &mut Value, our_records: &[Value], ledger_dir: &Path) {
    let by_peer = fold(our_records, &read_requested(ledger_dir));
    let Some(rows) = pane_b.get_mut("rows").and_then(Value::as_array_mut) else {
        return;
    };
    for row in rows {
        if row.get("peer_id").is_none_or(Value::is_null) {
            continue;
        }
        let mut verdicts: Vec<PeerVerdict> = Vec::new();
        for id in row_peer_ids(row) {
            for v in by_peer.get(id).into_iter().flatten() {
                if !verdicts
                    .iter()
                    .any(|seen| seen.verdict_capsule_id == v.verdict_capsule_id)
                {
                    verdicts.push(v.clone());
                }
            }
        }
        row["referee_verdicts"] = counts_json(&verdicts);
        row["referee_verdicts_by_model"] = counts_by_model(&verdicts);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A verdict record as this node's chain holds one: `referee` judged the
    /// halves `halves` served by `nodes`.
    fn record(
        block: &str,
        verdict: &str,
        id: &str,
        referee: &str,
        halves: [&str; 2],
        nodes: [&str; 2],
    ) -> Value {
        let at = if block == ADJUDICATION_ISSUED_BLOCK {
            "issued_at"
        } else {
            "received_at"
        };
        json!({ "model_attestation": { "compute_attestation": { block: {
            "verdict": verdict,
            "verdict_capsule_id": id,
            "referee_node_id": referee,
            "halves": halves,
            "half_node_ids": nodes,
            at: "2026-09-28T15:00:00Z",
        }}}})
    }

    fn received(verdict: &str, id: &str) -> Value {
        record(
            ADJUDICATION_RECEIVED_BLOCK,
            verdict,
            id,
            "ref",
            ["ha", "hb"],
            ["a", "b"],
        )
    }

    fn asked(pairs: &[(&str, [&str; 2])]) -> HashSet<PairKey> {
        pairs
            .iter()
            .map(|(referee, [a, b])| pair_key(referee, a, b))
            .collect()
    }

    fn count(json: &Value, bucket: &str) -> u64 {
        json[bucket]["count"].as_u64().unwrap()
    }

    #[test]
    fn a_verdict_lands_in_the_bucket_for_each_twin_it_judged() {
        assert_eq!(
            ruling_about("contradicted:b", ["a", "b"], "b"),
            Some(CONTRADICTED)
        );
        assert_eq!(
            ruling_about("contradicted:b", ["a", "b"], "a"),
            Some(CORROBORATED)
        );
        assert_eq!(
            ruling_about("corroborated", ["a", "b"], "a"),
            Some(CORROBORATED)
        );
        assert_eq!(
            ruling_about("inconclusive", ["a", "b"], "b"),
            Some(INCONCLUSIVE)
        );
        assert_eq!(
            ruling_about("not_comparable", ["a", "b"], "a"),
            Some(NOT_COMPARABLE)
        );
        assert_eq!(
            ruling_about("contradicted:b", ["a", "b"], "c"),
            None,
            "not a judged twin"
        );
        assert_eq!(
            ruling_about("contradicted:c", ["a", "b"], "a"),
            None,
            "names no twin"
        );
        assert_eq!(
            ruling_about("figure:9", ["a", "b"], "a"),
            None,
            "an unknown ruling counts nowhere"
        );
    }

    #[test]
    fn only_the_receiver_verified_record_kinds_count() {
        let bare = json!({ "model_attestation": { "compute_attestation": { "adjudication": {
            "verdict": "contradicted:b", "verdict_capsule_id": "v0", "referee_node_id": "ref",
            "halves": ["ha", "hb"], "half_node_ids": ["a", "b"],
        }}}});
        let refused = json!({ "model_attestation": { "compute_attestation": { "adjudication_ack_refused": {
            "verdict": "contradicted:b", "verdict_capsule_id": "v9", "referee_node_id": "ref",
            "halves": ["ha", "hb"], "half_node_ids": ["a", "b"],
        }}}});
        let records = vec![
            bare,
            refused,
            received("contradicted:b", "v1"),
            record(
                ADJUDICATION_ISSUED_BLOCK,
                "inconclusive",
                "v2",
                "me",
                ["hc", "hd"],
                ["b", "c"],
            ),
        ];
        let by_peer = fold(&records, &asked(&[("ref", ["ha", "hb"])]));
        let b = counts_json(&by_peer["b"]);
        assert_eq!(count(&b, CONTRADICTED), 1);
        assert_eq!(
            count(&b, INCONCLUSIVE),
            1,
            "a verdict this node issued counts without a request"
        );
        assert_eq!(b[CONTRADICTED]["verdict_capsule_ids"], json!(["v1"]));
        assert_eq!(count(&counts_json(&by_peer["a"]), CORROBORATED), 1);
    }

    /// The adversarial case: peer X, with an announced key, signs N
    /// contradictions of Y, each citing one of our record ids and a made-up
    /// one, each with its own verdict id. This node never asked X anything,
    /// so none of them count.
    #[test]
    fn verdicts_from_a_referee_this_node_did_not_ask_count_nowhere() {
        let forged: Vec<Value> = (0..5)
            .map(|i| {
                record(
                    ADJUDICATION_RECEIVED_BLOCK,
                    "contradicted:y",
                    &format!("forged-{i}"),
                    "x",
                    ["ours", &format!("made-up-{i}")],
                    ["z", "y"],
                )
            })
            .collect();
        assert!(fold(&forged, &HashSet::new()).is_empty());
        // Asking X about one pair counts that pair only, never the others.
        let by_peer = fold(&forged, &asked(&[("x", ["made-up-0", "ours"])]));
        assert_eq!(by_peer["y"].len(), 1);
        // Asking a different referee about the same pair does not let X in.
        assert!(fold(&forged, &asked(&[("ref", ["ours", "made-up-0"])])).is_empty());
    }

    #[test]
    fn one_referee_counts_once_per_pair_whatever_it_signs() {
        let records = vec![
            received("contradicted:b", "v1"),
            received("contradicted:b", "v2"),
            received("contradicted:b", "v1"),
            record(
                ADJUDICATION_RECEIVED_BLOCK,
                "contradicted:b",
                "v3",
                "ref",
                ["hb", "ha"],
                ["a", "b"],
            ),
        ];
        let by_peer = fold(&records, &asked(&[("ref", ["ha", "hb"])]));
        assert_eq!(by_peer["b"].len(), 1);
        assert_eq!(
            by_peer["b"][0].verdict_capsule_id, "v1",
            "the first recorded verdict stands"
        );
    }

    #[test]
    fn a_referee_that_is_one_of_the_twins_counts_nowhere() {
        let records = vec![record(
            ADJUDICATION_RECEIVED_BLOCK,
            "contradicted:b",
            "v1",
            "a",
            ["ha", "hb"],
            ["a", "b"],
        )];
        assert!(fold(&records, &asked(&[("a", ["ha", "hb"])])).is_empty());
    }

    #[test]
    fn a_malformed_block_counts_nowhere() {
        let requested = asked(&[("ref", ["ha", "hb"])]);
        let records = vec![
            record(
                ADJUDICATION_RECEIVED_BLOCK,
                "contradicted:b",
                "v1",
                "ref",
                ["ha", "hb"],
                ["b", "b"],
            ),
            record(
                ADJUDICATION_RECEIVED_BLOCK,
                "contradicted:b",
                "v2",
                "ref",
                ["ha", "ha"],
                ["a", "b"],
            ),
            json!({ "model_attestation": { "compute_attestation": { "adjudication_received": {
                "verdict": "contradicted:b", "verdict_capsule_id": "v3", "referee_node_id": "ref",
                "halves": ["ha", "hb"], "half_node_ids": ["a"],
            }}}}),
            json!({ "model_attestation": { "compute_attestation": { "adjudication_received": {
                "verdict": "contradicted:b", "verdict_capsule_id": "v4",
                "halves": ["ha", "hb"], "half_node_ids": ["a", "b"],
            }}}}),
        ];
        assert!(fold(&records, &requested).is_empty());
    }

    #[test]
    fn the_requested_file_is_read_and_a_missing_one_asks_nothing() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_requested(dir.path()).is_empty());
        std::fs::write(
            dir.path().join(REQUESTED_FILENAME),
            "{\"referee\":\"ref\",\"halves\":[\"hb\",\"ha\"],\"twin_bracket_id\":\"t\",\"asked_at\":\"x\"}\nnot json\n{\"referee\":\"ref\",\"halves\":[\"only-one\"]}\n",
        )
        .unwrap();
        assert_eq!(read_requested(dir.path()), asked(&[("ref", ["ha", "hb"])]));
    }

    #[test]
    fn the_records_this_plugin_seals_are_the_ones_counted() {
        use crate::capsule_emit::{CapsuleState, VerdictFacts};
        let dir = tempfile::tempdir().unwrap();
        let state = CapsuleState::open(dir.path(), "node-under-test").unwrap();
        let facts = |verdict, id| VerdictFacts {
            verdict,
            verdict_capsule_id: id,
            referee_node_id: "ref",
            halves: ["ha", "hb"],
            half_node_ids: ["a", "b"],
            twin_bracket_id: None,
            model_hash: None,
        };
        let (first, second) = ("1".repeat(64), "2".repeat(64));
        state
            .emit_adjudication_received(
                &facts("contradicted:b", &first),
                "ha",
                "a",
                "2026-09-28T15:01:00Z",
            )
            .unwrap();
        state
            .emit_adjudication_issued(
                &facts("not_comparable", &second),
                None,
                "2026-09-28T15:02:00Z",
            )
            .unwrap();
        let records = crate::evidence_panes::read_capsule_records(state.ledger_dir());
        // Same referee, same pair: the received one was first and stands.
        let b = &fold(&records, &asked(&[("ref", ["ha", "hb"])]))["b"];
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].bucket, CONTRADICTED);
        assert_eq!(b[0].recorded_at.as_deref(), Some("2026-09-28T15:01:00Z"));
        // Not asked: only this node's own issued verdict counts.
        let b = &fold(&records, &HashSet::new())["b"];
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].bucket, NOT_COMPARABLE);
    }

    #[test]
    fn rows_get_all_four_counts_matched_on_any_of_their_ids() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(REQUESTED_FILENAME),
            "{\"referee\":\"ref\",\"halves\":[\"ha\",\"hb\"]}\n",
        )
        .unwrap();
        let mut pane = json!({ "rows": [
            { "peer_id": "endpoint:b", "identity": { "endpoint_id": "b", "node_id": null } },
            { "peer_id": "key:k", "identity": { "endpoint_id": null, "node_id": "a" } },
            { "peer_id": "node:z", "identity": null },
            { "peer_id": null },
        ]});
        attach(&mut pane, &[received("contradicted:b", "v1")], dir.path());
        let rows = pane["rows"].as_array().unwrap();
        assert_eq!(count(&rows[0]["referee_verdicts"], CONTRADICTED), 1);
        assert_eq!(count(&rows[1]["referee_verdicts"], CORROBORATED), 1);
        for bucket in BUCKETS {
            assert_eq!(
                count(&rows[2]["referee_verdicts"], bucket),
                0,
                "zero is shown, not hidden"
            );
        }
        assert!(
            rows[3].get("referee_verdicts").is_none(),
            "the unattributed residual names no peer"
        );
    }
}
