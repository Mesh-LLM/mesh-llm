//! A referee's verdict, delivered to a node it concerns.
//!
//! Over record-push the body is `{"adjudication_delivery": 1,
//! "verdict_capsule": <the signed verdict>}`; the node that delivers it is
//! only the courier. What this door checks, in order:
//!
//! 1. The verdict verifies, and the referee it names signed it with that
//!    node's announced key ([`verdict_facts`]).
//! 2. The referee is neither twin, and the ruling is `corroborated`,
//!    `inconclusive`, `not_comparable`, or a contradiction naming one of the
//!    twins.
//! 3. Every cited half this node holds is the record of the node the verdict
//!    names for it.
//! 4. The verdict concerns this node: it asked that referee about exactly
//!    that pair and holds both halves (the requester), or one half is its
//!    own served record (a judged provider). A verdict from a referee nobody
//!    here asked, about records this node merely holds, is refused.
//! 5. A requester holds a verdict only when it seals the tier this node
//!    asked at (`tier_not_as_asked`).
//! 6. At most one verdict per referee and pair (requester), or per referee
//!    and own record (provider).
//!
//! The verdict is then held beside the ledger (`received-adjudications.jsonl`),
//! never written into `capsules.jsonl`; the bridge seals the
//! `adjudication_received` record that cites it. A repeat delivery of the
//! same verdict is answered from what is held.

use std::collections::HashMap;
use std::path::Path;

use serde_json::{json, Map, Value};

use super::half::{record_checks, served_by, signer};
use super::verdict::{
    VERDICT_CONTRADICTED_PREFIX, VERDICT_CORROBORATED, VERDICT_INCONCLUSIVE, VERDICT_NOT_COMPARABLE,
};
use crate::peer_keys::{announced_key_in, peer_id_for_key_in};

pub const DELIVERY_MARKER: &str = crate::adjudication_records::DELIVERY_MARKER;
pub const DELIVERY_VERSION: u64 = 1;
pub const RECEIVED_ADJUDICATIONS_FILENAME: &str =
    crate::adjudication_records::RECEIVED_ADJUDICATIONS_FILENAME;
pub const REQUESTED_ADJUDICATIONS_FILENAME: &str = crate::verdict_counts::REQUESTED_FILENAME;

pub const REASON_REQUEST_MALFORMED: &str = "request_malformed";
pub const REASON_VERDICT_UNVERIFIED: &str = "verdict_unverified";
pub const REASON_NOT_ABOUT_THIS_NODE: &str = "not_about_this_node";
pub const REASON_HALVES_MISATTRIBUTED: &str = "halves_misattributed";
pub const REASON_VERDICT_ALREADY_HELD: &str = "verdict_already_held";
pub const REASON_TIER_NOT_AS_ASKED: &str = "tier_not_as_asked";
pub const REASON_POLICY_DECLINE: &str = "policy_decline";

/// The facts of a verdict the referee it names signed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Facts {
    pub verdict: String,
    pub verdict_capsule_id: String,
    pub referee_node_id: String,
    pub halves: [String; 2],
    pub half_node_ids: [String; 2],
    pub twin_bracket_id: Option<String>,
    /// The tier the referee sealed (`selection_tier`), when it sealed one.
    pub selection_tier: Option<u64>,
    /// The model the twins served (`model_hash`), when the verdict seals it.
    pub model_hash: Option<String>,
}

impl Facts {
    /// The facts as a held line and a reply carry them.
    fn to_map(&self) -> Map<String, Value> {
        let mut map = Map::new();
        map.insert("verdict".into(), json!(self.verdict));
        map.insert("verdict_capsule_id".into(), json!(self.verdict_capsule_id));
        map.insert("referee_node_id".into(), json!(self.referee_node_id));
        map.insert("halves".into(), json!(self.halves));
        map.insert("half_node_ids".into(), json!(self.half_node_ids));
        map.insert("twin_bracket_id".into(), json!(self.twin_bracket_id));
        map
    }
}

fn block(capsule: &Value) -> Option<&Map<String, Value>> {
    capsule
        .pointer("/model_attestation/compute_attestation/adjudication")?
        .as_object()
}

fn text(block: &Map<String, Value>, key: &str) -> Option<String> {
    block
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// The facts of `capsule` when it verifies as a verdict signed, with the key
/// it announced (`peer_keys`: the raw announced-key registry), by the referee
/// it names; `None` otherwise.
pub fn verdict_facts(capsule: &Value, peer_keys: Option<&str>) -> Option<Facts> {
    let id = capsule
        .get("capsule_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())?;
    let block = block(capsule)?;
    let referee = text(block, "referee_node_id")?;
    let verdict = text(block, "verdict")?;
    let halves = [
        text(block, "half_a_capsule_id")?,
        text(block, "half_b_capsule_id")?,
    ];
    let nodes = [
        text(block, "half_a_node_id")?,
        text(block, "half_b_node_id")?,
    ];
    if nodes.contains(&referee) || nodes[0] == nodes[1] {
        return None;
    }
    let ruling = [
        VERDICT_CORROBORATED,
        VERDICT_INCONCLUSIVE,
        VERDICT_NOT_COMPARABLE,
    ]
    .contains(&verdict.as_str())
        || nodes
            .iter()
            .any(|n| verdict == format!("{VERDICT_CONTRADICTED_PREFIX}{n}"));
    if !ruling || !record_checks(capsule) {
        return None;
    }
    let signed_by = signer(capsule)?;
    if capsule.get("key_id").and_then(Value::as_str) != Some(signed_by.as_str())
        || announced_key_in(peer_keys, &referee).as_deref() != Some(signed_by.as_str())
    {
        return None;
    }
    Some(Facts {
        verdict,
        verdict_capsule_id: id.to_string(),
        referee_node_id: referee,
        halves,
        half_node_ids: nodes,
        twin_bracket_id: text(block, "twin_bracket_id"),
        selection_tier: block.get("selection_tier").and_then(Value::as_u64),
        model_hash: text(block, "model_hash"),
    })
}

fn json_lines(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .map(|text| {
            text.lines()
                .filter_map(|line| serde_json::from_str(line).ok())
                .collect()
        })
        .unwrap_or_default()
}

/// `(own records, halves pushed to this node)`, each by capsule id.
fn held_records(ledger_dir: &Path) -> (HashMap<String, Value>, HashMap<String, Value>) {
    let by_id = |records: Vec<Value>| {
        records
            .into_iter()
            .filter_map(|r| Some((r.get("capsule_id")?.as_str()?.to_string(), r)))
            .collect()
    };
    (
        by_id(crate::evidence_panes::read_capsule_records(ledger_dir)),
        by_id(json_lines(
            &ledger_dir.join(crate::record_push_receive::RECEIVED_CAPSULES_FILENAME),
        )),
    )
}

fn sorted_pair(halves: &[String; 2]) -> [String; 2] {
    let mut pair = halves.clone();
    pair.sort();
    pair
}

/// The adjudicate request this node sent to `referee` about exactly this
/// pair, when it sent one.
pub fn asked(ledger_dir: &Path, referee: &str, halves: &[String; 2]) -> Option<Value> {
    json_lines(&ledger_dir.join(REQUESTED_ADJUDICATIONS_FILENAME))
        .into_iter()
        .find(|line| {
            line.get("referee").and_then(Value::as_str) == Some(referee)
                && line
                    .get("halves")
                    .and_then(Value::as_array)
                    .and_then(|h| match h.as_slice() {
                        [a, b] => Some([a.as_str()?.to_string(), b.as_str()?.to_string()]),
                        _ => None,
                    })
                    .is_some_and(|h| sorted_pair(&h) == sorted_pair(halves))
        })
}

fn received_reply(held: &Value) -> Value {
    let mut adjudication = held.as_object().cloned().unwrap_or_default();
    adjudication.remove("verdict_capsule");
    adjudication.remove("hold_key");
    json!({"status": "received", "adjudication": adjudication})
}

/// This node, as the door sees it.
pub struct Node<'a> {
    pub ledger_dir: &'a Path,
    /// The raw announced-key registry.
    pub peer_keys: Option<&'a str>,
    /// This node's own key id (hex): its own id is the one peer announcing it.
    pub own_key_id: &'a str,
}

/// Hold a delivered verdict (the body of a record-push carrying
/// [`DELIVERY_MARKER`]) from courier `sender` at `now`. `Ok(reply)` when it is
/// held (or already was), `Err(reason)` for the caller to refuse with.
pub fn hold_delivered_verdict(
    node: &Node<'_>,
    body: &Value,
    sender: &str,
    now: &str,
) -> std::io::Result<Result<Value, &'static str>> {
    let exact = body.as_object().is_some_and(|o| {
        o.len() == 2 && o.contains_key("verdict_capsule") && o.contains_key(DELIVERY_MARKER)
    });
    if body.get(DELIVERY_MARKER).and_then(Value::as_u64) != Some(DELIVERY_VERSION)
        || body.get(DELIVERY_MARKER).is_some_and(Value::is_f64)
        || !exact
    {
        return Ok(Err(REASON_REQUEST_MALFORMED));
    }
    let capsule = &body["verdict_capsule"];
    let Some(facts) = verdict_facts(capsule, node.peer_keys) else {
        return Ok(Err(REASON_VERDICT_UNVERIFIED));
    };

    let path = node.ledger_dir.join(RECEIVED_ADJUDICATIONS_FILENAME);
    let already = json_lines(&path);
    if let Some(held) = already.iter().find(|held| {
        held.get("verdict_capsule_id").and_then(Value::as_str) == Some(&facts.verdict_capsule_id)
    }) {
        return Ok(Ok(received_reply(held)));
    }

    let (own, pushed) = held_records(node.ledger_dir);
    for (half, named) in facts.halves.iter().zip(&facts.half_node_ids) {
        if let Some(record) = own.get(half).or_else(|| pushed.get(half)) {
            if served_by(record) != Some(named.as_str()) {
                return Ok(Err(REASON_HALVES_MISATTRIBUTED));
            }
        }
    }

    let referee = facts.referee_node_id.clone();
    let asked_line = if cfg!(feature = "mutant-referee-hold-skips-asked-check") {
        Some(json!({}))
    } else {
        asked(node.ledger_dir, &referee, &facts.halves)
    };
    let (held_half, key) = if let Some(asked_line) = asked_line {
        // The requester: it chose this referee for this pair, and holds both.
        if !facts
            .halves
            .iter()
            .all(|h| own.contains_key(h) || pushed.contains_key(h))
        {
            return Ok(Err(REASON_NOT_ABOUT_THIS_NODE));
        }
        let asked_tier = asked_line.get("selection_tier").and_then(Value::as_u64);
        if !cfg!(feature = "mutant-referee-hold-ignores-the-asked-tier")
            && asked_tier.is_some()
            && asked_tier != facts.selection_tier
        {
            return Ok(Err(REASON_TIER_NOT_AS_ASKED));
        }
        (
            facts.halves[0].clone(),
            json!(["pair", referee, sorted_pair(&facts.halves)]),
        )
    } else {
        // A judged provider: the verdict must be about its own served record.
        let self_id = peer_id_for_key_in(node.peer_keys, node.own_key_id);
        let mine = facts.halves.iter().find(|h| {
            self_id.is_some()
                && own
                    .get(*h)
                    .is_some_and(|r| served_by(r) == self_id.as_deref())
        });
        let Some(mine) = mine else {
            return Ok(Err(REASON_NOT_ABOUT_THIS_NODE));
        };
        (mine.clone(), json!(["own", referee, mine]))
    };
    if already
        .iter()
        .any(|held| held.get("hold_key") == Some(&key))
    {
        return Ok(Err(REASON_VERDICT_ALREADY_HELD));
    }

    let mut held = facts.to_map();
    held.insert("held_half_capsule_id".into(), json!(held_half));
    held.insert("hold_key".into(), key);
    held.insert("received_from".into(), json!(sender));
    held.insert("received_at".into(), json!(now));
    held.insert("verdict_capsule".into(), capsule.clone());
    let held = Value::Object(held);
    append(node.ledger_dir, RECEIVED_ADJUDICATIONS_FILENAME, &held)?;
    Ok(Ok(received_reply(&held)))
}

/// The rule for a verdict delivered at `/evidence/deliver` (the body is the
/// verdict record itself), judged by the parity corpus. This plugin carries
/// verdicts only over record-push ([`hold_delivered_verdict`]), so no
/// transport reaches this door yet; one added later gets this rule.
///
/// A verdict delivered at `/evidence/deliver`: the body is the verdict record
/// itself. It is held when it cites one of this node's own records and the
/// referee it names signed it; a node the verdict contradicts may decline it
/// (`policy_decline`). `Ok(reply)` or `Err(reason)`.
#[cfg_attr(not(test), allow(dead_code))]
pub fn handle_delivery(
    node: &Node<'_>,
    body: &[u8],
    now: &str,
) -> std::io::Result<Result<Value, &'static str>> {
    let Ok(capsule) = crate::strict_json::parse(body) else {
        return Ok(Err(REASON_REQUEST_MALFORMED));
    };
    if capsule
        .get("capsule_id")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
        || !record_checks(&capsule)
    {
        return Ok(Err(REASON_REQUEST_MALFORMED));
    }
    let cited: Vec<&str> = block(&capsule)
        .map(|b| {
            b.iter()
                .filter(|(k, _)| k.ends_with("_capsule_id"))
                .filter_map(|(_, v)| v.as_str().filter(|s| !s.is_empty()))
                .collect()
        })
        .unwrap_or_default();
    let (own, _) = held_records(node.ledger_dir);
    let Some(own_half) = cited.iter().find_map(|id| own.get(*id)) else {
        return Ok(Err(REASON_REQUEST_MALFORMED));
    };
    // Only the referee signs: a ruling no referee signed with its announced
    // key is never held, whoever sealed it.
    let Some(facts) = verdict_facts(&capsule, node.peer_keys) else {
        return Ok(Err(REASON_VERDICT_UNVERIFIED));
    };
    let owner = own_half
        .pointer("/model_attestation/compute_attestation/owner/owner_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty());
    if owner.is_some_and(|o| facts.verdict == format!("{VERDICT_CONTRADICTED_PREFIX}{o}")) {
        return Ok(Err(REASON_POLICY_DECLINE));
    }
    let own_half_id = own_half["capsule_id"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    let path = node.ledger_dir.join(RECEIVED_ADJUDICATIONS_FILENAME);
    let seen = json_lines(&path).into_iter().any(|held| {
        held.get("verdict_capsule_id").and_then(Value::as_str) == Some(&facts.verdict_capsule_id)
    });
    if !seen {
        let mut held = facts.to_map();
        held.insert("held_half_capsule_id".into(), json!(own_half_id));
        held.insert(
            "hold_key".into(),
            json!(["own", facts.referee_node_id, own_half_id]),
        );
        held.insert("received_from".into(), Value::Null);
        held.insert("received_at".into(), json!(now));
        held.insert("verdict_capsule".into(), capsule.clone());
        append(
            node.ledger_dir,
            RECEIVED_ADJUDICATIONS_FILENAME,
            &Value::Object(held),
        )?;
    }
    Ok(Ok(json!({"status": "received"})))
}

/// The model hash a held verdict seals, read from the held record itself.
pub fn held_model_hash(ledger_dir: &Path, verdict_capsule_id: &str) -> Option<String> {
    json_lines(&ledger_dir.join(RECEIVED_ADJUDICATIONS_FILENAME))
        .into_iter()
        .find(|held| {
            held.get("verdict_capsule_id").and_then(Value::as_str) == Some(verdict_capsule_id)
        })
        .and_then(|held| block(&held["verdict_capsule"]).and_then(|b| text(b, "model_hash")))
}

fn append(ledger_dir: &Path, filename: &str, line: &Value) -> std::io::Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(ledger_dir)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(ledger_dir.join(filename))?;
    writeln!(file, "{line}")?;
    file.sync_data()
}
