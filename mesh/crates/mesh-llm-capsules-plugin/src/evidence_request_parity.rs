//! The evidence-request parity run: the in-process responder
//! (`evidence_answer::answer_and_log`) against the corpus and golden answers
//! in `vectors/parity/evidence_request/`.
//!
//! The golden answers are a snapshot this implementation wrote, reviewed
//! case by case against draft-mih-agent-evidence-request-00. They are not the
//! draft's own conformance vectors (those run in the protocol crate's
//! repository), and not the Python reference's answers: that reference departs
//! from -00 in six ways (see that directory's README), and -00 is what counts.
//! The served summary is the one part the Python reference still judges: its
//! value in the golden answers is
//! checked against `served_summary.py` by `test_served_summary_parity.py`.
//!
//! `EVIDENCE_REQUEST_PARITY_OUT=<path>` also writes this run's answers.
//! `EVIDENCE_REQUEST_PARITY_WRITE=1` with `--ignored` regenerates the corpus
//! and the golden answers from this implementation: review every changed
//! line before committing it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use capsule_emit_evidence_request::digest::request_digest;
use capsule_emit_evidence_request::refusal;
use cll::checkpoint::{sign_checkpoint_digest, CheckpointRecord, WitnessRecord};
use cll::mmr::{add_leaf, leaf_hash, peaks, root_from_peaks, MemoryNodeStore, NodeReader};
use ed25519_dalek::SigningKey;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::evidence_answer::{self, Outcome, Responder};

fn parity_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("vectors/parity/evidence_request")
}

// ---------------------------------------------------------------------------
// The corpus.
// ---------------------------------------------------------------------------

const NOW: &str = "2026-09-29T00:00:00Z";
/// A throwaway test key, public by design: it signs only this corpus.
pub(crate) const NODE_KEY_SEED: [u8; 32] = [0x5e; 32];

/// The answering node's own mesh id, which a signed request's `audience`
/// names.
pub(crate) const RESPONDER_NODE: &str = "node-r";

/// The nodes that ask. node-a and node-b are the corpus's counterparties;
/// node-y and node-z are named by no record.
pub(crate) const ASKERS: [&str; 4] = ["node-a", "node-b", "node-y", "node-z"];

/// An asking node's throwaway test key, public by design: its seed is the
/// SHA-256 of the node's id. The corpus carries only the public half.
pub(crate) fn asker_key(node: &str) -> SigningKey {
    SigningKey::from_bytes(&Sha256::digest(node.as_bytes()).into())
}

/// The key the host announced for each asking node.
pub(crate) fn announced_keys() -> BTreeMap<String, String> {
    ASKERS
        .iter()
        .map(|node| {
            let key = asker_key(node).verifying_key().to_bytes();
            (node.to_string(), hex::encode(key))
        })
        .collect()
}

/// `request` signed by `key` at [`NOW`], with a nonce fixed by the case's
/// `name` so the corpus is the same bytes every time it is written.
fn signed_as(key: &SigningKey, name: &str, mut request: Value) -> Value {
    request["nonce"] = json!(hex::encode(Sha256::digest(name.as_bytes())));
    let now = chrono::DateTime::parse_from_rfc3339(NOW).unwrap().into();
    crate::requester_auth::sign(
        crate::requester_auth::Channel::EvidenceRequest,
        &mut request,
        key,
        now,
        RESPONDER_NODE,
    );
    request
}

/// A case whose request `node` signs.
fn case_by(name: &str, covers: &str, history_segments: &str, node: &str, request: Value) -> Value {
    case(
        name,
        covers,
        history_segments,
        signed_as(&asker_key(node), name, request),
    )
}

pub(crate) fn sealed(mut body: Value) -> Value {
    let id =
        crate::producer::jcs::compute_capsule_id(&body).expect("a corpus record has a capsule_id");
    body["capsule_id"] = json!(id);
    body
}

fn served(counterparty: &str, model: &str, status: &str, latency: &str, extra: Value) -> Value {
    let mut poc = json!({
        "role": "served",
        "latency_ms": latency,
        "serving_provenance": {"model_canonical_ref": model, "quantization": "q4_k_m", "requesting_party": counterparty},
    });
    if let (Some(p), Value::Object(e)) = (poc.as_object_mut(), extra) {
        p.extend(e);
    }
    sealed(json!({
        "action_type": "inference",
        "effect": {"status": status},
        "model_attestation": {"compute_attestation": {"x-mesh-poc-v1": poc}},
    }))
}

fn padding(i: usize) -> Value {
    sealed(json!({
        "record_type": crate::producer::padding::RECORD_TYPE_PADDING,
        "epistemic_type": crate::producer::padding::PADDING_EPISTEMIC_TYPE,
        "store_nonce": hex::encode(Sha256::digest(format!("padding-{i}"))),
    }))
}

/// The half another node holds, which leaf 6 cites.
fn foreign_half() -> String {
    hex::encode(Sha256::digest(b"the counterparty's own half"))
}

/// 301 lines: eight records of several kinds (0-7), padding to 16
/// (checkpoint 0), five more served records (16-20) and a local block (21),
/// padding to 300 (checkpoint 1), and one record no checkpoint covers yet.
pub(crate) fn corpus_ledger() -> Vec<Value> {
    let mut ledger = vec![
        served(
            "node-a",
            "model-x",
            "confirmed",
            "10.000",
            json!({"nonce": "n-1"}),
        ),
        served(
            "node-b",
            "model-x",
            "confirmed",
            "20.000",
            json!({"nonce": "n-2"}),
        ),
        served(
            "node-a",
            "model-x",
            "failed",
            "30.000",
            json!({"exchange_id": "ex-1"}),
        ),
        served("node-a", "model-y", "confirmed", "5.000", json!({})),
        served(
            "node-a",
            "model-x",
            "confirmed",
            "40.000",
            json!({"nonce": "n-1"}),
        ),
        sealed(json!({
            "action_type": "inference",
            "effect": {"status": "confirmed"},
            "model_attestation": {"compute_attestation": {"x-mesh-poc-v1": {
                "role": "requested",
                "serving_provenance": {"model_canonical_ref": "model-x", "served_by_node_id": "node-b"},
            }}},
        })),
    ];
    ledger.push(sealed(json!({
        "action_type": "fyi",
        "references": [{"type": "capsule", "digest_alg": "SHA-256", "digest": foreign_half(), "citation_purpose": "counterparty_half"}],
        "model_attestation": {"compute_attestation": {"x-mesh-poc-v1": {
            "role": "served",
            "serving_provenance": {"model_canonical_ref": "model-z", "requesting_party": "node-b"},
        }}},
    })));
    let judged = ledger[0]["capsule_id"].clone();
    ledger.push(sealed(json!({
        "action_type": "fyi",
        "chain": {"parent_capsule_id": ledger[6]["capsule_id"], "relation": "adjudicates"},
        "model_attestation": {"compute_attestation": {"adjudication": {"half_a_capsule_id": judged, "half_b_capsule_id": "f".repeat(64)}}},
    })));
    while ledger.len() < 16 {
        ledger.push(padding(ledger.len()));
    }
    for latency in ["50.000", "60.000", "70.000", "80.000", "90.000"] {
        ledger.push(served("node-a", "model-x", "confirmed", latency, json!({})));
    }
    ledger.push(sealed(json!({
        "action_type": "fyi",
        "model_attestation": {"compute_attestation": {"local_routing_choice": {"peer_commitment": "c".repeat(64), "choice": "stop"}}},
    })));
    while ledger.len() < 300 {
        ledger.push(padding(ledger.len()));
    }
    ledger.push(served(
        "node-a",
        "model-x",
        "confirmed",
        "99.000",
        json!({}),
    ));
    ledger
}

/// Checkpoints after 16 and after 300 lines, signed by the node key; the
/// second carries one witness.
pub(crate) fn corpus_checkpoints(ledger: &[Value], key: &SigningKey) -> Vec<CheckpointRecord> {
    let mut nodes = MemoryNodeStore::new();
    let mut out: Vec<CheckpointRecord> = Vec::new();
    for (i, record) in ledger.iter().enumerate() {
        let id: [u8; 32] = hex::decode(record["capsule_id"].as_str().unwrap())
            .unwrap()
            .try_into()
            .unwrap();
        add_leaf(&mut nodes, leaf_hash(&id)).unwrap();
        if i + 1 == 16 || i + 1 == 300 {
            let size = nodes.size();
            let peak_hashes: Vec<_> = peaks(size)
                .unwrap()
                .iter()
                .map(|&p| nodes.node(p))
                .collect();
            let prev = out.last();
            let mut cp = CheckpointRecord {
                v: 1,
                kind: "mmr_checkpoint".into(),
                log_id: "corpus-node-log".into(),
                mmr_size: size,
                root: hex::encode(root_from_peaks(&peak_hashes)),
                prev_size: prev.map_or(0, |p| p.mmr_size),
                prev_root: prev.map_or(String::new(), |p| p.root.clone()),
                key_id: hex::encode(key.verifying_key().to_bytes()),
                timestamp: if out.is_empty() {
                    "2026-09-28T00:00:00Z"
                } else {
                    "2026-09-28T12:00:00Z"
                }
                .into(),
                signature: String::new(),
                witnesses: Vec::new(),
            };
            cp.signature = sign_checkpoint_digest(&cp, key);
            if !out.is_empty() {
                cp.witnesses.push(WitnessRecord {
                    ts_url: "https://witness.example".into(),
                    entry_hash: hex::encode(Sha256::digest(cp.digest().as_bytes())),
                    receipt_b64: String::new(),
                    leaf_index: 0,
                    tree_size: 1,
                    is_stub: true,
                });
            }
            out.push(cp);
        }
    }
    out
}

fn case(name: &str, covers: &str, history_segments: &str, request: Value) -> Value {
    json!({"name": name, "covers": covers, "history_segments": history_segments, "request": request})
}

/// `case` on a node that shares Standard History with any requester: the
/// cases about what an answer is, not about who gets one.
fn shared(name: &str, covers: &str, history_segments: &str, request: Value) -> Value {
    let mut c = case(name, covers, history_segments, request);
    c["share_standard_history"] = json!("on");
    c
}

fn corpus_cases(ledger: &[Value], cps: &[CheckpointRecord]) -> Vec<Value> {
    let id = |i: usize| ledger[i]["capsule_id"].as_str().unwrap().to_string();
    let latest = json!({"expected_pin": cps[1].digest()});
    let first = json!({"expected_pin": cps[0].digest()});
    let fresh = json!({"min_freshness": 1});
    let with = |mut r: Value, k: &str, v: Value| {
        r[k] = v;
        r
    };
    let req = |subject: Value, coverage: &Value| json!({"subject": subject, "coverage": coverage});
    let mut cases = vec![
        // Records, and who may have them.
        case_by(
            "record_to_its_counterparty",
            "a record goes to the node it names as the other side",
            "counterparties",
            "node-a",
            with(req(json!({"record": id(0)}), &latest), "requester_id", json!("node-a")),
        ),
        case_by(
            "record_to_another_node",
            "a record does not go to a node it does not name",
            "counterparties",
            "node-b",
            with(req(json!({"record": id(0)}), &latest), "requester_id", json!("node-b")),
        ),
        case_by(
            "record_to_a_stranger",
            "a proven node no record names is nobody's counterparty",
            "counterparties",
            "node-z",
            req(json!({"record": id(0)}), &latest),
        ),
        case_by(
            "record_unknown_to_a_stranger",
            "a stranger is refused before the subject is resolved: it learns nothing of what this node holds",
            "counterparties",
            "node-z",
            req(json!({"record": "a".repeat(64)}), &latest),
        ),
        case_by(
            "range_over_limit_to_a_stranger",
            "a stranger is refused before the size is counted: it learns nothing of this log's size",
            "counterparties",
            "node-z",
            req(json!({"range": [0, 299]}), &latest),
        ),
        case(
            "record_unsigned_naming_its_counterparty",
            "an unsigned request is nobody's, whatever id it states",
            "counterparties",
            with(
                req(json!({"record": id(0)}), &latest),
                "requester_id",
                json!("node-a"),
            ),
        ),
        case(
            "record_unsigned_asking_nothing_held",
            "an unsigned request is refused before the subject is resolved",
            "counterparties",
            req(json!({"record": "a".repeat(64)}), &latest),
        ),
        case_by(
            "record_signed_by_one_claiming_another",
            "a node stating another node's id is refused before lookup",
            "counterparties",
            "node-y",
            with(
                req(json!({"record": id(0)}), &latest),
                "requester_id",
                json!("node-a"),
            ),
        ),
        case(
            "record_signed_under_another_nodes_key_id",
            "a signature by one node under another node's key id does not verify",
            "counterparties",
            {
                let mut r = signed_as(
                    &asker_key("node-y"),
                    "record_signed_under_another_nodes_key_id",
                    req(json!({"record": id(0)}), &latest),
                );
                r["requester_auth"]["key_id"] = json!(announced_keys()["node-a"]);
                r
            },
        ),
        case(
            "record_signature_over_another_request",
            "a member changed after signing: the signature no longer covers the request",
            "counterparties",
            {
                let mut r = signed_as(
                    &asker_key("node-a"),
                    "record_signature_over_another_request",
                    req(json!({"record": id(0)}), &latest),
                );
                r["coverage"] = json!({"expected_pin": cps[0].digest()});
                r
            },
        ),
        case(
            "record_signed_by_an_unannounced_key",
            "a valid signature under a key no host binds reads as unsigned where no identity is needed",
            "peers",
            signed_as(
                &SigningKey::from_bytes(&[0x77; 32]),
                "record_signed_by_an_unannounced_key",
                req(json!({"record": id(0)}), &latest),
            ),
        ),
        case(
            "record_signed_by_an_unannounced_key_needing_identity",
            "a key no host binds proves no node, where the answer needs one",
            "counterparties",
            signed_as(
                &SigningKey::from_bytes(&[0x77; 32]),
                "record_signed_by_an_unannounced_key_needing_identity",
                req(json!({"record": id(0)}), &latest),
            ),
        ),
        case(
            "record_signed_without_an_audience",
            "a signed request must name the node it is addressed to",
            "peers",
            {
                let mut r = signed_as(
                    &asker_key("node-a"),
                    "record_signed_without_an_audience",
                    req(json!({"record": id(0)}), &latest),
                );
                r.as_object_mut().unwrap().remove(crate::requester_auth::AUDIENCE);
                r
            },
        ),
        case(
            "record_relayed_from_another_audience",
            "a request signed for another node is refused before lookup, even under peers",
            "peers",
            {
                let mut r = req(json!({"record": id(0)}), &latest);
                r["nonce"] = json!(hex::encode(Sha256::digest(b"record_relayed_from_another_audience")));
                let now = chrono::DateTime::parse_from_rfc3339(NOW).unwrap().into();
                crate::requester_auth::sign(crate::requester_auth::Channel::EvidenceRequest, &mut r, &asker_key("node-a"), now, "node-x");
                r
            },
        ),
        case(
            "requester_auth_short_key",
            "requester_auth with a 63-hex key is malformed",
            "peers",
            {
                let mut r = signed_as(
                    &asker_key("node-a"),
                    "requester_auth_short_key",
                    req(json!({"record": id(0)}), &latest),
                );
                let short = r["requester_auth"]["key_id"].as_str().unwrap()[1..].to_string();
                r["requester_auth"]["key_id"] = json!(short);
                r
            },
        ),
        case(
            "record_under_peers",
            "under peers a record goes to anyone",
            "peers",
            req(json!({"record": id(0)}), &latest),
        ),
        case_by(
            "record_sharing_off",
            "off serves no record, even to its counterparty",
            "off",
            "node-a",
            with(req(json!({"record": id(0)}), &latest), "requester_id", json!("node-a")),
        ),
        case_by(
            "record_counterparties_tier",
            "counterparties serves a record to its counterparty",
            "counterparties",
            "node-b",
            with(req(json!({"record": id(1)}), &first), "requester_id", json!("node-b")),
        ),
        case(
            "record_local_block_never_leaves",
            "a local block is never served, even under peers",
            "peers",
            req(json!({"record": id(21)}), &latest),
        ),
        case(
            "record_unknown",
            "a record this node does not hold",
            "peers",
            req(json!({"record": "a".repeat(64)}), &latest),
        ),
        case(
            "record_padding",
            "padding is a leaf, never a record",
            "peers",
            req(json!({"record": id(9)}), &latest),
        ),
        case(
            "record_prefix",
            "a record id must be the whole digest (no prefix match)",
            "peers",
            req(json!({"record": id(0)[..16]}), &latest),
        ),
        case(
            "record_not_yet_checkpointed",
            "a record after the newest checkpoint is not committed under any anchor",
            "peers",
            req(json!({"record": id(300)}), &fresh),
        ),
        case(
            "record_under_first_anchor",
            "a pinned older anchor serves what it covers",
            "peers",
            req(json!({"record": id(3)}), &first),
        ),
        // Ranges and the full history.
        case(
            "range_under_peers",
            "a range carries every leaf, padding included, with a range proof",
            "peers",
            req(json!({"range": [5, 12]}), &first),
        ),
        case_by(
            "range_to_a_counterparty",
            "a range naming only one counterparty's records goes to it",
            "counterparties",
            "node-a",
            with(req(json!({"range": [16, 20]}), &latest), "requester_id", json!("node-a")),
        ),
        case_by(
            "range_mixed_counterparties",
            "a range with another node's record is refused whole, never filtered",
            "counterparties",
            "node-a",
            with(req(json!({"range": [0, 4]}), &latest), "requester_id", json!("node-a")),
        ),
        case(
            "range_with_a_local_block",
            "a range holding a local block is refused whole, even under peers",
            "peers",
            req(json!({"range": [20, 22]}), &latest),
        ),
        case(
            "range_past_anchor",
            "a range ending past the anchor",
            "peers",
            req(json!({"range": [10, 16]}), &first),
        ),
        case(
            "range_over_limit",
            "more than 256 records is declined: ask for a smaller range",
            "peers",
            req(json!({"range": [0, 299]}), &latest),
        ),
        case(
            "full_history_over_limit",
            "the full history of 300 records is over the limit",
            "peers",
            req(json!({"full_history": null}), &latest),
        ),
        case(
            "full_history_under_first_anchor",
            "the full history under a pinned anchor",
            "peers",
            req(json!({"full_history": null}), &first),
        ),
        // Correlation and exchange.
        case_by(
            "correlation_to_its_counterparty",
            "records carrying one nonce, all naming the asker",
            "counterparties",
            "node-a",
            with(req(json!({"correlation": "n-1"}), &latest), "requester_id", json!("node-a")),
        ),
        case(
            "correlation_by_exchange_id",
            "an exchange_id is a correlation identifier",
            "peers",
            req(json!({"correlation": "ex-1"}), &latest),
        ),
        case(
            "correlation_unknown",
            "no record carries this identifier",
            "peers",
            req(json!({"correlation": "n-9"}), &latest),
        ),
        case(
            "exchange_citing_the_half",
            "the record citing the asker's half, proven under the earliest checkpoint covering it",
            "peers",
            req(
                json!({"exchange": foreign_half()}),
                &json!({"expected_pin": foreign_half()}),
            ),
        ),
        case(
            "exchange_nothing_cites",
            "no record cites this half",
            "peers",
            req(json!({"exchange": "b".repeat(64)}), &latest),
        ),
        // Checkpoints and derivations: no record bodies, so never gated.
        shared(
            "checkpoints_with_sharing_off",
            "the checkpoint list carries no records and is served under off",
            "off",
            req(json!({"checkpoints": null}), &latest),
        ),
        shared(
            "history_card",
            "the history_card/1 derivation, with consistency proofs",
            "off",
            with(
                req(json!({"full_history": null}), &latest),
                "derivation",
                json!("history_card/1"),
            ),
        ),
        shared(
            "history_card_wrong_subject",
            "history_card/1 is not derived for a record",
            "peers",
            with(
                req(json!({"record": id(0)}), &latest),
                "derivation",
                json!("history_card/1"),
            ),
        ),
        shared(
            "derivation_unknown",
            "a derivation this node does not derive",
            "peers",
            with(
                req(json!({"full_history": null}), &latest),
                "derivation",
                json!("unregistered_fold/1"),
            ),
        ),
        shared(
            "served_summary",
            "served_summary/1 pinned to the newest checkpoint",
            "off",
            with(
                req(json!({"full_history": null}), &latest),
                "derivation",
                json!("served_summary/1"),
            ),
        ),
        shared(
            "served_summary_older_pin",
            "served_summary/1 is a static export of the newest checkpoint only",
            "peers",
            with(
                req(json!({"full_history": null}), &first),
                "derivation",
                json!("served_summary/1"),
            ),
        ),
        shared(
            "served_summary_on_demand",
            "served_summary/1 under min_freshness asks for an on-demand export",
            "peers",
            with(
                req(json!({"full_history": null}), &fresh),
                "derivation",
                json!("served_summary/1"),
            ),
        ),
        // Coverage.
        shared(
            "coverage_unknown_pin",
            "a pin that is none of this node's checkpoints",
            "peers",
            req(
                json!({"checkpoints": null}),
                &json!({"expected_pin": "d".repeat(64)}),
            ),
        ),
        shared(
            "coverage_both_members",
            "coverage carries exactly one member",
            "peers",
            req(
                json!({"checkpoints": null}),
                &json!({"expected_pin": cps[1].digest(), "min_freshness": 1}),
            ),
        ),
        shared(
            "coverage_missing",
            "coverage is required",
            "peers",
            json!({"subject": {"checkpoints": null}}),
        ),
        shared(
            "coverage_fresher_than_held",
            "min_freshness beyond every checkpoint",
            "peers",
            req(
                json!({"checkpoints": null}),
                &json!({"min_freshness": 1000}),
            ),
        ),
        shared(
            "coverage_by_time",
            "min_freshness as a time picks a checkpoint issued since",
            "peers",
            req(
                json!({"checkpoints": null}),
                &json!({"min_freshness": "2026-09-28T06:00:00Z"}),
            ),
        ),
        // Malformed, late, large.
        shared(
            "pre_draft_request_shape",
            "a request in the shape used before -00 (a kind and capsule_id subject) is not a -00 request",
            "peers",
            json!({"subject": {"kind": "record", "capsule_id": id(0)}}),
        ),
        shared(
            "deadline_passed",
            "a deadline already past at receipt",
            "peers",
            with(
                req(json!({"checkpoints": null}), &latest),
                "deadline",
                json!("2026-09-28T00:00:00Z"),
            ),
        ),
    ];
    cases.extend(standard_history_cases(&latest, &id(0)));
    cases.push(
        json!({"name": "not_json", "covers": "bytes that are not JSON", "history_segments": "peers",
                      "request_text": "{\"subject\": "}),
    );
    cases.push(json!({"name": "request_too_large", "covers": "a request over 16 KiB is declined before it is parsed",
                      "history_segments": "peers",
                      "request": with(req(json!({"checkpoints": null}), &latest), "pad", json!("x".repeat(17 * 1024)))}));
    cases.push(
        json!({"name": "busy", "covers": "a request past the in-flight limit is declined unread",
                      "history_segments": "peers", "busy": true,
                      "request": req(json!({"checkpoints": null}), &latest)}),
    );
    cases
}

/// PRD T7's default half: Standard History (checkpoints, `history_card/1`,
/// `served_summary/1`) goes to a counterparty, and every other requester is
/// refused `policy_declined` before the subject is looked up.
fn standard_history_cases(latest: &Value, record: &str) -> Vec<Value> {
    let with_asker = |subject: Value, derivation: Option<&str>, asker: Option<&str>| {
        let mut r = json!({"subject": subject, "coverage": latest});
        if let Some(d) = derivation {
            r["derivation"] = json!(d);
        }
        if let Some(a) = asker {
            r["requester_id"] = json!(a);
        }
        r
    };
    let full = || json!({"full_history": null});
    // A case asked by `asker`: signed by it, so it is that node.
    let by = |name: &str, covers: &str, segments: &str, request: Value| match request
        .get("requester_id")
        .and_then(Value::as_str)
    {
        Some(node) => case_by(name, covers, segments, node, request.clone()),
        None => case(name, covers, segments, request),
    };
    vec![
        by(
            "checkpoints_to_a_non_counterparty",
            "checkpoints are not shared with a node this one never exchanged with",
            "counterparties",
            with_asker(json!({"checkpoints": null}), None, Some("node-z")),
        ),
        case(
            "checkpoints_to_a_stranger",
            "an asker naming no id is nobody's counterparty",
            "counterparties",
            with_asker(json!({"checkpoints": null}), None, None),
        ),
        by(
            "checkpoints_to_a_non_counterparty_under_peers",
            "share_history_segments governs records, never Standard History",
            "peers",
            with_asker(json!({"checkpoints": null}), None, Some("node-z")),
        ),
        by(
            "checkpoints_to_a_counterparty",
            "a counterparty is answered checkpoints",
            "counterparties",
            with_asker(json!({"checkpoints": null}), None, Some("node-a")),
        ),
        by(
            "history_card_to_a_non_counterparty",
            "history_card/1 is not shared with a node this one never exchanged with",
            "counterparties",
            with_asker(full(), Some("history_card/1"), Some("node-z")),
        ),
        by(
            "history_card_to_a_counterparty",
            "a counterparty is answered history_card/1",
            "counterparties",
            with_asker(full(), Some("history_card/1"), Some("node-b")),
        ),
        by(
            "served_summary_to_a_non_counterparty",
            "served_summary/1 is not shared with a node this one never exchanged with",
            "counterparties",
            with_asker(full(), Some("served_summary/1"), Some("node-z")),
        ),
        by(
            "served_summary_to_a_counterparty",
            "a counterparty is answered served_summary/1",
            "counterparties",
            with_asker(full(), Some("served_summary/1"), Some("node-a")),
        ),
        by(
            "derivation_unknown_to_a_non_counterparty",
            "the refusal comes before the derivation is looked at: it says nothing about what this node derives",
            "counterparties",
            with_asker(full(), Some("unregistered_fold/1"), Some("node-z")),
        ),
        by(
            "standard_history_unknown_pin_to_a_non_counterparty",
            "the refusal comes before the anchor is resolved: it says nothing about what this node holds",
            "counterparties",
            json!({"subject": {"checkpoints": null}, "coverage": {"expected_pin": "d".repeat(64)}, "requester_id": "node-z"}),
        ),
        by(
            "record_to_its_counterparty_under_prospective",
            "prospective, deprecated, is read as counterparties",
            "prospective",
            json!({"subject": {"record": record}, "coverage": latest, "requester_id": "node-a"}),
        ),
        by(
            "record_to_another_node_under_prospective",
            "prospective, deprecated, serves no one counterparties does not",
            "prospective",
            json!({"subject": {"record": record}, "coverage": latest, "requester_id": "node-b"}),
        ),
        case(
            "checkpoints_unsigned_naming_a_counterparty",
            "an unsigned request is nobody's, whatever id it states",
            "counterparties",
            with_asker(json!({"checkpoints": null}), None, Some("node-a")),
        ),
    ]
}

fn build_corpus() -> Value {
    let key = SigningKey::from_bytes(&NODE_KEY_SEED);
    let ledger = corpus_ledger();
    let cps = corpus_checkpoints(&ledger, &key);
    let cases = corpus_cases(&ledger, &cps);
    json!({
        "v": 1,
        "path": "evidence-request",
        "now": NOW,
        "announced_keys": announced_keys(),
        "node_id": RESPONDER_NODE,
        "ledger": ledger,
        "checkpoints": cps,
        "cases": cases,
    })
}

// ---------------------------------------------------------------------------
// Running it.
// ---------------------------------------------------------------------------

pub(crate) fn write_jsonl(path: &Path, lines: &[Value]) {
    let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
    std::fs::write(path, text).unwrap();
}

/// One answer, as the golden file records it: the checks, never the
/// signatures themselves.
fn summarize(request_bytes: &[u8], outcome: &Outcome, key: &SigningKey, cps: &[Value]) -> Value {
    match outcome {
        Outcome::Refused { wire, reason, .. } => json!({
            "refused": reason.token(),
            "signed_by_node": refusal::verify_for(wire, &key.verifying_key(), &request_digest(request_bytes)) == Ok(*reason),
            "request_digest_is_body_sha256": wire["request_digest"] == json!(hex::encode(Sha256::digest(request_bytes))),
            "issued_at_is_now": wire["issued_at"] == json!(NOW),
        }),
        Outcome::Answered { wire, subject_kind } => {
            let mut verification = evidence_answer::verify_response(
                request_bytes,
                wire,
                &key.verifying_key(),
                &Default::default(),
            );
            // The anchor by its place in the corpus, not its digest.
            if let Some(anchor) = verification
                .get("anchor")
                .and_then(Value::as_str)
                .map(str::to_string)
            {
                let at = cps.iter().position(|cp| {
                    serde_json::from_value::<CheckpointRecord>(cp.clone())
                        .is_ok_and(|cp| cp.digest() == anchor)
                });
                verification["anchor"] = json!(at.map(|i| format!("checkpoints[{i}]")));
            }
            let mut reply = json!({"answered": subject_kind, "verification": verification});
            let artifact: Value = wire["artifact"]
                .as_str()
                .and_then(|a| serde_json::from_str(a).ok())
                .unwrap_or(Value::Null);
            if let Some(summary) = artifact.get("served_summary") {
                reply["served_summary"] = summary.clone();
            }
            reply
        }
        Outcome::Unanswerable(why) => json!({"unanswerable": why}),
    }
}

fn run(corpus: &Value) -> BTreeMap<String, Value> {
    // The node's key is the runner's own fixed test key, never read from the
    // corpus: a data file carries no private key material, not even a test
    // key's.
    let key = SigningKey::from_bytes(&NODE_KEY_SEED);
    let data = tempfile::tempdir().unwrap();
    let ledger_dir = data.path().join("ledger");
    std::fs::create_dir_all(&ledger_dir).unwrap();
    write_jsonl(
        &ledger_dir.join("capsules.jsonl"),
        corpus["ledger"].as_array().unwrap(),
    );
    let cps = corpus["checkpoints"].as_array().unwrap();
    write_jsonl(&ledger_dir.join("checkpoints.jsonl"), cps);
    let now = corpus["now"].as_str().unwrap();
    // The keys the host announced for the asking nodes: public halves only.
    let announced: BTreeMap<String, String> =
        serde_json::from_value(corpus["announced_keys"].clone()).unwrap_or_default();

    let mut answers = BTreeMap::new();
    for case in corpus["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap().to_string();
        let bytes = match case.get("request_text").and_then(Value::as_str) {
            Some(text) => text.as_bytes().to_vec(),
            None => serde_json::to_vec(&case["request"]).unwrap(),
        };
        let log_dir = data.path().join(format!("received-log-{name}"));
        std::fs::create_dir_all(&log_dir).unwrap();
        // Each case's settings as the operator would write them, read the way
        // the plugin reads them. Standard History is not shared unless the
        // case says so.
        // A fresh replay cache per case: each case is one request.
        let replay = crate::requester_auth::ReplayCache::default();
        let responder = Responder {
            ledger_dir: &ledger_dir,
            signing_key: &key,
            now,
            history_segments: crate::share_policy::history_segments_for(
                case["history_segments"].as_str(),
            ),
            share_standard_history: case["share_standard_history"] == json!("on"),
            keys: crate::requester_auth::Keys {
                announced: &announced,
                configured: None,
            },
            replay: &replay,
            node_id: corpus["node_id"].as_str(),
        };
        let busy = case.get("busy").and_then(Value::as_bool).unwrap_or(false);
        let outcome = evidence_answer::answer_and_log(&bytes, busy, &responder, Some(&log_dir));
        let mut appended = serde_json::Map::new();
        if let Ok(text) =
            std::fs::read_to_string(log_dir.join(crate::received_log::RECEIVED_LOG_FILE))
        {
            let lines: Vec<Value> = text
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect();
            appended.insert(crate::received_log::RECEIVED_LOG_FILE.into(), json!(lines));
        }
        answers.insert(
            name,
            json!({"reply": summarize(&bytes, &outcome, &key, cps), "appended": appended}),
        );
    }
    answers
}

/// The layout every file here is written in: each top-level array or map
/// member one entry per line, so a changed case is a one-line diff.
fn write_layout(path: &Path, doc: &Value) {
    let obj = doc.as_object().unwrap();
    let mut out = String::from("{\n");
    let members: Vec<String> = obj
        .iter()
        .map(|(k, v)| match v {
            Value::Array(items) => format!(
                " {}: [\n{}\n ]",
                json!(k),
                items
                    .iter()
                    .map(|i| format!("  {i}"))
                    .collect::<Vec<_>>()
                    .join(",\n")
            ),
            Value::Object(map) if map.len() > 1 => format!(
                " {}: {{\n{}\n }}",
                json!(k),
                map.iter()
                    .map(|(k, v)| format!("  {}: {v}", json!(k)))
                    .collect::<Vec<_>>()
                    .join(",\n")
            ),
            _ => format!(" {}: {v}", json!(k)),
        })
        .collect();
    out.push_str(&members.join(",\n"));
    out.push_str("\n}\n");
    assert_eq!(
        serde_json::from_str::<Value>(&out).unwrap(),
        *doc,
        "the layout must not change the value"
    );
    std::fs::write(path, out).unwrap();
}

fn golden_doc(answers: BTreeMap<String, Value>) -> Value {
    json!({
        "v": 1,
        "path": "evidence-request",
        "implementation": "draft-mih-agent-evidence-request-00 (reviewed)",
        "answers": answers,
    })
}

fn read(path: &Path) -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap()
}

#[test]
fn every_answer_is_the_golden_answer() {
    let corpus = read(&parity_dir().join("corpus.json"));
    let golden = read(&parity_dir().join("golden.json"));
    let answers = run(&corpus);
    if let Ok(out) = std::env::var("EVIDENCE_REQUEST_PARITY_OUT") {
        write_layout(Path::new(&out), &golden_doc(answers.clone()));
    }
    let expected = golden["answers"].as_object().unwrap();
    let mut differ = Vec::new();
    for (name, want) in expected {
        match answers.get(name) {
            Some(got) if got == want => {}
            Some(got) => differ.push(format!("{name}:\n  golden: {want}\n  got:    {got}")),
            None => differ.push(format!("{name}: no answer")),
        }
    }
    for name in answers.keys().filter(|n| !expected.contains_key(*n)) {
        differ.push(format!("{name}: not in the golden answers"));
    }
    assert!(
        differ.is_empty(),
        "{} case(s) differ:\n{}",
        differ.len(),
        differ.join("\n")
    );
}

#[test]
fn the_corpus_reaches_every_outcome() {
    let golden = read(&parity_dir().join("golden.json"));
    let replies: Vec<&Value> = golden["answers"]
        .as_object()
        .unwrap()
        .values()
        .map(|a| &a["reply"])
        .collect();
    let refused: std::collections::BTreeSet<&str> = replies
        .iter()
        .filter_map(|r| r["refused"].as_str())
        .collect();
    for reason in [
        "not_authorized",
        "no_such_subject",
        "coverage_unsatisfiable",
        "derivation_unsupported",
        "policy_declined",
        "deadline_unmet",
        "request_malformed",
    ] {
        assert!(refused.contains(reason), "no case is refused {reason}");
    }
    for form in [
        "record",
        "range",
        "full_history",
        "checkpoints",
        "correlation",
        "exchange",
    ] {
        assert!(
            replies
                .iter()
                .any(|r| r["answered"] == json!(form)
                    && r["verification"]["state"] == json!("artifact")),
            "no case answers a {form} subject with an artifact that verifies"
        );
    }
    assert!(
        replies
            .iter()
            .all(|r| r.get("refused").is_none() || r["signed_by_node"] == json!(true)),
        "every refusal is signed by the node and binds the request"
    );
}

#[test]
#[ignore = "regenerates the corpus and golden answers; review before committing"]
fn write_corpus_and_golden() {
    if std::env::var("EVIDENCE_REQUEST_PARITY_WRITE").as_deref() != Ok("1") {
        return;
    }
    let corpus = build_corpus();
    std::fs::create_dir_all(parity_dir()).unwrap();
    write_layout(&parity_dir().join("corpus.json"), &corpus);
    write_layout(&parity_dir().join("golden.json"), &golden_doc(run(&corpus)));
}
