//! The evidence-request parity run, both halves against what judges them:
//!
//! - **The protocol**, against agent-action-capsule's conformance vectors
//!   for draft-mih-agent-evidence-request-00 (`vectors/aac-evidence-request/`,
//!   copied verbatim at the `crates/evidence-request-v0.2.0` tag): every
//!   request in `request.json` and `digest.json` through this node's
//!   responder (`evidence_answer::answer_and_log`), which refuses it with the
//!   vector's reason and binds the vector's digest; every refusal in
//!   `refusal.json` through this node's requester
//!   (`evidence_answer::verify_response`), which accepts exactly the
//!   conformant ones.
//! - **This node's own rules**, which no vector states: who may have which
//!   records (the sharing policy), who asks (`requester_auth`), Standard
//!   History, the served summary and the limits. The corpus below is a
//!   ledger and the requests made of it; [`EXPECTED`] states each one's
//!   answer and what the "asked of you" log records. Each answer is also
//!   checked as a requester would check it.
//!
//! The vectors' other files (resolution, outcomes, invariance, retention)
//! hold no bytes this node sends or receives: the `evidence-request` crate
//! is tested against them where it is developed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use cll::checkpoint::{
    checkpoint_to_cose, sign_checkpoint_digest, CheckpointRecord, WitnessRecord,
};
use cll::mmr::{
    add_leaf, consistency_proof, leaf_hash, peaks, root_from_peaks, MemoryNodeStore, NodeReader,
};
use ed25519_dalek::SigningKey;
use evidence_request::digest::request_digest;
use evidence_request::refusal;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::evidence_answer::{self, Outcome, Responder};

fn aac_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("vectors/aac-evidence-request")
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

/// A corpus checkpoint: its signed JSON record and its COSE wire statement,
/// as the checkpointer writes them.
pub(crate) struct CorpusCheckpoint {
    pub record: CheckpointRecord,
    pub statement: Vec<u8>,
}

impl CorpusCheckpoint {
    /// The digest a requester pins it by: the SHA-256 of its statement.
    pub fn anchor(&self) -> String {
        hex::encode(Sha256::digest(&self.statement))
    }

    /// Its line in `checkpoints.jsonl`.
    pub fn line(&self) -> Value {
        let mut line = serde_json::to_value(&self.record).unwrap();
        line["checkpoint_cose"] = json!(hex::encode(&self.statement));
        line
    }
}

/// Checkpoints after 16 and after 300 lines, signed by the node key, each
/// with its COSE wire statement; the second carries one witness.
pub(crate) fn corpus_checkpoints(ledger: &[Value], key: &SigningKey) -> Vec<CorpusCheckpoint> {
    let mut nodes = MemoryNodeStore::new();
    let mut out: Vec<CorpusCheckpoint> = Vec::new();
    for (i, record) in ledger.iter().enumerate() {
        let id: [u8; 32] = hex::decode(record["capsule_id"].as_str().unwrap())
            .unwrap()
            .try_into()
            .unwrap();
        add_leaf(&mut nodes, leaf_hash(&id)).unwrap();
        if i + 1 == 16 || i + 1 == 300 {
            let size = nodes.size();
            let peak_hashes = |size: u64| -> Vec<_> {
                peaks(size)
                    .unwrap()
                    .iter()
                    .map(|&p| nodes.node(p))
                    .collect()
            };
            let prev = out.last().map(|c| &c.record);
            let mut cp = CheckpointRecord {
                v: 1,
                kind: "mmr_checkpoint".into(),
                log_id: "corpus-node-log".into(),
                mmr_size: size,
                root: hex::encode(root_from_peaks(&peak_hashes(size))),
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
            let prev_peaks = (cp.prev_size > 0).then(|| peak_hashes(cp.prev_size));
            let consistency =
                (cp.prev_size > 0).then(|| consistency_proof(&nodes, cp.prev_size, size).unwrap());
            let statement = checkpoint_to_cose(
                &cp,
                key,
                &peak_hashes(size),
                prev_peaks.as_deref(),
                consistency.as_ref(),
                None,
            )
            .unwrap();
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
            out.push(CorpusCheckpoint {
                record: cp,
                statement,
            });
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

fn corpus_cases(ledger: &[Value], cps: &[CorpusCheckpoint]) -> Vec<Value> {
    let id = |i: usize| ledger[i]["capsule_id"].as_str().unwrap().to_string();
    let latest = json!({"expected_pin": cps[1].anchor()});
    let first = json!({"expected_pin": cps[0].anchor()});
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
                r["coverage"] = json!({"expected_pin": cps[0].anchor()});
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
                &json!({"expected_pin": cps[1].anchor(), "min_freshness": 1}),
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

// ---------------------------------------------------------------------------
// This node's own rules: running the corpus.
// ---------------------------------------------------------------------------

pub(crate) fn write_jsonl(path: &Path, lines: &[Value]) {
    let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
    std::fs::write(path, text).unwrap();
}

/// A ledger directory holding the corpus ledger and its checkpoints.
pub(crate) fn corpus_ledger_dir(dir: &Path, key: &SigningKey) -> Vec<CorpusCheckpoint> {
    let ledger = corpus_ledger();
    let cps = corpus_checkpoints(&ledger, key);
    std::fs::create_dir_all(dir).unwrap();
    write_jsonl(&dir.join(crate::evidence_log::CAPSULES_FILE), &ledger);
    let lines: Vec<Value> = cps.iter().map(CorpusCheckpoint::line).collect();
    write_jsonl(&dir.join(crate::evidence_log::CHECKPOINTS_FILE), &lines);
    cps
}

/// One answer as [`EXPECTED`] states it, and anything wrong with it that
/// holds for every answer: a refusal must be signed by the node, bind the
/// request as received and be issued now; an artifact answer must verify.
fn summarize(
    request_bytes: &[u8],
    outcome: &Outcome,
    key: &SigningKey,
    cps: &[CorpusCheckpoint],
    problems: &mut Vec<String>,
) -> Value {
    match outcome {
        Outcome::Refused { wire, reason, .. } => {
            let digest = request_digest(request_bytes);
            if refusal::verify_for(wire, &key.verifying_key(), &digest) != Ok(*reason) {
                problems.push("a refusal is not the node's signed refusal of this request".into());
            }
            if wire["request_digest"] != json!(hex::encode(Sha256::digest(request_bytes))) {
                problems.push("a refusal does not bind the request as received".into());
            }
            if wire["issued_at"] != json!(NOW) {
                problems.push("a refusal is not issued now".into());
            }
            json!({"refused": reason.token()})
        }
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
                let at = cps.iter().position(|cp| cp.anchor() == anchor);
                verification["anchor"] = json!(at.map(|i| format!("checkpoints[{i}]")));
            }
            let mut reply = json!({"answered": subject_kind, "verification": verification});
            let artifact: Value = wire["artifact"]
                .as_str()
                .and_then(|a| {
                    use base64::Engine as _;
                    base64::engine::general_purpose::URL_SAFE_NO_PAD
                        .decode(a)
                        .ok()
                })
                .and_then(|a| serde_json::from_slice(&a).ok())
                .unwrap_or(Value::Null);
            if artifact.get("served_summary").is_some_and(Value::is_object) {
                reply["served_summary"] = json!(true);
            }
            reply
        }
        Outcome::Unanswerable(why) => json!({"unanswerable": why}),
    }
}

/// The "asked of you" log line an answer appended, as [`EXPECTED`] states
/// it: who asked and how that was established. Its time, path, status,
/// reason and request digest must be the answer's own.
fn logged(
    log_dir: &Path,
    request_bytes: &[u8],
    outcome: &Outcome,
    problems: &mut Vec<String>,
) -> Value {
    let lines: Vec<Value> =
        std::fs::read_to_string(log_dir.join(crate::received_log::RECEIVED_LOG_FILE))
            .unwrap_or_default()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
    let [line] = lines.as_slice() else {
        problems.push(format!("{} log lines, not one", lines.len()));
        return Value::Null;
    };
    let (status, reason) = outcome.log_status();
    if line["ts"] != json!(NOW)
        || line["path"] != json!("evidence-request")
        || line["status"] != json!(status)
        || line["reason"] != json!(reason)
        || line["request_digest"] != json!(request_digest(request_bytes))
    {
        problems.push(format!("the log line is not this answer's: {line}"));
    }
    let mut out = serde_json::Map::new();
    for member in [
        "requester_id",
        "subject_kind",
        "principal_key_id",
        "principal_node_id",
        "principal_binding",
        "audience",
        "replay",
    ] {
        if let Some(v) = line.get(member).filter(|v| !v.is_null()) {
            out.insert(member.into(), v.clone());
        }
    }
    Value::Object(out)
}

/// Every corpus case through the responder: `{name: {"reply", "log"}}`, and
/// what is wrong with any of them.
fn run() -> (BTreeMap<String, Value>, Vec<String>) {
    let key = SigningKey::from_bytes(&NODE_KEY_SEED);
    let data = tempfile::tempdir().unwrap();
    let ledger_dir = data.path().join("ledger");
    let cps = corpus_ledger_dir(&ledger_dir, &key);
    let cases = corpus_cases(&corpus_ledger(), &cps);
    let announced = announced_keys();

    let mut answers = BTreeMap::new();
    let mut problems = Vec::new();
    for case in &cases {
        let name = case["name"].as_str().unwrap().to_string();
        let bytes = match case.get("request_text").and_then(Value::as_str) {
            Some(text) => text.as_bytes().to_vec(),
            None => serde_json::to_vec(&case["request"]).unwrap(),
        };
        let log_dir = data.path().join(format!("received-log-{name}"));
        std::fs::create_dir_all(&log_dir).unwrap();
        // Each case's settings as the operator would write them, read the
        // way the plugin reads them. Standard History is not shared unless
        // the case says so. A fresh replay cache per case: each case is one
        // request.
        let replay = crate::requester_auth::ReplayCache::default();
        let responder = Responder {
            ledger_dir: &ledger_dir,
            signing_key: &key,
            now: NOW,
            history_segments: crate::share_policy::history_segments_for(
                case["history_segments"].as_str(),
            ),
            share_standard_history: case["share_standard_history"] == json!("on"),
            share_exchange_corroboration: case["share_exchange_corroboration"] == json!("on"),
            keys: crate::requester_auth::Keys {
                announced: &announced,
                configured: None,
            },
            replay: &replay,
            node_id: Some(RESPONDER_NODE),
        };
        let busy = case.get("busy").and_then(Value::as_bool).unwrap_or(false);
        let outcome = evidence_answer::answer_and_log(&bytes, busy, &responder, Some(&log_dir));
        let mut found = Vec::new();
        let reply = summarize(&bytes, &outcome, &key, &cps, &mut found);
        let log = logged(&log_dir, &bytes, &outcome, &mut found);
        problems.extend(found.into_iter().map(|p| format!("{name}: {p}")));
        answers.insert(name, json!({"reply": reply, "log": log}));
    }
    (answers, problems)
}

#[test]
fn every_case_is_answered_as_expected() {
    let (answers, problems) = run();
    assert!(problems.is_empty(), "{}", problems.join("\n"));
    let mut differ = Vec::new();
    for (name, want) in EXPECTED {
        let want: Value = serde_json::from_str(want).unwrap();
        match answers.get(*name) {
            Some(got) if *got == want => {}
            Some(got) => differ.push(format!("{name}:\n  expected: {want}\n  got:      {got}")),
            None => differ.push(format!("{name}: not in the corpus")),
        }
    }
    for name in answers.keys() {
        if !EXPECTED.iter().any(|(n, _)| n == name) {
            differ.push(format!("{name}: no expected answer"));
        }
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
    let replies: Vec<Value> = EXPECTED
        .iter()
        .map(|(_, e)| serde_json::from_str::<Value>(e).unwrap()["reply"].clone())
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
}

// ---------------------------------------------------------------------------
// The protocol: agent-action-capsule's vectors.
// ---------------------------------------------------------------------------

/// A vector file, checked against the copy's own `SHA256SUMS` first.
fn aac(file: &str) -> Value {
    let dir = aac_dir();
    let bytes = std::fs::read(dir.join(file)).unwrap();
    let sums = std::fs::read_to_string(dir.join("SHA256SUMS")).unwrap();
    let pinned = sums
        .lines()
        .find_map(|l| l.strip_suffix(&format!("  {file}")))
        .unwrap_or_else(|| panic!("{file} is not in SHA256SUMS"));
    assert_eq!(hex::encode(Sha256::digest(&bytes)), pinned, "{file}");
    serde_json::from_slice(&bytes).unwrap()
}

/// A node over the corpus ledger that serves records to any peer and shares
/// Standard History: what it answers depends on the request alone.
fn answer_openly(dir: &Path, bytes: &[u8]) -> Outcome {
    let key = SigningKey::from_bytes(&NODE_KEY_SEED);
    let announced = announced_keys();
    let replay = crate::requester_auth::ReplayCache::default();
    evidence_answer::answer_and_log(
        bytes,
        false,
        &Responder {
            ledger_dir: dir,
            signing_key: &key,
            now: NOW,
            history_segments: "peers",
            share_standard_history: true,
            share_exchange_corroboration: false,
            keys: crate::requester_auth::Keys {
                announced: &announced,
                configured: None,
            },
            replay: &replay,
            node_id: Some(RESPONDER_NODE),
        },
        None,
    )
}

/// `request.json`: a request the vectors call malformed is refused with
/// their reason; a well-formed one parses here and is never refused as
/// malformed (the vectors expect nothing more of it). Every
/// refusal binds the vector's digest of the bytes as received. The JSON
/// binding only: this node takes no CBOR requests.
#[test]
fn the_responder_refuses_each_aac_request_as_the_vectors_say() {
    let data = tempfile::tempdir().unwrap();
    corpus_ledger_dir(data.path(), &SigningKey::from_bytes(&NODE_KEY_SEED));
    let vectors = aac("request.json");
    let cases = vectors["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 33);
    let mut differ = Vec::new();
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let bytes = case["request_json"].as_str().unwrap().as_bytes();
        assert_eq!(request_digest(bytes), case["request_json_digest"], "{id}");
        let outcome = answer_openly(data.path(), bytes);
        let refused = match &outcome {
            Outcome::Refused { wire, reason, .. } => {
                if wire["request_digest"] != case["request_json_digest"] {
                    differ.push(format!("{id}: the refusal binds another digest"));
                }
                Some(reason.token())
            }
            _ => None,
        };
        let expect = &case["expect"];
        let ok = if expect["well_formed"] == json!(true) {
            evidence_request::request::parse_json(bytes).is_ok()
                && refused != Some("request_malformed")
        } else {
            refused == expect["reason"].as_str()
        };
        if !ok {
            differ.push(format!("{id}: expected {expect}, got {refused:?}"));
        }
    }
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}

/// `digest.json`: the digest a refusal binds is over the bytes as received,
/// whitespace and all, so it is the vector's. The requester takes a refusal
/// as the answer to its request only when it carries the digest of the bytes
/// it sent.
#[test]
fn a_refusal_binds_the_aac_digest_of_the_bytes_as_received() {
    let data = tempfile::tempdir().unwrap();
    corpus_ledger_dir(data.path(), &SigningKey::from_bytes(&NODE_KEY_SEED));
    let refusals = aac("refusal.json");
    let seed: [u8; 32] = hex::decode(refusals["keys"]["primary"]["seed_hex"].as_str().unwrap())
        .unwrap()
        .try_into()
        .unwrap();
    let signer = SigningKey::from_bytes(&seed);
    let (mut answered, mut identified) = (0, 0);
    for case in aac("digest.json")["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let Some(bytes) = case["request_json"].as_str().map(str::as_bytes) else {
            assert_eq!(case["encoding"], json!("cbor"), "{id}");
            continue;
        };
        if let Some(want) = case["expect"].get("request_digest") {
            let Outcome::Refused { wire, .. } = answer_openly(data.path(), bytes) else {
                panic!("{id}: the corpus node answers none of these requests");
            };
            assert_eq!(&wire["request_digest"], want, "{id}");
            answered += 1;
        }
        if let Some(want) = case["expect"].get("identifies_request") {
            let refusal = refusal::sign(
                case["refusal_request_digest"].as_str().unwrap(),
                evidence_request::registry::Reason::NoSuchSubject,
                NOW,
                &signer,
            )
            .unwrap()
            .to_json();
            let got = evidence_answer::verify_response(
                bytes,
                &refusal,
                &signer.verifying_key(),
                &Default::default(),
            );
            assert_eq!(
                got["state"] == json!("refusal"),
                *want == json!(true),
                "{id}: {got}"
            );
            identified += 1;
        }
    }
    assert_eq!((answered, identified), (3, 2));
}

/// `refusal.json`: the requester accepts a refusal exactly when the vectors
/// call it conformant, as the refusal of the request it names. The CBOR
/// frame and the imported file name no request this node sent.
#[test]
fn the_requester_accepts_exactly_the_conformant_aac_refusals() {
    let requests = aac("request.json");
    let sent_with_digest = |digest: &Value| {
        requests["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["request_json_digest"] == *digest)
            .map(|c| c["request_json"].as_str().unwrap().as_bytes().to_vec())
    };
    let vectors = aac("refusal.json");
    let key_hex = vectors["keys"]["primary"]["public_key_hex"]
        .as_str()
        .unwrap();
    let key =
        ed25519_dalek::VerifyingKey::from_bytes(&hex::decode(key_hex).unwrap().try_into().unwrap())
            .unwrap();
    let mut checked = 0;
    let mut differ = Vec::new();
    for case in vectors["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        if case.get("refusal_cbor_hex").is_some() || case.get("file").is_some() {
            continue;
        }
        let refusal = &case["refusal"];
        // The request it refuses, else one it does not: a tampered digest
        // names no request.
        let sent = sent_with_digest(&refusal["request_digest"]).unwrap_or_else(|| {
            requests["cases"][0]["request_json"]
                .as_str()
                .unwrap()
                .as_bytes()
                .to_vec()
        });
        let got = evidence_answer::verify_response(&sent, refusal, &key, &Default::default());
        let conformant = case["expect"]["conformant"] == json!(true);
        let accepted = got["state"] == json!("refusal") && got["reason"] == refusal["reason"];
        if accepted != conformant {
            differ.push(format!("{id}: conformant {conformant}, got {got}"));
        }
        if !accepted && got["state"] != json!("not_evidence") {
            differ.push(format!("{id}: neither accepted nor not evidence: {got}"));
        }
        checked += 1;
    }
    assert!(differ.is_empty(), "{}", differ.join("\n"));
    assert_eq!(checked, 19);
}

/// What each corpus case is answered, and what the log records of who asked
/// (`run`). Reviewed case by case against draft-mih-agent-evidence-request-00
/// and this node's sharing policy (`docs/SHARING-POLICY.md`).
const EXPECTED: &[(&str, &str)] = &[
    (
        "record_to_its_counterparty",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"3e7d04d3813a4c90546bf2dec79da711152dbdc5f927f152485c6555887f2dde","principal_node_id":"node-a","replay":"checked","requester_id":"node-a","subject_kind":"record"},"reply":{"answered":"record","verification":{"anchor":"checkpoints[1]","checkpoints":0,"records":[0],"state":"artifact"}}}"#,
    ),
    (
        "record_to_another_node",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"86f14770b6f56d58727d65f274104d73ecf947c2e0ed1330d09818fad8b8fa05","principal_node_id":"node-b","replay":"checked","requester_id":"node-b","subject_kind":"record"},"reply":{"refused":"not_authorized"}}"#,
    ),
    (
        "record_to_a_stranger",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"7808d43d8cf3775d0d679359fd0771ff3090bc1d23aecc650170af8a0c5bca2d","principal_node_id":"node-z","replay":"checked","subject_kind":"record"},"reply":{"refused":"not_authorized"}}"#,
    ),
    (
        "record_unknown_to_a_stranger",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"7808d43d8cf3775d0d679359fd0771ff3090bc1d23aecc650170af8a0c5bca2d","principal_node_id":"node-z","replay":"checked","subject_kind":"record"},"reply":{"refused":"not_authorized"}}"#,
    ),
    (
        "range_over_limit_to_a_stranger",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"7808d43d8cf3775d0d679359fd0771ff3090bc1d23aecc650170af8a0c5bca2d","principal_node_id":"node-z","replay":"checked","subject_kind":"range"},"reply":{"refused":"not_authorized"}}"#,
    ),
    (
        "record_unsigned_naming_its_counterparty",
        r#"{"log":{"requester_id":"node-a","subject_kind":"record"},"reply":{"refused":"not_authorized"}}"#,
    ),
    (
        "record_unsigned_asking_nothing_held",
        r#"{"log":{"subject_kind":"record"},"reply":{"refused":"not_authorized"}}"#,
    ),
    (
        "record_signed_by_one_claiming_another",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"78461563dd9830102634c9eadaaad51c39787a472cef9303f7a23ab31e5ba0d4","principal_node_id":"node-y","replay":"checked","requester_id":"node-a","subject_kind":"record"},"reply":{"refused":"not_authorized"}}"#,
    ),
    (
        "record_signed_under_another_nodes_key_id",
        r#"{"log":{"subject_kind":"record"},"reply":{"refused":"not_authorized"}}"#,
    ),
    (
        "record_signature_over_another_request",
        r#"{"log":{"subject_kind":"record"},"reply":{"refused":"not_authorized"}}"#,
    ),
    (
        "record_signed_by_an_unannounced_key",
        r#"{"log":{"audience":"node-r","principal_binding":"unbound","principal_key_id":"c853ad0f0cd2b619aea92ceec4fd56a24d6499d584ce79257e45cfd8139b60a7","replay":"checked","subject_kind":"record"},"reply":{"answered":"record","verification":{"anchor":"checkpoints[1]","checkpoints":0,"records":[0],"state":"artifact"}}}"#,
    ),
    (
        "record_signed_by_an_unannounced_key_needing_identity",
        r#"{"log":{"audience":"node-r","principal_binding":"unbound","principal_key_id":"c853ad0f0cd2b619aea92ceec4fd56a24d6499d584ce79257e45cfd8139b60a7","replay":"checked","subject_kind":"record"},"reply":{"refused":"not_authorized"}}"#,
    ),
    (
        "record_signed_without_an_audience",
        r#"{"log":{"subject_kind":"record"},"reply":{"refused":"request_malformed"}}"#,
    ),
    (
        "record_relayed_from_another_audience",
        r#"{"log":{"audience":"node-x","principal_key_id":"3e7d04d3813a4c90546bf2dec79da711152dbdc5f927f152485c6555887f2dde","subject_kind":"record"},"reply":{"refused":"not_authorized"}}"#,
    ),
    (
        "requester_auth_short_key",
        r#"{"log":{"subject_kind":"record"},"reply":{"refused":"request_malformed"}}"#,
    ),
    (
        "record_under_peers",
        r#"{"log":{"subject_kind":"record"},"reply":{"answered":"record","verification":{"anchor":"checkpoints[1]","checkpoints":0,"records":[0],"state":"artifact"}}}"#,
    ),
    (
        "record_sharing_off",
        r#"{"log":{"requester_id":"node-a","subject_kind":"record"},"reply":{"refused":"not_authorized"}}"#,
    ),
    (
        "record_counterparties_tier",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"86f14770b6f56d58727d65f274104d73ecf947c2e0ed1330d09818fad8b8fa05","principal_node_id":"node-b","replay":"checked","requester_id":"node-b","subject_kind":"record"},"reply":{"answered":"record","verification":{"anchor":"checkpoints[0]","checkpoints":0,"records":[1],"state":"artifact"}}}"#,
    ),
    (
        "record_local_block_never_leaves",
        r#"{"log":{"subject_kind":"record"},"reply":{"refused":"not_authorized"}}"#,
    ),
    (
        "record_unknown",
        r#"{"log":{"subject_kind":"record"},"reply":{"refused":"no_such_subject"}}"#,
    ),
    (
        "record_padding",
        r#"{"log":{"subject_kind":"record"},"reply":{"refused":"no_such_subject"}}"#,
    ),
    (
        "record_prefix",
        r#"{"log":{},"reply":{"refused":"request_malformed"}}"#,
    ),
    (
        "record_not_yet_checkpointed",
        r#"{"log":{"subject_kind":"record"},"reply":{"refused":"coverage_unsatisfiable"}}"#,
    ),
    (
        "record_under_first_anchor",
        r#"{"log":{"subject_kind":"record"},"reply":{"answered":"record","verification":{"anchor":"checkpoints[0]","checkpoints":0,"records":[3],"state":"artifact"}}}"#,
    ),
    (
        "range_under_peers",
        r#"{"log":{"subject_kind":"range"},"reply":{"answered":"range","verification":{"anchor":"checkpoints[0]","checkpoints":0,"records":[5,6,7,8,9,10,11,12],"state":"artifact"}}}"#,
    ),
    (
        "range_to_a_counterparty",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"3e7d04d3813a4c90546bf2dec79da711152dbdc5f927f152485c6555887f2dde","principal_node_id":"node-a","replay":"checked","requester_id":"node-a","subject_kind":"range"},"reply":{"answered":"range","verification":{"anchor":"checkpoints[1]","checkpoints":0,"records":[16,17,18,19,20],"state":"artifact"}}}"#,
    ),
    (
        "range_mixed_counterparties",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"3e7d04d3813a4c90546bf2dec79da711152dbdc5f927f152485c6555887f2dde","principal_node_id":"node-a","replay":"checked","requester_id":"node-a","subject_kind":"range"},"reply":{"refused":"not_authorized"}}"#,
    ),
    (
        "range_with_a_local_block",
        r#"{"log":{"subject_kind":"range"},"reply":{"refused":"not_authorized"}}"#,
    ),
    (
        "range_past_anchor",
        r#"{"log":{"subject_kind":"range"},"reply":{"refused":"coverage_unsatisfiable"}}"#,
    ),
    (
        "range_over_limit",
        r#"{"log":{"subject_kind":"range"},"reply":{"refused":"policy_declined"}}"#,
    ),
    (
        "full_history_over_limit",
        r#"{"log":{"subject_kind":"full_history"},"reply":{"refused":"policy_declined"}}"#,
    ),
    (
        "full_history_under_first_anchor",
        r#"{"log":{"subject_kind":"full_history"},"reply":{"answered":"full_history","verification":{"anchor":"checkpoints[0]","checkpoints":0,"records":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],"state":"artifact"}}}"#,
    ),
    (
        "correlation_to_its_counterparty",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"3e7d04d3813a4c90546bf2dec79da711152dbdc5f927f152485c6555887f2dde","principal_node_id":"node-a","replay":"checked","requester_id":"node-a","subject_kind":"correlation"},"reply":{"answered":"correlation","verification":{"anchor":"checkpoints[1]","checkpoints":0,"records":[0,4],"state":"artifact"}}}"#,
    ),
    (
        "correlation_by_exchange_id",
        r#"{"log":{"subject_kind":"correlation"},"reply":{"answered":"correlation","verification":{"anchor":"checkpoints[1]","checkpoints":0,"records":[2],"state":"artifact"}}}"#,
    ),
    (
        "correlation_unknown",
        r#"{"log":{"subject_kind":"correlation"},"reply":{"refused":"no_such_subject"}}"#,
    ),
    (
        "exchange_citing_the_half",
        r#"{"log":{"subject_kind":"exchange"},"reply":{"answered":"exchange","verification":{"anchor":"checkpoints[0]","checkpoints":0,"records":[6],"state":"artifact"}}}"#,
    ),
    (
        "exchange_nothing_cites",
        r#"{"log":{"subject_kind":"exchange"},"reply":{"refused":"no_such_subject"}}"#,
    ),
    (
        "checkpoints_with_sharing_off",
        r#"{"log":{"subject_kind":"checkpoints"},"reply":{"answered":"checkpoints","verification":{"anchor":"checkpoints[1]","checkpoints":2,"records":[],"state":"artifact"}}}"#,
    ),
    (
        "history_card",
        r#"{"log":{"subject_kind":"full_history"},"reply":{"answered":"full_history","verification":{"anchor":"checkpoints[1]","checkpoints":2,"records":[],"state":"artifact"}}}"#,
    ),
    (
        "history_card_wrong_subject",
        r#"{"log":{"subject_kind":"record"},"reply":{"refused":"no_such_subject"}}"#,
    ),
    (
        "derivation_unknown",
        r#"{"log":{"subject_kind":"full_history"},"reply":{"refused":"derivation_unsupported"}}"#,
    ),
    (
        "served_summary",
        r#"{"log":{"subject_kind":"full_history"},"reply":{"answered":"full_history","served_summary":true,"verification":{"anchor":"checkpoints[1]","checkpoints":0,"records":[],"state":"artifact"}}}"#,
    ),
    (
        "served_summary_older_pin",
        r#"{"log":{"subject_kind":"full_history"},"reply":{"refused":"coverage_unsatisfiable"}}"#,
    ),
    (
        "served_summary_on_demand",
        r#"{"log":{"subject_kind":"full_history"},"reply":{"refused":"policy_declined"}}"#,
    ),
    (
        "coverage_unknown_pin",
        r#"{"log":{"subject_kind":"checkpoints"},"reply":{"refused":"coverage_unsatisfiable"}}"#,
    ),
    (
        "coverage_both_members",
        r#"{"log":{},"reply":{"refused":"coverage_unsatisfiable"}}"#,
    ),
    (
        "coverage_missing",
        r#"{"log":{},"reply":{"refused":"coverage_unsatisfiable"}}"#,
    ),
    (
        "coverage_fresher_than_held",
        r#"{"log":{"subject_kind":"checkpoints"},"reply":{"refused":"coverage_unsatisfiable"}}"#,
    ),
    (
        "coverage_by_time",
        r#"{"log":{"subject_kind":"checkpoints"},"reply":{"answered":"checkpoints","verification":{"anchor":"checkpoints[1]","checkpoints":2,"records":[],"state":"artifact"}}}"#,
    ),
    (
        "pre_draft_request_shape",
        r#"{"log":{},"reply":{"refused":"request_malformed"}}"#,
    ),
    (
        "deadline_passed",
        r#"{"log":{"subject_kind":"checkpoints"},"reply":{"refused":"deadline_unmet"}}"#,
    ),
    (
        "checkpoints_to_a_non_counterparty",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"7808d43d8cf3775d0d679359fd0771ff3090bc1d23aecc650170af8a0c5bca2d","principal_node_id":"node-z","replay":"checked","requester_id":"node-z","subject_kind":"checkpoints"},"reply":{"refused":"policy_declined"}}"#,
    ),
    (
        "checkpoints_to_a_stranger",
        r#"{"log":{"subject_kind":"checkpoints"},"reply":{"refused":"policy_declined"}}"#,
    ),
    (
        "checkpoints_to_a_non_counterparty_under_peers",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"7808d43d8cf3775d0d679359fd0771ff3090bc1d23aecc650170af8a0c5bca2d","principal_node_id":"node-z","replay":"checked","requester_id":"node-z","subject_kind":"checkpoints"},"reply":{"refused":"policy_declined"}}"#,
    ),
    (
        "checkpoints_to_a_counterparty",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"3e7d04d3813a4c90546bf2dec79da711152dbdc5f927f152485c6555887f2dde","principal_node_id":"node-a","replay":"checked","requester_id":"node-a","subject_kind":"checkpoints"},"reply":{"answered":"checkpoints","verification":{"anchor":"checkpoints[1]","checkpoints":2,"records":[],"state":"artifact"}}}"#,
    ),
    (
        "history_card_to_a_non_counterparty",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"7808d43d8cf3775d0d679359fd0771ff3090bc1d23aecc650170af8a0c5bca2d","principal_node_id":"node-z","replay":"checked","requester_id":"node-z","subject_kind":"full_history"},"reply":{"refused":"policy_declined"}}"#,
    ),
    (
        "history_card_to_a_counterparty",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"86f14770b6f56d58727d65f274104d73ecf947c2e0ed1330d09818fad8b8fa05","principal_node_id":"node-b","replay":"checked","requester_id":"node-b","subject_kind":"full_history"},"reply":{"answered":"full_history","verification":{"anchor":"checkpoints[1]","checkpoints":2,"records":[],"state":"artifact"}}}"#,
    ),
    (
        "served_summary_to_a_non_counterparty",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"7808d43d8cf3775d0d679359fd0771ff3090bc1d23aecc650170af8a0c5bca2d","principal_node_id":"node-z","replay":"checked","requester_id":"node-z","subject_kind":"full_history"},"reply":{"refused":"policy_declined"}}"#,
    ),
    (
        "served_summary_to_a_counterparty",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"3e7d04d3813a4c90546bf2dec79da711152dbdc5f927f152485c6555887f2dde","principal_node_id":"node-a","replay":"checked","requester_id":"node-a","subject_kind":"full_history"},"reply":{"answered":"full_history","served_summary":true,"verification":{"anchor":"checkpoints[1]","checkpoints":0,"records":[],"state":"artifact"}}}"#,
    ),
    (
        "derivation_unknown_to_a_non_counterparty",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"7808d43d8cf3775d0d679359fd0771ff3090bc1d23aecc650170af8a0c5bca2d","principal_node_id":"node-z","replay":"checked","requester_id":"node-z","subject_kind":"full_history"},"reply":{"refused":"policy_declined"}}"#,
    ),
    (
        "standard_history_unknown_pin_to_a_non_counterparty",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"7808d43d8cf3775d0d679359fd0771ff3090bc1d23aecc650170af8a0c5bca2d","principal_node_id":"node-z","replay":"checked","requester_id":"node-z","subject_kind":"checkpoints"},"reply":{"refused":"policy_declined"}}"#,
    ),
    (
        "record_to_its_counterparty_under_prospective",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"3e7d04d3813a4c90546bf2dec79da711152dbdc5f927f152485c6555887f2dde","principal_node_id":"node-a","replay":"checked","requester_id":"node-a","subject_kind":"record"},"reply":{"answered":"record","verification":{"anchor":"checkpoints[1]","checkpoints":0,"records":[0],"state":"artifact"}}}"#,
    ),
    (
        "record_to_another_node_under_prospective",
        r#"{"log":{"audience":"node-r","principal_binding":"bound","principal_key_id":"86f14770b6f56d58727d65f274104d73ecf947c2e0ed1330d09818fad8b8fa05","principal_node_id":"node-b","replay":"checked","requester_id":"node-b","subject_kind":"record"},"reply":{"refused":"not_authorized"}}"#,
    ),
    (
        "checkpoints_unsigned_naming_a_counterparty",
        r#"{"log":{"requester_id":"node-a","subject_kind":"checkpoints"},"reply":{"refused":"policy_declined"}}"#,
    ),
    (
        "not_json",
        r#"{"log":{},"reply":{"refused":"request_malformed"}}"#,
    ),
    (
        "request_too_large",
        r#"{"log":{},"reply":{"refused":"policy_declined"}}"#,
    ),
    (
        "busy",
        r#"{"log":{},"reply":{"refused":"policy_declined"}}"#,
    ),
];
