//! Tier 2: the checks run over a real two-node run.
//!
//! `scripts/e2e-two-node.sh` starts two unmodified `mesh-llm` nodes on one
//! machine, each running this plugin: the provider serves a small model, the
//! requester serves nothing and sends its chat completions to the provider
//! over the mesh. After both nodes have stopped, the script copies each
//! node's plugin data directory and runs these tests over the copies:
//!
//! ```sh
//! E2E_REQUESTER_DIR=<copy> E2E_PROVIDER_DIR=<copy> E2E_EXCHANGES=<n> \
//!   cargo test --locked --bin capsules -- --ignored --test-threads=1 two_node_e2e::
//! ```
//!
//! Every check calls the plugin's own code (the ledger reload, the receiver,
//! the Evidence pane builder) on what the two nodes actually wrote. Every
//! mutation happens on a scratch copy; the collected directories are never
//! changed.
//!
//! **The host names the other side** (mesh-llm 0.78 and later): the
//! provider's events carry `requested_by_node_id`, the requester's carry
//! `served_by_node_id`, with matching digests. The script turns record push
//! on at both ends and gives the requester the provider's key, so
//! [`the_host_names_both_sides_and_the_requester_holds_the_providers_record`]
//! checks the confirmed rows on the requester. The provider cannot hold the
//! requester's record: the requester is a client node, whose id is new each
//! time it starts, so no key can be configured for it; that test checks the
//! provider refused it for exactly that reason.
//!
//! Ignored by default: they need the data directories of a real run.

use std::path::{Path, PathBuf};

use capsule_emit_lib::ledger::LedgerError;
use ed25519_dalek::SigningKey;
use serde_json::{json, Value};

use crate::record_push_receive::{
    receive, Receiver, MAX_REJECTED_LOG_BYTES, REASON_MODEL_MISMATCH, REASON_REQUEST_MALFORMED,
    REASON_SERVED_BY_MISMATCH, REASON_SIGNATURE_UNVERIFIED,
};

const NOW: &str = "2026-01-01T00:00:00Z";
const OTHER_WEIGHTS: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

/// One node's plugin data directory, as the run left it.
struct Node {
    data_dir: PathBuf,
}

impl Node {
    fn from_env(var: &str) -> Self {
        let dir = std::env::var(var)
            .unwrap_or_else(|_| panic!("{var} must name a node's plugin data directory"));
        let data_dir = PathBuf::from(dir);
        assert!(
            data_dir.join("ledger").join("capsules.jsonl").is_file(),
            "{var}: no ledger/capsules.jsonl under {}",
            data_dir.display()
        );
        Self { data_dir }
    }

    fn ledger_dir(&self) -> PathBuf {
        self.data_dir.join("ledger")
    }

    /// Every line of this node's own chained log, padding included.
    fn ledger_lines(&self) -> Vec<Value> {
        read_jsonl(&self.ledger_dir().join("capsules.jsonl"))
    }

    /// This node's own sealed halves of exchanges with `role`.
    fn halves(&self, role: &str) -> Vec<Value> {
        self.ledger_lines()
            .into_iter()
            .filter(|record| role_of(record) == Some(role))
            .collect()
    }

    fn signing_key(&self) -> SigningKey {
        let pem = std::fs::read_to_string(self.data_dir.join("keys").join("node-key.pem"))
            .expect("the node's signing key");
        capsule_emit_lib::keys::load_signing_key_pem(&pem).expect("parse the node's signing key")
    }

    /// The raw Ed25519 public key, lowercase hex: the `key_id` a record
    /// carries and the value the announced-key map ([`crate::peer_keys`])
    /// gives a peer.
    fn key_id(&self) -> String {
        hex::encode(self.signing_key().verifying_key().to_bytes())
    }

    fn peer_id(&self) -> String {
        let id = std::fs::read_to_string(self.data_dir.join("self-peer-id"))
            .expect("the node wrote its own peer id");
        let id = id.trim().to_string();
        assert!(!id.is_empty(), "empty self-peer-id");
        id
    }

    fn lifecycle_events(&self) -> Vec<Value> {
        read_jsonl(&self.data_dir.join("lifecycle-events.jsonl"))
    }
}

fn read_jsonl(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("a JSON line"))
        .collect()
}

fn write_jsonl(path: &Path, lines: &[Value]) {
    let text: String = lines.iter().map(|line| format!("{line}\n")).collect();
    std::fs::write(path, text).expect("write a scratch ledger");
}

fn role_of(record: &Value) -> Option<&str> {
    record
        .pointer("/model_attestation/compute_attestation/x-mesh-poc-v1/role")
        .and_then(Value::as_str)
}

fn serving_provenance_mut(record: &mut Value) -> &mut Value {
    record
        .pointer_mut("/model_attestation/compute_attestation/x-mesh-poc-v1/serving_provenance")
        .expect("a mesh half carries serving_provenance")
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).unwrap();
        }
    }
}

/// A scratch copy of `node`'s ledger directory.
fn scratch_ledger(node: &Node) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    copy_dir(&node.ledger_dir(), dir.path());
    dir
}

/// Give a changed record a fresh `capsule_id` and sign it again with `key`,
/// so it verifies on its own bytes: what the holder of that key can do.
fn reseal(record: &mut Value, key: &SigningKey) {
    let id = capsule_emit_lib::jcs::compute_capsule_id(record).expect("recompute capsule_id");
    record["capsule_id"] = json!(id);
    capsule_emit_lib::capsule::attach_producer_envelope(record, key).expect("sign the record");
}

/// Push `body` from `sender` to the requester, as a scratch copy of its
/// ledger that knows `peer_keys`. Returns the refusal reason, or `None` when
/// the record was received.
fn push_to_requester(
    requester: &Node,
    peer_keys: &str,
    sender: &str,
    body: &Value,
) -> Option<String> {
    let scratch = scratch_ledger(requester);
    let key = requester.signing_key();
    let node = Receiver {
        ledger_dir: scratch.path(),
        signing_key: &key,
        peer_keys: Some(peer_keys),
        record_at_completion_off: false,
        rejected_log_limit: MAX_REJECTED_LOG_BYTES,
    };
    let reply = receive(&node, body.to_string().as_bytes(), Some(sender), NOW)
        .expect("the receiver's own writes succeed");
    match reply.get("reason").and_then(Value::as_str) {
        Some(reason) => Some(reason.to_string()),
        None => {
            assert_eq!(reply["status"], json!("received"), "reply: {reply}");
            None
        }
    }
}

struct Run {
    requester: Node,
    provider: Node,
    exchanges: usize,
}

impl Run {
    fn from_env() -> Self {
        let exchanges = std::env::var("E2E_EXCHANGES")
            .expect("E2E_EXCHANGES must give the number of exchanges the run sent")
            .parse()
            .expect("E2E_EXCHANGES is a count");
        Self {
            requester: Node::from_env("E2E_REQUESTER_DIR"),
            provider: Node::from_env("E2E_PROVIDER_DIR"),
            exchanges,
        }
    }

    /// The provider's announced key, as the requester's operator would
    /// configure it.
    fn peer_keys(&self) -> String {
        json!({ self.provider.peer_id(): self.provider.key_id() }).to_string()
    }

    /// One of the provider's served halves, with a later record chained
    /// after it (so dropping or replacing it breaks a link).
    fn served_half_inside_the_chain(&self) -> (usize, Value) {
        let lines = self.provider.ledger_lines();
        let last_chained = lines
            .iter()
            .rposition(|line| !crate::producer::padding::is_padding(line))
            .expect("the provider's ledger has chained records");
        lines
            .into_iter()
            .enumerate()
            .find(|(at, line)| *at < last_chained && role_of(line) == Some("served"))
            .expect("a served half with a record chained after it")
    }
}

fn open_ledger(dir: &Path) -> Result<usize, LedgerError> {
    crate::producer::index::open_ledger(dir).map(|(_, report)| report.valid_entries)
}

/// Each node sealed its own record of every exchange: the requester one
/// `requested` half per exchange, the provider one `served` half. Both logs
/// reload clean through the ledger's own checks (every `capsule_id`
/// recomputes, the chain is unbroken, every signed statement verifies), and
/// every half verifies offline under its own node's key.
#[test]
#[ignore = "needs the data directories of a real two-node run (scripts/e2e-two-node.sh)"]
fn each_side_sealed_its_own_record_of_every_exchange() {
    let run = Run::from_env();
    for (node, role) in [(&run.requester, "requested"), (&run.provider, "served")] {
        open_ledger(&node.ledger_dir())
            .unwrap_or_else(|error| panic!("{role} side: the ledger does not reload: {error}"));
        let (ledger, _) = crate::producer::index::open_ledger(&node.ledger_dir()).unwrap();
        let key = node.signing_key().verifying_key();
        let key_id = node.key_id();
        let halves = node.halves(role);
        assert!(
            halves.len() >= run.exchanges,
            "{role} side sealed {} halves for {} exchanges",
            halves.len(),
            run.exchanges
        );
        for half in &halves {
            let id = half["capsule_id"].as_str().expect("capsule_id");
            let entry = ledger
                .lookup(id)
                .unwrap()
                .expect("the half is in the ledger");
            let report = capsule_emit_lib::verify::verify_offline(
                &entry.capsule,
                &entry.signed_statement,
                &key,
                None,
            );
            assert!(report.ok(), "{role} half {id}: {:?}", report.findings);
            assert_eq!(
                capsule_emit_lib::cose::verify_producer_envelope(half).as_deref(),
                Ok(key_id.as_str()),
                "{role} half {id} is signed by its own node"
            );
        }
    }
}

/// A record that did not come, unchanged, from the provider's announced key
/// is refused. The control first: the provider's real half is received.
#[test]
#[ignore = "needs the data directories of a real two-node run (scripts/e2e-two-node.sh)"]
fn a_forged_record_is_refused() {
    let run = Run::from_env();
    let keys = run.peer_keys();
    let provider = run.provider.peer_id();
    let (_, honest) = run.served_half_inside_the_chain();

    assert_eq!(
        push_to_requester(&run.requester, &keys, &provider, &honest),
        None,
        "control: the provider's real half is received"
    );

    // Changed after signing: the capsule_id no longer recomputes.
    let mut edited = honest.clone();
    edited["effect"]["response_digest"] = json!(OTHER_WEIGHTS);
    assert_eq!(
        push_to_requester(&run.requester, &keys, &provider, &edited).as_deref(),
        Some(REASON_REQUEST_MALFORMED),
        "a record changed after signing"
    );

    // Changed and signed again by a key that is not the provider's.
    let mut forged = edited.clone();
    reseal(&mut forged, &SigningKey::from_bytes(&[7; 32]));
    assert_eq!(
        push_to_requester(&run.requester, &keys, &provider, &forged).as_deref(),
        Some(REASON_SIGNATURE_UNVERIFIED),
        "a record signed by a key other than the sender's announced one"
    );

    // The real record, from a sender with no announced key.
    assert_eq!(
        push_to_requester(&run.requester, &keys, &run.requester.peer_id(), &honest).as_deref(),
        Some(REASON_SIGNATURE_UNVERIFIED),
        "a sender with no announced key"
    );
}

/// Attack A: tamper with a sealed record in the provider's own log. Changed
/// in place, its `capsule_id` no longer recomputes. Changed and sealed again
/// with the provider's own key (an insider), with a new signed statement,
/// it verifies on its own, but the next record's chain link still names the
/// original.
#[test]
#[ignore = "needs the data directories of a real two-node run (scripts/e2e-two-node.sh)"]
fn attack_a_a_tampered_record_breaks_the_providers_log() {
    let run = Run::from_env();
    let (at, honest) = run.served_half_inside_the_chain();
    let key = run.provider.signing_key();
    let mut lines = run.provider.ledger_lines();

    let naive = scratch_ledger(&run.provider);
    lines[at]["model_attestation"]["compute_attestation"]["agent_output_digest"] =
        json!(OTHER_WEIGHTS);
    write_jsonl(&naive.path().join("capsules.jsonl"), &lines);
    assert!(
        matches!(
            open_ledger(naive.path()),
            Err(LedgerError::CapsuleIdMismatch { .. })
        ),
        "A1: a record changed in place"
    );

    let insider = scratch_ledger(&run.provider);
    reseal(&mut lines[at], &key);
    let id = lines[at]["capsule_id"].as_str().unwrap().to_string();
    assert_ne!(Some(id.as_str()), honest["capsule_id"].as_str());
    let payload = capsule_emit_lib::capsule::payload_bytes(&lines[at]);
    let statement = capsule_emit_lib::cose::build_signed_statement(
        &capsule_emit_lib::cose::SignedStatementInput {
            payload: &payload,
            issuer: &run.provider.key_id(),
            subject: &id,
            content_type: "application/json",
        },
        &key,
    );
    std::fs::write(
        insider
            .path()
            .join("signed-statements")
            .join(format!("{id}.cose")),
        statement,
    )
    .unwrap();
    write_jsonl(&insider.path().join("capsules.jsonl"), &lines);
    assert!(
        matches!(
            open_ledger(insider.path()),
            Err(LedgerError::ChainBroken { .. })
        ),
        "A2: a record replaced by its own key holder"
    );
}

/// Attack B: the provider lies about who served. It changes its own served
/// half to name another node and signs it again with its own (announced)
/// key; the requester refuses it.
#[test]
#[ignore = "needs the data directories of a real two-node run (scripts/e2e-two-node.sh)"]
fn attack_b_a_half_naming_another_server_is_refused() {
    let run = Run::from_env();
    let (_, mut lying) = run.served_half_inside_the_chain();
    serving_provenance_mut(&mut lying)["served_by_node_id"] = json!(run.requester.peer_id());
    reseal(&mut lying, &run.provider.signing_key());
    assert_eq!(
        push_to_requester(
            &run.requester,
            &run.peer_keys(),
            &run.provider.peer_id(),
            &lying
        )
        .as_deref(),
        Some(REASON_SERVED_BY_MISMATCH)
    );
}

/// Attack C: drop one served exchange from the provider's log. The record
/// chained after it names the dropped one, so the log no longer reloads.
#[test]
#[ignore = "needs the data directories of a real two-node run (scripts/e2e-two-node.sh)"]
fn attack_c_a_dropped_record_breaks_the_providers_log() {
    let run = Run::from_env();
    let (at, _) = run.served_half_inside_the_chain();
    let mut lines = run.provider.ledger_lines();
    lines.remove(at);
    let scratch = scratch_ledger(&run.provider);
    write_jsonl(&scratch.path().join("capsules.jsonl"), &lines);
    assert!(matches!(
        open_ledger(scratch.path()),
        Err(LedgerError::ChainBroken { .. })
    ));
}

/// Attack D: the provider claims other model weights. The served half names
/// the weights twice (the producer's `weights_digest` and the host's
/// `serving_provenance.model.weights_digest`); a half that changes one and
/// is signed again with the provider's own key names two models, and is
/// refused.
#[test]
#[ignore = "needs the data directories of a real two-node run (scripts/e2e-two-node.sh)"]
fn attack_d_a_half_naming_other_weights_is_refused() {
    let run = Run::from_env();
    let (_, mut swapped) = run.served_half_inside_the_chain();
    let real = swapped
        .pointer("/model_attestation/compute_attestation/weights_digest/digest")
        .and_then(Value::as_str)
        .expect("the served half names the weights it loaded")
        .to_string();
    assert_ne!(real, OTHER_WEIGHTS);
    serving_provenance_mut(&mut swapped)["model"]["weights_digest"] = json!(OTHER_WEIGHTS);
    reseal(&mut swapped, &run.provider.signing_key());
    assert_eq!(
        push_to_requester(
            &run.requester,
            &run.peer_keys(),
            &run.provider.peer_id(),
            &swapped
        )
        .as_deref(),
        Some(REASON_MODEL_MISMATCH)
    );
}

/// What a run on a host that names the other side holds:
///
/// - the host named both sides: every served event on the provider names the
///   requester (`requested_by_node_id`), every routed event on the requester
///   names the provider (`served_by_node_id`), and the two sides' request and
///   response digests are the same;
/// - the requester holds the provider's record of every exchange, and its
///   Evidence page shows each as confirmed, the digests verified;
/// - the provider holds nothing from the requester: a client node's id is new
///   each start, so no key names it, and every push from it was refused
///   `signature_unverified`;
/// - a provider that lies consistently about its weights is still not caught
///   at receipt: the requester's own half names no weights to compare with.
#[test]
#[ignore = "needs the data directories of a real two-node run (scripts/e2e-two-node.sh)"]
fn the_host_names_both_sides_and_the_requester_holds_the_providers_record() {
    let run = Run::from_env();
    let exchanges = run.exchanges;
    let (requester_id, provider_id) = (run.requester.peer_id(), run.provider.peer_id());
    let terminal = |events: Vec<Value>, path: &str| -> Vec<Value> {
        events
            .into_iter()
            .filter(|e| e["phase"] == json!("terminal") && e["dispatch_path"] == json!(path))
            .collect()
    };
    let served = terminal(run.provider.lifecycle_events(), "raw_proxy");
    let routed = terminal(run.requester.lifecycle_events(), "remote_mesh");
    assert!(served.len() >= exchanges && routed.len() >= exchanges);
    for event in &served {
        assert_eq!(
            event["serving_provenance"]["requested_by_node_id"],
            json!(requester_id),
            "the provider's event names the requester"
        );
    }
    for event in &routed {
        assert_eq!(
            event["serving_provenance"]["served_by_node_id"],
            json!(provider_id),
            "the requester's event names the provider"
        );
    }
    let digests = |events: &[Value]| -> std::collections::BTreeSet<(String, String)> {
        events
            .iter()
            .map(|e| {
                (
                    e["request_digest"].as_str().unwrap_or_default().to_string(),
                    e["response_digest"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                )
            })
            .collect()
    };
    assert_eq!(
        digests(&served),
        digests(&routed),
        "both sides saw the same exchanges"
    );

    let held = read_jsonl(&run.requester.ledger_dir().join("received-capsules.jsonl"));
    assert_eq!(
        held.len(),
        exchanges,
        "the requester holds the provider's record of each exchange"
    );
    let pane = crate::evidence_panes::build_pane_json("pane-b", &run.requester.ledger_dir(), None)
        .expect("the requester's peers pane");
    let confirmed: Vec<&Value> = pane["rows"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|row| row["confirmed_siblings"].as_array().into_iter().flatten())
        .collect();
    assert_eq!(
        confirmed.len(),
        exchanges,
        "each exchange is confirmed on the requester's page"
    );
    for sibling in confirmed {
        assert_eq!(
            sibling["digest_match"]["state"],
            json!("verified"),
            "{sibling}"
        );
    }

    assert!(
        read_jsonl(&run.provider.ledger_dir().join("received-capsules.jsonl")).is_empty(),
        "the provider holds nothing from a client node"
    );
    let refused = read_jsonl(
        &run.provider
            .ledger_dir()
            .join(crate::record_push_receive::REJECTED_PUSHES_FILENAME),
    );
    assert_eq!(
        refused.len(),
        exchanges,
        "every push from the requester was refused"
    );
    for line in &refused {
        assert_eq!(line["reason"], json!(REASON_SIGNATURE_UNVERIFIED), "{line}");
        assert_eq!(
            line["claimed_sender_peer_id"],
            json!(requester_id),
            "{line}"
        );
    }

    for half in run.requester.halves("requested") {
        assert!(
            crate::record_push_receive::weights_claims(&half).is_empty(),
            "the requester's own half names weights: a consistent liar can now be caught at receipt; check this note"
        );
    }
}
