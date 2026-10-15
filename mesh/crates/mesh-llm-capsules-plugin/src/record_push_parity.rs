//! The record-push parity run: the repository's parity corpus
//! (`vectors/parity/`) through [`crate::record_push_receive::receive`], every
//! answer held to the Python reference's golden answer (or, for the cases in
//! `intended_differences.json`, to the stricter answer listed there). The
//! answer format is specified in `vectors/parity/README.md`.
//!
//! Set `RECORD_PUSH_PARITY_OUT=<path>` to also write this implementation's
//! answers, to compare them case by case with another implementation's.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use base64::Engine;
use ed25519_dalek::{Signature, SigningKey, VerifyingKey};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::record_push_receive::{receive, Receiver};

pub(crate) fn parity_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("vectors/parity")
}

/// Read one of the harness's own files. The golden answers hold a record
/// nested at the parser's limit inside the answer's own nesting, so this
/// reader (and only this one) parses without a depth limit.
pub(crate) fn read_json(path: &Path) -> Value {
    use serde::Deserialize;
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut deserializer = serde_json::Deserializer::from_str(&text);
    deserializer.disable_recursion_limit();
    Value::deserialize(&mut deserializer)
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// Canonical JSON (sorted keys, compact) for comparing answers by value.
pub(crate) fn canonical(value: &Value) -> String {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let ordered: BTreeMap<&String, Value> =
                    map.iter().map(|(k, v)| (k, sorted(v))).collect();
                Value::Object(ordered.into_iter().map(|(k, v)| (k.clone(), v)).collect())
            }
            Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }
    sorted(value).to_string()
}

pub(crate) fn body_bytes(push: &Value) -> Vec<u8> {
    match (push.get("body"), push.get("body_b64")) {
        (Some(Value::String(text)), _) => text.as_bytes().to_vec(),
        (_, Some(Value::String(b64))) => base64::engine::general_purpose::STANDARD
            .decode(b64)
            .expect("corpus body_b64 is base64"),
        _ => panic!("a push carries body or body_b64"),
    }
}

pub(crate) fn line_counts(dir: &Path) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for entry in std::fs::read_dir(dir).expect("read ledger dir").flatten() {
        if entry.path().is_file() {
            let text = std::fs::read_to_string(entry.path()).unwrap_or_default();
            counts.insert(
                entry.file_name().to_string_lossy().into_owned(),
                text.lines().count(),
            );
        }
    }
    counts
}

pub(crate) fn appended(dir: &Path, before: &BTreeMap<String, usize>) -> Value {
    let mut out = Map::new();
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .expect("read ledger dir")
        .flatten()
        .collect();
    names.sort_by_key(|e| e.file_name());
    for entry in names {
        if !entry.path().is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(entry.path()).unwrap_or_default();
        let skip = before.get(&name).copied().unwrap_or(0);
        let lines: Vec<Value> = text
            .lines()
            .skip(skip)
            .map(|l| serde_json::from_str(l).expect("every stored line is JSON"))
            .collect();
        if !lines.is_empty() {
            out.insert(name, Value::Array(lines));
        }
    }
    Value::Object(out)
}

/// The reply as the harness compares it: a refusal's signature is checked
/// against this node's key rather than copied.
pub(crate) fn normalize_reply(
    reply: &Value,
    body: &[u8],
    node_key: &VerifyingKey,
    now: &str,
) -> Value {
    let Some(reason) = reply.get("reason") else {
        return reply.clone();
    };
    let text = |k: &str| {
        reply
            .get(k)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    let signed = (|| -> Option<bool> {
        if text("key_id") != hex::encode(node_key.to_bytes()) {
            return Some(false);
        }
        let sig: [u8; 64] = hex::decode(text("sig")).ok()?.try_into().ok()?;
        let body = json!({"issued_at": text("issued_at"), "reason": reason, "request_digest": text("request_digest")});
        Some(
            node_key
                .verify_strict(canonical(&body).as_bytes(), &Signature::from_bytes(&sig))
                .is_ok(),
        )
    })()
    .unwrap_or(false);
    json!({
        "refused": reason,
        "status": reply.get("status").cloned().unwrap_or(Value::Null),
        "signed_by_node": signed,
        "request_digest_is_body_sha256": text("request_digest") == hex::encode(Sha256::digest(body)),
        "issued_at_is_now": text("issued_at") == now,
    })
}

fn run_case(case: &Value, now: &str, seed: u8) -> Vec<Value> {
    let scratch = tempfile::tempdir().expect("tempdir");
    let ledger_dir = scratch.path().join("ledger");
    std::fs::create_dir_all(&ledger_dir).expect("ledger dir");
    let node = &case["node"];
    let own: String = node["own_records"]
        .as_array()
        .expect("own_records")
        .iter()
        .map(|r| r.to_string() + "\n")
        .collect();
    std::fs::write(ledger_dir.join("capsules.jsonl"), own).expect("own records");
    let signing_key = SigningKey::from_bytes(&[seed; 32]);
    let receiver = Receiver {
        ledger_dir: &ledger_dir,
        signing_key: &signing_key,
        peer_keys: node["peer_keys_env"].as_str(),
        record_at_completion_off: node["record_at_completion"].as_str() == Some("off"),
        rejected_log_limit: crate::record_push_receive::MAX_REJECTED_LOG_BYTES,
    };
    case["pushes"]
        .as_array()
        .expect("pushes")
        .iter()
        .map(|push| {
            let body = body_bytes(push);
            let before = line_counts(&ledger_dir);
            let reply = receive(&receiver, &body, push["sender"].as_str(), now)
                .expect("local writes succeed");
            json!({
                "reply": normalize_reply(&reply, &body, &signing_key.verifying_key(), now),
                "appended": appended(&ledger_dir, &before),
            })
        })
        .collect()
}

#[test]
fn the_receiver_answers_the_corpus_as_recorded() {
    let dir = parity_dir();
    let corpus = read_json(&dir.join("corpus/record_push.json"));
    let golden = read_json(&dir.join("golden/record_push.json"));
    let intended = read_json(&dir.join("intended_differences.json"));
    let now = corpus["now"].as_str().expect("now");

    let mut answers = Map::new();
    let mut differs = Vec::new();
    for (i, case) in corpus["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .enumerate()
    {
        let name = case["name"].as_str().expect("name");
        let got = Value::Array(run_case(case, now, u8::try_from(i % 250).unwrap_or(0) + 1));
        let expected = intended["cases"]
            .get(name)
            .map(|entry| &entry["answers"])
            .unwrap_or(&golden["answers"][name]);
        if canonical(&got) != canonical(expected) {
            differs.push(format!(
                "{name}\n  expected {}\n  got      {}",
                canonical(expected),
                canonical(&got)
            ));
        }
        answers.insert(name.to_string(), got);
    }

    if let Ok(out) = std::env::var("RECORD_PUSH_PARITY_OUT") {
        let doc = json!({"v": 1, "path": corpus["path"], "implementation": "rust-plugin", "answers": answers});
        std::fs::write(&out, doc.to_string() + "\n").expect("write answers");
    }
    assert!(
        differs.is_empty(),
        "{} case(s) differ:\n{}",
        differs.len(),
        differs.join("\n")
    );
}
