//! A referee's signed verdict, on the chains it concerns.
//!
//! The referee holds each verdict it issues beside its ledger
//! (`issued-adjudications.jsonl`); a node the verdict concerns holds the ones
//! delivered to it (`received-adjudications.jsonl`, `crate::referee::hold`).
//! Neither file is a chain: each node's chain carries its own
//! `adjudication_issued` / `adjudication_received` record citing the verdict.
//!
//! Delivery is a courier's push over the record-push stream
//! (`deliver_adjudication`); when the receiver refuses it, the courier seals
//! an `adjudication_ack_refused` record, so its own chain shows the decline.
use std::sync::Arc;

use mesh_llm_plugin::{PluginContext, PluginError, PluginResult};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::capsule_emit::CapsuleState;

/// The member of a record-push body that marks a delivered verdict.
pub const DELIVERY_MARKER: &str = "adjudication_delivery";
pub const DELIVER_OPERATION: &str = "deliver_adjudication";

pub fn is_delivery_body(body: &Value) -> bool {
    body.get(DELIVERY_MARKER).is_some()
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeliverAdjudicationArgs {
    /// The node the verdict goes to.
    pub peer_id: String,
    /// The referee's signed verdict record, as it issued it.
    pub verdict_capsule: Value,
}

/// The `deliver_adjudication` tool: push a referee's signed verdict to a node
/// it concerns, as its courier. The receiver's answer comes back unchanged.
/// A refusal is sealed on this node's own chain (`adjudication_ack_refused`)
/// before the answer is returned.
pub async fn deliver(
    args: DeliverAdjudicationArgs,
    context: &mut PluginContext<'_>,
    capsules: Arc<CapsuleState>,
    self_id: Option<String>,
) -> PluginResult<Value> {
    let Some(self_id) = self_id else {
        return Err(PluginError::invalid_request(
            "this node does not know its own peer id yet, so it cannot name itself as the courier",
        ));
    };
    push_verdict(
        context,
        &capsules,
        &args.peer_id,
        &self_id,
        &args.verdict_capsule,
    )
    .await
    .map_err(|error| PluginError::internal(error.to_string()))
}

/// Push `verdict_capsule` to `peer_id` as its courier (`self_id`) and return
/// the receiver's answer; a refusal is sealed on this node's chain first
/// (`adjudication_ack_refused`).
pub(crate) async fn push_verdict(
    context: &mut PluginContext<'_>,
    capsules: &CapsuleState,
    peer_id: &str,
    self_id: &str,
    verdict_capsule: &Value,
) -> anyhow::Result<Value> {
    let verdict_id = verdict_capsule
        .get("capsule_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| anyhow::anyhow!("verdict_capsule has no capsule_id"))?
        .to_string();
    let body = json!({ DELIVERY_MARKER: 1, "verdict_capsule": verdict_capsule });
    let reply = crate::record_push_bridge::send_push(context, peer_id, self_id, &body).await?;
    if let Some(reason) = reply.get("reason").and_then(Value::as_str) {
        let digest = hex::encode(Sha256::digest(reply.to_string().as_bytes()));
        let refused = crate::producer::capsule::RefusedDelivery {
            verdict_capsule_id: &verdict_id,
            refused_by: peer_id,
            reason,
            refusal_digest: &digest,
            refusal_key_id: reply.get("key_id").and_then(Value::as_str),
            refused_at: &crate::producer::timestamp::utc_now_iso8601(),
        };
        capsules.emit_adjudication_ack_refused(&refused)?;
    }
    Ok(reply)
}

/// The held verdict files beside the ledger, as an earlier run may have
/// left them.
pub const ISSUED_ADJUDICATIONS_FILENAME: &str = "issued-adjudications.jsonl";
pub const RECEIVED_ADJUDICATIONS_FILENAME: &str = "received-adjudications.jsonl";

fn held_verdict(ledger_dir: &std::path::Path, verdict_capsule_id: &str) -> Option<Value> {
    for file in [
        ISSUED_ADJUDICATIONS_FILENAME,
        RECEIVED_ADJUDICATIONS_FILENAME,
    ] {
        let Ok(text) = std::fs::read_to_string(ledger_dir.join(file)) else {
            continue;
        };
        for line in text.lines() {
            let Ok(held) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            if held.get("verdict_capsule_id").and_then(Value::as_str) == Some(verdict_capsule_id) {
                if let Some(capsule) = held.get("verdict_capsule") {
                    return Some(capsule.clone());
                }
            }
        }
    }
    None
}

/// `issued` / `received` when this node's chain records the verdict.
fn recorded_as(ledger_dir: &std::path::Path, verdict_capsule_id: &str) -> Option<&'static str> {
    let text = std::fs::read_to_string(ledger_dir.join("capsules.jsonl")).ok()?;
    for line in text.lines() {
        let Ok(record) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let Some(attestation) = record.pointer("/model_attestation/compute_attestation") else {
            continue;
        };
        for (block, kind) in [
            (
                crate::producer::capsule::ADJUDICATION_ISSUED_BLOCK,
                "issued",
            ),
            (
                crate::producer::capsule::ADJUDICATION_RECEIVED_BLOCK,
                "received",
            ),
        ] {
            if attestation
                .pointer(&format!("/{block}/verdict_capsule_id"))
                .and_then(Value::as_str)
                == Some(verdict_capsule_id)
            {
                return Some(kind);
            }
        }
    }
    None
}

/// `http/ledger/verdict?capsule_id=`: one held verdict. `verify_ok` is true
/// only when it verifies as signed, with its announced key, by the referee
/// it names (`crate::referee::hold::verdict_facts`). The signature's own
/// check (`signed_by_key_id` / `signature_error`) and how this node's chain
/// records it (`recorded_as`) are reported beside it. `null` capsule when
/// none is held.
pub fn verdict_json(ledger_dir: &std::path::Path, verdict_capsule_id: &str) -> Value {
    let peer_keys = crate::peer_keys::registry();
    verdict_json_with(ledger_dir, verdict_capsule_id, peer_keys.as_deref())
}

fn verdict_json_with(
    ledger_dir: &std::path::Path,
    verdict_capsule_id: &str,
    peer_keys: Option<&str>,
) -> Value {
    let Some(capsule) = held_verdict(ledger_dir, verdict_capsule_id) else {
        return json!({ "capsule": null, "signed_by_key_id": null, "verify_ok": false });
    };
    let signature = crate::producer::cose::verify_producer_envelope(&capsule);
    let recorded = recorded_as(ledger_dir, verdict_capsule_id);
    let referee = capsule
        .pointer("/model_attestation/compute_attestation/adjudication/referee_node_id")
        .cloned()
        .unwrap_or(Value::Null);
    let verified = crate::referee::hold::verdict_facts(&capsule, peer_keys).is_some();
    json!({
        "signed_by_key_id": signature.as_ref().ok(),
        "verify_ok": verified,
        "signature_error": signature.as_ref().err(),
        "recorded_as": recorded,
        "referee_node_id": referee,
        "capsule": capsule,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/referee-verdict.json")).unwrap()
    }

    #[test]
    fn a_verdict_the_reference_referee_signed_verifies_here() {
        let fx = fixture();
        let key = crate::producer::cose::verify_producer_envelope(&fx["verdict_capsule"])
            .expect("verifies");
        assert_eq!(json!(key), fx["referee_key_id"]);
    }

    fn held_dir(recorded: bool) -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let capsule = fixture()["verdict_capsule"].clone();
        let id = capsule["capsule_id"].as_str().unwrap().to_string();
        let held = json!({ "verdict_capsule_id": id, "verdict_capsule": capsule });
        std::fs::write(
            dir.path().join(RECEIVED_ADJUDICATIONS_FILENAME),
            format!("{held}\n"),
        )
        .unwrap();
        if recorded {
            let record = json!({ "model_attestation": { "compute_attestation": {
                "adjudication_received": { "verdict_capsule_id": id } } } });
            std::fs::write(dir.path().join("capsules.jsonl"), format!("{record}\n")).unwrap();
        }
        (dir, id)
    }

    #[test]
    fn a_held_verdict_verifies_only_under_its_referees_announced_key() {
        let (dir, id) = held_dir(true);
        let out = verdict_json_with(dir.path(), &id, None);
        assert_eq!(
            out["verify_ok"],
            json!(false),
            "no announced key for its referee"
        );
        assert_eq!(out["recorded_as"], json!("received"));
        assert_eq!(out["signed_by_key_id"], fixture()["referee_key_id"]);

        // A verdict from the referee corpus, with the corpus's announced keys.
        let corpus = crate::record_push_parity::read_json(
            &crate::referee::parity::referee_dir().join("corpus/hold.json"),
        );
        let case = &corpus["cases"][0];
        let body: Value =
            serde_json::from_str(case["pushes"][0]["body"].as_str().unwrap()).unwrap();
        let capsule = body["verdict_capsule"].clone();
        let id = capsule["capsule_id"].as_str().unwrap().to_string();
        let dir = tempfile::tempdir().unwrap();
        let held = json!({ "verdict_capsule_id": id, "verdict_capsule": capsule });
        std::fs::write(
            dir.path().join(RECEIVED_ADJUDICATIONS_FILENAME),
            format!("{held}\n"),
        )
        .unwrap();
        let keys = case["node"]["peer_keys_env"].as_str();
        assert_eq!(
            verdict_json_with(dir.path(), &id, keys)["verify_ok"],
            json!(true)
        );
        assert_eq!(
            verdict_json_with(dir.path(), &id, Some("{}"))["verify_ok"],
            json!(false),
            "the same record, with its referee's key not announced"
        );
    }

    #[test]
    fn an_altered_held_verdict_does_not_verify() {
        let (dir, id) = held_dir(true);
        let path = dir.path().join(RECEIVED_ADJUDICATIONS_FILENAME);
        let altered = std::fs::read_to_string(&path)
            .unwrap()
            .replace("contradicted:", "contradicted:x");
        std::fs::write(&path, altered).unwrap();
        let out = verdict_json(dir.path(), &id);
        assert_eq!(out["verify_ok"], json!(false));
        assert!(out["signature_error"].is_string());
    }

    #[test]
    fn an_unknown_verdict_is_null_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            verdict_json(dir.path(), &"0".repeat(64))["capsule"],
            Value::Null
        );
    }
}
