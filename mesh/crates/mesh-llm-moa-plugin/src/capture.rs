//! Opt-in capture of committee turns as local distillation pairs.
//!
//! A node that runs the MoA committee sees, for every answer turn, the small
//! workers' drafts *and* the synthesized answer the caller actually received.
//! That pairing is the training signal for a node that learns from its own
//! traffic (`mesh-llm#2575`): the drafts are what the local model said, the
//! final answer is what it should have said.
//!
//! Capture is off unless `MESH_MOA_CAPTURE_PATH` names a file. Records are
//! appended as JSON lines, raw and unredacted, to that file only. Nothing is
//! gossiped or sent to a peer. This is a research surface for an operator's
//! own node; redaction and filtering belong to the training step that reads
//! the file.

use mesh_mixture_of_agents as moa;
use serde_json::{Value, json};
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

pub const CAPTURE_PATH_ENV: &str = "MESH_MOA_CAPTURE_PATH";

fn capture_path() -> Option<&'static PathBuf> {
    static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    PATH.get_or_init(|| {
        std::env::var_os(CAPTURE_PATH_ENV)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    })
    .as_ref()
}

/// Build one capture record, or `None` when the turn carries no drafts (tool
/// turns, tool-result turns, failures, single-worker passthrough).
pub fn record(request: &Value, result: &moa::TurnResult) -> Option<Value> {
    if result.drafts.is_empty() {
        return None;
    }
    let final_message = result
        .response_body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .cloned()?;
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    Some(json!({
        "ts": ts,
        "messages": request.get("messages").cloned().unwrap_or(Value::Array(Vec::new())),
        "drafts": result.drafts.iter().map(|draft| json!({
            "model": draft.model,
            "role": draft.role.label(),
            "confidence": draft.confidence,
            "text": draft.text,
        })).collect::<Vec<_>>(),
        "final": final_message,
        "reducer_used": result.reducer_used,
        "turn_kind": result.turn_kind.label(),
        "elapsed_ms": result.elapsed_ms,
        "workers": result.worker_summaries.iter().map(|w| json!({
            "model": w.model,
            "role": w.role.label(),
            "succeeded": w.succeeded,
            "elapsed_ms": w.elapsed_ms,
        })).collect::<Vec<_>>(),
    }))
}

/// Append the turn to the capture file when capture is enabled. Never fails
/// the turn: a capture problem is reported once per occurrence via tracing.
pub fn capture_turn(request: &Value, result: &moa::TurnResult) {
    let Some(path) = capture_path() else {
        return;
    };
    let Some(record) = record(request, result) else {
        return;
    };
    if let Err(error) = append_line(path, &record) {
        tracing::warn!(path = %path.display(), %error, "moa capture: append failed");
    }
}

fn append_line(path: &PathBuf, record: &Value) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let mut line = serde_json::to_vec(record)?;
    line.push(b'\n');
    file.write_all(&line)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_mixture_of_agents::{Draft, TurnKind, TurnResult, WorkerSummary};

    fn turn(drafts: Vec<Draft>) -> TurnResult {
        TurnResult {
            response_body: json!({
                "choices": [{"message": {"role": "assistant", "content": "synthesized"}}]
            }),
            worker_summaries: vec![WorkerSummary {
                model: "small".into(),
                role: moa::WorkerRole::Generalist,
                succeeded: true,
                elapsed_ms: 10,
                output_kind: None,
                confidence: None,
            }],
            reducer_used: true,
            reducer_attempts: 1,
            turn_kind: TurnKind::Fanout,
            elapsed_ms: 42,
            drafts,
        }
    }

    #[test]
    fn answer_turn_pairs_drafts_with_final() {
        let request = json!({"messages": [{"role": "user", "content": "hi"}]});
        let result = turn(vec![Draft {
            model: "small".into(),
            role: moa::WorkerRole::Generalist,
            text: "draft".into(),
            confidence: 0.5,
        }]);
        let record = record(&request, &result).expect("record");
        assert_eq!(record["drafts"][0]["text"], "draft");
        assert_eq!(record["final"]["content"], "synthesized");
        assert_eq!(record["messages"][0]["content"], "hi");
        assert_eq!(record["reducer_used"], true);
        assert_eq!(record["turn_kind"], "fanout");
    }

    #[test]
    fn turns_without_drafts_are_skipped() {
        let request = json!({"messages": []});
        assert!(record(&request, &turn(Vec::new())).is_none());
    }
}
