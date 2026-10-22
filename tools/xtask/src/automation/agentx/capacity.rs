//! Preserve native allocation evidence without equating offered load with slots.
use super::{Profile, trace_context};
use crate::command::DynResult;
use serde_json::{Value, json};
use std::path::Path;

pub(super) fn policy(engine: &str, profile: Profile) -> &'static str {
    if engine == "skippy" {
        "engine_default"
    } else if profile == Profile::Comparison {
        "common_128k_per_slot_explicit_parallel"
    } else {
        "auto_context_explicit_parallel"
    }
}

pub(super) fn read(output: &Path) -> Value {
    let mut observations = Vec::new();
    for stream in ["server.stdout.log", "server.stderr.log"] {
        if let Ok(log) = std::fs::read_to_string(output.join(stream)) {
            observations.extend(
                log.lines()
                    .flat_map(|line| observations_in_line(stream, line)),
            );
        }
    }
    json!({
        "status": if observations.is_empty() { "unavailable" } else { "observed" },
        "observations": observations,
    })
}

/// The CLI reports the effective stage configuration before model load. Keep
/// this distinct from native allocation observations, which may be unavailable.
pub(super) fn read_configured(output: &Path) -> Value {
    let observations = std::fs::read_to_string(output.join("server.stdout.log"))
        .ok()
        .map(|log| {
            log.lines()
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .filter(|event| event["type"] == "serving_capacity")
                .filter_map(|event| {
                    let context_tokens_per_lane =
                        event["data"]["context_tokens_per_lane"].as_u64()?;
                    let lanes = event["data"]["lanes"].as_u64()?;
                    (context_tokens_per_lane > 0 && lanes > 0).then(|| {
                        json!({
                            "source":"server.stdout.log:serving_capacity",
                            "context_tokens_per_lane":context_tokens_per_lane,
                            "lanes":lanes,
                        })
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    json!({
        "status": if observations.is_empty() { "unavailable" } else { "observed" },
        "observations": observations,
    })
}

pub(super) fn read_verified(
    output: &Path,
    engine: &str,
    profile: Profile,
    concurrency: u32,
) -> DynResult<Value> {
    let evidence = read(output);
    if engine == "llama" && profile == Profile::Comparison {
        let observations = evidence["observations"]
            .as_array()
            .ok_or("missing native context observations")?;
        for (field, expected) in [
            ("n_slots", u64::from(concurrency)),
            ("n_ctx_slot", trace_context::LIMIT),
        ] {
            let values: Vec<_> = observations
                .iter()
                .filter(|observation| observation["field"] == field)
                .filter_map(|observation| observation["value"].as_u64())
                .collect();
            if values.is_empty() || values.iter().any(|&value| value != expected) {
                return Err(format!(
                    "llama comparison {field} must be {expected}; observed {values:?}"
                )
                .into());
            }
        }
    }
    Ok(evidence)
}

fn observations_in_line(stream: &str, line: &str) -> Vec<Value> {
    // Retain each observation: warmup and model initialization can emit multiple
    // context shapes, so selecting the last line would invent a final capacity.
    // Server summaries can put several fields on one line. Keep the original
    // field names: a per-slot limit is not an observed total KV allocation.
    [
        "n_ctx_per_seq",
        "n_ctx_seq",
        "n_seq_max",
        "n_ctx",
        "n_slots",
        "n_ctx_slot",
    ]
    .into_iter()
    .filter_map(|field| {
        let (_, suffix) = line.split_once(field)?;
        let value = suffix.trim_start().strip_prefix('=')?;
        let digits: String = value
            .trim_start()
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        let value = digits.parse::<u64>().ok()?;
        Some(json!({"source":stream,"field":field,"value":value}))
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_capacity_policy_distinguishes_matched_context() {
        assert_eq!(
            policy("llama", Profile::Comparison),
            "common_128k_per_slot_explicit_parallel"
        );
        assert_eq!(
            policy("llama", Profile::Smoke),
            "auto_context_explicit_parallel"
        );
        assert_eq!(policy("skippy", Profile::Comparison), "engine_default");
    }

    #[test]
    fn configured_capacity_is_separate_from_native_evidence() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("server.stdout.log"),
            concat!(
                "{\"type\":\"serving_capacity\",\"data\":{\"context_tokens_per_lane\":131072,\"lanes\":4}}\n",
                "{\"type\":\"status\",\"data\":{\"message\":\"ready\"}}\n",
            ),
        )
        .unwrap();
        let configured = read_configured(dir.path());
        assert_eq!(configured["status"], "observed");
        assert_eq!(
            configured["observations"][0]["context_tokens_per_lane"],
            131072
        );
        assert_eq!(configured["observations"][0]["lanes"], 4);
        assert_eq!(read(dir.path())["status"], "unavailable");
    }

    #[test]
    fn comparison_requires_observed_matched_llama_slots() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_verified(dir.path(), "llama", Profile::Comparison, 4).is_err());
        std::fs::write(
            dir.path().join("server.stderr.log"),
            "srv load_model: initializing, n_slots = 4, n_ctx_slot = 131072, kv_unified = 'false'",
        )
        .unwrap();
        assert!(read_verified(dir.path(), "llama", Profile::Comparison, 4).is_ok());
        assert!(read_verified(dir.path(), "llama", Profile::Comparison, 2).is_err());
        std::fs::write(
            dir.path().join("server.stderr.log"),
            "srv load_model: initializing, n_slots = 4, n_ctx_slot = 262144, kv_unified = 'false'",
        )
        .unwrap();
        assert!(read_verified(dir.path(), "llama", Profile::Comparison, 4).is_err());
        assert!(read_verified(dir.path(), "llama", Profile::Smoke, 4).is_ok());
        assert!(read_verified(dir.path(), "skippy", Profile::Comparison, 4).is_ok());
    }

    #[test]
    fn retains_context_shapes_without_claiming_offered_load_is_capacity() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read(dir.path())["status"], "unavailable");
        std::fs::write(
            dir.path().join("server.stderr.log"),
            "llama_context: n_ctx = 65536\nllama_context: n_ctx_per_seq = 16384\nllama_context: n_seq_max = 4\nllama_context: n_ctx_seq = 16384\n",
        )
        .unwrap();
        let evidence = read(dir.path());
        assert_eq!(evidence["observations"][1]["field"], "n_ctx_per_seq");
        assert_eq!(evidence["observations"][1]["value"], 16384);
        assert_eq!(evidence["observations"][3]["field"], "n_ctx_seq");
        assert_eq!(evidence["observations"][3]["value"], 16384);
        assert_eq!(evidence["observations"].as_array().unwrap().len(), 4);
        assert!(observations_in_line("stdout", "n_ctx_train = 262144").is_empty());
    }

    #[test]
    fn retains_each_field_in_server_slot_summaries() {
        let observations = observations_in_line(
            "server.stderr.log",
            "srv load_model: initializing, n_slots = 4, n_ctx_slot = 262144, kv_unified = 'false'",
        );
        assert_eq!(
            observations,
            vec![
                json!({"source":"server.stderr.log","field":"n_slots","value":4}),
                json!({"source":"server.stderr.log","field":"n_ctx_slot","value":262144}),
            ]
        );
        assert!(observations_in_line("stderr", "n_slots = auto, n_ctx_slot = -1").is_empty());
    }
}
