//! Preserve native allocation evidence without equating offered load with slots.
use serde_json::{Value, json};
use std::path::Path;

pub(super) fn policy(engine: &str) -> &'static str {
    if engine == "skippy" {
        "engine_default"
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
