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
            observations.extend(log.lines().filter_map(|line| observation(stream, line)));
        }
    }
    json!({
        "status": if observations.is_empty() { "unavailable" } else { "observed" },
        "observations": observations,
    })
}

fn observation(stream: &str, line: &str) -> Option<Value> {
    // Retain each observation: warmup and model initialization can emit multiple
    // context shapes, so selecting the last line would invent a final capacity.
    for field in ["n_ctx_per_seq", "n_seq_max", "n_ctx"] {
        let Some((_, suffix)) = line.split_once(field) else {
            continue;
        };
        let Some(value) = suffix.trim_start().strip_prefix('=') else {
            continue;
        };
        let digits: String = value
            .trim_start()
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        if let Ok(value) = digits.parse::<u64>() {
            return Some(json!({"source":stream,"field":field,"value":value}));
        }
    }
    None
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
            "llama_context: n_ctx = 65536\nllama_context: n_ctx_per_seq = 16384\nllama_context: n_seq_max = 4\n",
        )
        .unwrap();
        let evidence = read(dir.path());
        assert_eq!(evidence["observations"][1]["field"], "n_ctx_per_seq");
        assert_eq!(evidence["observations"][1]["value"], 16384);
        assert_eq!(evidence["observations"].as_array().unwrap().len(), 3);
        assert!(observation("stdout", "n_ctx_train = 262144").is_none());
    }
}
