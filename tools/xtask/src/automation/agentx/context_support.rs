//! Keep context-limited workloads visible without publishing unequal-work metrics.
use crate::command::DynResult;
use serde_json::{Value, json};
use std::path::Path;

pub(super) fn evidence(output: &Path) -> Option<Value> {
    read(output).ok()
}

fn read(output: &Path) -> DynResult<Value> {
    let lifecycle: Value = serde_json::from_slice(&std::fs::read(output.join("lifecycle.json"))?)?;
    if lifecycle["clean"] != true || lifecycle["cleanup_complete"] != true {
        return Err("context support requires clean process completion".into());
    }
    let export_path = output.join("upstream/profile_export_aiperf.json");
    let export: Value = serde_json::from_slice(&std::fs::read(&export_path)?)?;
    let overflows = request_count(&export, "context_overflow_count")?;
    if export["was_cancelled"] != false
        || export["metadata"]["submission_valid"] != false
        || optional_errors(&export)? != 0
    {
        return Err("export contains other failures or invalid provenance".into());
    }
    let phase = super::phase_outcome::context_limited(
        &std::fs::read_to_string(output.join("upstream/logs/aiperf.log"))?,
        request_count(&export, "request_count")?,
        overflows,
    )?;
    Ok(json!({
        "reason":"recorded requests exceed the engine's context capacity; traces were not truncated",
        "context_overflow_requests":overflows,
        "phase_outcome":phase,
        "export_sha256":crate::product::digest::file_sha256(&export_path).map_err(|e|e.error)?,
        "export_path":"upstream/profile_export_aiperf.json",
        "trace_diagnostics_path":"upstream/logs/aiperf.log"
    }))
}

fn optional_errors(export: &Value) -> DynResult<u64> {
    if export["error_request_count"].is_null() {
        Ok(0)
    } else {
        request_count(export, "error_request_count")
    }
}

fn request_count(export: &Value, name: &str) -> DynResult<u64> {
    let value = export[name]["avg"]
        .as_f64()
        .ok_or("missing request count")?;
    if export[name]["unit"] != "requests"
        || !value.is_finite()
        || value < 0.0
        || value.fract() != 0.0
        || value > (1_u64 << 53) as f64
    {
        return Err("invalid request count".into());
    }
    Ok(value as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(output: &Path, export: &Value, log: &str) {
        std::fs::create_dir_all(output.join("upstream/logs")).unwrap();
        std::fs::write(
            output.join("lifecycle.json"),
            r#"{"clean":true,"cleanup_complete":true}"#,
        )
        .unwrap();
        std::fs::write(
            output.join("upstream/profile_export_aiperf.json"),
            export.to_string(),
        )
        .unwrap();
        std::fs::write(output.join("upstream/logs/aiperf.log"), log).unwrap();
    }

    #[test]
    fn identifies_context_only_rejection_without_publishing_throughput() {
        let dir = tempfile::tempdir().unwrap();
        let export = json!({"was_cancelled":false,"metadata":{"submission_valid":false},
            "context_overflow_count":{"unit":"requests","avg":1.0},
            "request_count":{"unit":"requests","avg":4.0}});
        let log = "Phase profiling (profiling) complete | completed=5, cancelled=0, errors=1 | elapsed=151.96s";
        fixture(dir.path(), &export, log);
        let record = evidence(dir.path()).unwrap();
        assert_eq!(record["context_overflow_requests"], 1);
        assert!(record.get("output_tok_s").is_none());
        for (field, value) in [
            ("was_cancelled", json!(true)),
            ("error_request_count", json!({"unit":"requests","avg":1})),
            ("error_request_count", json!({"unit":"requests","avg":-1})),
            ("context_overflow_count", json!({"unit":"requests","avg":0})),
            (
                "context_overflow_count",
                json!({"unit":"requests","avg":1.5}),
            ),
            ("context_overflow_count", Value::Null),
        ] {
            let mut invalid = export.clone();
            invalid[field] = value;
            fixture(dir.path(), &invalid, log);
            assert!(evidence(dir.path()).is_none(), "{field}");
        }
        for invalid in [
            log.replace("cancelled=0", "cancelled=1"),
            log.replace("errors=1", "errors=2"),
            log.replace("completed=5", "completed=4"),
            String::new(),
        ] {
            fixture(dir.path(), &export, &invalid);
            assert!(evidence(dir.path()).is_none());
        }
        fixture(dir.path(), &export, log);
        std::fs::write(
            dir.path().join("lifecycle.json"),
            r#"{"clean":false,"cleanup_complete":true}"#,
        )
        .unwrap();
        assert!(evidence(dir.path()).is_none());
    }
}
