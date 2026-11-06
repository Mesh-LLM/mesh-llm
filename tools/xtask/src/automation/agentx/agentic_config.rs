//! Explicit AgentX replay configuration shared by fixed-session profiles.
use super::{Prepared, trace_context};
use crate::command::DynResult;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

pub(super) fn tokenizer_hub_cache(root: &Path) -> PathBuf {
    let cache = std::env::var_os("HF_HUB_CACHE").map_or_else(
        || {
            std::env::var_os("HF_HOME")
                .map_or_else(|| root.join(".cache/huggingface"), PathBuf::from)
                .join("hub")
        },
        PathBuf::from,
    );
    if cache.is_absolute() {
        cache
    } else {
        root.join(cache)
    }
}

pub(super) fn tokenizer_snapshot(root: &Path, model: &super::Model) -> PathBuf {
    tokenizer_hub_cache(root)
        .join(format!("models--{}", model.tokenizer.replace('/', "--")))
        .join("snapshots")
        .join(&model.tokenizer_revision)
}

pub(super) fn verify_tokenizer_snapshot(pinned: &Path, resolved: &Path) -> DynResult<()> {
    if pinned.canonicalize()? != resolved.canonicalize()? {
        return Err("offline tokenizer cache must resolve to the exact pinned snapshot".into());
    }
    Ok(())
}

pub(super) fn configuration(
    input: &Prepared,
    port: u16,
    concurrency: u32,
    sessions: usize,
    namespace_label: &str,
) -> DynResult<Value> {
    if let Some(tools) = &input.tools {
        verify_tokenizer_snapshot(
            &tools.tokenizer.path,
            &tokenizer_snapshot(&input.root, &input.model),
        )?;
    }
    let cohort = input
        .dataset
        .parent()
        .ok_or("agentic cohort parent")?
        .join("cohort.json");
    let cohort_digest =
        crate::product::digest::file_sha256(&cohort).map_err(|failure| failure.error)?;
    let marker_namespace = format!("{namespace_label}:{}:{cohort_digest}", input.model.key);
    // An explicit config avoids AgentX scenario defaults reinstating an
    // hour-long duration or consuming the selected turns in warmup.
    Ok(json!({"schema_version":"2.0","random_seed":42,"benchmark":{
        "model":input.model.key,
        "endpoint":{"url":format!("http://127.0.0.1:{port}/v1/chat/completions"),
            "type":"chat","streaming":true,"timeout":1800,
            "use_server_token_count":true,"session_header":"X-Session-ID","extra":{"ignore_eos":true}},
        // The locked offline client resolves repository IDs through the HF cache;
        // passing an absolute snapshot path triggers its repo-ID validation.
        "tokenizer":{"name":input.model.tokenizer,"revision":input.model.tokenizer_revision},
        "dataset":{"type":"file","path":input.dataset,"format":"weka_trace",
            "sampling":"sequential","entries":sessions,"max_context_length":trace_context::LIMIT,
            "ignore_trace_delays":false,"use_think_time_only":true,
            "cache_bust":{"target":"first_turn_prefix","namespace":marker_namespace}},
        "phases":[{"name":"profiling","type":"concurrency","timing_mode":"agentic_replay",
            "concurrency":concurrency,"sessions":sessions,"trajectory_start_min_ratio":0,
            "trajectory_start_max_ratio":0,"burst_phase_starts":true}],
        // Start the record processor before the one-shot dataset notification.
        "runtime":{"record_processors":1},
        "artifacts":{"dir":input.output.join("upstream"),"summary":["json"],
            "records":["jsonl"],"trace":true},"gpu_telemetry":{"enabled":false}
    }}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_resolution_rejects_a_different_or_missing_pinned_snapshot() {
        let root = tempfile::tempdir().unwrap();
        let pinned = root.path().join("pinned");
        let other = root.path().join("other");
        std::fs::create_dir(&pinned).unwrap();
        std::fs::create_dir(&other).unwrap();
        verify_tokenizer_snapshot(&pinned, &pinned).unwrap();
        assert!(verify_tokenizer_snapshot(&pinned, &other).is_err());
        assert!(verify_tokenizer_snapshot(&pinned, &root.path().join("missing")).is_err());
    }

    #[test]
    fn child_uses_the_same_explicit_cache_as_snapshot_admission() {
        let root = tempfile::tempdir().unwrap();
        let spec = super::super::command::spec(root.path(), "true", &[]).unwrap();
        assert!(
            matches!(spec.environment.get(std::ffi::OsStr::new("HF_HUB_CACHE")),
            Some(crate::process::Value::Public(path)) if path == tokenizer_hub_cache(root.path()).as_os_str())
        );
    }
}
