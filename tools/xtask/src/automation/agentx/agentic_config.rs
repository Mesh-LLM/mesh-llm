//! Explicit AgentX replay configuration shared by fixed-session profiles.
use super::{Prepared, trace_context};
use crate::command::DynResult;
use serde_json::{Value, json};

pub(super) fn configuration(
    input: &Prepared,
    port: u16,
    concurrency: u32,
    sessions: usize,
    namespace_label: &str,
) -> DynResult<Value> {
    let cache = std::env::var_os("HF_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| input.root.join(".cache/huggingface"));
    let tokenizer = cache
        .join("hub")
        .join(format!(
            "models--{}",
            input.model.tokenizer.replace('/', "--")
        ))
        .join("snapshots")
        .join(&input.model.tokenizer_revision);
    let tokenizer = input
        .tools
        .as_ref()
        .map_or(tokenizer, |tools| tools.tokenizer.path.clone());
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
        "tokenizer":{"name":tokenizer,"revision":input.model.tokenizer_revision},
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
