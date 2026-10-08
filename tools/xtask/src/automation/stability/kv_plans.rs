//! describe complete KV intent without HTTP or evidence I/O.
use super::kv_options::Options;
use serde_json::{Value, json};
pub(super) fn build(options: &Options) -> Value {
    let mut checks = Vec::new();
    for model in &options.models {
        for attempt in 1..=options.attempts {
            checks.push(json!({"model":model,"attempt":attempt,"phase":"tool_loop",
                "pressure_turns":options.pressure_turns,
                "steps":["primary_tool_call","primary_tool_result","pressure_turns","secondary_tool_call","secondary_tool_result","recall_both_facts_and_pin"]}));
            checks.push(json!({"model":model,"attempt":attempt,"phase":"overlap_tool_loop","overlap_requests":options.overlap_requests,
            "admission":"all participants reach start barrier before concurrent requests",
            "steps":["simultaneous_title_and_tools","complete_individual_tool_histories","warm_and_measure_overlap_prefix"]}));
        }
        for phase in ["same_prefix_cache", "exact_prefix_cache"] {
            checks.push(json!({"model":model,"phase":phase,"min_cached_tokens":options.minimum_cached,
                "suffix_prefill_limit":options.suffix_limit,"steps":["warm","measured"],
                "geometry":if phase == "same_prefix_cache" { "same prefix with a different short tail" } else { "identical complete request body" }}));
        }
    }
    if !options.native_logs.is_empty() {
        checks.push(json!({"phase":"native_log_scan","paths":options.native_logs,
            "scan_mode":"appended_since_run_start","capture_before_requests":true,
            "rescan_on":["file_identity_change","truncation","checkpoint_tail_rewrite"],
            "fatal_patterns":["failed to find a memory slot","RuntimeError: llama_decode failed","llama_decode failed","proactive_eviction status=error"]}));
    }
    let mut evidence_files = vec![
        "manifest.json",
        "results.jsonl",
        "summary.json",
        "summary.md",
        "transcripts/",
    ];
    if !options.native_logs.is_empty() {
        evidence_files.push("native-log-scan.json");
    }
    json!({"name":"kv-tool-loop-stability","base_url":options.base.as_str(),
        "models":options.models,"attempts":options.attempts,"pressure_turns":options.pressure_turns,
        "overlap_requests":options.overlap_requests,"timeout_seconds":options.timeout.as_secs_f64(),
        "output_dir":options.output,"min_cached_tokens":options.minimum_cached,
        "suffix_prefill_limit":options.suffix_limit,"native_logs":options.native_logs,
        "native_log_scan_mode":if options.native_logs.is_empty() { None } else { Some("appended_since_run_start") },
        "checks":checks,"evidence_files":evidence_files})
}
