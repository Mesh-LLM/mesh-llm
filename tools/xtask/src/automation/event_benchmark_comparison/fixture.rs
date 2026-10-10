//! Finite recorded-manifest fixtures, never a claim of benchmark execution.
use serde_json::{Value, json};

pub(super) fn manifest(mode: &str) -> Value {
    let trials=(0..20).map(|pair_index|json!({"scenario":"__primary__","pair_index":pair_index,"status":"succeeded","decode_tok_s":50.0,"decode_only_tok_s":55.0,"ttft_ms":100.0})).collect::<Vec<_>>();
    let executed_order=(0..20).map(|pair_index|json!({"scenario":"__primary__","pair_index":pair_index,"order":if pair_index%2==0 {vec!["production","event-disabled"]} else {vec!["event-disabled","production"]}})).collect::<Vec<_>>();
    json!({
        "schema_version":1,"metrics_schema":"streaming_v1","mode":mode,"seed":42,"attempt":1,
        "environment":{
            "MESH_LLM_LIFECYCLE_LOG_PARSER":{"value":"auto","redacted":false},
            "MESH_LLM_BENCHMARK_TUNE_TRIAL":{"value":true,"redacted":false},
            "MESH_LLM_EVENT_SYSTEM_TRIAL_MODE":{"value":mode,"redacted":false}
        },
        "host":{"system":"Darwin","machine":"arm64","certification_host":"macos-arm64-metal","p99_gate":"enforced"},
        "health":{"terminal_delivery_failed":0,"cancelled_reservation_rejected":0,"dropped_progress":0,"dropped_diagnostic":0},
        "callback_ingress_p99_us":50.0,"expected_dropped_progress":0,"expected_dropped_diagnostic":0,
        "scenarios":[],"trials":trials,"executed_order":executed_order,
        "binary":{"path":"/fixtures/mesh-llm","sha256":"a".repeat(64),"version":"mesh-llm fixture"},
        "thermal_state":{"available":true,"source":"fixture","raw":"No thermal warning"}
    })
}
