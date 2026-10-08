use super::{
    acceptance::Version, adaptive_identity as identity, mixed_counters::Projection,
    mixed_summary as summary, mixed_workload::*,
};
use serde_json::{Value, json};
fn shape() -> Shape {
    serde_json::from_value(json!({"rounds":2,"anchors":1,"prefills":1,"anchor_prompt_blocks":8,"prefill_prompt_blocks":256,"anchor_output_tokens":128,"prefill_output_tokens":8,"prefill_delay_ms":50.0,"prefill_stagger_ms":5.0,"lanes":2,"n_batch":1024,"n_ubatch":256,"prefill_adaptive_start":256,"prefill_adaptive_step":256,"prefill_adaptive_max":256,"adaptive_target_new_only":true})).unwrap()
}
#[test]
fn mixed_stable_role_prompts_are_deterministic_nonempty_and_preserve_request_identity() {
    let first = stable_prompt(4, 2, Role::Anchor).unwrap();
    assert_eq!(first, stable_prompt(4, 2, Role::Anchor).unwrap());
    assert!(first.contains("context-block-0000") && first.contains("Request 2"));
    assert!(first.contains("numbered implementation checklist"));
    assert!(
        stable_prompt(4, 2, Role::Prefill)
            .unwrap()
            .contains("owner of invariant 2")
    );
    assert!(
        stable_prompt(16, -1, Role::Prefill)
            .unwrap()
            .contains("owner of invariant 15")
    );
    assert!(stable_prompt(0, 2, Role::Anchor).is_err());
}
#[test]
fn mixed_two_stage_and_local_configs_preserve_lanes_batches_model_context_and_target_forwarding() {
    let root = tempfile::tempdir().unwrap();
    let mut arm:identity::Input=serde_json::from_value(json!({"schema_version":1,"round":1,"version":"old","binary":root.path().join("binary"),"binary_sha256":"a".repeat(64),"commit":"b".repeat(40),"native_build":root.path().join("native"),"native_build_sha256":"c".repeat(64),"native_profile":"standalone-static-skippy-server","model":root.path().join("model"),"model_sha256":"d".repeat(64),"model_id":"fixture","ctx_size":32768,"split_layer":14,"layer_end":27,"n_gpu_layers":999,"adaptive_target_ms":100.0,"stage_ports":[9000,9001],"openai_port":9002})).unwrap();
    let mut profile = shape();
    profile.lanes = 12;
    let first = config(&arm, &profile, 0, true).unwrap();
    let last = config(&arm, &profile, 1, true).unwrap();
    assert_eq!(first["downstream"]["endpoint"], "tcp://127.0.0.1:9001");
    assert!(first["upstream"].is_null());
    assert_eq!(last["upstream"]["endpoint"], "tcp://127.0.0.1:9000");
    assert!(last["downstream"].is_null());
    assert_eq!(first["layer_end"], last["layer_start"]);
    assert_eq!(first["lane_count"], 12);
    assert_eq!(first["n_batch"], 1024);
    assert_eq!(first["n_ubatch"], 256);
    assert_eq!(first["ctx_size"], 32768);
    assert_eq!(first["source_model_sha256"], arm.model_sha256);
    assert!(first.get("kv_cache").is_none());
    let local = config(&arm, &profile, 0, false).unwrap();
    assert_eq!(local["layer_end"], 27);
    assert!(local["upstream"].is_null() && local["downstream"].is_null());
    assert!(config(&arm, &profile, 1, false).is_err());
    let old = arguments(&arm, &profile, &root.path().join("config.json"), 0, true).unwrap();
    assert!(
        !old.iter()
            .any(|a| a == "--openai-prefill-adaptive-target-ms")
    );
    arm.version = Version::New;
    let new = arguments(&arm, &profile, &root.path().join("config.json"), 0, true).unwrap();
    assert!(
        new.windows(2)
            .any(|a| a[0] == "--openai-bind-addr" && a[1] == "127.0.0.1:9002")
    );
    assert!(
        new.windows(2)
            .any(|a| a[0] == "--openai-generation-concurrency" && a[1] == "12")
    );
    assert!(
        new.windows(2)
            .any(|a| a[0] == "--openai-prefill-adaptive-target-ms" && a[1] == "100")
    );
    arm.version = Version::Old;
    profile.adaptive_target_new_only = false;
    assert!(
        arguments(&arm, &profile, &root.path().join("config.json"), 0, false)
            .unwrap()
            .iter()
            .any(|a| a == "--openai-prefill-adaptive-target-ms")
    );
    profile.n_ubatch = 1025;
    assert!(profile.validate().is_err());
    root.close().unwrap();
}
fn manifest() -> Manifest {
    serde_json::from_value(json!({"metadata":{"dataset_revision":"pinned"},"prompts":[{"family":"trajectory-1","bucket":"8k-16k","source_id":"session-1","prompt":"real trace1"},{"family":"trajectory-2","source_id":"session-2","prompt":"real trace2"}]})).unwrap()
}
#[test]
fn mixed_manifest_partitions_trace_provenance_exactly_by_round_and_preserves_metadata() {
    let shape = shape();
    let manifest = manifest();
    manifest.validate(&shape).unwrap();
    assert_eq!(manifest.metadata["dataset_revision"], "pinned");
    let first = requests(&shape, 1, Some(&manifest)).unwrap();
    let second = requests(&shape, 2, Some(&manifest)).unwrap();
    assert_eq!(first[0].role, Role::Anchor);
    assert_eq!(first[1].role, Role::Prefill);
    assert_eq!(first[1].prompt.family, "trajectory-1");
    assert_eq!(first[1].prompt.provenance["bucket"], "8k-16k");
    assert_eq!(first[1].prompt.provenance["source_id"], "session-1");
    assert_eq!(second[1].prompt.provenance["source_id"], "session-2");
    assert_eq!(first[0].delay_ms, 0.0);
    assert_eq!(first[1].delay_ms, 50.0);
    assert_eq!(first[0].output_tokens, 128);
    assert_eq!(first[1].output_tokens, 8);
    assert!(requests(&shape, 0, Some(&manifest)).is_err());
}
#[test]
fn mixed_manifest_missing_prompt_family_or_exact_round_roster_refuses() {
    let profile = shape();
    for mode in ["prompt", "family", "count"] {
        let mut m = manifest();
        match mode {
            "prompt" => m.prompts[0].prompt.clear(),
            "family" => m.prompts[0].family.clear(),
            "count" => {
                m.prompts.pop();
            }
            _ => unreachable!(),
        };
        assert!(m.validate(&profile).is_err());
    }
}
fn prefill(tokens: u64) -> Value {
    json!({"event":"stage.openai_prefill","attributes":{"llama_stage.prefill_token_count":tokens,"llama_stage.prefill_chunk_count":3,"llama_stage.prefill_max_chunk_size":256,"llama_stage.prefill_bottleneck_stage_index":1,"skippy.kv.chain_cache_errors":0,"skippy.kv.stage0_cache_errors":0}})
}
#[test]
fn mixed_scheduler_feature_numeric_owner_preserves_event_tag_and_refuses_unqualified_or_incomplete_capture()
 {
    let mut p = Projection::default();
    p.observe(serde_json::to_vec(&prefill(8)).unwrap().as_slice());
    let boundary = p.warmup_boundary().unwrap();
    p.observe(br#"{"event":"stage.scheduler_feature_iteration","attributes":{"skippy.scheduler.token_count":7}}"#);
    for n in [10, 100] {
        p.observe(serde_json::to_vec(&prefill(n)).unwrap().as_slice());
    }
    let measured = p.measured(&boundary, 2, &"a".repeat(64), true).unwrap();
    assert_eq!(measured.scheduler.len(), 1);
    let row = serde_json::to_value(&measured.scheduler[0]).unwrap();
    assert_eq!(row["skippy.scheduler.token_count"], 7);
    assert_eq!(row["_event"], "stage.scheduler_feature_iteration");
    assert!(p.measured(&boundary, 2, &"a".repeat(64), false).is_err());
    assert!(p.measured(&boundary, 3, &"a".repeat(64), true).is_err());
    p.observe(br#"{"event":"stage.scheduler_iteration","attributes":{"skippy.scheduler.decode_tokens":1}}"#);
    assert!(p.measured(&boundary, 2, &"a".repeat(64), true).is_err());
}
fn worker(scale: f64) -> Value {
    json!({"schema_version":1,"round":1,"version":"old","model":"fixture","input_sha256":"a".repeat(64),"error":null,"makespan_ms":100.0*scale,"requests":[{"role":"anchor","request_index":0,"prompt_sha256":"b".repeat(64),"prompt_provenance":{"family":"anchor"},"content_sha256":"c".repeat(64),"completion_tokens":128,"ttft_ms":2.0*scale,"content_gaps_ms":[1.0*scale,3.0*scale]},{"role":"prefill","request_index":1,"prompt_sha256":"d".repeat(64),"prompt_provenance":{"family":"trace","source_id":"one"},"content_sha256":"e".repeat(64),"completion_tokens":8,"ttft_ms":4.0*scale,"content_gaps_ms":[]}]})
}
#[test]
fn mixed_paired_role_metrics_are_deterministic_and_require_explicit_scheduler_phase_qualification()
{
    let mut cells = Vec::new();
    for round in 1..=4 {
        for version in ["old", "new"] {
            let mut cell = worker(if version == "old" { 1.0 } else { 1.1 });
            cell["round"] = json!(round);
            cell["version"] = json!(version);
            cell["summary"] = summary::requests(&cell).unwrap();
            cells.push(cell);
        }
    }
    let first = summary::compare(&cells, 4).unwrap();
    assert_eq!(first, summary::compare(&cells, 4).unwrap());
    assert!(
        (first["paired_delta_percent"]["makespan_ms"]["median"]
            .as_f64()
            .unwrap()
            - 10.0)
            .abs()
            < 1e-9
    );
    assert_eq!(first["qualified"], false);
    assert_eq!(first["output_parity"]["exact_matches"], 8);
    assert!(summary::compare(&cells[..7], 4).is_err());
    let mut changed = cells;
    changed[1]["requests"][1]["role"] = json!("anchor");
    assert!(summary::compare(&changed, 4).is_err());
}
#[test]
fn mixed_report_skips_missing_role_metrics_and_renders_zero_denominator_as_na() {
    let value = json!({"aggregate":{"old":{"makespan_ms":0.0,"anchor_gap_ms_p95":null},"new":{"makespan_ms":2.0,"prefill_ttft_ms_p95":null}},"paired_delta_percent":{},"output_parity":{"exact_matches":0,"comparable_requests":0}});
    let rendered = summary::render(&value);
    assert!(rendered.contains("Mixed scheduling"));
    assert!(!rendered.contains("Anchor stream gap p95 ms"));
    assert!(!rendered.contains("Prefill TTFT p95 ms"));
    assert!(rendered.contains("| Makespan ms | 0.000 | 2.000 | n/a |"));
}
