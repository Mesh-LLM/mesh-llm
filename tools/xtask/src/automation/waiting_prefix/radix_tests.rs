use super::{
    radix_cell, radix_gate,
    radix_identity::Arm,
    radix_owner::Owner,
    radix_projection::Projection,
    radix_summary,
    radix_workload::{self, Shape},
};
use crate::process::{
    self,
    retained::{Action, Context, Coordinator, MemberId, MemberState, Snapshot},
};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    path::PathBuf,
    time::{Duration, Instant},
};
fn shape() -> Shape {
    Shape {
        rounds: 2,
        requests: 4,
        levels: vec![1, 4],
        prefix_blocks: 2,
        output_tokens: 8,
        lanes: 4,
        n_gpu_layers: 999,
    }
}
fn event() -> Value {
    json!({"event":"stage.openai_generation_summary","attributes":{"skippy.kv.status":"hit","skippy.kv.matched_prefix_tokens":100,"skippy.kv.suffix_prefill_tokens":5,"llama_stage.prompt_token_count":105}})
}
fn owner() -> Owner {
    Owner {
        server: None,
        batches: VecDeque::new(),
        active: Some((MemberId::WorkerOne, 1, false, Instant::now())),
        projection: Projection::default(),
        boundaries: vec![],
        consumed: 0,
        batch_budget: Duration::from_millis(100),
        telemetry_budget: Duration::from_millis(100),
        stopping: false,
    }
}
fn context<'a>(snapshots: &'a [Snapshot]) -> Context<'a> {
    Context {
        elapsed: Duration::from_millis(10),
        remaining: Duration::from_secs(2),
        members: snapshots,
    }
}
fn snapshots() -> Vec<Snapshot> {
    vec![
        Snapshot {
            member: MemberId::Seed,
            pid: 1,
            started: Instant::now(),
            state: MemberState::Ready {
                elapsed: Duration::from_millis(1),
            },
        },
        Snapshot {
            member: MemberId::WorkerOne,
            pid: 2,
            started: Instant::now(),
            state: MemberState::ExpectedExit {
                status: 0,
                elapsed: Duration::from_millis(5),
            },
        },
    ]
}
#[test]
fn radix_delayed_summary_after_worker_exit_keeps_exact_batch_barrier() {
    let mut owner = owner();
    let snapshots = snapshots();
    assert!(matches!(owner.tick(context(&snapshots)), Action::Pending));
    let bytes = serde_json::to_vec(&event()).unwrap();
    owner.captured_line(
        MemberId::Seed,
        process::ObservedLine {
            stream: process::Stream::Stderr,
            bytes: &bytes,
            ending: process::LineEnding::Lf,
        },
    );
    assert!(matches!(
        owner.tick(context(&snapshots)),
        Action::Stop(MemberId::Seed)
    ));
    assert_eq!(owner.boundaries, vec![(0, 1, false)]);
}
#[test]
fn radix_missing_summary_refuses_observed_zero_of_one_instead_of_inventing_metrics() {
    let mut owner = owner();
    owner.active.as_mut().unwrap().3 = Instant::now() - Duration::from_secs(1);
    let snapshots = snapshots();
    match owner.tick(context(&snapshots)) {
        Action::Reject(error) => assert!(error.contains("observed 0/1")),
        _ => panic!("missing summary was admitted"),
    };
}
#[test]
fn radix_single_stage_config_preserves_explicit_offload_and_warm_cache_profile() {
    let arm = Arm {
        binary: PathBuf::from("/binary"),
        binary_sha256: "a".repeat(64),
        commit: "b".repeat(40),
        model: PathBuf::from("/model"),
        model_sha256: "c".repeat(64),
        model_id: "local/test".into(),
        native_build: PathBuf::from("/native"),
        native_build_sha256: "d".repeat(64),
        ctx_size: 4096,
        layer_end: 8,
        payload: "resident-kv".into(),
    };
    let cold = radix_cell::config(&arm, &shape(), false);
    let warm = radix_cell::config(&arm, &shape(), true);
    assert_eq!(cold["n_gpu_layers"], 999);
    assert!(cold.get("kv_cache").is_none());
    assert_eq!(warm["kv_cache"]["shared_prefix_record_limit"], 4);
    assert_eq!(warm["kv_cache"]["payload"], "resident-kv");
}
#[test]
fn radix_divergent_prompts_are_unique_nonempty_and_preserve_base() {
    let a = radix_workload::divergent("stable", 1, 0);
    let b = radix_workload::divergent("stable", 1, 1);
    assert!(a.starts_with("stable") && !a.is_empty());
    assert_ne!(a, b);
}
#[test]
fn radix_coding_trace_grows_from_stable_prior_transcript() {
    let a = radix_workload::coding("stable", 1, 0);
    let b = radix_workload::coding("stable", 1, 1);
    assert!(
        b.starts_with(
            a.strip_suffix("Assistant: return the latest invariant only.")
                .unwrap()
        )
    );
    assert!(b.len() > a.len());
    let batches = radix_workload::batches(&shape(), true).unwrap();
    assert_eq!(batches.len(), 12);
    assert_eq!(batches.iter().filter(|b| b.warmup).count(), 6);
}
#[test]
fn radix_absent_telemetry_leaves_optional_cache_metrics_null_and_retains_prompt_hashes() {
    let row = json!({"prompt_sha256":"prompt","content_sha256":"canonical-output","ttft_ms":4,"tpot_ms":2,"elapsed_ms":10,"error":null});
    let value = radix_summary::summarize(&[row], &[]);
    assert!(
        value["matched_prefix_tokens_median"].is_null()
            && value["suffix_prefill_tokens_median"].is_null()
    );
    assert_eq!(
        value["outputs_by_prompt"]["prompt"],
        json!(["canonical-output"])
    );
}
fn cell(
    version: &str,
    cache: &str,
    output: &str,
    ttft: f64,
    sample: (&str, u32, u32, u32, u32),
) -> Value {
    let (scenario, hits, n, requests, round) = sample;
    let summary = json!({"requests":requests,"successful":requests,"cache_hits":hits,"cache_misses":requests-hits,"matched_prefix_tokens":vec![100;requests as usize],"suffix_prefill_tokens":vec![5;requests as usize],"ttft_ms":vec![ttft;requests as usize],"tpot_ms":vec![2.0;requests as usize],"outputs_by_prompt":{"prompt":[output]}});
    json!({"version":version,"cache":cache,"round":round,"suspect_log":false,"observations":[{"scenario":scenario,"concurrency":n,"summary":summary}]})
}
fn gate(cells: &[Value], payload: &str) -> Value {
    radix_gate::evaluate(payload, cells, &radix_summary::aggregate(cells))
}
#[test]
fn radix_cache_lift_is_independent_of_old_stale_output_and_new_preservation() {
    let cells = vec![
        cell("old", "cold", "correct", 100.0, ("divergent", 0, 1, 1, 1)),
        cell("old", "warm", "stale", 30.0, ("divergent", 1, 1, 1, 1)),
        cell("new", "cold", "correct", 100.0, ("divergent", 0, 1, 1, 1)),
        cell("new", "warm", "correct", 20.0, ("divergent", 1, 1, 1, 1)),
    ];
    let rows = radix_summary::aggregate(&cells);
    let old = rows
        .iter()
        .find(|r| r["version"] == "old" && r["cache"] == "warm")
        .unwrap();
    let new = rows
        .iter()
        .find(|r| r["version"] == "new" && r["cache"] == "warm")
        .unwrap();
    assert_eq!(old["cache_lift_ttft_ms"], 70.0);
    assert_eq!(new["cache_lift_ttft_ms"], 80.0);
    let (_, preservation) = radix_summary::comparisons(&rows);
    assert!(
        preservation
            .iter()
            .any(|r| r["version"] == "old" && r["cache_preserves_output"] == false)
    );
    assert!(
        preservation
            .iter()
            .any(|r| r["version"] == "new" && r["cache_preserves_output"] == true)
    );
    assert_eq!(gate(&cells, "resident-kv")["passed"], true);
}
#[test]
fn radix_recurrent_gate_requires_only_exact_checkpoint_hits() {
    let mut cells = vec![];
    for version in ["old", "new"] {
        for cache in ["cold", "warm"] {
            for scenario in ["exact", "divergent", "coding"] {
                cells.push(cell(
                    version,
                    cache,
                    "correct",
                    25.0,
                    (
                        scenario,
                        u32::from(cache == "warm" && scenario == "exact"),
                        1,
                        1,
                        1,
                    ),
                ));
            }
        }
    }
    assert_eq!(gate(&cells, "kv-recurrent")["passed"], true);
    assert_eq!(gate(&cells, "resident-kv")["passed"], false);
}
fn concurrent(new_hits: u32) -> Vec<Value> {
    let mut cells = vec![];
    for round in [1, 2] {
        for (version, hits) in [("old", 4), ("new", new_hits)] {
            for cache in ["cold", "warm"] {
                cells.push(cell(
                    version,
                    cache,
                    "correct",
                    25.0,
                    (
                        "divergent",
                        if cache == "warm" { hits } else { 0 },
                        4,
                        4,
                        round,
                    ),
                ));
            }
        }
    }
    cells
}
#[test]
fn radix_concurrent_hit_gate_tolerates_one_transient_miss_per_round() {
    assert_eq!(gate(&concurrent(3), "resident-kv")["passed"], true);
}
#[test]
fn radix_concurrent_hit_gate_rejects_regression_beyond_round_tolerance() {
    let result = gate(&concurrent(2), "resident-kv");
    assert_eq!(result["passed"], false);
    assert!(
        result["failures"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v.as_str().unwrap().contains("2-request round tolerance"))
    );
}
#[test]
fn radix_new_n1_output_change_fails_only_when_old_preserved_cold_baseline() {
    let run = |old_warm: &str| {
        let cells = vec![
            cell("old", "cold", "baseline", 100.0, ("divergent", 0, 1, 1, 1)),
            cell("old", "warm", old_warm, 25.0, ("divergent", 1, 1, 1, 1)),
            cell("new", "cold", "baseline", 100.0, ("divergent", 0, 1, 1, 1)),
            cell("new", "warm", "changed", 20.0, ("divergent", 1, 1, 1, 1)),
        ];
        gate(&cells, "resident-kv")
    };
    assert_eq!(run("baseline")["passed"], false);
    assert_eq!(run("changed")["passed"], true);
}
#[test]
fn radix_typed_projection_omits_secret_unknown_fields_and_rejects_extra_boundary() {
    let mut projection = Projection::default();
    let mut value = event();
    value["attributes"]["secret"] = json!("raw-user-token");
    projection.observe(&serde_json::to_vec(&value).unwrap());
    assert!(
        !serde_json::to_string(&projection.rows)
            .unwrap()
            .contains("raw-user-token")
    );
    let mut owner = owner();
    owner.projection = projection;
    owner
        .projection
        .observe(&serde_json::to_vec(&event()).unwrap());
    assert!(matches!(
        owner.tick(context(&snapshots())),
        Action::Reject(_)
    ));
}

#[test]
fn radix_initial_model_startup_uses_configured_batch_budget_before_short_rechecks() {
    for batch in [2, 4, 901, 10800] {
        assert_eq!(radix_cell::readiness_budget(0, batch), batch - 1);
        assert!(radix_cell::readiness_budget(0, batch) < batch);
        assert_eq!(radix_cell::readiness_budget(1, batch), 1);
        assert_eq!(radix_cell::readiness_budget(127, batch), 1);
    }
}
