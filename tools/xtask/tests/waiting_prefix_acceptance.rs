//! Actual offline waiting-prefix command status and published evidence.
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};

fn aggregate(version: &str) -> Value {
    json!({
        "version": version, "rounds": 4, "requests": 24, "successful": 24,
        "capacity_rejections": 0, "cache_hits_median": 1,
        "suffix_prefill_tokens_median": 8237, "family_switches_median": 1,
        "ttft_ms_p50_median": 3161.1, "ttft_ms_p95_median": 3313.7,
        "makespan_ms_median": 5326.8, "output_tokens_per_second_median": 71.34,
        "resident_evicted_tokens_median": 0, "resident_evicted_entries_median": 0,
        "predicted_recompute_cost_median": null,
    })
}

fn run(directory: &Path) -> std::process::Output {
    let catalog =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../evals/skippy-scheduler-fixtures.json");
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(directory)
        .env_clear()
        .args([
            "automation",
            "waiting-prefix",
            "evaluate",
            "--comparison",
            "comparison.json",
            "--output",
            "acceptance.json",
            "--report",
            "report.md",
            "--catalog",
        ])
        .arg(catalog)
        .args(["--profile", "warm-affinity"])
        .output()
        .unwrap()
}

#[test]
fn native_offline_acceptance_publishes_measured_success_and_failure() {
    let directory = tempfile::tempdir().unwrap();
    let mut input = json!({"aggregate": [aggregate("old"), aggregate("new")]});
    let path = directory.path().join("comparison.json");
    fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
    let passed = run(directory.path());
    assert!(
        passed.status.success(),
        "{}",
        String::from_utf8_lossy(&passed.stderr)
    );
    let evidence: Value =
        serde_json::from_slice(&fs::read(directory.path().join("acceptance.json")).unwrap())
            .unwrap();
    assert_eq!(evidence["passed"], true);
    assert!(
        fs::read_to_string(directory.path().join("report.md"))
            .unwrap()
            .contains("Fixture acceptance: **PASS**")
    );
    input["aggregate"][1]["ttft_ms_p95_median"] = json!(4000.0);
    fs::write(path, serde_json::to_vec(&input).unwrap()).unwrap();
    let failed = run(directory.path());
    assert_eq!(failed.status.code(), Some(1));
    let evidence: Value =
        serde_json::from_slice(&fs::read(directory.path().join("acceptance.json")).unwrap())
            .unwrap();
    assert_eq!(evidence["passed"], false);
    assert!(
        fs::read_to_string(directory.path().join("report.md"))
            .unwrap()
            .contains("Fixture acceptance: **FAIL**")
    );
}

#[test]
fn native_offline_acceptance_preserves_prior_output_when_input_is_invalid() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("comparison.json"), b"{}").unwrap();
    fs::write(directory.path().join("acceptance.json"), b"keep-me").unwrap();
    let failed = run(directory.path());
    assert_eq!(failed.status.code(), Some(1));
    assert_eq!(
        fs::read(directory.path().join("acceptance.json")).unwrap(),
        b"keep-me"
    );
    assert!(!directory.path().join("report.md").exists());
}

#[test]
fn native_measurement_commands_preserve_unknown_cost_and_aggregate_actual_rounds() {
    let directory = tempfile::tempdir().unwrap();
    let mut cells = Vec::new();
    for version in ["old", "new"] {
        let input = json!({
            "round": 1, "version": version, "makespan_ms": 10,
            "requests": [{"request_id": 0, "family": "one", "first_token_ms": 5,
                "ttft_ms": 5, "tokens_predicted": 2}],
            "events": [], "capacity_events": [],
            "record_events": [{"attributes": {"skippy.kv.decision": "proactive_eviction",
                "skippy.kv.proactive_evicted_tokens": 4, "skippy.kv.proactive_evicted_entries": 1}}],
        });
        fs::write(
            directory.path().join("input.json"),
            serde_json::to_vec(&input).unwrap(),
        )
        .unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(directory.path())
            .env_clear()
            .args([
                "automation",
                "waiting-prefix",
                "summarize",
                "--input",
                "input.json",
                "--output",
                "cell.json",
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let cell: Value =
            serde_json::from_slice(&fs::read(directory.path().join("cell.json")).unwrap()).unwrap();
        assert!(cell["summary"]["predicted_recompute_cost_total"].is_null());
        assert_eq!(cell["summary"]["resident_evicted_tokens_total"], 4.0);
        cells.push(cell);
    }
    fs::write(
        directory.path().join("cells.json"),
        serde_json::to_vec(&json!({"cells": cells})).unwrap(),
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(directory.path())
        .env_clear()
        .args([
            "automation",
            "waiting-prefix",
            "aggregate",
            "--input",
            "cells.json",
            "--output",
            "aggregate.json",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let aggregate: Value =
        serde_json::from_slice(&fs::read(directory.path().join("aggregate.json")).unwrap())
            .unwrap();
    for row in aggregate["aggregate"].as_array().unwrap() {
        assert_eq!(row["rounds"], 1);
        assert_eq!(row["successful"], 1);
        assert!(row["predicted_recompute_cost_median"].is_null());
        assert_eq!(row["ttft_ms_p50_median"], 5.0);
    }
}
