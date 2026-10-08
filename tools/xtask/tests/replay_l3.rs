use serde_json::{Value, json};
use std::process::{Command, Output};
fn invoke(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix"])
        .args(arguments)
        .current_dir(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .ancestors()
                .nth(2)
                .unwrap(),
        )
        .output()
        .unwrap()
}
#[test]
fn manual_l3_plan_retains_defaults_and_does_not_read_manifest_or_build() {
    let result = invoke(&[
        "l3-plan",
        "--ref",
        "candidate=HEAD",
        "--model",
        "hf:org/model@immutable/model.gguf",
        "--trajectory-manifest",
        "/nonexistent/captured-l3.json",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let plan: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(plan["kind"], "disk-l3-lifecycle");
    assert_eq!(plan["config"]["concurrency"], json!([64, 128, 256]));
    assert_eq!(plan["config"]["cold_samples"], 3);
    assert_eq!(plan["config"]["restart_samples"], 3);
    assert_eq!(plan["config"]["identical_repeats"], 100);
    assert_eq!(plan["gates"]["post_restart_l3_ttft_p50_ratio_max"], 0.5);
    assert_eq!(plan["gates"]["payload_write_amplification_max"], 1.2);
    assert_eq!(
        plan["gates"]["high_load_decode_inter_token_p99_regression_max_pct"],
        5.0
    );
    assert!(
        plan["server_commands"]["disk_on"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "<persistent-cache-root>")
    );
}
#[test]
fn manual_l3_run_rejects_missing_captured_sources_before_creating_output() {
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("output");
    let result = invoke(&[
        "l3-run",
        "--ref",
        "candidate=HEAD",
        "--model",
        "fixture",
        "--trajectory-manifest",
        "/not-present",
        "--output",
        output.to_str().unwrap(),
    ]);
    assert!(!result.status.success());
    assert!(!output.exists());
    assert!(String::from_utf8_lossy(&result.stderr).contains("configuration"));
}
#[test]
fn manual_l3_run_preserves_plan_but_rejects_invalid_manifest_before_any_build() {
    let parent = tempfile::tempdir().unwrap();
    let source = parent.path().join("manifest.json");
    let output = parent.path().join("output");
    std::fs::write(
        &source,
        serde_json::to_vec(&json!({"cohorts":{"l3":[],"1":[]}})).unwrap(),
    )
    .unwrap();
    let result = invoke(&[
        "l3-run",
        "--ref",
        "candidate=HEAD",
        "--model",
        "fixture",
        "--trajectory-manifest",
        source.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--concurrency",
        "1",
        "--require-source-dataset",
        "buzz",
        "--require-source-dataset",
        "opencode",
        "--require-source-dataset",
        "goose",
    ]);
    assert!(!result.status.success());
    assert!(output.join("plan.json").is_file());
    assert!(!output.join("build-candidate.input.json").exists());
    assert!(!output.join("logs").exists());
}
#[test]
fn manual_l3_report_reads_retained_legacy_receipt_and_inventory_without_python() {
    let parent = tempfile::tempdir().unwrap();
    let document = json!({"schema_version":1,"kind":"disk-l3-lifecycle","build":{"commit":"legacy"},"config":{"model":"legacy-model","backend":"metal"},"gates":{"evaluated":true,"passed":true,"checks":[{"name":"single_physical_fill","passed":true,"detail":"fills=1"}],"cold_ttft_p50_seconds":2.0,"restart_l3_ttft_p50_seconds":0.5,"restart_l3_ttft_ratio":0.25}});
    std::fs::write(
        parent.path().join("run.json"),
        serde_json::to_vec(&document).unwrap(),
    )
    .unwrap();
    let result = invoke(&["l3-report", "--artifact", parent.path().to_str().unwrap()]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        std::fs::read_to_string(parent.path().join("REPORT.md"))
            .unwrap()
            .contains("legacy-model")
    );
    assert!(parent.path().join("artifact-sha256.txt").is_file());
}
