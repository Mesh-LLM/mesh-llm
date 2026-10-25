//! Required command-level evidence for comparison help, reproducibility and refusal.
use serde_json::Value;
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
#[path = "../../src/automation/event_benchmark_comparison/fixture.rs"]
mod fixture;

fn invoke(directory: &Path, args: &[String]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(directory)
        .args(args)
        .output()
        .unwrap()
}

fn inputs(directory: &Path) {
    for (name, mode) in [
        ("production.json", "production"),
        ("reference.json", "event-disabled"),
        ("baseline.json", "production"),
    ] {
        fs::write(
            directory.join(name),
            serde_json::to_vec(&fixture::manifest(mode)).unwrap(),
        )
        .unwrap();
    }
}

fn arguments(output: &str) -> Vec<String> {
    [
        "automation",
        "event-benchmark-compare",
        "--production",
        "production.json",
        "--event-disabled",
        "reference.json",
        "--baseline",
        "baseline.json",
        "--output",
        output,
        "--bootstrap-samples",
        "10000",
        "--seed",
        "42",
        "--max-degradation-percent",
        "3",
        "--min-primary-pairs",
        "20",
        "--min-scenario-pairs",
        "10",
        "--max-mdd-percent",
        "10",
        "--report-holm",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

#[test]
fn native_comparator_help_exits_zero_and_lists_every_frozen_flag() {
    let directory = tempfile::tempdir().unwrap();
    let args = ["automation", "event-benchmark-compare", "--help"].map(String::from);
    let output = invoke(directory.path(), &args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let help = String::from_utf8(output.stdout).unwrap();
    for flag in [
        "--production",
        "--event-disabled",
        "--baseline",
        "--output",
        "--bootstrap-samples",
        "--seed",
        "--max-degradation-percent",
        "--min-primary-pairs",
        "--min-scenario-pairs",
        "--max-mdd-percent",
        "--report-holm",
    ] {
        assert!(help.contains(flag), "missing {flag}");
    }
}

#[test]
fn native_comparator_separate_processes_produce_identical_complete_reports() {
    let directory = tempfile::tempdir().unwrap();
    inputs(directory.path());
    // Nonconstant paired measurements ensure this checks actual resampling.
    let path = directory.path().join("production.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    for (index, trial) in manifest["trials"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        trial["decode_tok_s"] = serde_json::json!(50.0 + if index % 2 == 0 { 0.2 } else { -0.2 });
    }
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    for name in ["first.json", "second.json"] {
        let output = invoke(directory.path(), &arguments(name));
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(summary["certification_status"], "pass");
    }
    let first = fs::read(directory.path().join("first.json")).unwrap();
    assert_eq!(
        first,
        fs::read(directory.path().join("second.json")).unwrap()
    );
    let report: Value = serde_json::from_slice(&first).unwrap();
    assert_eq!(
        report["resampling_algorithm"],
        "sha256-counter-paired-bootstrap-v1"
    );
    assert!(
        report["comparison_a"]["metrics"][0]["minimal_detectable_degradation_pct"]
            .as_f64()
            .unwrap()
            > 0.0
    );
}

#[test]
fn native_comparator_rejects_nonfinite_and_oversized_numeric_input_without_replacing_report() {
    let directory = tempfile::tempdir().unwrap();
    inputs(directory.path());
    let source = serde_json::to_string(&fixture::manifest("production")).unwrap();
    let marker = "\"decode_tok_s\":50.0";
    assert!(source.contains(marker));
    let output_path = directory.path().join("report.json");
    for value in ["NaN".to_owned(), "1e10000".to_owned(), "9".repeat(1000)] {
        let malformed = source.replacen(marker, &format!("\"decode_tok_s\":{value}"), 1);
        fs::write(directory.path().join("production.json"), malformed).unwrap();
        fs::write(&output_path, b"previous report").unwrap();
        let output = invoke(directory.path(), &arguments("report.json"));
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert_eq!(fs::read(&output_path).unwrap(), b"previous report");
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 4);
    }
}

#[test]
fn native_comparator_blocked_health_publishes_actual_reason_and_exits_nonzero() {
    let directory = tempfile::tempdir().unwrap();
    inputs(directory.path());
    let mut manifest = fixture::manifest("production");
    manifest["health"] = Value::Null;
    fs::write(
        directory.path().join("production.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let output = invoke(directory.path(), &arguments("report.json"));
    assert_eq!(output.status.code(), Some(1));
    let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["certification_status"], "blocked");
    let report: Value =
        serde_json::from_slice(&fs::read(directory.path().join("report.json")).unwrap()).unwrap();
    assert!(
        report["blocking_reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason == "health_unavailable")
    );
}
