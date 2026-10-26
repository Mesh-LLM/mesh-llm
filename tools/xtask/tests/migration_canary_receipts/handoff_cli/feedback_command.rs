use super::*;
use std::path::Path;

fn evidence(fixture: &Fixture, package_input: &Value, family: &str, outcome: &str) {
    let mut input = package_input.clone();
    let directory = fixture.0.join("evidence").join(family);
    fs::create_dir_all(&directory).unwrap();
    let results = format!(
        "{{\"family\":\"{family}\",\"exit_code\":0,\"split_layer\":10,\"outcomes\":[{{\"name\":\"chain\",\"status\":\"pass\",\"exit_code\":0}},{{\"name\":\"single-step\",\"status\":\"pass\",\"exit_code\":0}},{{\"name\":\"state-handoff\",\"status\":\"pass\",\"exit_code\":0}}]}}\n"
    );
    fs::write(directory.join("results.jsonl"), results).unwrap();
    input["evidence"] = json!(directory);
    input["family"] = json!(family);
    input["outcome"] = json!(outcome);
    input["runner"] = json!("fixture-runner");
    let output = invoke(fixture, &input, "receipt");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
fn aggregate_cli(fixture: &Fixture, input: &Value, feedback: Option<&Path>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .args(["automation", "canary-receipts", "aggregate", "--package"])
        .arg(fixture.0.join("package"))
        .args([
            "--identity",
            input["identity_sha256"].as_str().unwrap(),
            "--run-id",
            "123",
            "--run-attempt",
            "4",
            "--controller-revision",
            input["controller_revision"].as_str().unwrap(),
            "--evidence",
        ])
        .arg(fixture.0.join("evidence"))
        .env("GITHUB_OUTPUT", fixture.0.join("outputs"))
        .env_remove("GITHUB_STEP_SUMMARY");
    if let Some(destination) = feedback {
        command.arg("--feedback-output").arg(destination);
    }
    command.output().unwrap()
}

#[test]
fn actual_aggregate_cli_exports_candidate_feedback_without_claiming_green() {
    let (fixture, input) = package("repair-1", false);
    evidence(&fixture, &input, "dense", "failure");
    evidence(&fixture, &input, "hybrid", "success");
    let destination = fixture.0.join("feedback");
    let result = aggregate_cli(&fixture, &input, Some(&destination));
    assert!(!result.status.success());
    let metadata: Value =
        serde_json::from_slice(&fs::read(destination.join("feedback.json")).unwrap()).unwrap();
    assert_eq!(metadata["state"], "candidate_repairable");
    assert_eq!(metadata["candidate_failures"], json!(["dense"]));
    assert!(destination.join("dense/results.jsonl").is_file());
    assert!(!destination.join("hybrid").exists());
    let outputs = fs::read_to_string(fixture.0.join("outputs")).unwrap();
    assert!(outputs.contains("feedback_ready=true\n"));
    assert!(outputs.contains("repairable=true\n"));
    assert!(!outputs.contains("green=true"));
    assert!(outputs.contains("retry_matrix={\"include\":[]}\n"));
}

#[test]
fn actual_aggregate_cli_refuses_unsafe_feedback_without_emitting_ready() {
    let (fixture, input) = package("repair-1", false);
    evidence(&fixture, &input, "dense", "failure");
    evidence(&fixture, &input, "hybrid", "success");
    std::os::unix::fs::symlink("results.jsonl", fixture.0.join("evidence/dense/extra.log"))
        .unwrap();
    fs::write(fixture.0.join("outputs"), b"previous=true\n").unwrap();
    let destination = fixture.0.join("feedback");
    let result = aggregate_cli(&fixture, &input, Some(&destination));
    assert!(!result.status.success());
    assert!(!destination.exists());
    assert_eq!(
        fs::read(fixture.0.join("outputs")).unwrap(),
        b"previous=true\n"
    );
}

#[test]
fn actual_aggregate_cli_preserves_legacy_failure_and_missing_infrastructure() {
    let (fixture, input) = package("repair-1", false);
    evidence(&fixture, &input, "dense", "failure");
    let legacy = aggregate_cli(&fixture, &input, None);
    assert!(!legacy.status.success());
    assert!(!fixture.0.join("outputs").exists());
    let destination = fixture.0.join("feedback");
    let result = aggregate_cli(&fixture, &input, Some(&destination));
    assert!(!result.status.success());
    let metadata: Value =
        serde_json::from_slice(&fs::read(destination.join("feedback.json")).unwrap()).unwrap();
    assert_eq!(metadata["state"], "infrastructure_retryable");
    assert_eq!(metadata["infrastructure_failures"], json!(["hybrid"]));
    assert!(!destination.join("hybrid").exists());
    assert!(
        fs::read_to_string(fixture.0.join("outputs"))
            .unwrap()
            .contains("repairable=false\n")
    );
    let outputs = fs::read_to_string(fixture.0.join("outputs")).unwrap();
    let matrix = outputs
        .lines()
        .find_map(|line| line.strip_prefix("retry_matrix="))
        .unwrap();
    let matrix: Value = serde_json::from_str(matrix).unwrap();
    assert_eq!(matrix["include"].as_array().unwrap().len(), 1);
    let row = &matrix["include"][0];
    assert_eq!(row["families"], "hybrid");
    assert_eq!(row["shard_index"], 1);
    assert_eq!(row["id"], "family-hybrid");
    assert_eq!(row["memory_tier"], "accelerator-memory-256plus");
    assert_eq!(row["minimum_runner_memory_gib"], 256);
    assert_eq!(row["resident_model_bytes"], 8_u64 * 1024 * 1024 * 1024);
    assert_eq!(row["estimated_peak_bytes"], 16_u64 * 1024 * 1024 * 1024);
    assert_eq!(row["historical_row_owner"], json!({"opaque":"retained"}));
    let plan_path = fixture.0.join("package/plan.json");
    assert_eq!(fs::read(&plan_path).unwrap(), super::scheduled_plan());
}

fn reconcile_cli(fixture: &Fixture, input: &Value, graph: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "canary-receipts", "reconcile", "--package"])
        .arg(fixture.0.join("package"))
        .args([
            "--identity",
            input["identity_sha256"].as_str().unwrap(),
            "--run-id",
            "123",
            "--run-attempt",
            "5",
            "--controller-revision",
            input["controller_revision"].as_str().unwrap(),
            "--family-result",
            graph,
            "--previous-feedback",
        ])
        .arg(fixture.0.join("feedback"))
        .arg("--evidence")
        .arg(fixture.0.join("evidence"))
        .arg("--feedback-output")
        .arg(fixture.0.join("reconciled-feedback"))
        .env("GITHUB_OUTPUT", fixture.0.join("reconcile-outputs"))
        .env_remove("GITHUB_STEP_SUMMARY")
        .output()
        .unwrap()
}

#[test]
fn actual_reconcile_cli_publishes_retained_candidate_after_infrastructure_pass() {
    let (fixture, mut input) = package("repair-1", false);
    evidence(&fixture, &input, "dense", "failure");
    let destination = fixture.0.join("feedback");
    assert!(
        !aggregate_cli(&fixture, &input, Some(&destination))
            .status
            .success()
    );
    fs::remove_dir_all(fixture.0.join("evidence")).unwrap();
    input["run_attempt"] = json!("5");
    evidence(&fixture, &input, "hybrid", "success");
    let result = reconcile_cli(&fixture, &input, "success");
    assert!(!result.status.success());
    let metadata: Value = serde_json::from_slice(
        &fs::read(fixture.0.join("reconciled-feedback/feedback.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(metadata["state"], "candidate_repairable");
    assert_eq!(metadata["candidate_failures"], json!(["dense"]));
    assert!(
        fixture
            .0
            .join("reconciled-feedback/dense/results.jsonl")
            .exists()
    );
    let outputs = fs::read_to_string(fixture.0.join("reconcile-outputs")).unwrap();
    assert!(outputs.contains("repairable=true\n"));
    assert!(outputs.contains("feedback_ready=true\n"));
    assert!(!outputs.contains("green=true"));
}

#[test]
fn actual_reconcile_cli_emits_green_only_after_successful_retry_graph() {
    for graph in ["success", "failure"] {
        let (fixture, mut input) = package("repair-1", false);
        evidence(&fixture, &input, "hybrid", "success");
        let destination = fixture.0.join("feedback");
        assert!(
            !aggregate_cli(&fixture, &input, Some(&destination))
                .status
                .success()
        );
        fs::remove_dir_all(fixture.0.join("evidence")).unwrap();
        input["run_attempt"] = json!("5");
        evidence(&fixture, &input, "dense", "success");
        let result = reconcile_cli(&fixture, &input, graph);
        assert_eq!(
            result.status.success(),
            graph == "success",
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let outputs = fs::read_to_string(fixture.0.join("reconcile-outputs")).unwrap();
        assert_eq!(outputs.contains("green=true\n"), graph == "success");
        assert!(!fixture.0.join("reconciled-feedback").exists());
        assert!(outputs.contains("feedback_ready=false\n"));
    }
}

#[test]
fn actual_reconcile_cli_refuses_corrupt_feedback_before_output() {
    let (fixture, mut input) = package("repair-1", false);
    evidence(&fixture, &input, "hybrid", "success");
    let destination = fixture.0.join("feedback");
    assert!(
        !aggregate_cli(&fixture, &input, Some(&destination))
            .status
            .success()
    );
    fs::write(destination.join("feedback.json"), b"{}\n").unwrap();
    input["run_attempt"] = json!("5");
    fs::write(fixture.0.join("reconcile-outputs"), b"previous=true\n").unwrap();
    let result = reconcile_cli(&fixture, &input, "success");
    assert!(!result.status.success());
    assert_eq!(
        fs::read(fixture.0.join("reconcile-outputs")).unwrap(),
        b"previous=true\n"
    );
    assert!(!fixture.0.join("reconciled-feedback").exists());
}

#[test]
fn actual_aggregate_cli_emits_complete_optional_green_outputs_without_feedback() {
    let (fixture, input) = package("repair-1", false);
    evidence(&fixture, &input, "dense", "success");
    evidence(&fixture, &input, "hybrid", "success");
    let destination = fixture.0.join("feedback");
    let result = aggregate_cli(&fixture, &input, Some(&destination));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!destination.exists());
    let output = fs::read_to_string(fixture.0.join("outputs")).unwrap();
    for required in [
        "green=true\n",
        "state=green\n",
        "repairable=false\n",
        "feedback_ready=false\n",
        "retry_matrix={\"include\":[]}\n",
    ] {
        assert!(output.contains(required), "{output}");
    }
}

#[test]
fn actual_aggregate_cli_terminal_outputs_override_stale_ready_flags() {
    let (fixture, input) = package("repair-1", false);
    evidence(&fixture, &input, "dense", "failure");
    evidence(&fixture, &input, "hybrid", "success");
    fs::write(
        fixture.0.join("evidence/dense/results.jsonl"),
        b"corrupt results",
    )
    .unwrap();
    fs::write(
        fixture.0.join("outputs"),
        b"green=true\nstate=green\nrepairable=true\nfeedback_ready=true\n",
    )
    .unwrap();
    let destination = fixture.0.join("feedback");
    let result = aggregate_cli(&fixture, &input, Some(&destination));
    assert!(!result.status.success());
    assert!(!destination.exists());
    let output = fs::read_to_string(fixture.0.join("outputs")).unwrap();
    for (key, expected) in [
        ("green", "false"),
        ("state", "terminal_contract"),
        ("repairable", "false"),
        ("feedback_ready", "false"),
        ("failure_class", "contract"),
        ("failure_stage", "family-certification"),
        ("retry_matrix", "{\"include\":[]}"),
    ] {
        let last = output
            .lines()
            .filter_map(|line| line.split_once('='))
            .rfind(|(name, _)| *name == key)
            .map(|(_, value)| value);
        assert_eq!(last, Some(expected), "{key}: {output}");
    }
}
