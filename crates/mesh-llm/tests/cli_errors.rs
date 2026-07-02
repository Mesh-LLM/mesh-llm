use std::process::Command;

use tempfile::TempDir;

#[test]
fn early_command_errors_are_visible_on_stderr() {
    let output = Command::new(env!("CARGO_BIN_EXE_mesh-llm"))
        .args(["auth", "revoke-node"])
        .output()
        .expect("mesh-llm command should run");

    assert!(!output.status.success(), "command must exit non-zero");
    assert!(
        output.stdout.is_empty(),
        "fatal error output should not be routed to stdout"
    );
    let stderr = String::from_utf8(output.stderr).expect("stderr should be utf-8");
    assert!(
        stderr.contains("Pass --cert-id, --node-id, or both."),
        "stderr should include the fatal error, got: {stderr:?}"
    );
}

#[test]
fn hidden_gpu_benchmark_backend_errors_are_visible_on_stderr() {
    let output = Command::new(env!("CARGO_BIN_EXE_mesh-llm"))
        .args(["gpus", "run-benchmark", "--backend", "cuda"])
        .output()
        .expect("mesh-llm command should run");

    assert!(!output.status.success(), "command must exit non-zero");
    assert!(
        output.stdout.is_empty(),
        "fatal benchmark errors should not be routed to stdout"
    );
    let stderr = String::from_utf8(output.stderr).expect("stderr should be utf-8");
    assert!(
        stderr.contains("CUDA benchmark backend was not compiled")
            || stderr.contains("benchmark backend"),
        "stderr should include the benchmark backend error, got: {stderr:?}"
    );
}

#[test]
fn benchmark_tune_missing_target_names_model_and_reason() {
    let temp = TempDir::new().expect("tempdir should be created");
    let config_path = temp.path().join("config.toml");
    let missing_target = "definitely-missing-benchmark-tune-target";
    let output = Command::new(env!("CARGO_BIN_EXE_mesh-llm"))
        .args([
            "--config",
            config_path.to_str().expect("config path should be utf-8"),
            "benchmark",
            "tune",
            "--model",
            missing_target,
        ])
        .output()
        .expect("mesh-llm command should run");

    assert!(!output.status.success(), "command must exit non-zero");
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    assert!(
        stdout.contains(missing_target),
        "stdout should include the failed target report, got: {stdout:?}"
    );
    assert!(
        stdout.contains("installed cache ref"),
        "stdout should include the failure reason in the report, got: {stdout:?}"
    );
    let stderr = String::from_utf8(output.stderr).expect("stderr should be utf-8");
    assert!(
        stderr.contains(missing_target),
        "stderr should include the missing model target, got: {stderr:?}"
    );
    assert!(
        stderr.contains("target is not an existing local path or installed cache ref"),
        "stderr should include the failure reason, got: {stderr:?}"
    );
}

#[test]
fn benchmark_tune_remote_only_target_reports_local_only_rejection() {
    let remote_target = "hf://meshllm/example@rev/Q4_K_M/model.gguf";
    let output = Command::new(env!("CARGO_BIN_EXE_mesh-llm"))
        .args(["benchmark", "tune", "--model", remote_target])
        .output()
        .expect("mesh-llm command should run");

    assert!(!output.status.success(), "command must exit non-zero");
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    assert!(
        stdout.contains(remote_target),
        "stdout should include the rejected remote target report, got: {stdout:?}"
    );
    assert!(
        stdout.contains("local-only") && stdout.contains("download"),
        "stdout should explain the no-download local-only contract, got: {stdout:?}"
    );
    let stderr = String::from_utf8(output.stderr).expect("stderr should be utf-8");
    assert!(
        stderr.contains(remote_target),
        "stderr should include the rejected remote target, got: {stderr:?}"
    );
    assert!(
        stderr.contains("local-only") && stderr.contains("download"),
        "stderr should explain the no-download local-only contract, got: {stderr:?}"
    );
}
