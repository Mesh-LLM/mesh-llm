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
fn gpus_json_without_native_runtime_keeps_json_stdout_and_warns_on_stderr() {
    // Isolate executable-adjacent bundles as well as user state: `--json`
    // must stay machine-readable, and the OS-survey notice belongs on stderr.
    let temp = TempDir::new().expect("tempdir should be created");
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).expect("isolated home should be created");
    let executable = temp
        .path()
        .join(format!("mesh-llm{}", std::env::consts::EXE_SUFFIX));
    std::fs::copy(env!("CARGO_BIN_EXE_mesh-llm"), &executable)
        .expect("isolated executable should be copied");
    let output = Command::new(&executable)
        .args(["gpus", "--json"])
        .env("HOME", &home)
        .env("XDG_CONFIG_HOME", home.join("xdg-config"))
        .env("XDG_CACHE_HOME", home.join("xdg-cache"))
        .env("XDG_DATA_HOME", home.join("xdg-data"))
        .env("MESH_LLM_NO_RUNTIME_DOWNLOAD", "1")
        .env(
            "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR",
            home.join("runtime-cache"),
        )
        .env_remove("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR")
        .output()
        .expect("mesh-llm command should run");

    assert!(
        output.status.success(),
        "gpus must exit 0 without a native runtime, got: {:?}",
        output.status
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    serde_json::from_str::<serde_json::Value>(&stdout)
        .expect("gpus --json stdout should parse as JSON");
    if cfg!(feature = "dynamic-native-runtime") {
        // Without the feature the loader is a no-op that never fails, so no
        // notice is printed; the notice is specific to the dynamic build.
        let stderr = String::from_utf8(output.stderr).expect("stderr should be utf-8");
        assert!(
            stderr.contains("Native runtime not loaded"),
            "stderr should carry the OS-survey fallback notice, got: {stderr:?}"
        );
    }
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
            || stderr.contains("benchmark backend")
            || stderr.contains("cuda native runtime with a benchmark tool"),
        "stderr should include the benchmark backend or native-runtime error, got: {stderr:?}"
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
