use std::process::Command;

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
        .args(["benchmark", "run-gpu", "--backend", "cuda"])
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
