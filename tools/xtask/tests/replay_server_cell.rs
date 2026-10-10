#[cfg(unix)]
#[test]
fn server_cell_owns_server_and_worker_until_complete_cleanup() {
    runtime_context_case(131072, 0, true);
}

#[cfg(unix)]
#[test]
fn insufficient_runtime_context_retains_snapshot_without_measuring() {
    runtime_context_case(262144, 0, false);
}

#[cfg(unix)]
#[test]
fn insufficient_session_prompt_coverage_retains_negative_cell_acceptance() {
    runtime_context_case(131072, 32768, false);
}

#[cfg(unix)]
fn runtime_context_case(required: u64, minimum_prompt: u64, successful: bool) {
    use sha2::{Digest, Sha256};
    use std::{net::TcpListener, os::unix::fs::PermissionsExt, process::Command};
    let state = tempfile::tempdir().unwrap();
    let model = state.path().join("model.gguf");
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(2_u64.to_le_bytes());
    for (key, value) in [
        ("general.architecture", Some("fixture")),
        ("fixture.context_length", None),
    ] {
        bytes.extend(u64::try_from(key.len()).unwrap().to_le_bytes());
        bytes.extend(key.as_bytes());
        if let Some(value) = value {
            bytes.extend(8_u32.to_le_bytes());
            bytes.extend(u64::try_from(value.len()).unwrap().to_le_bytes());
            bytes.extend(value.as_bytes());
        } else {
            bytes.extend(4_u32.to_le_bytes());
            bytes.extend(262144_u32.to_le_bytes());
        }
    }
    let digest = hex::encode(Sha256::digest(&bytes));
    std::fs::write(&model, bytes).unwrap();
    let reservation = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = reservation.local_addr().unwrap().port();
    let binary = state.path().join("server-fixture");
    std::fs::write(
        &binary,
        format!(
            "#!/bin/sh\nexec '{}' --port {port} \"$@\"\n",
            env!("CARGO_BIN_EXE_laya-product-fixture")
        ),
    )
    .unwrap();
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
    let workload = state.path().join("workload.json");
    let requests = state.path().join("requests.jsonl");
    let summary = state.path().join("summary.json");
    let log = state.path().join("server.log");
    std::fs::write(&workload, serde_json::to_vec(&serde_json::json!({
        "trajectories":[{"session_id":"s","source_dataset":"fixture","agent_framework":"fixture",
            "messages":[{"role":"user","content":"task"},{"role":"assistant","content":"recorded"}]}],
        "model":"pending","base_url":format!("http://127.0.0.1:{port}/v1"),"concurrency":1,
        "max_output_tokens":2048,"request_timeout_seconds":2,"qualification_probe":true,
        "runtime_context":{"required_tokens":required,"output":state.path().join("runtime.json")},
        "model_pin":{"sha256":digest,"minimum_context_tokens":required,"output":state.path().join("model-identity.json")},
        "eligibility":{"context_tokens":1,"maximum_output_tokens":2048,"minimum_session_prompt_tokens":minimum_prompt}
    })).unwrap()).unwrap();
    drop(reservation);
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "server-cell", "--binary"])
        .arg(binary)
        .arg("--native-runtime-root")
        .arg(state.path())
        .arg("--model")
        .arg(model)
        .arg("--workload")
        .arg(workload)
        .arg("--requests-output")
        .arg(&requests)
        .arg("--summary-output")
        .arg(&summary)
        .arg("--server-log")
        .arg(log)
        .args(["--timeout", "10", "--startup-timeout", "3"])
        .output()
        .unwrap();
    assert!(
        result.status.success() == successful,
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let runtime: serde_json::Value =
        serde_json::from_slice(&std::fs::read(state.path().join("runtime.json")).unwrap()).unwrap();
    assert_eq!(runtime["models"][0]["context_length"], 131072);
    if successful {
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(summary).unwrap()).unwrap();
        assert_eq!(report["acceptance"]["passed"], true);
        assert_eq!(
            std::fs::read_to_string(requests).unwrap().lines().count(),
            1
        );
    } else if minimum_prompt == 0 {
        assert!(!requests.exists());
        assert!(!summary.exists());
    } else {
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(summary).unwrap()).unwrap();
        assert_eq!(report["acceptance"]["passed"], false);
        assert_eq!(report["eligibility"]["context_tokens"], 131072);
        assert_eq!(report["eligibility"]["passed"], false);
        assert_eq!(
            std::fs::read_to_string(requests).unwrap().lines().count(),
            1
        );
    }
    assert!(TcpListener::bind(("127.0.0.1", port)).is_ok());
}

#[cfg(unix)]
#[test]
fn server_early_exit_rejects_before_cell_success() {
    use std::{os::unix::fs::PermissionsExt, process::Command};
    let state = tempfile::tempdir().unwrap();
    let reservation = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = reservation.local_addr().unwrap().port();
    let binary = state.path().join("server-fixture");
    std::fs::write(&binary, "#!/bin/sh\nexit 7\n").unwrap();
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
    let workload = state.path().join("workload.json");
    std::fs::write(&workload, serde_json::to_vec(&serde_json::json!({"base_url":format!("http://127.0.0.1:{port}/v1"),"model":"fixture","trajectories":[{"session_id":"s","source_dataset":"fixture","agent_framework":"fixture","messages":[{"role":"user","content":"task"},{"role":"assistant","content":"answer"}]}],"concurrency":1,"max_output_tokens":1,"request_timeout_seconds":1})).unwrap()).unwrap();
    let summary = state.path().join("summary.json");
    drop(reservation);
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "server-cell", "--binary"])
        .arg(binary)
        .arg("--native-runtime-root")
        .arg(state.path())
        .args(["--model", "fixture.gguf", "--workload"])
        .arg(workload)
        .arg("--requests-output")
        .arg(state.path().join("requests.jsonl"))
        .arg("--summary-output")
        .arg(&summary)
        .arg("--server-log")
        .arg(state.path().join("server.log"))
        .args(["--timeout", "5", "--startup-timeout", "1"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("replay server/cell failed"),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!summary.exists());
}

#[cfg(unix)]
#[test]
fn warmup_runs_only_the_requested_initial_turns() {
    warmup_then_cells(false);
}

#[cfg(unix)]
#[test]
fn failed_measured_cell_retains_later_cell_evidence_and_fails_pass() {
    warmup_then_cells(true);
}

#[cfg(unix)]
fn warmup_then_cells(failed: bool) {
    use std::{net::TcpListener, os::unix::fs::PermissionsExt, process::Command};
    let state = tempfile::tempdir().unwrap();
    let reservation = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = reservation.local_addr().unwrap().port();
    let binary = state.path().join("server-fixture");
    std::fs::write(
        &binary,
        format!(
            "#!/bin/sh\nexec '{}' --port {port}\n",
            env!("CARGO_BIN_EXE_laya-product-fixture")
        ),
    )
    .unwrap();
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
    let workload = state.path().join("workload.json");
    let requests = state.path().join("requests.jsonl");
    let summary = state.path().join("summary.json");
    std::fs::write(&workload, serde_json::to_vec(&serde_json::json!({
        "trajectories":[{"session_id":"warmup","source_dataset":"fixture","agent_framework":"fixture",
            "messages":[{"role":"user","content":"task"},{"role":"assistant","content":"first"},
                {"role":"user","content":"next"},{"role":"assistant","content":"second"}]}],
        "model":"pending","base_url":format!("http://127.0.0.1:{port}/v1"),"concurrency":1,
        "max_output_tokens":2048,"request_timeout_seconds":2,"warmup_turns":1,
        "following_cells":[{
            "requests_output":state.path().join("first.jsonl"),
            "summary_output":state.path().join("first.json"),
            "workload":{
                "trajectories":[{"session_id":if failed {"failed"} else {"first"},"source_dataset":"fixture","agent_framework":"fixture",
                    "messages":[{"role":"user","content":"task"},{"role":"assistant","content":"first"}]}],
                "model":"pending","base_url":format!("http://127.0.0.1:{port}/v1"),"concurrency":1,
                "max_output_tokens":2048,"request_timeout_seconds":2
            }
        }, {
            "requests_output":state.path().join("measured.jsonl"),
            "summary_output":state.path().join("measured.json"),
            "workload":{
                "trajectories":[{"session_id":"measured","source_dataset":"fixture","agent_framework":"fixture",
                    "messages":[{"role":"user","content":"task"},{"role":"assistant","content":"first"},
                        {"role":"user","content":"next"},{"role":"assistant","content":"second"}]}],
                "model":"pending","base_url":format!("http://127.0.0.1:{port}/v1"),"concurrency":1,
                "max_output_tokens":2048,"request_timeout_seconds":2,"minimum_recurrent_restored_tokens":1,
                "measured_prefix":{"expected_prompt_tokens":{"measured:0":40,"measured:1":40},"require_later_turn_reuse":true}
            }
        }]
    })).unwrap()).unwrap();
    drop(reservation);
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "server-cell", "--binary"])
        .arg(binary)
        .arg("--native-runtime-root")
        .arg(state.path())
        .args(["--model", "fixture.gguf", "--workload"])
        .arg(workload)
        .arg("--requests-output")
        .arg(&requests)
        .arg("--summary-output")
        .arg(&summary)
        .arg("--server-log")
        .arg(state.path().join("server.log"))
        .args(["--timeout", "10", "--startup-timeout", "3"])
        .output()
        .unwrap();
    assert!(
        result.status.success() != failed,
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(summary).unwrap()).unwrap();
    assert_eq!(report["acceptance"]["passed"], true);
    assert_eq!(report["acceptance"]["expected_turns"], 1);
    assert_eq!(report["completeness"]["passed"], false);
    let rows = std::fs::read_to_string(requests).unwrap();
    assert_eq!(rows.lines().count(), 1);
    let row: serde_json::Value = serde_json::from_str(rows.trim()).unwrap();
    assert_eq!(row["warmup"], true);
    assert_eq!(row["request_id"], "warmup:0");
    let measured: serde_json::Value =
        serde_json::from_slice(&std::fs::read(state.path().join("measured.json")).unwrap())
            .unwrap();
    assert_eq!(measured["acceptance"]["passed"], true);
    assert_eq!(measured["recurrent_state"]["passed"], true);
    let first: serde_json::Value =
        serde_json::from_slice(&std::fs::read(state.path().join("first.json")).unwrap()).unwrap();
    assert_eq!(first["acceptance"]["passed"], !failed);
    assert_eq!(
        std::fs::read_to_string(state.path().join("measured.jsonl"))
            .unwrap()
            .lines()
            .count(),
        2
    );
}
