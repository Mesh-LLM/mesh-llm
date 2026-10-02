use std::process::Command;

#[test]
fn recurrent_log_reader_deduplicates_raw_lines_but_retains_distinct_encodings() {
    let state = tempfile::tempdir().unwrap();
    let input = state.path().join("input.json");
    let output = state.path().join("report.json");
    let log = state.path().join("server.log");
    let event = serde_json::json!({"event":"stage.openai_kv_lookup_decision","start_time_unix_nanos":1,
        "attributes":{"openai.prompt_cache_key":"s","skippy.kv.decision":"exact_hit",
            "skippy.exact_cache.payload_kind":"kv-recurrent","skippy.exact_cache.restored_tokens":39000}});
    let compact = serde_json::to_string(&event).unwrap();
    let distinct = format!(" {compact}");
    std::fs::write(&log, format!("noise\n{compact}\n{compact}\n{distinct}\n")).unwrap();
    let document = serde_json::json!({
        "trajectories":[{"session_id":"s","messages":[{"role":"assistant"},{"role":"assistant"}]}],
        "requests":[{"session_id":"s","request_id":"s:0","assistant_turn":0,"prompt_tokens":40000},
            {"session_id":"s","request_id":"s:1","assistant_turn":1,"prompt_tokens":41000}],
        "runtime":{"models":[{"context_length":131072}]},"required_context":131072,
        "recurrent":{"minimum_restored_tokens":32768,"log_paths":[log,log]}
    });
    std::fs::write(&input, serde_json::to_vec(&document).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "session-evidence", "--input"])
        .arg(input)
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(
        report["recurrent_state"]["turns"].as_array().unwrap().len(),
        2
    );
}

#[test]
fn recorded_requests_cli_preserves_ordered_prefixes() {
    let state = tempfile::tempdir().unwrap();
    let input = state.path().join("trajectory.json");
    let output = state.path().join("requests.json");
    let trajectory = serde_json::json!({
        "session_id":"s", "source_dataset":"capture", "agent_framework":"goose",
        "messages":[{"role":"user","content":"task"},
            {"role":"assistant","content":"recorded first"},
            {"role":"tool","content":"recorded observation","tool_call_id":"call"},
            {"role":"assistant","content":"recorded final"}]
    });
    std::fs::write(&input, serde_json::to_vec(&trajectory).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "automation",
            "replay-matrix",
            "recorded-requests",
            "--input",
        ])
        .arg(input)
        .arg("--output")
        .arg(&output)
        .args(["--model", "target"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let requests: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(requests.as_array().unwrap().len(), 2);
    assert_eq!(
        requests[1]["body"]["messages"][1]["content"],
        "recorded first"
    );
    assert_eq!(
        requests[1]["body"]["messages"][2]["content"],
        "recorded observation"
    );
    assert_eq!(requests[1]["body"]["model"], "target");
}

#[test]
fn empty_workload_fails_and_writes_a_negative_report() {
    let state = tempfile::tempdir().unwrap();
    let input = state.path().join("input.json");
    let output = state.path().join("report.json");
    std::fs::write(&input, br#"{"trajectories":[],"requests":[],"runtime":{"models":[{"context_length":131072}]},"required_context":131072}"#).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "session-evidence", "--input"])
        .arg(input)
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(report["passed"], false);
    assert!(!report["problems"].as_array().unwrap().is_empty());
}

#[test]
fn context_fit_failure_reaches_top_level_report() {
    let state = tempfile::tempdir().unwrap();
    let input = state.path().join("input.json");
    let output = state.path().join("report.json");
    let document = serde_json::json!({
        "trajectories":[{"session_id":"s","messages":[{"role":"assistant","content":"answer"}]}],
        "requests":[{"session_id":"s","request_id":"s:0","prompt_tokens":131070}],
        "runtime":{"models":[{"context_length":131072}]},"required_context":131072,
        "eligibility":{"context_tokens":131072,"maximum_output_tokens":2048,"minimum_session_prompt_tokens":32768}
    });
    std::fs::write(&input, serde_json::to_vec(&document).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "session-evidence", "--input"])
        .arg(input)
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(report["passed"], false);
    assert_eq!(report["eligibility"]["passed"], false);
    assert!(!report["problems"].as_array().unwrap().is_empty());
}

#[test]
fn evidence_cli_writes_success_and_retains_failure_report() {
    let state = tempfile::tempdir().unwrap();
    let input = state.path().join("input.json");
    let output = state.path().join("report.json");
    let mut document = serde_json::json!({
        "trajectories": [{"session_id":"s", "messages":[{"role":"user"},{"role":"assistant"}]}],
        "requests": [{"session_id":"s","request_id":"s:0"}],
        "runtime":{"models":[{"context_length":131072}]},
        "required_context":131072
    });
    std::fs::write(&input, serde_json::to_vec(&document).unwrap()).unwrap();
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "replay-matrix", "session-evidence", "--input"])
            .arg(&input)
            .arg("--output")
            .arg(&output)
            .output()
            .unwrap()
    };
    assert!(run().status.success());
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    assert_eq!(report["expected_request_ids"], serde_json::json!(["s:0"]));
    document["runtime"]["models"][0]["context_length"] = 32768.into();
    std::fs::write(&input, serde_json::to_vec(&document).unwrap()).unwrap();
    assert!(!run().status.success());
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(report["passed"], false);
}

#[cfg(unix)]
#[test]
fn retained_reader_adapter_bounds_an_explicit_executable_and_forwards_selection() {
    use std::os::unix::fs::PermissionsExt;
    let state = tempfile::tempdir().unwrap();
    let executable = state.path().join("reader-fixture");
    std::fs::write(&executable, "#!/bin/sh\ncase \"$1\" in */evals/agentic-trajectory-manifest.py) ;; *) exit 8;; esac\nshift\n[ \"$1\" = '--dataset-file' ] && [ \"$2\" = 'input.parquet' ] || exit 9\nexit 0\n").unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "automation",
            "replay-matrix",
            "trajectory-reader",
            "--python",
        ])
        .arg(&executable)
        .args(["--timeout", "2", "--dataset-file", "input.parquet"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "automation",
            "replay-matrix",
            "trajectory-reader",
            "--python",
            "relative",
            "--timeout",
            "2",
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
}

#[cfg(unix)]
#[test]
fn retained_reader_deadline_terminates_the_fixture_tree() {
    use std::os::unix::fs::PermissionsExt;
    let state = tempfile::tempdir().unwrap();
    let executable = state.path().join("reader-fixture");
    std::fs::write(&executable, "#!/bin/sh\nsleep 60 &\nwait\n").unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "automation",
            "replay-matrix",
            "trajectory-reader",
            "--python",
        ])
        .arg(executable)
        .args(["--timeout", "1", "--dataset-file", "fixture.parquet"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("Deadline"));
}
