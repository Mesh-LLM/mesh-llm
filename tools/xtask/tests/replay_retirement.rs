#![cfg(unix)]
use serde_json::json;
use std::{os::unix::fs::PermissionsExt, path::Path, process::Command};

fn executable(path: &Path, version: &str) {
    std::fs::write(path, format!("#!/bin/sh\nprintf '{version}\\n'\n")).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
}

#[test]
fn bare_executable_uses_runner_path_instead_of_arm_cwd_decoy() {
    let temporary = tempfile::tempdir().unwrap();
    let search = temporary.path().join("path");
    let cwd = temporary.path().join("cwd");
    std::fs::create_dir(&search).unwrap();
    std::fs::create_dir(&cwd).unwrap();
    executable(&search.join("engine"), "runner-path-version");
    executable(&cwd.join("engine"), "must-not-use-cwd");
    let config = temporary.path().join("engines.json");
    std::fs::write(&config, serde_json::to_vec(&json!({"schema_version":1,"comparison":{"model":"opaque"},"arms":[{"label":"fixture","engine":"llama","executable":"engine","cwd":cwd,"model":"opaque","context_size":32768,"max_concurrency":1}]})).unwrap()).unwrap();
    let output = temporary.path().join("plan.json");
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .env("PATH", &search)
        .args(["automation", "replay-matrix", "external-config", "--config"])
        .arg(&config)
        .args(["--model", "opaque", "--output"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let plan: serde_json::Value = serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(plan["builds"][0]["version"], "runner-path-version");
    assert_eq!(
        plan["builds"][0]["provenance"]["resolved_executable"],
        json!(search.join("engine"))
    );
    assert_eq!(
        plan["external_server_commands"][0][0],
        json!(search.join("engine"))
    );
}

#[test]
fn missing_relative_executable_fails_before_plan_output() {
    let temporary = tempfile::tempdir().unwrap();
    let config = temporary.path().join("engines.json");
    std::fs::write(&config, serde_json::to_vec(&json!({"schema_version":1,"comparison":{"model":"opaque"},"arms":[{"label":"fixture","engine":"llama","executable":"./missing","cwd":temporary.path(),"model":"opaque","context_size":32768,"max_concurrency":1}]})).unwrap()).unwrap();
    let output = temporary.path().join("plan.json");
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "external-config", "--config"])
        .arg(config)
        .args(["--model", "opaque", "--output"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(error.contains("missing"), "{error}");
    assert!(!output.exists());
}

#[test]
fn configured_acceptance_rejects_failed_requests_missing_delta_and_empty_measurements() {
    let temporary = tempfile::tempdir().unwrap();
    let input = temporary.path().join("input.json");
    let output = temporary.path().join("gates.json");
    let baseline = json!({"rows":[{"label":"candidate","concurrency":1,"failed_requests":0,"prompt_tokens_min":19000,"prompt_tokens_max":19000,"cache_pct":75,"content_identity_known":true,"delta_comparable":true,"ttft_p50_seconds_delta_pct":0}],"prompt_token_range":[18000,22000],"min_cache_pct":70,"require_output_match":true,"max_ttft_regression_pct":5});
    for (mutation, succeeds, failed_name) in [
        (None, true, ""),
        (
            Some(("failed_requests", json!(1))),
            false,
            "failed-requests:candidate/c1",
        ),
        (
            Some(("ttft_p50_seconds_delta_pct", json!(null))),
            false,
            "ttft-regression:candidate/c1",
        ),
        (
            Some(("cache_pct", json!(0))),
            false,
            "cached-prompt:candidate/c1",
        ),
    ] {
        let mut document = baseline.clone();
        if let Some((key, value)) = mutation {
            document["rows"][0][key] = value;
        }
        std::fs::write(&input, serde_json::to_vec(&document).unwrap()).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "replay-matrix", "acceptance-gates", "--input"])
            .arg(&input)
            .arg("--output")
            .arg(&output)
            .output()
            .unwrap();
        assert_eq!(
            result.status.success(),
            succeeds,
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(report["passed"], succeeds);
        if !succeeds {
            assert!(
                report["checks"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|check| check["name"] == failed_name && check["passed"] == false)
            );
        }
    }
    std::fs::write(&input, b"{\"rows\":[],\"min_cache_pct\":70}").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "acceptance-gates", "--input"])
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    assert_eq!(report["checks"][0]["name"], "measured-rows");
    assert_eq!(report["passed"], false);
}

fn receive_http(socket: &mut std::net::TcpStream) -> String {
    use std::io::Read;
    socket.set_nonblocking(false).unwrap();
    socket
        .set_read_timeout(Some(std::time::Duration::from_secs(2)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut chunk = [0; 4096];
    loop {
        let count = socket.read(&mut chunk).unwrap();
        assert!(count > 0);
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(end) = bytes.windows(4).position(|value| value == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]);
            let length = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|value| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            if bytes.len() >= end + 4 + length {
                return headers.lines().next().unwrap().to_owned();
            }
        }
    }
}

#[test]
fn runtime_context_missing_then_ready_retries_before_measured_cell() {
    use std::{
        io::Write,
        net::TcpListener,
        time::{Duration, Instant},
    };
    let temporary = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let port = listener.local_addr().unwrap().port();
    let runtime_output = temporary.path().join("runtime.json");
    let workload = temporary.path().join("workload.json");
    let requests = temporary.path().join("requests.jsonl");
    let summary = temporary.path().join("summary.json");
    std::fs::write(&workload,serde_json::to_vec(&json!({"trajectories":[{"session_id":"s","source_dataset":"fixture","agent_framework":"goose","messages":[{"role":"user","content":"task"},{"role":"assistant","content":"answer"}]}],"model":"placeholder","base_url":format!("http://127.0.0.1:{port}/v1"),"concurrency":1,"max_output_tokens":8,"request_timeout_seconds":1,"runtime_context":{"required_tokens":32768,"output":runtime_output}})).unwrap()).unwrap();
    let stopped = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        let server=scope.spawn(|| {
            let deadline=Instant::now()+Duration::from_secs(8);let mut contexts=0;let mut measured=false;
            while Instant::now()<deadline&&!measured&&!stopped.load(std::sync::atomic::Ordering::SeqCst) {
                let (mut socket,_)=match listener.accept(){Ok(value)=>value,Err(error) if error.kind()==std::io::ErrorKind::WouldBlock=>{std::thread::sleep(Duration::from_millis(5));continue;},Err(error)=>panic!("{error}")};
                let request=receive_http(&mut socket);
                let body=if request.contains("/v1/models") {"{\"data\":[{\"id\":\"fixture\"}]}".to_owned()} else if request.contains("/api/runtime") {
                    contexts+=1;if contexts==1 {"{\"models\":[{\"name\":\"fixture\"}]}".to_owned()} else {"{\"models\":[{\"name\":\"fixture\",\"context_length\":32768}]}".to_owned()}
                } else {assert!(request.contains("/v1/chat/completions"),"{request}");assert_eq!(contexts,2);measured=true;"data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":1,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\ndata: [DONE]\n\n".to_owned()};
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).as_bytes()).unwrap();
            }
            if !stopped.load(std::sync::atomic::Ordering::SeqCst) { assert!(measured,"finite server fixture never reached measurement"); }
        });
        let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "replay-matrix", "server-cell-worker", "5"])
            .arg(port.to_string())
            .arg(&workload)
            .arg(&requests)
            .arg(&summary)
            .output()
            .unwrap();
        if !result.status.success() {
            stopped.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        server.join().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    });
    let retained: serde_json::Value =
        serde_json::from_slice(&std::fs::read(runtime_output).unwrap()).unwrap();
    assert_eq!(retained["models"][0]["context_length"], 32768);
    assert_eq!(
        std::fs::read_to_string(requests).unwrap().lines().count(),
        1
    );
}
