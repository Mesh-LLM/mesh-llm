//! Actual offline waiting-prefix command status and published evidence.
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};

fn request_fixture_accept(listener: &std::net::TcpListener) -> std::net::TcpStream {
    use std::time::{Duration, Instant};
    listener.set_nonblocking(true).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match listener.accept() {
            Ok((connection, _)) => {
                connection
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                return connection;
            }
            Err(error)
                if error.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline =>
            {
                std::thread::sleep(Duration::from_millis(5))
            }
            Err(error) => panic!("HTTP fixture admission failed: {error}"),
        }
    }
}

fn request_fixture_body(connection: &mut std::net::TcpStream) -> Value {
    use std::io::Read;
    let mut bytes = Vec::new();
    let mut chunk = [0; 1024];
    loop {
        let count = connection.read(&mut chunk).unwrap();
        assert!(count > 0, "incomplete request");
        bytes.extend_from_slice(&chunk[..count]);
        assert!(bytes.len() < 1024 * 1024, "oversized fixture request");
        if let Some(end) = bytes.windows(4).position(|b| b == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]).to_lowercase();
            let length: usize = headers
                .lines()
                .find_map(|l| l.strip_prefix("content-length: "))
                .unwrap()
                .parse()
                .unwrap();
            if bytes.len() >= end + 4 + length {
                return serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap();
            }
        }
    }
}

fn request_cli(
    state: &Path,
    address: std::net::SocketAddr,
    prompts: Value,
    timeout: f64,
) -> std::process::Output {
    fs::write(state.join("requests-input.json"),serde_json::to_vec(&serde_json::json!({
        "schema_version":1,"round":1,"version":"old","base_url":format!("http://{address}/v1"),
        "model":"fixture","output_tokens":2,"request_timeout_secs":timeout,"stagger_ms":5.0,"prompts":prompts
    })).unwrap()).unwrap();
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(state)
        .env_clear()
        .args([
            "automation",
            "waiting-prefix",
            "execute-requests",
            "--input",
            "requests-input.json",
            "--output",
            "requests.json",
        ])
        .output()
        .unwrap()
}

fn request_fixture_responses(listener: std::net::TcpListener) -> Vec<Value> {
    use std::io::Write;
    let mut bodies = Vec::new();
    for _ in 0..2 {
        let mut connection = request_fixture_accept(&listener);
        let body = request_fixture_body(&mut connection);
        let good = body["messages"][0]["content"] == "success";
        bodies.push(body);
        let data = if good {
            "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: {\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":30}}}\n\ndata: [DONE]\n\n"
        } else {
            "failure"
        };
        write!(connection,"HTTP/1.1 {}\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",if good {"200 OK"} else {"500 Internal Server Error"},data.len(),data).unwrap();
    }
    bodies
}

#[test]
fn native_request_phase_preserves_measured_usage_and_http_failure_in_sorted_evidence() {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || request_fixture_responses(listener));
    let state = tempfile::tempdir().unwrap();
    let result = request_cli(
        state.path(),
        address,
        serde_json::json!([{"family":"one","prompt":"success"},{"family":"two","prompt":"fail"}]),
        2.0,
    );
    let bodies = server.join().unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(bodies.len(), 2);
    for body in bodies {
        assert_eq!(body["model"], "fixture");
        assert_eq!(body["seed"], 0);
        assert_eq!(body["stream_options"]["include_usage"], true);
    }
    let phase: Value =
        serde_json::from_slice(&fs::read(state.path().join("requests.json")).unwrap()).unwrap();
    assert_eq!(phase["requests"][0]["request_id"], 0);
    assert_eq!(phase["requests"][0]["tokens_predicted"], 2);
    assert_eq!(phase["requests"][0]["cached_tokens"], 30);
    assert!(phase["requests"][0]["ttft_ms"].as_f64().unwrap() >= 0.0);
    assert_eq!(phase["requests"][1]["request_id"], 1);
    assert!(
        phase["requests"][1]["error"]
            .as_str()
            .unwrap()
            .contains("HTTP 500")
    );
}

#[test]
fn native_request_phase_enforces_deadline_and_refuses_invalid_input_before_publication() {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let connection = request_fixture_accept(&listener);
        // The client can time out before sending headers. Hold the socket
        // without responding so the fixture tests the deadline itself.
        std::thread::sleep(std::time::Duration::from_millis(250));
        drop(connection);
    });
    let state = tempfile::tempdir().unwrap();
    let result = request_cli(
        state.path(),
        address,
        serde_json::json!([{"family":"one","prompt":"wait"}]),
        0.05,
    );
    server.join().unwrap();
    assert_eq!(result.status.code(), Some(1));
    let prior = fs::read(state.path().join("requests.json")).unwrap();
    let phase: Value = serde_json::from_slice(&prior).unwrap();
    assert!(
        phase["requests"][0]["error"]
            .as_str()
            .unwrap()
            .contains("deadline")
    );
    fs::write(state.path().join("requests-input.json"), b"{}").unwrap();
    let refused = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(state.path())
        .env_clear()
        .args([
            "automation",
            "waiting-prefix",
            "execute-requests",
            "--input",
            "requests-input.json",
            "--output",
            "requests.json",
        ])
        .output()
        .unwrap();
    assert_eq!(refused.status.code(), Some(1));
    assert_eq!(fs::read(state.path().join("requests.json")).unwrap(), prior);
}

#[test]
fn native_workload_plan_binds_checked_in_shape_and_preserves_prior_output_on_refusal() {
    let directory = tempfile::tempdir().unwrap();
    let catalog =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../evals/skippy-scheduler-fixtures.json");
    let document: Value = serde_json::from_slice(&fs::read(&catalog).unwrap()).unwrap();
    let model = &document["profiles"]["warm-affinity"]["model"];
    let output = directory.path().join("plan.json");
    let invoke = |id: &str| {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(directory.path())
            .env_clear()
            .args(["automation", "waiting-prefix", "plan", "--catalog"])
            .arg(&catalog)
            .args([
                "--profile",
                "warm-affinity",
                "--model-id",
                id,
                "--model-sha256",
                model["sha256"].as_str().unwrap(),
                "--output",
            ])
            .arg(&output)
            .output()
            .unwrap()
    };
    let result = invoke(model["id"].as_str().unwrap());
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let bytes = fs::read(&output).unwrap();
    let plan: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        plan["workload"],
        document["profiles"]["warm-affinity"]["workload"]
    );
    assert_eq!(plan["successful_requests_per_binary"], 24);
    assert_eq!(plan["requests_per_round"], 6);
    assert!(plan["cache_seed"].is_null());
    assert_eq!(invoke("wrong-model").status.code(), Some(1));
    assert_eq!(fs::read(output).unwrap(), bytes);
}

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
