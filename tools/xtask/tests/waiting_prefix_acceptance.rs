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
                connection.set_nonblocking(false).unwrap();
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
        use std::io::Read;
        let mut connection = request_fixture_accept(&listener);
        let mut chunk = [0; 1024];
        // Send no response, and require the timed-out client to close its
        // socket. Reading bytes needs no complete HTTP header or body.
        loop {
            let count = connection
                .read(&mut chunk)
                .expect("client deadline must close the stalled HTTP connection");
            if count == 0 {
                break;
            }
        }
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

#[test]
fn native_synthetic_prompts_interleave_tasks_and_preserve_output_on_invalid_counts() {
    let directory = tempfile::tempdir().unwrap();
    let run = |families: &str| {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(directory.path())
            .env_clear()
            .args([
                "automation",
                "waiting-prefix",
                "synthetic-prompts",
                "--families",
                families,
                "--requests-per-family",
                "2",
                "--prefix-blocks",
                "2",
                "--output",
                "prompts.json",
            ])
            .output()
            .unwrap()
    };
    let successful = run("2");
    assert!(
        successful.status.success(),
        "{}",
        String::from_utf8_lossy(&successful.stderr)
    );
    let path = directory.path().join("prompts.json");
    let bytes = fs::read(&path).unwrap();
    let document: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(document["metadata"]["generator"], "stable-prefix-v1");
    let prompts = document["prompts"].as_array().unwrap();
    assert_eq!(prompts.len(), 4);
    assert_eq!(
        prompts
            .iter()
            .map(|row| row["family"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["family-0", "family-1", "family-0", "family-1"]
    );
    assert!(
        prompts[2]["prompt"]
            .as_str()
            .unwrap()
            .ends_with("Unique task 1: inspect module_1.rs and return its invariant only.")
    );
    assert_eq!(run("0").status.code(), Some(1));
    assert_eq!(fs::read(path).unwrap(), bytes);
}

#[test]
fn native_stage_config_binds_actual_file_hash_and_preserves_output_on_model_change() {
    let directory = tempfile::tempdir().unwrap();
    let model = directory.path().join("fixture-model.bin");
    fs::write(&model, b"model fixture bytes").unwrap();
    let input = json!({"model_id":"fixture", "model_path":model,
        "source_model_sha256":hex::encode(<sha2::Sha256 as sha2::Digest>::digest(b"model fixture bytes")),
        "layer_end":28,"ctx_size":65536,"lane_count":6,"n_gpu_layers":0,
        "payload":"resident-kv","cache_entries":1});
    fs::write(
        directory.path().join("input.json"),
        serde_json::to_vec(&input).unwrap(),
    )
    .unwrap();
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(directory.path())
            .env_clear()
            .args([
                "automation",
                "waiting-prefix",
                "stage-config",
                "--input",
                "input.json",
                "--output",
                "stage.json",
            ])
            .output()
            .unwrap()
    };
    let success = run();
    assert!(
        success.status.success(),
        "{}",
        String::from_utf8_lossy(&success.stderr)
    );
    let path = directory.path().join("stage.json");
    let bytes = fs::read(&path).unwrap();
    let stage: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(stage["source_model_sha256"], input["source_model_sha256"]);
    assert_eq!(stage["kv_cache"]["shared_prefix_record_limit"], 1);
    assert_eq!(stage["kv_cache"]["payload"], "resident-kv");
    assert_eq!(stage["lane_count"], 6);
    fs::write(model, b"changed model fixture bytes").unwrap();
    let failure = run();
    assert_eq!(failure.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&failure.stderr).contains("SHA-256 mismatch"));
    assert_eq!(fs::read(path).unwrap(), bytes);
}

#[test]
fn native_cell_telemetry_excludes_seed_and_preserves_output_on_wrong_snapshot() {
    use std::io::Write;
    let directory = tempfile::tempdir().unwrap();
    let log = directory.path().join("server.log");
    let events = |offset: u64| {
        [
            "stage.openai_generation_summary",
            "stage.openai_kv_capacity_decision",
            "stage.openai_kv_record_decision",
        ]
        .into_iter()
        .enumerate()
        .map(|(index, event)| {
            serde_json::to_string(
                &json!({"event":event,"attributes":{"count":offset+index as u64}}),
            )
            .unwrap()
                + "\n"
        })
        .collect::<String>()
    };
    fs::write(&log, events(1)).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(directory.path())
            .env_clear()
            .args(["automation", "waiting-prefix", "telemetry-log"])
            .args(args)
            .output()
            .unwrap()
    };
    let snapshot = run(&["snapshot", "--log", "server.log", "--output", "cursor.json"]);
    assert!(
        snapshot.status.success(),
        "{}",
        String::from_utf8_lossy(&snapshot.stderr)
    );
    fs::OpenOptions::new()
        .append(true)
        .open(&log)
        .unwrap()
        .write_all(events(4).as_bytes())
        .unwrap();
    let collect = || {
        run(&[
            "collect",
            "--log",
            "server.log",
            "--cursor",
            "cursor.json",
            "--expected-generations",
            "1",
            "--output",
            "measured.json",
        ])
    };
    let success = collect();
    assert!(
        success.status.success(),
        "{}",
        String::from_utf8_lossy(&success.stderr)
    );
    let output = directory.path().join("measured.json");
    let bytes = fs::read(&output).unwrap();
    let measured: Value = serde_json::from_slice(&bytes).unwrap();
    for (key, count) in [("events", 4), ("capacity_events", 5), ("record_events", 6)] {
        assert_eq!(measured[key].as_array().unwrap().len(), 1);
        assert_eq!(measured[key][0]["attributes"]["count"], count);
    }
    let failure = run(&[
        "collect",
        "--log",
        "server.log",
        "--cursor",
        "cursor.json",
        "--expected-generations",
        "2",
        "--output",
        "measured.json",
    ]);
    assert_eq!(failure.status.code(), Some(1));
    assert_eq!(fs::read(&output).unwrap(), bytes);
    let mut cursor: Value =
        serde_json::from_slice(&fs::read(directory.path().join("cursor.json")).unwrap()).unwrap();
    cursor["generations"] = json!(0);
    fs::write(
        directory.path().join("cursor.json"),
        serde_json::to_vec(&cursor).unwrap(),
    )
    .unwrap();
    assert_eq!(collect().status.code(), Some(1));
    assert_eq!(fs::read(&output).unwrap(), bytes);
    fs::write(log, events(9)).unwrap();
    assert_eq!(collect().status.code(), Some(1));
    assert_eq!(fs::read(output).unwrap(), bytes);
}

fn cell_fixture_request(connection: &mut std::net::TcpStream) -> (String, Value) {
    use std::io::Read;
    let mut bytes = Vec::new();
    let mut chunk = [0; 1024];
    loop {
        let count = connection.read(&mut chunk).unwrap();
        assert!(count > 0, "incomplete cell request");
        bytes.extend_from_slice(&chunk[..count]);
        assert!(bytes.len() < 1024 * 1024);
        if let Some(end) = bytes.windows(4).position(|row| row == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]).to_lowercase();
            let length = headers
                .lines()
                .find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    (key == "content-length").then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            if bytes.len() >= end + 4 + length {
                let body = if length == 0 {
                    Value::Null
                } else {
                    serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap()
                };
                return (headers.lines().next().unwrap().to_string(), body);
            }
        }
    }
}

fn cell_fixture_response(connection: &mut std::net::TcpStream, content_type: &str, body: &str) {
    use std::io::Write;
    write!(connection,"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
}

fn cell_fixture_telemetry(log: &Path, seeded: bool) {
    use std::io::Write;
    let events = [
        (
            "stage.openai_generation_summary",
            json!({"skippy.kv.status":"hit",
            "skippy.kv.matched_prefix_tokens":if seeded {100} else {10},"skippy.kv.suffix_prefill_tokens":3}),
        ),
        (
            "stage.openai_kv_capacity_decision",
            json!({"skippy.kv.capacity_status":if seeded {"rejected"} else {"evicted"},
            "skippy.kv.capacity_evicted_tokens":if seeded {100} else {2},"skippy.kv.capacity_evicted_entries":1,
            "skippy.kv.capacity_predicted_recompute_cost":4}),
        ),
        (
            "stage.openai_kv_record_decision",
            json!({"skippy.kv.decision":"proactive_eviction",
            "skippy.kv.proactive_evicted_tokens":if seeded {999} else {1},"skippy.kv.proactive_evicted_entries":1}),
        ),
    ];
    let mut output = fs::OpenOptions::new().append(true).open(log).unwrap();
    for (event, attributes) in events {
        writeln!(output, "{}", json!({"event":event,"attributes":attributes})).unwrap();
    }
    output.flush().unwrap();
}

fn cell_fixture_server(listener: std::net::TcpListener, log: std::path::PathBuf) -> Vec<Value> {
    let mut bodies = Vec::new();
    for _ in 0..5 {
        let mut connection = request_fixture_accept(&listener);
        let (request, body) = cell_fixture_request(&mut connection);
        if request.starts_with("get /v1/models ") {
            cell_fixture_response(
                &mut connection,
                "application/json",
                r#"{"data":[{"id":"fixture"}]}"#,
            );
            continue;
        }
        assert!(request.starts_with("post /v1/chat/completions "));
        let seeded = body["max_tokens"] == 1;
        let generated = if seeded { 1 } else { 2 };
        cell_fixture_telemetry(&log, seeded);
        let data = format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            json!({"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}),
            json!({"usage":{"prompt_tokens":40,"completion_tokens":generated,
                "prompt_tokens_details":{"cached_tokens":30}}})
        );
        cell_fixture_response(&mut connection, "text/event-stream", &data);
        bodies.push(body);
    }
    bodies
}

fn cell_worker_input(address: std::net::SocketAddr, log: &Path) -> Value {
    json!({"schema_version":1,"server_log":log,"startup_timeout_secs":2,"telemetry_timeout_secs":2,
        "phase":{"schema_version":1,"round":1,"version":"new","base_url":format!("http://{address}/v1"),
            "model":"fixture","output_tokens":2,"request_timeout_secs":2.0,"stagger_ms":0.0,
            "prompts":[{"family":"family-0","prompt":"first"},{"family":"family-1","prompt":"second"}]},
        "cache_seed":{"families":2,"prefix_blocks":2,"output_tokens":1,"stagger_ms":0.0}})
}

fn cell_worker_cli(directory: &Path, input: &Value) -> std::process::Output {
    fs::write(
        directory.join("cell-input.json"),
        serde_json::to_vec(input).unwrap(),
    )
    .unwrap();
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(directory)
        .env_clear()
        .args([
            "automation",
            "waiting-prefix",
            "cell-worker",
            "--input",
            "cell-input.json",
            "--output",
            "cell.json",
        ])
        .output()
        .unwrap()
}

#[test]
fn native_cell_worker_separates_seed_telemetry_from_complete_measured_requests() {
    let directory = tempfile::tempdir().unwrap();
    let log = directory.path().join("server.log");
    fs::write(&log, b"").unwrap();
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let server_log = log.clone();
    let server = std::thread::spawn(move || cell_fixture_server(listener, server_log));
    let result = cell_worker_cli(directory.path(), &cell_worker_input(address, &log));
    let bodies = server.join().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(bodies.len(), 4);
    assert_eq!(
        bodies.iter().filter(|body| body["max_tokens"] == 1).count(),
        2
    );
    assert_eq!(
        bodies.iter().filter(|body| body["max_tokens"] == 2).count(),
        2
    );
    let output: Value =
        serde_json::from_slice(&fs::read(directory.path().join("cell.json")).unwrap()).unwrap();
    assert_eq!(output["model"], "fixture");
    assert_eq!(output["version"], "new");
    assert!(output["error"].is_null());
    assert_eq!(
        output["cache_seed"]["requests"].as_array().unwrap().len(),
        2
    );
    assert_eq!(
        output["measurement"]["requests"].as_array().unwrap().len(),
        2
    );
    assert_eq!(output["telemetry"]["events"].as_array().unwrap().len(), 2);
    assert_eq!(output["summary"]["summary"]["requests"], 2);
    assert_eq!(output["summary"]["summary"]["successful"], 2);
    assert_eq!(
        output["summary"]["summary"]["matched_prefix_tokens_total"],
        20.0
    );
    assert_eq!(output["summary"]["summary"]["capacity_rejections"], 0);
    assert_eq!(
        output["summary"]["summary"]["resident_evicted_tokens_total"],
        6.0
    );
}

#[test]
fn native_cell_worker_wrong_served_model_retains_failure_before_any_seed_phase() {
    let directory = tempfile::tempdir().unwrap();
    let log = directory.path().join("server.log");
    fs::write(&log, b"").unwrap();
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let mut connection = request_fixture_accept(&listener);
        let (request, _) = cell_fixture_request(&mut connection);
        assert!(request.starts_with("get /v1/models "));
        cell_fixture_response(
            &mut connection,
            "application/json",
            r#"{"data":[{"id":"other-model"}]}"#,
        );
    });
    let result = cell_worker_cli(directory.path(), &cell_worker_input(address, &log));
    server.join().unwrap();
    assert_eq!(result.status.code(), Some(1));
    let path = directory.path().join("cell.json");
    let prior = fs::read(&path).unwrap();
    let output: Value = serde_json::from_slice(&prior).unwrap();
    assert!(output["error"].as_str().unwrap().contains("differs"));
    assert!(output["cache_seed"].is_null());
    assert!(output["measurement"].is_null());
    assert_eq!(output["model"], "fixture");
    let invalid = cell_worker_cli(directory.path(), &json!({}));
    assert_eq!(invalid.status.code(), Some(1));
    assert_eq!(fs::read(path).unwrap(), prior);
}

#[test]
fn native_cell_worker_missing_telemetry_retains_completed_measurement_at_deadline() {
    let directory = tempfile::tempdir().unwrap();
    let log = directory.path().join("server.log");
    fs::write(&log, b"").unwrap();
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        for _ in 0..2 {
            let mut connection = request_fixture_accept(&listener);
            let (request, _) = cell_fixture_request(&mut connection);
            if request.starts_with("get /v1/models ") {
                cell_fixture_response(
                    &mut connection,
                    "application/json",
                    r#"{"data":[{"id":"fixture"}]}"#,
                );
            } else {
                assert!(request.starts_with("post /v1/chat/completions "));
                let data = format!(
                    "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                    json!({"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}),
                    json!({"usage":{"prompt_tokens":40,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":30}}})
                );
                cell_fixture_response(&mut connection, "text/event-stream", &data);
            }
        }
    });
    let mut input = cell_worker_input(address, &log);
    input["cache_seed"] = Value::Null;
    input["phase"]["prompts"]
        .as_array_mut()
        .unwrap()
        .truncate(1);
    input["telemetry_timeout_secs"] = json!(1);
    let result = cell_worker_cli(directory.path(), &input);
    server.join().unwrap();
    assert_eq!(result.status.code(), Some(1));
    let output: Value =
        serde_json::from_slice(&fs::read(directory.path().join("cell.json")).unwrap()).unwrap();
    assert!(
        output["error"]
            .as_str()
            .unwrap()
            .contains("telemetry deadline")
    );
    assert_eq!(
        output["measurement"]["requests"].as_array().unwrap().len(),
        1
    );
    assert!(output["measurement"]["requests"][0]["error"].is_null());
    assert!(output["telemetry"].is_null());
    assert!(output["summary"].is_null());
}

#[path = "waiting_prefix_acceptance/server_cell.rs"]
mod server_cell;

#[path = "waiting_prefix_acceptance/round_runner.rs"]
mod round_runner;
