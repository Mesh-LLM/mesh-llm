use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::Command;

fn read_request(connection: &mut std::net::TcpStream) -> Vec<u8> {
    connection
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .unwrap();
    let mut request = Vec::new();
    loop {
        let mut buffer = [0; 4096];
        let count = connection.read(&mut buffer).unwrap();
        assert!(count > 0);
        request.extend_from_slice(&buffer[..count]);
        if let Some(end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&request[..end]);
            let length = headers
                .lines()
                .find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap();
            if request.len() >= end + 4 + length {
                return request;
            }
        }
    }
}

#[test]
fn concurrent_cell_runs_every_recorded_turn_with_matching_summary() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let mut observed = Vec::new();
        for _ in 0..4 {
            let (mut connection, _) = listener.accept().unwrap();
            let request = read_request(&mut connection);
            let end = request
                .windows(4)
                .position(|bytes| bytes == b"\r\n\r\n")
                .unwrap();
            observed
                .push(serde_json::from_slice::<serde_json::Value>(&request[end + 4..]).unwrap());
            let body = b"data: {\"choices\":[{\"delta\":{\"content\":\"saved answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: {\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":30}}}\n\ndata: [DONE]\n\n";
            write!(connection, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
            connection.write_all(body).unwrap();
        }
        observed
    });
    let state = tempfile::tempdir().unwrap();
    let input = state.path().join("cell.json");
    let requests = state.path().join("requests.jsonl");
    let summary = state.path().join("summary.json");
    let trajectories: Vec<_> = ["z", "a"].into_iter().map(|session| serde_json::json!({
        "session_id":session,"source_dataset":"fixture","agent_framework":"fixture",
        "messages":[{"role":"user","content":"task"},{"role":"assistant","content":"recorded first"},
            {"role":"user","content":"next"},{"role":"assistant","content":"recorded final"}]
    })).collect();
    std::fs::write(
        &input,
        serde_json::to_vec(&serde_json::json!({
            "trajectories":trajectories,"model":"fixture","base_url":format!("http://{address}/v1"),
            "concurrency":2,"max_output_tokens":2048,"request_timeout_seconds":5
        }))
        .unwrap(),
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "execute-cell", "--input"])
        .arg(input)
        .arg("--requests-output")
        .arg(&requests)
        .arg("--summary-output")
        .arg(&summary)
        .output()
        .unwrap();
    let observed = server.join().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    for session in ["z", "a"] {
        let turns: Vec<_> = observed
            .iter()
            .filter(|body| body["prompt_cache_key"] == session)
            .collect();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0]["messages"].as_array().unwrap().len(), 1);
        assert_eq!(turns[1]["messages"][1]["content"], "recorded first");
    }
    let raw = std::fs::read_to_string(requests).unwrap();
    assert_eq!(raw.lines().count(), 4);
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(summary).unwrap()).unwrap();
    assert_eq!(report["completeness"]["passed"], true);
    assert_eq!(report["successful_requests"], 4);
    assert_eq!(report["completion_tokens"], 8.0);
    use sha2::{Digest, Sha256};
    let expected = hex::encode(Sha256::digest(
        serde_json::to_vec(&serde_json::Value::Array(trajectories)).unwrap(),
    ));
    assert_eq!(report["session_cohort_sha256"], expected);
    assert_eq!(
        report["completeness"]["expected_request_ids"],
        serde_json::json!(["z:0", "z:1", "a:0", "a:1"])
    );
}

#[test]
fn trajectory_execution_preserves_stream_evidence_through_http() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut connection, _) = listener.accept().unwrap();
        connection
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        loop {
            let mut buffer = [0; 4096];
            let count = connection.read(&mut buffer).unwrap();
            assert!(count > 0);
            request.extend_from_slice(&buffer[..count]);
            if let Some(end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&request[..end]);
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (key, value) = line.split_once(':')?;
                        key.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap();
                if request.len() >= end + 4 + length {
                    break;
                }
            }
        }
        let body = b"data: {\"choices\":[{\"delta\":{\"content\":\"saved answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: {\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":30}}}\n\ndata: [DONE]\n\n";
        write!(connection, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
        connection.write_all(body).unwrap();
        request
    });
    let state = tempfile::tempdir().unwrap();
    let input = state.path().join("trajectory.json");
    let output = state.path().join("requests.jsonl");
    std::fs::write(&input, br#"{"session_id":"s","source_dataset":"fixture","agent_framework":"fixture","messages":[{"role":"user","content":"task"},{"role":"assistant","content":"recorded answer"}]}"#).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "automation",
            "replay-matrix",
            "execute-trajectory",
            "--input",
        ])
        .arg(input)
        .arg("--output")
        .arg(&output)
        .args([
            "--base-url",
            &format!("http://{address}/v1"),
            "--model",
            "fixture",
            "--timeout",
            "5",
        ])
        .output()
        .unwrap();
    let request = server.join().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(request.starts_with(b"POST /v1/chat/completions "));
    let record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(record["request_id"], "s:0");
    assert_eq!(record["prompt_tokens"], 40);
    assert_eq!(record["cached_tokens"], 30);
    assert_eq!(record["completion_tokens"], 2);
    assert!(record.get("error").is_none());
}

#[test]
fn cell_execution_retains_failed_turns_and_negative_completeness() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        for _ in 0..2 {
            let (mut connection, _) = listener.accept().unwrap();
            connection
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut buffer = [0; 8192];
            let count = connection.read(&mut buffer).unwrap();
            assert!(count > 0);
            connection.write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
        }
    });
    let state = tempfile::tempdir().unwrap();
    let input = state.path().join("cell.json");
    let requests = state.path().join("requests.jsonl");
    let summary = state.path().join("summary.json");
    let workload = serde_json::json!({
        "trajectories":[{"session_id":"s","source_dataset":"fixture","agent_framework":"fixture",
            "messages":[{"role":"user","content":"task"},{"role":"assistant","content":"first"},
                {"role":"user","content":"next"},{"role":"assistant","content":"second"}]}],
        "model":"fixture","base_url":format!("http://{address}/v1"),"concurrency":2,
        "max_output_tokens":2048,"request_timeout_seconds":5
    });
    std::fs::write(&input, serde_json::to_vec(&workload).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "execute-cell", "--input"])
        .arg(input)
        .arg("--requests-output")
        .arg(&requests)
        .arg("--summary-output")
        .arg(&summary)
        .output()
        .unwrap();
    server.join().unwrap();
    assert!(!result.status.success());
    let raw = std::fs::read_to_string(requests).unwrap();
    let rows: Vec<serde_json::Value> = raw
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["request_id"], "s:0");
    assert_eq!(rows[1]["request_id"], "s:1");
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(summary).unwrap()).unwrap();
    assert_eq!(report["completeness"]["passed"], false);
    assert_eq!(report["failed_requests"], 2);
}
