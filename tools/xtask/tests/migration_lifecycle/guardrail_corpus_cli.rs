//! Actual native command, synthetic corpus and owned local HTTP refusal evidence.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use serde_json::{Value as Json, json};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::{Ipv4Addr, TcpListener},
    path::Path,
    time::{Duration, Instant},
};
fn perform(root: &Path, args: Vec<String>) -> process::ProcessReport {
    process::supervise(
        &ProcessSpec {
            executable: Path::new(env!("CARGO_BIN_EXE_xtask")).into(),
            arguments: args.into_iter().map(|a| Value::Public(a.into())).collect(),
            cwd: root.into(),
            environment: BTreeMap::new(),
        },
        &Limits {
            execution: Duration::from_secs(6),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap()
}
fn clean(result: &process::ProcessReport) {
    assert_eq!(result.outcome, process::Outcome::Exited);
    assert!(result.failure.is_none());
    assert!(
        result.cleanup.complete
            && !result.cleanup.forced
            && result.cleanup.failure.is_none()
            && !result.cleanup.graceful_signal_failed
    );
    assert!(
        result.stdout.line_capture_complete
            && result.stderr.line_capture_complete
            && !result.stdout.truncated
            && !result.stderr.truncated
    );
}
fn run(root: &Path, args: Vec<String>) -> process::ProcessReport {
    let result = perform(root, args);
    clean(&result);
    result
}
fn args(base: &str, out: &Path) -> Vec<String> {
    [
        "automation",
        "guardrail-corpus",
        "--base-url",
        base,
        "--model",
        "fixture",
        "--guardrail-mode",
        "metrics",
        "--trials",
        "2",
        "--out",
        out.to_str().unwrap(),
        "--timeout-secs",
        "3",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}
#[test]
fn actual_fake_cli_retains_corpus_summary_and_synthetic_labels() {
    let root = tempfile::tempdir().unwrap();
    let out = root.path().join("fresh/nested/evidence/receipt.json");
    let result = run(root.path(), args("fake://owned", &out));
    assert!(result.status.unwrap().success());
    let value: Json = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(value["backend_mode"], "fake");
    assert_eq!(value["expected_server_mode"], "metrics");
    assert_eq!(value["total_requests"], 10);
    assert_eq!(value["success_count"], 8);
    assert_eq!(value["failure_count"], 2);
    assert_eq!(value["retry_count"], 2);
    assert_eq!(value["physical_retries"], 0);
    assert_eq!(value["corpus"].as_array().unwrap().len(), 5);
    assert!(
        value["results"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["latency_origin"] == "deterministic_synthetic_not_measured")
    );
}
#[test]
fn actual_cli_refuses_existing_symlink_and_invalid_mode_without_overwrite() {
    let root = tempfile::tempdir().unwrap();
    let target = root.path().join("target");
    std::fs::write(&target, b"unchanged").unwrap();
    let out = root.path().join("receipt");
    std::os::unix::fs::symlink(&target, &out).unwrap();
    assert!(
        !run(root.path(), args("fake://owned", &out))
            .status
            .unwrap()
            .success()
    );
    assert_eq!(std::fs::read(&target).unwrap(), b"unchanged");
    assert!(std::fs::symlink_metadata(&out).unwrap().is_symlink());
    let fresh = root.path().join("fresh.json");
    let mut arguments = args("fake://owned", &fresh);
    let index = arguments.iter().position(|v| v == "metrics").unwrap();
    arguments[index] = "unknown".into();
    assert!(!run(root.path(), arguments).status.unwrap().success());
    assert!(!fresh.exists());
    let file_parent = root.path().join("file-parent");
    std::fs::write(&file_parent, b"unchanged").unwrap();
    assert!(
        !run(
            root.path(),
            args("fake://owned", &file_parent.join("nested/report.json"))
        )
        .status
        .unwrap()
        .success()
    );
    assert_eq!(std::fs::read(&file_parent).unwrap(), b"unchanged");
    let directory = root.path().join("directory");
    std::fs::create_dir(&directory).unwrap();
    let link = root.path().join("linked-parent");
    std::os::unix::fs::symlink(&directory, &link).unwrap();
    assert!(
        !run(
            root.path(),
            args("fake://owned", &link.join("nested/report.json"))
        )
        .status
        .unwrap()
        .success()
    );
    assert!(!directory.join("nested").exists());
    root.close().unwrap();
}
fn peer(listener: TcpListener, complete: bool) -> Vec<String> {
    listener.set_nonblocking(true).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut requests = Vec::new();
    while requests.len() < (if complete { 6 } else { 2 }) && Instant::now() < deadline {
        let (mut stream, _) = match listener.accept() {
            Ok(v) => v,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
            Err(e) => panic!("{e}"),
        };
        stream.set_nonblocking(false).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let mut bytes = Vec::new();
        let mut buffer = [0; 4096];
        loop {
            match stream.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    bytes.extend_from_slice(&buffer[..n]);
                    if bytes.len() > 65536 {
                        break;
                    }
                    if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&bytes[..end]);
                        let length = headers
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .and_then(|v| v.trim().parse::<usize>().ok())
                            })
                            .unwrap_or(0);
                        if bytes.len() >= end + 4 + length {
                            break;
                        }
                    }
                }
            }
        }
        let text = String::from_utf8(bytes).unwrap();
        let body = if requests.is_empty() {
            json!({"data":[{"id":"fixture"}]}).to_string()
        } else if complete && requests.len() > 1 {
            json!({"choices":[{"message":{"content":"observed fixture content"}}]}).to_string()
        } else {
            "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":\"stop\"}]}\n\n".into()
        };
        let body = if complete && requests.len() == 1 {
            format!("{body}data: [DONE]\n\n")
        } else {
            body
        };
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        requests.push(text);
    }
    requests
}
#[test]
fn actual_live_cli_refuses_truncated_sse_and_retains_partial_without_fake_switch() {
    let root = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let base = format!(
        "http://127.0.0.1:{}/v1",
        listener.local_addr().unwrap().port()
    );
    let handle = std::thread::spawn(move || peer(listener, false));
    let out = root.path().join("receipt.json");
    let result = perform(root.path(), args(&base, &out));
    let requests = handle.join().unwrap();
    clean(&result);
    assert!(!result.status.unwrap().success());
    assert_eq!(requests.len(), 2);
    assert!(requests[0].starts_with("GET /v1/models "));
    assert!(requests[1].starts_with("POST /v1/chat/completions "));
    assert!(requests[1].contains("\"mesh_guardrails\":true"));
    let report: Json = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(report["backend_mode"], "live");
    assert_eq!(report["status"], "incomplete");
    assert_eq!(report["total_requests"], 1);
    assert_eq!(report["results"][0]["status"], 200);
    assert_eq!(report["results"][0]["response_kind"], "transport_failure");
    assert!(report["fallback_reason"].is_null());
}

#[test]
fn actual_live_cli_consumes_all_five_request_shapes_without_physical_retries() {
    let root = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let base = format!(
        "http://127.0.0.1:{}/v1",
        listener.local_addr().unwrap().port()
    );
    let handle = std::thread::spawn(move || peer(listener, true));
    let out = root.path().join("receipt.json");
    let mut arguments = args(&base, &out);
    let at = arguments.iter().position(|a| a == "--trials").unwrap() + 1;
    arguments[at] = "1".into();
    let result = perform(root.path(), arguments);
    let requests = handle.join().unwrap();
    assert!(result.status.unwrap().success());
    assert_eq!(requests.len(), 6);
    let bodies: Vec<Json> = requests
        .iter()
        .skip(1)
        .map(|r| serde_json::from_str(r.split_once("\r\n\r\n").unwrap().1).unwrap())
        .collect();
    assert_eq!(bodies[0]["stream"], true);
    assert_eq!(bodies[1]["tools"][0]["function"]["name"], "calculator");
    assert_eq!(bodies[2]["response_format"]["type"], "json_object");
    assert_eq!(bodies[3]["response_format"]["json_schema"]["strict"], true);
    assert_eq!(bodies[4]["tools"][0]["function"]["name"], "calculator");
    assert_eq!(bodies[4]["response_format"]["json_schema"]["strict"], true);
    let report: Json = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(report["status"], "completed");
    assert_eq!(report["backend_mode"], "live");
    assert_eq!(report["success_count"], 4);
    assert_eq!(report["failure_count"], 1);
    assert_eq!(report["physical_retries"], 0);
    assert_eq!(report["results"].as_array().unwrap().len(), 5);
}
