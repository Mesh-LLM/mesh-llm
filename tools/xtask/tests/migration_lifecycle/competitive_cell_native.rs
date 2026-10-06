//! Actual native request-worker CLI against bounded owned loopback fixtures.
use crate::process;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::TcpListener,
    num::NonZeroUsize,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
struct Server {
    base: String,
    requests: Arc<Mutex<Vec<Value>>>,
    readiness_seen: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    task: Option<std::thread::JoinHandle<()>>,
}
impl Server {
    fn new(reply: &'static str) -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = requests.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let quitting = stop.clone();
        let readiness_seen = Arc::new(AtomicBool::new(false));
        let observed = readiness_seen.clone();
        let task = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(8);
            while !quitting.load(Ordering::Acquire) && Instant::now() < deadline {
                let Ok((mut socket, _)) = listener.accept() else {
                    std::thread::sleep(Duration::from_millis(5));
                    continue;
                };
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(1)))
                    .unwrap();
                socket
                    .set_write_timeout(Some(Duration::from_secs(1)))
                    .unwrap();
                let mut bytes = Vec::new();
                let mut buffer = [0; 4096];
                let end = loop {
                    let Ok(count) = socket.read(&mut buffer) else {
                        break None;
                    };
                    if count == 0 {
                        break None;
                    }
                    bytes.extend_from_slice(&buffer[..count]);
                    if bytes.len() > 65536 {
                        break None;
                    }
                    if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                        break Some(end + 4);
                    }
                };
                let Some(end) = end else {
                    continue;
                };
                let headers = String::from_utf8_lossy(&bytes[..end]);
                let models = headers.starts_with("GET /v1/models ");
                if models {
                    observed.store(true, Ordering::Release);
                }
                let length = headers
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .and_then(|value| value.trim().parse::<usize>().ok())
                    })
                    .unwrap_or(0);
                if length > 65536 {
                    continue;
                }
                while bytes.len() < end + length {
                    let Ok(count) = socket.read(&mut buffer) else {
                        break;
                    };
                    if count == 0 {
                        break;
                    }
                    bytes.extend_from_slice(&buffer[..count]);
                }
                if bytes.len() < end + length {
                    continue;
                }
                if !models {
                    captured
                        .lock()
                        .unwrap()
                        .push(serde_json::from_slice(&bytes[end..end + length]).unwrap());
                }
                if (!models && reply == "__HOLD__") || (models && reply == "__HOLD_READY__") {
                    while !quitting.load(Ordering::Acquire) && Instant::now() < deadline {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    continue;
                }
                let payload = if models {
                    "{\"data\":[{\"id\":\"fixture-model\"}]}"
                } else {
                    reply
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                    payload.len()
                );
                let _ = socket.write_all(response.as_bytes());
            }
        });
        Self {
            base,
            requests,
            readiness_seen,
            stop,
            task: Some(task),
        }
    }
    fn close(mut self) -> Vec<Value> {
        self.stop.store(true, Ordering::Release);
        self.task.take().unwrap().join().unwrap();
        self.requests.lock().unwrap().clone()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(task) = self.task.take() {
            task.join().unwrap();
        }
    }
}
fn input(root: &Path, base: &str) -> std::path::PathBuf {
    let mut config: Value = serde_json::from_slice(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../evals/skippy-competitive-benchmark.json"
    )))
    .unwrap();
    config["thoughtworks"]["minimum_prompts"] = json!(1);
    let prompts: Vec<_> = (0..256)
        .map(|index| json!({"family":index%8,"prompt":"πabcdefghijkl"}))
        .collect();
    let manifest = json!({"metadata":{"dataset_revision":config["thoughtworks"]["dataset"]["revision"],"rows":config["thoughtworks"]["selection"]["rows"]},"prompts":prompts});
    let bytes = serde_json::to_vec_pretty(&manifest).unwrap();
    let manifest_path = root.join("manifest.json");
    std::fs::write(&manifest_path, &bytes).unwrap();
    config["thoughtworks"]["selection"]["manifest_sha256"] =
        json!(hex::encode(Sha256::digest(&bytes)));
    let bytes = serde_json::to_vec_pretty(&config).unwrap();
    let config_path = root.join("config.json");
    std::fs::write(&config_path, &bytes).unwrap();
    let doc = json!({"config":config_path,"config_sha256":hex::encode(Sha256::digest(&bytes)),"manifest":manifest_path,"cell":{"platform":"metal","model":"llama32-dense","workload":"thoughtworks","arm":"mesh","output_tokens":8,"context_size":131072,"active_lanes":16,"concurrency":1,"prompt_count":1},"base_url":base,"served_model":"fixture-model","launch_provenance":{"binary_sha256":"a".repeat(64),"model_sha256":config["models"][0]["sha256"]},"output":root.join("output"),"timeout_seconds":4,"request_timeout_seconds":1});
    let path = root.join("input.json");
    std::fs::write(&path, serde_json::to_vec(&doc).unwrap()).unwrap();
    path
}
fn run(root: &Path, path: &Path) -> bool {
    let raw = process::supervise_raw(
        &process::ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            cwd: root.into(),
            arguments: [
                "automation",
                "replay-matrix",
                "competitive-cell",
                "--input",
                path.to_str().unwrap(),
            ]
            .map(|value| process::Value::Public(value.into()))
            .into(),
            environment: BTreeMap::new(),
        },
        &process::Limits {
            execution: Duration::from_secs(6),
            graceful_shutdown: Duration::from_millis(200),
            forced_shutdown: Duration::from_millis(200),
            retained_bytes_per_stream: 65536,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        &process::Cancellation::default(),
        process::RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    let report = raw.process;
    assert_eq!(report.outcome, process::Outcome::Exited);
    assert!(
        report.failure.is_none()
            && report.cleanup.complete
            && !report.cleanup.forced
            && report.cleanup.failure.is_none()
            && !report.cleanup.graceful_signal_failed,
        "{report:?}"
    );
    assert!(!report.stdout.truncated && !report.stderr.truncated);
    assert_eq!(report.stdout.suppressed_lines, 0);
    assert_eq!(report.stderr.suppressed_lines, 0);
    let stdout = raw.stdout.unwrap();
    let stderr = raw.stderr.unwrap();
    assert_eq!(stdout.as_bytes().len() as u64, report.stdout.bytes_seen);
    assert_eq!(stderr.as_bytes().len() as u64, report.stderr.bytes_seen);
    report.status.unwrap().success()
}
const GOOD: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: {\"usage\":{\"prompt_tokens\":32,\"prompt_tokens_details\":{\"cached_tokens\":24},\"completion_tokens\":8}}\n\ndata: [DONE]\n\n";
#[test]
fn competitive_owned_cli_preserves_prefix_order_prompt_hash_and_usage_without_completion_marker() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::new(GOOD);
    let path = input(root.path(), &server.base);
    assert!(run(root.path(), &path));
    let requests = server.close();
    assert_eq!(requests.len(), 3);
    let contents: Vec<_> = requests
        .iter()
        .map(|body| body["messages"][0]["content"].as_str().unwrap())
        .collect();
    assert_eq!(contents, ["πabcdef", "πabcdefghi", "πabcdefghijkl"]);
    let rows: Vec<Value> = std::fs::read_to_string(root.path().join("output/requests.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 3);
    let row = &rows[2];
    assert_eq!(row["phase"], "measured");
    assert_eq!(
        row["prompt_sha256"],
        hex::encode(Sha256::digest("πabcdefghijkl".as_bytes()))
    );
    assert_eq!(row["prompt_tokens"], 32);
    assert_eq!(row["cached_prompt_tokens"], 24);
    assert_eq!(row["new_prompt_tokens"], 8);
    assert_eq!(row["completion_tokens"], 8);
    assert!(row["error"].is_null());
    let summary: Value = serde_json::from_slice(
        &std::fs::read(root.path().join("output/worker-summary.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(summary["passed"], true);
    assert_eq!(summary["terminal_complete"], true);
    assert!(summary["terminal_error"].is_null());
    assert_eq!(summary["requests"], 1);
    assert!(!root.path().join("output/complete.json").exists());
    root.close().unwrap();
}
#[test]
fn competitive_owned_cli_refuses_hidden_errors_short_output_and_missing_terminal() {
    for reply in [
        "data: {\"error\":{\"message\":\"hidden failure\"}}\n\ndata: [DONE]\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"short\"}}]}\n\ndata: {\"usage\":{\"prompt_tokens\":32,\"prompt_tokens_details\":{\"cached_tokens\":0},\"completion_tokens\":7}}\n\ndata: [DONE]\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
    ] {
        let root = tempfile::tempdir().unwrap();
        let server = Server::new(reply);
        let path = input(root.path(), &server.base);
        assert!(!run(root.path(), &path));
        server.close();
        let summary: Value = serde_json::from_slice(
            &std::fs::read(root.path().join("output/worker-summary.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(summary["passed"], false);
        assert!(!root.path().join("output/complete.json").exists());
        root.close().unwrap();
    }
}
#[test]
fn competitive_owned_cli_refuses_cell_and_prompt_identity_drift_before_http_or_outputs() {
    for drift in ["cell", "prompt", "provenance"] {
        let root = tempfile::tempdir().unwrap();
        let server = Server::new(GOOD);
        let path = input(root.path(), &server.base);
        if drift == "cell" {
            let mut doc: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            doc["cell"]["concurrency"] = json!(3);
            std::fs::write(&path, serde_json::to_vec(&doc).unwrap()).unwrap();
        } else if drift == "prompt" {
            std::fs::write(root.path().join("manifest.json"), "changed").unwrap();
        } else {
            let mut manifest: Value =
                serde_json::from_slice(&std::fs::read(root.path().join("manifest.json")).unwrap())
                    .unwrap();
            manifest["metadata"]["rows"][0]["session_id"] = json!("wrong");
            let bytes = serde_json::to_vec(&manifest).unwrap();
            std::fs::write(root.path().join("manifest.json"), &bytes).unwrap();
            let mut config: Value =
                serde_json::from_slice(&std::fs::read(root.path().join("config.json")).unwrap())
                    .unwrap();
            config["thoughtworks"]["selection"]["manifest_sha256"] =
                json!(hex::encode(Sha256::digest(&bytes)));
            let bytes = serde_json::to_vec(&config).unwrap();
            std::fs::write(root.path().join("config.json"), &bytes).unwrap();
            let mut doc: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            doc["config_sha256"] = json!(hex::encode(Sha256::digest(&bytes)));
            std::fs::write(&path, serde_json::to_vec(&doc).unwrap()).unwrap();
        }

        assert!(!run(root.path(), &path));
        let requests = server.close();
        assert!(requests.is_empty());
        assert!(!root.path().join("output").exists());
        root.close().unwrap();
    }
}
#[test]
fn competitive_owned_cli_deadline_keeps_partial_rows_and_never_marks_complete() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::new("__HOLD__");
    let path = input(root.path(), &server.base);
    let mut document: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    document["timeout_seconds"] = json!(1);
    std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
    assert!(!run(root.path(), &path));
    server.close();
    let summary: Value = serde_json::from_slice(
        &std::fs::read(root.path().join("output/worker-summary.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(summary["completed"], false);
    assert!(summary["error"].as_str().unwrap().contains("deadline"));
    let rows = std::fs::read_to_string(root.path().join("output/requests.jsonl")).unwrap();
    assert!(rows.contains("competitive request deadline exceeded"));
    assert!(!root.path().join("output/complete.json").exists());
    root.close().unwrap();
}

#[test]
fn competitive_workers_interrupt_actual_held_readiness_before_forced_cleanup() {
    for command in ["competitive-cell", "competitive-synthetic-cell"] {
        let root = tempfile::tempdir().unwrap();
        let server = Server::new("__HOLD_READY__");
        let path = input(root.path(), &server.base);
        if command == "competitive-synthetic-cell" {
            let mut document: Value =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            document.as_object_mut().unwrap().remove("manifest");
            document["request_timeout_seconds"] = 4.into();
            document["cell"] = json!({"platform":"metal","model":"llama32-dense","workload":"synthetic","arm":"mesh","prompt_tokens":512,"output_tokens":8,"concurrency":1});
            let binary = Path::new(env!("CARGO_BIN_EXE_xtask"));
            document["benchy"] = json!({"path":binary,"sha256":hex::encode(Sha256::digest(std::fs::read(binary).unwrap()))});
            document["tokenizer"] = json!(root.path());
            std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        }
        let cancellation = process::Cancellation::default();
        let ready = server.readiness_seen.clone();
        let raw = std::thread::scope(|scope| {
            scope.spawn(|| {
                let deadline = Instant::now() + Duration::from_secs(3);
                while !ready.load(Ordering::Acquire) && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(5));
                }
                cancellation.cancel();
            });
            process::supervise_raw(
                &process::ProcessSpec {
                    executable: env!("CARGO_BIN_EXE_xtask").into(),
                    cwd: root.path().into(),
                    arguments: [
                        "automation",
                        "replay-matrix",
                        command,
                        "--input",
                        path.to_str().unwrap(),
                    ]
                    .map(|value| process::Value::Public(value.into()))
                    .into(),
                    environment: BTreeMap::new(),
                },
                &process::Limits {
                    execution: Duration::from_secs(6),
                    graceful_shutdown: Duration::from_secs(1),
                    forced_shutdown: Duration::from_secs(1),
                    retained_bytes_per_stream: 65536,
                    readiness: process::Readiness::None,
                    completion: process::Completion::Exit,
                },
                &cancellation,
                process::RawCaptureOptions {
                    stdout: NonZeroUsize::new(65536),
                    stderr: NonZeroUsize::new(65536),
                },
            )
            .unwrap()
        });
        server.close();
        let report = raw.process;
        assert!(
            ready.load(Ordering::Acquire),
            "cancellation must follow actual readiness GET receipt"
        );
        assert_eq!(report.outcome, process::Outcome::Cancelled);
        assert!(report.status.is_some_and(|status| !status.success()));
        assert!(
            report.failure.is_none()
                && report.cleanup.complete
                && !report.cleanup.forced
                && !report.cleanup.graceful_signal_failed
                && report.cleanup.failure.is_none(),
            "{report:?}"
        );
        assert!(!report.stdout.truncated && !report.stderr.truncated);
        assert_eq!(report.stdout.suppressed_lines, 0);
        assert_eq!(report.stderr.suppressed_lines, 0);
        assert_eq!(
            raw.stdout.unwrap().as_bytes().len() as u64,
            report.stdout.bytes_seen
        );
        assert_eq!(
            raw.stderr.unwrap().as_bytes().len() as u64,
            report.stderr.bytes_seen
        );
        let summary: Value = serde_json::from_slice(
            &std::fs::read(root.path().join("output/worker-summary.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(summary["completed"], false);
        assert!(!root.path().join("output/complete.json").exists());
        root.close().unwrap();
    }
}
