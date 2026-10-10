#![allow(dead_code, unused_imports)]
#[path = "../src/process/mod.rs"]
mod process;

#[path = "migration_stability/tls_cases.rs"]
mod tls_cases;

#[path = "migration_stability/nightly_cases.rs"]
mod nightly_cases;

#[path = "migration_stability/kv_cases.rs"]
mod kv_cases;

mod fixture {
    use super::process::{
        self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
    };
    use serde_json::Value as Json;
    use std::{
        collections::BTreeMap,
        fs,
        io::{Read, Write},
        net::{TcpListener, TcpStream},
        path::Path,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
        thread::{self, JoinHandle},
        time::{Duration, Instant},
    };

    pub struct Server {
        pub base: String,
        pub requests: Arc<Mutex<Vec<(String, Json)>>>,
        stop: Arc<AtomicBool>,
        thread: Option<JoinHandle<()>>,
    }
    impl Server {
        pub fn new(replies: Vec<(u16, String, bool)>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let base = format!("http://{}/tenant/v1", listener.local_addr().unwrap());
            let requests = Arc::new(Mutex::new(Vec::new()));
            let output = requests.clone();
            let stop = Arc::new(AtomicBool::new(false));
            let stopping = stop.clone();
            let handle = thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_secs(8);
                for (status, body, stream) in replies {
                    let mut socket = loop {
                        if stopping.load(Ordering::SeqCst) || Instant::now() >= deadline {
                            return;
                        }
                        match listener.accept() {
                            Ok((socket, _)) => break socket,
                            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                                thread::sleep(Duration::from_millis(5))
                            }
                            Err(error) => panic!("fixture accept: {error}"),
                        }
                    };
                    socket.set_nonblocking(false).unwrap();
                    socket
                        .set_read_timeout(Some(Duration::from_secs(2)))
                        .unwrap();
                    socket
                        .set_write_timeout(Some(Duration::from_secs(2)))
                        .unwrap();
                    let (head, payload) = read_request(&mut socket);
                    output.lock().unwrap().push((head, payload));
                    let content_type = if stream {
                        "text/event-stream"
                    } else {
                        "application/json"
                    };
                    let header = format!(
                        "HTTP/1.1 {status} fixture\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    if socket.write_all(header.as_bytes()).is_err() {
                        return;
                    }
                    for fragment in body.as_bytes().chunks(7) {
                        if socket.write_all(fragment).is_err() {
                            return;
                        }
                    }
                }
            });
            Self {
                base,
                requests,
                stop,
                thread: Some(handle),
            }
        }
    }
    impl Drop for Server {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            let outcome = self.thread.take().unwrap().join();
            if !thread::panicking() {
                assert!(outcome.is_ok(), "HTTP fixture failed");
            }
        }
    }
    fn read_request(socket: &mut TcpStream) -> (String, Json) {
        let mut bytes = Vec::new();
        let mut chunk = [0; 1024];
        let boundary = loop {
            let count = socket.read(&mut chunk).unwrap();
            assert!(count > 0 && bytes.len() < 2 * 1024 * 1024);
            bytes.extend_from_slice(&chunk[..count]);
            if let Some(index) = bytes.windows(4).position(|value| value == b"\r\n\r\n") {
                break index + 4;
            }
        };
        let head = String::from_utf8(bytes[..boundary].to_vec()).unwrap();
        let length = head
            .lines()
            .find_map(|line| {
                let (key, value) = line.split_once(':')?;
                key.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().unwrap())
            })
            .unwrap_or(0);
        assert!(length <= 2 * 1024 * 1024);
        while bytes.len() < boundary + length {
            let count = socket.read(&mut chunk).unwrap();
            assert!(count > 0);
            bytes.extend_from_slice(&chunk[..count]);
        }
        let json = if length == 0 {
            Json::Null
        } else {
            serde_json::from_slice(&bytes[boundary..boundary + length]).unwrap()
        };
        (head, json)
    }
    pub fn invoke(cwd: &Path, args: &[String]) -> process::RawProcessReport {
        invoke_with(cwd, args, BTreeMap::new())
    }
    pub fn invoke_with(
        cwd: &Path,
        args: &[String],
        environment: BTreeMap<std::ffi::OsString, Value>,
    ) -> process::RawProcessReport {
        invoke_with_cancellation(cwd, args, environment, &Cancellation::default())
    }
    pub fn invoke_with_cancellation(
        cwd: &Path,
        args: &[String],
        environment: BTreeMap<std::ffi::OsString, Value>,
        cancellation: &Cancellation,
    ) -> process::RawProcessReport {
        let result = process::supervise_raw(
            &ProcessSpec {
                executable: Path::new(env!("CARGO_BIN_EXE_xtask"))
                    .canonicalize()
                    .unwrap(),
                cwd: cwd.canonicalize().unwrap(),
                arguments: args.iter().map(|arg| Value::Public(arg.into())).collect(),
                environment,
            },
            &Limits {
                execution: Duration::from_secs(10),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            cancellation,
            RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(65536),
                stderr: std::num::NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        assert!(
            result.process.cleanup.complete && result.process.failure.is_none(),
            "{:?}",
            result.process
        );
        result
    }
    pub fn rows(path: &Path) -> Vec<Json> {
        fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }
}

use serde_json::{Value, json};
use std::{fs, path::Path};
fn arguments(mode: &str, base: &str, output: &Path) -> Vec<String> {
    [
        "automation",
        "stability",
        mode,
        "--base-url",
        base,
        "--models",
        "fixture",
        "--attempts",
        "1",
        if mode == "nightly" {
            "--output-dir"
        } else {
            "--output"
        },
        output.to_str().unwrap(),
    ]
    .iter()
    .map(|value| (*value).into())
    .collect()
}
fn answer(content: &str) -> Value {
    json!({"model":"served-fixture","choices":[{"message":{"role":"assistant","content":content},"finish_reason":"stop"}],
        "usage":{"completion_tokens":4}})
}
fn call() -> Value {
    json!({"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","reasoning_content":"discard me",
        "tool_calls":[{"id":"fixture-call","type":"function","function":{"name":"lookup_fixture_fact","arguments":"{\"key\":\"codeword\"}"}}]}}]})
}
fn sse(values: &[Value]) -> String {
    let mut output = values
        .iter()
        .map(|value| format!("data: {value}\n\n"))
        .collect::<String>();
    output.push_str("data: [DONE]\n\n");
    output
}
fn stream_call() -> Vec<Value> {
    vec![
        json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"stream-call","function":{"name":"lookup_fixture_","arguments":"{\"key\":\"code"}}]}}]}),
        json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"fact","arguments":"word\"}"}}]}}]}),
        json!({"choices":[{"finish_reason":"tool_calls","delta":{}}]}),
    ]
}

#[test]
fn stability_cli_nightly_preserves_surface_tool_history_and_evidence_from_foreign_cwd() {
    let replies = vec![
        (200, json!({"data":[{"id":"fixture"}]}).to_string(), false),
        (200, answer("STABILITY_OK").to_string(), false),
        (
            200,
            sse(&[json!({"model":"served-fixture","choices":[{"delta":{"content":"STREAM_OK"}}]})]),
            true,
        ),
        (200, call().to_string(), false),
        (200, answer("signal-7429").to_string(), false),
        (200, sse(&stream_call()), true),
        (
            200,
            sse(&[json!({"choices":[{"delta":{"content":"signal-7429"}}]})]),
            true,
        ),
    ];
    let server = fixture::Server::new(replies);
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("evidence with spaces");
    let result = fixture::invoke(temp.path(), &arguments("nightly", &server.base, &output));
    assert_eq!(
        result.process.status.unwrap().code(),
        Some(0),
        "{:?}",
        result.process
    );
    let summary: Value =
        serde_json::from_slice(&fs::read(output.join("summary.json")).unwrap()).unwrap();
    assert_eq!(summary["ok"], true);
    assert_eq!(summary["passed"], 4);
    assert_eq!(summary["prereq"], 1);
    assert_eq!(summary["release_attestation"]["status"], "not_configured");
    let manifest: Value =
        serde_json::from_slice(&fs::read(output.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["name"], "nightly-stability");
    assert!(manifest["created_at"].as_str().unwrap().ends_with("+00:00"));
    assert_eq!(fixture::rows(&output.join("results.jsonl")).len(), 3);
    assert_eq!(
        fixture::rows(&output.join("agent-tool-call-reliability/results.jsonl")).len(),
        4
    );
    assert_eq!(
        fixture::rows(&output.join("commands.jsonl"))[0]["status"],
        "PASS"
    );
    assert!(output.join("summary.md").is_file());
    let requests = server.requests.lock().unwrap();
    assert_eq!(requests.len(), 7);
    assert!(requests[0].0.starts_with("GET /tenant/v1/models "));
    let continuation = &requests[4].1;
    assert_eq!(
        continuation["messages"][2]["tool_calls"][0]["id"],
        "fixture-call"
    );
    assert!(
        continuation["messages"][2]
            .get("reasoning_content")
            .is_none()
    );
    assert_eq!(continuation["messages"][3]["tool_call_id"], "fixture-call");
    assert!(
        continuation["messages"][3]["content"]
            .as_str()
            .unwrap()
            .contains("signal-7429")
    );
    assert_eq!(requests[6].1["messages"][3]["tool_call_id"], "stream-call");
    assert_eq!(requests[3].1["parallel_tool_calls"], false);
}

#[test]
fn stability_cli_wrong_answer_retains_served_model_and_failure_verdict() {
    let server = fixture::Server::new(vec![
        (200, call().to_string(), false),
        (
            200,
            answer("<think>signal-7429</think>wrong answer").to_string(),
            false,
        ),
    ]);
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("results.jsonl");
    let mut args = arguments("tool-call", &server.base, &output);
    args.push("--skip-streaming".into());
    let result = fixture::invoke(temp.path(), &args);
    assert_eq!(result.process.status.unwrap().code(), Some(1));
    let rows = fixture::rows(&output);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["ok"], true);
    assert_eq!(rows[1]["ok"], false);
    assert_eq!(rows[1]["actual_model"], "served-fixture");
    assert_eq!(rows[1]["status_code"], 200);
    assert!(rows[1]["tok_per_sec"].as_f64().unwrap() > 0.0);
}

#[test]
fn stability_cli_plan_and_bad_inputs_create_no_evidence_or_network_dependency() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("uncreated/results.jsonl");
    let mut args = arguments("tool-call", "https://fixture.invalid/service/v1", &output);
    args.push("--print-plan".into());
    let result = fixture::invoke(temp.path(), &args);
    assert_eq!(result.process.status.unwrap().code(), Some(0));
    let plan: Value = serde_json::from_slice(result.stdout.unwrap().as_bytes()).unwrap();
    assert_eq!(plan["name"], "agent-tool-call-reliability");
    assert_eq!(plan["checks"][0]["phases"].as_array().unwrap().len(), 4);
    assert!(!output.parent().unwrap().exists());
    let mut invalid = arguments("nightly", "http://127.0.0.1:1", output.parent().unwrap());
    invalid.extend(["--attempts".into(), "0".into()]);
    let result = fixture::invoke(temp.path(), &invalid);
    assert_eq!(result.process.status.unwrap().code(), Some(2));
    assert!(!output.parent().unwrap().exists());
}

#[test]
fn stability_cli_transport_and_tool_failures_never_issue_a_continuation() {
    for (status, body) in [
        (503, json!({"error":"finite unavailable"}).to_string()),
        (200, "not JSON".into()),
        (200, json!({"choices":[]}).to_string()),
    ] {
        let server = fixture::Server::new(vec![(status, body, false)]);
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("results.jsonl");
        let mut args = arguments("tool-call", &server.base, &output);
        args.push("--skip-streaming".into());
        let result = fixture::invoke(temp.path(), &args);
        assert_eq!(result.process.status.unwrap().code(), Some(1));
        let rows = fixture::rows(&output);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["phase"], "tool_call");
        assert_eq!(rows[0]["ok"], false);
        assert_eq!(rows[0]["status_code"], status);
        assert_eq!(server.requests.lock().unwrap().len(), 1);
    }
}

#[test]
fn stability_cli_attestation_is_inspected_directly_and_compared_with_expected_status() {
    let replies = vec![
        (200, json!({"data":[{"id":"fixture"}]}).to_string(), false),
        (200, answer("STABILITY_OK").to_string(), false),
        (200, call().to_string(), false),
        (200, answer("signal-7429").to_string(), false),
    ];
    let server = fixture::Server::new(replies);
    let temp = tempfile::tempdir().unwrap();
    let binary = temp.path().join("unattested fixture");
    fs::write(&binary, b"finite unattested artifact; not executed").unwrap();
    let output = temp.path().join("nightly");
    let mut args = arguments("nightly", &server.base, &output);
    args.extend([
        "--skip-streaming".into(),
        "--mesh-binary".into(),
        binary.to_str().unwrap().into(),
        "--release-attestation-expected-status".into(),
        "missing".into(),
    ]);
    let result = fixture::invoke(temp.path(), &args);
    assert_eq!(result.process.status.unwrap().code(), Some(0));
    let attestation: Value =
        serde_json::from_slice(&fs::read(output.join("release-attestation.json")).unwrap())
            .unwrap();
    assert_eq!(attestation["status"], "missing");
    assert_eq!(attestation["expected_status"], "missing");
    assert_eq!(attestation["ok"], true);
    let summary: Value =
        serde_json::from_slice(&fs::read(output.join("summary.json")).unwrap()).unwrap();
    assert_eq!(summary["prereq"], 0);
    assert_eq!(summary["attestation"]["passed"], 1);
}

#[cfg(unix)]
#[test]
fn stability_cli_optional_agent_uses_selected_root_absolute_writers_and_preserves_failure() {
    use std::{collections::BTreeMap, os::unix::fs::PermissionsExt};
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("selected source");
    fs::create_dir_all(source.join("tools/xtask")).unwrap();
    fs::create_dir_all(source.join("scripts")).unwrap();
    fs::create_dir_all(temp.path().join("bin")).unwrap();
    fs::write(source.join("Cargo.toml"), "fixture workspace").unwrap();
    fs::write(source.join("tools/xtask/Cargo.toml"), "fixture tool").unwrap();
    let pi = temp.path().join("bin/pi");
    fs::write(&pi, "#!/bin/bash\nexit 99\n").unwrap();
    fs::set_permissions(pi, fs::Permissions::from_mode(0o755)).unwrap();
    fs::write(source.join("scripts/ci-pi-smoke.sh"),
        "#!/bin/bash\nset -eu\n/bin/mkdir -p \"$PI_SMOKE_WORK_DIR\"\nprintf '%s\\n' \"$PWD\" \"$PI_SMOKE_WORK_DIR\" \"$MESH_AGENT_BASE_URL\" > \"$PI_SMOKE_OUTPUT\"\nprintf 'finite agent fixture\\n'\nexit 7\n").unwrap();
    let replies = vec![
        (200, json!({"data":[{"id":"fixture"}]}).to_string(), false),
        (200, answer("STABILITY_OK").to_string(), false),
        (200, call().to_string(), false),
        (200, answer("signal-7429").to_string(), false),
    ];
    let server = fixture::Server::new(replies);
    let output = temp.path().join("evidence with spaces");
    let mut args = vec!["--repo-root".into(), source.to_str().unwrap().into()];
    args.extend(arguments("nightly", &server.base, &output));
    args.extend([
        "--skip-streaming".into(),
        "--agent-smokes".into(),
        "pi,goose".into(),
    ]);
    let environment = BTreeMap::from([(
        "PATH".into(),
        process::Value::Public(format!("{}:/bin", temp.path().join("bin").display()).into()),
    )]);
    let result = fixture::invoke_with(temp.path(), &args, environment);
    assert_eq!(result.process.status.unwrap().code(), Some(1));
    let rows = fixture::rows(&output.join("commands.jsonl"));
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["status"], "PASS");
    assert_eq!(rows[1]["status"], "FAIL");
    assert_eq!(rows[1]["exit_code"], 7);
    assert_eq!(rows[2]["status"], "PREREQ");
    let capture = fs::read_to_string(output.join("agent-smokes/pi/pi-output.jsonl")).unwrap();
    let lines = capture.lines().collect::<Vec<_>>();
    assert_eq!(lines[0], source.canonicalize().unwrap().to_str().unwrap());
    assert_eq!(lines[1], output.join("agent-smokes/pi").to_str().unwrap());
    assert_eq!(lines[2], server.base);
    assert!(
        fs::read_to_string(output.join("logs/pi-agent-smoke.log"))
            .unwrap()
            .contains("finite agent fixture")
    );
    let summary: Value =
        serde_json::from_slice(&fs::read(output.join("summary.json")).unwrap()).unwrap();
    assert_eq!(summary["ok"], false);
    assert_eq!(summary["failed"], 1);
    assert_eq!(summary["prereq"], 2);
    let markdown = fs::read_to_string(output.join("summary.md")).unwrap();
    assert!(markdown.contains("| FAIL | pi-agent-smoke | 7 |"));
    assert!(markdown.contains("| PREREQ | goose-agent-smoke |"));
}
