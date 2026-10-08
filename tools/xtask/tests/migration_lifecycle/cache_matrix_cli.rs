use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::{Value as Json, json};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::{Ipv4Addr, TcpListener, TcpStream},
    num::NonZeroUsize,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};
struct Peer {
    url: String,
    stop: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<Json>>>,
    worker: Option<JoinHandle<()>>,
}
impl Peer {
    fn new(mode: &str) -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let flag = stop.clone();
        let seen = requests.clone();
        let mode = mode.to_owned();
        let worker = std::thread::spawn(move || {
            let mut count = BTreeMap::new();
            while !flag.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => reply(stream, &mode, &flag, &seen, &mut count),
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => panic!("{error}"),
                }
            }
        });
        Self {
            url,
            stop,
            requests,
            worker: Some(worker),
        }
    }
    fn close(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}
impl Drop for Peer {
    fn drop(&mut self) {
        self.close();
    }
}
fn reply(
    mut stream: TcpStream,
    mode: &str,
    stop: &AtomicBool,
    requests: &Mutex<Vec<Json>>,
    counts: &mut BTreeMap<String, u64>,
) {
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut bytes = Vec::new();
    let end = loop {
        let mut buffer = [0; 1024];
        let n = stream.read(&mut buffer).unwrap();
        if n == 0 {
            return;
        }
        bytes.extend_from_slice(&buffer[..n]);
        assert!(bytes.len() < 65536);
        if let Some(n) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
            break n + 4;
        }
    };
    let header = String::from_utf8(bytes[..end].to_vec()).unwrap();
    let path = header.split_whitespace().nth(1).unwrap().to_owned();
    let length = header
        .lines()
        .find_map(|line| {
            line.to_lowercase()
                .strip_prefix("content-length:")
                .map(str::trim)
                .map(str::parse::<usize>)
        })
        .unwrap()
        .unwrap();
    assert!(length < 65536);
    while bytes.len() - end < length {
        let mut buffer = [0; 1024];
        let n = stream.read(&mut buffer).unwrap();
        assert!(n > 0);
        bytes.extend_from_slice(&buffer[..n]);
    }
    let body: Json = serde_json::from_slice(&bytes[end..end + length]).unwrap();
    let auth = header
        .lines()
        .find(|line| line.to_lowercase().starts_with("authorization:"))
        .map(str::to_owned);
    requests
        .lock()
        .unwrap()
        .push(json!({"path":path,"body":body,"authorization":auth}));
    let occurrence = counts.entry(path.clone()).or_insert(0);
    let first = *occurrence == 0;
    *occurrence += 1;
    if mode == "held" && path.contains("warm-skippy") {
        while !stop.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(5));
        }
        return;
    }
    let enabled = path.contains("warm");
    let cached = if enabled && !first && mode != "zero" {
        if path.contains("native") { 8 } else { 9 }
    } else {
        0
    };
    let mut value = if path.ends_with("/completion") {
        json!({"content":"same inert output","stop":true,"tokens_predicted":2,"tokens_evaluated":10,"tokens_cached":99,"truncated":false,"stop_type":"limit","timings":{"prompt_n":10-cached,"cache_n":cached,"predicted_n":2,"prompt_ms":1.0,"predicted_ms":2.0}})
    } else {
        json!({"choices":[{"message":{"content":"same inert output"}}],"usage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":cached}}})
    };
    if mode == "missing" && path.contains("warm-skippy") {
        value["usage"]
            .as_object_mut()
            .unwrap()
            .remove("prompt_tokens_details");
    }
    if mode == "malformed" && path.contains("cold-skippy") {
        value["usage"]["prompt_tokens_details"]["cached_tokens"] = json!(999);
    }
    let code = if mode == "late-error" && path.contains("warm-skippy") && *occurrence == 3 {
        503
    } else {
        200
    };
    let bytes = serde_json::to_vec(&value).unwrap();
    let _ = write!(
        stream,
        "HTTP/1.1 {code} Fixture\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        bytes.len()
    );
    let _ = stream.write_all(&bytes);
}
fn args(root: &Path, peer: &Peer, pattern: &str) -> Vec<String> {
    vec![
        "--llama-cold-base-url".into(),
        format!("{}/cold-native", peer.url),
        "--llama-warm-base-url".into(),
        format!("{}/warm-native", peer.url),
        "--skippy-cold-base-url".into(),
        format!("{}/cold-skippy/v1", peer.url),
        "--skippy-warm-base-url".into(),
        format!("{}/warm-skippy/v1", peer.url),
        "--model".into(),
        "inert-model".into(),
        "--api-key".into(),
        "inert-private-token".into(),
        "--prefix-repetitions".into(),
        "2".into(),
        "--repeats".into(),
        "2".into(),
        "--max-tokens".into(),
        "2".into(),
        "--pattern".into(),
        pattern.into(),
        "--timeout".into(),
        "0.5".into(),
        "--output-dir".into(),
        root.join("new-parent/results").to_str().unwrap().into(),
    ]
}
fn cli(root: &Path, args: Vec<String>, cancel: &Cancellation) -> process::RawProcessReport {
    cli_with_environment(
        root,
        args,
        cancel,
        BTreeMap::new(),
        Duration::from_secs(10),
        Duration::from_secs(2),
    )
}
fn cli_with_environment(
    root: &Path,
    args: Vec<String>,
    cancel: &Cancellation,
    environment: BTreeMap<std::ffi::OsString, Value>,
    execution: Duration,
    graceful_shutdown: Duration,
) -> process::RawProcessReport {
    let spec = ProcessSpec {
        executable: env!("CARGO_BIN_EXE_xtask").into(),
        arguments: ["automation".to_owned(), "openai-cache-matrix".into()]
            .into_iter()
            .chain(args)
            .map(|s| Value::Public(s.into()))
            .collect(),
        cwd: root.into(),
        environment,
    };
    let limits = Limits {
        execution,
        graceful_shutdown,
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 1024 * 1024,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise_raw(
        &spec,
        &limits,
        cancel,
        RawCaptureOptions {
            stdout: NonZeroUsize::new(1024 * 1024),
            stderr: NonZeroUsize::new(1024 * 1024),
        },
    )
    .unwrap();
    assert!(
        report.process.cleanup.complete
            && !report.process.cleanup.forced
            && !report.process.cleanup.graceful_signal_failed
            && report.process.cleanup.failure.is_none()
            && report.process.failure.is_none(),
        "{report:?}"
    );
    for (stream, raw) in [
        (&report.process.stdout, report.stdout.as_ref()),
        (&report.process.stderr, report.stderr.as_ref()),
    ] {
        assert!(
            !stream.truncated && stream.line_capture_complete,
            "{report:?}"
        );
        assert_eq!(stream.bytes_seen, raw.unwrap().as_bytes().len() as u64);
        assert!(!String::from_utf8_lossy(raw.unwrap().as_bytes()).contains("inert-private-token"));
    }
    report
}
fn receipt(root: &Path) -> Json {
    serde_json::from_slice(
        &std::fs::read(root.join("new-parent/results/cache-matrix.json")).unwrap(),
    )
    .unwrap()
}
#[test]
fn exact_and_shared_prefix_complete_four_serial_rows_and_exclude_two_warmups() {
    for pattern in ["exact", "shared-prefix"] {
        let root = tempfile::tempdir().unwrap();
        let mut peer = Peer::new("success");
        let result = cli(
            root.path(),
            args(root.path(), &peer, pattern),
            &Cancellation::default(),
        );
        assert!(result.process.success(), "{result:?}");
        let report = receipt(root.path());
        assert_eq!(report["status"], "completed");
        let rows = report["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 4);
        for (index, row) in rows.iter().enumerate() {
            assert_eq!(row["runs"].as_array().unwrap().len(), 2);
            assert_eq!(row["warmup"].is_null(), index < 2);
            assert!(row["verdict"].as_str().unwrap().starts_with("PASS"));
        }
        assert_eq!(rows[2]["max_prompt_tokens"], 10);
        assert_eq!(rows[2]["max_cached_tokens"], 8);
        assert_eq!(rows[2]["max_suffix_prefill_tokens"], 2);
        assert_eq!(rows[3]["max_cacheable_prefix_tokens"], 9);
        assert_eq!(rows[3]["max_suffix_prefill_tokens"], 0);
        assert_eq!(rows[3]["max_cache_efficiency"], 1.0);
        assert_eq!(rows[2]["warmup"]["cached_tokens"], 0);
        let seen = peer.requests.lock().unwrap().clone();
        assert_eq!(seen.len(), 10);
        let positions = seen
            .iter()
            .map(|row| row["path"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            positions,
            [
                "/cold-native/completion",
                "/cold-native/completion",
                "/cold-skippy/v1/chat/completions",
                "/cold-skippy/v1/chat/completions",
                "/warm-native/completion",
                "/warm-native/completion",
                "/warm-native/completion",
                "/warm-skippy/v1/chat/completions",
                "/warm-skippy/v1/chat/completions",
                "/warm-skippy/v1/chat/completions"
            ]
        );
        assert_eq!(seen[0]["body"]["cache_prompt"], false);
        assert_eq!(seen[4]["body"]["cache_prompt"], true);
        assert_eq!(seen[0]["authorization"], Json::Null);
        assert_eq!(
            seen[2]["authorization"],
            "authorization: Bearer inert-private-token"
        );
        assert_eq!(seen[2]["body"]["model"], "inert-model");
        assert_eq!(seen[2]["body"]["max_tokens"], 2);
        assert_eq!(seen[2]["body"]["temperature"], 0);
        assert_eq!(seen[2]["body"]["top_p"], 1);
        let measured = seen[8]["body"]["messages"][1]["content"].as_str().unwrap();
        let warmup = seen[7]["body"]["messages"][1]["content"].as_str().unwrap();
        assert_eq!(measured == warmup, pattern == "exact");
        assert_eq!(
            rows[0]["runs"][0]["content_sha256"],
            rows[1]["runs"][0]["content_sha256"]
        );
        for name in ["cache-matrix.json", "cache-matrix.md"] {
            assert!(
                !std::fs::read_to_string(root.path().join("new-parent/results").join(name))
                    .unwrap()
                    .contains("inert-private-token")
            );
        }
        peer.close();
        root.close().unwrap();
    }
}
#[test]
fn zero_and_missing_warm_usage_retain_comparisons_and_exploration_never_invents_a_hit() {
    for mode in ["zero", "missing"] {
        for allow in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let mut peer = Peer::new(mode);
            let mut values = args(root.path(), &peer, "exact");
            if allow {
                values.push("--allow-missing-warm-cache".into());
            }
            let result = cli(root.path(), values, &Cancellation::default());
            assert_eq!(
                result.process.status.unwrap().code(),
                Some(if allow { 0 } else { 1 })
            );
            let report = receipt(root.path());
            assert_eq!(report["rows"].as_array().unwrap().len(), 4);
            assert_eq!(
                report["status"],
                if allow {
                    "completed"
                } else {
                    "warm-cache-proof-failed"
                }
            );
            assert_eq!(
                report["rows"][3]["verdict"],
                if mode == "missing" {
                    "UNQUALIFIED missing cache usage"
                } else {
                    "FAIL missed-hit"
                }
            );
            peer.close();
            root.close().unwrap();
        }
    }
}
#[test]
fn malformed_usage_and_late_http_failure_keep_completed_and_partial_rows_without_next_request() {
    for (mode, completed, requests) in [("malformed", 1, 3), ("late-error", 3, 10)] {
        let root = tempfile::tempdir().unwrap();
        let mut peer = Peer::new(mode);
        let result = cli(
            root.path(),
            args(root.path(), &peer, "exact"),
            &Cancellation::default(),
        );
        assert_eq!(result.process.status.unwrap().code(), Some(1));
        let report = receipt(root.path());
        assert_eq!(report["status"], "failed");
        assert_eq!(report["rows"].as_array().unwrap().len(), completed);
        assert!(report["incomplete_row"].is_object());
        if mode == "late-error" {
            assert_eq!(
                report["incomplete_row"]["runs"].as_array().unwrap().len(),
                1
            );
            assert!(report["incomplete_row"]["warmup"].is_object());
        }
        assert_eq!(peer.requests.lock().unwrap().len(), requests);
        peer.close();
        root.close().unwrap();
    }
}
#[test]
fn held_response_deadline_and_causal_cancellation_publish_partial_evidence_without_stopping_external_peer()
 {
    for cancel_case in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut peer = Peer::new("held");
        let cancel = Cancellation::default();
        let values = args(root.path(), &peer, "exact");
        std::thread::scope(|scope| {
            let token = cancel.clone();
            let seen = peer.requests.clone();
            let waiter = scope.spawn(move || {
                if cancel_case {
                    let deadline = Instant::now() + Duration::from_secs(5);
                    while seen.lock().unwrap().len() < 8 && Instant::now() < deadline {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    assert_eq!(seen.lock().unwrap().len(), 8);
                    token.cancel();
                }
            });
            let result = cli(root.path(), values, &cancel);
            waiter.join().unwrap();
            if cancel_case {
                assert_eq!(result.process.outcome, process::Outcome::Cancelled);
            } else {
                assert_eq!(result.process.status.unwrap().code(), Some(1));
            }
        });
        let report = receipt(root.path());
        assert_eq!(report["rows"].as_array().unwrap().len(), 3);
        assert_eq!(
            report["status"],
            if cancel_case { "cancelled" } else { "failed" }
        );
        assert!(!peer.worker.as_ref().unwrap().is_finished());
        assert_eq!(peer.requests.lock().unwrap().len(), 8);
        peer.close();
        root.close().unwrap();
    }
}
#[test]
fn invalid_endpoint_count_and_existing_output_refuse_before_external_requests() {
    for attack in ["zero", "nan", "credentials", "scheme", "query", "existing"] {
        let root = tempfile::tempdir().unwrap();
        let mut peer = Peer::new("success");
        let mut values = args(root.path(), &peer, "exact");
        let replace = |values: &mut Vec<String>, name: &str, value: &str| {
            let index = values.iter().position(|s| s == name).unwrap();
            values[index + 1] = value.into();
        };
        match attack {
            "zero" => replace(&mut values, "--repeats", "0"),
            "nan" => replace(&mut values, "--timeout", "NaN"),
            "credentials" => replace(
                &mut values,
                "--skippy-cold-base-url",
                "http://user:secret@127.0.0.1:9337/v1",
            ),
            "scheme" => replace(
                &mut values,
                "--skippy-cold-base-url",
                "ftp://127.0.0.1:9337/v1",
            ),
            "query" => replace(
                &mut values,
                "--skippy-cold-base-url",
                "https://unqualified.invalid/v1?secret=hidden",
            ),
            "existing" => {
                std::fs::create_dir_all(root.path().join("new-parent/results")).unwrap();
                std::fs::write(root.path().join("new-parent/results/sentinel"), "preserved")
                    .unwrap();
            }
            _ => unreachable!(),
        };
        let result = cli(root.path(), values, &Cancellation::default());
        assert_eq!(result.process.status.unwrap().code(), Some(1));
        assert!(peer.requests.lock().unwrap().is_empty());
        if attack == "existing" {
            assert_eq!(
                std::fs::read_to_string(root.path().join("new-parent/results/sentinel")).unwrap(),
                "preserved"
            );
        }
        peer.close();
        root.close().unwrap();
    }
}

#[path = "cache_matrix_transport_cli.rs"]
mod transport_cli;
