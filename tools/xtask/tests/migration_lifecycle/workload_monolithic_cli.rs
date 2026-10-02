//! Actual CLI, finite HTTP endpoints and an independent fake native completion process.
#![cfg(unix)]
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use serde_json::{Value as Json, json};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    os::unix::fs::PermissionsExt,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

struct Server {
    url: String,
    requests: Arc<Mutex<Vec<Json>>>,
    stop: Arc<AtomicBool>,
    worker: Mutex<Option<thread::JoinHandle<Result<(), String>>>>,
}
fn read_payload(stream: &mut TcpStream) -> Result<Json, String> {
    stream
        .set_nonblocking(false)
        .map_err(|error| error.to_string())?;
    stream
        .set_read_timeout(Some(Duration::from_millis(100)))
        .map_err(|error| error.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut raw = Vec::new();
    let mut buffer = [0; 4096];
    loop {
        if Instant::now() >= deadline {
            return Err(format!(
                "request read deadline; received {} bytes; headers: {}",
                raw.len(),
                String::from_utf8_lossy(&raw[..raw.len().min(4096)])
            ));
        }
        let n = match stream.read(&mut buffer) {
            Ok(0) => return Err("request EOF before complete body".into()),
            Ok(n) => n,
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::Interrupted
                ) =>
            {
                continue;
            }
            Err(error) => return Err(format!("request read failed: {error}")),
        };
        raw.extend_from_slice(&buffer[..n]);
        if raw.len() > 65536 {
            return Err("fixture request exceeds 64 KiB".into());
        }
        let Some(split) = raw.windows(4).position(|bytes| bytes == b"\r\n\r\n") else {
            continue;
        };
        let headers = std::str::from_utf8(&raw[..split]).map_err(|error| error.to_string())?;
        let length = headers
            .lines()
            .find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .map(|value| value.trim().parse::<usize>())
            })
            .ok_or_else(|| format!("missing Content-Length in fixture request: {headers}"))?
            .map_err(|error| error.to_string())?;
        if length > 65536 - split - 4 {
            return Err("declared fixture body exceeds bound".into());
        }
        if raw.len() >= split + 4 + length {
            return serde_json::from_slice(&raw[split + 4..split + 4 + length])
                .map_err(|error| error.to_string());
        }
    }
}
impl Server {
    fn new(
        mut reply: impl FnMut(usize, &Json) -> (u16, &'static str, String) + Send + 'static,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = requests.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let halted = stop.clone();
        let worker = thread::spawn(move || {
            let end = Instant::now() + Duration::from_secs(12);
            while !halted.load(Ordering::SeqCst) && Instant::now() < end {
                let (mut stream, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(error) => return Err(error.to_string()),
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .map_err(|error| error.to_string())?;
                stream
                    .set_write_timeout(Some(Duration::from_secs(2)))
                    .map_err(|error| error.to_string())?;
                let payload = read_payload(&mut stream)?;
                let index = {
                    let mut rows = captured.lock().unwrap();
                    let index = rows.len();
                    rows.push(payload.clone());
                    index
                };
                let (status, mime, body) = reply(index, &payload);
                write!(stream,"HTTP/1.1 {status} Fixture\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).map_err(|error| error.to_string())?;
            }
            Ok(())
        });
        Self {
            url,
            requests,
            stop,
            worker: Mutex::new(Some(worker)),
        }
    }
    fn finish(&self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(worker) = self.worker.lock().unwrap().take() {
            let result = worker.join().expect("HTTP fixture worker panicked");
            assert!(result.is_ok(), "HTTP fixture worker failed: {result:?}");
        }
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Ok(slot) = self.worker.get_mut()
            && let Some(worker) = slot.take()
        {
            let _ = worker.join();
        }
    }
}
fn invoke(directory: &Path, args: Vec<String>) -> process::ProcessReport {
    invoke_cancel(directory, args, &Cancellation::default())
}
fn invoke_cancel(
    directory: &Path,
    args: Vec<String>,
    cancellation: &Cancellation,
) -> process::ProcessReport {
    let spec = ProcessSpec {
        executable: env!("CARGO_BIN_EXE_xtask").into(),
        arguments: ["automation", "workload-monolithic-oracle"]
            .into_iter()
            .map(|s| Value::Public(s.into()))
            .chain(args.into_iter().map(|s| Value::Public(s.into())))
            .collect(),
        cwd: directory.into(),
        environment: BTreeMap::new(),
    };
    let limits = Limits {
        execution: Duration::from_secs(8),
        graceful_shutdown: Duration::from_secs(4),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let output = process::supervise(&spec, &limits, cancellation, OutputFiles::default()).unwrap();
    assert!(output.cleanup.complete, "{output:?}");
    output
}
fn arguments(candidate: &Server, reference: &Server, class: &str) -> Vec<String> {
    [
        "--candidate-url",
        &candidate.url,
        "--oracle-url",
        &reference.url,
        "--model",
        "fixture",
        "--class",
        class,
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}
fn embeddings(_index: usize, payload: &Json) -> (u16, &'static str, String) {
    let count = payload["input"].as_array().map_or(1, Vec::len);
    (200,"application/json",json!({"data":(0..count).map(|index|json!({"index":index,"embedding":[1,0]})).collect::<Vec<_>>()} ).to_string())
}
#[test]
fn independent_http_embeddings_compare_batch_and_every_single_even_after_a_divergence() {
    for divergence in [None, Some(0), Some(2)] {
        let state = tempfile::tempdir().unwrap();
        let reference = Server::new(embeddings);
        let candidate = Server::new(move |index, payload| {
            if divergence == Some(index) {
                (
                    200,
                    "application/json",
                    json!({"data":(0..payload["input"].as_array().map_or(1,Vec::len)).map(|index|json!({"index":index,"embedding":[0,1]})).collect::<Vec<_>>()}).to_string(),
                )
            } else {
                embeddings(index, payload)
            }
        });
        let output = invoke(state.path(), arguments(&candidate, &reference, "embedding"));
        candidate.finish();
        reference.finish();
        assert_eq!(output.success(), divergence.is_none(), "{output:?}");
        if divergence.is_none() {
            assert!(
                String::from_utf8_lossy(&output.stdout.bytes_retained)
                    .starts_with("embedding local-monolithic oracle passed: "),
                "{output:?}"
            );
        }
        for server in [&candidate, &reference] {
            let rows = server.requests.lock().unwrap();
            assert_eq!(rows.len(), 4, "{rows:?}");
            assert_eq!(
                rows[0]["input"],
                json!([
                    "search_query: distributed GPU inference",
                    "search_document: GPUs share one language model over a mesh",
                    "search_document: A recipe for tomato soup"
                ])
            );
            for index in 0..3 {
                assert_eq!(rows[index + 1]["input"], rows[0]["input"][index]);
                assert_eq!(rows[index + 1]["encoding_format"], "float");
            }
        }
        if let Some(index) = divergence {
            assert!(
                String::from_utf8_lossy(&output.stderr.bytes_retained).contains(if index == 0 {
                    "batched"
                } else {
                    "single[1]"
                }),
                "{output:?}"
            );
        }
    }
}
#[test]
fn independent_http_rerank_rejects_score_and_wire_order_disagreement() {
    for changed in [0, 1, 2] {
        let state = tempfile::tempdir().unwrap();
        let reference = Server::new(|_, _| {
            (200,"application/json",json!({"results":[{"index":0,"relevance_score":2},{"index":1,"relevance_score":-1}]}).to_string())
        });
        let candidate = Server::new(move |_, payload| {
            assert_eq!(payload["return_documents"], true);
            assert_eq!(payload["documents"].as_array().unwrap().len(), 2);
            let mut rows =
                json!([{"index":0,"relevance_score":2},{"index":1,"relevance_score":-1}]);
            if changed == 1 {
                rows.as_array_mut().unwrap().reverse();
            }
            if changed == 2 {
                rows[0]["relevance_score"] = json!(2.1);
            }
            (200, "application/json", json!({"results":rows}).to_string())
        });
        let output = invoke(state.path(), arguments(&candidate, &reference, "rerank"));
        candidate.finish();
        reference.finish();
        assert_eq!(output.success(), changed == 0, "{output:?}");
    }
}
#[test]
fn independent_native_translation_preserves_deterministic_argv_and_terminal_marker_only() {
    for mode in [
        "equal",
        "equal-token",
        "oversized",
        "different",
        "empty",
        "failed",
        "internal-marker",
    ] {
        let state = tempfile::tempdir().unwrap();
        let native = state.path().join("independent completion with spaces");
        let trace = state.path().join("argv");
        let model = state.path().join("immutable fixture.gguf");
        fs::write(&model, b"GGUF finite fake").unwrap();
        let answer = match mode {
            "equal" => "Das   Haus ist wunderbar. [end of text]",
            "equal-token" => "Das token authorization Haus ist wunderbar. [end of text]",
            "different" => "Das Auto ist wunderbar. [end of text]",
            "empty" => " [end of text]",
            "internal-marker" => "Das [end of text] Haus ist wunderbar. [end of text]",
            _ => "",
        };
        fs::write(
            &native,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\nprintf '%s\\n' '{}'\nexit {}\n",
                trace.display(),
                answer,
                if mode == "failed" { 7 } else { 0 }
            ),
        )
        .unwrap();
        if mode == "oversized" {
            fs::write(&native, format!("#!/bin/sh\nprintf '%s\n' \"$@\" > '{}'\nexec /usr/bin/head -c 4194305 /dev/zero\n", trace.display())).unwrap();
        }
        fs::set_permissions(&native, fs::Permissions::from_mode(0o700)).unwrap();

        let candidate = Server::new(move |_, payload| {
            assert_eq!(payload["seed"], 1);
            assert_eq!(payload["temperature"], 0.0);
            assert_eq!(payload["max_tokens"], 32);
            assert_eq!(
                payload["prompt"],
                "translate English to German: The house is wonderful."
            );
            (
                200,
                "application/json",
                json!({"choices":[{"text":if mode=="equal-token" {"Das token authorization Haus ist wunderbar."} else {" Das Haus ist wunderbar.\n"}}]}).to_string(),
            )
        });
        let args = vec![
            "--candidate-url".into(),
            candidate.url.clone(),
            "--oracle-completion".into(),
            native.to_str().unwrap().into(),
            "--model-path".into(),
            model.to_str().unwrap().into(),
            "--model".into(),
            "fixture".into(),
            "--class".into(),
            "encoder_decoder".into(),
        ];
        let output = invoke(state.path(), args);
        candidate.finish();
        assert_eq!(
            output.success(),
            mode.starts_with("equal"),
            "mode={mode}: {output:?}"
        );
        let argv = fs::read_to_string(trace).unwrap();
        let expected = format!(
            "-m\n{}\n-p\ntranslate English to German: The house is wonderful.\n-n\n32\n-c\n0\n-b\n2048\n-ub\n2048\n-ngl\n0\n-s\n1\n--temp\n0\n--no-repack\n--no-display-prompt\n--simple-io\n",
            model.display()
        );
        assert_eq!(argv, expected);
        if !mode.starts_with("equal") {
            assert!(output.stdout.bytes_retained.is_empty(), "{output:?}");
        }
    }
}
#[test]
fn http_status_content_type_and_object_admission_cannot_pass_comparison() {
    for (status, mime, body) in [
        (503, "application/json", "{}"),
        (200, "text/plain", "{}"),
        (200, "application/json", "[]"),
        (200, "application/json", "{broken"),
    ] {
        let state = tempfile::tempdir().unwrap();
        let candidate = Server::new(move |_, _| (status, mime, body.into()));
        let reference = Server::new(embeddings);
        let output = invoke(state.path(), arguments(&candidate, &reference, "embedding"));
        candidate.finish();
        reference.finish();
        assert!(!output.success(), "{output:?}");
        assert!(output.stdout.bytes_retained.is_empty());
        assert!(reference.requests.lock().unwrap().is_empty());
    }
}
#[test]
fn invalid_reference_selection_is_rejected_before_http_or_native_launch() {
    let state = tempfile::tempdir().unwrap();
    let candidate = Server::new(embeddings);
    let reference = Server::new(embeddings);
    let mut args = arguments(&candidate, &reference, "embedding");
    args.extend(["--oracle-completion".into(), "/unlaunched-native".into()]);
    let output = invoke(state.path(), args);
    candidate.finish();
    reference.finish();
    assert!(!output.success(), "{output:?}");
    assert!(candidate.requests.lock().unwrap().is_empty());
    assert!(reference.requests.lock().unwrap().is_empty());
    let output = invoke(
        state.path(),
        vec![
            "--candidate-url".into(),
            candidate.url.clone(),
            "--model".into(),
            "fixture".into(),
            "--class".into(),
            "encoder_decoder".into(),
        ],
    );
    assert!(!output.success(), "{output:?}");
    assert!(candidate.requests.lock().unwrap().is_empty());
}

struct UnrelatedSentinel(std::process::Child);
impl Drop for UnrelatedSentinel {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
#[test]
fn outer_cancellation_of_native_comparator_reaps_detached_oracle_tree_and_preserves_sentinel() {
    let state = tempfile::tempdir().unwrap();
    let model = state.path().join("model.gguf");
    fs::write(&model, b"inert GGUF fixture").unwrap();
    let marker = state.path().join("descendant.pid");
    let native = state.path().join("native fixture");
    fs::write(&native, format!(
        "#!/bin/bash\nset -euo pipefail\n/bin/sleep 30 &\nchild=$!\ntrap 'kill \"$child\" 2>/dev/null || true; wait \"$child\" 2>/dev/null || true; exit 143' TERM INT\nprintf '%s' \"$child\" > '{}'\nwait \"$child\"\n", marker.display()
    )).unwrap();
    fs::set_permissions(&native, fs::Permissions::from_mode(0o700)).unwrap();
    let mut sentinel = UnrelatedSentinel(
        std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .unwrap(),
    );
    let candidate = Server::new(|_, _| {
        (
            200,
            "application/json",
            json!({"choices":[{"text":"Das Haus ist wunderbar."}]}).to_string(),
        )
    });
    let arguments = vec![
        "--candidate-url".into(),
        candidate.url.clone(),
        "--oracle-completion".into(),
        native.to_str().unwrap().into(),
        "--model-path".into(),
        model.to_str().unwrap().into(),
        "--model".into(),
        "fixture".into(),
        "--class".into(),
        "encoder_decoder".into(),
    ];
    let cancellation = Cancellation::default();
    let trigger = cancellation.clone();
    let output = std::thread::scope(|scope| {
        scope.spawn(|| {
            let deadline = Instant::now() + Duration::from_secs(6);
            while !marker.exists() {
                assert!(Instant::now() < deadline, "native descendant did not start");
                thread::sleep(Duration::from_millis(10));
            }
            trigger.cancel();
        });
        invoke_cancel(state.path(), arguments, &cancellation)
    });
    candidate.finish();
    assert_eq!(output.outcome, process::Outcome::Cancelled, "{output:?}");
    assert!(
        output.cleanup.complete && !output.cleanup.forced,
        "{output:?}"
    );
    assert!(output.stdout.bytes_retained.is_empty(), "{output:?}");
    let pid: i32 = fs::read_to_string(marker).unwrap().parse().unwrap();
    // SAFETY: signal zero observes only the descendant recorded by this fixture.
    assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
    assert!(
        sentinel.0.try_wait().unwrap().is_none(),
        "unrelated sentinel was killed"
    );
}
