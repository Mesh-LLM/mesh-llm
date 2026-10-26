use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Argument,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
struct Peer {
    port: u16,
    stopped: Arc<AtomicBool>,
    received: Arc<AtomicBool>,
    task: Option<thread::JoinHandle<Result<Vec<Value>, String>>>,
}
impl Drop for Peer {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::SeqCst);
        if let Some(task) = self.task.take() {
            let _ = task.join();
        }
    }
}
fn read_request(socket: &mut TcpStream) -> Result<Value, String> {
    socket
        .set_read_timeout(Some(Duration::from_secs(1)))
        .map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    loop {
        let mut buf = [0; 4096];
        let n = socket.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("early request EOF".into());
        }
        bytes.extend_from_slice(&buf[..n]);
        if bytes.len() > 128 * 1024 {
            return Err("oversized fixture request".into());
        }
        if let Some(end) = bytes.windows(4).position(|b| b == b"\r\n\r\n") {
            let h = std::str::from_utf8(&bytes[..end]).map_err(|e| e.to_string())?;
            let length = h
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(str::trim)
                        .map(str::to_owned)
                })
                .ok_or("content length")?
                .parse::<usize>()
                .map_err(|e| e.to_string())?;
            if bytes.len() >= end + 4 + length {
                let v: Value = serde_json::from_slice(&bytes[end + 4..end + 4 + length])
                    .map_err(|e| e.to_string())?;
                return Ok(json!({"headers":h,"body":v}));
            }
        }
    }
}
impl Peer {
    fn new(
        cohort: &str,
        count: usize,
        hold_after: Option<usize>,
        fail_after: Option<usize>,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let stopped = Arc::new(AtomicBool::new(false));
        let stop = stopped.clone();
        let received = Arc::new(AtomicBool::new(false));
        let signal = received.clone();
        let cohort = cohort.to_owned();
        let task = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut requests = Vec::new();
            while requests.len() < count
                && !stop.load(Ordering::SeqCst)
                && Instant::now() < deadline
            {
                let (mut socket, _) = match listener.accept() {
                    Ok(v) => v,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(e) => return Err(e.to_string()),
                };
                socket.set_nonblocking(false).map_err(|e| e.to_string())?;
                socket
                    .set_write_timeout(Some(Duration::from_secs(1)))
                    .map_err(|e| e.to_string())?;
                requests.push(read_request(&mut socket)?);
                signal.store(true, Ordering::SeqCst);
                let mut native = json!({"stop":true,"content":"hi","tokens_predicted":1,"tokens_evaluated":40,"tokens_cached":41,
                    "truncated":false,"model":"fixture","stop_type":"limit","timings":{"prompt_n":10,"cache_n":30,"prompt_ms":2,"predicted_n":1,"predicted_ms":3}});
                if fail_after == Some(requests.len() - 1) {
                    native.as_object_mut().unwrap().remove("tokens_predicted");
                }
                let body=match cohort.as_str(){
                    "native-serial"=>serde_json::to_vec(&native).unwrap(),
                    "native-concurrent"=>{let mut final_event=native;final_event["content"]=json!("");format!("data: {}\n\ndata: {}\n\n",json!({"stop":false,"content":"hi"}),final_event).into_bytes()},
                    _=>format!("data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",json!({"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}),json!({"usage":{"prompt_tokens":40,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":30}}})).into_bytes(),
                };
                let hold = hold_after == Some(requests.len() - 1);
                let length = if hold { 100 } else { body.len() };
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n").as_bytes()).map_err(|e|e.to_string())?;
                if hold {
                    while !stop.load(Ordering::SeqCst) && Instant::now() < deadline {
                        thread::sleep(Duration::from_millis(5));
                    }
                } else {
                    socket.write_all(&body).map_err(|e| e.to_string())?;
                }
            }
            Ok(requests)
        });
        Self {
            port,
            stopped,
            received,
            task: Some(task),
        }
    }
    fn finish(mut self) -> Vec<Value> {
        self.stopped.store(true, Ordering::SeqCst);
        self.task.take().unwrap().join().unwrap().unwrap()
    }
}
struct Fixture {
    directory: tempfile::TempDir,
    root: PathBuf,
    input: PathBuf,
    output: PathBuf,
}
impl Fixture {
    fn new(cohort: &str, port: u16) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let input = root.join("input.json");
        let output = root.join("receipt.json");
        let serial = cohort == "native-serial";
        let openai = cohort == "openai-concurrent";
        fs::write(&input,serde_json::to_vec(&json!({"schema_version":1,"cohort":cohort,"base_url":format!("http://127.0.0.1:{port}{}",if openai{"/v1"}else{"/"}),
            "prompt":"fixed prompt","model_id":if openai{Some("declared-model")}else{None},"requests":if serial{3}else{2},"concurrency":if serial{1}else{2},
            "output_tokens":if serial{1}else if openai{32}else{128},"request_timeout_ms":5000,"execution_timeout_ms":8000})).unwrap()).unwrap();
        Self {
            directory,
            root,
            input,
            output,
        }
    }
    fn spec(&self) -> ProcessSpec {
        ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            arguments: [
                "automation".into(),
                "cache-family-measure".into(),
                "--input".into(),
                self.input.clone().into_os_string(),
                "--output".into(),
                self.output.clone().into_os_string(),
            ]
            .into_iter()
            .map(Argument::Public)
            .collect(),
            cwd: self.root.clone(),
            environment: BTreeMap::new(),
        }
    }
    fn receipt(&self) -> Value {
        serde_json::from_slice(&fs::read(&self.output).unwrap()).unwrap()
    }
}
fn invoke(spec: &ProcessSpec, cancel: &Cancellation) -> process::ProcessReport {
    let r = process::supervise(
        spec,
        &Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancel,
        OutputFiles::default(),
    )
    .unwrap();
    assert!(
        r.failure.is_none()
            && r.cleanup.complete
            && !r.cleanup.forced
            && r.cleanup.failure.is_none()
            && !r.cleanup.graceful_signal_failed
    );
    assert!(
        [&r.stdout, &r.stderr]
            .iter()
            .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0)
    );
    r
}
#[test]
fn cache_measure_actual_cli_preserves_all_three_cohorts_and_request_correlation() {
    for cohort in ["native-serial", "native-concurrent", "openai-concurrent"] {
        let peer = Peer::new(
            cohort,
            if cohort == "native-serial" { 3 } else { 2 },
            None,
            None,
        );
        let fixture = Fixture::new(cohort, peer.port);
        let r = invoke(&fixture.spec(), &Cancellation::default());
        let requests = peer.finish();
        assert_eq!(r.outcome, process::Outcome::Exited);
        assert_eq!(
            r.status.as_ref().and_then(std::process::ExitStatus::code),
            Some(0)
        );
        let receipt = fixture.receipt();
        assert_eq!(receipt["status"], "completed");
        use sha2::{Digest, Sha256};
        assert_eq!(
            receipt["request_sha256"],
            hex::encode(Sha256::digest(fs::read(&fixture.input).unwrap()))
        );
        assert_eq!(receipt["rows"].as_array().unwrap().len(), requests.len());
        for (id, row) in receipt["rows"].as_array().unwrap().iter().enumerate() {
            assert_eq!(row["request_id"], id);
            assert_eq!(row["tokens_predicted"], 1);
            assert_eq!(row["status"], "completed");
        }
        if cohort == "native-serial" {
            assert_eq!(receipt["rows"][0]["excluded_warmup"], true);
            assert!(receipt["rows"][0]["ttft_ms"].is_null());
            assert_eq!(receipt["summary"]["excluded_runs"], 1);
        } else {
            assert!(
                receipt["summary"]["output_tokens_per_sec"]
                    .as_f64()
                    .unwrap()
                    > 0.0
            );
            assert!(receipt["rows"][0]["ttft_ms"].as_f64().unwrap() >= 0.0);
        }
        for request in requests {
            let body = &request["body"];
            let headers = request["headers"].as_str().unwrap();
            if cohort == "openai-concurrent" {
                assert!(headers.starts_with("POST /v1/chat/completions HTTP/1.1"));
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("authorization: bearer empty")
                );
                assert_eq!(
                    *body,
                    json!({"model":"declared-model","messages":[{"role":"user","content":"fixed prompt"}],
                    "max_tokens":32,"temperature":0,"seed":0,"stream":true,"stream_options":{"include_usage":true}})
                );
            } else {
                assert!(headers.starts_with("POST /completion HTTP/1.1"));
                assert!(!headers.to_ascii_lowercase().contains("authorization:"));
                assert_eq!(
                    *body,
                    json!({"prompt":"fixed prompt","n_predict":if cohort=="native-serial"{1}else{128},
                    "temperature":0,"top_k":1,"cache_prompt":true,"stream":cohort!="native-serial"})
                );
            }
            assert_eq!(
                body[if cohort == "openai-concurrent" {
                    "max_tokens"
                } else {
                    "n_predict"
                }],
                if cohort == "native-serial" {
                    1
                } else if cohort == "openai-concurrent" {
                    32
                } else {
                    128
                }
            );
        }
        fixture.directory.close().unwrap();
    }
}
#[test]
fn cache_measure_actual_cli_refuses_existing_output_without_http_request() {
    let peer = Peer::new("native-serial", 3, None, None);
    let fixture = Fixture::new("native-serial", peer.port);
    fs::write(&fixture.output, b"sentinel").unwrap();
    let r = invoke(&fixture.spec(), &Cancellation::default());
    let requests = peer.finish();
    assert_eq!(r.outcome, process::Outcome::Exited);
    assert_eq!(
        r.status.as_ref().and_then(std::process::ExitStatus::code),
        Some(1)
    );
    assert!(requests.is_empty());
    assert_eq!(fs::read(&fixture.output).unwrap(), b"sentinel");
    fixture.directory.close().unwrap();
}
#[cfg(unix)]
#[test]
fn cache_measure_actual_cli_inflight_cancel_retains_partial_roster_and_reaps_owned_worker() {
    let peer = Peer::new("native-serial", 3, Some(0), None);
    let fixture = Fixture::new("native-serial", peer.port);
    let cancellation = Cancellation::default();
    let child_cancel = cancellation.clone();
    let spec = fixture.spec();
    let task = thread::spawn(move || invoke(&spec, &child_cancel));
    let deadline = Instant::now() + Duration::from_secs(3);
    while !peer.received.load(Ordering::SeqCst) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    let admitted = peer.received.load(Ordering::SeqCst);
    cancellation.cancel();
    let r = task.join().unwrap();
    let requests = peer.finish();
    assert!(admitted, "did not cancel before a real request");
    assert!(!requests.is_empty());
    assert_eq!(r.outcome, process::Outcome::Cancelled);
    let receipt = fixture.receipt();
    assert_eq!(receipt["status"], "incomplete");
    assert!(receipt["summary"].is_null());
    assert_eq!(receipt["rows"].as_array().unwrap().len(), 3);
    assert!(receipt["rows"][0]["evidence"].is_null());
    fixture.directory.close().unwrap();
}

#[test]
fn cache_measure_actual_cli_protocol_failure_preserves_prior_measured_receipt() {
    let peer = Peer::new("native-serial", 3, None, Some(2));
    let fixture = Fixture::new("native-serial", peer.port);
    let report = invoke(&fixture.spec(), &Cancellation::default());
    let requests = peer.finish();
    assert_eq!(report.outcome, process::Outcome::Exited);
    assert_eq!(
        report
            .status
            .as_ref()
            .and_then(std::process::ExitStatus::code),
        Some(1)
    );
    assert_eq!(requests.len(), 3);
    let receipt = fixture.receipt();
    assert_eq!(receipt["status"], "incomplete");
    assert!(receipt["summary"].is_null());
    assert_eq!(receipt["rows"][0]["excluded_warmup"], true);
    assert_eq!(receipt["rows"][1]["status"], "completed");
    assert_eq!(receipt["rows"][1]["tokens_predicted"], 1);
    assert_eq!(receipt["rows"][1]["evidence"]["tokens_evaluated"], 40);
    assert_eq!(receipt["rows"][2]["status"], "failed");
    assert!(receipt["rows"][2]["evidence"].is_null());
    fixture.directory.close().unwrap();
}

#[test]
fn cache_measure_actual_cli_whole_budget_retains_earlier_success_before_held_response() {
    let peer = Peer::new("native-serial", 3, Some(2), None);
    let fixture = Fixture::new("native-serial", peer.port);
    let mut input: Value = serde_json::from_slice(&fs::read(&fixture.input).unwrap()).unwrap();
    input["execution_timeout_ms"] = json!(1500);
    // The whole budget, not the much longer per-request cap, must end this held response.
    input["request_timeout_ms"] = json!(5000);
    fs::write(&fixture.input, serde_json::to_vec(&input).unwrap()).unwrap();
    let report = invoke(&fixture.spec(), &Cancellation::default());
    let requests = peer.finish();
    assert_eq!(report.outcome, process::Outcome::Exited);
    assert_eq!(
        report
            .status
            .as_ref()
            .and_then(std::process::ExitStatus::code),
        Some(1)
    );
    assert_eq!(
        requests.len(),
        3,
        "deadline was not exercised after the third actual POST"
    );
    let receipt = fixture.receipt();
    assert_eq!(receipt["status"], "incomplete");
    assert!(receipt["summary"].is_null());
    assert_eq!(receipt["rows"].as_array().unwrap().len(), 3);
    assert_eq!(receipt["rows"][1]["status"], "completed");
    assert_eq!(receipt["rows"][1]["tokens_predicted"], 1);
    assert_ne!(receipt["rows"][2]["status"], "completed");
    assert!(receipt["rows"][2]["evidence"].is_null());
    fixture.directory.close().unwrap();
}

#[test]
fn cache_measure_actual_one_repeat_preserves_observation_with_null_warm_statistics() {
    let peer = Peer::new("native-serial", 1, None, None);
    let fixture = Fixture::new("native-serial", peer.port);
    let mut input: Value = serde_json::from_slice(&fs::read(&fixture.input).unwrap()).unwrap();
    input["requests"] = json!(1);
    fs::write(&fixture.input, serde_json::to_vec(&input).unwrap()).unwrap();
    let report = invoke(&fixture.spec(), &Cancellation::default());
    let requests = peer.finish();
    assert_eq!(report.outcome, process::Outcome::Exited);
    assert_eq!(report.status.and_then(|status| status.code()), Some(0));
    assert_eq!(requests.len(), 1);
    let receipt = fixture.receipt();
    assert_eq!(receipt["status"], "completed");
    assert_eq!(receipt["rows"].as_array().unwrap().len(), 1);
    assert_eq!(receipt["rows"][0]["status"], "completed");
    assert_eq!(receipt["rows"][0]["excluded_warmup"], true);
    assert_eq!(receipt["rows"][0]["tokens_predicted"], 1);
    assert_eq!(receipt["summary"]["excluded_runs"], 1);
    assert!(receipt["summary"]["warm_mean_ms"].is_null());
    assert!(receipt["summary"]["warm_median_ms"].is_null());
    fixture.directory.close().unwrap();
}
