//! Actual inert caller fixtures; not MTP runtime or external endpoint qualification.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value as Arg,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
struct Peer {
    port: u16,
    stop: Arc<AtomicBool>,
    seen: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<Value>>>,
    task: Option<thread::JoinHandle<Result<(), String>>>,
}
fn request(s: &mut TcpStream) -> Result<Value, String> {
    s.set_nonblocking(false).map_err(|e| e.to_string())?;
    s.set_read_timeout(Some(Duration::from_secs(1)))
        .map_err(|e| e.to_string())?;
    s.set_write_timeout(Some(Duration::from_secs(1)))
        .map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    loop {
        let mut b = [0; 4096];
        let n = s.read(&mut b).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("request EOF".into());
        }
        bytes.extend_from_slice(&b[..n]);
        if bytes.len() > 131072 {
            return Err("large fixture request".into());
        }
        if let Some(end) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
            let h = std::str::from_utf8(&bytes[..end]).map_err(|e| e.to_string())?;
            let length = h
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().to_owned())
                })
                .map_or(Ok(0), |v| v.parse::<usize>())
                .map_err(|e| e.to_string())?;
            if bytes.len() >= end + 4 + length {
                let body = if length == 0 {
                    Value::Null
                } else {
                    serde_json::from_slice(&bytes[end + 4..end + 4 + length])
                        .map_err(|e| e.to_string())?
                };
                return Ok(json!({"headers":h,"body":body}));
            }
        }
    }
}
fn response(source: &str) -> Value {
    json!({"choices":[{"message":{"content":"same"},"finish_reason":"stop"}],"timings":{"predicted_n":2,"predicted_per_second":50.0,"draft_n":2,"draft_n_accepted":1,"native_mtp_ngram_proposer":source,"native_mtp_hybrid_ngram_tokens":2,"native_mtp_hybrid_accepted_tail_tokens":1,"native_mtp_ngram_proposer_attempts":1,"native_mtp_ngram_proposer_hits":if source=="suffix"{1}else{0},"native_mtp_ngram_proposer_match_length_max":2,"native_mtp_ngram_proposer_candidates_examined":1,"native_mtp_ngram_proposer_appended_tokens":2,"native_mtp_ngram_proposer_rebuilds":1,"native_mtp_ngram_proposer_sync_us":4,"native_mtp_ngram_proposer_lookup_us":5}})
}
impl Peer {
    fn new(source: &str, mode: &str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let seen = Arc::new(AtomicBool::new(false));
        let signal = seen.clone();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let records = requests.clone();
        let source = source.to_owned();
        let mode = mode.to_owned();
        let task = thread::spawn(move || {
            let until = Instant::now() + Duration::from_secs(15);
            let mut posts = 0;
            while !stopped.load(Ordering::SeqCst) && Instant::now() < until {
                let (mut s, _) = match listener.accept() {
                    Ok(v) => v,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(e) => return Err(e.to_string()),
                };
                let r = request(&mut s)?;
                let post = r["headers"].as_str().unwrap().starts_with("POST");
                records.lock().unwrap().push(r);
                if post {
                    posts += 1;
                    signal.store(true, Ordering::SeqCst)
                }
                let body = if post {
                    response(if mode == "wrong-source" {
                        "off"
                    } else {
                        &source
                    })
                } else {
                    json!({"data":[{"id":"model"}]})
                };
                let bad = post && mode == "later-failure" && posts == 2;
                let bytes = serde_json::to_vec(&body).unwrap();
                let hold = post && mode == "hold";
                s.write_all(
                    format!(
                        "HTTP/1.1 {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        if bad { "503 Unavailable" } else { "200 OK" },
                        if hold { bytes.len() + 100 } else { bytes.len() }
                    )
                    .as_bytes(),
                )
                .map_err(|e| e.to_string())?;
                if hold {
                    while !stopped.load(Ordering::SeqCst) && Instant::now() < until {
                        let mut b = [0; 1];
                        match s.read(&mut b) {
                            Ok(0) => break,
                            Ok(_) => (),
                            Err(e)
                                if matches!(
                                    e.kind(),
                                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                                ) => {}
                            Err(e) => return Err(e.to_string()),
                        }
                    }
                } else {
                    let _ = s.write_all(&bytes);
                }
            }
            Ok(())
        });
        Self {
            port,
            stop,
            seen,
            requests,
            task: Some(task),
        }
    }
    fn finish(mut self) -> Vec<Value> {
        self.stop.store(true, Ordering::SeqCst);
        self.task.take().unwrap().join().unwrap().unwrap();
        self.requests.lock().unwrap().clone()
    }
}
impl Drop for Peer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.task.take() {
            let _ = t.join();
        }
    }
}
struct Fixture {
    directory: tempfile::TempDir,
    input: PathBuf,
    output: PathBuf,
}
impl Fixture {
    fn new(a: u16, b: u16, warmups: u32) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let input = root.join("input.json");
        let output = root.join("out");
        let corpus = root.join("empty.jsonl");
        std::fs::write(&corpus, "").unwrap();
        std::fs::write(&input,serde_json::to_vec(&json!({"schema_version":1,"arms":[{"name":"off","base_url":format!("http://127.0.0.1:{a}"),"declared_stages":2,"declared_mtp_capable":true},{"name":"suffix","base_url":format!("http://127.0.0.1:{b}"),"declared_stages":2,"declared_mtp_capable":true}],"model":"model","corpus":corpus,"warmups":warmups,"runs":2,"max_tokens":256,"request_timeout_ms":3000,"execution_timeout_ms":10000})).unwrap()).unwrap();
        Self {
            directory,
            input,
            output,
        }
    }
    fn spec(&self) -> ProcessSpec {
        ProcessSpec {
            executable: PathBuf::from(env!("CARGO_BIN_EXE_xtask")),
            arguments: [
                "automation".into(),
                "suffix-proposer".into(),
                "--input".into(),
                self.input.as_os_str().into(),
                "--output-directory".into(),
                self.output.as_os_str().into(),
            ]
            .into_iter()
            .map(Arg::Public)
            .collect(),
            cwd: self.directory.path().canonicalize().unwrap(),
            environment: BTreeMap::new(),
        }
    }
    fn report(&self) -> Value {
        serde_json::from_slice(&std::fs::read(self.output.join("report.json")).unwrap()).unwrap()
    }
}
fn invoke(spec: &ProcessSpec, c: &Cancellation) -> process::RawProcessReport {
    let r = process::supervise_raw(
        spec,
        &Limits {
            execution: Duration::from_secs(12),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        c,
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    let p = &r.process;
    assert!(
        p.failure.is_none()
            && p.cleanup.complete
            && !p.cleanup.forced
            && !p.cleanup.graceful_signal_failed
            && p.cleanup.failure.is_none()
    );
    for (s, b) in [
        (&p.stdout, r.stdout.as_ref().unwrap()),
        (&p.stderr, r.stderr.as_ref().unwrap()),
    ] {
        assert!(s.line_capture_complete && !s.truncated);
        assert_eq!(s.bytes_seen, b.as_bytes().len() as u64);
    }
    r
}
#[test]
fn suffix_actual_cli_excludes_warmups_and_reports_matched_measured_activation() {
    let a = Peer::new("off", "");
    let b = Peer::new("suffix", "");
    let f = Fixture::new(a.port, b.port, 1);
    let r = invoke(&f.spec(), &Cancellation::default());
    let requests = [a.finish(), b.finish()];
    assert_eq!(r.process.outcome, Outcome::Exited);
    assert!(r.process.status.unwrap().success());
    let report = f.report();
    assert_eq!(report["status"], "PASS");
    assert_eq!(report["completed_warmups"], 4);
    assert_eq!(report["samples"].as_array().unwrap().len(), 8);
    assert_eq!(report["source_unchanged"], true);
    assert_eq!(report["identity_before"], report["identity_after"]);
    assert!(
        report["summary"]["output_hash_mismatches"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    for arm in requests {
        assert_eq!(
            arm.iter()
                .filter(|r| r["headers"].as_str().unwrap().starts_with("POST"))
                .count(),
            6
        );
        for r in arm {
            let h = r["headers"].as_str().unwrap();
            assert!(!h.to_ascii_lowercase().contains("authorization:"));
            if h.starts_with("POST") {
                assert!(h.starts_with("POST /v1/chat/completions HTTP/1.1"));
                assert_eq!(r["body"]["model"], "model");
                assert_eq!(r["body"]["max_tokens"], 256);
                assert_eq!(r["body"]["temperature"], 0.0);
                assert!(r["body"].get("stream").is_none());
            }
        }
    }
    assert_eq!(
        std::fs::read_to_string(f.output.join("results.jsonl"))
            .unwrap()
            .lines()
            .count(),
        8
    );
    assert!(
        std::fs::read_to_string(f.output.join("summary.md"))
            .unwrap()
            .contains("Output hash mismatches: 0")
    );
    f.directory.close().unwrap();
}
#[test]
fn suffix_actual_cli_required_source_and_later_failure_publish_honest_evidence() {
    for mode in ["wrong-source", "later-failure"] {
        let a = Peer::new("off", "");
        let b = Peer::new("suffix", mode);
        let f = Fixture::new(a.port, b.port, 0);
        let r = invoke(&f.spec(), &Cancellation::default());
        a.finish();
        b.finish();
        assert_eq!(r.process.status.unwrap().code(), Some(1));
        let report = f.report();
        assert_eq!(report["status"], "FAILED");
        assert!(!report["samples"].as_array().unwrap().is_empty());
        if mode == "later-failure" {
            assert!(report["summary"].is_null());
            assert!(!report["failed_cell"].is_null());
        } else {
            assert!(!report["summary"].is_null());
        }
        f.directory.close().unwrap();
    }
}
#[test]
fn suffix_actual_cli_causal_cancel_and_whole_deadline_retain_partial_report() {
    for cancel in [true, false] {
        let a = Peer::new("off", "");
        let b = Peer::new("suffix", "hold");
        let f = Fixture::new(a.port, b.port, 0);
        let mut input: Value = serde_json::from_slice(&std::fs::read(&f.input).unwrap()).unwrap();
        if !cancel {
            input["execution_timeout_ms"] = json!(1500);
            input["request_timeout_ms"] = json!(5000);
            std::fs::write(&f.input, serde_json::to_vec(&input).unwrap()).unwrap();
        }
        let c = Cancellation::default();
        let observer = if cancel {
            let signal = b.seen.clone();
            let stop = c.clone();
            Some(thread::spawn(move || {
                let until = Instant::now() + Duration::from_secs(4);
                while !signal.load(Ordering::SeqCst) && Instant::now() < until {
                    thread::sleep(Duration::from_millis(5));
                }
                let seen = signal.load(Ordering::SeqCst);
                stop.cancel();
                seen
            }))
        } else {
            None
        };
        let r = invoke(&f.spec(), &c);
        let seen = observer.map(|t| t.join().unwrap());
        a.finish();
        b.finish();
        if cancel {
            assert_eq!(seen, Some(true));
            assert!(c.is_cancelled());
        } else {
            assert!(!c.is_cancelled());
        }
        assert!(!r.process.status.unwrap().success());
        let report = f.report();
        assert_eq!(report["status"], "FAILED");
        assert!(report["summary"].is_null());
        assert!(!report["failed_cell"].is_null());
        f.directory.close().unwrap();
    }
}
#[test]
fn suffix_actual_cli_prior_output_and_writerless_fifo_refuse_before_http() {
    let a = Peer::new("off", "");
    let b = Peer::new("suffix", "");
    let f = Fixture::new(a.port, b.port, 0);
    std::fs::create_dir(&f.output).unwrap();
    std::fs::write(f.output.join("sentinel"), "prior").unwrap();
    let r = invoke(&f.spec(), &Cancellation::default());
    assert_eq!(r.process.status.unwrap().code(), Some(1));
    assert_eq!(std::fs::read(f.output.join("sentinel")).unwrap(), b"prior");
    let fifo = f.directory.path().join("fifo");
    let path = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    let mut spec = f.spec();
    spec.arguments[3] = Arg::Public(fifo.into_os_string());
    let r = invoke(&spec, &Cancellation::default());
    assert_eq!(r.process.status.unwrap().code(), Some(1));
    assert!(a.finish().is_empty() && b.finish().is_empty());
    f.directory.close().unwrap();
}
