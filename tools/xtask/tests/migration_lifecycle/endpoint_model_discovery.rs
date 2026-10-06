//! Actual CLI and thin wrapper fixtures against an owned local HTTP peer.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::TcpListener,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
struct Peer {
    base: String,
    stop: Arc<AtomicBool>,
    seen: Arc<AtomicBool>,
    headers: Arc<Mutex<String>>,
    task: Option<thread::JoinHandle<()>>,
}
impl Peer {
    fn new(status: u16, body: Vec<u8>, hold: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!(
            "http://127.0.0.1:{}/v1",
            listener.local_addr().unwrap().port()
        );
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let seen = Arc::new(AtomicBool::new(false));
        let observed = seen.clone();
        let headers = Arc::new(Mutex::new(String::new()));
        let captured = headers.clone();
        let task = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(12);
            while !stopped.load(Ordering::SeqCst) && Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).unwrap();
                        stream
                            .set_read_timeout(Some(Duration::from_secs(2)))
                            .unwrap();
                        stream
                            .set_write_timeout(Some(Duration::from_secs(2)))
                            .unwrap();
                        let mut bytes = Vec::new();
                        let mut block = [0; 1024];
                        while bytes.len() < 8192 && !bytes.windows(4).any(|b| b == b"\r\n\r\n") {
                            match stream.read(&mut block) {
                                Ok(0) | Err(_) => break,
                                Ok(n) => bytes.extend_from_slice(&block[..n]),
                            }
                        }
                        *captured.lock().unwrap() = String::from_utf8_lossy(&bytes).into_owned();
                        observed.store(true, Ordering::SeqCst);
                        if hold {
                            while !stopped.load(Ordering::SeqCst) && Instant::now() < deadline {
                                thread::sleep(Duration::from_millis(5));
                            }
                        } else {
                            let header = format!(
                                "HTTP/1.1 {status} fixture\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                                body.len()
                            );
                            let _ = stream.write_all(header.as_bytes());
                            let _ = stream.write_all(&body);
                        }
                        break;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5))
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            base,
            stop,
            seen,
            headers,
            task: Some(task),
        }
    }
    fn close(mut self) -> String {
        self.stop.store(true, Ordering::SeqCst);
        self.task.take().unwrap().join().unwrap();
        self.headers.lock().unwrap().clone()
    }
}
impl Drop for Peer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(task) = self.task.take() {
            task.join().unwrap();
        }
    }
}
fn invoke(spec: &ProcessSpec, cancellation: &Cancellation) -> process::RawProcessReport {
    let report = process::supervise_raw(
        spec,
        &Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(2),
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
    let p = &report.process;
    assert!(
        p.failure.is_none()
            && p.cleanup.complete
            && !p.cleanup.forced
            && !p.cleanup.graceful_signal_failed
            && p.cleanup.failure.is_none()
    );
    for (stream, bytes) in [
        (&p.stdout, report.stdout.as_ref().unwrap()),
        (&p.stderr, report.stderr.as_ref().unwrap()),
    ] {
        assert!(stream.line_capture_complete && !stream.truncated);
        assert_eq!(stream.bytes_seen, bytes.as_bytes().len() as u64);
        assert!(!String::from_utf8_lossy(bytes.as_bytes()).contains("private-fixture-key"));
    }
    report
}
fn spec(root: &std::path::Path, base: &str, seconds: &str) -> ProcessSpec {
    ProcessSpec {
        executable: PathBuf::from(env!("CARGO_BIN_EXE_xtask")),
        arguments: [
            "automation",
            "endpoint-model-discovery",
            "--base-url",
            base,
            "--timeout-secs",
            seconds,
        ]
        .into_iter()
        .map(|s| Value::Public(s.into()))
        .collect(),
        cwd: root.to_owned(),
        environment: BTreeMap::from([(
            "API_KEY".into(),
            Value::Secret("private-fixture-key".into()),
        )]),
    }
}
#[test]
fn discovery_actual_cli_authenticates_and_selects_first_model_refusing_status_shape_and_size() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let first = br#"{"data":[{"id":"first"},{"id":"second"}]}"#.to_vec();
    let peer = Peer::new(200, first.clone(), false);
    let report = invoke(&spec(&root, &peer.base, "3"), &Cancellation::default());
    let headers = peer.close();
    assert_eq!(report.process.outcome, Outcome::Exited);
    assert!(report.process.status.unwrap().success());
    assert_eq!(report.stdout.unwrap().as_bytes(), b"first\n");
    assert!(headers.starts_with("GET /v1/models HTTP/1.1"));
    assert!(
        headers
            .to_ascii_lowercase()
            .contains("authorization: bearer private-fixture-key")
    );
    for (status, body) in [
        (302, first.clone()),
        (503, first),
        (200, b"{\"data\":[]}".to_vec()),
        (200, b"{\"data\":[{\"id\":5}]}".to_vec()),
        (200, vec![b'x'; 1024 * 1024 + 1]),
    ] {
        let peer = Peer::new(status, body, false);
        let report = invoke(&spec(&root, &peer.base, "3"), &Cancellation::default());
        peer.close();
        assert!(!report.process.status.unwrap().success());
        assert!(report.stdout.unwrap().as_bytes().is_empty());
    }
    // URL aliases must reach the numeric HTTP path with canonical Host bytes.
    for alias in ["127.1", "2130706433"] {
        let peer = Peer::new(200, br#"{"data":[{"id":"alias-first"}]}"#.to_vec(), false);
        let base = peer.base.replace("127.0.0.1", alias);
        let report = invoke(&spec(&root, &base, "3"), &Cancellation::default());
        let headers = peer.close();
        assert!(report.process.status.unwrap().success());
        assert_eq!(report.stdout.unwrap().as_bytes(), b"alias-first\n");
        assert!(headers.to_ascii_lowercase().contains("host: 127.0.0.1:"));
    }
    directory.close().unwrap();
}
#[test]
fn discovery_actual_local_dns_uses_owned_private_curl_authorization() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let peer = Peer::new(200, br#"{"data":[{"id":"dns-first"}]}"#.to_vec(), false);
    let base = peer.base.replace("127.0.0.1", "localhost");
    let mut command = spec(&root, &base, "8");
    command.environment.insert(
        "PATH".into(),
        Value::Public("/usr/bin:/bin:/usr/local/bin".into()),
    );
    let report = invoke(&command, &Cancellation::default());
    let headers = peer.close();
    assert!(report.process.status.unwrap().success());
    assert_eq!(report.stdout.unwrap().as_bytes(), b"dns-first\n");
    assert!(
        headers
            .to_ascii_lowercase()
            .contains("authorization: bearer private-fixture-key")
    );
    directory.close().unwrap();
}

#[test]
fn discovery_actual_held_get_refuses_deadline_and_interrupt_without_publishing_identity() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    for alias in ["127.0.0.1", "127.1", "2130706433"] {
        let peer = Peer::new(200, vec![], true);
        let base = peer.base.replace("127.0.0.1", alias);
        let report = invoke(&spec(&root, &base, "1"), &Cancellation::default());
        let headers = peer.close();
        assert!(headers.to_ascii_lowercase().contains("host: 127.0.0.1:"));
        assert!(!report.process.status.unwrap().success());
        assert!(report.stdout.unwrap().as_bytes().is_empty());
        let peer = Peer::new(200, vec![], true);
        let seen = peer.seen.clone();
        let base = peer.base.replace("127.0.0.1", alias);
        let token = Cancellation::default();
        let trigger = token.clone();
        let cancel = thread::spawn(move || {
            let until = Instant::now() + Duration::from_secs(5);
            while !seen.load(Ordering::SeqCst) && Instant::now() < until {
                thread::sleep(Duration::from_millis(5));
            }
            let observed = seen.load(Ordering::SeqCst);
            trigger.cancel();
            observed
        });
        let report = invoke(&spec(&root, &base, "8"), &token);
        let observed = cancel.join().unwrap();
        let headers = peer.close();
        assert!(observed);
        assert!(headers.to_ascii_lowercase().contains("host: 127.0.0.1:"));
        assert!(!report.process.status.unwrap().success());
        assert!(report.stdout.unwrap().as_bytes().is_empty());
    }
    directory.close().unwrap();
}
fn executable(path: &std::path::Path, source: &str) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, source).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
}
#[test]
fn benchy_wrapper_actual_native_discovery_preserves_defaults_overrides_flags_and_redacts_display() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let uvx = root.join("uvx");
    let output = root.join("argv");
    executable(
        &uvx,
        "#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$FIXTURE_ARGV\"\n",
    );
    let peer = Peer::new(
        200,
        br#"{"data":[{"id":"first"},{"id":"second"}]}"#.to_vec(),
        false,
    );
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/run-llama-benchy-openai.sh")
        .canonicalize()
        .unwrap();
    let mut environment = BTreeMap::from([
        ("PATH".into(), Value::Public(root.as_os_str().into())),
        (
            "FIXTURE_ARGV".into(),
            Value::Public(output.as_os_str().into()),
        ),
        (
            "MESH_LLM_AUTOMATION_BIN".into(),
            Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
        ),
        (
            "API_KEY".into(),
            Value::Secret("private-fixture-key".into()),
        ),
        ("BASE_URL".into(), Value::Public(peer.base.as_str().into())),
        ("PP".into(), Value::Public("128 512".into())),
        ("TG".into(), Value::Public("16 32".into())),
        (
            "TOKENIZER".into(),
            Value::Public("declared-tokenizer".into()),
        ),
        ("SAVE_RESULT".into(), Value::Public("owned-result".into())),
        ("NO_CACHE".into(), Value::Public("1".into())),
    ]);
    let mut command = ProcessSpec {
        executable: PathBuf::from("/bin/bash"),
        arguments: vec![Value::Public(script.as_os_str().into())],
        cwd: root.clone(),
        environment: environment.clone(),
    };
    let report = invoke(&command, &Cancellation::default());
    peer.close();
    assert!(report.process.status.unwrap().success());
    let argv = std::fs::read_to_string(&output).unwrap();
    assert!(argv.contains("--model\nfirst\n--served-model-name\nfirst\n"));
    assert!(argv.contains("--pp\n128\n512\n--tg\n16\n32\n"));
    assert!(argv.contains("--tokenizer\ndeclared-tokenizer\n"));
    assert!(argv.contains("--skip-coherence\n") && argv.contains("--no-cache\n"));
    assert!(argv.contains("--api-key\nprivate-fixture-key\n")); // External CLI contract, explicitly not private argv custody.
    environment.insert("MODEL".into(), Value::Public("override".into()));
    environment.insert("SERVED_MODEL_NAME".into(), Value::Public("alias".into()));
    environment.insert(
        "BASE_URL".into(),
        Value::Public("http://127.0.0.1:1/v1".into()),
    );
    command.environment = environment;
    let report = invoke(&command, &Cancellation::default());
    assert!(report.process.status.unwrap().success());
    assert!(
        std::fs::read_to_string(&output)
            .unwrap()
            .contains("--model\noverride\n--served-model-name\nalias\n")
    );
    command
        .environment
        .insert("BASE_URL".into(), Value::Public("--api-key".into()));
    command
        .environment
        .insert("MODEL".into(), Value::Public("--api-key".into()));
    let report = invoke(&command, &Cancellation::default());
    assert!(report.process.status.unwrap().success());
    let argv = std::fs::read_to_string(&output).unwrap();
    assert!(argv.contains("--base-url\n--api-key\n--api-key\nprivate-fixture-key\n"));
    assert!(argv.contains("--model\n--api-key\n--served-model-name\nalias\n"));
    directory.close().unwrap();
}
