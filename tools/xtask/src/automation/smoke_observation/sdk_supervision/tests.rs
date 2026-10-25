use super::*;
use crate::process::{Outcome, OutputFiles};
use std::{
    io::{Read, Write},
    net::TcpListener,
    os::unix::fs::PermissionsExt,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
fn fixture(script: &str) -> (tempfile::TempDir, child::Admitted) {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    std::fs::create_dir_all(root.join("scripts")).unwrap();
    std::fs::create_dir_all(root.join("ci/required-sdk-python")).unwrap();
    std::fs::write(
        root.join("scripts/ci-litellm-smoke.py"),
        "inert source pin only",
    )
    .unwrap();
    std::fs::write(
        root.join("ci/required-sdk-python/requirements.lock"),
        "inert lock",
    )
    .unwrap();
    let exe = root.join("inert-sdk");
    std::fs::write(&exe, format!("#!/bin/sh\n{script}\n")).unwrap();
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o700)).unwrap();
    let admitted = child::admit(
        &root,
        "litellm",
        &exe,
        "http://127.0.0.1:1/v1",
        Some("fixture-model"),
    )
    .unwrap();
    (temp, admitted)
}
#[test]
fn retained_sdk_child_literal_argv_closed_environment_and_sanitized_capture() {
    let (temp, input) = fixture(
        r#"test "$#" = 6 && test "$1" = -I && test "$3" = --base-url && test "$5" = --model && test "$6" = fixture-model || exit 12
 test -z "${HF_TOKEN+x}" || exit 13
 printf 'safe SDK observation\ntoken=fixture-private-sdk-key\n'
 exit 0"#,
    );
    let log = temp.path().join("stdout.log");
    let report = child::execute(
        &input,
        Instant::now() + Duration::from_secs(8),
        &Cancellation::default(),
        OutputFiles {
            stdout: Some(log.clone()),
            stderr: None,
        },
    )
    .unwrap();
    assert!(report.success() && report.cleanup.complete && report.stdout.line_capture_complete);
    assert_eq!(report.stdout.suppressed_lines, 1);
    let persisted = std::fs::read_to_string(log).unwrap();
    assert!(persisted.contains("safe SDK observation"));
    assert!(persisted.contains("[output line suppressed]"));
    assert!(!persisted.contains("fixture-private-sdk-key"));
    assert_eq!(
        persisted.as_bytes(),
        report.stdout.bytes_retained.as_slice()
    );
    assert!(
        !String::from_utf8(report.stdout.bytes_retained)
            .unwrap()
            .contains("fixture-private-sdk-key")
    );
    drop(input);
    temp.close().unwrap();
}
#[test]
fn retained_sdk_nonzero_deadline_and_inflight_cancel_preserve_partial_capture_and_cleanup() {
    for mode in ["failure", "deadline", "cancel"] {
        let marker = tempfile::tempdir().unwrap();
        let seen = marker.path().join("seen");
        let script = if mode == "failure" {
            "printf 'partial SDK observation\n'; exit 23".into()
        } else {
            format!(
                "printf 'partial SDK observation\n'; : > '{}'; while :; do sleep 0.02; done",
                seen.display()
            )
        };
        let (temp, input) = fixture(&script);
        let cancel = Cancellation::default();
        let token = cancel.clone();
        let deadline = Instant::now() + Duration::from_secs(if mode == "deadline" { 4 } else { 8 });
        let worker = std::thread::spawn(move || {
            let report = child::execute(&input, deadline, &token, OutputFiles::default());
            drop(input);
            report.map_err(|error| error.to_string())
        });
        let mut observed = false;
        if mode == "cancel" {
            let until = Instant::now() + Duration::from_secs(3);
            while Instant::now() < until {
                if seen.exists() {
                    observed = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            cancel.cancel();
        }
        let report = worker.join().unwrap().unwrap();
        assert!(
            report.cleanup.complete && report.stdout.line_capture_complete && !report.success()
        );
        assert!(
            String::from_utf8(report.stdout.bytes_retained)
                .unwrap()
                .contains("partial SDK observation")
        );
        assert_eq!(
            report.outcome,
            match mode {
                "deadline" => Outcome::Deadline,
                "cancel" => Outcome::Cancelled,
                _ => Outcome::Exited,
            }
        );
        if mode == "failure" {
            assert_eq!(report.status.and_then(|s| s.code()), Some(23));
        }
        if mode == "cancel" {
            assert!(observed && cancel.is_cancelled());
        }
        marker.close().unwrap();
        temp.close().unwrap();
    }
}
#[test]
fn retained_sdk_pin_drift_refuses_before_spawn_and_cleanup_reserve_is_required() {
    let (temp, input) = fixture("exit 0");
    let path = temp.path().join("scripts/ci-litellm-smoke.py");
    std::fs::write(&path, "changed source").unwrap();
    assert!(
        child::execute(
            &input,
            Instant::now() + Duration::from_secs(8),
            &Cancellation::default(),
            OutputFiles::default()
        )
        .is_err()
    );
    std::fs::write(path, "inert source pin only").unwrap();
    assert!(
        child::execute(
            &input,
            Instant::now() + Duration::from_secs(2),
            &Cancellation::default(),
            OutputFiles::default()
        )
        .is_err()
    );
    drop(input);
    temp.close().unwrap();
}
struct Peer {
    port: u16,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    seen: Arc<AtomicBool>,
}
impl Peer {
    fn new(hold: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let seen = Arc::new(AtomicBool::new(false));
        let (s, o) = (stop.clone(), seen.clone());
        let thread = std::thread::spawn(move || {
            while !s.load(Ordering::SeqCst) {
                if let Ok((mut stream, _)) = listener.accept() {
                    stream.set_nonblocking(false).unwrap();
                    stream
                        .set_read_timeout(Some(Duration::from_millis(100)))
                        .unwrap();
                    stream
                        .set_write_timeout(Some(Duration::from_millis(100)))
                        .unwrap();
                    let mut request = [0; 4096];
                    let _ = stream.read(&mut request);
                    o.store(true, Ordering::SeqCst);
                    if hold {
                        while !s.load(Ordering::SeqCst) {
                            std::thread::sleep(Duration::from_millis(5));
                        }
                    } else {
                        let body = r#"{"data":[{"id":"fixture-model"}]}"#;
                        let _ = write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                    }
                } else {
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        });
        Self {
            port,
            stop,
            thread: Some(thread),
            seen,
        }
    }
}
impl Drop for Peer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}
#[test]
fn retained_sdk_owned_readiness_success_held_deadline_and_causal_cancel() {
    for mode in ["success", "deadline", "cancel"] {
        let peer = Peer::new(mode != "success");
        let base = format!("http://127.0.0.1:{}/v1", peer.port);
        let cancel = Cancellation::default();
        let token = cancel.clone();
        let worker = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime
                .block_on(ready(
                    &base,
                    Instant::now()
                        + Duration::from_millis(if mode == "deadline" { 250 } else { 3000 }),
                    &token,
                ))
                .map_err(|e| e.to_string())
        });
        let mut observed = false;
        if mode == "cancel" {
            let until = Instant::now() + Duration::from_secs(2);
            while Instant::now() < until {
                if peer.seen.load(Ordering::SeqCst) {
                    observed = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            cancel.cancel();
        }
        let result = worker.join().unwrap();
        if mode == "success" {
            assert_eq!(result.unwrap(), "fixture-model");
        } else {
            assert!(result.is_err());
        }
        if mode == "cancel" {
            assert!(observed && cancel.is_cancelled());
        }
        drop(peer);
    }
}

#[test]
fn retained_sdk_dead_owned_product_refuses_live_endpoint_readiness() {
    let peer = Peer::new(false);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let base = format!("http://127.0.0.1:{}/v1", peer.port);
    assert_eq!(
        runtime
            .block_on(ready(
                &base,
                Instant::now() + Duration::from_secs(3),
                &Cancellation::default()
            ))
            .unwrap(),
        "fixture-model"
    );
    let source = std::fs::read_to_string(
        crate::repo_consistency::repo_root()
            .unwrap()
            .join("scripts/ci-compat-smoke.sh"),
    )
    .unwrap();
    let function = source
        .split("sdk_owned_model_ready() {")
        .nth(1)
        .unwrap()
        .split("\n}\nsdk_owned_model_ready")
        .next()
        .unwrap();
    assert_eq!(function.matches("kill -0 \"$MESH_PID\"").count(), 2);
    let temp = tempfile::tempdir().unwrap();
    let code = format!(
        "fixture_readiness() {{ printf 'wrongly queried live endpoint\\n'; return 0; }}\nautomation=(fixture_readiness)\nBASE_URL='{base}'\nMAX_WAIT=3\n/bin/sh -c 'exit 0' &\nMESH_PID=$!\nwait \"$MESH_PID\"\nsdk_owned_model_ready() {{{function}\n}}\nsdk_owned_model_ready\n"
    );
    let spec = crate::process::ProcessSpec {
        executable: "/bin/bash".into(),
        arguments: vec![
            crate::process::Value::Public("-c".into()),
            crate::process::Value::Public(code.into()),
        ],
        cwd: temp.path().canonicalize().unwrap(),
        environment: std::collections::BTreeMap::new(),
    };
    let report = crate::process::supervise(
        &spec,
        &crate::process::Limits {
            execution: Duration::from_secs(3),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 4096,
            readiness: crate::process::Readiness::None,
            completion: crate::process::Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert_eq!(report.status.and_then(|s| s.code()), Some(1));
    assert!(
        report.cleanup.complete
            && report.stdout.line_capture_complete
            && report.stderr.line_capture_complete
    );
    assert!(report.stdout.bytes_retained.is_empty());
    drop(peer);
    temp.close().unwrap();
}
