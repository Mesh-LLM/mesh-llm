use super::*;
use crate::process::{RawCaptureOptions, RawProcessReport, Value};
use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::Path, time::Duration};

fn spec(root: &Path, script: &str) -> ProcessSpec {
    ProcessSpec {
        executable: "/bin/sh".into(),
        arguments: vec![Value::Public("-c".into()), Value::Public(script.into())],
        cwd: root.into(),
        environment: BTreeMap::from([(
            "FIXTURE_PRIVATE".into(),
            Value::Secret("private-fixture-value".into()),
        )]),
    }
}
fn limits() -> Limits {
    Limits {
        execution: Duration::from_secs(5),
        graceful_shutdown: Duration::from_millis(500),
        forced_shutdown: Duration::from_millis(500),
        retained_bytes_per_stream: 4096,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}
fn files(root: &Path) -> OutputFiles {
    OutputFiles {
        stdout: Some(root.join("stdout.log")),
        stderr: Some(root.join("stderr.log")),
    }
}
fn raw(limit: usize) -> RawCaptureOptions {
    RawCaptureOptions {
        stdout: NonZeroUsize::new(limit),
        stderr: NonZeroUsize::new(limit),
    }
}
fn clean(report: &RawProcessReport, root: &Path) {
    let p = &report.process;
    assert!(p.failure.is_none(), "{:?}", p.failure);
    assert!(p.cleanup.complete);
    assert!(!p.cleanup.forced);
    assert!(!p.cleanup.graceful_signal_failed);
    assert!(p.cleanup.failure.is_none());
    for (stream, bytes, name) in [
        (&p.stdout, report.stdout.as_ref().unwrap(), "stdout.log"),
        (&p.stderr, report.stderr.as_ref().unwrap(), "stderr.log"),
    ] {
        assert!(stream.line_capture_complete && !stream.truncated);
        assert_eq!(bytes.as_bytes().len() as u64, stream.bytes_seen);
        assert_eq!(fs::read(root.join(name)).unwrap(), stream.bytes_retained);
        assert!(!String::from_utf8_lossy(&stream.bytes_retained).contains("private-fixture-value"));
    }
}

#[test]
fn raw_files_capture_keeps_exact_payload_and_persists_only_sanitized_diagnostics() {
    let root = tempfile::tempdir().unwrap();
    let report = supervise_raw_with_files(
        &spec(root.path(), "printf 'ordinary-out\n'; printf 'token=%s\n' \"$FIXTURE_PRIVATE\"; printf 'ordinary-err\n' >&2; printf '%s\n' \"$FIXTURE_PRIVATE\" >&2"),
        &limits(), &Cancellation::default(), files(root.path()), raw(4096),
    ).unwrap();
    clean(&report, root.path());
    assert_eq!(report.process.outcome, Outcome::Exited);
    assert!(report.process.status.unwrap().success());
    assert_eq!(
        report.stdout.unwrap().as_bytes(),
        b"ordinary-out\ntoken=private-fixture-value\n"
    );
    assert_eq!(
        report.stderr.unwrap().as_bytes(),
        b"ordinary-err\nprivate-fixture-value\n"
    );
    assert!(report.process.stdout.suppressed_lines > 0);
    root.close().unwrap();
}

#[test]
fn raw_files_capture_preserves_nonzero_status_and_complete_clean_streams() {
    let root = tempfile::tempdir().unwrap();
    let report = supervise_raw_with_files(
        &spec(
            root.path(),
            "printf 'out-before-failure\n'; printf 'error-before-failure\n' >&2; exit 7",
        ),
        &limits(),
        &Cancellation::default(),
        files(root.path()),
        raw(4096),
    )
    .unwrap();
    clean(&report, root.path());
    assert_eq!(report.process.outcome, Outcome::Exited);
    assert_eq!(report.process.status.unwrap().code(), Some(7));
    assert_eq!(report.stdout.unwrap().as_bytes(), b"out-before-failure\n");
    assert_eq!(report.stderr.unwrap().as_bytes(), b"error-before-failure\n");
    root.close().unwrap();
}

#[test]
fn raw_files_capture_causal_cancellation_joins_child_before_stream_assertions() {
    let root = tempfile::tempdir().unwrap();
    let cancellation = Cancellation::default();
    let observer_cancel = cancellation.clone();
    let marker = root.path().join("stdout.log");
    let observer = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut seen = false;
        while Instant::now() < deadline {
            if fs::read(&marker).is_ok_and(|bytes| bytes == b"READY\n") {
                seen = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        observer_cancel.cancel();
        seen
    });
    let result = supervise_raw_with_files(
        &spec(
            root.path(),
            "trap 'exit 0' TERM; printf 'READY\n'; printf 'token=%s\n' \"$FIXTURE_PRIVATE\" >&2; while :; do :; done",
        ),
        &limits(),
        &cancellation,
        files(root.path()),
        raw(4096),
    );
    // Always join the marker observer after the synchronous owner returned;
    // a missing marker cannot panic before cancellation and child cleanup.
    let seen = observer.join().unwrap();
    let report = result.unwrap();
    clean(&report, root.path());
    assert!(seen && cancellation.is_cancelled());
    assert_eq!(report.process.outcome, Outcome::Cancelled);
    assert_eq!(report.stdout.unwrap().as_bytes(), b"READY\n");
    assert_eq!(
        report.stderr.unwrap().as_bytes(),
        b"token=private-fixture-value\n"
    );
    root.close().unwrap();
}
