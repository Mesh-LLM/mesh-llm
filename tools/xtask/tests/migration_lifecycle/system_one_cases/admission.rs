use super::server;
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{collections::BTreeMap, fs, path::Path, time::Duration};
fn run(args: Vec<String>, cwd: &Path) -> process::RawProcessReport {
    let report = process::supervise_raw(
        &ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            arguments: args
                .into_iter()
                .map(|arg| Value::Public(arg.into()))
                .collect(),
            cwd: cwd.into(),
            environment: BTreeMap::new(),
        },
        &Limits {
            execution: Duration::from_secs(5),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert!(
        report.process.cleanup.complete && report.process.failure.is_none(),
        "{:?}",
        report.process
    );
    report
}
fn args(url: &str, output: &Path) -> Vec<String> {
    vec![
        "automation".into(),
        "system-one-cases".into(),
        "--base-url".into(),
        url.into(),
        "--model".into(),
        "fixture-model".into(),
        "--mode".into(),
        "contract".into(),
        "--json-out".into(),
        output.to_str().unwrap().into(),
    ]
}
#[test]
fn actual_cli_input_admission_rejects_bad_endpoint_deadline_duplicates_before_http_or_report() {
    let scratch = tempfile::tempdir().unwrap();
    let output = scratch.path().join("old.json");
    fs::write(&output, "old").unwrap();
    let server = server::Server::new("contract", "");
    let mut variants = Vec::new();
    for bad in [
        "https://127.0.0.1:1",
        "/",
        "http://user@127.0.0.1:1",
        "http://127.0.0.1:1?credential=x",
        "http://127.0.0.1:1#x",
        "http://127.0.0.1:99999",
        "http://127.0.0.1:",
        "http://127.0.0.1:+80",
        "http://[::1]:99999",
        "http://[invalid]:80",
    ] {
        variants.push(args(bad, &output));
    }
    for timeout in ["0", "-1", "NaN", "inf", "3601"] {
        let mut a = args(&server.url, &output);
        a.extend(["--timeout".into(), timeout.into()]);
        variants.push(a);
    }
    let mut duplicate = args(&server.url, &output);
    duplicate.extend(["--model".into(), "another".into()]);
    variants.push(duplicate);
    for a in variants {
        let report = run(a, scratch.path());
        assert_eq!(report.process.status.unwrap().code(), Some(2));
        assert!(report.stdout.unwrap().as_bytes().is_empty());
        assert_eq!(fs::read_to_string(&output).unwrap(), "old");
    }
    assert!(server.calls.lock().unwrap().is_empty());
    server.finish();
}
#[test]
fn actual_report_directory_fifo_symlink_and_write_failure_are_bounded_nonpassing_outcomes() {
    let scratch = tempfile::tempdir().unwrap();
    let server = server::Server::new("contract", "");
    let old = scratch.path().join("old.json");
    fs::write(&old, "old").unwrap();
    let link = scratch.path().join("link");
    std::os::unix::fs::symlink(&old, &link).unwrap();
    let fifo = scratch.path().join("fifo");
    let path = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
    // SAFETY: owned fixture path is a NUL-terminated string and mode has no special bits.
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    for output in [scratch.path(), fifo.as_path(), link.as_path()] {
        let report = run(args(&server.url, output), scratch.path());
        assert_eq!(report.process.status.unwrap().code(), Some(2));
        assert!(!report.process.success());
    }
    assert!(server.calls.lock().unwrap().is_empty());
    assert_eq!(fs::read_to_string(old).unwrap(), "old");
    let missing = scratch.path().join("absent/parent/report.json");
    let report = run(args(&server.url, &missing), scratch.path());
    assert_eq!(report.process.status.unwrap().code(), Some(2));
    assert!(!missing.exists());
    assert_eq!(server.calls.lock().unwrap().len(), 14);
    server.finish();
    assert!(
        String::from_utf8_lossy(report.stderr.unwrap().as_bytes())
            .contains("report destination failed")
    );
}
