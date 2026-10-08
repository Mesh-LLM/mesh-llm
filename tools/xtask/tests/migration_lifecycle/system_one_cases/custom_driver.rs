//! Local supplied case drivers are supervised; these fixtures do not qualify inference.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt as _,
    path::Path,
    time::{Duration, Instant},
};
fn executable(path: &Path, body: &str) {
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
fn run(
    root: &Path,
    driver: &Path,
    budget: &str,
    cancellation: &Cancellation,
) -> process::RawProcessReport {
    let args = vec![
        "automation".to_owned(),
        "system-one-cases".into(),
        "--driver-executable".into(),
        driver.to_str().unwrap().into(),
        "--driver-timeout".into(),
        budget.into(),
        "--base-url".into(),
        "http://127.0.0.1:9337".into(),
        "--model".into(),
        "literal model with spaces".into(),
        "--alias".into(),
        "literal alias".into(),
        "--mode".into(),
        "contract".into(),
        "--timeout".into(),
        "900".into(),
        "--json-out".into(),
        root.join("report.json").to_str().unwrap().into(),
    ];
    run_arguments(root, args, cancellation)
}
fn run_arguments(
    root: &Path,
    args: Vec<String>,
    cancellation: &Cancellation,
) -> process::RawProcessReport {
    let report = process::supervise_raw(
        &ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            cwd: root.into(),
            arguments: args.into_iter().map(|a| Value::Public(a.into())).collect(),
            environment: BTreeMap::from([("PATH".into(), Value::Public("/usr/bin:/bin".into()))]),
        },
        &Limits {
            execution: Duration::from_secs(8),
            graceful_shutdown: Duration::from_secs(4),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancellation,
        RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert!(report.process.failure.is_none() && report.process.cleanup.failure.is_none());
    assert!(
        report.process.cleanup.complete
            && !report.process.cleanup.forced
            && !report.process.cleanup.graceful_signal_failed
    );
    for (stream, raw) in [
        (&report.process.stdout, report.stdout.as_ref().unwrap()),
        (&report.process.stderr, report.stderr.as_ref().unwrap()),
    ] {
        assert!(stream.line_capture_complete && !stream.truncated);
        assert_eq!(stream.oversized_lines, 0);
        assert_eq!(
            stream.bytes_seen,
            u64::try_from(raw.as_bytes().len()).unwrap()
        );
    }
    report
}
#[test]
fn actual_custom_driver_forwards_literal_six_pairs_and_consumes_exit_without_interpreter() {
    for status in [0, 23] {
        let scratch = tempfile::tempdir().unwrap();
        let root = scratch.path().canonicalize().unwrap();
        let driver = root.join("driver with spaces");
        executable(
            &driver,
            &format!(
                "#!/bin/bash\nset -euo pipefail\n[[ $# == 12 ]]\nprintf '%s\\0' \"$@\" > arguments\n[[ \"$9\" == --timeout && \"${{10}}\" == 900 && \"${{11}}\" == --json-out ]]\nprintf '{{\"status\":\"fixture\"}}\\n' > \"${{12}}\"\nprintf 'custom driver observed\\n'\nprintf 'token=private-driver-fixture-secret\\n'\n\nexit {status}\n"
            ),
        );
        let result = run(&root, &driver, "5", &Cancellation::default());
        assert_eq!(result.process.outcome, Outcome::Exited);
        assert_eq!(result.process.status.unwrap().code(), Some(status));
        let stdout = String::from_utf8_lossy(result.stdout.as_ref().unwrap().as_bytes());
        assert!(stdout.contains("custom driver observed"));
        assert!(!stdout.contains("private-driver-fixture-secret"));
        assert!(
            !String::from_utf8_lossy(result.stderr.as_ref().unwrap().as_bytes())
                .contains("private-driver-fixture-secret")
        );
        let expected = [
            "--base-url",
            "http://127.0.0.1:9337",
            "--model",
            "literal model with spaces",
            "--alias",
            "literal alias",
            "--mode",
            "contract",
            "--timeout",
            "900",
            "--json-out",
            root.join("report.json").to_str().unwrap(),
        ]
        .join("\0")
            + "\0";
        assert_eq!(
            fs::read(root.join("arguments")).unwrap(),
            expected.as_bytes()
        );
        assert!(fs::read(root.join("report.json")).is_ok());
        scratch.close().unwrap();
    }
    let scratch = tempfile::tempdir().unwrap();
    let root = scratch.path().canonicalize().unwrap();
    let driver = root.join("inline driver");
    executable(&driver, "#!/bin/bash\nprintf '%s\\0' \"$@\" > arguments\n");
    let report_path = root.join("report with spaces");
    let args = vec![
        "automation".into(),
        "system-one-cases".into(),
        format!("--driver-executable={}", driver.display()),
        "--driver-timeout=5".into(),
        "--base-url=http://127.0.0.1:9337/".into(),
        "--model=--driver-timeout".into(),
        "--alias".into(),
        "literal ; $() alias".into(),
        "--mode=contract".into(),
        "--timeout=0900.0".into(),
        format!("--json-out={}", report_path.display()),
    ];
    let result = run_arguments(&root, args, &Cancellation::default());
    assert!(result.process.status.unwrap().success());
    let expected = [
        "--base-url",
        "http://127.0.0.1:9337/",
        "--model",
        "--driver-timeout",
        "--alias",
        "literal ; $() alias",
        "--mode",
        "contract",
        "--timeout",
        "0900.0",
        "--json-out",
        report_path.to_str().unwrap(),
    ]
    .join("\0")
        + "\0";
    assert_eq!(
        fs::read(root.join("arguments")).unwrap(),
        expected.as_bytes()
    );
    scratch.close().unwrap();
}
#[test]
fn actual_custom_driver_refuses_python_relative_nonexecutables_and_special_paths_before_launch() {
    let scratch = tempfile::tempdir().unwrap();
    let root = scratch.path().canonicalize().unwrap();
    let native = root.join("native");
    executable(&native, "#!/bin/bash\nprintf bad > launched\n");
    let py = root.join("old.py");
    fs::copy(&native, &py).unwrap();
    let noexec = root.join("noexec");
    fs::write(&noexec, b"inert").unwrap();
    let link = root.join("link");
    std::os::unix::fs::symlink(&native, &link).unwrap();
    let fifo = root.join("fifo");
    let c = std::ffi::CString::new(fifo.to_str().unwrap()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
    for driver in [&py, Path::new("native"), &noexec, &root, &link, &fifo] {
        let result = run(&root, driver, "5", &Cancellation::default());
        assert_eq!(result.process.status.unwrap().code(), Some(2));
        assert!(!root.join("launched").exists());
        assert!(!root.join("report.json").exists());
    }
    scratch.close().unwrap();
}
#[test]
fn actual_custom_driver_deadline_and_causal_cancel_reap_owned_sleep_without_success() {
    for cancelled in [false, true] {
        let scratch = tempfile::tempdir().unwrap();
        let root = scratch.path().canonicalize().unwrap();
        let driver = root.join("held");
        executable(
            &driver,
            "#!/bin/bash\nprintf started > started\nexec /bin/sleep 60\n",
        );
        let cancellation = Cancellation::default();
        let token = cancellation.clone();
        let marker = root.join("started");
        let trigger = cancelled.then(|| {
            std::thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_secs(5);
                while !marker.exists() && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(10));
                }
                let seen = marker.exists();
                token.cancel();
                seen
            })
        });
        let result = run(
            &root,
            &driver,
            if cancelled { "5" } else { "0.2" },
            &cancellation,
        );
        if let Some(trigger) = trigger {
            let seen = trigger.join().unwrap();
            assert!(seen);
            assert!(cancellation.is_cancelled());
            assert_eq!(result.process.outcome, Outcome::Cancelled);
        } else {
            assert!(!cancellation.is_cancelled());
            assert_eq!(result.process.outcome, Outcome::Exited);
            assert_eq!(result.process.status.unwrap().code(), Some(2));
        }
        assert!(root.join("started").exists());
        assert!(!root.join("report.json").exists());
        scratch.close().unwrap();
    }
}
