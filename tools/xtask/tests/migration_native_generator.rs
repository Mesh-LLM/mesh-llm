#[path = "../src/process/mod.rs"]
pub mod process;
use process::*;
use std::collections::BTreeMap;
use std::time::Duration;

#[cfg(unix)]
#[path = "migration_process/raw_capture_tree.rs"]
mod raw_tree;

#[cfg(unix)]
fn raw_tree_report(budget: &Limits, cancellation: &Cancellation) -> RawProcessReport {
    let root = tempfile::tempdir().unwrap();
    let profile = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned();
    let executable = profile.join("examples").join(format!(
        "migration_process_driver{}",
        std::env::consts::EXE_SUFFIX
    ));
    let mut sentinel = std::process::Command::new(&executable)
        .env("MIGRATION_PROCESS_DRIVER", "1")
        .env("MIGRATION_PROCESS_MODE", "sentinel")
        .env("MIGRATION_PROCESS_ROOT", root.path())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let spec = ProcessSpec {
        executable,
        arguments: Vec::new(),
        cwd: root.path().to_owned(),
        environment: BTreeMap::from([
            ("MIGRATION_PROCESS_DRIVER".into(), Value::Public("1".into())),
            (
                "MIGRATION_PROCESS_MODE".into(),
                Value::Public("tree-hang".into()),
            ),
            (
                "MIGRATION_PROCESS_ROOT".into(),
                Value::Public(root.path().into()),
            ),
        ]),
    };
    let report = supervise_raw(&spec, budget, cancellation, options(64)).unwrap();
    let sentinel_alive = sentinel.try_wait().unwrap().is_none();
    sentinel.kill().unwrap();
    sentinel.wait().unwrap();
    assert!(sentinel_alive);
    assert!(report.process.cleanup.complete);
    assert_eq!(report.stdout.as_ref().unwrap().as_bytes(), b"TREE_READY\n");
    for mode in ["tree-hang", "branch", "leaf"] {
        let pid = std::fs::read_to_string(root.path().join(format!("{mode}.pid"))).unwrap();
        let output = std::process::Command::new("/bin/kill")
            .args(["-0", pid.trim()])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{mode} survived");
    }
    report
}
fn spec(root: &std::path::Path, mode: &str) -> ProcessSpec {
    let executable = std::env::current_exe().unwrap();
    let profile = executable.parent().unwrap().parent().unwrap();
    let fixture = profile.join("examples").join(format!(
        "migration_raw_capture_fixture{}",
        std::env::consts::EXE_SUFFIX
    ));
    assert!(
        fixture.is_file(),
        "build the migration_raw_capture_fixture example first"
    );
    ProcessSpec {
        executable: fixture,
        arguments: vec![Value::Public(mode.into())],
        cwd: root.to_owned(),
        environment: ["SYSTEMROOT", "WINDIR"]
            .into_iter()
            .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
            .collect::<BTreeMap<_, _>>(),
    }
}
fn limits() -> Limits {
    Limits {
        execution: Duration::from_secs(5),
        graceful_shutdown: Duration::from_millis(100),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 64,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}
fn options(cap: usize) -> RawCaptureOptions {
    RawCaptureOptions {
        stdout: Some(std::num::NonZeroUsize::new(cap).unwrap()),
        stderr: None,
    }
}
#[test]
fn exact_payload_and_sanitized_diagnostics_when_child_emits_arbitrary_bytes() {
    let root = tempfile::tempdir().unwrap();
    let mut expected = b"token\0\xff\r\npassword secret authorization invite\n".to_vec();
    expected.extend(vec![b'x'; 9000]);
    expected.extend_from_slice(b"\r\nend\0");
    let report = supervise_raw(
        &spec(root.path(), "raw"),
        &limits(),
        &Cancellation::default(),
        options(expected.len()),
    )
    .unwrap();
    assert!(report.process.success(), "{:?}", report.process);
    assert_eq!(report.stdout.unwrap().as_bytes(), expected);
    assert!(report.stderr.is_none());
    assert_eq!(
        report.process.stderr.bytes_retained,
        b"[output line suppressed]\nordinary diagnostic\r\n"
    );
    assert!(report.process.stdout.truncated);
    assert!(!format!("{:?}", report.process).contains("password secret"));
}
#[test]
fn typed_overflow_and_cleanup_when_child_exceeds_raw_cap() {
    let root = tempfile::tempdir().unwrap();
    let report = supervise_raw(
        &spec(root.path(), "raw"),
        &limits(),
        &Cancellation::default(),
        options(1),
    )
    .unwrap();
    assert!(matches!(
        report.process.failure,
        Some(Failure::RawCaptureOverflow {
            stream: Stream::Stdout,
            limit: 1
        })
    ));
    assert!(report.stdout.is_none());
    assert!(report.process.cleanup.complete);
    assert!(report.process.status.is_some());
}
#[test]
fn timeout_still_cleans_owned_child_when_raw_capture_is_enabled() {
    let root = tempfile::tempdir().unwrap();
    let mut budget = limits();
    budget.execution = Duration::from_millis(100);
    let report = supervise_raw(
        &spec(root.path(), "held"),
        &budget,
        &Cancellation::default(),
        options(64),
    )
    .unwrap();
    assert_eq!(report.process.outcome, Outcome::Deadline);
    assert!(report.process.cleanup.complete);
    assert!(report.process.status.is_some());
}
#[test]
fn cancellation_still_cleans_owned_child_when_raw_capture_is_enabled() {
    static CANCELLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    fn cancel_on_line(_: ObservedLine<'_>) -> bool {
        CANCELLED.store(true, std::sync::atomic::Ordering::SeqCst);
        false
    }
    CANCELLED.store(false, std::sync::atomic::Ordering::SeqCst);
    let root = tempfile::tempdir().unwrap();
    let cancellation = Cancellation::from_static(&CANCELLED);
    let mut budget = limits();
    budget.readiness = Readiness::ObservedLines {
        deadline: Duration::from_secs(2),
        matcher: cancel_on_line,
    };
    budget.completion = Completion::StopAfterReady;
    let report = supervise_raw(
        &spec(root.path(), "held"),
        &budget,
        &cancellation,
        options(64),
    )
    .unwrap();
    assert_eq!(report.process.outcome, Outcome::Cancelled);
    assert!(report.process.cleanup.complete);
}
#[test]
fn raw_capture_exceeds_diagnostic_ceiling_without_changing_bytes() {
    let root = tempfile::tempdir().unwrap();
    let length = 4097 * 4096;
    let report = supervise_raw(
        &spec(root.path(), "large"),
        &limits(),
        &Cancellation::default(),
        options(length),
    )
    .unwrap();
    assert!(report.process.success(), "{:?}", report.process);
    let bytes = report.stdout.unwrap();
    assert_eq!(bytes.as_bytes(), vec![b'x'; length]);
}
#[test]
fn stderr_payload_is_exact_only_when_explicitly_selected() {
    let root = tempfile::tempdir().unwrap();
    let report = supervise_raw(
        &spec(root.path(), "raw"),
        &limits(),
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: None,
            stderr: Some(std::num::NonZeroUsize::new(64).unwrap()),
        },
    )
    .unwrap();
    assert!(report.process.success(), "{:?}", report.process);
    assert!(report.stdout.is_none());
    assert_eq!(
        report.stderr.unwrap().as_bytes(),
        b"token diagnostic\nordinary diagnostic\r\n"
    );
}
