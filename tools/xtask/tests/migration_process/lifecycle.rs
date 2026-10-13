use crate::{process::*, support::*};
use std::time::{Duration, Instant};

#[test]
fn migration_process_ready_then_successful_exit() {
    let root = tempfile::tempdir().unwrap();
    let mut limits = limits();
    ready(&mut limits, b"READY");
    let report = supervise(
        &spec(root.path(), "exit"),
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(report.success(), "{report:?}");
    assert!(report.ready);
    assert_eq!(report.status.unwrap().code(), Some(0));
}

#[test]
fn migration_process_early_crash_propagates_exit_and_diagnostics() {
    let root = tempfile::tempdir().unwrap();
    let mut limits = limits();
    ready(&mut limits, b"READY");
    let report = supervise(
        &spec(root.path(), "crash"),
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert_eq!(report.outcome, Outcome::EarlyExit);
    assert_eq!(report.status.unwrap().code(), Some(23));
    assert!(report.cleanup.complete);
    assert!(
        report
            .stderr
            .bytes_retained
            .windows(13)
            .any(|bytes| bytes == b"early failure")
    );
    assert!(!report.success());
}

fn tree_case(mode: &str, expected: Outcome) {
    let root = tempfile::tempdir().unwrap();
    let mut sentinel = Sentinel::new(root.path());
    let mut limits = limits();
    limits.execution = Duration::from_millis(600);
    let report = supervise(
        &spec(root.path(), mode),
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    sentinel.assert_alive();
    assert_stopped(root.path(), &[mode, "branch", "leaf"]);
    assert_eq!(report.outcome, expected, "{report:?}");
    assert!(report.cleanup.complete, "{report:?}");
    assert!(
        report.cleanup.forced,
        "stubborn leaf must require escalation: {report:?}"
    );
    assert!(!report.success());
    assert!(report.elapsed < Duration::from_secs(5));
}

#[test]
fn migration_process_successful_leader_cleans_child_and_grandchild() {
    tree_case("tree-exit", Outcome::Exited);
}
#[test]
fn migration_process_failed_leader_cleans_child_and_grandchild() {
    tree_case("tree-crash", Outcome::Exited);
}
#[test]
fn migration_process_hung_tree_deadline_preserves_sentinel() {
    tree_case("tree-hang", Outcome::Deadline);
}

#[test]
fn migration_process_readiness_deadline_does_not_accept_sleep_as_readiness() {
    let root = tempfile::tempdir().unwrap();
    let mut limits = limits();
    limits.readiness = Readiness::Line {
        stream: Stream::Stdout,
        bytes: b"MISSING".to_vec(),
        deadline: Duration::from_millis(100),
    };
    let report = supervise(
        &spec(root.path(), "hang"),
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert_eq!(report.outcome, Outcome::ReadinessDeadline);
    assert!(!report.ready);
    assert!(report.cleanup.complete);
    assert!(!alive(report.pid), "timed-out leader survived");
}

#[test]
fn migration_process_cancellation_cleans_live_tree() {
    let root = tempfile::tempdir().unwrap();
    let cancellation = Cancellation::default();
    let cancel = cancellation.clone();
    let marker = root.path().join("leaf.ready");
    let worker = std::thread::spawn(move || {
        let until = Instant::now() + Duration::from_secs(3);
        while !marker.is_file() && Instant::now() < until {
            std::thread::park_timeout(Duration::from_millis(5));
        }
        cancel.cancel();
    });
    let report = supervise(
        &spec(root.path(), "tree-hang"),
        &limits(),
        &cancellation,
        OutputFiles::default(),
    )
    .unwrap();
    worker.join().unwrap();
    assert_eq!(report.outcome, Outcome::Cancelled);
    assert!(report.cleanup.complete);
    assert_stopped(root.path(), &["tree-hang", "branch", "leaf"]);
}

#[test]
fn migration_process_stops_after_ready_with_graceful_exit() {
    let root = tempfile::tempdir().unwrap();
    let mut limits = limits();
    ready(&mut limits, b"READY");
    limits.completion = Completion::StopAfterReady;
    let report = supervise(
        &spec(root.path(), "ready-hang"),
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert_eq!(report.outcome, Outcome::Ready);
    assert!(report.success(), "{report:?}");
}

#[test]
fn migration_process_invalid_executable_fails_without_spawning() {
    let root = tempfile::tempdir().unwrap();
    let mut spec = spec(root.path(), "exit");
    spec.executable = root.path().join("missing.exe");
    let result = supervise(
        &spec,
        &limits(),
        &Cancellation::default(),
        OutputFiles::default(),
    );
    assert!(matches!(
        result,
        Err(Failure::Io {
            operation: "spawn",
            ..
        })
    ));
    assert!(!root.path().join("exit.pid").exists());
}

#[cfg(unix)]
#[test]
fn migration_process_signal_exit_retains_signal_identity() {
    use std::os::unix::process::ExitStatusExt;
    let root = tempfile::tempdir().unwrap();
    let report = supervise(
        &spec(root.path(), "signal-exit"),
        &limits(),
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert_eq!(report.status.unwrap().signal(), Some(libc::SIGKILL));
    assert!(report.cleanup.complete);
    assert!(!report.success());
}

#[test]
fn migration_process_cancelled_before_spawn_creates_no_child() {
    let root = tempfile::tempdir().unwrap();
    let cancel = Cancellation::default();
    cancel.cancel();
    let result = supervise(
        &spec(root.path(), "exit"),
        &limits(),
        &cancel,
        OutputFiles::default(),
    );
    assert!(matches!(result, Err(Failure::InvalidSpec(_))));
    assert!(!root.path().join("exit.pid").exists());
}

#[test]
fn migration_process_successful_tree_exits_without_escalation() {
    let root = tempfile::tempdir().unwrap();
    let report = supervise(
        &spec(root.path(), "tree-graceful"),
        &limits(),
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(report.success(), "{report:?}");
    assert_stopped(root.path(), &["tree-graceful", "branch-graceful", "leaf"]);
}
