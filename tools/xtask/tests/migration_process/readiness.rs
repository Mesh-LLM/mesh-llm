use crate::{process::*, support::*};
use std::time::{Duration, Instant};

fn cleanup_case(mode: &str, limits: &Limits) -> ProcessReport {
    let root = tempfile::tempdir().unwrap();
    let mut sentinel = Sentinel::new(root.path());
    let report = supervise(
        &spec(root.path(), mode),
        limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    sentinel.assert_alive();
    assert_stopped(root.path(), &[mode, "cleanup-leaf"]);
    assert!(root.path().join("shutdown-observed").is_file());
    assert!(
        report
            .stdout
            .bytes_retained
            .windows(6)
            .any(|line| line == b"READY\n")
    );
    assert!(report.cleanup.complete, "{report:?}");
    report
}

#[test]
fn migration_process_readiness_cleanup_marker_cannot_repair_early_exit() {
    let mut limits = limits();
    ready(&mut limits, b"READY");
    limits.graceful_shutdown = Duration::from_secs(1);
    let report = cleanup_case("cleanup-leader", &limits);
    assert_eq!(report.outcome, Outcome::EarlyExit, "{report:?}");
    assert!(!report.ready);
    assert!(!report.success());
    assert_eq!(report.status.unwrap().code(), Some(0));
}

#[test]
fn migration_process_readiness_cleanup_marker_cannot_repair_deadline() {
    let mut limits = limits();
    limits.readiness = Readiness::Line {
        stream: Stream::Stdout,
        bytes: b"READY".to_vec(),
        deadline: Duration::from_millis(500),
    };
    limits.graceful_shutdown = Duration::from_secs(1);
    let report = cleanup_case("cleanup-timeout", &limits);
    assert_eq!(report.outcome, Outcome::ReadinessDeadline, "{report:?}");
    assert!(!report.ready);
    assert!(!report.success());
}

#[test]
fn migration_process_readiness_accepted_before_shutdown_is_preserved() {
    let mut limits = limits();
    ready(&mut limits, b"READY");
    limits.completion = Completion::StopAfterReady;
    limits.graceful_shutdown = Duration::from_secs(1);
    let report = cleanup_case("cleanup-ready", &limits);
    assert_eq!(report.outcome, Outcome::Ready, "{report:?}");
    assert!(report.ready);
    assert!(report.success(), "{report:?}");
}

#[test]
fn migration_process_readiness_cleanup_marker_cannot_repair_cancellation() {
    let root = tempfile::tempdir().unwrap();
    let mut limits = limits();
    ready(&mut limits, b"READY");
    limits.graceful_shutdown = Duration::from_secs(1);
    let cancellation = Cancellation::default();
    let cancel = cancellation.clone();
    let marker = root.path().join("armed");
    let worker = std::thread::spawn(move || {
        let until = Instant::now() + Duration::from_secs(3);
        while !marker.is_file() && Instant::now() < until {
            std::thread::park_timeout(Duration::from_millis(1));
        }
        cancel.cancel();
    });
    let report = supervise(
        &spec(root.path(), "cleanup-cancel"),
        &limits,
        &cancellation,
        OutputFiles::default(),
    )
    .unwrap();
    worker.join().unwrap();
    assert_stopped(root.path(), &["cleanup-cancel", "cleanup-leaf"]);
    assert!(root.path().join("shutdown-observed").is_file());
    assert_eq!(report.outcome, Outcome::Cancelled);
    assert!(!report.ready);
    assert!(report.cleanup.complete);
}
