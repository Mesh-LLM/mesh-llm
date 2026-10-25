use crate::{process::*, support::*};
use std::time::{Duration, Instant};

fn observed_limits() -> Limits {
    let mut limits = limits();
    limits.readiness = Readiness::ObservedLines {
        deadline: Duration::from_secs(2),
        matcher: |line| {
            line.ending == LineEnding::Lf
                && line.bytes.strip_suffix(b"\r").unwrap_or(line.bytes) == b"READY"
        },
    };
    limits.completion = Completion::StopAfterReady;
    limits.graceful_shutdown = Duration::from_secs(1);
    limits
}

#[test]
fn migration_process_observed_stderr_ready_stop_with_zero_and_saturated_retention() {
    for cap in [0, 128] {
        let root = tempfile::tempdir().unwrap();
        let mut sentinel = Sentinel::new(root.path());
        let mut limits = observed_limits();
        limits.retained_bytes_per_stream = cap;
        let report = supervise(
            &spec(root.path(), "observed-ready"),
            &limits,
            &Cancellation::default(),
            OutputFiles {
                stdout: Some(root.path().join("out")),
                stderr: Some(root.path().join("err")),
            },
        )
        .unwrap();
        assert!(report.success(), "{report:?}");
        assert!(
            matches!(report.readiness_stop, ReadinessStop::Admitted { observation, request: GracefulRequest::RequestedAfterLiveObservation } if observation.stream == Stream::Stderr && observation.elapsed < Duration::from_secs(2))
        );
        assert!(root.path().join("observed-stop").is_file());
        for (name, stream) in [("out", &report.stdout), ("err", &report.stderr)] {
            assert_eq!(stream.bytes_retained.len(), cap);
            assert!(stream.truncated);
            assert_eq!(
                std::fs::read(root.path().join(name)).unwrap(),
                stream.bytes_retained
            );
        }
        sentinel.assert_alive();
        assert_stopped(root.path(), &["observed-ready"]);
        drop(sentinel);
        root.close().unwrap();
    }
}

#[test]
fn migration_process_observed_stubborn_ready_requires_force_and_non_success() {
    let root = tempfile::tempdir().unwrap();
    let mut limits = observed_limits();
    limits.graceful_shutdown = Duration::from_millis(100);
    let report = supervise(
        &spec(root.path(), "observed-stubborn"),
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(report.ready);
    assert!(report.cleanup.forced && report.cleanup.complete);
    assert!(!report.success());
    assert!(!root.path().join("observed-stop").exists());
    assert_stopped(root.path(), &["observed-stubborn"]);
}

#[test]
fn migration_process_observed_nonzero_handler_retains_exit_identity() {
    let root = tempfile::tempdir().unwrap();
    let report = supervise(
        &spec(root.path(), "observed-nonzero"),
        &observed_limits(),
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(report.ready && report.cleanup.complete);
    assert!(!report.success());
    assert_eq!(report.status.unwrap().code(), Some(23));
    assert!(root.path().join("observed-stop").is_file());
    assert_stopped(root.path(), &["observed-nonzero"]);
}

#[test]
fn migration_process_observed_cleanup_only_line_cannot_repair_deadline() {
    let root = tempfile::tempdir().unwrap();
    let mut limits = observed_limits();
    limits.readiness = Readiness::ObservedLines {
        deadline: Duration::from_millis(500),
        matcher: |line| line.ending == LineEnding::Lf && line.bytes == b"READY",
    };
    let report = supervise(
        &spec(root.path(), "observed-timeout"),
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert_eq!(report.outcome, Outcome::ReadinessDeadline);
    assert!(!report.ready && !report.success());
    assert!(matches!(report.readiness_stop, ReadinessStop::NotAdmitted));
    assert!(root.path().join("observed-stop").is_file());
    assert!(report.stderr.bytes_retained.ends_with(b"READY\n"));
    assert_stopped(root.path(), &["observed-timeout"]);
}

#[test]
fn migration_process_observed_descendant_cleanup_only_line_cannot_repair_exit() {
    let root = tempfile::tempdir().unwrap();
    let mut sentinel = Sentinel::new(root.path());
    let report = supervise(
        &spec(root.path(), "cleanup-leader"),
        &observed_limits(),
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert_eq!(report.outcome, Outcome::EarlyExit);
    assert!(!report.ready && !report.success());
    assert!(matches!(report.readiness_stop, ReadinessStop::NotAdmitted));
    assert!(
        report
            .stdout
            .bytes_retained
            .split_inclusive(|byte| *byte == b'\n')
            .any(|line| line == b"READY\n")
    );
    assert!(root.path().join("shutdown-observed").is_file());
    assert!(report.cleanup.complete);
    sentinel.assert_alive();
    assert_stopped(root.path(), &["cleanup-leader", "cleanup-leaf"]);
}

#[test]
fn migration_process_observed_cancellation_freezes_cleanup_only_readiness() {
    let root = tempfile::tempdir().unwrap();
    let cancellation = Cancellation::default();
    let cancel = cancellation.clone();
    let marker = root.path().join("armed");
    let worker = std::thread::spawn(move || {
        let until = Instant::now() + Duration::from_secs(3);
        while !marker.is_file() && Instant::now() < until {
            std::thread::park_timeout(Duration::from_millis(1));
        }
        let armed = marker.is_file();
        cancel.cancel();
        armed
    });
    let report = supervise(
        &spec(root.path(), "cleanup-cancel"),
        &observed_limits(),
        &cancellation,
        OutputFiles::default(),
    )
    .unwrap();
    assert!(worker.join().unwrap());
    assert_eq!(report.outcome, Outcome::Cancelled);
    assert!(!report.ready);
    assert!(matches!(report.readiness_stop, ReadinessStop::NotAdmitted));
    assert!(root.path().join("shutdown-observed").is_file());
    assert!(report.cleanup.complete);
    assert_stopped(root.path(), &["cleanup-cancel", "cleanup-leaf"]);
}

#[test]
fn migration_process_observed_unterminated_eof_is_not_lf_readiness() {
    let root = tempfile::tempdir().unwrap();
    let report = supervise(
        &spec(root.path(), "buffered-eof"),
        &observed_limits(),
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert_eq!(report.outcome, Outcome::EarlyExit);
    assert!(!report.ready && !report.success());
    assert!(report.cleanup.complete);
    assert_stopped(root.path(), &["buffered-eof"]);
}

#[test]
fn migration_process_observed_flood_without_line_cannot_starve_deadline() {
    let root = tempfile::tempdir().unwrap();
    let mut limits = observed_limits();
    limits.retained_bytes_per_stream = 0;
    limits.readiness = Readiness::ObservedLines {
        deadline: Duration::from_millis(100),
        matcher: |line| {
            !line.bytes.is_empty() && line.bytes.iter().all(|byte| *byte == b'x' || *byte == b'y')
        },
    };
    let spec = spec(root.path(), "forever-flood");
    let report = supervise(
        &spec,
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert_eq!(report.outcome, Outcome::ReadinessDeadline);
    assert!(!report.ready && report.cleanup.complete);
    assert!(report.elapsed < Duration::from_secs(3));
}
