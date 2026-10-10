use super::*;

#[test]
fn raw_deadline_cleans_descendants_and_preserves_sentinel() {
    let mut budget = limits();
    budget.execution = Duration::from_millis(500);

    let report = raw_tree_report(&budget, &Cancellation::default());

    assert_eq!(report.process.outcome, Outcome::Deadline);
}

#[test]
fn raw_cancellation_cleans_descendants_and_preserves_sentinel() {
    static CANCELLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    fn cancel_on_tree(line: ObservedLine<'_>) -> bool {
        if line.bytes == b"TREE_READY" {
            CANCELLED.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        false
    }
    CANCELLED.store(false, std::sync::atomic::Ordering::SeqCst);
    let mut budget = limits();
    budget.readiness = Readiness::ObservedLines {
        deadline: Duration::from_secs(2),
        matcher: cancel_on_tree,
    };
    budget.completion = Completion::StopAfterReady;

    let report = raw_tree_report(&budget, &Cancellation::from_static(&CANCELLED));

    assert_eq!(report.process.outcome, Outcome::Cancelled);
}

struct ReleaseHolder(std::path::PathBuf);

impl Drop for ReleaseHolder {
    fn drop(&mut self) {
        std::fs::write(self.0.join("release-holder"), b"release").unwrap();
    }
}

fn held_pipe(cap: usize) -> RawProcessReport {
    let root = tempfile::tempdir().unwrap();
    let release = ReleaseHolder(root.path().to_owned());
    let mut budget = limits();
    budget.graceful_shutdown = Duration::from_millis(20);
    budget.forced_shutdown = Duration::from_millis(100);

    let report = supervise_raw(
        &spec(root.path(), "held-pipe"),
        &budget,
        &Cancellation::default(),
        options(cap),
    )
    .unwrap();

    drop(release);
    let until = std::time::Instant::now() + Duration::from_secs(2);
    while !root.path().join("holder-done").is_file() {
        assert!(
            std::time::Instant::now() < until,
            "holder did not acknowledge release"
        );
        std::thread::park_timeout(Duration::from_millis(1));
    }
    assert!(report.stdout.is_none());
    assert!(!report.process.success());
    assert!(report.process.status.is_some());
    report
}

#[test]
fn cleanup_deadline_precedes_incomplete_raw_capture_when_pipe_stays_open() {
    let report = held_pipe(64);

    assert!(matches!(
        report.process.failure,
        Some(Failure::CleanupDeadline)
    ));
    assert!(report.process.status.is_some_and(|status| status.success()));
}

#[test]
fn raw_overflow_precedes_cleanup_deadline_when_pipe_stays_open() {
    let report = held_pipe(1);

    assert!(matches!(
        report.process.failure,
        Some(Failure::RawCaptureOverflow {
            stream: Stream::Stdout,
            limit: 1
        })
    ));
}
