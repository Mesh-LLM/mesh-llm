use super::*;
#[path = "overlap.rs"]
mod endpoint;

// The primary may take its readiness budget plus coordinated.rs's 300 s
// allowance before it answers and the headless writes overlap.armed.
fn armed<T>(
    directory: &std::path::Path,
    readiness: std::time::Duration,
    worker: &std::thread::ScopedJoinHandle<'_, T>,
) {
    let deadline = std::time::Instant::now() + readiness + std::time::Duration::from_secs(300 + 30);
    while !directory.join("overlap.armed").exists() {
        assert!(
            !worker.is_finished(),
            "worker finished before overlap.armed was written"
        );
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for overlap.armed at live-primary barrier"
        );
        std::thread::park_timeout(std::time::Duration::from_millis(10));
    }
}

#[test]
fn primary_remains_live_when_headless_waits_on_barrier_for_all_variants() {
    for variant in Variant::REQUIRED {
        let directory = tempfile::tempdir().unwrap();
        let mut options = options(directory.path(), "overlap-success");
        options.variant = variant;
        options.readiness = std::time::Duration::from_secs(120);
        options.shutdown = std::time::Duration::from_secs(10);
        let cancel = Cancellation::default();
        std::thread::scope(|scope| {
            let worker = scope.spawn(|| {
                command::execute(directory.path(), &options, &cancel)
                    .map_err(|error| error.to_string())
            });
            armed(directory.path(), options.readiness, &worker);
            endpoint::query(directory.path()).unwrap();
            std::fs::write(directory.path().join("overlap.release"), b"release").unwrap();
            let receipt = worker.join().unwrap().unwrap();
            assert!(
                receipt
                    .reports()
                    .iter()
                    .all(|report| report.success() && report.status.unwrap().code() == Some(0))
            );
            assert_eq!(receipt.reports().len(), 2);
        });
    }
}

#[test]
fn cancellation_retains_both_reports_when_headless_waits_on_barrier() {
    let directory = tempfile::tempdir().unwrap();
    let options = options(directory.path(), "overlap-cancel");
    let cancel = Cancellation::default();
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            let error = command::execute(directory.path(), &options, &cancel).unwrap_err();
            (
                output::reason(error.as_ref()),
                output::reports(error.as_ref())
                    .iter()
                    .map(|report| (report.pid, report.cleanup.complete))
                    .collect::<Vec<_>>(),
            )
        });
        armed(directory.path(), options.readiness, &worker);
        endpoint::query(directory.path()).unwrap();
        cancel.cancel();
        let (reason, reports) = worker.join().unwrap();
        assert_eq!(reason, Rejection::Cancelled.to_string());
        assert_eq!(reports.len(), 2);
        assert!(reports.iter().all(|(pid, complete)| *pid > 0 && *complete));
    });
}

#[test]
fn primary_early_exit_rejects_when_headless_waits_on_barrier() {
    let directory = tempfile::tempdir().unwrap();
    let options = options(directory.path(), "overlap-early");
    let cancel = Cancellation::default();
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            let error = command::execute(directory.path(), &options, &cancel).unwrap_err();
            (
                output::reason(error.as_ref()),
                output::reports(error.as_ref()).len(),
            )
        });
        armed(directory.path(), options.readiness, &worker);
        std::fs::write(directory.path().join("primary.exit"), b"exit").unwrap();
        let (reason, count) = worker.join().unwrap();
        assert_eq!(reason, Rejection::EarlyExit.to_string());
        assert_eq!(count, 2);
    });
}

#[test]
fn cleanup_failure_retains_both_reports_when_overlap_checks_passed() {
    let directory = tempfile::tempdir().unwrap();
    let options = options(directory.path(), "nonzero");
    let error = command::execute(directory.path(), &options, &Cancellation::default()).unwrap_err();
    assert_eq!(
        output::reason(error.as_ref()),
        Rejection::ProcessCleanup.to_string()
    );
    let reports = output::reports(error.as_ref());
    assert_eq!(reports.len(), 2);
    assert!(
        reports
            .iter()
            .all(|report| report.status.unwrap().code() == Some(23))
    );
    assert!(directory.path().join("overlap.observed").exists());
}
