use super::*;
#[path = "overlap.rs"]
mod endpoint;

fn armed(directory: &std::path::Path) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !directory.join("overlap.armed").exists() {
        assert!(
            std::time::Instant::now() < deadline,
            "headless did not reach live-primary barrier"
        );
        std::thread::yield_now();
    }
}

#[test]
fn primary_remains_live_when_headless_waits_on_barrier_for_all_variants() {
    for variant in Variant::REQUIRED {
        let directory = tempfile::tempdir().unwrap();
        let mut options = options(directory.path(), "overlap-success");
        options.variant = variant;
        let cancel = Cancellation::default();
        std::thread::scope(|scope| {
            let worker = scope.spawn(|| {
                command::execute(directory.path(), &options, &cancel)
                    .map_err(|error| error.to_string())
            });
            armed(directory.path());
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
        armed(directory.path());
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
        armed(directory.path());
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
