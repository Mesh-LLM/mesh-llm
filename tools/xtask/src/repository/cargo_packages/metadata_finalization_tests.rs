use super::*;

#[test]
fn finalization_retains_setup_failure_when_signal_scope_fails() {
    let preceding = Err(process::Failure::InvalidSpec("cancelled before spawn"));

    let result = finalize(preceding, Err(Reason::Interrupted));

    assert!(
        matches!(result, Err(Error::Finalization { reason: Reason::Interrupted, preceding })
        if matches!(*preceding, Err(process::Failure::InvalidSpec("cancelled before spawn"))))
    );
}

#[test]
fn finalization_retains_complete_report_when_signal_scope_fails() {
    let preceding = Ok(process::RawProcessReport {
        process: process::ProcessReport {
            pid: 42,
            outcome: process::Outcome::Cancelled,
            status: None,
            ready: false,
            readiness_stop: process::ReadinessStop::NotAdmitted,
            elapsed: Duration::from_secs(3),
            stdout: process::StreamReport::default(),
            stderr: process::StreamReport::default(),
            cleanup: process::Cleanup {
                complete: false,
                forced: true,
                graceful_signal_failed: true,
                failure: Some(process::Failure::CleanupDeadline),
            },
            failure: Some(process::Failure::RawCaptureOverflow {
                stream: process::Stream::Stdout,
                limit: MAX_BYTES,
            }),
        },
        stdout: None,
        stderr: None,
    });
    let reason = Reason::Io {
        operation: "remove signal handler",
        kind: std::io::ErrorKind::PermissionDenied,
        code: Some(1),
    };

    let result = finalize(preceding, Err(reason));

    let Err(Error::Finalization { preceding, .. }) = result else {
        panic!("expected retained report");
    };
    let report = preceding.unwrap();
    assert_eq!(report.process.pid, 42);
    assert_eq!(report.process.outcome, process::Outcome::Cancelled);
    assert!(matches!(
        report.process.failure,
        Some(process::Failure::RawCaptureOverflow {
            stream: process::Stream::Stdout,
            limit: MAX_BYTES
        })
    ));
    assert!(matches!(
        report.process.cleanup.failure,
        Some(process::Failure::CleanupDeadline)
    ));
    assert!(!report.process.cleanup.complete);
    assert!(report.process.cleanup.forced);
    assert!(report.process.cleanup.graceful_signal_failed);
}
