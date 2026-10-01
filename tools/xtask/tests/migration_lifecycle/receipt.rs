use super::accept;
use crate::process::*;
use std::process::ExitStatus;
use std::time::Duration;

fn status(code: u8) -> ExitStatus {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        ExitStatus::from_raw(i32::from(code) << 8)
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::ExitStatusExt;
        ExitStatus::from_raw(u32::from(code))
    }
}

fn ready_report() -> ProcessReport {
    ProcessReport {
        pid: 1,
        outcome: Outcome::Ready,
        status: Some(status(0)),
        ready: true,
        readiness_stop: ReadinessStop::Admitted {
            observation: ReadinessObservation {
                stream: Stream::Stderr,
                elapsed: Duration::from_millis(1),
            },
            request: GracefulRequest::RequestedAfterLiveObservation,
        },
        elapsed: Duration::from_secs(2),
        stdout: StreamReport::default(),
        stderr: StreamReport::default(),
        cleanup: Cleanup {
            complete: true,
            ..Cleanup::default()
        },
        failure: None,
    }
}

#[test]
fn migration_lifecycle_accepts_only_clean_requested_stop() {
    let report = ready_report();

    let result = accept(report);

    assert!(result.is_ok());
}

macro_rules! rejected {
    ($name:ident, $edit:expr) => {
        #[test]
        fn $name() {
            let mut report = ready_report();
            ($edit)(&mut report);

            let result = accept(report);

            assert!(result.is_err());
        }
    };
}

rejected!(
    migration_lifecycle_rejects_no_receipt,
    |report: &mut ProcessReport| report.readiness_stop = ReadinessStop::NotAdmitted
);
rejected!(
    migration_lifecycle_rejects_probe_receipt,
    |report: &mut ProcessReport| report.readiness_stop = ReadinessStop::ProbeAdmitted {
        elapsed: Duration::from_millis(1),
        request: GracefulRequest::RequestedAfterLiveObservation,
    }
);
rejected!(
    migration_lifecycle_rejects_observation_rejected,
    |report: &mut ProcessReport| report.outcome = Outcome::ObservationRejected
);
rejected!(
    migration_lifecycle_rejects_missing_status,
    |report: &mut ProcessReport| report.status = None
);
rejected!(
    migration_lifecycle_rejects_nonzero_status,
    |report: &mut ProcessReport| report.status = Some(status(23))
);
rejected!(
    migration_lifecycle_rejects_force,
    |report: &mut ProcessReport| report.cleanup.forced = true
);
rejected!(
    migration_lifecycle_rejects_incomplete_cleanup,
    |report: &mut ProcessReport| report.cleanup.complete = false
);
rejected!(
    migration_lifecycle_rejects_signal_failure,
    |report: &mut ProcessReport| report.cleanup.graceful_signal_failed = true
);
rejected!(
    migration_lifecycle_rejects_cleanup_failure,
    |report: &mut ProcessReport| report.cleanup.failure = Some(Failure::CleanupDeadline)
);
rejected!(
    migration_lifecycle_rejects_output_failure,
    |report: &mut ProcessReport| report.failure = Some(Failure::CleanupDeadline)
);
rejected!(
    migration_lifecycle_rejects_cancellation,
    |report: &mut ProcessReport| report.outcome = Outcome::Cancelled
);
rejected!(
    migration_lifecycle_rejects_deadline,
    |report: &mut ProcessReport| report.outcome = Outcome::Deadline
);
rejected!(
    migration_lifecycle_rejects_readiness_deadline,
    |report: &mut ProcessReport| report.outcome = Outcome::ReadinessDeadline
);
rejected!(
    migration_lifecycle_rejects_early_exit,
    |report: &mut ProcessReport| report.outcome = Outcome::EarlyExit
);
rejected!(
    migration_lifecycle_rejects_unadmitted_ready_flag,
    |report: &mut ProcessReport| report.ready = false
);

macro_rules! rejected_request {
    ($name:ident, $request:expr) => {
        rejected!($name, |report: &mut ProcessReport| {
            report.readiness_stop = ReadinessStop::Admitted {
                observation: ReadinessObservation {
                    stream: Stream::Stdout,
                    elapsed: Duration::ZERO,
                },
                request: $request,
            };
        });
    };
}

rejected_request!(
    migration_lifecycle_rejects_known_pre_request_exit,
    GracefulRequest::LeaderExitedBeforeRequest
);
rejected_request!(
    migration_lifecycle_rejects_unknown_leader,
    GracefulRequest::LeaderObservationFailed(Failure::CleanupDeadline)
);
rejected_request!(
    migration_lifecycle_rejects_skipped_request,
    GracefulRequest::SkippedInactiveTree
);
rejected_request!(
    migration_lifecycle_rejects_unknown_tree,
    GracefulRequest::TreeObservationFailed
);
rejected_request!(
    migration_lifecycle_rejects_failed_request,
    GracefulRequest::RequestFailed(Failure::CleanupDeadline)
);
