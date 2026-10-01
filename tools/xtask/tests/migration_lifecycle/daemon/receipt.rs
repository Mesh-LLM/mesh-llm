use super::*;
use crate::automation::daemon_readiness::http::Transfer;
use crate::process::*;
use std::time::Duration;

fn report() -> ProbeReport<Rejection> {
    #[cfg(unix)]
    let status = {
        use std::os::unix::process::ExitStatusExt;
        std::process::ExitStatus::from_raw(0)
    };
    #[cfg(windows)]
    let status = {
        use std::os::windows::process::ExitStatusExt;
        std::process::ExitStatus::from_raw(0)
    };
    ProbeReport {
        rejection: None,
        process: ProcessReport {
            pid: 42,
            outcome: Outcome::Ready,
            status: Some(status),
            ready: true,
            readiness_stop: ReadinessStop::ProbeAdmitted {
                elapsed: Duration::from_millis(1),
                request: GracefulRequest::RequestedAfterLiveObservation,
            },
            elapsed: Duration::from_secs(1),
            stdout: StreamReport::default(),
            stderr: StreamReport::default(),
            cleanup: Cleanup {
                complete: true,
                ..Cleanup::default()
            },
            failure: None,
        },
    }
}

fn facts() -> Facts {
    Facts {
        phase: Phase::Complete,
        status: Some(Transfer {
            status_code: 200,
            body_bytes: 64,
        }),
        models: Some(Transfer {
            status_code: 200,
            body_bytes: 0,
        }),
        status_attempts: 1,
    }
}

#[test]
fn d18_clean_probe_receipt() {
    assert!(accept(report(), facts(), (false, true)).is_ok());
}
#[test]
fn d20_worker_join_failure_retains_process_receipt() {
    let error = accept(report(), facts(), (false, false)).unwrap_err();
    assert!(
        matches!(error, Error::Lifecycle(rejected) if rejected.process.ready && rejected.cause == "worker_failed")
    );
}
#[test]
fn d19_interruption_retains_ready_history() {
    assert!(accept(report(), facts(), (true, true)).is_err());
}

#[test]
fn d19_interruption_after_state_cleanup_retains_ready_history() {
    let accepted = accept(report(), facts(), (false, true)).unwrap();
    let error = accepted
        .finish(Err(
            crate::automation::command_interrupt::Reason::Interrupted,
        ))
        .unwrap_err();
    assert!(matches!(error, Error::Completion { receipt, .. } if receipt.process.ready));
}

macro_rules! rejected {
    ($name:ident, $edit:expr) => {
        #[test]
        fn $name() {
            let mut report = report();
            ($edit)(&mut report.process);
            assert!(accept(report, facts(), (false, true)).is_err());
        }
    };
}
rejected!(d20_status_missing, |report: &mut ProcessReport| report
    .status =
    None);
rejected!(d20_output_failure, |report: &mut ProcessReport| report
    .failure =
    Some(Failure::CleanupDeadline));
rejected!(d20_tree_failure, |report: &mut ProcessReport| report
    .cleanup
    .failure =
    Some(Failure::EnumerationLimit));
rejected!(d20_signal_failure, |report: &mut ProcessReport| report
    .cleanup
    .graceful_signal_failed =
    true);
rejected!(d20_incomplete, |report: &mut ProcessReport| report
    .cleanup
    .complete =
    false);
rejected!(d18_forced, |report: &mut ProcessReport| report
    .cleanup
    .forced = true);
rejected!(d17_leader_exited, |report: &mut ProcessReport| report
    .readiness_stop =
    ReadinessStop::ProbeAdmitted {
        elapsed: Duration::ZERO,
        request: GracefulRequest::LeaderExitedBeforeRequest
    });
rejected!(d20_request_failed, |report: &mut ProcessReport| report
    .readiness_stop =
    ReadinessStop::ProbeAdmitted {
        elapsed: Duration::ZERO,
        request: GracefulRequest::RequestFailed(Failure::CleanupDeadline)
    });
rejected!(d16_not_admitted, |report: &mut ProcessReport| report
    .readiness_stop =
    ReadinessStop::NotAdmitted);

#[test]
fn d02_attribution_unavailable_only_on_owner_deadline() {
    let mut report = report();
    report.process.outcome = Outcome::Deadline;
    let mut facts = facts();
    facts.phase = Phase::Attribution;
    assert!(
        matches!(accept(report, facts, (false, true)), Err(Error::Lifecycle(error)) if error.cause == "attribution_unavailable")
    );
}
