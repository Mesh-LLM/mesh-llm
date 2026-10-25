use super::{support::*, *};
use crate::process::{Failure, GracefulRequest, Outcome, ProcessReport, ReadinessStop};
use std::time::Duration;

#[test]
fn missing_headless_receipt_cannot_certify_standalone() {
    let session = completed(variant(Model::Dense));
    let result = session.finish((report(41), None), Ok(()));
    assert_eq!(result.unwrap_err().reason, Rejection::Incomplete);
}

#[test]
fn missing_decisions_cannot_be_replaced_by_successful_process_reports() {
    let result = session().finish((report(41), Some(report(42))), Ok(()));
    assert_eq!(result.unwrap_err().reason, Rejection::Incomplete);
}

#[test]
fn duplicated_process_receipt_cannot_replace_headless_process() {
    let session = completed(variant(Model::Dense));
    let result = session.finish((report(41), Some(report(41))), Ok(()));
    assert_eq!(result.unwrap_err().reason, Rejection::ProcessIdentity);
}

#[test]
fn zero_pid_cannot_supply_process_identity() {
    let session = completed(variant(Model::Dense));
    let result = session.finish((report(0), Some(report(42))), Ok(()));
    assert_eq!(result.unwrap_err().reason, Rejection::ProcessIdentity);
}

#[test]
fn early_exit_before_any_http_response_retains_its_actual_cause() {
    let session = session();
    let mut primary = report(41);
    primary.outcome = Outcome::EarlyExit;
    let result = session.finish((primary, None), Ok(()));
    assert_eq!(result.unwrap_err().reason, Rejection::EarlyExit);
}

#[test]
fn malformed_evidence_reason_survives_the_supervisor_rejection() {
    let mut session = session();
    session
        .observe(
            (Check::Runtime, response(br#"{"llama_ready":true}"#)),
            Duration::ZERO,
        )
        .unwrap();
    session
        .observe(
            (Check::Models, response(br#"{"data":[{"id":"chosen"}]}"#)),
            Duration::ZERO,
        )
        .unwrap();
    assert_eq!(
        session.observe((Check::Chat, response(b"{")), Duration::ZERO),
        Err(Rejection::Evidence(Check::Chat))
    );
    let mut primary = report(41);
    primary.outcome = Outcome::ObservationRejected;
    let result = session.finish((primary, None), Ok(()));
    assert_eq!(result.unwrap_err().reason, Rejection::Evidence(Check::Chat));
}

#[test]
fn early_exit_and_deadlines_reject_complete_http_evidence() {
    for (outcome, reason) in [
        (Outcome::Exited, Rejection::EarlyExit),
        (Outcome::EarlyExit, Rejection::EarlyExit),
        (Outcome::Deadline, Rejection::ProcessDeadline),
        (Outcome::ReadinessDeadline, Rejection::ProcessDeadline),
        (Outcome::Cancelled, Rejection::Cancelled),
        (Outcome::ObservationRejected, Rejection::ProcessFailure),
        (Outcome::IoFailure, Rejection::ProcessFailure),
    ] {
        for primary_failed in [true, false] {
            let session = completed(variant(Model::Recurrent));
            let mut primary = report(41);
            let mut headless = report(42);
            if primary_failed {
                primary.outcome = outcome;
            } else {
                headless.outcome = outcome;
            }
            let result = session.finish((primary, Some(headless)), Ok(()));
            assert_eq!(result.unwrap_err().reason, reason);
        }
    }
}

#[test]
fn post_process_teardown_errors_preserve_both_process_reports() {
    for reason in [
        Rejection::Cancelled,
        Rejection::WorkerFailed,
        Rejection::StateCleanup,
    ] {
        let session = completed(variant(Model::Dense));
        let error: Box<Rejected> = session
            .finish((report(41), Some(report(42))), Err(reason))
            .unwrap_err();
        assert_eq!(error.reason, reason);
        assert_eq!(error.primary.pid, 41);
        assert_eq!(error.headless.as_ref().unwrap().pid, 42);
    }
}

#[test]
fn clean_exit_does_not_replace_probe_admission() {
    for stop in [
        ReadinessStop::NotAdmitted,
        ReadinessStop::Admitted {
            observation: crate::process::ReadinessObservation {
                stream: crate::process::Stream::Stdout,
                elapsed: Duration::from_secs(1),
            },
            request: GracefulRequest::RequestedAfterLiveObservation,
        },
        ReadinessStop::ProbeAdmitted {
            elapsed: Duration::from_secs(1),
            request: GracefulRequest::LeaderExitedBeforeRequest,
        },
    ] {
        let session = completed(variant(Model::Dense));
        let mut primary = report(41);
        primary.readiness_stop = stop;
        let result = session.finish((primary, Some(report(42))), Ok(()));
        assert_eq!(result.unwrap_err().reason, Rejection::StopNotAdmitted);
    }
}

macro_rules! cleanup_rejected {
    ($name:ident, $change:expr) => {
        #[test]
        fn $name() {
            let session = completed(variant(Model::Dense));
            let mut headless = report(42);
            ($change)(&mut headless);
            let result = session.finish((report(41), Some(headless)), Ok(()));
            assert_eq!(result.unwrap_err().reason, Rejection::ProcessCleanup);
        }
    };
}

cleanup_rejected!(
    forced_shutdown_rejects_completion,
    |report: &mut ProcessReport| report.cleanup.forced = true
);
cleanup_rejected!(
    unfinished_tree_rejects_completion,
    |report: &mut ProcessReport| report.cleanup.complete = false
);
cleanup_rejected!(
    failed_signal_rejects_completion,
    |report: &mut ProcessReport| report.cleanup.graceful_signal_failed = true
);
cleanup_rejected!(
    cleanup_deadline_rejects_completion,
    |report: &mut ProcessReport| report.cleanup.failure = Some(Failure::CleanupDeadline)
);
cleanup_rejected!(
    capture_failure_rejects_completion,
    |report: &mut ProcessReport| report.failure = Some(Failure::EnumerationLimit)
);
cleanup_rejected!(
    missing_exit_rejects_completion,
    |report: &mut ProcessReport| report.status = None
);
cleanup_rejected!(
    nonzero_exit_rejects_completion,
    |report: &mut ProcessReport| report.status = Some(exit(7))
);
cleanup_rejected!(
    missing_ready_fact_rejects_completion,
    |report: &mut ProcessReport| report.ready = false
);

#[test]
fn receipt_diagnostics_do_not_serialize_child_output_or_responses() {
    let session = completed(variant(Model::Dense));
    let mut primary = report(41);
    primary.stdout.bytes_retained = b"invite-token-fixture".to_vec();
    let error = session
        .finish((primary, None), Err(Rejection::StateCleanup))
        .unwrap_err();
    assert!(!format!("{error:?} {error}").contains("invite-token-fixture"));
    assert_eq!(error.primary.stdout.bytes_retained, b"invite-token-fixture");
}
