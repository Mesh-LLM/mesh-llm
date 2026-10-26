use super::*;
use crate::process::{
    Cancellation, Completion, Outcome, ProcessReport, Readiness, ReadinessStop, Stream,
    StreamReport,
    observed::{self, Admission, Decision, Lines},
};
use std::collections::VecDeque;

enum Step {
    Leader(Result<bool, Failure>),
    Active(Result<bool, Failure>),
    Graceful(Result<(), Failure>),
    Force,
    Reap,
}

struct Script {
    steps: VecDeque<Step>,
    status: Option<ExitStatus>,
}

impl Leader for Script {
    fn exited(&mut self) -> Result<bool, Failure> {
        match self.steps.pop_front() {
            Some(Step::Leader(result)) => result,
            _ => panic!("unexpected leader observation"),
        }
    }
}

impl Tree for Script {
    fn active(&mut self) -> Result<bool, Failure> {
        match self.steps.pop_front() {
            Some(Step::Active(result)) => result,
            _ => panic!("unexpected tree observation"),
        }
    }
    fn graceful(&mut self) -> Result<(), Failure> {
        match self.steps.pop_front() {
            Some(Step::Graceful(result)) => result,
            _ => panic!("unexpected graceful request"),
        }
    }
    fn force(&mut self) -> Result<(), Failure> {
        assert!(matches!(self.steps.pop_front(), Some(Step::Force)));
        Ok(())
    }
    fn reap(&mut self) -> Result<Option<ExitStatus>, Failure> {
        match self.steps.pop_front() {
            Some(Step::Reap) => Ok(self.status),
            None if self.status.is_none() => Ok(None),
            _ => panic!("unexpected reap"),
        }
    }
}

struct Candidate;
impl Lines for Candidate {
    fn poll_lines(&mut self, _: &Readiness) -> Result<Option<Stream>, Failure> {
        Ok(Some(Stream::Stderr))
    }
}

fn status(code: i32) -> ExitStatus {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        ExitStatus::from_raw(code * 256)
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::ExitStatusExt;
        ExitStatus::from_raw(u32::try_from(code).unwrap())
    }
}

fn failure() -> Failure {
    Failure::io(
        "scripted observation",
        std::io::ErrorKind::PermissionDenied.into(),
    )
}

fn limits() -> Limits {
    Limits {
        execution: Duration::from_secs(2),
        graceful_shutdown: Duration::from_millis(1),
        forced_shutdown: Duration::from_millis(1),
        retained_bytes_per_stream: 0,
        readiness: Readiness::ObservedLines {
            deadline: Duration::from_secs(1),
            matcher: |_| true,
        },
        completion: Completion::StopAfterReady,
    }
}

fn run(mut script: Script) -> ProcessReport {
    let limits = limits();
    let now = std::cell::Cell::new(Duration::from_secs(1) - Duration::from_nanos(1));
    let cancellation = Cancellation::default();
    let decision = observed::monitor(
        &mut script,
        &mut Candidate,
        &Admission {
            limits: &limits,
            cancellation: &cancellation,
            elapsed: || now.get(),
        },
    );
    let observation = match decision {
        Decision::Admitted(observation) => observation,
        _ => panic!("admission failed"),
    };
    now.set(Duration::from_secs(3));
    cancellation.cancel();
    let (cleanup, status, request) = shutdown_ready(&mut script, &limits, || {});
    assert!(script.steps.is_empty(), "script not consumed");
    ProcessReport {
        pid: 1,
        outcome: Outcome::Ready,
        ready: true,
        readiness_stop: ReadinessStop::Admitted {
            observation,
            request,
        },
        status,
        cleanup,
        elapsed: now.get(),
        stdout: StreamReport::default(),
        stderr: StreamReport::default(),
        failure: None,
    }
}

fn signalled(leader: Result<bool, Failure>, signal: Result<(), Failure>, code: i32) -> Script {
    Script {
        steps: VecDeque::from([
            Step::Leader(Ok(false)),
            Step::Active(Ok(true)),
            Step::Leader(leader),
            Step::Graceful(signal),
            Step::Active(Ok(false)),
            Step::Active(Ok(false)),
            Step::Reap,
        ]),
        status: Some(status(code)),
    }
}

#[test]
fn migration_process_observed_admission_survives_shutdown_deadline_and_later_cancel() {
    let report = run(signalled(Ok(false), Ok(()), 0));
    assert!(report.success(), "{report:?}");
    assert!(
        matches!(report.readiness_stop, ReadinessStop::Admitted { observation, request: GracefulRequest::RequestedAfterLiveObservation } if observation.stream == Stream::Stderr && observation.elapsed < Duration::from_secs(1))
    );
}

#[test]
fn migration_process_observed_pre_request_exit_cannot_use_live_descendant() {
    let report = run(signalled(Ok(true), Ok(()), 0));
    assert!(!report.success());
    assert!(report.cleanup.complete);
    assert_eq!(report.status.unwrap().code(), Some(0));
    assert!(matches!(
        report.readiness_stop,
        ReadinessStop::Admitted {
            request: GracefulRequest::LeaderExitedBeforeRequest,
            ..
        }
    ));
}

#[test]
fn migration_process_observed_final_leader_query_failure_is_not_live() {
    let report = run(signalled(Err(failure()), Ok(()), 0));
    assert!(!report.success());
    assert!(report.cleanup.complete);
    assert!(matches!(
        report.readiness_stop,
        ReadinessStop::Admitted {
            request: GracefulRequest::LeaderObservationFailed(Failure::Io { .. }),
            ..
        }
    ));
}

#[test]
fn migration_process_observed_skipped_signal_is_explicit_despite_zero_exit() {
    let report = run(Script {
        steps: VecDeque::from([
            Step::Leader(Ok(false)),
            Step::Active(Ok(false)),
            Step::Active(Ok(false)),
            Step::Reap,
        ]),
        status: Some(status(0)),
    });
    assert!(!report.success());
    assert!(!report.cleanup.graceful_signal_failed);
    assert!(matches!(
        report.readiness_stop,
        ReadinessStop::Admitted {
            request: GracefulRequest::SkippedInactiveTree,
            ..
        }
    ));
}

#[test]
fn migration_process_observed_tree_query_failure_forces_cleanup_without_request() {
    let report = run(Script {
        steps: VecDeque::from([
            Step::Leader(Ok(false)),
            Step::Active(Err(failure())),
            Step::Force,
            Step::Active(Ok(false)),
            Step::Reap,
        ]),
        status: Some(status(0)),
    });
    assert!(!report.success());
    assert!(report.cleanup.forced && report.cleanup.complete);
    assert!(matches!(
        report.readiness_stop,
        ReadinessStop::Admitted {
            request: GracefulRequest::TreeObservationFailed,
            ..
        }
    ));
}

#[test]
fn migration_process_observed_signal_failure_is_typed_even_with_clean_exit() {
    let report = run(signalled(Ok(false), Err(failure()), 0));
    assert!(!report.success());
    assert!(report.cleanup.graceful_signal_failed);
    assert!(matches!(
        report.readiness_stop,
        ReadinessStop::Admitted {
            request: GracefulRequest::RequestFailed(Failure::Io { .. }),
            ..
        }
    ));
}

#[test]
fn migration_process_observed_nonzero_stop_is_not_success() {
    let report = run(signalled(Ok(false), Ok(()), 23));
    assert!(!report.success());
    assert_eq!(report.status.unwrap().code(), Some(23));
}

#[test]
fn migration_process_observed_missing_final_status_is_not_success() {
    let mut script = signalled(Ok(false), Ok(()), 0);
    script.status = None;
    let report = run(script);
    assert!(!report.success());
    assert!(report.status.is_none());
    assert!(matches!(
        report.cleanup.failure,
        Some(Failure::CleanupDeadline)
    ));
}
