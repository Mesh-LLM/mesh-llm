use super::*;
use crate::process::{
    Completion, ObservedLine, ProbeDecision, Readiness,
    retained::{Action, Context, MemberId},
};
use std::time::Duration;
pub(crate) struct Observer(bool);
#[cfg(unix)]
pub(crate) fn blocked_observer() -> Observer {
    Observer(false)
}
#[cfg(unix)]
pub(crate) fn test_limits() -> Limits {
    limits()
}
impl Coordinator for Observer {
    type Rejection = ();
    fn line(&mut self, _: MemberId, _: ObservedLine<'_>) -> ProbeDecision<()> {
        unreachable!()
    }
    fn tick(&mut self, _: Context<'_>) -> Action<()> {
        self.0 = true;
        Action::Reject(())
    }
}
fn limits() -> Limits {
    Limits {
        execution: Duration::from_secs(1),
        graceful_shutdown: Duration::from_millis(10),
        forced_shutdown: Duration::from_millis(10),
        retained_bytes_per_stream: 0,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}
#[test]
fn signal_scope_overlap_rejects_retained_callback() {
    if std::env::var_os("RETAINED_SCOPE_TEST").is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap()).args(["--exact", "automation::retained_session::tests::signal_scope_overlap_rejects_retained_callback", "--nocapture"]).env("RETAINED_SCOPE_TEST", "1").output().unwrap();
        assert!(output.status.success(), "{output:?}");
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
        return;
    }
    let mut observer = Observer(false);
    let scope = Interrupt::install().unwrap();
    assert!(matches!(
        run(&mut observer, &limits()),
        Err(Error::Interrupt(Reason::ScopeBusy))
    ));
    scope.finish().unwrap();
    assert!(!observer.0);
}

#[test]
fn finalization_failure_preserves_preceding_process_error() {
    let preceding: Result<Report<()>, Failure> = Err(Failure::Io {
        operation: "spawn",
        kind: std::io::ErrorKind::NotFound,
        code: Some(2),
    });
    let result = finalize(preceding, Err(Reason::Interrupted));
    assert!(matches!(
        result,
        Err(Error::Finalization {
            reason: Reason::Interrupted,
            preceding: Err(Failure::Io {
                operation: "spawn",
                kind: std::io::ErrorKind::NotFound,
                code: Some(2),
            }),
        })
    ));
}

#[test]
fn restoration_failure_preserves_preceding_rejection_and_failure() {
    let preceding = retained::run(
        &mut Observer(false),
        &limits(),
        &crate::process::Cancellation::default(),
    )
    .unwrap();
    let preceding = Report {
        failure: Some(Failure::CleanupDeadline),
        ..preceding
    };
    let result = finalize(
        Ok(preceding),
        Err(Reason::Io {
            operation: "restore SIGINT",
            kind: std::io::ErrorKind::PermissionDenied,
            code: Some(1),
        }),
    );
    let Err(Error::Finalization {
        reason:
            Reason::Io {
                operation: "restore SIGINT",
                kind: std::io::ErrorKind::PermissionDenied,
                code: Some(1),
            },
        preceding: Ok(report),
    }) = result
    else {
        panic!("restoration failure must retain the rejected report");
    };
    assert_eq!(report.outcome, crate::process::Outcome::ObservationRejected);
    assert_eq!(report.rejection, Some(()));
    assert!(matches!(report.failure, Some(Failure::CleanupDeadline)));
    assert!(!report.success());
    assert!(!report.recovery_success());
}

#[test]
fn successful_finalization_preserves_preceding_process_error() {
    let preceding: Result<Report<()>, Failure> = Err(Failure::EnumerationLimit);
    let result = finalize(preceding, Ok(()));
    assert!(matches!(
        result,
        Err(Error::Process(Failure::EnumerationLimit))
    ));
}

#[test]
fn successful_finalization_preserves_preceding_report() {
    let preceding = retained::run(
        &mut Observer(false),
        &limits(),
        &crate::process::Cancellation::default(),
    )
    .unwrap();
    let report = finalize(Ok(preceding), Ok(())).unwrap();
    assert_eq!(report.outcome, crate::process::Outcome::ObservationRejected);
    assert_eq!(report.rejection, Some(()));
    assert!(!report.success());
}
