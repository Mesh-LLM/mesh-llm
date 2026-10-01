use super::*;
use crate::process::{Completion, Outcome, ProcessReport, Readiness, StreamReport};

struct StuckTree {
    force_error: bool,
    reap_calls: usize,
}

impl Leader for StuckTree {
    fn exited(&mut self) -> Result<bool, Failure> {
        panic!("ordinary cleanup must not query the leader")
    }
}

impl Tree for StuckTree {
    fn active(&mut self) -> Result<bool, Failure> {
        Ok(true)
    }
    fn graceful(&mut self) -> Result<(), Failure> {
        Ok(())
    }
    fn force(&mut self) -> Result<(), Failure> {
        if self.force_error {
            Err(Failure::io(
                "injected kill failure",
                std::io::ErrorKind::PermissionDenied.into(),
            ))
        } else {
            Ok(())
        }
    }
    fn reap(&mut self) -> Result<Option<ExitStatus>, Failure> {
        self.reap_calls += 1;
        Ok(None)
    }
}

fn limits() -> Limits {
    Limits {
        execution: Duration::from_secs(1),
        graceful_shutdown: Duration::from_millis(1),
        forced_shutdown: Duration::from_millis(1),
        retained_bytes_per_stream: 0,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}

#[test]
fn migration_process_cleanup_error_is_not_success_and_retains_wait_ownership() {
    let mut tree = StuckTree {
        force_error: true,
        reap_calls: 0,
    };
    let (cleanup, status) = shutdown(&mut tree, &limits(), || {});
    let report = ProcessReport {
        pid: 1,
        outcome: Outcome::Exited,
        status,
        ready: false,
        readiness_stop: Default::default(),
        elapsed: Duration::ZERO,
        stdout: StreamReport::default(),
        stderr: StreamReport::default(),
        cleanup,
        failure: None,
    };
    assert!(!report.success());
    assert!(!report.cleanup.complete);
    assert!(matches!(
        report.cleanup.failure,
        Some(Failure::Io {
            kind: std::io::ErrorKind::PermissionDenied,
            ..
        })
    ));
    assert_eq!(tree.reap_calls, 0);
}

#[test]
fn migration_process_successful_kill_with_survivors_is_not_cleanup_proof() {
    let mut tree = StuckTree {
        force_error: false,
        reap_calls: 0,
    };
    let (cleanup, _) = shutdown(&mut tree, &limits(), || {});
    assert!(!cleanup.complete);
    assert!(matches!(cleanup.failure, Some(Failure::CleanupDeadline)));
    assert_eq!(tree.reap_calls, 0);
}
