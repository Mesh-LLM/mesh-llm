//! Git protocol completion admission, including post-exit drain failures.
use super::*;
#[cfg(unix)]
fn completed() -> process::ProcessReport {
    use std::os::unix::process::ExitStatusExt;
    process::ProcessReport {
        pid: 1,
        outcome: process::Outcome::Exited,
        status: Some(std::process::ExitStatus::from_raw(0)),
        ready: false,
        readiness_stop: process::ReadinessStop::default(),
        elapsed: Duration::ZERO,
        stdout: process::StreamReport::default(),
        stderr: process::StreamReport::default(),
        cleanup: process::Cleanup {
            complete: true,
            ..Default::default()
        },
        failure: None,
    }
}
#[cfg(unix)]
#[test]
fn post_exit_drain_or_cleanup_failures_never_admit_successful_git_protocol() {
    let output = completed();
    assert!(admitted_completion(&output));
    let mut drain_failed = completed();
    drain_failed.failure = Some(process::Failure::CleanupDeadline);
    assert_eq!(drain_failed.outcome, process::Outcome::Exited);
    assert!(drain_failed.status.unwrap().success() && drain_failed.cleanup.complete);
    assert!(!admitted_completion(&drain_failed));
    let mut cleanup_failed = completed();
    cleanup_failed.cleanup.failure = Some(process::Failure::CleanupDeadline);
    assert!(!admitted_completion(&cleanup_failed));
    let mut forced = completed();
    forced.cleanup.forced = true;
    assert!(!admitted_completion(&forced));
    let mut cancelled = completed();
    cancelled.outcome = process::Outcome::Cancelled;
    assert!(!admitted_completion(&cancelled));
    let mut truncated = completed();
    truncated.stdout.truncated = true;
    assert!(!admitted_completion(&truncated));
}
