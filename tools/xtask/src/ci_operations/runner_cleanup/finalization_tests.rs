use super::{Error, error::Failure, finalization::finalize};
use crate::{command_interrupt::Reason, process};
use std::{io, time::Duration};

fn cancelled() -> process::ProcessReport {
    process::ProcessReport {
        pid: 731,
        outcome: process::Outcome::Cancelled,
        status: None,
        ready: false,
        readiness_stop: process::ReadinessStop::NotAdmitted,
        elapsed: Duration::from_millis(17),
        stdout: process::StreamReport {
            bytes_seen: 19,
            bytes_retained: b"private-git-payload".to_vec(),
            truncated: true,
            suppressed_lines: 2,
        },
        stderr: process::StreamReport::default(),
        cleanup: process::Cleanup {
            complete: true,
            forced: true,
            ..process::Cleanup::default()
        },
        failure: Some(process::Failure::CleanupDeadline),
    }
}

#[test]
fn git_cancellation_when_finish_interrupts_retains_complete_report() {
    let primary = Err::<(), _>(Error::Git(Box::new(cancelled())));

    let result = finalize(primary, Err(Reason::Interrupted));

    let Failure::Finalization { reason, preceding } = result.unwrap_err() else {
        panic!("both outcomes must survive");
    };
    assert!(matches!(reason, Reason::Interrupted));
    let Error::Git(report) = preceding.unwrap_err() else {
        panic!("Git report must survive");
    };
    assert_eq!(report.pid, 731);
    assert_eq!(report.outcome, process::Outcome::Cancelled);
    assert_eq!(report.elapsed, Duration::from_millis(17));
    assert_eq!(report.stdout.bytes_retained, b"private-git-payload");
    assert_eq!(report.stdout.bytes_seen, 19);
    assert!(report.stdout.truncated);
    assert_eq!(report.stdout.suppressed_lines, 2);
    assert!(report.cleanup.complete && report.cleanup.forced);
    assert!(matches!(
        report.failure,
        Some(process::Failure::CleanupDeadline)
    ));
}

#[test]
fn io_primary_when_restoration_fails_retains_original_error() {
    let primary = Err::<(), _>(Error::Io(io::Error::from_raw_os_error(13)));
    let finish = Err(Reason::Io {
        operation: "restore signal handler",
        kind: io::ErrorKind::PermissionDenied,
        code: Some(1),
    });

    let result = finalize(primary, finish);

    let Failure::Finalization { reason, preceding } = result.unwrap_err() else {
        panic!("restoration and primary failures must survive");
    };
    assert!(matches!(reason, Reason::Io { code: Some(1), .. }));
    let Error::Io(error) = preceding.unwrap_err() else {
        panic!("I/O primary must survive");
    };
    assert_eq!(error.raw_os_error(), Some(13));
}

#[cfg(unix)]
#[test]
fn nonzero_git_when_finish_interrupts_retains_exit_status() {
    use std::os::unix::process::ExitStatusExt;
    let mut report = cancelled();
    report.outcome = process::Outcome::Exited;
    report.status = Some(std::process::ExitStatus::from_raw(23 << 8));
    let primary = Err::<(), _>(Error::Git(Box::new(report)));

    let result = finalize(primary, Err(Reason::Interrupted));

    let Failure::Finalization { preceding, .. } = result.unwrap_err() else {
        panic!("nonzero Git and interruption must survive");
    };
    let Error::Git(report) = preceding.unwrap_err() else {
        panic!("nonzero Git report must survive");
    };
    assert_eq!(report.status.unwrap().code(), Some(23));
    assert!(report.cleanup.complete);
}

#[test]
fn success_when_finish_interrupts_retains_success_payload() {
    let primary = Ok::<_, Error>(vec![0, 255, 17]);

    let result = finalize(primary, Err(Reason::Interrupted));

    let Failure::Finalization { reason, preceding } = result.unwrap_err() else {
        panic!("successful primary must survive failed finish");
    };
    assert!(matches!(reason, Reason::Interrupted));
    assert_eq!(preceding.unwrap(), vec![0, 255, 17]);
}

#[test]
fn primary_error_when_finish_succeeds_keeps_single_error_behavior() {
    let primary = Err::<(), _>(Error::Io(io::Error::from_raw_os_error(13)));

    let result = finalize(primary, Ok(()));

    let Failure::Operation(Error::Io(error)) = result.unwrap_err() else {
        panic!("ordinary error must remain unwrapped by finalization");
    };
    assert_eq!(error.raw_os_error(), Some(13));
}

#[test]
fn success_when_finish_succeeds_returns_original_payload() {
    let primary = Ok::<_, Error>(vec![0, 255, 17]);

    let result = finalize(primary, Ok(()));

    assert_eq!(result.unwrap(), vec![0, 255, 17]);
}

#[test]
fn diagnostics_when_git_finish_fails_preserve_metadata_without_raw_payload() {
    let primary = Err::<(), _>(Error::Git(Box::new(cancelled())));

    let error = finalize(primary, Err(Reason::Interrupted)).unwrap_err();

    for text in [error.to_string(), format!("{error:?}")] {
        assert!(text.contains("Cancelled"), "{text}");
        assert!(text.contains("731"), "{text}");
        assert!(text.contains("complete: true"), "{text}");
        assert!(!text.contains("private-git-payload"), "{text}");
        assert!(!text.contains("bytes_retained"), "{text}");
    }
}

#[test]
fn diagnostics_when_io_finish_fails_do_not_expose_primary_message() {
    let primary = Err::<(), _>(Error::Io(io::Error::other("private-io-payload")));

    let error = finalize(primary, Err(Reason::Interrupted)).unwrap_err();

    for text in [error.to_string(), format!("{error:?}")] {
        assert!(text.contains("Io(kind=Other"), "{text}");
        assert!(!text.contains("private-io-payload"), "{text}");
    }
    let Failure::Finalization { preceding, .. } = error else {
        panic!("I/O primary must remain typed");
    };
    assert_eq!(preceding.unwrap_err().to_string(), "private-io-payload");
}

#[test]
fn diagnostics_when_success_finish_fails_do_not_expose_success_payload() {
    let primary = Ok::<_, Error>("private-success-payload");

    let error = finalize(primary, Err(Reason::Interrupted)).unwrap_err();

    for text in [error.to_string(), format!("{error:?}")] {
        assert!(text.contains("Ok(retained)"), "{text}");
        assert!(!text.contains("private-success-payload"), "{text}");
    }
}
