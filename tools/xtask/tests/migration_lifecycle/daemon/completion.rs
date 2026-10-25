use super::*;
use crate::process::{Cleanup, Failure, GracefulRequest, Outcome, ProcessReport, ReadinessStop};
use observer::{Facts, Phase};

fn teardown() -> Result<(), Reason> {
    Err(Reason::Io {
        operation: "injected interruption teardown",
        kind: io::ErrorKind::Other,
        code: Some(73),
    })
}

fn report() -> process::ProbeReport<Rejection> {
    #[cfg(unix)]
    use std::os::unix::process::ExitStatusExt;
    #[cfg(windows)]
    use std::os::windows::process::ExitStatusExt;
    process::ProbeReport {
        rejection: None,
        process: ProcessReport {
            pid: 42,
            outcome: Outcome::Ready,
            status: Some(std::process::ExitStatus::from_raw(0)),
            ready: true,
            readiness_stop: ReadinessStop::ProbeAdmitted {
                elapsed: Duration::from_millis(1),
                request: GracefulRequest::RequestedAfterLiveObservation,
            },
            elapsed: Duration::from_secs(1),
            stdout: process::StreamReport::default(),
            stderr: process::StreamReport::default(),
            cleanup: Cleanup {
                complete: true,
                ..Cleanup::default()
            },
            failure: None,
        },
    }
}

fn accepted() -> Result<(Ports, receipt::Accepted), Error> {
    let facts = Facts {
        phase: Phase::Complete,
        status: Some(http::Transfer {
            status_code: 200,
            body_bytes: 64,
        }),
        models: Some(http::Transfer {
            status_code: 200,
            body_bytes: 0,
        }),
        status_attempts: 1,
    };
    receipt::accept(report(), facts, (false, true)).map(|accepted| {
        (
            Ports {
                api: 1234,
                console: 1235,
                quic: 1236,
            },
            accepted,
        )
    })
}

fn rejected() -> Error {
    let mut report = report();
    report.rejection = Some(Rejection::MalformedStatus);
    report.process.outcome = Outcome::ObservationRejected;
    report.process.ready = false;
    report.process.readiness_stop = ReadinessStop::NotAdmitted;
    let facts = Facts {
        phase: Phase::StatusInFlight,
        status: None,
        models: None,
        status_attempts: 1,
    };
    receipt::accept(report, facts, (false, true)).unwrap_err()
}

fn deletion(preceding: Option<Error>) -> Error {
    Error::StateDeletion {
        kind: io::ErrorKind::NotADirectory,
        code: Some(20),
        preceding: preceding.map(Box::new),
    }
}

fn preceding(error: Error) -> Error {
    let Error::Interruption { preceding, cause } = error else {
        panic!("both failures must be retained: {error:?}");
    };
    assert!(matches!(
        *cause,
        Error::Io {
            operation: "injected interruption teardown",
            kind: io::ErrorKind::Other,
            code: Some(73),
        }
    ));
    *preceding
}

#[test]
fn supervision_failure_and_teardown_failure_are_retained() {
    let result = Err(Error::Process(Failure::Io {
        operation: "injected supervision failure",
        kind: io::ErrorKind::PermissionDenied,
        code: Some(13),
    }));
    let error = finish(result, teardown()).unwrap_err();
    assert!(matches!(
        preceding(error),
        Error::Process(Failure::Io {
            operation: "injected supervision failure",
            kind: io::ErrorKind::PermissionDenied,
            code: Some(13),
        })
    ));
}

#[test]
fn lifecycle_failure_and_teardown_failure_retain_process_history() {
    let result = Err(rejected());
    let error = finish(result, teardown()).unwrap_err();
    let diagnostic = error.to_string();
    assert!(diagnostic.contains("injected interruption teardown"));
    assert!(diagnostic.contains("malformed_status"));
    assert!(diagnostic.contains("ObservationRejected"));
    assert!(diagnostic.contains("NotAdmitted"));
    assert!(matches!(preceding(error), Error::Lifecycle(_)));
}

#[test]
fn state_deletion_failure_and_teardown_failure_are_retained() {
    let result = Err(deletion(None));
    let error = finish(result, teardown()).unwrap_err();
    assert!(matches!(
        preceding(error),
        Error::StateDeletion {
            kind: io::ErrorKind::NotADirectory,
            code: Some(20),
            preceding: None,
        }
    ));
}

#[test]
fn lifecycle_state_deletion_and_teardown_failures_are_all_retained() {
    let result = Err(deletion(Some(rejected())));
    let error = finish(result, teardown()).unwrap_err();
    let Error::StateDeletion {
        kind: io::ErrorKind::NotADirectory,
        code: Some(20),
        preceding: Some(prior),
    } = preceding(error)
    else {
        panic!("state deletion failure must retain its preceding failure");
    };
    assert!(matches!(*prior, Error::Lifecycle(_)));
    assert!(prior.to_string().contains("ObservationRejected"));
}

#[test]
fn supervision_failure_alone_is_unchanged() {
    let result = Err(Error::Process(Failure::EnumerationLimit));
    let error = finish(result, Ok(())).unwrap_err();
    assert!(matches!(error, Error::Process(Failure::EnumerationLimit)));
}

#[test]
fn state_deletion_failure_alone_is_unchanged() {
    let result = Err(deletion(None));
    let error = finish(result, Ok(())).unwrap_err();
    assert!(matches!(
        error,
        Error::StateDeletion {
            kind: io::ErrorKind::NotADirectory,
            code: Some(20),
            preceding: None,
        }
    ));
}

#[test]
fn teardown_failure_alone_keeps_successful_history() {
    let result = accepted();
    let error = finish(result, teardown()).unwrap_err();
    let Error::Completion { cause, receipt } = error else {
        panic!("successful history must accompany the teardown failure");
    };
    assert!(matches!(*cause, Error::Io { code: Some(73), .. }));
    assert!(receipt.to_string().contains("outcome=Ready"));
    assert!(receipt.to_string().contains("ProbeAdmitted"));
}

#[test]
fn successful_completion_preserves_selected_ports() {
    let result = accepted();
    let ports = finish(result, Ok(())).unwrap();
    assert_eq!((ports.api, ports.console, ports.quic), (1234, 1235, 1236));
}
