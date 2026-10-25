use super::{
    adapter_args,
    adapter_error::{self, Error},
    native_lint::{self, NativeFailure},
};
use crate::{automation::command_interrupt::Reason, process};

#[test]
fn rejects_arguments_when_limits_or_tool_identity_are_invalid() {
    for arguments in [
        vec!["--template", "template", "--plutil", "relative"],
        vec![
            "--template",
            "template",
            "--plutil",
            "/tool",
            "--timeout-ms",
            "0",
        ],
        vec![
            "--template",
            "template",
            "--plutil",
            "/tool",
            "--max-output-bytes",
            "16777217",
        ],
        vec![
            "--template",
            "template",
            "--plutil",
            "/tool",
            "--plutil",
            "/second",
        ],
        vec!["--template", "template", "--plutil"],
    ] {
        let arguments: Vec<String> = arguments.into_iter().map(str::to_owned).collect();
        let result = adapter_args::parse(&arguments);
        assert!(matches!(result, Err(Error::Arguments(_))));
    }
}

#[test]
fn preserves_both_errors_when_primary_and_finalization_fail() {
    let primary: Result<(), Error> = Err(Error::Arguments("primary"));
    let result = adapter_error::finish(primary, Err(Reason::Interrupted));
    assert!(
        matches!(result, Err(Error::Finalization { primary, secondary: Reason::Interrupted })
        if matches!(*primary, Error::Arguments("primary")))
    );
}

#[test]
fn rejects_success_when_finalization_fails() {
    let result = adapter_error::finish(Ok(7), Err(Reason::Interrupted));
    assert!(matches!(result, Err(Error::Interrupt(Reason::Interrupted))));
}

#[test]
fn preserves_primary_when_finalization_succeeds() {
    let primary: Result<(), Error> = Err(Error::Arguments("primary"));
    let result = adapter_error::finish(primary, Ok(()));
    assert!(matches!(result, Err(Error::Arguments("primary"))));
}

#[test]
fn returns_verified_count_when_both_operations_succeed() {
    let result = adapter_error::finish(Ok(7), Ok(()));
    assert_eq!(result.unwrap(), 7);
}

#[cfg(unix)]
#[test]
fn rejects_native_completion_when_cleanup_was_forced_or_incomplete() {
    use std::os::unix::process::ExitStatusExt;
    for cleanup in [
        process::Cleanup {
            complete: true,
            forced: true,
            ..Default::default()
        },
        process::Cleanup {
            complete: false,
            ..Default::default()
        },
        process::Cleanup {
            complete: true,
            failure: Some(process::Failure::CleanupDeadline),
            ..Default::default()
        },
    ] {
        let report = process::RawProcessReport {
            process: process::ProcessReport {
                pid: 1,
                outcome: process::Outcome::Exited,
                status: Some(std::process::ExitStatus::from_raw(0)),
                ready: false,
                readiness_stop: process::ReadinessStop::NotAdmitted,
                elapsed: std::time::Duration::ZERO,
                stdout: Default::default(),
                stderr: Default::default(),
                cleanup,
                failure: None,
            },
            stdout: None,
            stderr: None,
        };
        let result = native_lint::classify(report);
        assert!(matches!(result, Err((NativeFailure::Child { .. }, _))));
    }
}

#[cfg(unix)]
#[test]
fn rejects_native_completion_when_complete_raw_eof_is_missing() {
    use std::os::unix::process::ExitStatusExt;
    let report = process::RawProcessReport {
        process: process::ProcessReport {
            pid: 1,
            outcome: process::Outcome::Exited,
            status: Some(std::process::ExitStatus::from_raw(0)),
            ready: false,
            readiness_stop: process::ReadinessStop::NotAdmitted,
            elapsed: std::time::Duration::ZERO,
            stdout: Default::default(),
            stderr: Default::default(),
            cleanup: process::Cleanup {
                complete: true,
                ..Default::default()
            },
            failure: None,
        },
        stdout: None,
        stderr: None,
    };
    let result = native_lint::classify(report);
    assert!(matches!(result, Err((NativeFailure::Incomplete, _))));
}
