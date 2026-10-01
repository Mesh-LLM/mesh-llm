use super::super::{command, coordinated as execution, failure::Failure, output};
use super::{support::*, *};
use crate::{automation::command_interrupt::Reason, process};

fn retained_report() -> process::retained::Report<Rejection> {
    process::retained::Report {
        outcome: process::Outcome::ObservationRejected,
        rejection: Some(Rejection::Evidence(Check::Chat)),
        failure: None,
        members: vec![process::retained::MemberReport {
            member: process::retained::MemberId::Seed,
            admitted: None,
            disposition: process::retained::Disposition::Failure(
                process::Outcome::ObservationRejected,
            ),
            process: report(71),
        }],
    }
}

#[test]
fn state_deletion_preserves_worker_failure_when_obstructed() {
    let directory = tempfile::tempdir().unwrap();
    let state =
        crate::automation::private_state::PrivateState::create(directory.path(), "required-smoke")
            .unwrap();
    let root = state
        .output_files()
        .stdout
        .unwrap()
        .parent()
        .unwrap()
        .to_owned();
    std::fs::rename(&root, root.with_extension("displaced")).unwrap();
    std::fs::write(&root, b"obstruction").unwrap();
    let preceding =
        execution::finish_workers(Ok(retained_report()), vec![(false, true), (true, true)]);
    let preceding = preceding.map(|_| panic!("worker join must reject"));
    let error = execution::finish_state(state, preceding).err().unwrap();
    assert_eq!(
        output::reason(error.as_ref()),
        Rejection::StateCleanup.to_string()
    );
    assert_eq!(output::reports(error.as_ref())[0].pid, 71);
    assert!(matches!(
        std::error::Error::source(error.as_ref())
            .unwrap()
            .downcast_ref::<Failure>(),
        Some(Failure::CoordinatedWorkers { .. })
    ));
}

#[test]
fn worker_panic_retains_report_when_join_fails() {
    let preceding = retained_report();
    let joined = std::thread::scope(|scope| {
        let daemon = scope.spawn(|| ());
        let smoke = scope.spawn(|| panic!("fixture worker panic"));
        (daemon.join().is_ok(), smoke.join().is_ok())
    });
    let error = execution::finish_workers(Ok(preceding), vec![joined, (true, true)])
        .err()
        .unwrap();
    assert_eq!(output::reports(error.as_ref())[0].pid, 71);
    match error.downcast_ref::<Failure>().unwrap() {
        Failure::CoordinatedWorkers { preceding, .. } => assert_eq!(
            preceding.as_ref().unwrap().rejection,
            Some(Rejection::Evidence(Check::Chat))
        ),
        _ => panic!("missing worker failure"),
    }
}

#[test]
fn restoration_failure_retains_completed_reports_when_no_signal_occurred() {
    let completed = completed(variant(Model::Dense))
        .finish((report(71), Some(report(72))), Ok(()))
        .unwrap();
    let result = command::finalize(
        Ok(output::Receipt::new(completed)),
        Err(Reason::Io {
            operation: "restore signal handler",
            kind: std::io::ErrorKind::PermissionDenied,
            code: Some(13),
        }),
    );
    let error = result.unwrap_err();
    assert_eq!(
        output::reason(error.as_ref()),
        "signal_scope_finalization_failed"
    );
    assert_eq!(
        output::reports(error.as_ref())
            .iter()
            .map(|report| report.pid)
            .collect::<Vec<_>>(),
        [71, 72]
    );
}

#[test]
fn interruption_retains_rejection_when_smoke_already_failed() {
    let preceding = completed(variant(Model::Dense))
        .finish((report(71), Some(report(72))), Err(Rejection::StateCleanup))
        .err()
        .unwrap();
    let error = command::finalize(Err(preceding), Err(Reason::Interrupted)).unwrap_err();
    assert_eq!(
        output::reason(error.as_ref()),
        Rejection::Cancelled.to_string()
    );
    match error.downcast_ref::<Failure>().unwrap() {
        Failure::Interrupt { preceding, .. } => assert_eq!(
            preceding
                .as_ref()
                .unwrap_err()
                .downcast_ref::<Rejected>()
                .unwrap()
                .reason,
            Rejection::StateCleanup
        ),
        _ => panic!("missing interruption failure"),
    }
    assert_eq!(output::reports(error.as_ref()).len(), 2);
}

#[test]
fn worker_failure_retains_supervisor_error_when_no_report_exists() {
    let error = execution::finish_workers(
        Err(process::Failure::EnumerationLimit),
        vec![(false, true), (true, true)],
    )
    .err()
    .unwrap();
    assert!(matches!(
        std::error::Error::source(error.as_ref())
            .unwrap()
            .downcast_ref::<process::Failure>(),
        Some(process::Failure::EnumerationLimit)
    ));
    assert!(output::reports(error.as_ref()).is_empty());
}
