use super::{super::coordinated, support::*, *};
use crate::process::{
    self,
    retained::{Disposition, MemberId, MemberReport, Report},
};
use std::time::Duration;

#[test]
fn headless_budget_starts_when_launch_occurs_after_primary_checks() {
    let mut session = before_headless(variant(Model::Dense));
    session.headless_launched(Duration::from_secs(500)).unwrap();
    let result = session.tick(Duration::from_secs(679));
    assert_eq!(result, Ok(Progress::Pending));
    assert_eq!(
        session.tick(Duration::from_secs(680)),
        Err(Rejection::Deadline(Check::HeadlessModels))
    );
}

#[test]
fn headless_retry_cannot_restart_the_launch_clock() {
    let mut session = before_headless(variant(Model::Dense));
    session.headless_launched(Duration::from_secs(500)).unwrap();
    session
        .observe(
            (Check::HeadlessModels, Transfer::Failed),
            Duration::from_secs(679),
        )
        .unwrap();
    let result = session.headless_launched(Duration::from_secs(679));
    assert_eq!(result, Err(Rejection::OutOfOrder));
}

#[test]
fn intentional_stop_recovery_cannot_certify_smoke_when_cleanup_was_forced() {
    let session = completed(variant(Model::Dense));
    let mut headless = report(42);
    headless.cleanup.forced = true;
    let report = Report {
        outcome: process::Outcome::Ready,
        rejection: None,
        failure: None,
        members: vec![
            MemberReport {
                member: MemberId::Seed,
                admitted: Some(Duration::from_secs(1)),
                disposition: Disposition::IntentionalStop,
                completion: None,
                process: report(41),
            },
            MemberReport {
                member: MemberId::WorkerOne,
                admitted: Some(Duration::from_secs(1)),
                disposition: Disposition::IntentionalStop,
                completion: None,
                process: headless,
            },
        ],
    };
    assert!(report.recovery_success());
    let error =
        coordinated::finish_report(session, report, &process::Cancellation::default()).unwrap_err();
    assert_eq!(
        output::reason(error.as_ref()),
        Rejection::ProcessCleanup.to_string()
    );
    assert_eq!(output::reports(error.as_ref()).len(), 2);
}

#[test]
fn state_cleanup_retains_completed_reports_when_directory_is_obstructed() {
    let directory = tempfile::tempdir().unwrap();
    let state =
        crate::automation::private_state::PrivateState::create(directory.path(), "overlap-state")
            .unwrap();
    state.prepare().unwrap();
    let root = state
        .output_files()
        .stdout
        .unwrap()
        .parent()
        .unwrap()
        .to_owned();
    std::fs::rename(&root, root.with_extension("displaced")).unwrap();
    std::fs::write(&root, b"obstruction").unwrap();
    let receipt = output::Receipt::new(
        completed(variant(Model::Dense))
            .finish((report(41), Some(report(42))), Ok(()))
            .unwrap(),
    );
    let error = coordinated::finish_state(state, Ok(receipt)).unwrap_err();
    assert_eq!(
        output::reason(error.as_ref()),
        Rejection::StateCleanup.to_string()
    );
    assert_eq!(
        output::reports(error.as_ref())
            .iter()
            .map(|report| report.pid)
            .collect::<Vec<_>>(),
        [41, 42]
    );
}
