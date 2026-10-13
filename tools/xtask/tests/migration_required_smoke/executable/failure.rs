use super::*;

#[cfg(unix)]
#[test]
fn headless_spawn_failure_retains_primary_when_fixture_removes_private_executable() {
    let directory = tempfile::tempdir().unwrap();
    let mut options = options(directory.path(), "headless-spawn");
    let private_binary = directory.path().join("private-smoke-fixture");
    std::fs::copy(&options.binary, &private_binary).unwrap();
    options.binary = private_binary;
    let error = command::execute(directory.path(), &options, &Cancellation::default()).unwrap_err();
    let reports = output::reports(error.as_ref());
    assert_eq!(reports.len(), 1);
    assert!(reports[0].cleanup.complete);
    match error
        .downcast_ref::<super::super::super::failure::Failure>()
        .unwrap()
    {
        super::super::super::failure::Failure::Coordinated { report } => {
            assert!(report.failure.is_some());
        }
        _ => panic!("missing headless launch failure"),
    }
}
