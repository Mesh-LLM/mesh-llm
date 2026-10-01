use crate::{driver_expectations::*, process::*, support::*};

#[test]
fn migration_process_driver_accepts_expected_success() {
    let root = tempfile::tempdir().unwrap();
    let mut spec = spec(root.path(), "exit");
    spec.environment
        .insert("ASAN_OPTIONS".into(), Value::Public("symbolize=0".into()));
    let report = supervise(
        &spec,
        &driver_limits(),
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    let result = Scenario::parse("exit")
        .unwrap()
        .validate(&report, root.path());
    assert!(result.is_ok(), "{result:?}: {report:?}");
}

#[test]
fn migration_process_driver_accepts_expected_deadline() {
    let root = tempfile::tempdir().unwrap();
    let scenario = Scenario::parse("tree-hang").unwrap();
    let report = supervise(
        &spec(root.path(), scenario.mode()),
        &driver_limits(),
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    scenario.validate(&report, root.path()).unwrap();
    assert_stopped(root.path(), &["tree-hang", "branch", "leaf"]);
}

#[cfg(unix)]
#[test]
fn migration_process_driver_rejects_injected_tree_hang_sigkill() {
    let root = tempfile::tempdir().unwrap();
    let stage = root.path().to_path_buf();
    let worker = std::thread::spawn(move || {
        let until = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while !stage.join("leaf.ready").is_file() {
            assert!(
                std::time::Instant::now() < until,
                "fixture tree did not start"
            );
            std::thread::park_timeout(std::time::Duration::from_millis(1));
        }
        let pid: i32 = std::fs::read_to_string(stage.join("tree-hang.pid"))
            .unwrap()
            .parse()
            .unwrap();
        // SAFETY: the live leader is recorded by our unique still-running fixture.
        assert_eq!(unsafe { libc::kill(pid, libc::SIGKILL) }, 0);
    });
    let report = supervise(
        &spec(root.path(), "tree-hang"),
        &driver_limits(),
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    worker.join().unwrap();
    assert_stopped(root.path(), &["tree-hang", "branch", "leaf"]);
    assert!(report.cleanup.complete);
    assert!(Scenario::TreeHang.validate(&report, root.path()).is_err());
}

#[test]
fn migration_process_driver_rejects_incomplete_pid_roster() {
    let root = tempfile::tempdir().unwrap();
    let report = supervise(
        &spec(root.path(), "tree-exit"),
        &driver_limits(),
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    std::fs::remove_file(root.path().join("leaf.pid")).unwrap();
    assert!(Scenario::TreeExit.validate(&report, root.path()).is_err());
}
