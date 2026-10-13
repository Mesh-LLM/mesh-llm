use crate::{process::*, support::*};

#[test]
fn migration_process_windows_without_console_forces_owned_job_only() {
    let root = tempfile::tempdir().unwrap();
    let mut spec = spec(root.path(), "exit");
    spec.environment
        .remove(std::ffi::OsStr::new("MIGRATION_PROCESS_MODE"));
    spec.arguments = [
        "--exact",
        "windows::migration_process_windows_consoleless_fixture",
        "--nocapture",
    ]
    .into_iter()
    .map(|arg| Value::Public(arg.into()))
    .collect();
    spec.environment.insert(
        "MIGRATION_PROCESS_CONSOLELESS".into(),
        Value::Public(root.path().into()),
    );
    let report = supervise(
        &spec,
        &limits(),
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(report.success(), "{report:?}");
}

#[test]
fn migration_process_windows_consoleless_fixture() {
    let Some(root) = std::env::var_os("MIGRATION_PROCESS_CONSOLELESS") else {
        return;
    };
    let root = std::path::Path::new(&root);
    // SAFETY: only this disposable fixture detaches its own console. The parent
    // test process keeps its console and can still observe its child job.
    assert_ne!(
        unsafe { windows_sys::Win32::System::Console::FreeConsole() },
        0
    );
    let mut sentinel = Sentinel::new(root);
    let mut limits = limits();
    limits.execution = std::time::Duration::from_millis(600);
    let report = supervise(
        &spec(root, "tree-hang"),
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    sentinel.assert_alive();
    assert!(report.cleanup.graceful_signal_failed);
    assert!(report.cleanup.forced);
    assert!(report.cleanup.complete);
    assert!(!report.success());
    assert_stopped(root, &["tree-hang", "branch", "leaf"]);
}
