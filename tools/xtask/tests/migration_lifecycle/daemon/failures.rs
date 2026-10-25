use super::{
    cli::{case, command},
    protocol::{Behavior, Plan},
};
use crate::support::{Sentinel, assert_absent};

#[test]
fn d20_state_deletion_obstruction_fails_after_clean_stop() {
    let case = case(&Plan {
        behavior: Behavior::DeleteFailure,
        ..Plan::default()
    });
    let mut sentinel = Sentinel::new(&case);
    let output = command(&case).output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("private state deletion failed"));
    assert!(case.native.join("handler").exists());
    assert_absent(case.audit().pid);
    assert!(sentinel.0.try_wait().unwrap().is_none());
}

#[test]
fn d24_result_write_failure_is_nonzero_after_cleanup() {
    let case = case(&Plan::default());
    let mut sentinel = Sentinel::new(&case);
    let read_only = std::fs::File::open(case.native.join("daemon.json")).unwrap();
    let output = command(&case).stdout(read_only).output().unwrap();
    assert!(
        !output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("write readiness result"));
    assert!(sentinel.0.try_wait().unwrap().is_none());
    case.assert_removed();
}

#[test]
fn d20_spawn_failure_cleans_state() {
    let case = case(&Plan::default());
    let mut sentinel = Sentinel::new(&case);
    let invalid = case.root.path().join("invalid-executable");
    std::fs::write(&invalid, b"not an executable").unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&invalid, std::fs::Permissions::from_mode(0o700)).unwrap();
    let mut arguments = case.arguments();
    arguments[1] = invalid.to_str().unwrap().into();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "daemon-readiness"])
        .args(arguments)
        .current_dir(crate::support::repository())
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(!case.native.join("audit.json").exists());
    assert_eq!(std::fs::read_dir(&case.state).unwrap().count(), 0);
    assert!(sentinel.0.try_wait().unwrap().is_none());
}

#[test]
fn d21_stale_owner_and_other_private_root_are_preserved() {
    let case = case(&Plan {
        record: String::new(),
        ..Plan::default()
    });
    let mut sentinel = Sentinel::new(&case);
    let stale = case.state.join("mld-state.stale");
    std::fs::create_dir(&stale).unwrap();
    let owner = format!("{{\"pid\":{},\"request_id\":\"old\"}}", sentinel.0.id());
    std::fs::write(stale.join("owner.json"), &owner).unwrap();
    std::fs::write(stale.join("stdout.log"), Plan::default().record).unwrap();
    let output = command(&case).output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("attribution_unavailable"));
    assert_eq!(
        std::fs::read_to_string(stale.join("owner.json")).unwrap(),
        owner
    );
    assert_eq!(std::fs::read_dir(&case.state).unwrap().count(), 1);
    assert_absent(case.audit().pid);
    assert!(sentinel.0.try_wait().unwrap().is_none());
}

#[test]
fn d14_absolute_five_second_models_timeout() {
    let case = case(&Plan {
        behavior: Behavior::SlowBody,
        ..Plan::default()
    });
    let mut sentinel = Sentinel::new(&case);
    let mut arguments = case.arguments();
    arguments[5] = "12".into();
    let started = std::time::Instant::now();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "daemon-readiness"])
        .args(arguments)
        .current_dir(crate::support::repository())
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("models_transfer_failed"));
    assert!(started.elapsed() < std::time::Duration::from_secs(10));
    assert!(sentinel.0.try_wait().unwrap().is_none());
    case.assert_removed();
}

#[test]
fn d24_help_and_bad_option_do_not_spawn() {
    let case = case(&Plan::default());
    for (arguments, success) in [(vec!["--help"], true), (vec!["--unknown", "x"], false)] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "daemon-readiness"])
            .args(arguments)
            .current_dir(crate::support::repository())
            .output()
            .unwrap();
        assert_eq!(output.status.success(), success);
    }
    assert!(!case.native.join("audit.json").exists());
    assert_eq!(std::fs::read_dir(&case.state).unwrap().count(), 0);
}
