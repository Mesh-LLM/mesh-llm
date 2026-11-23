use super::*;
use crate::{LeaderState, platform};
use std::process::{Command, Stdio};

#[test]
fn forced_shutdown_captures_live_leader_before_killing_it() {
    let directory = tempfile::tempdir().unwrap();
    let shell = std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|path| path.join("sh"))
        .find(|path| path.is_file())
        .unwrap();
    let mut command = Command::new(shell);
    command
        .current_dir(directory.path())
        .args([
            "-c",
            "trap '' TERM; printf ready > ready; while :; do :; done",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut child = platform::OwnedChild::spawn_inherited(&mut command).unwrap();
    let until = Instant::now() + Duration::from_secs(10);
    while !directory.path().join("ready").is_file() && Instant::now() < until {
        thread::sleep(POLL);
    }
    assert!(
        directory.path().join("ready").is_file(),
        "signal handler ready event missing"
    );
    let limits = Limits {
        graceful_shutdown: Duration::from_millis(10),
        forced_shutdown: Duration::from_secs(10),
        ..super::tests::limits()
    };

    let (cleanup, status) = shutdown(&mut child, &limits, || {});

    assert!(cleanup.forced && cleanup.complete, "{cleanup:?}");
    let observation = cleanup
        .pre_force
        .expect("forced cleanup must capture pre-force observation");
    assert_eq!(observation.leader.unwrap(), LeaderState::Alive);
    assert!(observation.members.is_none());
    use std::os::unix::process::ExitStatusExt;
    assert_eq!(status.unwrap().signal(), Some(libc::SIGKILL));
}

#[test]
fn exited_leader_status_is_observed_without_reaping_before_force() {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "control::pre_force_tests::exit_fixture",
            "--nocapture",
        ])
        .env("PRE_FORCE_EXIT_FIXTURE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut child = platform::OwnedChild::spawn_inherited(&mut command).unwrap();
    let until = Instant::now() + Duration::from_secs(10);
    while !child.exited().unwrap() && Instant::now() < until {
        thread::sleep(POLL);
    }
    assert!(child.exited().unwrap());

    let observation = child.pre_force();

    let LeaderState::Exited(status) = observation.leader.unwrap() else {
        panic!("expected exit")
    };
    assert_eq!(status.code(), Some(23));
    assert_eq!(child.reap().unwrap(), Some(status));
}

#[test]
fn exit_fixture() {
    if std::env::var_os("PRE_FORCE_EXIT_FIXTURE").is_some() {
        std::process::exit(23);
    }
}
