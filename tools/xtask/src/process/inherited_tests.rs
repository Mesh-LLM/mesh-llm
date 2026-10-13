use super::*;
use crate::process::Value;
use std::{collections::BTreeMap, fs, process::Command};

fn spec(script: &str, cwd: &std::path::Path) -> ProcessSpec {
    ProcessSpec {
        executable: "/bin/sh".into(),
        cwd: cwd.into(),
        arguments: vec![Value::Public("-c".into()), Value::Public(script.into())],
        environment: BTreeMap::new(),
    }
}
fn limits() -> Limits {
    Limits {
        execution: Duration::from_millis(200),
        graceful_shutdown: Duration::from_millis(200),
        forced_shutdown: Duration::from_secs(3),
        retained_bytes_per_stream: 0,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}

#[test]
fn inherited_streams_are_visible_before_completion() {
    const CHILD: &str = "MESH_INHERITED_STREAM_FIXTURE";
    if let Some(path) = std::env::var_os(CHILD) {
        let directory = std::path::PathBuf::from(path);
        let mut budget = limits();
        budget.execution = Duration::from_secs(8);
        let result = supervise_inherited(&spec("echo early-out; echo early-err >&2; while [ ! -f release ]; do /bin/sleep 0.02; done; exit 7", &directory), &budget, &Cancellation::default()).unwrap();
        assert!(result.cleanup.complete);
        assert_eq!(result.outcome, Outcome::Exited);
        assert_eq!(result.status.unwrap().code(), Some(7));
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let out = directory.path().join("out");
    let err = directory.path().join("err");
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "process::inherited::tests::inherited_streams_are_visible_before_completion",
            "--nocapture",
        ])
        .env(CHILD, directory.path())
        .stdout(fs::File::create(&out).unwrap())
        .stderr(fs::File::create(&err).unwrap())
        .spawn()
        .unwrap();
    let until = Instant::now() + Duration::from_secs(6);
    let visible = loop {
        if fs::read_to_string(&out).unwrap().contains("early-out")
            && fs::read_to_string(&err).unwrap().contains("early-err")
        {
            break true;
        }
        if Instant::now() >= until || child.try_wait().unwrap().is_some() {
            break false;
        }
        thread::sleep(Duration::from_millis(10));
    };
    let running = child.try_wait().unwrap().is_none();
    fs::write(directory.path().join("release"), "release").unwrap();
    let status = child.wait().unwrap();
    assert!(
        visible && running,
        "inherited output must reach caller while leader is blocked"
    );
    assert!(status.success());
}

#[test]
fn deadline_cancellation_and_normal_exit_clean_only_owned_tree() {
    let directory = tempfile::tempdir().unwrap();
    let mut sentinel = Command::new("/bin/sleep").arg("30").spawn().unwrap();
    let result = supervise_inherited(
        &spec(
            "trap '' TERM; printf ready > term-ready; /bin/sleep 30 & wait",
            directory.path(),
        ),
        &limits(),
        &Cancellation::default(),
    )
    .unwrap();
    assert!(
        result.cleanup.complete && result.cleanup.forced,
        "deadline cleanup={:?}, outcome={:?}, failure={:?}, elapsed={:?}, fixture_ready={}",
        result.cleanup,
        result.outcome,
        result.failure,
        result.elapsed,
        directory.path().join("term-ready").is_file(),
    );
    assert_eq!(result.outcome, Outcome::Deadline);
    assert!(sentinel.try_wait().unwrap().is_none());
    let token = Cancellation::default();
    let trigger = token.clone();
    let cancel = thread::spawn(move || {
        thread::sleep(Duration::from_millis(50));
        trigger.cancel();
    });
    let result = supervise_inherited(
        &spec("trap '' TERM; /bin/sleep 30 & wait", directory.path()),
        &limits(),
        &token,
    )
    .unwrap();
    cancel.join().unwrap();
    assert!(result.cleanup.complete);
    assert_eq!(result.outcome, Outcome::Cancelled);
    let result = supervise_inherited(
        &spec("/bin/sleep 30 & exit 7", directory.path()),
        &limits(),
        &Cancellation::default(),
    )
    .unwrap();
    assert!(result.cleanup.complete);
    assert_eq!(result.status.unwrap().code(), Some(7));
    assert!(sentinel.try_wait().unwrap().is_none());
    sentinel.kill().unwrap();
    sentinel.wait().unwrap();
}

#[test]
fn completed_leader_at_expired_deadline_preserves_actual_exit_and_ignores_zombie_group() {
    for code in [0, 7] {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", &format!("exit {code}")]);
        command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let mut child = platform::OwnedChild::spawn_inherited(&mut command).unwrap();
        let until = Instant::now() + Duration::from_secs(3);
        while !child.exited().unwrap() && Instant::now() < until {
            thread::sleep(Duration::from_millis(5));
        }
        assert!(
            child.exited().unwrap(),
            "fixture leader must exit before deadline observation"
        );
        let started = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        let (outcome, failure) = monitor(&mut child, &limits(), &Cancellation::default(), started);
        let (cleanup, status) = control::shutdown(&mut child, &limits(), || {});
        assert_eq!(outcome, Outcome::Exited);
        assert!(failure.is_none());
        assert!(cleanup.complete && !cleanup.forced && cleanup.failure.is_none());
        assert!(!cleanup.graceful_signal_failed);
        assert_eq!(status.unwrap().code(), Some(code));
    }
}

#[test]
fn cancellation_after_spawn_keeps_cleanup_ownership_and_stops_descendant_writes() {
    let directory = tempfile::tempdir().unwrap();
    let mut sentinel = Command::new("/bin/sleep").arg("30").spawn().unwrap();
    let mut command = Command::new("/bin/sh");
    command.current_dir(directory.path()).args([
        "-c",
        "trap '' TERM; i=0; while :; do printf '%s' \"$i\" > writes; i=$((i+1)); /bin/sleep 0.01; done",
    ]);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut child = platform::OwnedChild::spawn_inherited(&mut command).unwrap();
    let until = Instant::now() + Duration::from_secs(3);
    while !directory.path().join("writes").is_file() && Instant::now() < until {
        thread::sleep(Duration::from_millis(5));
    }
    let ready = directory.path().join("writes").is_file();
    // Cancellation recorded while ownership is acquired cannot perform cleanup
    // inside spawn or wait. The monitor reports it, then the owning scope cleans.
    let token = Cancellation::default();
    token.cancel();
    let (outcome, failure) = monitor(&mut child, &limits(), &token, Instant::now());
    let (cleanup, _) = control::shutdown(&mut child, &limits(), || token.cancel());
    let sentinel_alive = sentinel.try_wait().unwrap().is_none();
    sentinel.kill().unwrap();
    sentinel.wait().unwrap();
    assert!(ready && sentinel_alive);
    assert_eq!(outcome, Outcome::Cancelled);
    assert!(failure.is_none());
    assert!(cleanup.complete && cleanup.forced && cleanup.failure.is_none());
    let before = fs::read(directory.path().join("writes")).unwrap();
    thread::sleep(Duration::from_millis(100));
    assert_eq!(fs::read(directory.path().join("writes")).unwrap(), before);
}
