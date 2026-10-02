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
        &spec("trap '' TERM; /bin/sleep 30 & wait", directory.path()),
        &limits(),
        &Cancellation::default(),
    )
    .unwrap();
    assert!(result.cleanup.complete && result.cleanup.forced);
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
