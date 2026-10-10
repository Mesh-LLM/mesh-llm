use super::{capture, error::Error, support};
use crate::process::{Cancellation, Outcome};
use std::time::Duration;

#[test]
fn swift_checksum_refuses_spawn_when_interrupted_before_spawn() {
    let root = tempfile::tempdir().unwrap();
    let native = support::native(root.path(), "success");
    let cancellation = Cancellation::default();
    cancellation.cancel();
    let result = capture::compute(&native, &cancellation);
    assert!(matches!(result, Err(Error::Process(_))));
    assert!(!root.path().join("argv.bin").exists());
}

#[cfg(unix)]
#[test]
fn swift_checksum_cleans_descendant_when_timeout_preserves_sentinel() {
    let root = tempfile::tempdir().unwrap();
    let mut native = support::native(root.path(), "tree");
    native.timeout = Duration::from_secs(1);
    let mut sentinel = support::Sentinel::start(root.path());
    let result = capture::compute(&native, &Cancellation::default());
    let Err(Error::Native(report)) = result else {
        panic!("expected deadline process receipt");
    };
    assert_eq!(report.process.outcome, Outcome::Deadline);
    assert!(report.process.cleanup.complete);
    let pid = std::fs::read_to_string(root.path().join("descendant.pid")).unwrap();
    let alive = std::process::Command::new("/bin/kill")
        .args(["-0", &pid])
        .status()
        .unwrap();
    assert!(!alive.success());
    sentinel.assert_alive();
}

#[cfg(unix)]
#[test]
fn swift_checksum_cleans_descendant_when_cancelled_during_capture() {
    let root = tempfile::tempdir().unwrap();
    let native = support::native(root.path(), "tree");
    let cancellation = Cancellation::default();
    let mut sentinel = support::Sentinel::start(root.path());
    let result = std::thread::scope(|scope| {
        scope.spawn(|| {
            support::wait_for(&root.path().join("ready"));
            cancellation.cancel();
        });
        capture::compute(&native, &cancellation)
    });
    let Err(Error::Native(report)) = result else {
        panic!("expected cancelled process receipt");
    };
    assert_eq!(report.process.outcome, Outcome::Cancelled);
    assert!(report.process.cleanup.complete);
    let pid = std::fs::read_to_string(root.path().join("descendant.pid")).unwrap();
    let alive = std::process::Command::new("/bin/kill")
        .args(["-0", &pid])
        .status()
        .unwrap();
    assert!(!alive.success());
    sentinel.assert_alive();
}
