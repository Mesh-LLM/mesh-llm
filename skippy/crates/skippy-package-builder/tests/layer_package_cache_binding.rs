use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
const COMMIT: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
fn fixture(home: &Path, revision: &str) -> PathBuf {
    let repo = home.join("hub/models--org--repo");
    let snapshot = repo.join("snapshots").join(COMMIT);
    fs::create_dir_all(&snapshot).unwrap();
    fs::write(
        snapshot.join("model-package.json"),
        br#"{"layer_count":2,"activation_width":4096,"model_id":"fixture"}"#,
    )
    .unwrap();
    let reference = repo.join("refs").join(revision);
    fs::create_dir_all(reference.parent().unwrap()).unwrap();
    fs::write(reference, COMMIT).unwrap();
    fs::create_dir_all(repo.join("snapshots/ffffffffffffffffffffffffffffffffffffffff")).unwrap();
    snapshot
}
fn builder() -> &'static str {
    env!("CARGO_BIN_EXE_skippy-package-builder")
}
fn command(home: &Path, revision: &str) -> Command {
    let mut cmd = Command::new(builder());
    cmd.args(["resolve-layer-package-cache", "--reference"])
        .arg(format!("hf://org/repo@{revision}"))
        .arg("--cache-root")
        .arg(home.join("hub"));
    cmd
}
#[test]
fn actual_cli_returns_exact_snapshot_and_missing_or_malformed_ref_has_no_output() {
    let home = tempfile::tempdir().unwrap();
    let snapshot = fixture(home.path(), "release/étape#v2%");
    let success = command(home.path(), "release/étape#v2%").output().unwrap();
    assert!(
        success.status.success(),
        "{}",
        String::from_utf8_lossy(&success.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&success.stdout).unwrap();
    assert_eq!(report["commit"], COMMIT);
    assert_eq!(report["requested_revision"], "release/étape#v2%");
    assert_eq!(
        report["snapshot_path"],
        snapshot.canonicalize().unwrap().to_str().unwrap()
    );
    for revision in ["missing", "malformed"] {
        if revision == "malformed" {
            fs::write(
                home.path().join("hub/models--org--repo/refs/malformed"),
                "../foreign",
            )
            .unwrap();
        }
        let result = command(home.path(), revision).output().unwrap();
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
    }
}
#[cfg(unix)]
#[test]
fn no_writer_fifo_ref_refuses_with_finite_child_cleanup() {
    use std::{
        process::Stdio,
        time::{Duration, Instant},
    };
    let home = tempfile::tempdir().unwrap();
    fixture(home.path(), "main");
    let fifo = home.path().join("hub/models--org--repo/refs/fifo");
    assert!(Command::new("mkfifo").arg(fifo).status().unwrap().success());
    let stdout = home.path().join("stdout");
    let mut child = command(home.path(), "fifo")
        .stdout(fs::File::create(&stdout).unwrap())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("cache ref FIFO admission blocked");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    assert!(!status.success());
    assert!(fs::read(stdout).unwrap().is_empty());
}
