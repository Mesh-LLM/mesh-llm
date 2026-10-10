//! Actual product declaration planner; payloads need not be downloaded.
#[path = "../src/layer_package_inspection/fixtures.rs"]
mod fixtures;
use std::{fs, path::Path, process::Command};
fn builder() -> &'static str {
    env!("CARGO_BIN_EXE_skippy-package-builder")
}
fn plan(manifest: &Path) -> std::process::Output {
    Command::new(builder())
        .args(["plan-layer-package-artifacts", "--manifest"])
        .arg(manifest)
        .args([
            "--stage-index",
            "0",
            "--stage-count",
            "1",
            "--layer-start",
            "0",
            "--layer-end",
            "2",
        ])
        .output()
        .unwrap()
}
#[test]
fn real_cli_plans_without_payloads_and_refuses_unsafe_duplicate_declarations() {
    let temp = tempfile::tempdir().unwrap();
    let good = fixtures::fixture(temp.path());
    for name in [
        "metadata.gguf",
        "embeddings.gguf",
        "output.gguf",
        "projector.gguf",
    ] {
        fs::remove_file(temp.path().join(name)).unwrap();
    }
    fs::remove_dir_all(temp.path().join("layers")).unwrap();
    let manifest = temp.path().join("model-package.json");
    let result = plan(&manifest);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(String::from_utf8(result.stdout).unwrap().lines().count(), 6);
    for path in ["../escape", "metadata.gguf", "layers/x\ny"] {
        let mut bad = good.clone();
        bad["layers"][1]["path"] = serde_json::json!(path);
        fixtures::save(temp.path(), &bad);
        let result = plan(&manifest);
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
    }
    let result = Command::new(builder())
        .args(["even-layer-stage-range", "1", "3", "8"])
        .output()
        .unwrap();
    assert!(result.status.success());
    assert_eq!(result.stdout, b"3 6\n");
    for args in [
        ["0", "0", "8"],
        ["0", "9", "8"],
        ["3", "3", "8"],
        ["-1", "3", "8"],
    ] {
        let result = Command::new(builder())
            .arg("even-layer-stage-range")
            .args(args)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
    }
}
#[cfg(unix)]
#[test]
fn fifo_without_writer_refuses_finitely_and_regular_blob_symlink_remains_valid() {
    use std::{
        os::unix::fs::symlink,
        process::Stdio,
        time::{Duration, Instant},
    };
    let temp = tempfile::tempdir().unwrap();
    fixtures::fixture(temp.path());
    let manifest = temp.path().join("model-package.json");
    let link = temp.path().join("blob-pointer.json");
    symlink(&manifest, &link).unwrap();
    let valid = plan(&link);
    assert!(
        valid.status.success(),
        "{}",
        String::from_utf8_lossy(&valid.stderr)
    );
    let fifo = temp.path().join("no-writer.fifo");
    assert!(
        Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success()
    );
    let stdout = temp.path().join("stdout");
    let mut child = Command::new(builder())
        .args(["plan-layer-package-artifacts", "--manifest"])
        .arg(&fifo)
        .args([
            "--stage-index",
            "0",
            "--stage-count",
            "1",
            "--layer-start",
            "0",
            "--layer-end",
            "2",
        ])
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
            panic!("manifest FIFO admission blocked before regular-file refusal");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    assert!(!status.success());
    assert!(fs::read(stdout).unwrap().is_empty());
}
