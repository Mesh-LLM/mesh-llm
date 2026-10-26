//! Actual local product command, with declared byte fixtures and no ABI or network invocation.
#[path = "../src/layer_package_inspection/fixtures.rs"]
mod fixtures;
use std::{fs, process::Command};
#[test]
fn actual_cli_checks_full_cache_and_refuses_missing_projector_before_output() {
    let temp = tempfile::tempdir().unwrap();
    fixtures::fixture(temp.path());
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_skippy-package-builder"))
            .args([
                "inspect-layer-package",
                "--package",
                temp.path().to_str().unwrap(),
                "--expected-layer-count",
                "2",
                "--expected-activation-width",
                "4096",
            ])
            .output()
            .unwrap()
    };
    let okay = run();
    assert!(
        okay.status.success(),
        "{}",
        String::from_utf8_lossy(&okay.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&okay.stdout).unwrap();
    assert_eq!(report["artifact_count"], 6);
    fs::remove_file(temp.path().join("projector.gguf")).unwrap();
    let refused = run();
    assert!(!refused.status.success());
    assert!(refused.stdout.is_empty());
}
#[test]
fn actual_cli_requires_closed_local_inspection_arguments() {
    for args in [
        vec!["inspect-layer-package"],
        vec![
            "inspect-layer-package",
            "--package",
            "missing",
            "--stage-index",
            "1",
        ],
        vec![
            "inspect-layer-package",
            "--package",
            "missing",
            "--expected-layer-count",
            "-1",
        ],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_skippy-package-builder"))
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
    }
}
