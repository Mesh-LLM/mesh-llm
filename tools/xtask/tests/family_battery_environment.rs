#![cfg(unix)]
use serde_json::Value;
use std::{fs, process::Command};

#[test]
fn actual_environment_cli_observes_local_free_bytes_and_writes_complete_receipt() {
    let state = tempfile::tempdir().unwrap();
    let output = state.path().join("environment.json");
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "family-battery-policy", "--environment"])
        .arg(state.path().join("missing/artifacts"))
        .arg(state.path().join("missing/models"))
        .arg("0")
        .arg(&output)
        .env("PATH", "")
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert!(result.stdout.is_empty());
    assert!(result.stderr.is_empty());
    let report: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(report["ports"]["allocation"], "os-assigned-at-launch");
    let entries = report["filesystems"].as_array().unwrap();
    assert_eq!(entries.len(), 2);
    for (entry, label) in entries.iter().zip(["artifacts", "models"]) {
        assert_eq!(entry["label"], label);
        assert_eq!(entry["path"], state.path().to_str().unwrap());
        assert!(entry["free_bytes"].as_u64().is_some());
        assert_eq!(entry["minimum_free_bytes"], 0);
        assert_eq!(entry["sufficient"], true);
    }
}

#[test]
fn actual_environment_cli_invalid_capacity_preserves_existing_receipt() {
    let state = tempfile::tempdir().unwrap();
    let output = state.path().join("environment.json");
    fs::write(&output, b"existing receipt").unwrap();
    for minimum in ["+1", "-1", "17179869184"] {
        let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "family-battery-policy", "--environment"])
            .arg(state.path())
            .arg(state.path())
            .arg(minimum)
            .arg(&output)
            .env("PATH", "")
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
        assert!(!result.stderr.is_empty());
        assert_eq!(fs::read(&output).unwrap(), b"existing receipt");
    }
}
