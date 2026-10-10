use std::process::Command;

#[test]
fn parity_cli_preserves_device_and_rejects_invalid_sources() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let executable = env!("CARGO_BIN_EXE_xtask");
    let result = Command::new(executable)
        .current_dir(&root)
        .args(["automation", "laya", "parity", "--help"])
        .output()
        .unwrap();
    assert!(result.status.success());
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .contains("--device")
    );
    let result = Command::new(executable)
        .current_dir(&root)
        .args([
            "automation",
            "laya",
            "parity",
            "--base-url",
            "http://localhost:9",
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
}

#[cfg(unix)]
#[test]
fn cli_golden_battery_runs_real_adapter_and_writes_report() {
    use std::os::unix::fs::PermissionsExt;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let state = tempfile::tempdir().unwrap();
    let cli = state.path().join("laya-fixture");
    std::fs::write(&cli, "#!/bin/sh\n[ \"$1\" = '-m' ] && [ \"$3\" = '-f' ] && [ \"$5\" = '--device' ] && [ \"$6\" = 'CPU' ] || exit 7\ncat \"$4\"\n").unwrap();
    std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o700)).unwrap();
    let output = state.path().join("report.json");
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(root)
        .args(["automation", "laya", "parity", "--cli"])
        .arg(cli)
        .args(["--gguf", "fixture.gguf", "--device", "CPU", "--json-out"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(report["passed"], true);
    assert_eq!(report["results"].as_array().unwrap().len(), 7);
}

#[cfg(unix)]
#[test]
fn product_server_early_exit_fails_without_publishing_success() {
    use std::os::unix::fs::PermissionsExt;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let state = tempfile::tempdir().unwrap();
    let binary = state.path().join("mesh-fixture");
    std::fs::write(&binary, "#!/bin/sh\nexit 7\n").unwrap();
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
    let model = state.path().join("model.gguf");
    std::fs::write(&model, b"fixture").unwrap();
    let report = state.path().join("report.json");
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(root)
        .args(["automation", "laya", "product", "--mesh-binary"])
        .arg(binary)
        .arg("--model")
        .arg(model)
        .args([
            "--device",
            "CPU",
            "--startup-timeout",
            "1",
            "--read-timeout",
            "1",
            "--json-out",
        ])
        .arg(&report)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!report.exists());
}

#[test]
fn product_lifecycle_runs_all_goldens_and_stops_the_fixture() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let executable = env!("CARGO_BIN_EXE_laya-product-fixture");
    let state = tempfile::tempdir().unwrap();
    let model = state.path().join("model.gguf");
    std::fs::write(&model, b"fixture").unwrap();
    let output = state.path().join("report.json");
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(root)
        .args(["automation", "laya", "product", "--mesh-binary"])
        .arg(executable)
        .arg("--model")
        .arg(model)
        .args([
            "--device",
            "CPU",
            "--startup-timeout",
            "5",
            "--read-timeout",
            "5",
            "--json-out",
        ])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(report["passed"], true);
    assert_eq!(report["results"].as_array().unwrap().len(), 7);
}
