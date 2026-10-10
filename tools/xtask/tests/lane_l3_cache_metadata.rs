#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::process::Command;

#[test]
fn cache_management_rejects_effective_target_and_build_directory_mismatches() {
    for (target_name, build_name, explicit, succeeds) in [
        ("target", "other-build", "target", false),
        ("configured-target", "configured-target", "target", false),
        ("target", "target", "other-target", false),
        (
            "configured-target",
            "configured-target",
            "configured-target",
            true,
        ),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let workspace = temporary.path().join("ws");
        let bin = temporary.path().join("bin");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(workspace.join("Cargo.toml"), "[workspace]\n").unwrap();
        let metadata = serde_json::json!({
            "target_directory": workspace.join(target_name),
            "build_directory": workspace.join(build_name),
            "packages": []
        });
        let stub = bin.join("just");
        std::fs::write(&stub, format!("#!/bin/sh\nprintf '%s' '{}'\n", metadata)).unwrap();
        std::fs::set_permissions(stub, std::fs::Permissions::from_mode(0o755)).unwrap();
        let path = std::env::join_paths(std::iter::once(bin).chain(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        )))
        .unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["ci-ops", "build-cache", "status", "--workspace"])
            .arg(&workspace)
            .arg("--target-dir")
            .arg(workspace.join(explicit))
            .env("PATH", path)
            .env_remove("CARGO_BUILD_BUILD_DIR")
            .output()
            .unwrap();
        assert_eq!(
            output.status.success(),
            succeeds,
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        if !succeeds {
            assert!(!workspace.join(explicit).exists());
        }
    }
}

#[test]
fn cache_management_rejects_separate_build_directory_environment() {
    let temporary = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["ci-ops", "build-cache", "status", "--workspace"])
        .arg(temporary.path())
        .env("CARGO_BUILD_BUILD_DIR", "elsewhere")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!temporary.path().join("target").exists());
}
