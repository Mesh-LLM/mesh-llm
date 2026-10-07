use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[cfg(unix)]
#[path = "migration_cargo_metadata/interruption.rs"]
mod interruption;

fn fixture() -> PathBuf {
    match std::env::var_os("MIGRATION_TEST_CARGO") {
        Some(path) => PathBuf::from(path),
        None => std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples")
            .join(format!(
                "migration_cargo_metadata_fixture{}",
                std::env::consts::EXE_SUFFIX
            )),
    }
}

fn workspace(mode: &str) -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(root.path().join("tools/xtask")).unwrap();
    std::fs::write(root.path().join("Cargo.toml"), b"workspace marker").unwrap();
    std::fs::write(root.path().join("tools/xtask/Cargo.toml"), b"marker").unwrap();
    std::fs::write(root.path().join("mode"), mode).unwrap();
    let names: Vec<String> = serde_json::from_str(include_str!(
        "../../../scripts/tests/fixtures/ci-source-layout/extracted-packages.json"
    ))
    .unwrap();
    let packages = names.iter().map(|name| serde_json::json!({
        "id": name, "name": name, "version": "0.1.0", "manifest_path": "Cargo.toml",
        "description": format!("password token secret authorization invite {}", "x".repeat(9000)),
    })).chain(std::iter::once(serde_json::json!({"id":"external-id", "name":"--invalid-nonmember", "version":"0.1.0", "manifest_path":"Cargo.toml"}))).collect::<Vec<_>>();
    let payload =
        serde_json::to_vec(&serde_json::json!({"workspace_members":names, "packages":packages}))
            .unwrap();
    std::fs::write(root.path().join("payload.json"), payload).unwrap();
    root
}

fn invoke(root: &Path, crates: &str) -> Output {
    invoke_cargo(root, crates, &fixture())
}

fn invoke_cargo(root: &Path, crates: &str, cargo: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["--repo-root"])
        .arg(root)
        .args([
            "repository",
            "cargo-packages",
            "--generation",
            "legacy",
            "--crates",
            crates,
            "--cargo",
        ])
        .arg(cargo)
        .args(["--timeout", "5"])
        .env("GH_TOKEN", "fixture-private-secret")
        .env("GIT_DIR", "invalid-git-override")
        .env("GIT_CONFIG_COUNT", "3")
        .env("UNRELATED", "fixture-private-unrelated")
        .env("CARGO_NET_OFFLINE", "true")
        .current_dir(std::env::temp_dir())
        .output()
        .unwrap()
}

#[derive(Deserialize)]
struct Invocation {
    argv: Vec<String>,
    cwd: PathBuf,
    environment: BTreeMap<String, String>,
}

#[test]
fn selected_root_and_fixed_argv_when_real_mode_discovers_metadata() {
    let root = workspace("success");
    let output = invoke(root.path(), "[\"model-hf\"]");
    assert!(output.status.success(), "{:?}", output);
    assert_eq!(output.stdout, b"[\"skippy-model-hf\", \"skippy-hf-hub\"]\n");
    let record: Invocation =
        serde_json::from_slice(&std::fs::read(root.path().join("invocation.json")).unwrap())
            .unwrap();
    assert_eq!(record.cwd, root.path().canonicalize().unwrap());
    assert_eq!(
        record.argv,
        ["metadata", "--locked", "--no-deps", "--format-version=1"]
    );
    assert_eq!(
        record
            .environment
            .get("CARGO_NET_OFFLINE")
            .map(String::as_str),
        Some("true")
    );
    for name in ["GH_TOKEN", "GIT_DIR", "GIT_CONFIG_COUNT", "UNRELATED"] {
        assert!(!record.environment.contains_key(name));
    }
    assert!(!String::from_utf8_lossy(&output.stderr).contains("fixture-private"));
}

#[test]
fn every_mapping_when_owned_metadata_has_all_extracted_members() {
    let root = workspace("success");
    for (old, expected) in [
        ("mesh-llm-gpu-bench", vec!["skippy-gpu-bench"]),
        ("mesh-llm-guardrails", vec!["skippy-guardrails"]),
        ("mesh-llm-hardware-profile", vec!["skippy-hardware-profile"]),
        ("mesh-llm-native-runtime", vec!["skippy-native-runtime"]),
        ("mesh-llm-runtime-install", vec!["skippy-runtime-install"]),
        ("model-artifact", vec!["skippy-model-artifact"]),
        ("model-hf", vec!["skippy-model-hf", "skippy-hf-hub"]),
        ("model-package", vec!["skippy-model-package"]),
        ("model-ref", vec!["skippy-model-ref"]),
        ("model-resolver", vec!["skippy-model-resolver"]),
        ("openai-frontend", vec!["skippy-inference-api"]),
        ("skippy-model-package", vec!["skippy-package-builder"]),
        (
            "skippy-server",
            vec![
                "skippy-serving",
                "skippy-api",
                "skippy-cli",
                "skippy-commands",
                "skippy-config",
                "skippy-events",
            ],
        ),
        (
            "mesh-llm-host-runtime",
            vec![
                "mesh-llm-host-runtime",
                "mesh-llm-skippy-adapter",
                "mesh-llm-control-api",
                "mesh-llm-membership",
                "mesh-llm-transport",
            ],
        ),
    ] {
        let output = invoke(root.path(), &format!("[\"{old}\"]"));
        assert!(output.status.success(), "{old}: {output:?}");
        assert_eq!(
            serde_json::from_slice::<Vec<String>>(&output.stdout).unwrap(),
            expected
        );
    }
}

#[test]
fn failure_has_no_translation_or_raw_leak_when_child_crashes_overflows_or_json_is_invalid() {
    for (mode, diagnostic) in [
        ("crash", "exit=Some(17)"),
        ("overflow", "exceeded 16777216 bytes"),
        ("invalid", "invalid Cargo metadata JSON"),
    ] {
        let root = workspace(mode);
        let output = invoke(root.path(), "[\"model-hf\"]");
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains(diagnostic), "{stderr}");
        assert!(!stderr.contains("sensitive-payload"));
        assert!(!stderr.contains("fixture-private"));
    }
}

#[test]
fn invalid_request_precedes_spawn_when_package_name_is_invalid() {
    let root = workspace("success");
    let output = invoke(root.path(), "[\"bad;name\"]");
    assert_eq!(output.status.code(), Some(2));
    assert!(!root.path().join("invocation.json").exists());
}

#[test]
fn request_validation_precedes_spawn_when_complete_batches_are_invalid() {
    for batches in [
        "[{\"crates\":[\"other\"]}]",
        "[{\"crates\":[\"a\"]},{\"crates\":[\"a\"]}]",
    ] {
        let root = workspace("success");
        let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args([
                "repository",
                "cargo-packages",
                "--generation",
                "current",
                "--crates",
                "[\"a\"]",
                "--batches",
                batches,
                "--cargo",
            ])
            .arg(fixture())
            .current_dir(root.path())
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(!root.path().join("invocation.json").exists());
    }
}

#[test]
fn usage_rejects_invalid_combinations_before_spawn() {
    for extra in [
        vec!["--metadata", "missing"],
        vec!["--timeout", "0"],
        vec!["--cargo", "relative"],
    ] {
        let root = workspace("success");
        let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args([
                "repository",
                "cargo-packages",
                "--generation",
                "current",
                "--crates",
                "[\"a\"]",
                "--cargo",
            ])
            .arg(fixture())
            .args(extra)
            .current_dir(root.path())
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(!root.path().join("invocation.json").exists());
    }
}

#[cfg(unix)]
#[test]
fn deadline_cleans_owned_descendant_when_unrelated_sentinel_is_running() {
    let root = workspace("tree");
    let mut sentinel = Command::new(fixture()).arg("hold").spawn().unwrap();
    let output = invoke(root.path(), "[\"model-hf\"]");
    let sentinel_alive = sentinel.try_wait().unwrap().is_none();
    sentinel.kill().unwrap();
    sentinel.wait().unwrap();
    assert!(sentinel_alive);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Deadline"));
    let pid = std::fs::read_to_string(root.path().join("descendant.pid")).unwrap();
    assert!(
        !Command::new("/bin/kill")
            .args(["-0", pid.trim()])
            .output()
            .unwrap()
            .status
            .success()
    );
}

#[cfg(windows)]
#[test]
fn windows_extensionless_cargo_resolves_native_exe_with_fixed_metadata_contract() {
    let root = workspace("success");
    let cargo = root.path().join("cargo.exe");
    std::fs::copy(fixture(), &cargo).unwrap();
    let output = invoke_cargo(root.path(), "[\"model-hf\"]", &cargo.with_extension(""));
    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stdout, b"[\"skippy-model-hf\", \"skippy-hf-hub\"]\n");
    let record: Invocation =
        serde_json::from_slice(&std::fs::read(root.path().join("invocation.json")).unwrap())
            .unwrap();
    assert_eq!(
        record.argv,
        ["metadata", "--locked", "--no-deps", "--format-version=1"]
    );
    assert_eq!(record.cwd, root.path().canonicalize().unwrap());
}

#[cfg(windows)]
#[test]
fn windows_cargo_command_scripts_and_missing_native_sibling_refuse_before_execution() {
    let root = workspace("success");
    let command = root.path().join("cargo.cmd");
    std::fs::write(&command, b"@echo forbidden\r\n").unwrap();
    // A sibling executable does not authorize changing an explicit script request.
    std::fs::copy(fixture(), root.path().join("cargo.exe")).unwrap();
    let output = invoke_cargo(root.path(), "[\"model-hf\"]", &command);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Windows requires an explicit .exe"));
    assert!(!root.path().join("invocation.json").exists());
    let missing = invoke_cargo(
        root.path(),
        "[\"model-hf\"]",
        &root.path().join("missing-cargo"),
    );
    assert!(!missing.status.success());
    assert!(missing.stdout.is_empty());
    assert!(!root.path().join("invocation.json").exists());
}
