//! Execute the protected consumer's inert guards and its actual native CLI.
use crate::{support, workflow_yaml::Node};
use sha2::{Digest, Sha256};
use std::{fs, os::unix::fs::symlink, path::Path, process::Command};

const SOURCE: &str = "0123456789012345678901234567890123456789";
const ACTION: &str = ".github/actions/audit-pr-authority-verified/action.yml";

fn document() -> Node {
    crate::workflow_yaml::parse(&fs::read_to_string(support::root().join(ACTION)).unwrap()).unwrap()
}

fn steps(document: &Node) -> &[Node] {
    let Node::Seq(steps) = document.get("runs").unwrap().get("steps").unwrap() else {
        panic!("composite steps")
    };
    steps
}

fn script(index: usize) -> String {
    steps(&document())[index]
        .get("run")
        .unwrap()
        .text()
        .unwrap()
        .to_owned()
}

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn command(directory: &Path) -> Command {
    let mut command = Command::new("/bin/bash");
    command.env_clear().current_dir(directory);
    command.env("PATH", "/usr/bin:/bin:/opt/homebrew/bin");
    command.env("RUNNER_TEMP", directory);
    command.env("RUNNER_OS", "macOS");
    command.env("RUNNER_ARCH", "ARM64");
    command.env("AUTOMATION_PRODUCER_OS", "macOS");
    command.env("AUTOMATION_PRODUCER_ARCH", "ARM64");
    command.env("AUTOMATION_SOURCE_SHA", SOURCE);
    command.env("AUTOMATION_ARTIFACT_ID", "1234");
    command.env("AUTOMATION_BINARY_SHA256", digest(b"fixture"));
    command.env("INPUT_DEPOT_SELECTED", "true");
    command.env("INPUT_ALLOW_NATIVE_GITHUB_CACHE", "false");
    command.env("INPUT_ALLOW_DEPOT_REMOTE_CACHE", "false");
    command.env("GITHUB_EVENT_NAME", "pull_request");
    command.env("DOCKER_CONFIG", directory);
    command
}

fn run(command: &mut Command, body: &str) -> std::process::Output {
    command.args(["-c", body]);
    let mut owned = Command::new(command.get_program());
    owned.env_clear().args(command.get_args());
    if let Some(directory) = command.get_current_dir() {
        owned.current_dir(directory);
    }
    for (name, value) in command.get_envs() {
        if let Some(value) = value {
            owned.env(name, value);
        }
    }
    support::Fixture::new().run(owned)
}

fn manifest(directory: &Path) {
    let stage = directory.join("immutable-automation-restored");
    fs::write(
        stage.join("SHA256SUMS"),
        format!(
            "{}  xtask\n{}  source.txt\n",
            digest(&fs::read(stage.join("xtask")).unwrap()),
            digest(&fs::read(stage.join("source.txt")).unwrap())
        ),
    )
    .unwrap();
}

fn staged(binary: &[u8]) -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    let output = run(&mut command(directory.path()), &script(0));
    assert!(output.status.success(), "{output:?}");
    let stage = directory.path().join("immutable-automation-restored");
    fs::write(stage.join("xtask"), binary).unwrap();
    fs::write(stage.join("source.txt"), format!("{SOURCE}\n")).unwrap();
    manifest(directory.path());
    directory
}

#[test]
fn consumer_requires_exact_same_run_download_before_verification_and_native_audit() {
    let document = document();
    let steps = steps(&document);
    assert_eq!(steps.len(), 4);
    let download = &steps[1];
    assert_eq!(
        download.get("uses").and_then(Node::text),
        Some("actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c")
    );
    let inputs = download.get("with").unwrap();
    assert_eq!(
        inputs.get("artifact-ids").and_then(Node::text),
        Some("${{ inputs.artifact_id }}")
    );
    for forbidden in ["name", "run-id", "github-token", "repository"] {
        assert!(inputs.get(forbidden).is_none(), "{forbidden}");
    }
    for input in [
        "source_sha",
        "artifact_id",
        "binary_sha256",
        "producer_os",
        "producer_arch",
    ] {
        let input = document.get("inputs").unwrap().get(input).unwrap();
        assert_eq!(input.get("required").and_then(Node::text), Some("true"));
        assert!(input.get("default").is_none());
    }
    let execution = script(3);
    assert!(execution.contains(
        "\"$RUNNER_TEMP/immutable-automation-restored/$binary\" ci-ops pr-authority-audit"
    ));
    assert!(!execution.contains("MESH_LLM_AUTOMATION_BIN"));
}

#[test]
fn consumer_identity_refuses_download_admission_before_creating_a_stage() {
    for (name, value) in [
        ("AUTOMATION_ARTIFACT_ID", ""),
        ("AUTOMATION_ARTIFACT_ID", "0"),
        ("AUTOMATION_ARTIFACT_ID", "123,456"),
        ("AUTOMATION_BINARY_SHA256", ""),
        ("AUTOMATION_SOURCE_SHA", "main"),
        ("AUTOMATION_PRODUCER_OS", "Linux"),
        ("AUTOMATION_PRODUCER_ARCH", "X64"),
        ("RUNNER_OS", "unknown"),
        ("RUNNER_ARCH", "unknown"),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let result = run(command(directory.path()).env(name, value), &script(0));
        assert!(!result.status.success(), "{name}");
        assert!(
            !directory
                .path()
                .join("immutable-automation-restored")
                .exists()
        );
    }
}

#[test]
fn substituted_untrusted_payloads_cannot_run_even_with_a_rehashed_manifest() {
    let payload = b"#!/bin/bash\ntouch \"$RUNNER_TEMP/executed\"\n";
    for attack in [
        "wrong-source",
        "rehash",
        "corrupt",
        "extra",
        "symlink",
        "source-no-lf",
        "manifest-escape",
        "missing-id",
        "wrong-platform",
    ] {
        let directory = staged(payload);
        let stage = directory.path().join("immutable-automation-restored");
        match attack {
            "wrong-source" => {
                fs::write(
                    stage.join("source.txt"),
                    "1123456789012345678901234567890123456789\n",
                )
                .unwrap();
                manifest(directory.path());
            }
            "rehash" => {
                fs::write(
                    stage.join("xtask"),
                    b"#!/bin/bash\ntouch \"$RUNNER_TEMP/executed\"\nexit 1\n",
                )
                .unwrap();
                manifest(directory.path());
            }
            "corrupt" => fs::write(stage.join("xtask"), b"corrupt").unwrap(),
            "extra" => fs::write(stage.join(".extra"), b"unexpected").unwrap(),
            "symlink" => {
                fs::write(directory.path().join("outside"), payload).unwrap();
                fs::remove_file(stage.join("xtask")).unwrap();
                symlink(directory.path().join("outside"), stage.join("xtask")).unwrap();
            }
            "source-no-lf" => {
                fs::write(stage.join("source.txt"), SOURCE).unwrap();
                manifest(directory.path());
            }
            "manifest-escape" => fs::write(
                stage.join("SHA256SUMS"),
                format!("{}  ../outside\n", digest(payload)),
            )
            .unwrap(),
            "missing-id" | "wrong-platform" => (),
            _ => unreachable!(),
        }
        let mut command = command(directory.path());
        command.env("AUTOMATION_BINARY_SHA256", digest(payload));
        if attack == "missing-id" {
            command.env("AUTOMATION_ARTIFACT_ID", "");
        }
        if attack == "wrong-platform" {
            command.env("AUTOMATION_PRODUCER_ARCH", "X64");
        }
        let result = run(&mut command, &format!("{}\n{}", script(2), script(3)));
        assert!(!result.status.success(), "{attack}");
        assert!(!directory.path().join("executed").exists(), "{attack}");
    }
}

#[test]
fn verified_native_consumer_runs_without_checkout_and_ignores_an_executable_override() {
    let binary = fs::read(env!("CARGO_BIN_EXE_xtask")).unwrap();
    let directory = staged(&binary);
    let mut command = command(directory.path());
    command.env("AUTOMATION_BINARY_SHA256", digest(&binary));
    command.env("MESH_LLM_AUTOMATION_BIN", "/missing/untrusted-executable");
    command.env("ACTIONS_CACHE_URL", "http://[::1]:1234/cache");
    let result = run(&mut command, &format!("{}\n{}", script(2), script(3)));
    assert!(result.status.success(), "{result:?}");
    assert!(!directory.path().join(".git").exists());
    assert!(result.stderr.is_empty());
}

#[test]
fn verified_native_consumer_preserves_policy_failure_and_private_diagnostics() {
    let binary = fs::read(env!("CARGO_BIN_EXE_xtask")).unwrap();
    let directory = staged(&binary);
    let mut command = command(directory.path());
    command.env("AUTOMATION_BINARY_SHA256", digest(&binary));
    command.env("DEPOT_TOKEN", "private-probe-value");
    let result = run(&mut command, &format!("{}\n{}", script(2), script(3)));
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("DEPOT_TOKEN"));
    assert!(!stderr.contains("private-probe-value"));
    assert!(!String::from_utf8_lossy(&result.stdout).contains("private-probe-value"));
}
