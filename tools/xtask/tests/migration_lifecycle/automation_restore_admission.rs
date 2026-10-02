//! Register only after applying the candidate restore action; never runs xtask.
use sha2::{Digest, Sha256};
use std::{
    fs,
    os::unix::fs::symlink,
    path::Path,
    process::{Command, Output},
};

const SOURCE: &str = "0123456789012345678901234567890123456789";
const BINARY: &[u8] = b"#!/bin/bash\necho must-not-execute\nexit 93\n";

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn verifier() -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source =
        fs::read_to_string(root.join(".github/actions/restore-automation/action.yml")).unwrap();
    source
        .split("    - name: Verify and export automation\n")
        .nth(1)
        .unwrap()
        .split("      run: |\n")
        .nth(1)
        .unwrap()
        .lines()
        .map(|line| line.strip_prefix("        ").unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n")
}

fn fixture() -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    let stage = directory.path().join("immutable-automation-restored");
    fs::create_dir(&stage).unwrap();
    fs::write(stage.join("xtask"), BINARY).unwrap();
    fs::write(stage.join("source.txt"), format!("{SOURCE}\n")).unwrap();
    checksums(&stage);
    directory
}

fn checksums(stage: &Path) {
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

fn invoke(directory: &Path, source: &str, id: &str, expected: &str) -> Output {
    Command::new("/bin/bash")
        .args(["-c", &verifier()])
        .env("RUNNER_TEMP", directory)
        .env("RUNNER_OS", "Linux")
        .env("AUTOMATION_SOURCE_SHA", source)
        .env("AUTOMATION_ARTIFACT_ID", id)
        .env("AUTOMATION_BINARY_SHA256", expected)
        .env("GITHUB_ENV", directory.join("export.env"))
        .output()
        .unwrap()
}

#[test]
fn admits_exact_producer_identity_without_executing_binary() {
    let directory = fixture();
    let result = invoke(directory.path(), SOURCE, "123", &digest(BINARY));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!String::from_utf8_lossy(&result.stdout).contains("must-not-execute"));
    let exported = fs::read_to_string(directory.path().join("export.env")).unwrap();
    assert_eq!(
        exported,
        format!(
            "MESH_LLM_AUTOMATION_BIN={}/immutable-automation-restored/xtask\n",
            directory.path().display()
        )
    );
}

#[test]
fn rejects_binary_substitution_even_when_its_manifest_is_rehashed() {
    let directory = fixture();
    let stage = directory.path().join("immutable-automation-restored");
    fs::write(stage.join("xtask"), b"substituted binary").unwrap();
    checksums(&stage);
    assert!(
        !invoke(directory.path(), SOURCE, "123", &digest(BINARY))
            .status
            .success()
    );
    assert!(!directory.path().join("export.env").exists());
}

#[test]
fn rejects_wrong_source_partial_identity_and_checksum_target_escape() {
    for (source, id, expected) in [
        ("bad", "123", digest(BINARY)),
        (
            "1123456789012345678901234567890123456789",
            "123",
            digest(BINARY),
        ),
        (SOURCE, "", digest(BINARY)),
        (SOURCE, "123", String::new()),
        (SOURCE, "0", digest(BINARY)),
        (SOURCE, "123,124", digest(BINARY)),
    ] {
        let directory = fixture();
        assert!(
            !invoke(directory.path(), source, id, &expected)
                .status
                .success()
        );
        assert!(!directory.path().join("export.env").exists());
    }
    for name in ["../outside", "/tmp/outside", "unexpected", "source.txt"] {
        let directory = fixture();
        let stage = directory.path().join("immutable-automation-restored");
        fs::write(
            stage.join("SHA256SUMS"),
            format!(
                "{}  {name}\n{}  source.txt\n",
                digest(BINARY),
                digest(format!("{SOURCE}\n").as_bytes())
            ),
        )
        .unwrap();
        assert!(
            !invoke(directory.path(), SOURCE, "123", &digest(BINARY))
                .status
                .success()
        );
        assert!(!directory.path().join("export.env").exists());
    }
}

#[test]
fn rejects_extra_members_symlink_and_noncanonical_source_bytes() {
    for attack in [
        "extra",
        "symlink",
        "source-newline",
        "source-nul",
        "source-no-lf",
        "corrupt",
    ] {
        let directory = fixture();
        let stage = directory.path().join("immutable-automation-restored");
        match attack {
            "extra" => fs::write(stage.join(".extra"), b"unexpected").unwrap(),
            "symlink" => {
                fs::write(directory.path().join("outside"), BINARY).unwrap();
                fs::remove_file(stage.join("xtask")).unwrap();
                symlink(directory.path().join("outside"), stage.join("xtask")).unwrap();
            }
            "source-newline" => {
                fs::write(stage.join("source.txt"), format!("{SOURCE}\n\n")).unwrap();
                checksums(&stage);
            }
            "source-nul" => {
                let mut bytes = SOURCE.as_bytes().to_vec();
                bytes.push(0);
                fs::write(stage.join("source.txt"), bytes).unwrap();
                checksums(&stage);
            }
            "source-no-lf" => {
                fs::write(stage.join("source.txt"), SOURCE.as_bytes()).unwrap();
                checksums(&stage);
            }
            "corrupt" => fs::write(stage.join("xtask"), b"corrupt").unwrap(),
            _ => unreachable!(),
        }
        assert!(
            !invoke(directory.path(), SOURCE, "123", &digest(BINARY))
                .status
                .success(),
            "{attack}"
        );
        assert!(!directory.path().join("export.env").exists());
    }
}

fn run_block(path: &str, step: &str, indent: &str) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source = fs::read_to_string(root.join(path)).unwrap();
    let after = source
        .split(step)
        .nth(1)
        .unwrap()
        .split("run: |\n")
        .nth(1)
        .unwrap();
    after
        .lines()
        .take_while(|line| line.starts_with(indent) || line.trim().is_empty())
        .map(|line| line.strip_prefix(indent).unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n")
        .trim_end()
        .to_owned()
}

#[test]
fn no_checkout_workflow_and_local_restore_have_identical_admission_contract() {
    let action = run_block(
        ".github/actions/restore-automation/action.yml",
        "- name: Verify and export automation\n",
        "        ",
    );
    let sentinel = run_block(
        ".github/workflows/ci-quality-slice.yml",
        "- name: Verify protected authority automation without executing it\n",
        "          ",
    );
    assert_eq!(action, sentinel);
}

#[test]
fn admits_standard_binary_checksum_marker_and_windows_executable_name() {
    let directory = fixture();
    let stage = directory.path().join("immutable-automation-restored");
    fs::rename(stage.join("xtask"), stage.join("xtask.exe")).unwrap();
    fs::write(
        stage.join("SHA256SUMS"),
        format!(
            "{} *xtask.exe\n{}  source.txt\n",
            digest(BINARY),
            digest(format!("{SOURCE}\n").as_bytes())
        ),
    )
    .unwrap();
    let output = Command::new("/bin/bash")
        .args(["-c", &verifier()])
        .env("RUNNER_TEMP", directory.path())
        .env("RUNNER_OS", "Windows")
        .env("AUTOMATION_SOURCE_SHA", SOURCE)
        .env("AUTOMATION_ARTIFACT_ID", "123")
        .env("AUTOMATION_BINARY_SHA256", digest(BINARY))
        .env("GITHUB_ENV", directory.path().join("export.env"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("must-not-execute"));
    assert!(
        fs::read_to_string(directory.path().join("export.env"))
            .unwrap()
            .contains("xtask.exe\n")
    );
}

#[test]
fn admits_complete_checksum_entries_without_final_manifest_newline() {
    let directory = fixture();
    let stage = directory.path().join("immutable-automation-restored");
    let mut bytes = fs::read(stage.join("SHA256SUMS")).unwrap();
    assert_eq!(bytes.pop(), Some(b'\n'));
    fs::write(stage.join("SHA256SUMS"), bytes).unwrap();
    let result = invoke(directory.path(), SOURCE, "123", &digest(BINARY));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!String::from_utf8_lossy(&result.stdout).contains("must-not-execute"));
    assert!(directory.path().join("export.env").is_file());
}
