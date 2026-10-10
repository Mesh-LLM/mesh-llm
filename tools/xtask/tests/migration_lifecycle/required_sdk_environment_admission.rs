//! Register after applying caller candidates. No product/model process is started.
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Command, Output},
};

fn admission(path: &str) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    fs::read_to_string(root.join(path))
        .unwrap()
        .split("# Required SDK environment admission begins.\n")
        .nth(1)
        .unwrap()
        .split("# Required SDK environment admission ends.")
        .next()
        .unwrap()
        .to_owned()
}

fn run(path: &str, variable: &str, value: Option<&str>, class: &str) -> Output {
    let script = format!("set -euo pipefail\n{}\nprintf admitted\\n", admission(path));
    let mut command = Command::new("/bin/bash");
    command
        .args(["-c", &script])
        .env("MODEL_CLASS", class)
        .env_remove(variable);
    if let Some(value) = value {
        command.env(variable, value);
    }
    command.output().unwrap()
}

#[test]
fn missing_empty_relative_directory_and_nonexecutable_sdk_environments_fail_closed() {
    let directory = tempfile::tempdir().unwrap();
    let nonexec = directory.path().join("nonexecutable python");
    fs::write(&nonexec, b"not an SDK interpreter").unwrap();
    fs::set_permissions(&nonexec, fs::Permissions::from_mode(0o600)).unwrap();
    for (path, variable) in [
        ("scripts/ci-compat-smoke.sh", "MESH_REQUIRED_SDK_PYTHON"),
        (
            "skippy/scripts/skippy-workload-certify.sh",
            "SKIPPY_WORKLOAD_SDK_PYTHON",
        ),
    ] {
        for value in [
            None,
            Some(""),
            Some("python3"),
            Some("/absent/python"),
            Some(directory.path().to_str().unwrap()),
            Some(nonexec.to_str().unwrap()),
        ] {
            let output = run(path, variable, value, "embedding");
            assert!(!output.status.success(), "{path}");
            assert!(output.stdout.is_empty());
        }
    }
}

#[test]
fn embedding_admission_defers_execution_to_supervised_client_and_nonembedding_needs_no_sdk() {
    let directory = tempfile::tempdir().unwrap();
    let interpreter = directory.path().join("SDK Python with spaces");
    fs::write(&interpreter, "#!/bin/bash\nexit 23\n").unwrap();
    fs::set_permissions(&interpreter, fs::Permissions::from_mode(0o700)).unwrap();
    let output = run(
        "skippy/scripts/skippy-workload-certify.sh",
        "SKIPPY_WORKLOAD_SDK_PYTHON",
        interpreter.to_str(),
        "embedding",
    );
    // Admission checks the executable path without an unbounded interpreter probe.
    // The failing executable is accepted here and must fail in the actual SDK owner.
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("admitted"));
    let receipt = directory.path().join("embedding-sdk.json");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let supervised = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "automation",
            "smoke-observation",
            "sdk-client",
            "--client",
            "embeddings",
            "--python",
            interpreter.to_str().unwrap(),
            "--base-url",
            "http://127.0.0.1:1/v1",
            "--model",
            "fixture-model",
            "--timeout-secs",
            "8",
            "--receipt",
            receipt.to_str().unwrap(),
        ])
        .current_dir(root)
        .output()
        .unwrap();
    assert!(!supervised.status.success());
    let observed: serde_json::Value = serde_json::from_slice(&fs::read(receipt).unwrap()).unwrap();
    assert_eq!(observed["status"], "SDK_CHILD_OBSERVED");
    assert_eq!(observed["child_success"], false);
    assert_eq!(observed["sdk_qualified"], false);
    assert_eq!(observed["process"]["exit_code"], 23);
    assert_eq!(observed["process"]["cleanup_complete"], true);
    let other = run(
        "skippy/scripts/skippy-workload-certify.sh",
        "SKIPPY_WORKLOAD_SDK_PYTHON",
        None,
        "ocr",
    );
    assert!(other.status.success());
}

#[test]
fn compatibility_setup_rejects_existing_files_directories_and_broken_symlinks_before_install() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let action =
        fs::read_to_string(root.join(".github/actions/setup-canary-python/action.yml")).unwrap();
    let script = action
        .split("      run: |\n")
        .nth(1)
        .unwrap()
        .lines()
        .map(|line| line.strip_prefix("        ").unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n");
    for kind in ["file", "directory", "broken-symlink"] {
        let temporary = tempfile::tempdir().unwrap();
        let bin = temporary.path().join("bin");
        fs::create_dir(&bin).unwrap();
        let python = bin.join("python3");
        fs::write(&python, "#!/bin/bash\nif [[ $* == --version ]]; then printf 'Python 3.12.9\\n'; exit 0; fi\nprintf called > \"$RUNNER_TEMP/installer-called\"\nexit 23\n").unwrap();
        fs::set_permissions(&python, fs::Permissions::from_mode(0o700)).unwrap();
        let environment = temporary.path().join("required-sdk-compatibility");
        match kind {
            "file" => fs::write(&environment, "existing").unwrap(),
            "directory" => fs::create_dir(&environment).unwrap(),
            _ => std::os::unix::fs::symlink("absent-target", &environment).unwrap(),
        }
        let controller = bin.join("source-admission-observer");
        fs::write(&controller,"#!/bin/bash\n[[ $* == 'automation smoke-observation sdk-source --kind root' || $* == 'automation smoke-observation sdk-source --kind compatibility' ]] || exit 97\nprintf admitted > \"$RUNNER_TEMP/source-admitted\"\nprintf '%s\n' \"$RUNNER_TEMP\"\n").unwrap();
        fs::set_permissions(&controller, fs::Permissions::from_mode(0o700)).unwrap();
        let github_environment = temporary.path().join("github-env");
        let output = Command::new("/bin/bash")
            .args(["-c", &script])
            .env_clear()
            .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
            .env("RUNNER_TEMP", temporary.path())
            .env("SDK_KIND", "compatibility")
            .env("MESH_LLM_AUTOMATION_BIN", &controller)
            .env("GITHUB_ENV", &github_environment)
            .current_dir(&root)
            .output()
            .unwrap();
        assert!(!output.status.success(), "{kind}");
        assert!(temporary.path().join("source-admitted").exists(), "{kind}");
        assert!(
            !temporary.path().join("installer-called").exists(),
            "{kind}"
        );
        assert!(!github_environment.exists(), "{kind}");
        assert!(fs::symlink_metadata(&environment).is_ok(), "{kind}");
    }
}
