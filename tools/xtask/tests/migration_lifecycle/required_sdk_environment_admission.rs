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
            "scripts/skippy-workload-certify.sh",
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
fn embedding_missing_package_rejects_before_native_work_and_nonembedding_needs_no_sdk() {
    let directory = tempfile::tempdir().unwrap();
    let interpreter = directory.path().join("SDK Python with spaces");
    fs::write(&interpreter, "#!/bin/bash\nexit 23\n").unwrap();
    fs::set_permissions(&interpreter, fs::Permissions::from_mode(0o700)).unwrap();
    let output = run(
        "scripts/skippy-workload-certify.sh",
        "SKIPPY_WORKLOAD_SDK_PYTHON",
        interpreter.to_str(),
        "embedding",
    );
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("openai package"));
    let other = run(
        "scripts/skippy-workload-certify.sh",
        "SKIPPY_WORKLOAD_SDK_PYTHON",
        None,
        "ocr",
    );
    assert!(other.status.success());
}
