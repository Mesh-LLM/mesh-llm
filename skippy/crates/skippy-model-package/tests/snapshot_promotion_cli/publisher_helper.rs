#![cfg(unix)]
use sha2::{Digest, Sha256};
use std::{
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::Path,
    process::{Command, Stdio},
};
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn input(root: &Path) -> serde_json::Value {
    let path = root.join("model.gguf");
    let bytes = b"GGUFfixture-publisher";
    fs::write(&path, bytes).unwrap();
    serde_json::json!({"schema_version":1,"repo":"fixture/model","parent_commit":"a".repeat(40),
        "shards":[{"path":path,"path_in_repo":"model.gguf","sha256":digest(bytes),"byte_size":bytes.len()}],
        "sidecars":[],"credential_file":null,"execution_timeout_ms":10000})
}
fn run(verb: &str, path: &Path, output: &Path) -> std::process::Output {
    let child = Command::new(env!("CARGO_BIN_EXE_model-package-publish"))
        .env_clear()
        .env("HF_ENDPOINT", "http://127.0.0.1:1")
        .env("HF_TOKEN", "must-not-use-ambient")
        .args([verb, "--input"])
        .arg(path)
        .arg("--output-directory")
        .arg(output)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let result = super::finish_fixture_child(child);
    assert!(result.stdout.len() < 4096 && result.stderr.len() < 4096);
    assert!(!String::from_utf8_lossy(&result.stderr).contains("must-not-use-ambient"));
    result
}
#[test]
fn publisher_admit_correlates_actual_local_pins_without_publication_or_credentials() {
    let root = tempfile::tempdir().unwrap();
    let root_path = root.path().canonicalize().unwrap();
    let value = input(&root_path);
    let path = root_path.join("input.json");
    fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    let output = root_path.join("out");
    let result = run("admit", &path, &output);
    assert!(result.status.success());
    let receipt: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("publication.json")).unwrap()).unwrap();
    assert_eq!(receipt["status"], "ADMITTED");
    assert!(receipt["publication"].is_null());
    assert_eq!(receipt["source_custody_verified"], true);
    let canonical = format!(
        r#"{{"schema_version":1,"repo":"fixture/model","parent_commit":"{}","shards":[{{"path":{},"path_in_repo":"model.gguf","sha256":"{}","byte_size":{}}}],"sidecars":[],"credential_file":null,"execution_timeout_ms":10000}}"#,
        "a".repeat(40),
        serde_json::to_string(&value["shards"][0]["path"]).unwrap(),
        value["shards"][0]["sha256"].as_str().unwrap(),
        value["shards"][0]["byte_size"]
    );
    assert_eq!(receipt["request_sha256"], digest(canonical.as_bytes()));
    assert!(!output.join("progress.json").exists());
    assert_eq!(
        fs::metadata(&output).unwrap().permissions().mode() & 0o777,
        0o700
    );
    root.close().unwrap();
}
#[test]
fn publisher_refuses_nonregular_input_without_fifo_writer_and_preserves_outputs() {
    let root = tempfile::tempdir().unwrap();
    let base = root.path().canonicalize().unwrap();
    let valid = base.join("valid.json");
    fs::write(&valid, serde_json::to_vec(&input(&base)).unwrap()).unwrap();
    let fifo = base.join("fifo");
    let c = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
    // SAFETY: path is a live, NUL-terminated owned fixture path; mkfifo does not open it.
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
    let link = base.join("link");
    symlink(&valid, &link).unwrap();
    for (index, path) in [fifo, link, base.clone()].iter().enumerate() {
        let output = base.join(format!("out{index}"));
        let result = run("admit", path, &output);
        assert_eq!(result.status.code(), Some(1));
        assert!(!output.exists());
    }
    let output = base.join("existing");
    fs::create_dir(&output).unwrap();
    fs::write(output.join("sentinel"), b"keep").unwrap();
    assert_eq!(run("admit", &valid, &output).status.code(), Some(1));
    assert_eq!(fs::read(output.join("sentinel")).unwrap(), b"keep");
    root.close().unwrap();
}
#[test]
fn publisher_pin_or_private_credential_refusal_has_correlated_failed_receipt() {
    let root = tempfile::tempdir().unwrap();
    let base = root.path().canonicalize().unwrap();
    for private in [false, true] {
        let mut value = input(&base);
        let verb = if private { "publish" } else { "admit" };
        if private {
            let path = base.join("credential");
            fs::write(&path, b"explicit-secret\n").unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
            value["credential_file"] = serde_json::json!(path);
        } else {
            value["shards"][0]["sha256"] = serde_json::json!("0".repeat(64));
        }
        let path = base.join("input.json");
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let output = base.join(format!("out{private}"));
        assert_eq!(run(verb, &path, &output).status.code(), Some(1));
        let bytes = fs::read(output.join("publication.json")).unwrap();
        assert!(!String::from_utf8_lossy(&bytes).contains("explicit-secret"));
        let receipt: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(receipt["status"], "FAILED");
        assert!(receipt["publication"].is_null());
        assert_eq!(receipt["request_sha256"].as_str().unwrap().len(), 64);
        assert!(!output.join("progress.json").exists());
    }
    root.close().unwrap();
}

#[test]
fn publisher_help_and_argument_refusal_do_not_echo_credential_arguments() {
    for flag in ["--help", "--unrecognized-secret=fixture-private"] {
        let child = Command::new(env!("CARGO_BIN_EXE_model-package-publish"))
            .env_clear()
            .arg(flag)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let output = super::finish_fixture_child(child);
        assert!(output.stdout.len() < 8192 && output.stderr.len() < 4096);
        if flag == "--help" {
            assert!(output.status.success());
            assert!(String::from_utf8_lossy(&output.stdout).contains("publish"));
        } else {
            assert_eq!(output.status.code(), Some(1));
            assert!(!String::from_utf8_lossy(&output.stderr).contains("fixture-private"));
        }
    }
    let root = tempfile::tempdir().unwrap();
    for arguments in [
        vec!["admit"],
        vec!["admit", "--input", "fixture-private"],
        vec!["admit", "--output-directory", "fixture-private"],
    ] {
        let child = Command::new(env!("CARGO_BIN_EXE_model-package-publish"))
            .env_clear()
            .current_dir(root.path())
            .args(arguments)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let output = super::finish_fixture_child(child);
        assert_eq!(output.status.code(), Some(1));
        assert!(!String::from_utf8_lossy(&output.stderr).contains("fixture-private"));
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
    }
    root.close().unwrap();
}

#[test]
fn regular_receipt_actual_cli_refuses_fifo_wrong_correlation_and_public_credentials_before_network()
{
    let root = tempfile::tempdir().unwrap();
    let base = root.path().canonicalize().unwrap();
    let body=serde_json::to_vec(&serde_json::json!({"schema_version":1,"request_sha256":"a".repeat(64),"status":"CERTIFIED"})).unwrap();
    let artifact = base.join("native-job.json");
    fs::write(&artifact, &body).unwrap();
    let credential = base.join("credential");
    fs::write(&credential, b"inert-explicit-secret").unwrap();
    fs::set_permissions(&credential, fs::Permissions::from_mode(0o644)).unwrap();
    let mut value = serde_json::json!({"schema_version":1,"repo":"fixture/evidence","parent_commit":"b".repeat(40),"artifact":{"path":artifact,"path_in_repo":"runs/native-job.json","sha256":digest(&body),"byte_size":body.len()},"receipt_request_sha256":"a".repeat(64),"credential_file":credential,"execution_timeout_ms":10000});
    let source = base.join("input.json");
    fs::write(&source, serde_json::to_vec(&value).unwrap()).unwrap();
    let output = base.join("out-credential");
    assert_eq!(
        run("publish-regular-receipt", &source, &output)
            .status
            .code(),
        Some(1)
    );
    assert!(!output.exists());
    value["receipt_request_sha256"] = serde_json::json!("c".repeat(64));
    fs::write(&source, serde_json::to_vec(&value).unwrap()).unwrap();
    let output = base.join("out-correlation");
    assert_eq!(
        run("publish-regular-receipt", &source, &output)
            .status
            .code(),
        Some(1)
    );
    assert!(!output.exists());
    let fifo = base.join("fifo");
    let name = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
    // SAFETY: owned NUL-terminated path is valid; this creates but never opens the FIFO.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let output = base.join("out-fifo");
    assert_eq!(
        run("publish-regular-receipt", &fifo, &output).status.code(),
        Some(1)
    );
    assert!(!output.exists());
    root.close().unwrap();
}
