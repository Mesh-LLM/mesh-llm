//! CLI input verification against a fake POSIX Skippy executable.
#![cfg(unix)]

use super::verify;
use std::os::unix::fs::PermissionsExt as _;
use std::path::Path;

const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef01234567";

/// A fake `skippy` shell executable answering `build-contract` and `--version`.
fn write_fake_cli(directory: &Path, source_sha: &str) {
    let script = format!(
        "#!/bin/sh\nif [ \"$1\" = \"build-contract\" ]; then\nprintf '%s\\n' '{{\"schema_version\":1,\"product\":\"skippy\",\"product_version\":\"1.2.3\",\"runtime_release\":\"4.5.6\",\"skippy_abi\":\"7\",\"source_sha\":\"{source_sha}\",\"os\":\"linux\",\"architecture\":\"x86_64\"}}'\nexit 0\nfi\nif [ \"$1\" = \"--version\" ]; then\nprintf 'skippy 1.2.3\\n'\nexit 0\nfi\nexit 9\n"
    );
    let binary = directory.join("skippy");
    std::fs::write(&binary, script).unwrap();
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
    let digest = super::super::contract::digest(&binary).unwrap();
    std::fs::write(
        directory.join("skippy.sha256"),
        format!("{digest}  skippy\n"),
    )
    .unwrap();
    std::fs::write(
        directory.join("host-imports.json"),
        format!(
            "{{\"binary\":\"skippy\",\"binary_sha256\":\"{digest}\",\"policy\":\"mesh-llm-dynamic-host-v2\",\"rejected_imports\":[],\"imports\":[\"libc.so.6\"],\"format\":\"elf\"}}"
        ),
    )
    .unwrap();
    std::fs::write(
        directory.join("build-contract.json"),
        format!(
            "{{\"schema_version\":1,\"product\":\"skippy\",\"product_version\":\"1.2.3\",\"runtime_release\":\"4.5.6\",\"skippy_abi\":\"7\",\"source_sha\":\"{source_sha}\",\"os\":\"linux\",\"architecture\":\"x86_64\"}}\n"
        ),
    )
    .unwrap();
}

#[test]
fn fake_cli_input_verifies_and_returns_contract() {
    let temp = tempfile::tempdir().unwrap();
    write_fake_cli(temp.path(), SOURCE_SHA);
    let contract = verify(temp.path(), SOURCE_SHA).expect("valid CLI input");
    assert_eq!(
        contract.get("skippy_abi").and_then(|v| v.as_str()),
        Some("7")
    );
}

#[test]
fn foreign_source_or_tampered_bytes_fail() {
    let temp = tempfile::tempdir().unwrap();
    write_fake_cli(temp.path(), SOURCE_SHA);
    let error = verify(temp.path(), &"f".repeat(40)).unwrap_err();
    assert!(error.contains("different source commit"), "{error}");
    std::fs::write(temp.path().join("skippy"), b"#!/bin/sh\nexit 1\n").unwrap();
    let error = verify(temp.path(), SOURCE_SHA).unwrap_err();
    assert!(error.contains("sidecar"), "{error}");
}

#[test]
fn duplicate_build_contract_keys_fail_closed() {
    let temp = tempfile::tempdir().unwrap();
    write_fake_cli(temp.path(), SOURCE_SHA);
    std::fs::write(
        temp.path().join("build-contract.json"),
        "{\"schema_version\":1,\"schema_version\":2}\n",
    )
    .unwrap();
    let error = verify(temp.path(), SOURCE_SHA).unwrap_err();
    assert!(error.contains("duplicate JSON key"), "{error}");
}

#[test]
fn missing_or_dual_executables_fail() {
    let temp = tempfile::tempdir().unwrap();
    assert!(verify(temp.path(), SOURCE_SHA).is_err());
    write_fake_cli(temp.path(), SOURCE_SHA);
    std::fs::write(temp.path().join("skippy.exe"), b"mz").unwrap();
    let error = verify(temp.path(), SOURCE_SHA).unwrap_err();
    assert!(error.contains("exactly one"), "{error}");
}
