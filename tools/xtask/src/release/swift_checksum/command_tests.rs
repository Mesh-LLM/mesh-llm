use super::support;
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::time::Duration;

fn command(root: &std::path::Path, operation: &str, mode: &str) -> process::RawProcessReport {
    let artifact = root.join(mode);
    std::fs::write(&artifact, b"inert").unwrap();
    let spec = ProcessSpec {
        executable: support::fixture()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("xtask"),
        arguments: [
            "release".into(),
            "swift-manifest".into(),
            operation.into(),
            "not-a-version".into(),
            artifact.into_os_string(),
            root.join("Package.swift").into_os_string(),
            support::fixture().into_os_string(),
            "3".into(),
            "1024".into(),
        ]
        .into_iter()
        .map(Value::Public)
        .collect(),
        cwd: root.to_owned(),
        environment: std::collections::BTreeMap::new(),
    };
    let limits = Limits {
        execution: Duration::from_secs(6),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    process::supervise_raw(
        &spec,
        &limits,
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    )
    .unwrap()
}

#[cfg(target_os = "macos")]
#[test]
fn swift_checksum_command_updates_when_native_fixture_succeeds() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("Package.swift"),
        include_bytes!("../../../tests/fixtures/release/swift_manifest/template.swift"),
    )
    .unwrap();
    let report = command(root.path(), "update", "success");
    assert!(report.process.success());
    assert_eq!(
        std::fs::read(root.path().join("Package.swift")).unwrap(),
        include_bytes!("../../../tests/fixtures/release/swift_manifest/updated.swift")
    );
    assert_eq!(report.stdout.unwrap().as_bytes(), b"updated Swift package manifest for not-a-version\n  url: https://github.com/Mesh-LLM/mesh-llm/releases/download/not-a-version/MeshLLMFFI.xcframework.zip\n  checksum: opaque-checksum\n");
}

#[cfg(target_os = "macos")]
#[test]
fn swift_checksum_command_verifies_when_native_fixture_matches() {
    let root = tempfile::tempdir().unwrap();
    let original = include_bytes!("../../../tests/fixtures/release/swift_manifest/updated.swift");
    std::fs::write(root.path().join("Package.swift"), original).unwrap();
    let report = command(root.path(), "verify", "success");
    assert!(report.process.success());
    assert_eq!(
        report.stdout.unwrap().as_bytes(),
        b"verified Swift package manifest for not-a-version\n"
    );
    assert_eq!(
        std::fs::read(root.path().join("Package.swift")).unwrap(),
        original
    );
}

#[cfg(target_os = "macos")]
#[test]
fn swift_checksum_command_preserves_verify_error_order() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("Package.swift"),
        b"let remoteFFIXCFrameworkURL = \"__MESH_SWIFT_RELEASE_TAG__\"",
    )
    .unwrap();
    let report = command(root.path(), "verify", "success");
    assert!(!report.process.success());
    assert_eq!(
        report.stderr.unwrap().as_bytes(),
        format!(
            "missing remoteFFIXCFrameworkChecksum in {}\n",
            root.path().join("Package.swift").display()
        )
        .as_bytes()
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn swift_checksum_command_refuses_native_execution_on_other_platforms() {
    let root = tempfile::tempdir().unwrap();
    let report = command(root.path(), "verify", "success");
    assert!(!report.process.success());
    assert_eq!(
        report.stderr.unwrap().as_bytes(),
        b"error: Swift package manifest verification must run on macOS\n"
    );
    assert!(!root.path().join("argv.bin").exists());
}

#[cfg(target_os = "macos")]
#[test]
fn swift_checksum_command_preserves_manifest_when_native_fails() {
    let root = tempfile::tempdir().unwrap();
    let original = b"sentinel\r\n";
    std::fs::write(root.path().join("Package.swift"), original).unwrap();
    let report = command(root.path(), "update", "failure");
    assert_eq!(
        report.process.status.and_then(|status| status.code()),
        Some(7)
    );
    assert_eq!(
        std::fs::read(root.path().join("Package.swift")).unwrap(),
        original
    );
}

#[cfg(target_os = "macos")]
#[test]
fn swift_checksum_command_preserves_manifest_when_native_times_out() {
    let root = tempfile::tempdir().unwrap();
    let original = b"sentinel\r\n";
    std::fs::write(root.path().join("Package.swift"), original).unwrap();
    let report = command(root.path(), "update", "tree");
    assert!(!report.process.success());
    assert_eq!(
        std::fs::read(root.path().join("Package.swift")).unwrap(),
        original
    );
    assert!(
        std::str::from_utf8(report.stderr.unwrap().as_bytes())
            .unwrap()
            .contains("Deadline")
    );
}
