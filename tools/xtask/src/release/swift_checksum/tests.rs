use super::{arguments, capture, error::Error, finalize, support};
use crate::automation::command_interrupt::Reason;
use crate::process::Cancellation;

#[test]
fn swift_checksum_argv_and_cwd_when_artifact_contains_spaces() {
    let root = tempfile::tempdir().unwrap();
    let native = support::native(root.path(), "artifact with spaces.zip");
    let result = capture::compute(&native, &Cancellation::default()).unwrap();
    assert_eq!(result, "opaque-checksum");
    assert_eq!(
        std::fs::read(root.path().join("argv.bin")).unwrap(),
        b"package\0compute-checksum\0artifact with spaces.zip\0"
    );
    assert_eq!(
        std::fs::read(root.path().join("cwd.bin")).unwrap(),
        root.path()
            .canonicalize()
            .unwrap()
            .as_os_str()
            .as_encoded_bytes()
    );
}

#[test]
fn swift_checksum_empty_when_stdout_is_empty() {
    let root = tempfile::tempdir().unwrap();
    let native = support::native(root.path(), "empty");
    let result = capture::compute(&native, &Cancellation::default()).unwrap();
    assert_eq!(result, "");
}

#[test]
fn swift_checksum_retains_cr_when_stripping_trailing_lf() {
    let root = tempfile::tempdir().unwrap();
    let native = support::native(root.path(), "crlf");
    let result = capture::compute(&native, &Cancellation::default()).unwrap();
    assert_eq!(result, "opaque\r");
}

#[test]
fn swift_checksum_rejects_when_stdout_is_invalid_utf8() {
    let root = tempfile::tempdir().unwrap();
    let native = support::native(root.path(), "invalid");
    let result = capture::compute(&native, &Cancellation::default());
    assert!(matches!(result, Err(Error::Utf8(_))));
}

#[test]
fn swift_checksum_rejects_when_capture_overflows() {
    let root = tempfile::tempdir().unwrap();
    let native = support::native(root.path(), "overflow");
    let result = capture::compute(&native, &Cancellation::default());
    let Err(Error::Native(report)) = result else {
        panic!("expected typed process receipt");
    };
    assert!(matches!(
        report.process.failure,
        Some(crate::process::Failure::RawCaptureOverflow { .. })
    ));
}

#[test]
fn swift_checksum_keeps_primary_when_finalization_also_fails() {
    let result = finalize(
        Err(Error::Artifact("first".into())),
        Err(Reason::Interrupted),
    );
    let Err(Error::Finalization { primary, reason }) = result else {
        panic!("expected dual failure evidence");
    };
    assert!(matches!(*primary, Error::Artifact(ref path) if path == "first"));
    assert!(matches!(reason, Reason::Interrupted));
}

#[test]
fn swift_checksum_rejects_success_when_finalization_fails() {
    let result = finalize(Ok("checksum".into()), Err(Reason::Interrupted));
    assert!(matches!(result, Err(Error::Interrupt(Reason::Interrupted))));
}

#[test]
fn swift_checksum_arguments_reject_when_executable_is_relative() {
    let args = [
        "update", "tag", "artifact", "manifest", "swift", "3", "1024",
    ]
    .map(str::to_owned);
    let result = arguments::parse(&args);
    assert!(matches!(result, Err("Swift executable must be absolute")));
}

#[test]
fn swift_checksum_checks_artifact_before_manifest_and_native_spawn() {
    let root = tempfile::tempdir().unwrap();
    let artifact = root.path().join("missing artifact");
    let manifest = root.path().join("missing manifest");
    let args = [
        "update".to_owned(),
        "tag".to_owned(),
        artifact.display().to_string(),
        manifest.display().to_string(),
        "/missing/swift".to_owned(),
        "3".to_owned(),
        "1024".to_owned(),
    ];
    let options = arguments::parse(&args).unwrap();
    let result = super::execute(&options);
    assert!(matches!(result, Err(Error::Artifact(_))));
}

#[test]
fn swift_checksum_checks_manifest_before_native_spawn() {
    let root = tempfile::tempdir().unwrap();
    let artifact = root.path().join("artifact");
    std::fs::write(&artifact, b"inert").unwrap();
    let args = [
        "verify".to_owned(),
        "tag".to_owned(),
        artifact.display().to_string(),
        root.path().display().to_string(),
        "/missing/swift".to_owned(),
        "3".to_owned(),
        "1024".to_owned(),
    ];
    let options = arguments::parse(&args).unwrap();
    let result = super::execute(&options);
    assert!(matches!(result, Err(Error::Manifest(_))));
}

#[test]
fn swift_checksum_arguments_reject_when_budgets_exceed_bounds() {
    let args = [
        "verify", "tag", "artifact", "manifest", "/swift", "3601", "1024",
    ]
    .map(str::to_owned);
    let result = arguments::parse(&args);
    assert!(matches!(result, Err("timeout must be 1..3600 seconds")));
}
