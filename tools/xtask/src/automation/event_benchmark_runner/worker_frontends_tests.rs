use super::*;
use serde_json::json;
fn args(pairs: &[&str]) -> Vec<String> {
    pairs.iter().map(|value| (*value).into()).collect()
}
fn expected(path: PathBuf, bytes: &[u8]) -> ExpectedFile {
    ExpectedFile {
        path,
        sha256: hex::encode(Sha256::digest(bytes)),
    }
}
#[test]
fn worker_flags_are_closed_absolute_distinct_and_nonduplicated() {
    assert!(
        flags(&args(&[
            "--output",
            "/fresh/output",
            "--input",
            "/fresh/input"
        ]))
        .is_ok()
    );
    for values in [
        vec!["--input", "relative", "--output", "/fresh/output"],
        vec!["--input", "/same", "--output", "/same"],
        vec!["--input", "/a", "--input", "/b"],
        vec!["--remote", "/a", "--output", "/b"],
        vec!["--input", "/a"],
    ] {
        assert!(flags(&args(&values)).is_err());
    }
    assert!(run("not-worker", &[]).is_err());
}
#[test]
fn atomic_identity_receipt_requires_exact_operation_request_correlation() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().canonicalize().unwrap().join("receipt");
    let request = IdentityRequest::Revalidate {
        binaries: [
            expected(PathBuf::from("/a"), b"a"),
            expected(PathBuf::from("/b"), b"b"),
        ],
        model: expected(PathBuf::from("/m"), b"m"),
    };
    evidence_io::publish(
        &output,
        &Receipt {
            schema_version: 1,
            request_sha256: request_sha256(&request).unwrap(),
            data: IdentityData::Revalidated,
        },
        IDENTITY_BYTES,
    )
    .unwrap();
    assert!(matches!(
        correlated::<IdentityData, _>(&output, &request).unwrap(),
        IdentityData::Revalidated
    ));
    let changed = IdentityRequest::Revalidate {
        binaries: [
            expected(PathBuf::from("/a"), b"changed"),
            expected(PathBuf::from("/b"), b"b"),
        ],
        model: expected(PathBuf::from("/m"), b"m"),
    };
    assert!(correlated::<IdentityData, _>(&output, &changed).is_err());
    assert!(evidence_io::publish(&output, &json!({}), IDENTITY_BYTES).is_err());
}
#[test]
fn revalidation_worker_detects_changed_bytes_and_cancel_without_coordinator_hashing() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().canonicalize().unwrap().join("bytes");
    fs::write(&path, b"same").unwrap();
    let request = IdentityRequest::Revalidate {
        binaries: [
            expected(path.clone(), b"same"),
            expected(path.clone(), b"same"),
        ],
        model: expected(path.clone(), b"same"),
    };
    assert!(matches!(
        identity(&request, &Cancellation::default()).unwrap(),
        IdentityData::Revalidated
    ));
    fs::write(&path, b"changed").unwrap();
    assert!(identity(&request, &Cancellation::default()).is_err());
    let cancel = Cancellation::default();
    cancel.cancel();
    assert!(identity(&request, &cancel).is_err());
}
#[test]
fn runtime_package_enumeration_requires_local_nonempty_bounded_manifest_graph() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    assert!(packages(&root).is_err());
    fs::create_dir(root.join("runtime")).unwrap();
    fs::write(root.join("runtime/manifest.json"), b"{}").unwrap();
    assert_eq!(packages(&root).unwrap(), vec![root.join("runtime")]);
    fs::create_dir(root.join("missing-manifest")).unwrap();
    assert!(packages(&root).is_err());
}
#[test]
fn measurement_receipt_refuses_missing_or_changed_full_request_hash() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().canonicalize().unwrap().join("receipt");
    let prompt = "fixture".to_string();
    let input = measurement_worker::Input {
        schema_version: 1,
        port: 1234,
        prompt_sha256: hex::encode(Sha256::digest(prompt.as_bytes())),
        prompt,
        max_tokens: 64,
        readiness_timeout_ms: 100,
        request_timeout_ms: 100,
        readiness_poll_ms: 1,
    };
    let mut receipt = measurement_worker::Evidence {
        schema_version: 1,
        prompt_sha256: input.prompt_sha256.clone(),
        ..Default::default()
    };
    evidence_io::publish(&output, &receipt, evidence_io::RECEIPT_BYTES).unwrap();
    assert!(measurement_receipt(&output, &input).is_err());
    fs::remove_file(&output).unwrap();
    receipt.request_sha256 = Some(request_sha256(&input).unwrap());
    evidence_io::publish(&output, &receipt, evidence_io::RECEIPT_BYTES).unwrap();
    assert!(measurement_receipt(&output, &input).is_ok());
    let changed = measurement_worker::Input {
        port: 1235,
        ..input
    };
    assert!(measurement_receipt(&output, &changed).is_err());
}
#[test]
fn identity_json_schema_refuses_unknown_operations_and_fields() {
    assert!(
        serde_json::from_value::<IdentityRequest>(
            json!({"operation":"download","url":"https://remote.invalid"})
        )
        .is_err()
    );
    assert!(
        serde_json::from_value::<ExpectedFile>(
            json!({"path":"/a","sha256":"a".repeat(64),"ignored":true})
        )
        .is_err()
    );
}

#[cfg(unix)]
#[test]
fn dangling_receipt_symlink_is_rejected_before_any_measurement() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().canonicalize().unwrap().join("output");
    std::os::unix::fs::symlink(directory.path().join("missing"), &output).unwrap();
    assert!(
        flags(&[
            "--input".into(),
            directory
                .path()
                .join("missing-input")
                .to_string_lossy()
                .into_owned(),
            "--output".into(),
            output.to_string_lossy().into_owned()
        ])
        .is_err()
    );
    assert!(output.symlink_metadata().unwrap().file_type().is_symlink());
}
