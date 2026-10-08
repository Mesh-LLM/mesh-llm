use super::*;
fn fixture(root: &Path) -> Input {
    let body = serde_json::json!({"schema_version":1,"request_sha256":"a".repeat(64),"status":"CERTIFIED"});
    let bytes = serde_json::to_vec(&body).unwrap();
    let path = root.join("native-job.json");
    std::fs::write(&path, &bytes).unwrap();
    Input {
        schema_version: 1,
        repo: "fixture/evidence".into(),
        parent_commit: "b".repeat(40),
        artifact: Artifact {
            path,
            path_in_repo: "runs/native-job.json".into(),
            sha256: digest(&bytes),
            byte_size: bytes.len() as u64,
        },
        receipt_request_sha256: "a".repeat(64),
        credential_file: root.join("private-token"),
        execution_timeout_ms: 10000,
    }
}
#[test]
fn regular_receipt_admits_exact_native_job_bytes_and_closed_request_correlation() {
    let root = tempfile::tempdir().unwrap();
    let base = root.path().canonicalize().unwrap();
    let input = fixture(&base);
    validate(&input).unwrap();
    let admitted = artifact(&input, Instant::now() + Duration::from_secs(1)).unwrap();
    assert_eq!(admitted.identity.sha256, input.artifact.sha256);
    assert_eq!(admitted.identity.byte_size, input.artifact.byte_size);
    drop(admitted);
    let mut value = serde_json::to_value(&input).unwrap();
    value["done"] = serde_json::json!(true);
    assert!(serde_json::from_value::<Input>(value).is_err());
    root.close().unwrap();
}
#[test]
fn regular_receipt_refuses_wrong_body_pin_request_and_model_blob_without_publication() {
    let root = tempfile::tempdir().unwrap();
    let base = root.path().canonicalize().unwrap();
    let mut input = fixture(&base);
    input.receipt_request_sha256 = "c".repeat(64);
    assert!(artifact(&input, Instant::now() + Duration::from_secs(1)).is_err());
    let mut input = fixture(&base);
    input.artifact.sha256 = "d".repeat(64);
    assert!(artifact(&input, Instant::now() + Duration::from_secs(1)).is_err());
    let mut input = fixture(&base);
    std::fs::write(&input.artifact.path, b"GGUF").unwrap();
    input.artifact.byte_size = 4;
    input.artifact.sha256 = digest(b"GGUF");
    assert!(artifact(&input, Instant::now() + Duration::from_secs(1)).is_err());
    let mut input = fixture(&base);
    input.artifact.byte_size = 1024 * 1024 + 1;
    assert!(validate(&input).is_err());
    let mut input = fixture(&base);
    input.artifact.path_in_repo = "../native-job.json".into();
    assert!(validate(&input).is_err());
    assert!(artifact(&fixture(&base), Instant::now()).is_err());
    root.close().unwrap();
}
