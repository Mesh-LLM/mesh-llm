use super::*;
fn fixture() -> (tempfile::TempDir, Request) {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let source = root.join("mounted");
    std::fs::create_dir(&source).unwrap();
    let values=[("README.md",b"original card\n".to_vec()),("skippy-convert-manifest.json",serde_json::to_vec(&json!({"expected_splits":2,"output_basename":"model","source_repo":"original/bf16","target_prefix":"BF16"})).unwrap()),("model-00001-of-00002.gguf",b"GGUFfirst".to_vec()),("model-00002-of-00002.gguf",b"GGUFsecond".to_vec()),("status.json",b"{\"phase\":\"verified\"}".to_vec())];
    let mut files = Vec::new();
    for (name, bytes) in values {
        std::fs::write(source.join(name), &bytes).unwrap();
        files.push(FilePin {
            name: name.into(),
            sha256: admission::digest(&bytes),
            byte_size: bytes.len() as u64,
        });
    }
    let request = Request {
        schema_version: 1,
        repo: "fixture/converted".into(),
        revision: "a".repeat(40),
        source_directory: source,
        work_directory: root.join("work"),
        target_prefix: "BF16".into(),
        output_basename: "model".into(),
        requested_splits: 1,
        files,
        timeout_seconds: 10,
    };
    (temp, request)
}
#[test]
fn mounted_upload_workspace_preserves_complete_effective_roster_and_sidecar_bytes() {
    let (temp, request) = fixture();
    let receipt = execute(
        &request,
        Instant::now() + Duration::from_secs(10),
        &Cancellation::default(),
    )
    .unwrap();
    assert_eq!(receipt["effective_splits"], 2);
    assert_eq!(receipt["built"], false);
    assert_eq!(receipt["converted"], false);
    for pin in &request.files {
        assert_eq!(
            std::fs::read(request.source_directory.join(&pin.name)).unwrap(),
            std::fs::read(request.work_directory.join("target/BF16").join(&pin.name)).unwrap()
        );
    }
    assert!(
        !request
            .work_directory
            .join("convert-manifest.json")
            .exists()
    );
    temp.close().unwrap();
}
#[test]
fn mounted_upload_workspace_refuses_overlap_missing_roster_and_wrong_pins_before_copy() {
    for mode in ["overlap", "missing", "pin", "effective"] {
        let (temp, mut request) = fixture();
        match mode {
            "overlap" => request.work_directory = request.source_directory.join("work"),
            "missing" => {
                request.files.pop();
            }
            "pin" => request.files[0].sha256 = "0".repeat(64),
            _ => request.requested_splits = 3,
        };
        assert!(
            execute(
                &request,
                Instant::now() + Duration::from_secs(10),
                &Cancellation::default()
            )
            .is_err()
        );
        assert!(!request.work_directory.exists());
        temp.close().unwrap();
    }
}
#[test]
fn mounted_upload_workspace_cancel_and_expiry_never_create_work() {
    let (temp, request) = fixture();
    let cancel = Cancellation::default();
    cancel.cancel();
    assert!(execute(&request, Instant::now() + Duration::from_secs(10), &cancel).is_err());
    assert!(execute(&request, Instant::now(), &Cancellation::default()).is_err());
    assert!(!request.work_directory.exists());
    temp.close().unwrap();
}
