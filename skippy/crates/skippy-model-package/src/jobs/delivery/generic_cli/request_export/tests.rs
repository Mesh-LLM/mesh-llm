use super::*;
use crate::jobs::delivery::ModelMount;
fn request(large: bool) -> Value {
    let (mut worker, mut mounts, plan) = super::super::super::generic::facade_fixture();
    if large {
        worker["operator"]["conversion"]["source_files"] = json!((0..1024).map(|i| {
            json!({"path":format!("/models/target/tensor-{i:04}.safetensors"),"sha256":"1".repeat(64)})
        }).collect::<Vec<_>>());
    }
    mounts.push(ModelMount {
        repo: "fixture/requests".into(),
        revision: "a".repeat(40),
        mount_path: "/models/requests".into(),
    });
    json!({"schema_version":1,"delivery":{"schema_version":1,"namespace":"fixture","worker_input":worker,"mounts":mounts,"cpu_plan":plan},"request_destination":{"path":"/models/requests/worker-input.json","repo":"fixture/requests","revision":"a".repeat(40)}})
}
fn invoke(temp: &Path, operation: &str, value: &Value, extra: &[&str]) -> Result<bool> {
    let path = temp.join("input.json");
    std::fs::write(&path, serde_json::to_vec(value)?)?;
    let mut args: Vec<std::ffi::OsString> = ["model-package-generic-jobs", operation, "--input"]
        .map(Into::into)
        .into();
    args.push(path.into_os_string());
    args.push("--output-directory".into());
    args.push(temp.join(operation).into_os_string());
    args.extend(extra.iter().map(|v| std::ffi::OsString::from(*v)));
    super::super::run_args(args)
}
#[test]
fn offline_export_actual_dispatch_large_roster_roundtrips_canonical_bytes_and_locator() {
    let temp = tempfile::tempdir().unwrap();
    let input = request(true);
    assert!(invoke(temp.path(), "export-request", &input, &[]).unwrap());
    let output = temp.path().join("export-request");
    let bytes = std::fs::read(output.join("worker-input.json")).unwrap();
    assert!(bytes.len() > 65536);
    assert_eq!(
        bytes,
        serde_json::to_vec(&input["delivery"]["worker_input"]).unwrap()
    );
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        value["operator"]["conversion"]["source_files"]
            .as_array()
            .unwrap()
            .len(),
        1024
    );
    let locator: MountedRequest =
        serde_json::from_slice(&std::fs::read(output.join("mounted-request.json")).unwrap())
            .unwrap();
    let delivery: Input =
        serde_json::from_slice(&std::fs::read(output.join("delivery.json")).unwrap()).unwrap();
    locator.admit(&bytes, &delivery.mounts).unwrap();
    assert_eq!(
        locator.sha256,
        super::super::super::admission::digest(&bytes)
    );
    let result: Value =
        serde_json::from_slice(&std::fs::read(output.join("result.json")).unwrap()).unwrap();
    assert_eq!(result["delivery_input_sha256"], hash(&delivery).unwrap());
    assert_eq!(result["submitted"], false);
    assert_eq!(result["mount_bytes_observed"], false);
    let delivery_value = serde_json::to_value(&delivery).unwrap();
    assert!(invoke(temp.path(), "prepare", &delivery_value, &[]).unwrap());
    assert_eq!(
        bytes,
        std::fs::read(temp.path().join("prepare/worker-input.json")).unwrap()
    );
    let prepared: Value =
        serde_json::from_slice(&std::fs::read(temp.path().join("prepare/result.json")).unwrap())
            .unwrap();
    assert_eq!(prepared["worker_input_sha256"], locator.sha256);
    assert_eq!(prepared["worker_input_byte_size"], bytes.len() as u64);
    use std::os::unix::fs::PermissionsExt as _;
    assert_eq!(
        std::fs::metadata(output.join("worker-input.json"))
            .unwrap()
            .permissions()
            .mode()
            & 0o077,
        0
    );
    temp.close().unwrap();
}
#[test]
fn offline_export_refuses_unmounted_destination_and_invalid_worker_before_output() {
    for (pointer, change) in [
        (
            "/request_destination/path",
            json!("/unmounted/worker-input.json"),
        ),
        ("/request_destination/revision", json!("b".repeat(40))),
        (
            "/delivery/worker_input/workflow",
            json!("unreviewed-workflow"),
        ),
        (
            "/delivery/worker_input/operator/conversion/credential_file",
            json!("/ambient/token"),
        ),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let mut input = request(true);
        *input.pointer_mut(pointer).unwrap() = change;
        assert!(invoke(temp.path(), "export-request", &input, &[]).is_err());
        assert!(!temp.path().join("export-request").exists());
        temp.close().unwrap();
    }
    let temp = tempfile::tempdir().unwrap();
    assert!(
        invoke(
            temp.path(),
            "export-request",
            &request(false),
            &["--confirm-submission"]
        )
        .is_err()
    );
    assert!(!temp.path().join("export-request").exists());
    temp.close().unwrap();
}
#[test]
fn canonical_prepare_export_is_private_fresh_and_refuses_leaf_replacement() {
    let temp = tempfile::tempdir().unwrap();
    let input = request(false);
    assert!(invoke(temp.path(), "prepare", &input["delivery"], &[]).unwrap());
    let root = temp.path().join("prepare");
    let before = std::fs::read(root.join("worker-input.json")).unwrap();
    assert_eq!(
        before,
        serde_json::to_vec(&input["delivery"]["worker_input"]).unwrap()
    );
    assert!(worker_file(&root, &json!({"changed":true})).is_err());
    assert_eq!(
        before,
        std::fs::read(root.join("worker-input.json")).unwrap()
    );
    assert!(invoke(temp.path(), "prepare", &input["delivery"], &[]).is_err());
    temp.close().unwrap();
}
