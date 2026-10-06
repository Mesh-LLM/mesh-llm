use super::*;

fn measured(root: &Path, status: &str) -> Measured {
    Measured {
        evidence: root.to_owned(),
        output: json!({"schema_version":1,"status":status,"completed_receipts":1,
            "receipts":[{"status":"process_completed_manifest_bound","row_sha256":"observed"}],
            "effective_profile":{"path":"/owned/tools"},"prior_error":if status=="failed"{json!("earlier protocol refused")}else{Value::Null}}),
        deadline: Instant::now() + Duration::from_secs(30),
        cancellation: process::Cancellation::default(),
    }
}
fn published(root: &Path) -> Value {
    serde_json::from_slice(&fs::read(root.join("run-receipt.json")).unwrap()).unwrap()
}
#[test]
fn parity_final_admission_success_publishes_only_after_observed_result_admitted() {
    let root = tempfile::tempdir().unwrap();
    let observed = measured(root.path(), "completed");
    assert!(!root.path().join("run-receipt.json").exists());
    let expected = observed.output.clone();
    assert_eq!(observed.publish(Ok(())).unwrap(), expected);
    assert_eq!(published(root.path()), expected);
    root.close().unwrap();
}
#[test]
fn parity_final_admission_prior_failure_retains_error_rows_and_profile() {
    let root = tempfile::tempdir().unwrap();
    let observed = measured(root.path(), "failed");
    let expected = observed.output.clone();
    assert!(observed.publish(Err("finish failed".into())).is_err());
    let actual = published(root.path());
    assert_eq!(actual["status"], "failed");
    for key in ["prior_error", "receipts", "effective_profile"] {
        assert_eq!(actual[key], expected[key]);
    }
    assert_eq!(actual["terminal_errors"], json!(["finish failed"]));
    root.close().unwrap();
}
#[test]
fn parity_final_admission_cancelled_after_measurement_refuses_completed_receipt() {
    let root = tempfile::tempdir().unwrap();
    let observed = measured(root.path(), "completed");
    observed.cancellation.cancel();
    assert!(observed.publish(Ok(())).is_err());
    let actual = published(root.path());
    assert_eq!(actual["status"], "refused");
    assert_eq!(actual["completed_receipts"], 1);
    assert_eq!(
        actual["terminal_errors"],
        json!(["local parity admission cancelled"])
    );
    root.close().unwrap();
}
#[test]
fn parity_final_admission_expired_after_measurement_preserves_observed_rows() {
    let root = tempfile::tempdir().unwrap();
    let mut observed = measured(root.path(), "completed");
    observed.deadline = Instant::now();
    assert!(observed.publish(Ok(())).is_err());
    let actual = published(root.path());
    assert_eq!(actual["status"], "refused");
    assert_eq!(actual["receipts"][0]["row_sha256"], "observed");
    assert_eq!(
        actual["terminal_errors"],
        json!(["local parity admission deadline"])
    );
    root.close().unwrap();
}
#[test]
fn parity_final_admission_source_guard_error_is_preserved_with_failed_final_receipt() {
    let root = tempfile::tempdir().unwrap();
    let observed = measured(root.path(), "completed");
    let error = observed
        .publish(Err("source authority changed".into()))
        .unwrap_err();
    assert_eq!(error.to_string(), "source authority changed");
    let actual = published(root.path());
    assert_eq!(actual["status"], "refused");
    assert_eq!(actual["completed_receipts"], 1);
    assert_eq!(
        actual["terminal_errors"],
        json!(["source authority changed"])
    );
    root.close().unwrap();
}
