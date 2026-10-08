use super::super::{
    Digest, SourceFamilyPlan, TestPackageInputs, WorkflowRun, aggregate::aggregate,
    feedback::verify,
};
use super::*;
use serde_json::json;
use std::{fs, path::PathBuf};

const IDENTITY: &[u8] =
    include_bytes!("../../../../tests/migration_canary_receipts/fixtures/identity.json");
const PLAN: &[u8] =
    include_bytes!("../../../../tests/migration_canary_receipts/fixtures/plan.json");
const RESULTS: &[u8] =
    include_bytes!("../../../../tests/migration_canary_receipts/fixtures/dense.jsonl");

fn context() -> ReceiptContext {
    ReceiptContext::from_test_package(
        TestPackageInputs {
            identity: serde_json::from_slice(IDENTITY).unwrap(),
            identity_sha256: Digest::of_bytes(IDENTITY),
            plan: SourceFamilyPlan::parse(PLAN).unwrap(),
        },
        WorkflowRun {
            run_id: "123".into(),
            run_attempt: "4".to_owned().try_into().unwrap(),
        },
    )
    .unwrap()
}
fn source(root: &Path, context: &ReceiptContext) -> PathBuf {
    let directory = root.join("dense");
    fs::create_dir(&directory).unwrap();
    fs::write(directory.join("results.jsonl"), RESULTS).unwrap();
    let receipt = json!({"candidate":context.package.identity.candidate,"family":"dense",
        "identity_sha256":context.package.identity_sha256,"outcome":"failure",
        "pass_id":context.package.identity.pass_id,"results_sha256":Digest::of_bytes(RESULTS),
        "run_attempt":"4","run_id":"123","runner":"fixture"});
    fs::write(
        directory.join("receipt.json"),
        serde_json::to_vec(&receipt).unwrap(),
    )
    .unwrap();
    directory
}

#[test]
fn aggregate_export_preserves_missing_infrastructure_and_selected_candidate() {
    let root = tempfile::tempdir().unwrap();
    let context = context();
    source(root.path(), &context);
    let report = aggregate(&context, root.path()).unwrap();
    assert_eq!(report.state, AggregateState::InfrastructureRetryable);
    let destination = root.path().join("feedback");
    let published = publish(&context, &report, &destination).unwrap().unwrap();
    let verified = verify(
        &context,
        &destination,
        FeedbackState::InfrastructureRetryable,
    )
    .unwrap();
    assert_eq!(verified.candidate_failures()[0].as_str(), "dense");
    assert!(
        verified
            .missing_infrastructure()
            .iter()
            .any(|family| family.as_str() == "hybrid")
    );
    assert!(outputs(Some(&published)).contains("feedback_ready=true\n"));
    assert!(!destination.join("hybrid").exists());
}

#[test]
fn aggregate_export_refuses_selected_receipt_replacement_and_reclassification() {
    for mutation in ["receipt", "class"] {
        let root = tempfile::tempdir().unwrap();
        let context = context();
        let directory = source(root.path(), &context);
        let report = aggregate(&context, root.path()).unwrap();
        match mutation {
            "receipt" => {
                let path = directory.join("receipt.json");
                let mut value: serde_json::Value =
                    serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                value["runner"] = json!("replacement-runner");
                fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
            }
            "class" => fs::write(
                directory.join("memory-admission.json"),
                br#"{"status":"refused"}"#,
            )
            .unwrap(),
            _ => unreachable!(),
        }
        let destination = root.path().join("feedback");
        assert!(
            publish(&context, &report, &destination).is_err(),
            "{mutation}"
        );
        assert!(!destination.exists());
    }
}

#[test]
fn aggregate_export_refuses_unsafe_extra_evidence_before_publication() {
    let root = tempfile::tempdir().unwrap();
    let context = context();
    let directory = source(root.path(), &context);
    let report = aggregate(&context, root.path()).unwrap();
    std::os::unix::fs::symlink("results.jsonl", directory.join("extra.log")).unwrap();
    let destination = root.path().join("feedback");
    assert!(publish(&context, &report, &destination).is_err());
    assert!(!destination.exists());
}

#[test]
fn aggregate_terminal_state_never_exports_feedback() {
    let root = tempfile::tempdir().unwrap();
    let context = context();
    let directory = source(root.path(), &context);
    fs::write(directory.join("results.jsonl"), b"corrupt results").unwrap();
    let report = aggregate(&context, root.path()).unwrap();
    assert_eq!(report.state, AggregateState::TerminalContract);
    let destination = root.path().join("feedback");
    assert!(publish(&context, &report, &destination).unwrap().is_none());
    assert!(!destination.exists());
    assert_eq!(outputs(None), "feedback_ready=false\n");
}
