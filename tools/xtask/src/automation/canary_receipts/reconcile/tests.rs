use super::super::{Digest, SourceFamilyPlan, TestPackageInputs, WorkflowRun};
use super::*;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};
const IDENTITY: &[u8] =
    include_bytes!("../../../../tests/migration_canary_receipts/fixtures/identity.json");
const PLAN: &[u8] =
    include_bytes!("../../../../tests/migration_canary_receipts/fixtures/plan.json");
const DENSE: &[u8] =
    include_bytes!("../../../../tests/migration_canary_receipts/fixtures/dense.jsonl");
const HYBRID: &[u8] =
    include_bytes!("../../../../tests/migration_canary_receipts/fixtures/hybrid.jsonl");
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
fn family(value: &str) -> Family {
    value.to_owned().try_into().unwrap()
}
fn receipt(root: &Path, name: &str, family_name: &str, outcome: &str, attempt: &str) -> PathBuf {
    let directory = root.join(name);
    fs::create_dir_all(&directory).unwrap();
    let results = if family_name == "hybrid" {
        HYBRID
    } else {
        DENSE
    };
    fs::write(directory.join("results.jsonl"), results).unwrap();
    fs::write(directory.join("phase.log"), b"retained useful evidence").unwrap();
    fs::write(directory.join("receipt.json"),serde_json::to_vec(&json!({"candidate":"a".repeat(40),"family":family_name,"identity_sha256":Digest::of_bytes(IDENTITY),"outcome":outcome,"pass_id":"repair-1","results_sha256":Digest::of_bytes(results),"run_attempt":attempt,"run_id":"123","runner":"fixture"})).unwrap()).unwrap();
    directory
}
fn previous(root: &Path, context: &ReceiptContext, retain_candidate: bool) -> PathBuf {
    let admitted = if retain_candidate {
        let source = receipt(root, "candidate", "dense", "failure", "2");
        vec![FamilyEvidence::admit(context, &family("dense"), &source).unwrap()]
    } else {
        Vec::new()
    };
    let output = root.join("previous");
    FeedbackDraft::new(
        context,
        FeedbackState::InfrastructureRetryable,
        admitted,
        [family("hybrid")].into(),
        vec!["infrastructure absent".into()],
    )
    .unwrap()
    .publish(context, &output)
    .unwrap();
    output
}
fn change_receipt(directory: &Path, key: &str, value: Value) {
    let path = directory.join("receipt.json");
    let mut item: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    item[key] = value;
    fs::write(path, serde_json::to_vec(&item).unwrap()).unwrap();
}

#[test]
fn successful_retry_retains_candidate_snapshot_and_publishes_only_candidate_feedback() {
    let root = tempfile::tempdir().unwrap();
    let context = context();
    let previous = previous(root.path(), &context, true);
    let retries = root.path().join("retries");
    receipt(&retries, "retry-hybrid", "hybrid", "success", "4");
    let report = reconcile(&context, &previous, &retries, FamilyJobResult::Success).unwrap();
    assert_eq!(report.state, ReconcileState::CandidateRepairable);
    assert_eq!(report.passed, vec![family("hybrid")]);
    assert_eq!(report.candidate_failures, [family("dense")].into());
    fs::write(previous.join("dense/phase.log"), b"late replacement").unwrap();
    let output = root.path().join("reconciled");
    let published = report.publish_feedback(&context, &output).unwrap().unwrap();
    assert_eq!(published.state(), FeedbackState::CandidateRepairable);
    let verified = feedback::verify(&context, &output, FeedbackState::CandidateRepairable).unwrap();
    assert_eq!(verified.candidate_failures(), &[family("dense")]);
    assert!(verified.infrastructure_failures().is_empty());
    assert_eq!(
        fs::read(output.join("dense/phase.log")).unwrap(),
        b"retained useful evidence"
    );
}

#[test]
fn successful_receipts_require_successful_retry_graph_for_green() {
    for (job, expected) in [
        (FamilyJobResult::Success, ReconcileState::Green),
        (
            FamilyJobResult::Failure,
            ReconcileState::InfrastructureExhausted,
        ),
        (FamilyJobResult::Cancelled, ReconcileState::TerminalContract),
        (FamilyJobResult::Skipped, ReconcileState::TerminalContract),
    ] {
        let root = tempfile::tempdir().unwrap();
        let context = context();
        let previous = previous(root.path(), &context, false);
        let retries = root.path().join("retries");
        receipt(&retries, "retry-hybrid", "hybrid", "success", "4");
        let report = reconcile(&context, &previous, &retries, job).unwrap();
        assert_eq!(report.state, expected);
        let output = root.path().join("feedback");
        assert!(
            report
                .publish_feedback(&context, &output)
                .unwrap()
                .is_none()
        );
        assert!(!output.exists());
    }
}

#[test]
fn missing_or_remaining_infrastructure_is_exhausted_without_second_retry_feedback() {
    for fault in ["missing", "cancelled", "environment"] {
        let root = tempfile::tempdir().unwrap();
        let context = context();
        let previous = previous(root.path(), &context, true);
        let retries = root.path().join("retries");
        if fault != "missing" {
            let directory = receipt(
                &retries,
                "retry-hybrid",
                "hybrid",
                if fault == "cancelled" {
                    "cancelled"
                } else {
                    "failure"
                },
                "4",
            );
            if fault == "environment" {
                fs::write(
                    directory.join("memory-admission.json"),
                    b"{\"status\":\"failed\"}",
                )
                .unwrap();
            }
        }
        let report = reconcile(&context, &previous, &retries, FamilyJobResult::Failure).unwrap();
        assert_eq!(report.state, ReconcileState::InfrastructureExhausted);
        assert_eq!(report.infrastructure_failures, [family("hybrid")].into());
        assert!(
            report
                .publish_feedback(&context, &root.path().join("feedback"))
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn retry_candidate_failures_join_retained_candidates_after_infrastructure_clears() {
    let root = tempfile::tempdir().unwrap();
    let context = context();
    let previous = previous(root.path(), &context, true);
    let retries = root.path().join("retries");
    receipt(&retries, "retry-hybrid", "hybrid", "failure", "4");
    let report = reconcile(&context, &previous, &retries, FamilyJobResult::Failure).unwrap();
    assert_eq!(report.state, ReconcileState::CandidateRepairable);
    assert_eq!(
        report.candidate_failures,
        [family("dense"), family("hybrid")].into()
    );
    let output = root.path().join("feedback");
    report.publish_feedback(&context, &output).unwrap().unwrap();
    assert_eq!(
        feedback::verify(&context, &output, FeedbackState::CandidateRepairable)
            .unwrap()
            .candidate_failures(),
        &[family("dense"), family("hybrid")]
    );
}

#[test]
fn retry_custody_rejects_foreign_extraneous_duplicate_and_undeclared_artifacts() {
    for fault in [
        "foreign",
        "candidate-family",
        "duplicate",
        "future",
        "digest",
        "extra-root-file",
        "empty-directory",
        "wrong-pass",
    ] {
        let root = tempfile::tempdir().unwrap();
        let context = context();
        let previous = previous(root.path(), &context, false);
        let retries = root.path().join("retries");
        let directory = receipt(&retries, "retry-hybrid", "hybrid", "success", "4");
        match fault {
            "foreign" => change_receipt(&directory, "run_id", json!("999")),
            "candidate-family" => {
                receipt(&retries, "extra-dense", "dense", "failure", "4");
            }
            "duplicate" => {
                receipt(&retries, "second-hybrid", "hybrid", "success", "4");
            }
            "future" => change_receipt(&directory, "run_attempt", json!("5")),
            "digest" => change_receipt(&directory, "results_sha256", json!("e".repeat(64))),
            "extra-root-file" => fs::write(retries.join("unclaimed"), b"extra").unwrap(),
            "empty-directory" => fs::create_dir(retries.join("unclaimed")).unwrap(),
            "wrong-pass" => change_receipt(&directory, "pass_id", json!("verify-1")),
            _ => unreachable!(),
        }
        let report = reconcile(&context, &previous, &retries, FamilyJobResult::Success).unwrap();
        assert_eq!(report.state, ReconcileState::TerminalContract, "{fault}");
        assert!(
            report
                .publish_feedback(&context, &root.path().join("feedback"))
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn newest_retry_failure_never_falls_back_to_older_success() {
    let root = tempfile::tempdir().unwrap();
    let context = context();
    let previous = previous(root.path(), &context, false);
    let retries = root.path().join("retries");
    receipt(&retries, "old", "hybrid", "success", "3");
    receipt(&retries, "new", "hybrid", "failure", "4");
    let report = reconcile(&context, &previous, &retries, FamilyJobResult::Failure).unwrap();
    assert_eq!(report.state, ReconcileState::CandidateRepairable);
    assert!(report.passed.is_empty());
    assert_eq!(
        report.selected_attempts[&family("hybrid")],
        "4".to_owned().try_into().unwrap()
    );
}

#[test]
fn invalid_prior_feedback_refuses_before_retry_admission() {
    let root = tempfile::tempdir().unwrap();
    let context = context();
    let previous = previous(root.path(), &context, false);
    let metadata = previous.join("feedback.json");
    let mut payload: Value = serde_json::from_slice(&fs::read(&metadata).unwrap()).unwrap();
    payload["identity_sha256"] = json!("e".repeat(64));
    fs::write(metadata, serde_json::to_vec(&payload).unwrap()).unwrap();
    assert!(
        reconcile(
            &context,
            &previous,
            &root.path().join("absent"),
            FamilyJobResult::Success
        )
        .is_err()
    );
}

#[test]
fn linked_retry_artifact_is_terminal_without_reading_external_bytes() {
    use std::os::unix::fs::symlink;
    let root = tempfile::tempdir().unwrap();
    let context = context();
    let previous = previous(root.path(), &context, false);
    let retries = root.path().join("retries");
    fs::create_dir(&retries).unwrap();
    let outside = receipt(root.path(), "outside", "hybrid", "success", "4");
    symlink(&outside, retries.join("retry")).unwrap();
    let report = reconcile(&context, &previous, &retries, FamilyJobResult::Success).unwrap();
    assert_eq!(report.state, ReconcileState::TerminalContract);
    assert_eq!(
        fs::read(outside.join("phase.log")).unwrap(),
        b"retained useful evidence"
    );
}
