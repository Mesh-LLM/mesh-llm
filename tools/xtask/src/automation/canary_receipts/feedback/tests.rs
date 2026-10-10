use super::super::{SourceFamilyPlan, TestPackageInputs, WorkflowRun};
use super::*;
use serde_json::{Value, json};
use std::fs;

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
fn family(name: &str) -> Family {
    name.to_owned().try_into().unwrap()
}
fn source(root: &Path, context: &ReceiptContext, outcome: &str) -> PathBuf {
    let directory = root.join("source");
    fs::create_dir(&directory).unwrap();
    fs::write(directory.join("results.jsonl"), RESULTS).unwrap();
    fs::write(directory.join("phase.log"), b"useful candidate trace\n").unwrap();
    let receipt = json!({"candidate":context.package.identity.candidate,"family":"dense", "identity_sha256":context.package.identity_sha256,
        "outcome":outcome,"pass_id":context.package.identity.pass_id,"results_sha256":Digest::of_bytes(RESULTS),"run_attempt":"4","run_id":"123","runner":"fixture"});
    fs::write(
        directory.join("receipt.json"),
        serde_json::to_vec(&receipt).unwrap(),
    )
    .unwrap();
    directory
}
fn draft(context: &ReceiptContext, directory: &Path, state: FeedbackState) -> FeedbackDraft {
    let evidence = FamilyEvidence::admit(context, &family("dense"), directory).unwrap();
    let missing = if state == FeedbackState::InfrastructureRetryable {
        [family("hybrid")].into()
    } else {
        BTreeSet::new()
    };
    FeedbackDraft::new(
        context,
        state,
        vec![evidence],
        missing,
        vec!["candidate failed".into()],
    )
    .unwrap()
}
fn prepared() -> (tempfile::TempDir, ReceiptContext, PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let context = context();
    let source = source(root.path(), &context, "failure");
    let output = root.path().join("feedback");
    draft(&context, &source, FeedbackState::CandidateRepairable)
        .publish(&context, &output)
        .unwrap();
    (root, context, output)
}
fn mutate_metadata(root: &Path, operation: impl FnOnce(&mut Value)) {
    let path = root.join("feedback.json");
    let mut value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    operation(&mut value);
    fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
}

#[test]
fn feedback_export_preserves_owned_evidence_after_original_source_changes() {
    let root = tempfile::tempdir().unwrap();
    let context = context();
    let source = source(root.path(), &context, "failure");
    let draft = draft(&context, &source, FeedbackState::CandidateRepairable);
    fs::write(
        source.join("results.jsonl"),
        b"replaced source after admission",
    )
    .unwrap();
    let output = root.path().join("feedback");
    let published = draft.publish(&context, &output).unwrap();
    assert_eq!(published.directory(), output.canonicalize().unwrap());
    assert_eq!(published.state(), FeedbackState::CandidateRepairable);
    assert_eq!(
        fs::read(output.join("dense/results.jsonl")).unwrap(),
        RESULTS
    );
    let verified = verify(&context, &output, FeedbackState::CandidateRepairable).unwrap();
    assert_eq!(verified.state(), FeedbackState::CandidateRepairable);
    assert_eq!(verified.candidate_failures(), &[family("dense")]);
    assert!(verified.infrastructure_failures().is_empty());
    assert_eq!(
        verified.candidate_evidence(&context).unwrap()[0].family(),
        &family("dense")
    );
}

#[test]
fn feedback_missing_infrastructure_is_explicit_and_candidate_evidence_required() {
    let root = tempfile::tempdir().unwrap();
    let context = context();
    let source = source(root.path(), &context, "failure");
    let output = root.path().join("feedback");
    draft(&context, &source, FeedbackState::InfrastructureRetryable)
        .publish(&context, &output)
        .unwrap();
    let verified = verify(&context, &output, FeedbackState::InfrastructureRetryable).unwrap();
    assert_eq!(verified.missing_infrastructure(), [family("hybrid")].into());
    assert_eq!(verified.candidate_failures(), &[family("dense")]);
    assert!(!output.join("hybrid").exists());
    mutate_metadata(&output, |value| {
        value["evidence_sha256"]
            .as_object_mut()
            .unwrap()
            .remove("dense")
            .map(drop)
            .unwrap()
    });
    assert!(verify(&context, &output, FeedbackState::InfrastructureRetryable).is_err());
}

#[test]
fn feedback_contract_refuses_identity_attempt_state_and_family_corruption() {
    for (key, value) in [
        ("schema", json!(1)),
        ("identity_sha256", json!("e".repeat(64))),
        ("candidate", json!("c".repeat(40))),
        ("source_pass", json!("verify-1")),
        ("run_id", json!("999")),
        ("run_attempt", json!("1")),
        ("run_attempt", json!("5")),
        ("run_attempt", json!("04")),
        ("failure_stage", json!("build")),
        ("state", json!("green")),
        ("state", json!("infrastructure_retryable")),
        ("repairable", json!(false)),
        ("failure_class", json!("infrastructure")),
        ("candidate_failures", json!([])),
        ("candidate_failures", json!(["dense", "dense"])),
        ("candidate_failures", json!(["hybrid", "dense"])),
        ("infrastructure_failures", json!(["dense"])),
        ("failed_families", json!([])),
        ("unexpected", json!(true)),
    ] {
        let (_root, context, output) = prepared();
        mutate_metadata(&output, |metadata| metadata[key] = value);
        assert!(
            verify(&context, &output, FeedbackState::CandidateRepairable).is_err(),
            "{key}"
        );
    }
}

#[test]
fn feedback_exact_closure_rejects_changed_added_removed_and_empty_members() {
    for fault in [
        "changed",
        "added",
        "removed",
        "empty-directory",
        "top-file",
        "path",
        "empty-manifest",
    ] {
        let (_root, context, output) = prepared();
        match fault {
            "changed" => fs::write(output.join("dense/phase.log"), b"changed").unwrap(),
            "added" => fs::write(output.join("dense/unclaimed.log"), b"added").unwrap(),
            "removed" => fs::remove_file(output.join("dense/phase.log")).unwrap(),
            "empty-directory" => fs::create_dir(output.join("dense/empty")).unwrap(),
            "top-file" => fs::write(output.join("unclaimed.log"), b"added").unwrap(),
            "path" => mutate_metadata(&output, |metadata| {
                metadata["evidence_sha256"]["dense"]["../outside"] = json!("e".repeat(64))
            }),
            "empty-manifest" => mutate_metadata(&output, |metadata| {
                metadata["evidence_sha256"]["dense"] = json!({})
            }),
            _ => unreachable!(),
        }
        assert!(
            verify(&context, &output, FeedbackState::CandidateRepairable).is_err(),
            "{fault}"
        );
    }
}

#[test]
fn feedback_receipt_custody_refuses_success_corruption_and_wrong_family() {
    for fault in ["success", "identity", "digest", "family", "attempt"] {
        let root = tempfile::tempdir().unwrap();
        let context = context();
        let source = source(
            root.path(),
            &context,
            if fault == "success" {
                "success"
            } else {
                "failure"
            },
        );
        if fault != "success" {
            let path = source.join("receipt.json");
            let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            match fault {
                "identity" => value["identity_sha256"] = json!("e".repeat(64)),
                "digest" => value["results_sha256"] = json!("e".repeat(64)),
                "family" => value["family"] = json!("hybrid"),
                "attempt" => value["run_attempt"] = json!("5"),
                _ => unreachable!(),
            }
            fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
        }
        assert!(
            FamilyEvidence::admit(&context, &family("dense"), &source).is_err(),
            "{fault}"
        );
    }
}

#[test]
fn feedback_cancelled_receipt_without_results_remains_infrastructure_evidence() {
    let root = tempfile::tempdir().unwrap();
    let context = context();
    let source = source(root.path(), &context, "cancelled");
    fs::remove_file(source.join("results.jsonl")).unwrap();
    let evidence = FamilyEvidence::admit(&context, &family("dense"), &source).unwrap();
    assert!(!evidence.is_candidate());
    let draft = FeedbackDraft::new(
        &context,
        FeedbackState::InfrastructureRetryable,
        vec![evidence],
        BTreeSet::new(),
        vec![],
    )
    .unwrap();
    let output = root.path().join("feedback");
    draft.publish(&context, &output).unwrap();
    let verified = verify(&context, &output, FeedbackState::InfrastructureRetryable).unwrap();
    assert_eq!(verified.infrastructure_failures(), &[family("dense")]);
    assert!(verified.missing_infrastructure().is_empty());
}

#[test]
fn feedback_export_refuses_changed_owned_snapshot_and_existing_destination() {
    for fault in ["snapshot", "destination"] {
        let root = tempfile::tempdir().unwrap();
        let context = context();
        let source = source(root.path(), &context, "failure");
        let draft = draft(&context, &source, FeedbackState::CandidateRepairable);
        let destination = root.path().join("feedback");
        if fault == "snapshot" {
            fs::write(
                draft.evidence[&family("dense")]
                    .snapshot
                    .root()
                    .join("phase.log"),
                b"changed",
            )
            .unwrap();
        } else {
            fs::create_dir(&destination).unwrap();
            fs::write(destination.join("sentinel"), b"keep").unwrap();
        }
        assert!(draft.publish(&context, &destination).is_err());
        if fault == "destination" {
            assert_eq!(fs::read(destination.join("sentinel")).unwrap(), b"keep");
        } else {
            assert!(!destination.exists());
        }
        assert!(!fs::read_dir(root.path()).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".canary-feedback-")
        }));
    }
}

#[test]
fn feedback_results_cap_and_unsafe_family_components_fail_before_export() {
    let root = tempfile::tempdir().unwrap();
    let context = context();
    let source = source(root.path(), &context, "failure");
    fs::OpenOptions::new()
        .write(true)
        .open(source.join("results.jsonl"))
        .unwrap()
        .set_len(super::super::storage::RESULTS_LIMIT + 1)
        .unwrap();
    assert!(FamilyEvidence::admit(&context, &family("dense"), &source).is_err());
    for name in [".", ".."] {
        assert!(contract::safe_family(&family(name)).is_err());
    }
}

#[cfg(unix)]
#[test]
fn feedback_symlinks_and_fifos_cannot_escape_or_block_capture() {
    use std::{
        ffi::CString,
        os::unix::{ffi::OsStrExt, fs::symlink},
    };
    for fault in ["link", "directory-link", "fifo"] {
        let root = tempfile::tempdir().unwrap();
        let context = context();
        let source = source(root.path(), &context, "failure");
        let outside = root.path().join("outside");
        fs::write(&outside, b"private outside bytes").unwrap();
        if fault == "link" {
            symlink(&outside, source.join("linked.log")).unwrap();
        } else if fault == "directory-link" {
            symlink(root.path(), source.join("linked-directory")).unwrap();
        } else {
            let name = CString::new(source.join("pipe").as_os_str().as_bytes()).unwrap();
            // SAFETY: the terminated path is inside this test's owned directory.
            assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        }
        assert!(FamilyEvidence::admit(&context, &family("dense"), &source).is_err());
        assert_eq!(fs::read(outside).unwrap(), b"private outside bytes");
    }
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[test]
fn feedback_directory_publication_refuses_existing_empty_target_atomically() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("stage");
    let destination = root.path().join("feedback");
    fs::create_dir(&source).unwrap();
    fs::create_dir(&destination).unwrap();
    fs::write(source.join("sentinel"), b"keep").unwrap();
    assert!(
        export::publish_directory(
            &source,
            &destination,
            &evidence::open_directory(root.path()).unwrap()
        )
        .is_err()
    );
    assert_eq!(fs::read(source.join("sentinel")).unwrap(), b"keep");
    assert_eq!(fs::read_dir(destination).unwrap().count(), 0);
}

#[test]
fn feedback_metadata_log_and_depth_budgets_refuse_without_publication() {
    for fault in ["metadata", "log", "depth"] {
        let root = tempfile::tempdir().unwrap();
        let context = context();
        let source = source(root.path(), &context, "failure");
        match fault {
            "metadata" => fs::OpenOptions::new()
                .write(true)
                .open(source.join("receipt.json"))
                .unwrap()
                .set_len(super::super::storage::RECEIPT_LIMIT + 1)
                .unwrap(),
            "log" => fs::OpenOptions::new()
                .write(true)
                .open(source.join("phase.log"))
                .unwrap()
                .set_len(256 * 1024 * 1024 + 1)
                .unwrap(),
            "depth" => {
                let mut directory = source.clone();
                for _ in 0..34 {
                    directory.push("nested");
                }
                fs::create_dir_all(directory).unwrap();
            }
            _ => unreachable!(),
        }
        assert!(FamilyEvidence::admit(&context, &family("dense"), &source).is_err());
        assert!(!root.path().join("feedback").exists());
    }
}

#[test]
fn feedback_parent_descriptor_detects_directory_substitution() {
    let root = tempfile::tempdir().unwrap();
    let parent = root.path().join("parent");
    let displaced = root.path().join("original-parent");
    fs::create_dir(&parent).unwrap();
    let owned = evidence::open_directory(&parent).unwrap();
    fs::rename(&parent, &displaced).unwrap();
    fs::create_dir(&parent).unwrap();
    assert!(evidence::same_directory(&owned, &parent).is_err());
    assert!(evidence::same_directory(&owned, &displaced).is_ok());
}

#[test]
fn admitted_feedback_binds_selected_receipt_attempt_and_exact_bytes() {
    let root = tempfile::tempdir().unwrap();
    let context = context();
    let source = source(root.path(), &context, "failure");
    let digest = Digest::of_bytes(&fs::read(source.join("receipt.json")).unwrap());
    let evidence = FamilyEvidence::admit(&context, &family("dense"), &source).unwrap();
    let attempt = "4".to_owned().try_into().unwrap();
    assert!(evidence.matches_selection(&attempt, &digest));
    assert!(!evidence.matches_selection(&"3".to_owned().try_into().unwrap(), &digest));
    assert!(!evidence.matches_selection(&attempt, &Digest::of_bytes(b"replaced receipt")));
}

#[test]
fn feedback_stage_substitution_before_publication_preserves_original_and_refuses_target() {
    let root = tempfile::tempdir().unwrap();
    let stage = root.path().join("stage");
    let displaced = root.path().join("original-stage");
    let destination = root.path().join("feedback");
    fs::create_dir(&stage).unwrap();
    fs::write(stage.join("evidence"), b"verified original").unwrap();
    let parent = evidence::open_directory(root.path()).unwrap();
    let owned_stage = evidence::open_directory(&stage).unwrap();
    fs::rename(&stage, &displaced).unwrap();
    fs::create_dir(&stage).unwrap();
    fs::write(stage.join("evidence"), b"unverified replacement").unwrap();
    assert!(
        export::publish_owned_directory(&stage, &destination, &parent, &owned_stage, || {})
            .is_err()
    );
    assert!(!destination.exists());
    assert_eq!(
        fs::read(displaced.join("evidence")).unwrap(),
        b"verified original"
    );
    assert_eq!(
        fs::read(stage.join("evidence")).unwrap(),
        b"unverified replacement"
    );
}

#[test]
fn feedback_stage_substitution_after_source_guard_cannot_complete_publication_custody() {
    let root = tempfile::tempdir().unwrap();
    let stage = root.path().join("stage");
    let displaced = root.path().join("original-stage");
    let destination = root.path().join("feedback");
    fs::create_dir(&stage).unwrap();
    fs::write(stage.join("evidence"), b"verified original").unwrap();
    let parent = evidence::open_directory(root.path()).unwrap();
    let owned_stage = evidence::open_directory(&stage).unwrap();
    let outcome =
        export::publish_owned_directory(&stage, &destination, &parent, &owned_stage, || {
            fs::rename(&stage, &displaced).unwrap();
            fs::create_dir(&stage).unwrap();
            fs::write(stage.join("evidence"), b"unverified replacement").unwrap();
        });
    assert!(outcome.is_err());
    assert_eq!(
        fs::read(displaced.join("evidence")).unwrap(),
        b"verified original"
    );
    assert_eq!(
        fs::read(destination.join("evidence")).unwrap(),
        b"unverified replacement"
    );
    assert!(evidence::same_directory(&owned_stage, &destination).is_err());
}
