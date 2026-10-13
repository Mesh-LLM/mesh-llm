use super::support::{DENSE, Fixture, IDENTITY, PLAN, context, family};
use crate::canary_receipts::{
    Digest, ErrorKind, ProducerIdentity, ReceiptContext, SourceFamilyPlan, TestPackageInputs,
    WorkflowRun, aggregate, validate_results,
};
use serde_json::json;
use std::fs;

#[test]
fn migration_canary_receipts_root_family_accepts_last_planned_with_matching_digest() {
    let given = Fixture::complete();
    let bytes = include_bytes!("fixtures/root-family-planned-last.jsonl");
    fs::write(given.results("dense"), bytes).unwrap();
    given.edit(
        "dense",
        "results_sha256",
        json!(Digest::of_bytes(bytes).as_str()),
    );
    let when = aggregate(&context(), &given.0).unwrap();
    assert!(when.is_green(), "{}", when.summary());
    assert!(when.github_outputs().is_some());
}

#[test]
fn migration_canary_receipts_root_family_rejects_last_foreign_with_matching_digest() {
    let given = Fixture::complete();
    let bytes = include_bytes!("fixtures/root-family-foreign-last.jsonl");
    fs::write(given.results("dense"), bytes).unwrap();
    given.edit(
        "dense",
        "results_sha256",
        json!(Digest::of_bytes(bytes).as_str()),
    );
    let when = aggregate(&context(), &given.0).unwrap();
    assert_eq!(when.failures[0].error.kind, ErrorKind::Results);
    assert!(when.github_outputs().is_none());
}

#[test]
fn migration_canary_receipts_root_result_rejects_last_foreign_family() {
    let bytes = std::str::from_utf8(DENSE).unwrap().replace(
        r#""family":"dense""#,
        r#""family":"dense","family":"foreign""#,
    );
    let given = context();
    let family = family("dense");
    let when = validate_results(bytes.as_bytes(), &family, given.model(&family).unwrap());
    assert_eq!(when.unwrap_err().kind, ErrorKind::Results);
}

#[test]
fn migration_canary_receipts_root_plan_rejects_last_empty_models() {
    let given = std::str::from_utf8(PLAN)
        .unwrap()
        .replace(r#""shards":"#, r#""selected_models": [], "shards":"#);
    let when = SourceFamilyPlan::parse(given.as_bytes());
    assert_eq!(when.err().unwrap().kind, ErrorKind::Plan);
}

#[test]
fn migration_canary_receipts_root_identity_rejects_last_foreign_run() {
    let bytes = std::str::from_utf8(IDENTITY)
        .unwrap()
        .replace(r#""run_id": "123""#, r#""run_id": "123", "run_id": "999""#);
    let given = TestPackageInputs {
        identity: serde_json::from_str(&bytes).unwrap(),
        identity_sha256: Digest::of_bytes(bytes.as_bytes()),
        plan: SourceFamilyPlan::parse(PLAN).unwrap(),
    };
    let when = ReceiptContext::from_test_package(
        given,
        WorkflowRun {
            run_id: "123".to_owned(),
            run_attempt: "4".to_owned().try_into().unwrap(),
        },
    );
    assert_eq!(when.err().unwrap().kind, ErrorKind::ReceiptIdentity);
}

#[test]
fn migration_canary_receipts_root_receipt_rejects_last_foreign_run() {
    let given = Fixture::complete();
    let path = given.0.join("dense/receipt.json");
    let bytes = fs::read_to_string(&path)
        .unwrap()
        .replace(r#""run_id":"123""#, r#""run_id":"123","run_id":"999""#);
    fs::write(path, bytes).unwrap();
    let when = aggregate(&context(), &given.0).unwrap();
    assert_eq!(when.failures[0].error.kind, ErrorKind::ReceiptIdentity);
    assert!(when.github_outputs().is_none());
}

#[test]
fn migration_canary_receipts_root_outcome_uses_last_success() {
    let given = Fixture::complete();
    let path = given.0.join("dense/receipt.json");
    let bytes = fs::read_to_string(&path).unwrap().replace(
        r#""outcome":"success""#,
        r#""outcome":"failure","outcome":"success""#,
    );
    fs::write(path, bytes).unwrap();
    let when = aggregate(&context(), &given.0).unwrap();
    assert!(when.is_green(), "{}", when.summary());
}

#[test]
fn migration_canary_receipts_root_outcome_rejects_last_failure_without_fallback() {
    let given = Fixture::complete();
    given.put("new-dense", "dense", "3", "success");
    let path = given.0.join("new-dense/receipt.json");
    let bytes = fs::read_to_string(&path).unwrap().replace(
        r#""outcome":"success""#,
        r#""outcome":"success","outcome":"failure""#,
    );
    fs::write(path, bytes).unwrap();
    let when = aggregate(&context(), &given.0).unwrap();
    assert_eq!(when.selected_attempts[&family("dense")].as_str(), "3");
    assert_eq!(when.failures[0].error.kind, ErrorKind::WorkerOutcome);
    assert!(when.github_outputs().is_none());
}

#[test]
fn migration_canary_receipts_root_digest_uses_last_matching_hash() {
    let given = Fixture::complete();
    let path = given.0.join("dense/receipt.json");
    let bytes = fs::read_to_string(&path).unwrap().replace(
        r#""results_sha256":"#,
        r#""results_sha256":null,"results_sha256":"#,
    );
    fs::write(path, bytes).unwrap();
    let when = aggregate(&context(), &given.0).unwrap();
    assert!(when.is_green(), "{}", when.summary());
}

#[test]
fn migration_canary_receipts_root_digest_rejects_last_null_hash() {
    let given = Fixture::complete();
    let path = given.0.join("dense/receipt.json");
    let bytes = fs::read_to_string(&path).unwrap().replace(
        r#""run_attempt":"#,
        r#""results_sha256":null,"run_attempt":"#,
    );
    fs::write(path, bytes).unwrap();
    let when = aggregate(&context(), &given.0).unwrap();
    assert_eq!(when.failures[0].error.kind, ErrorKind::ResultsDigest);
    assert!(when.github_outputs().is_none());
}

#[test]
fn migration_canary_receipts_root_duplicate_replaces_invalid_type_before_decoding() {
    let given = std::str::from_utf8(IDENTITY).unwrap().replace(
        r#""run_attempt": "2""#,
        r#""run_attempt": {}, "run_attempt": "2""#,
    );
    let when = serde_json::from_str::<ProducerIdentity>(&given);
    assert_eq!(when.unwrap().run_attempt.as_str(), "2");
}

#[test]
fn migration_canary_receipts_root_duplicate_rejects_last_invalid_type() {
    let given = std::str::from_utf8(IDENTITY).unwrap().replace(
        r#""run_attempt": "2""#,
        r#""run_attempt": "2", "run_attempt": {}"#,
    );
    let when = serde_json::from_str::<ProducerIdentity>(&given);
    assert!(when.is_err());
}

#[test]
fn migration_canary_receipts_root_decoder_preserves_first_field_position() {
    for given in [
        r#"{"run_id":42,"branch":[]}"#,
        r#"{"run_id":"123","branch":[],"run_id":42}"#,
    ] {
        let when = serde_json::from_str::<ProducerIdentity>(given);
        assert!(when.unwrap_err().to_string().contains("integer `42`"));
    }
}

#[test]
fn migration_canary_receipts_root_decoder_preserves_reversed_field_position() {
    for given in [
        r#"{"branch":[],"run_id":42}"#,
        r#"{"branch":"fixture","run_id":42,"branch":[]}"#,
    ] {
        let when = serde_json::from_str::<ProducerIdentity>(given);
        assert!(when.unwrap_err().to_string().contains("sequence"));
    }
}
