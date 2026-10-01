use super::support::{DENSE, Fixture, IDENTITY, PLAN, context, context_with, family};
use crate::canary_receipts::{
    Digest, ErrorKind, ProducerIdentity, SourceFamilyPlan, aggregate, validate_results,
};
use serde_json::json;
use std::fs;

#[test]
fn migration_canary_receipts_nested_status_uses_last_pass_with_matching_digest() {
    let given = Fixture::complete();
    let bytes = include_bytes!("fixtures/nested-status-pass-last.jsonl");
    fs::write(given.results("dense"), bytes).unwrap();
    given.edit(
        "dense",
        "results_sha256",
        json!(Digest::of_bytes(bytes).as_str()),
    );
    let when = aggregate(&context(), &given.0).unwrap();
    assert!(when.is_green(), "{}", when.summary());
}

#[test]
fn migration_canary_receipts_nested_status_uses_last_failure_with_matching_digest() {
    let given = Fixture::complete();
    let bytes = include_bytes!("fixtures/nested-status-fail-last.jsonl");
    fs::write(given.results("dense"), bytes).unwrap();
    given.edit(
        "dense",
        "results_sha256",
        json!(Digest::of_bytes(bytes).as_str()),
    );
    let when = aggregate(&context(), &given.0).unwrap();
    assert_eq!(when.failures[0].error.kind, ErrorKind::RequiredLane);
    assert!(when.github_outputs().is_none());
}

#[test]
fn migration_canary_receipts_nested_model_uses_last_causal_class() {
    let bytes = std::str::from_utf8(PLAN).unwrap().replacen(
        r#""class": "causal_generation""#,
        r#""class": "embedding", "class": "causal_generation""#,
        1,
    );
    let given = context_with(bytes.as_bytes(), "4");
    let family = family("dense");
    let when = validate_results(DENSE, &family, given.model(&family).unwrap());
    assert!(when.is_ok(), "{when:?}");
}

#[test]
fn migration_canary_receipts_nested_model_uses_last_workload_class() {
    let bytes = std::str::from_utf8(PLAN).unwrap().replacen(
        r#""class": "causal_generation""#,
        r#""class": "causal_generation", "class": "embedding""#,
        1,
    );
    let given = context_with(bytes.as_bytes(), "4");
    let family = family("dense");
    let when = validate_results(DENSE, &family, given.model(&family).unwrap());
    assert_eq!(when.unwrap_err().kind, ErrorKind::Certification);
}

#[test]
fn migration_canary_receipts_nested_matrix_uses_last_complete_membership() {
    let given = std::str::from_utf8(PLAN).unwrap().replace(
        r#""github_matrix": {"include": ["#,
        r#""github_matrix": {"include": [], "include": ["#,
    );
    let when = SourceFamilyPlan::parse(given.as_bytes());
    assert!(when.is_ok());
}

#[test]
fn migration_canary_receipts_nested_matrix_uses_last_empty_membership() {
    let given = std::str::from_utf8(PLAN).unwrap().replace(
        r#""families": "dense"}]}"#,
        r#""families": "dense"}], "include": []}"#,
    );
    let when = SourceFamilyPlan::parse(given.as_bytes());
    assert_eq!(when.err().unwrap().kind, ErrorKind::Plan);
}

#[test]
fn migration_canary_receipts_root_result_uses_last_planned_family() {
    let bytes = std::str::from_utf8(DENSE).unwrap().replace(
        r#""family":"dense""#,
        r#""family":"foreign","family":"dense""#,
    );
    let given = context();
    let family = family("dense");
    let when = validate_results(bytes.as_bytes(), &family, given.model(&family).unwrap());
    assert!(when.is_ok(), "{when:?}");
}

#[test]
fn migration_canary_receipts_root_plan_uses_last_complete_models() {
    let given = std::str::from_utf8(PLAN).unwrap().replacen(
        r#""selected_models": ["#,
        r#""selected_models": [], "selected_models": ["#,
        1,
    );
    let when = SourceFamilyPlan::parse(given.as_bytes());
    assert!(when.is_ok());
}

#[test]
fn migration_canary_receipts_root_identity_uses_last_current_run() {
    let given = std::str::from_utf8(IDENTITY)
        .unwrap()
        .replace(r#""run_id": "123""#, r#""run_id": "999", "run_id": "123""#);
    let when = serde_json::from_str::<ProducerIdentity>(&given);
    assert_eq!(when.unwrap().run_id, "123");
}

#[test]
fn migration_canary_receipts_root_receipt_provenance_uses_last_current_run() {
    let given = Fixture::complete();
    let path = given.0.join("dense/receipt.json");
    let bytes = fs::read_to_string(&path)
        .unwrap()
        .replace(r#""run_id":"123""#, r#""run_id":"999","run_id":"123""#);
    fs::write(path, bytes).unwrap();
    let when = aggregate(&context(), &given.0).unwrap();
    assert!(when.is_green(), "{}", when.summary());
    assert!(when.github_outputs().is_some());
}
