use super::support::{DENSE, Fixture, PLAN, context, context_with, family, read};
use crate::canary_receipts::{WorkerOutcome, WorkerResult, aggregate, write_receipt};
use serde_json::json;
use std::fs;

#[test]
fn migration_canary_receipts_writer_matches_frozen_legacy_shape_and_bytes() {
    let given = Fixture::new();
    fs::write(given.results(""), DENSE).unwrap();
    let when = write_receipt(
        &context_with(PLAN, "3"),
        &given.0,
        WorkerResult {
            family: family("dense"),
            outcome: WorkerOutcome::Success,
            runner: Some("fixture-\u{e9}-\u{1f600}".to_owned()),
        },
    )
    .unwrap();
    assert_eq!(
        when.render().as_bytes(),
        include_bytes!("fixtures/expected-receipt.json")
    );
    assert_eq!(
        fs::read(given.0.join("receipt.json")).unwrap(),
        include_bytes!("fixtures/expected-receipt.json")
    );
    assert!(read(&given.0.join("receipt.json")).get("schema").is_none());
}

#[test]
fn migration_canary_receipts_writer_records_missing_results_without_certifying() {
    let given = Fixture::new();
    let when = write_receipt(
        &context(),
        &given.0,
        WorkerResult {
            family: family("dense"),
            outcome: WorkerOutcome::Failure,
            runner: None,
        },
    )
    .unwrap();
    assert!(when.results_sha256.is_none());
    assert_eq!(when.runner, "unknown");
    assert_eq!(when.outcome, WorkerOutcome::Failure);
}

#[test]
fn migration_canary_receipts_writer_replaces_current_receipt_without_touching_archived_attempt() {
    let given = Fixture::complete();
    let before = fs::read(given.0.join("dense/receipt.json")).unwrap();
    fs::create_dir(given.0.join("archived")).unwrap();
    fs::write(given.0.join("archived/receipt.json"), &before).unwrap();
    let when = write_receipt(
        &context(),
        &given.0.join("dense"),
        WorkerResult {
            family: family("dense"),
            outcome: WorkerOutcome::Failure,
            runner: None,
        },
    );
    assert!(when.is_ok());
    assert_eq!(
        fs::read(given.0.join("archived/receipt.json")).unwrap(),
        before
    );
    assert_eq!(
        read(&given.0.join("dense/receipt.json"))["run_attempt"],
        "4"
    );
    assert_eq!(
        read(&given.0.join("dense/receipt.json"))["outcome"],
        "failure"
    );
}

#[test]
fn migration_canary_receipts_writer_rejects_unplanned_family_before_writing() {
    let given = Fixture::new();
    let when = write_receipt(
        &context(),
        &given.0.join("foreign"),
        WorkerResult {
            family: family("foreign"),
            outcome: WorkerOutcome::Success,
            runner: None,
        },
    );
    assert!(when.is_err());
    assert!(!given.0.join("foreign").exists());
}

#[test]
fn migration_canary_receipts_full_local_receipt_flow_is_digest_bound() {
    let given = Fixture::new();
    for (name, bytes) in [("dense", DENSE), ("hybrid", super::support::HYBRID)] {
        fs::create_dir(given.0.join(name)).unwrap();
        fs::write(given.results(name), bytes).unwrap();
        write_receipt(
            &context(),
            &given.0.join(name),
            WorkerResult {
                family: family(name),
                outcome: WorkerOutcome::Success,
                runner: None,
            },
        )
        .unwrap();
    }
    let when = aggregate(&context(), &given.0).unwrap();
    assert_eq!(
        when.github_outputs().as_deref(),
        Some(
            "green=true\ncandidate=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nbranch=llama-canary/repair-123-2-aaaaaaaaaa\n"
        )
    );
}

#[test]
fn migration_canary_receipts_null_results_digest_is_not_success() {
    let given = Fixture::complete();
    given.edit("dense", "results_sha256", json!(null));
    let when = aggregate(&context(), &given.0).unwrap();
    assert_eq!(
        when.failures[0].error.kind,
        crate::canary_receipts::ErrorKind::ResultsDigest
    );
}

#[test]
fn migration_canary_receipts_receipt_size_limit_fails_closed() {
    let given = Fixture::complete();
    fs::write(
        given.0.join("dense/receipt.json"),
        vec![b' '; 1024 * 1024 + 1],
    )
    .unwrap();
    let when = aggregate(&context(), &given.0).unwrap();
    assert_eq!(
        when.failures[0].error.kind,
        crate::canary_receipts::ErrorKind::InputLimit
    );
    assert!(when.github_outputs().is_none());
}
