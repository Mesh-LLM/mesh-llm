use super::support::{DENSE, Fixture, context};
use crate::canary_receipts::{Digest, ErrorKind, aggregate};
use serde_json::json;
use std::fs;

#[test]
fn migration_canary_receipts_accepts_receipt_at_exact_limit() {
    let given = Fixture::complete();
    let path = given.0.join("dense/receipt.json");
    let mut bytes = fs::read(&path).unwrap();
    bytes.resize(1024 * 1024, b' ');
    fs::write(path, bytes).unwrap();

    let when = aggregate(&context(), &given.0).unwrap();

    assert!(when.is_green(), "{}", when.summary());
}

#[test]
fn migration_canary_receipts_accepts_results_at_exact_limit() {
    let given = Fixture::complete();
    let mut bytes = DENSE.to_vec();
    bytes.resize(64 * 1024 * 1024, b' ');
    fs::write(given.results("dense"), &bytes).unwrap();
    given.edit(
        "dense",
        "results_sha256",
        json!(Digest::of_bytes(&bytes).as_str()),
    );

    let when = aggregate(&context(), &given.0).unwrap();

    assert!(when.is_green(), "{}", when.summary());
}

#[test]
fn migration_canary_receipts_rejects_results_above_limit() {
    let given = Fixture::complete();
    let mut bytes = DENSE.to_vec();
    bytes.resize(64 * 1024 * 1024 + 1, b' ');
    fs::write(given.results("dense"), &bytes).unwrap();
    given.edit(
        "dense",
        "results_sha256",
        json!(Digest::of_bytes(&bytes).as_str()),
    );

    let when = aggregate(&context(), &given.0).unwrap();

    assert_eq!(when.failures[0].error.kind, ErrorKind::InputLimit);
    assert!(when.github_outputs().is_none());
}
