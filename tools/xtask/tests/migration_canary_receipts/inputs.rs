use super::support::{IDENTITY, PLAN, context_with, family};
use crate::canary_receipts::{
    Digest, ReceiptContext, RunAttempt, SourceFamilyPlan, TestPackageInputs, WorkflowRun,
};
use serde_json::{Value, json};

#[test]
fn migration_canary_receipts_attempts_require_positive_decimal_strings() {
    for given in [
        json!("0"),
        json!("-1"),
        json!("02"),
        json!("x"),
        json!(2),
        json!(null),
        json!(true),
        json!("2\n"),
    ] {
        let when = serde_json::from_value::<RunAttempt>(given.clone());
        assert!(when.is_err(), "{given}");
    }
}

#[test]
fn migration_canary_receipts_attempts_compare_exactly_beyond_machine_integer() {
    let given: RunAttempt = "18446744073709551616".to_owned().try_into().unwrap();
    let smaller: RunAttempt = "18446744073709551615".to_owned().try_into().unwrap();
    let when = given.cmp(&smaller);
    assert_eq!(when, std::cmp::Ordering::Greater);
}

#[test]
fn migration_canary_receipts_rejects_foreign_current_run_or_earlier_attempt() {
    for (run_id, attempt) in [("999", "4"), ("123", "1")] {
        let given = TestPackageInputs {
            identity: serde_json::from_slice(IDENTITY).unwrap(),
            identity_sha256: Digest::of_bytes(IDENTITY),
            plan: SourceFamilyPlan::parse(PLAN).unwrap(),
        };
        let when = ReceiptContext::from_test_package(
            given,
            WorkflowRun {
                run_id: run_id.to_owned(),
                run_attempt: attempt.to_owned().try_into().unwrap(),
            },
        );
        assert!(when.is_err());
    }
}

#[test]
fn migration_canary_receipts_rejects_plan_membership_drift() {
    for variant in [
        "empty",
        "duplicate-family",
        "missing-matrix",
        "duplicate-index",
        "wrong-membership",
        "missing-core",
        "unsafe-family",
    ] {
        let mut given: Value = serde_json::from_slice(PLAN).unwrap();
        match variant {
            "empty" => given["selected_models"] = json!([]),
            "duplicate-family" => given["selected_models"][1]["family"] = json!("dense"),
            "missing-matrix" => {
                given["github_matrix"]["include"]
                    .as_array_mut()
                    .unwrap()
                    .pop();
            }
            "duplicate-index" => given["github_matrix"]["include"][0]["shard_index"] = json!(0),
            "wrong-membership" => given["shards"][0]["families"] = json!(["hybrid"]),
            "missing-core" => given["required_certification_lanes"] = json!(["chain"]),
            "unsafe-family" => given["selected_models"][0]["family"] = json!("dense/foreign"),
            _ => unreachable!(),
        }
        let when = SourceFamilyPlan::parse(&serde_json::to_vec(&given).unwrap());
        assert!(when.is_err(), "{variant}");
    }
}

#[test]
fn migration_canary_receipts_accepts_historical_extra_fields_without_mutating_plan() {
    let given = PLAN.to_vec();
    let when = context_with(&given, "4");
    assert!(when.model(&family("dense")).is_ok());
    assert_eq!(given, PLAN);
}

#[test]
fn migration_canary_receipts_checks_frozen_fixture_hashes() {
    let given = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/migration_canary_receipts/fixtures");
    for row in include_str!("fixtures/FROZEN.sha256")
        .lines()
        .chain(include_str!("fixtures/EXPECTED.sha256").lines())
    {
        let (expected, name) = row.split_once("  ").unwrap();
        let when = Digest::of_bytes(&std::fs::read(given.join(name)).unwrap());
        assert_eq!(when.as_str(), expected, "{name}");
    }
}
