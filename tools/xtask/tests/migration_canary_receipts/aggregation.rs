use super::support::{Fixture, context, family};
use crate::canary_receipts::aggregate;
use serde::Deserialize;
use serde_json::json;
use std::fs;

#[derive(Debug, Deserialize)]
struct Case {
    case: String,
    green: bool,
    passed: Vec<String>,
    selected_dense: Option<String>,
    errors: Vec<String>,
}

fn given_case(name: &str) -> Fixture {
    let fixture = Fixture::complete();
    match name {
        "complete" => {}
        "rerun-success" => {
            fixture.edit("dense", "outcome", json!("failure"));
            fixture.put("new-dense", "dense", "3", "success");
        }
        "rerun-failure" => fixture.put("new-dense", "dense", "3", "failure"),
        "rerun-cancelled" => fixture.put("new-dense", "dense", "3", "cancelled"),
        "rerun-skipped" => fixture.put("new-dense", "dense", "3", "skipped"),
        "rerun-corrupt-results" => {
            fixture.put("new-dense", "dense", "3", "success");
            fs::write(fixture.results("new-dense"), b"{}\n").unwrap();
        }
        "rerun-missing-results" => {
            fixture.put("new-dense", "dense", "3", "success");
            fs::remove_file(fixture.results("new-dense")).unwrap();
        }
        "duplicate" => fixture.put("duplicate", "dense", "2", "success"),
        "missing" => fs::remove_file(fixture.0.join("dense/receipt.json")).unwrap(),
        "foreign-run" => fixture.edit("dense", "run_id", json!("999")),
        "foreign-candidate" => fixture.edit("dense", "candidate", json!("c".repeat(40))),
        "foreign-pass" => fixture.edit("dense", "pass_id", json!("verify-1")),
        "stale-producer" => fixture.edit("dense", "identity_sha256", json!("d".repeat(64))),
        "attempt-before-producer" => fixture.edit("dense", "run_attempt", json!("1")),
        "attempt-after-current" => fixture.edit("dense", "run_attempt", json!("5")),
        "malformed-attempt" => fixture.edit("dense", "run_attempt", json!("02")),
        "unplanned-family" => fixture.edit("dense", "family", json!("foreign")),
        "all-failed" => {
            fixture.edit("dense", "outcome", json!("failure"));
            fixture.edit("hybrid", "outcome", json!("failure"));
        }
        "newer-malformed-receipt" => {
            fixture.put("new-dense", "dense", "3", "success");
            fs::write(fixture.0.join("new-dense/receipt.json"), b"{").unwrap();
        }
        "old-corrupt-results" => {
            fixture.put("new-dense", "dense", "3", "success");
            fs::write(fixture.results("dense"), b"{}\n").unwrap();
        }
        "duplicate-old-attempt" => {
            fixture.put("duplicate", "dense", "2", "success");
            fixture.put("new-dense", "dense", "3", "success");
        }
        _ => panic!("unknown frozen case: {name}"),
    }
    fixture
}

#[test]
fn migration_canary_receipts_aggregate_matches_source_derived_cases() {
    let cases: Vec<Case> = serde_json::from_slice(include_bytes!("fixtures/cases.json")).unwrap();
    for case in cases {
        let given = given_case(&case.case);
        let when = aggregate(&context(), &given.0).unwrap();
        assert_eq!(when.is_green(), case.green, "{}", case.case);
        assert_eq!(
            when.passed
                .iter()
                .map(|family| family.as_str())
                .collect::<Vec<_>>(),
            case.passed,
            "{}",
            case.case
        );
        assert_eq!(
            when.selected_attempts
                .get(&family("dense"))
                .map(|attempt| attempt.as_str()),
            case.selected_dense.as_deref(),
            "{}",
            case.case
        );
        let errors: Vec<_> = when
            .failures
            .iter()
            .map(|failure| serde_json::to_value(failure.error.kind).unwrap())
            .collect();
        assert_eq!(
            errors,
            case.errors
                .iter()
                .map(|text| json!(text))
                .collect::<Vec<_>>(),
            "{}",
            case.case
        );
        assert_eq!(when.github_outputs().is_some(), case.green, "{}", case.case);
    }
}

#[test]
fn migration_canary_receipts_newest_selection_is_numeric_not_lexical() {
    let given = Fixture::complete();
    given.put("new-dense", "dense", "10", "failure");
    given.put("old-dense", "dense", "9", "success");
    let when = aggregate(
        &super::support::context_with(super::support::PLAN, "10"),
        &given.0,
    )
    .unwrap();
    assert_eq!(when.selected_attempts[&family("dense")].as_str(), "10");
    assert!(!when.is_green());
}

#[test]
fn migration_canary_receipts_ignores_superseded_payload_but_keeps_provenance_errors() {
    let given = Fixture::complete();
    given.edit("dense", "outcome", json!({"broken":"superseded payload"}));
    given.edit("dense", "results_sha256", json!([false]));
    given.put("new-dense", "dense", "3", "success");
    let when = aggregate(&context(), &given.0).unwrap();
    assert!(when.is_green());
}

#[test]
fn migration_canary_receipts_reports_all_failed_workers() {
    let given = given_case("all-failed");
    let when = aggregate(&context(), &given.0).unwrap();
    assert_eq!(when.failures.len(), 2);
    assert_eq!(
        when.summary(),
        "Canary repair-1: 0/2 family receipts passed for aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\n- dense: failed or mismatched worker receipt (runner=fixture, outcome=failure)\n- hybrid: failed or mismatched worker receipt (runner=fixture, outcome=failure)\n"
    );
    assert!(when.github_outputs().is_none());
}

#[test]
fn migration_canary_receipts_ignores_nonreceipt_files_and_nested_receipts() {
    let given = Fixture::complete();
    fs::write(given.0.join("note.txt"), b"not evidence").unwrap();
    fs::create_dir_all(given.0.join("outer/nested")).unwrap();
    fs::write(given.0.join("outer/nested/receipt.json"), b"invalid").unwrap();
    let when = aggregate(&context(), &given.0).unwrap();
    assert!(when.is_green());
}

#[test]
fn migration_canary_receipts_fails_closed_when_evidence_directory_missing() {
    let given = Fixture::new();
    let when = aggregate(&context(), &given.0.join("absent")).unwrap();
    assert_eq!(
        when.failures[0].error.kind,
        crate::canary_receipts::ErrorKind::MissingReceipts
    );
    assert!(when.github_outputs().is_none());
}

#[test]
fn migration_canary_receipts_newest_malformed_payload_cannot_fall_back() {
    let given = Fixture::complete();
    given.put("new-dense", "dense", "3", "success");
    given.edit("new-dense", "outcome", json!({"malformed":true}));
    let when = aggregate(&context(), &given.0).unwrap();
    assert_eq!(when.selected_attempts[&family("dense")].as_str(), "3");
    assert_eq!(
        when.failures[0].error.kind,
        crate::canary_receipts::ErrorKind::Json
    );
    assert!(!when.passed.contains(&family("dense")));
}

#[test]
fn migration_canary_receipts_stale_old_identity_invalidates_new_success() {
    let given = Fixture::complete();
    given.edit("dense", "identity_sha256", json!("d".repeat(64)));
    given.put("new-dense", "dense", "3", "success");
    let when = aggregate(&context(), &given.0).unwrap();
    assert!(when.passed.contains(&family("dense")));
    assert_eq!(
        when.failures[0].error.kind,
        crate::canary_receipts::ErrorKind::ReceiptIdentity
    );
    assert!(when.github_outputs().is_none());
}

#[test]
fn migration_canary_receipts_duplicate_is_rejected_in_either_directory_order() {
    for name in ["aaa-first", "zzz-last"] {
        let given = Fixture::complete();
        given.put(name, "dense", "2", "success");
        let when = aggregate(&context(), &given.0).unwrap();
        assert_eq!(
            when.failures[0].error.kind,
            crate::canary_receipts::ErrorKind::DuplicateReceipt
        );
        assert!(when.github_outputs().is_none());
    }
}

#[test]
fn migration_canary_receipts_missing_required_lane_still_fails_with_matching_digest() {
    let given = Fixture::complete();
    let bytes = br#"{"family":"dense","exit_code":0,"split_layer":10,"outcomes":[]}"#;
    fs::write(given.results("dense"), bytes).unwrap();
    given.edit(
        "dense",
        "results_sha256",
        json!(crate::canary_receipts::Digest::of_bytes(bytes).as_str()),
    );
    let when = aggregate(&context(), &given.0).unwrap();
    assert_eq!(
        when.failures[0].error.kind,
        crate::canary_receipts::ErrorKind::RequiredLane
    );
    assert!(when.github_outputs().is_none());
}
