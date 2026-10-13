use crate::automation::rollout;

#[path = "boundary.rs"]
mod boundary;
#[path = "evidence_failures.rs"]
mod evidence_failures;
#[path = "fixture.rs"]
mod fixture;
#[path = "sequence.rs"]
mod sequence;

#[test]
fn migration_rollout_valid_sequence_reports_local_consistency_not_acceptance() {
    let fixture = fixture::Fixture::new();

    let facts = rollout::validate_file(&fixture.input).expect("consistent local observations");

    let report = serde_json::to_value(facts).expect("serializable local facts");
    assert_eq!(report["scope"], "local_fact_consistency_only");
    assert_eq!(report["task26_acceptance"], "not_established");
    assert_eq!(report["source_sha"], fixture.source);
    assert_eq!(report["protected_sha"], fixture.protected);
    assert_eq!(report["predecessor_files"], 15);
    assert!(report.get("shadow_comparisons").is_none());
    assert_eq!(report["externally_unverified"].as_array().unwrap().len(), 4);
}

#[test]
fn migration_rollout_invalid_cli_shape_does_not_validate() {
    let args = ["--approve".to_owned(), "anything.json".to_owned()];

    let result = rollout::run(&args);

    assert!(result.is_err());
}

#[test]
fn migration_rollout_command_reads_existing_receipts_without_executing_them() {
    let fixture = fixture::Fixture::new();
    let args = [
        "--input".to_owned(),
        fixture.input.to_str().unwrap().to_owned(),
    ];

    let result = rollout::run(&args);

    assert!(result.is_ok(), "{result:?}");
}
