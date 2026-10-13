use super::support::{
    BATTERY, DENSE, EMBEDDING, PLAN, PRETTY, PROJECTOR, context, context_with, family,
};
use crate::canary_receipts::{Error, ErrorKind, validate_results};
use serde_json::{Value, json};

fn check(bytes: &[u8]) -> Result<(), Error> {
    let context = context();
    let family = family("dense");
    validate_results(bytes, &family, context.model(&family).unwrap())
}

#[test]
fn migration_canary_receipts_accepts_adjacent_pretty_and_battery_objects() {
    let given = [BATTERY, PRETTY].concat();
    let when = check(&given);
    assert!(when.is_ok(), "{when:?}");
}

#[test]
fn migration_canary_receipts_rejects_incomplete_result_streams() {
    let cases: Vec<(&[u8], ErrorKind)> = vec![
        (b"", ErrorKind::Results),
        (b"[]", ErrorKind::Json),
        (b"null", ErrorKind::Json),
        (b"{}{}", ErrorKind::Results),
        (b"{", ErrorKind::Json),
        (
            b"{\"family\":\"foreign\",\"exit_code\":0}",
            ErrorKind::Results,
        ),
        (
            b"{\"family\":\"dense\",\"exit_code\":1}",
            ErrorKind::Results,
        ),
        (BATTERY, ErrorKind::Results),
    ];
    for (given, expected) in cases {
        let when = check(given);
        assert_eq!(when.unwrap_err().kind, expected, "{given:?}");
    }
}

#[test]
fn migration_canary_receipts_rejects_duplicate_core_and_battery_certifications() {
    for (given, expected) in [
        ([DENSE, DENSE].concat(), ErrorKind::Certification),
        (
            [BATTERY, BATTERY, DENSE].concat(),
            ErrorKind::BatteryPreflight,
        ),
    ] {
        let when = check(&given);
        assert_eq!(when.unwrap_err().kind, expected);
    }
}

#[test]
fn migration_canary_receipts_requires_one_passing_result_per_lane() {
    for variant in [
        "absent",
        "duplicate",
        "skip",
        "failed-status",
        "failed-exit",
    ] {
        let mut given: Value = serde_json::from_slice(DENSE).unwrap();
        let outcomes = given["outcomes"].as_array_mut().unwrap();
        match variant {
            "absent" => {
                outcomes.remove(0);
            }
            "duplicate" => outcomes.push(outcomes[0].clone()),
            "skip" => outcomes[0]["status"] = json!("skip"),
            "failed-status" => outcomes[0]["status"] = json!("fail"),
            "failed-exit" => outcomes[0]["exit_code"] = json!(1),
            _ => unreachable!(),
        }
        let when = check(&serde_json::to_vec(&given).unwrap());
        assert_eq!(when.unwrap_err().kind, ErrorKind::RequiredLane, "{variant}");
    }
}

#[test]
fn migration_canary_receipts_requires_passing_global_preflight() {
    let mut given: Value = serde_json::from_slice(BATTERY).unwrap();
    given["outcomes"][0]["status"] = json!("fail");
    let when = check(&[serde_json::to_vec(&given).unwrap(), DENSE.to_vec()].concat());
    assert_eq!(when.unwrap_err().kind, ErrorKind::BatteryPreflight);
}

#[test]
fn migration_canary_receipts_requires_all_planned_native_heads_lane() {
    let mut plan: Value = serde_json::from_slice(PLAN).unwrap();
    plan["selected_models"][0]["certification_lanes"]
        .as_array_mut()
        .unwrap()
        .push(json!("native-mtp-heads"));
    let given = context_with(&serde_json::to_vec(&plan).unwrap(), "4");
    let family = family("dense");
    let when = validate_results(DENSE, &family, given.model(&family).unwrap());
    assert_eq!(when.unwrap_err().kind, ErrorKind::RequiredLane);
}

#[test]
fn migration_canary_receipts_preserves_workload_class_and_oracle_requirement() {
    let mut plan: Value = serde_json::from_slice(PLAN).unwrap();
    plan["selected_models"][0]["class"] = json!("embedding");
    plan["selected_models"][0]["certification_lanes"] =
        json!(["embedding-smoke", "embedding-oracle"]);
    let given = context_with(&serde_json::to_vec(&plan).unwrap(), "4");
    let family = family("dense");
    for (variant, expected) in [
        ("valid", None),
        ("wrong-class", Some(ErrorKind::WorkloadClass)),
        ("missing-oracle", Some(ErrorKind::RequiredLane)),
    ] {
        let mut row: Value = serde_json::from_slice(EMBEDDING).unwrap();
        match variant {
            "valid" => {}
            "wrong-class" => row["workload_class"] = json!("rerank"),
            "missing-oracle" => {
                row["outcomes"].as_array_mut().unwrap().pop();
            }
            _ => unreachable!(),
        }
        let when = validate_results(
            &serde_json::to_vec(&row).unwrap(),
            &family,
            given.model(&family).unwrap(),
        );
        assert_eq!(when.err().map(|error| error.kind), expected);
    }
}

#[test]
fn migration_canary_receipts_requires_exact_causal_projector_smoke_count() {
    let mut plan: Value = serde_json::from_slice(PLAN).unwrap();
    plan["selected_models"][0]["mmproj_artifact"] = json!({"files":["projector.gguf"]});
    let given = context_with(&serde_json::to_vec(&plan).unwrap(), "4");
    let family = family("dense");
    for (bytes, expected) in [
        (DENSE.to_vec(), Some(ErrorKind::Multimodal)),
        ([DENSE, PROJECTOR].concat(), None),
        (
            [DENSE, PROJECTOR, PROJECTOR].concat(),
            Some(ErrorKind::Multimodal),
        ),
    ] {
        let when = validate_results(&bytes, &family, given.model(&family).unwrap());
        assert_eq!(when.err().map(|error| error.kind), expected);
    }
}

#[test]
fn migration_canary_receipts_preserves_legacy_zero_exit_comparison() {
    for exit_code in [json!(false), json!(0.0), json!(0)] {
        let mut given: Value = serde_json::from_slice(DENSE).unwrap();
        given["exit_code"] = exit_code.clone();
        for lane in given["outcomes"].as_array_mut().unwrap() {
            lane["exit_code"] = exit_code.clone();
        }
        let when = check(&serde_json::to_vec(&given).unwrap());
        assert!(when.is_ok(), "{when:?}");
    }
}

#[test]
fn migration_canary_receipts_nonchat_projector_is_consolidated_in_workload_row() {
    let mut plan: Value = serde_json::from_slice(PLAN).unwrap();
    plan["selected_models"][0]["class"] = json!("embedding");
    plan["selected_models"][0]["certification_lanes"] =
        json!(["embedding-smoke", "embedding-oracle"]);
    plan["selected_models"][0]["mmproj_artifact"] = json!({"files":["projector.gguf"]});
    let given = context_with(&serde_json::to_vec(&plan).unwrap(), "4");
    let family = family("dense");
    let when = validate_results(EMBEDDING, &family, given.model(&family).unwrap());
    assert!(when.is_ok(), "{when:?}");
}

#[test]
fn migration_canary_receipts_supplied_native_heads_lane_can_certify() {
    let mut plan: Value = serde_json::from_slice(PLAN).unwrap();
    plan["selected_models"][0]["certification_lanes"]
        .as_array_mut()
        .unwrap()
        .push(json!("native-mtp-heads"));
    let given = context_with(&serde_json::to_vec(&plan).unwrap(), "4");
    let mut results: Value = serde_json::from_slice(DENSE).unwrap();
    results["outcomes"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name":"native-mtp-heads","status":"pass","exit_code":0}));
    let family = family("dense");
    let when = validate_results(
        &serde_json::to_vec(&results).unwrap(),
        &family,
        given.model(&family).unwrap(),
    );
    assert!(when.is_ok(), "{when:?}");
}

#[test]
fn migration_canary_receipts_zero_split_and_unrelated_lanes_follow_legacy_semantics() {
    let mut given: Value = serde_json::from_slice(DENSE).unwrap();
    given["split_layer"] = json!(0);
    given["outcomes"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name":"optional","status":"fail","exit_code":1}));
    let when = check(&serde_json::to_vec(&given).unwrap());
    assert!(when.is_ok(), "{when:?}");
}

#[test]
fn migration_canary_receipts_rejects_positional_arrays_as_lane_objects() {
    let mut given: Value = serde_json::from_slice(DENSE).unwrap();
    given["outcomes"] = json!([
        ["chain", "pass", 0],
        ["single-step", "pass", 0],
        ["state-handoff", "pass", 0]
    ]);
    let when = check(&serde_json::to_vec(&given).unwrap());
    assert_eq!(when.unwrap_err().kind, ErrorKind::Json);
}
