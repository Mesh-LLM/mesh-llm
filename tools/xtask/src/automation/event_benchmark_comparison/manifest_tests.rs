use super::super::fixture;
use super::*;
use serde_json::json;

fn accepted(value: &Value) -> bool {
    decode(&serde_json::to_vec(value).unwrap()).is_ok()
}

#[test]
fn valid_streaming_manifest_preserves_observed_metadata_and_defaults_only_attempt() {
    let mut value = fixture::manifest("production");
    value.as_object_mut().unwrap().remove("attempt");
    let parsed = decode(&serde_json::to_vec(&value).unwrap()).unwrap();
    assert_eq!(parsed.attempt, 1);
    assert_eq!(parsed.trials.len(), 20);
    assert_eq!(parsed.thermal_state, value["thermal_state"]);
    assert!(parsed.health.is_some());
}

#[test]
fn missing_or_incompatible_metrics_schema_is_never_mixed_into_comparison() {
    for schema in [json!("non_streaming_historical"), Value::Null] {
        let mut value = fixture::manifest("production");
        value["metrics_schema"] = schema;
        assert!(!accepted(&value));
    }
    let mut value = fixture::manifest("production");
    value["schema_version"] = json!(2);
    assert!(!accepted(&value));
}

#[test]
fn malformed_modes_attempts_health_and_host_waivers_are_refused() {
    for (pointer, value) in [
        ("/mode", json!("unknown")),
        ("/attempt", json!(0)),
        ("/health/cancelled_reservation_rejected", json!(-1)),
        ("/host/p99_gate", json!("informational")),
    ] {
        let mut input = fixture::manifest("production");
        *input.pointer_mut(pointer).unwrap() = value;
        assert!(!accepted(&input), "{pointer}");
    }
}

#[test]
fn duplicate_or_undeclared_scenarios_and_trial_identity_are_refused() {
    for names in [json!(["case", "case"]), json!([""]), json!(["__primary__"])] {
        let mut value = fixture::manifest("production");
        value["scenarios"] = names;
        assert!(!accepted(&value));
    }
    let mut value = fixture::manifest("production");
    value["trials"][0]["scenario"] = json!("undeclared");
    assert!(!accepted(&value));
    value = fixture::manifest("production");
    value["trials"][1] = value["trials"][0].clone();
    assert!(!accepted(&value));
}

#[test]
fn missing_health_and_order_are_preserved_for_their_actual_evidence_gates() {
    let mut value = fixture::manifest("production");
    value["health"] = Value::Null;
    value["executed_order"] = Value::Null;
    let parsed = decode(&serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(parsed.health.is_none());
    assert!(parsed.executed_order.is_none());
    assert!(decode(&vec![b' '; MAX_MANIFEST_BYTES + 1]).is_err());
}
