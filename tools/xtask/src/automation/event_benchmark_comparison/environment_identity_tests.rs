use super::*;
use serde_json::json;

fn input(value: Value) -> Environment {
    serde_json::from_value(value).unwrap()
}

#[test]
fn identical_normalized_and_redacted_presence_sets_are_comparable() {
    let env = input(
        json!({"MESH_LLM_LIFECYCLE_LOG_PARSER":{"value":"auto","redacted":false},"MESH_LLM_CONFIG":{"value":"<redacted:present>","redacted":true}}),
    );
    assert!(compare(&env, &env, false).unwrap().is_empty());
}

#[test]
fn only_the_explicit_comparison_a_selector_is_excluded() {
    let a = input(json!({(PLANNED_SELECTOR):{"value":"production","redacted":false}}));
    let b = input(json!({(PLANNED_SELECTOR):{"value":"event-disabled","redacted":false}}));
    assert!(compare(&a, &b, true).unwrap().is_empty());
    assert_eq!(compare(&a, &b, false).unwrap().len(), 1);
}

#[test]
fn unplanned_values_and_one_sided_presence_keep_named_violations_without_values() {
    let a = input(json!({"MESH_LLM_LIFECYCLE_LOG_PARSER":{"value":"auto","redacted":false}}));
    let b = input(json!({"MESH_LLM_LIFECYCLE_LOG_PARSER":{"value":"disabled","redacted":false}}));
    assert_eq!(
        compare(&a, &b, false).unwrap(),
        ["MESH_LLM_LIFECYCLE_LOG_PARSER: normalized value mismatch"]
    );
    assert!(compare(&a, &Environment::new(), false).unwrap()[0].contains("one side only"));
}

#[test]
fn redaction_mode_mismatch_and_boolean_numeric_mismatch_do_not_compare_equal() {
    let a = input(json!({"MESH_LLM_CONFIG":{"value":"<redacted:present>","redacted":true}}));
    let b = input(json!({"MESH_LLM_CONFIG":{"value":"/fixtures/config","redacted":false}}));
    assert!(compare(&a, &b, false).unwrap()[0].contains("redacted-presence mismatch"));
    let a = input(json!({"MESH_LLM_FLAG":{"value":true,"redacted":false}}));
    let b = input(json!({"MESH_LLM_FLAG":{"value":1,"redacted":false}}));
    assert_eq!(compare(&a, &b, false).unwrap().len(), 1);
}

#[test]
fn malformed_normalized_entries_and_non_marker_redaction_are_refused() {
    for entry in [
        json!({"value":{},"redacted":false}),
        json!({"value":"raw-value","redacted":true}),
        json!({"value":null,"redacted":false}),
    ] {
        assert!(validate(&input(json!({"MESH_LLM_FLAG":entry}))).is_err());
    }
}
