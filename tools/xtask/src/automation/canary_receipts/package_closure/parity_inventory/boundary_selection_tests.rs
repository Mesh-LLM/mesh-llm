use super::{super::*, set};

fn selected(rows: Vec<Value>, boundaries: &[&str]) -> Value {
    let sources = rows
        .iter()
        .map(|row| row["llama_model"].as_str().unwrap().into())
        .collect();
    let parity = json!({"candidates":rows});
    let boundaries = set(boundaries);
    validate(&parity, &json!({"models":[]}), &sources, &boundaries).unwrap();
    let rows = classifications(&parity, &sources, &boundaries).unwrap();
    serde_json::to_value(
        boundary_selection::next_target(&rows, Path::new("/prepared/llama.cpp")).unwrap(),
    )
    .unwrap()
}

#[test]
fn next_boundary_is_one_lexicographic_model_family_from_admitted_rows() {
    let rows = vec![
        json!({"llama_model":"zeta","family":"zeta_family","status":"needs_boundary_registration"}),
        json!({"llama_model":"alpha","family":"z_family","status":"needs_boundary_registration"}),
        json!({"llama_model":"alpha","family":"a_family","status":"needs_boundary_registration"}),
    ];
    for order in [rows.clone(), rows.into_iter().rev().collect()] {
        assert_eq!(
            selected(order, &[]),
            json!({"llama_model":"alpha","family":"a_family","source_file":"/prepared/llama.cpp/src/models/alpha.cpp","status":"needs_boundary_registration"})
        );
    }
}

#[test]
fn next_boundary_skips_registered_blocked_and_non_expansion_rows() {
    let rows = vec![
        json!({"llama_model":"alpha","status":"needs_boundary_registration"}),
        json!({"llama_model":"beta","status":"needs_boundary_registration","unsupported_reason":"blocked upstream"}),
        json!({"llama_model":"gamma","status":"needs_candidate"}),
        json!({"llama_model":"zeta-model","status":"needs_boundary_registration","unsupported_reason":""}),
    ];
    assert_eq!(
        selected(rows, &["alpha"]),
        json!({"llama_model":"zeta-model","family":"zeta_model","source_file":"/prepared/llama.cpp/src/models/zeta-model.cpp","status":"needs_boundary_registration"})
    );
}

#[test]
fn next_boundary_null_when_no_eligible_gap_remains() {
    assert_eq!(selected(vec![], &[]), Value::Null);
    let rows = vec![json!({"llama_model":"alpha","status":"needs_boundary_registration"})];
    assert_eq!(selected(rows, &["alpha"]), Value::Null);
}

#[test]
fn stale_runnable_diagnostic_is_sorted_and_does_not_admit_inventory() {
    let parity = json!({"candidates":[
        {"llama_model":"zeta","status":"candidate"},
        {"llama_model":"alpha","status":"certified"},
        {"llama_model":"blocked","status":"candidate","unsupported_reason":"deliberate"},
        {"llama_model":"registered","status":"candidate_stateful"},
        {"llama_model":"orphan","status":"candidate"},
        {"llama_model":"gap","status":"needs_boundary_registration"}
    ]});
    let sources = set(&["zeta", "alpha", "blocked", "registered", "gap"]);
    let boundaries = set(&["registered"]);
    let rows = parity["candidates"].as_array().unwrap();
    assert_eq!(
        boundary_selection::pending_reclassification(rows, &sources, &boundaries),
        set(&["alpha", "zeta"])
    );
    let error = validate(&parity, &json!({"models":[]}), &sources, &boundaries)
        .unwrap_err()
        .to_string();
    assert!(error.contains("pending_reclassification: [\"alpha\",\"zeta\"]"));
    assert!(error.contains("lacks paired boundary calls"));
}
