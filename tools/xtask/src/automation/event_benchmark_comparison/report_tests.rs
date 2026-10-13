use super::super::{fixture, manifest};
use super::*;

fn inputs() -> [Manifest; 3] {
    ["production", "event-disabled", "production"].map(|mode| {
        manifest::decode(&serde_json::to_vec(&fixture::manifest(mode)).unwrap()).unwrap()
    })
}
fn options() -> Options {
    Options {
        seed: 42,
        bootstrap_samples: 200,
        min_primary_pairs: 20,
        min_scenario_pairs: 10,
        max_degradation_percent: 3.0,
        max_mdd_percent: 10.0,
        report_holm: true,
    }
}
fn projected(inputs: &[Manifest; 3]) -> Value {
    build(&inputs[0], &inputs[1], &inputs[2], &options()).unwrap()
}
fn blocks(value: &Value, reason: &str) -> bool {
    value["blocking_reasons"]
        .as_array()
        .unwrap()
        .iter()
        .any(|v| v == reason)
}
fn worse_baseline(inputs: &mut [Manifest; 3]) {
    for trial in &mut inputs[2].trials {
        trial
            .measurements
            .insert("decode_tok_s".into(), json!(100.0));
    }
}

#[test]
fn complete_healthy_fixture_preserves_report_shape_and_explicit_native_sampler() {
    let inputs = inputs();
    let report = projected(&inputs);
    assert_eq!(report["certification_status"], "pass");
    assert_eq!(report["wording"], "not proven worse by this screen");
    assert_eq!(report["resampling_algorithm"], resampling::ALGORITHM);
    assert_eq!(
        report["comparison_a"]["metrics"].as_array().unwrap().len(),
        3
    );
    assert_eq!(
        report["thermal_state"]["production"],
        inputs[0].thermal_state
    );
    assert_eq!(report["binary_identity"]["baseline"], inputs[2].binary);
    for phrase in [
        "proven within",
        "proven to be within",
        "statistically proven",
    ] {
        assert!(!report.to_string().contains(phrase));
    }
}

#[test]
fn comparison_b_screens_the_current_binary_against_baseline_not_against_reference() {
    let mut inputs = inputs();
    worse_baseline(&mut inputs);
    let report = projected(&inputs);
    assert_eq!(report["comparison_a"]["status"], "pass");
    assert_eq!(report["comparison_b"]["status"], "fail");
    assert!(blocks(&report, "degradation_fail"));
    assert_eq!(report["retry"]["action"], "retry_permitted");
}

#[test]
fn mismatched_seed_and_unplanned_environment_values_each_block() {
    let mut inputs = inputs();
    inputs[2].seed = 43;
    inputs[1]
        .environment
        .get_mut("MESH_LLM_LIFECYCLE_LOG_PARSER")
        .unwrap()
        .value = json!("disabled");
    let report = projected(&inputs);
    assert!(blocks(&report, "seed_mismatch"));
    assert!(blocks(&report, "environment_mismatch_comparison_a"));
    assert_eq!(report["certification_status"], "blocked");
}

#[test]
fn unavailable_health_and_missing_order_have_actual_reasons_not_fabricated_violations() {
    let mut inputs = inputs();
    inputs[0].health = None;
    inputs[1].health = None;
    inputs[0].executed_order = None;
    let report = projected(&inputs);
    assert!(blocks(&report, "health_unavailable"));
    assert!(blocks(&report, "executed_order_inconsistent"));
    assert!(!blocks(&report, "health_expectation_violation"));
    assert_eq!(report["health_availability"]["production"], false);
}

#[test]
fn measured_state_loss_and_unmeasurable_callback_each_block_certification() {
    let mut inputs = inputs();
    inputs[0].health =
        Some(serde_json::from_value(json!({"state_transition_rejected":1})).unwrap());
    inputs[1].callback_ingress_p99_us = None;
    let report = projected(&inputs);
    assert!(blocks(&report, "health_expectation_violation"));
    assert!(blocks(&report, "callback_ingress_p99"));
    assert!(!blocks(&report, "health_unavailable"));
}

#[test]
fn per_metric_exclusion_retains_insufficient_pairs_and_nullable_statistics() {
    let mut inputs = inputs();
    inputs[0].trials[19]
        .measurements
        .insert("decode_tok_s".into(), Value::Null);
    let report = projected(&inputs);
    assert!(blocks(&report, "insufficient_pairs"));
    assert_eq!(report["comparison_a"]["metrics"][0]["valid_pairs"], 19);
    assert_eq!(report["comparison_a"]["metrics"][0]["required_pairs"], 20);
    assert!(report["comparison_a"]["metrics"][0]["ci_low_pct"].is_null());
    assert_eq!(report["comparison_a"]["metrics"][1]["status"], "pass");
}

#[test]
fn a_second_adverse_complete_set_blocks_retry_without_executing_any_trial() {
    let mut inputs = inputs();
    worse_baseline(&mut inputs);
    inputs[0].attempt = 2;
    let report = projected(&inputs);
    assert!(blocks(&report, "retry_exhausted"));
    assert_eq!(report["retry"]["action"], "blocked_retry_exhausted");
}

#[test]
fn complete_off_reference_preserves_intentional_absent_engine_health() {
    let mut values = [
        fixture::manifest("production"),
        fixture::manifest("off"),
        fixture::manifest("production"),
    ];
    values[1]["health"] = Value::Null;
    for value in &mut values[..2] {
        for record in value["executed_order"].as_array_mut().unwrap() {
            for side in record["order"].as_array_mut().unwrap() {
                if side == "event-disabled" {
                    *side = json!("off");
                }
            }
        }
    }
    let inputs = values.map(|v| manifest::decode(&serde_json::to_vec(&v).unwrap()).unwrap());
    let report = projected(&inputs);
    assert_eq!(report["certification_status"], "pass");
    assert_eq!(report["health_availability"]["event_disabled"], true);
}

#[test]
fn caller_cannot_reduce_the_frozen_pair_minimum_or_exceed_bootstrap_work() {
    let inputs = inputs();
    let mut settings = options();
    settings.min_primary_pairs = 19;
    assert!(build(&inputs[0], &inputs[1], &inputs[2], &settings).is_err());
    settings = options();
    settings.min_scenario_pairs = 9;
    assert!(build(&inputs[0], &inputs[1], &inputs[2], &settings).is_err());
    settings = options();
    settings.bootstrap_samples = usize::MAX;
    assert!(build(&inputs[0], &inputs[1], &inputs[2], &settings).is_err());
}
