use super::super::{fixture, manifest};
use super::*;
fn inputs() -> [Manifest; 3] {
    ["production", "event-disabled", "production"].map(|mode| {
        let mut input = fixture::manifest(mode);
        input["trial_plan_algorithm"] = Value::String(NATIVE_ALGORITHM.into());
        input["model"] = Value::String("approved-local-model".into());
        for row in input["trials"].as_array_mut().unwrap() {
            row["prompt_sha256"] = Value::String("a".repeat(64));
        }
        manifest::decode(&serde_json::to_vec(&input).unwrap()).unwrap()
    })
}
fn check(inputs: &[Manifest; 3]) -> Vec<String> {
    violations(&inputs[0], &inputs[1], &inputs[2])
}
#[test]
fn historical_manifests_remain_comparable_only_when_every_side_is_unversioned() {
    let inputs = ["production", "event-disabled", "production"].map(|mode| {
        manifest::decode(&serde_json::to_vec(&fixture::manifest(mode)).unwrap()).unwrap()
    });
    assert!(check(&inputs).is_empty());
}
#[test]
fn matched_native_plan_model_and_complete_prompt_identity_are_comparable() {
    assert!(check(&inputs()).is_empty());
}
#[test]
fn native_and_historical_or_changed_algorithm_inputs_cannot_pair_by_index_alone() {
    for changed in [None, Some("future-plan-v2".into())] {
        let mut inputs = inputs();
        inputs[2].trial_plan_algorithm = changed;
        assert!(
            check(&inputs)
                .iter()
                .any(|message| message.contains("baseline: trial plan algorithm"))
        );
    }
}
#[test]
fn differing_or_missing_model_reference_blocks_native_pairing() {
    for model in [None, Some("different-local-model".into())] {
        let mut inputs = inputs();
        inputs[1].model = model;
        assert!(
            check(&inputs)
                .iter()
                .any(|message| message.contains("event_disabled: paired trial model"))
        );
    }
}
#[test]
fn prompt_identity_mismatch_is_checked_independently_on_both_comparisons() {
    for (index, label) in [(1, "comparison_a"), (2, "comparison_b")] {
        let mut inputs = inputs();
        inputs[index].trials[0]
            .measurements
            .insert("prompt_sha256".into(), Value::String("b".repeat(64)));
        assert!(
            check(&inputs)
                .iter()
                .any(|message| message.starts_with(label))
        );
    }
}
#[test]
fn absent_or_malformed_hash_never_defaults_to_a_matching_prompt() {
    for hash in [
        Value::Null,
        Value::String("short".into()),
        Value::String("g".repeat(64)),
        Value::Bool(true),
    ] {
        let mut inputs = inputs();
        inputs[2].trials[0]
            .measurements
            .insert("prompt_sha256".into(), hash);
        assert!(
            check(&inputs)
                .iter()
                .any(|message| message.contains("baseline: complete prompt identity"))
        );
    }
}
#[test]
fn pairing_uses_row_identity_instead_of_input_order_or_hex_case() {
    let mut inputs = inputs();
    inputs[2].trials.reverse();
    for row in &mut inputs[2].trials {
        row.measurements
            .insert("prompt_sha256".into(), Value::String("A".repeat(64)));
    }
    assert!(check(&inputs).is_empty());
}

#[test]
fn complete_report_blocks_mixed_algorithm_or_replaced_prompt_with_its_actual_reason() {
    let options = super::super::options::Options {
        seed: 42,
        bootstrap_samples: 100,
        min_primary_pairs: 20,
        min_scenario_pairs: 10,
        max_degradation_percent: 3.0,
        max_mdd_percent: 10.0,
        report_holm: false,
    };
    for mixed_algorithm in [true, false] {
        let mut inputs = inputs();
        if mixed_algorithm {
            inputs[2].trial_plan_algorithm = None;
        } else {
            inputs[1].trials[0]
                .measurements
                .insert("prompt_sha256".into(), Value::String("b".repeat(64)));
        }
        let result =
            super::super::report::build(&inputs[0], &inputs[1], &inputs[2], &options).unwrap();
        assert_eq!(result["certification_status"], "blocked");
        let reasons = result["blocking_reasons"].as_array().unwrap();
        assert!(
            reasons
                .iter()
                .any(|reason| reason == "trial_plan_identity_mismatch")
        );
        assert!(!reasons.iter().any(|reason| reason == "seed_mismatch"));
        assert!(
            !result["trial_plan_violations"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
}
