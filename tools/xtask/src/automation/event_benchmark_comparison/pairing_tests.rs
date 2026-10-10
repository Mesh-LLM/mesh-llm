use super::*;
use serde_json::json;

fn trial(scenario: &str, pair_index: u64, throughput: Value, latency: Value) -> Trial {
    serde_json::from_value(json!({"scenario":scenario,"pair_index":pair_index,"status":"succeeded","decode_tok_s":throughput,"decode_only_tok_s":55,"ttft_ms":latency})).unwrap()
}

#[test]
fn pairing_uses_exact_group_and_pair_identity_independent_of_input_order() {
    let old = [
        trial("__primary__", 1, json!(100), json!(50)),
        trial("__primary__", 0, json!(50), json!(100)),
        trial("other", 0, json!(999), json!(999)),
    ];
    let new = [
        trial("__primary__", 0, json!(45), json!(110)),
        trial("__primary__", 1, json!(90), json!(55)),
    ];
    let throughput = pair(&old, &new, "__primary__", Metric::Decode).unwrap();
    let latency = pair(&old, &new, "__primary__", Metric::Ttft).unwrap();
    assert_eq!(throughput.deltas, [0.1, 0.1]);
    assert_eq!(latency.deltas, [0.1, 0.1]);
    assert!(throughput.exclusions.is_empty());
    assert_eq!(
        pair(&old, &new, "__primary__", Metric::DecodeOnly)
            .unwrap()
            .deltas,
        [0.0, 0.0]
    );
}

#[test]
fn missing_pairs_on_either_side_are_reported_without_substituting_other_pairs() {
    let old = [
        trial("case", 0, json!(50), json!(100)),
        trial("case", 1, json!(50), json!(100)),
    ];
    let new = [
        trial("case", 1, json!(50), json!(100)),
        trial("case", 2, json!(50), json!(100)),
    ];
    let result = pair(&old, &new, "case", Metric::Decode).unwrap();
    assert_eq!(result.deltas, [0.0]);
    assert_eq!(result.exclusions.len(), 2);
    assert!(result.exclusions[0].contains("case/0: missing on one side"));
    assert!(result.exclusions[1].contains("case/2: missing on one side"));
}

#[test]
fn null_metric_exclusion_is_pairwise_and_does_not_discard_other_metrics() {
    let old = [trial("case", 0, json!(50), json!(100))];
    let new = [trial("case", 0, Value::Null, json!(110))];
    let excluded = pair(&old, &new, "case", Metric::Decode).unwrap();
    assert!(excluded.deltas.is_empty());
    assert_eq!(excluded.exclusions.len(), 1);
    assert_eq!(
        pair(&old, &new, "case", Metric::Ttft).unwrap().deltas,
        [0.1]
    );
}

#[test]
fn failed_trial_and_missing_measurement_keep_distinct_exclusion_reasons() {
    let old = [trial("case", 0, json!(50), json!(100))];
    let mut candidate = old[0].clone();
    candidate.status = "failed".into();
    assert!(
        pair(&old, &[candidate.clone()], "case", Metric::Decode)
            .unwrap()
            .exclusions[0]
            .contains("non-succeeded")
    );
    candidate.status = "succeeded".into();
    candidate.measurements.remove("decode_tok_s");
    assert!(
        pair(&old, &[candidate], "case", Metric::Decode)
            .unwrap()
            .exclusions[0]
            .contains("null measurement")
    );
}

#[test]
fn booleans_strings_and_nonpositive_baselines_do_not_enter_resampling() {
    let candidate = [trial("case", 0, json!(50), json!(100))];
    for value in [json!(true), json!("50"), json!({}), json!(0), json!(-1)] {
        let old = [trial("case", 0, value, json!(100))];
        let result = pair(&old, &candidate, "case", Metric::Decode).unwrap();
        assert!(result.deltas.is_empty());
        assert_eq!(result.exclusions.len(), 1);
    }
}

#[test]
fn duplicate_or_empty_scenario_identity_refuses_the_input_instead_of_overwriting() {
    let row = trial("case", 0, json!(50), json!(100));
    let duplicates = [row.clone(), row.clone()];
    assert!(
        pair(
            &duplicates,
            std::slice::from_ref(&row),
            "case",
            Metric::Decode
        )
        .is_err()
    );
    assert!(
        pair(
            std::slice::from_ref(&row),
            &duplicates,
            "case",
            Metric::Decode
        )
        .is_err()
    );
    let empty = trial(" ", 0, json!(50), json!(100));
    assert!(pair(&[empty], &[], "case", Metric::Decode).is_err());
}

#[test]
fn finite_values_with_overflowing_relative_degradation_are_explicitly_excluded() {
    let old = [trial("case", 0, json!(1e-300), json!(100))];
    let new = [trial("case", 0, json!(1e300), json!(100))];
    let result = pair(&old, &new, "case", Metric::Decode).unwrap();
    assert!(result.deltas.is_empty());
    assert!(result.exclusions[0].contains("non-finite relative degradation"));
}
