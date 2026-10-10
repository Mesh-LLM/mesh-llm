use super::*;

fn trial(index: u64, rate: Value) -> Trial {
    serde_json::from_value(json!({"scenario":"__primary__","pair_index":index,"status":"succeeded","decode_tok_s":rate})).unwrap()
}

fn policy() -> Policy {
    Policy {
        required_pairs: 20,
        bootstrap_samples: 10_000,
        seed: 42,
        degradation_limit: 0.03,
        detectable_limit: 0.03,
    }
}

fn measured(rates: &[f64]) -> Screen {
    let old = rates
        .iter()
        .enumerate()
        .map(|(i, _)| trial(i as u64, json!(100.0)))
        .collect::<Vec<_>>();
    let new = rates
        .iter()
        .enumerate()
        .map(|(i, rate)| trial(i as u64, json!(rate)))
        .collect::<Vec<_>>();
    screen(&old, &new, "__primary__", Metric::Decode, &policy()).unwrap()
}

#[test]
fn insufficient_valid_pairs_remain_invalid_without_resampling_or_lowering_minimum() {
    let old = (0..20).map(|i| trial(i, json!(100))).collect::<Vec<_>>();
    let mut new = old.clone();
    new[19]
        .measurements
        .insert("decode_tok_s".into(), Value::Null);
    let result = screen(&old, &new, "__primary__", Metric::Decode, &policy()).unwrap();
    assert_eq!(result.status, Status::InvalidInput);
    assert_eq!((result.valid_pairs, result.required_pairs), (19, 20));
    assert_eq!(result.exclusions.len(), 1);
    assert!(result.evidence.is_none());
    let report = report(&[result], true).unwrap();
    assert!(report[0]["ci_low_pct"].is_null());
    assert!(report[0]["holm_adjusted_p_value"].is_null());
}

#[test]
fn clear_degradation_fails_and_holm_never_changes_the_verdict() {
    let result = measured(&[90.0; 20]);
    assert_eq!(result.status, Status::Fail);
    let screens = [result];
    assert_eq!(overall(&screens).unwrap(), Status::Fail);
    let report = report(&screens, true).unwrap();
    assert_eq!(report[0]["status"], "fail");
    assert_eq!(overall(&screens).unwrap(), Status::Fail);
    assert!((report[0]["ci_low_pct"].as_f64().unwrap() - 10.0).abs() < 1e-12);
}

#[test]
fn neutral_complete_screen_uses_only_the_permitted_wording() {
    let screens = [measured(&[100.0; 20])];
    assert_eq!(overall(&screens).unwrap(), Status::Pass);
    let projected = report(&screens, false).unwrap();
    assert_eq!(projected[0]["wording"], "not proven worse by this screen");
    assert!(projected[0].get("raw_p_value").is_none());
    assert!(projected[0].get("holm_adjusted_p_value").is_none());
    for phrase in [
        "proven within",
        "proven to be within",
        "statistically proven",
    ] {
        assert!(!projected.to_string().contains(phrase));
    }
}

#[test]
fn high_variance_neutral_mean_is_underpowered_and_not_a_pass() {
    let rates = (0..20)
        .map(|i| if i % 2 == 0 { 50.0 } else { 150.0 })
        .collect::<Vec<_>>();
    let screens = [measured(&rates)];
    assert_eq!(screens[0].status, Status::Underpowered);
    assert_eq!(overall(&screens).unwrap(), Status::Underpowered);
    assert!(
        report(&screens, true).unwrap()[0]["wording"]
            .as_str()
            .unwrap()
            .contains("UNDERPOWERED")
    );
}

#[test]
fn invalid_policy_and_empty_comparison_are_refused() {
    let mut configuration = policy();
    configuration.required_pairs = 0;
    assert!(screen(&[], &[], "case", Metric::Decode, &configuration).is_err());
    configuration = policy();
    configuration.degradation_limit = f64::NAN;
    assert!(screen(&[], &[], "case", Metric::Decode, &configuration).is_err());
    assert!(overall(&[]).is_err());
    assert!(report(&[], true).is_err());
    assert!(percentage(Some(f64::MAX)).is_err());
}

#[test]
fn failure_invalidity_and_power_have_explicit_conservative_precedence() {
    let failure = measured(&[90.0; 20]);
    let invalid = screen(&[], &[], "case", Metric::Decode, &policy()).unwrap();
    assert_eq!(overall(&[invalid, failure]).unwrap(), Status::Fail);
    let invalid = screen(&[], &[], "case", Metric::Decode, &policy()).unwrap();
    let rates = (0..20)
        .map(|i| if i % 2 == 0 { 50.0 } else { 150.0 })
        .collect::<Vec<_>>();
    assert_eq!(
        overall(&[invalid, measured(&rates)]).unwrap(),
        Status::InvalidInput
    );
}
