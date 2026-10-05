//! Complete event benchmark verdict assembled from independent measured gates.
use super::{
    environment_identity, gates,
    manifest::{Manifest, PRIMARY},
    options::Options,
    pairing::Metric,
    resampling, retry,
    screening::{self, Screen},
    statistics::Status,
};
use crate::command::DynResult;
use serde_json::{Value, json};

fn comparison(
    baseline: &Manifest,
    candidate: &Manifest,
    options: &Options,
) -> DynResult<Vec<Screen>> {
    let mut screens = Vec::new();
    for group in std::iter::once(PRIMARY).chain(candidate.scenarios.iter().map(String::as_str)) {
        for metric in [Metric::Decode, Metric::DecodeOnly, Metric::Ttft] {
            screens.push(screening::screen(
                &baseline.trials,
                &candidate.trials,
                group,
                metric,
                &options.policy(group),
            )?);
        }
    }
    Ok(screens)
}

fn statistical_reasons(a: Status, b: Status) -> Vec<String> {
    [
        (Status::Fail, "degradation_fail"),
        (Status::Underpowered, "underpowered"),
        (Status::InvalidInput, "insufficient_pairs"),
    ]
    .into_iter()
    .filter(|(status, _)| a == *status || b == *status)
    .map(|(_, name)| name.to_owned())
    .collect()
}

pub(super) fn build(
    production: &Manifest,
    reference: &Manifest,
    baseline: &Manifest,
    options: &Options,
) -> DynResult<Value> {
    for input in [production, reference, baseline] {
        input.validate()?;
    }
    options.validate(production)?;
    if production.mode != "production"
        || baseline.mode != "production"
        || !["event-disabled", "off"].contains(&reference.mode.as_str())
    {
        return Err(
            "comparison requires production current/baseline and event-disabled or off reference"
                .into(),
        );
    }
    let seeds=[("production",production),("event_disabled",reference),("baseline",baseline)].into_iter().filter(|(_,input)|input.seed!=options.seed).map(|(name,input)|format!("{name}: manifest seed {} does not match --seed {}; pairing requires the SAME seed on every side",input.seed,options.seed)).collect::<Vec<_>>();
    let environment_a =
        environment_identity::compare(&production.environment, &reference.environment, true)?;
    let environment_b =
        environment_identity::compare(&production.environment, &baseline.environment, false)?;
    let a = comparison(reference, production, options)?;
    let b = comparison(baseline, production, options)?;
    let status_a = screening::overall(&a)?;
    let status_b = screening::overall(&b)?;
    let gates = gates::evaluate(production, reference, baseline)?;
    let decision = retry::classify(
        status_a != Status::Pass || status_b != Status::Pass,
        production.attempt,
    )?;
    let mut reasons = Vec::new();
    for (blocked, name) in [
        (!seeds.is_empty(), "seed_mismatch"),
        (
            !environment_a.is_empty(),
            "environment_mismatch_comparison_a",
        ),
        (
            !environment_b.is_empty(),
            "environment_mismatch_comparison_b",
        ),
    ] {
        if blocked {
            reasons.push(name.into());
        }
    }
    reasons.extend(statistical_reasons(status_a, status_b));
    reasons.extend(gates.blocking);
    if decision.action == "blocked_retry_exhausted" {
        reasons.push("retry_exhausted".into());
    }
    Ok(json!({
        "schema_version":1,"generated_with_seed":options.seed,"bootstrap_samples":options.bootstrap_samples,
        "resampling_algorithm":resampling::ALGORITHM,
        "max_degradation_percent":options.max_degradation_percent,"max_mdd_percent":options.max_mdd_percent,
        "min_primary_pairs":options.min_primary_pairs,"min_scenario_pairs":options.min_scenario_pairs,"report_holm":options.report_holm,
        "input_violations":seeds,
        "comparison_a":{"description":format!("production vs {} on the current binary",reference.mode),"status":status_a,"environment_violations":environment_a,"metrics":screening::report(&a,options.report_holm)?},
        "comparison_b":{"description":"current-binary production vs baseline-binary production","status":status_b,"environment_violations":environment_b,"metrics":screening::report(&b,options.report_holm)?},
        "callback_ingress_p99":gates.callback,"health":gates.health,"health_availability":gates.availability,
        "executed_order_violations":gates.order_violations,"retry":decision,
        "binary_identity":{"production":production.binary,"event_disabled":reference.binary,"baseline":baseline.binary},
        "thermal_state":{"production":production.thermal_state,"event_disabled":reference.thermal_state,"baseline":baseline.thermal_state},
        "certification_status":if reasons.is_empty() {"pass"} else {"blocked"},"blocking_reasons":reasons,
        "wording":"not proven worse by this screen"
    }))
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod tests;
