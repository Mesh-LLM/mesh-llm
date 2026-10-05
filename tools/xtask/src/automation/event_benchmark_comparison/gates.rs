//! Measured callback, health and observed-order facts in the final report.
use super::{callback_latency, health, manifest::Manifest, order_identity};
use crate::command::DynResult;
use serde_json::{Value, json};

pub(super) struct Results {
    pub callback: Value,
    pub health: Value,
    pub availability: Value,
    pub order_violations: Vec<String>,
    pub blocking: Vec<String>,
}

pub(super) fn evaluate(
    production: &Manifest,
    reference: &Manifest,
    baseline: &Manifest,
) -> DynResult<Results> {
    let (production_p99, production_blocked) = callback_latency::evaluate(
        &production.host,
        production.callback_ingress_p99_us,
        callback_latency::BUDGET_US,
    )?;
    let (reference_p99, reference_blocked) = callback_latency::evaluate(
        &reference.host,
        reference.callback_ingress_p99_us,
        callback_latency::BUDGET_US,
    )?;
    let mut violations = Value::Object(Default::default());
    let mut availability = Value::Object(Default::default());
    let mut unavailable = false;
    let mut health_failed = false;
    for (name, input) in [
        ("production", production),
        ("event_disabled", reference),
        ("baseline", baseline),
    ] {
        let evidence = health::evaluate(
            &input.mode,
            input.health.as_ref(),
            input.expected_dropped_progress,
            input.expected_dropped_diagnostic,
        );
        unavailable |= !evidence.available;
        health_failed |= !evidence.violations.is_empty();
        violations[name] = json!(evidence.violations);
        availability[name] = json!(evidence.available);
    }
    let keys = production
        .trials
        .iter()
        .chain(&reference.trials)
        .map(|t| (t.scenario.clone(), t.pair_index))
        .collect();
    let order = order_identity::compare(
        production.executed_order.as_deref(),
        reference.executed_order.as_deref(),
        ["production", &reference.mode],
        &keys,
    );
    let blocking = [
        (
            production_blocked || reference_blocked,
            "callback_ingress_p99",
        ),
        (unavailable, "health_unavailable"),
        (health_failed, "health_expectation_violation"),
        (!order.is_empty(), "executed_order_inconsistent"),
    ]
    .into_iter()
    .filter(|(blocked, _)| *blocked)
    .map(|(_, name)| name.to_owned())
    .collect();
    Ok(Results {
        callback: json!({"production":production_p99,"event_disabled":reference_p99}),
        health: violations,
        availability,
        order_violations: order,
        blocking,
    })
}
