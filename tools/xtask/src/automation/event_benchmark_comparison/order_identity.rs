//! Observed interleaved order, not a label-only per-trial promise.
use serde::Deserialize;
use std::collections::BTreeSet;

#[derive(Clone, PartialEq, Eq, Deserialize)]
pub(super) struct Order {
    pub scenario: String,
    pub pair_index: u64,
    pub order: [String; 2],
}

pub(super) type Keys = BTreeSet<(String, u64)>;

fn identities(records: &[Order], sides: [&str; 2], keys: &Keys) -> Vec<String> {
    let mut observed = Keys::new();
    let mut violations = Vec::new();
    for record in records {
        let key = (record.scenario.clone(), record.pair_index);
        if !observed.insert(key) {
            violations.push("executed_order contains duplicate pair identity".into());
        }
        let forward = record.order[0] == sides[0] && record.order[1] == sides[1];
        let reverse = record.order[0] == sides[1] && record.order[1] == sides[0];
        if !forward && !reverse {
            violations.push("executed_order contains an invalid comparison-side identity".into());
        }
    }
    if &observed != keys {
        violations
            .push("executed_order does not cover the complete observed trial pair census".into());
    }
    violations
}

pub(super) fn compare(
    production: Option<&[Order]>,
    reference: Option<&[Order]>,
    sides: [&str; 2],
    keys: &Keys,
) -> Vec<String> {
    let mut violations = Vec::new();
    let production = production.filter(|v| !v.is_empty());
    let reference = reference.filter(|v| !v.is_empty());
    if production.is_none() {
        violations.push("production manifest is missing executed_order".into());
    }
    if reference.is_none() {
        violations.push("event_disabled manifest is missing executed_order".into());
    }
    let Some((production, reference)) = production.zip(reference) else {
        return violations;
    };
    if production != reference {
        violations.push(
            "executed_order disagrees between the production and event_disabled manifests".into(),
        );
        return violations;
    }
    violations.extend(identities(production, sides, keys));
    let orders = production
        .iter()
        .map(|record| &record.order)
        .collect::<BTreeSet<_>>();
    if orders.len() < 2 {
        violations.push("executed_order is constant across all matched pairs; randomized per pair trial-unit requirement is not honored".into());
    }
    violations
}

#[cfg(test)]
#[path = "order_identity_tests.rs"]
mod tests;
