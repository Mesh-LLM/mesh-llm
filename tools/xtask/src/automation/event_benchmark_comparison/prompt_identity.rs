//! Prevent seeded plans from pairing different prompts under identical row keys.
use super::manifest::Manifest;
use serde_json::Value;
use std::collections::BTreeMap;
const NATIVE_ALGORITHM: &str = "sha256-paired-trial-plan-v1";

fn note(output: &mut Vec<String>, message: String) {
    if output.len() < 128 {
        output.push(message);
    }
}
fn hash(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_str)
        .filter(|hash| hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

fn paired_prompts(production: &Manifest, other: &Manifest, label: &str, output: &mut Vec<String>) {
    let rows = other
        .trials
        .iter()
        .map(|row| ((row.scenario.as_str(), row.pair_index), row))
        .collect::<BTreeMap<_, _>>();
    for row in &production.trials {
        if let Some(other) = rows.get(&(row.scenario.as_str(), row.pair_index))
            && let (Some(a), Some(b)) = (
                hash(row.measurements.get("prompt_sha256")),
                hash(other.measurements.get("prompt_sha256")),
            )
            && !a.eq_ignore_ascii_case(b)
        {
            note(
                output,
                format!(
                    "{label}: prompt identity differs for {}/{}",
                    row.scenario, row.pair_index
                ),
            );
        }
    }
}

pub(super) fn violations(
    production: &Manifest,
    reference: &Manifest,
    baseline: &Manifest,
) -> Vec<String> {
    let inputs = [
        ("production", production),
        ("event_disabled", reference),
        ("baseline", baseline),
    ];
    if inputs
        .iter()
        .all(|(_, input)| input.trial_plan_algorithm.is_none())
    {
        return Vec::new();
    }
    let mut output = Vec::new();
    for (name, input) in inputs {
        if input.trial_plan_algorithm.as_deref() != Some(NATIVE_ALGORITHM) {
            note(
                &mut output,
                format!(
                    "{name}: trial plan algorithm is missing or incompatible with the native paired plan"
                ),
            );
        }
        if input
            .model
            .as_deref()
            .is_none_or(|model| model.trim().is_empty())
            || input.model != production.model
        {
            note(
                &mut output,
                format!(
                    "{name}: paired trial model reference is missing or differs from production"
                ),
            );
        }
        for row in &input.trials {
            if hash(row.measurements.get("prompt_sha256")).is_none() {
                note(
                    &mut output,
                    format!(
                        "{name}: complete prompt identity is missing for {}/{}",
                        row.scenario, row.pair_index
                    ),
                );
            }
        }
    }
    paired_prompts(production, reference, "comparison_a", &mut output);
    paired_prompts(production, baseline, "comparison_b", &mut output);
    output
}

#[cfg(test)]
#[path = "prompt_identity_tests.rs"]
mod tests;
