//! Side manifests retain actual pair order and single-last-trial health scope.
use super::{paired_execution::Batch, plan, trial_contract, trial_environment, trial_profile};
use crate::command::DynResult;
use serde::Serialize;
use serde_json::{Value, json};
#[cfg(test)]
use std::ffi::OsString;
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Serialize)]
pub(super) struct Binary {
    pub path: PathBuf,
    pub sha256: String,
    pub version: Option<String>,
}

#[derive(Serialize)]
pub(super) struct Host {
    pub system: String,
    pub machine: String,
    pub certification_host: Option<&'static str>,
    pub p99_gate: &'static str,
}
impl Host {
    pub fn classify(system: String, machine: String) -> Self {
        let certification_host = match (system.as_str(), machine.as_str()) {
            ("Darwin", "arm64") => Some("macos-arm64-metal"),
            ("Linux", "x86_64") => Some("linux-x86_64-cuda"),
            _ => None,
        };
        Self {
            system,
            machine,
            certification_host,
            p99_gate: if certification_host.is_some() {
                "enforced"
            } else {
                "informational"
            },
        }
    }
}

pub(super) struct Context<'a> {
    pub spec: &'a plan::Spec,
    pub model: &'a str,
    pub source_model_sha256: &'a str,
    pub attempt: u64,
    pub generated_at: &'a str,
    pub host: &'a Host,
    pub thermal_state: &'a Value,
}

fn digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
#[cfg(test)]
pub(super) fn build(
    context: &Context<'_>,
    side: &plan::Side,
    identity: &Binary,
    child_environment: &BTreeMap<OsString, OsString>,
    batch: &Batch,
    index: usize,
) -> DynResult<Value> {
    let environment = trial_environment::snapshot(child_environment);
    let inheritance = trial_profile::Inheritance::default();
    build_normalized(
        context,
        side,
        identity,
        Some(&environment),
        Some(&inheritance),
        batch,
        index,
    )
}

pub(super) fn build_normalized(
    context: &Context<'_>,
    side: &plan::Side,
    identity: &Binary,
    environment: Option<&BTreeMap<String, trial_environment::Entry>>,
    inheritance: Option<&trial_profile::Inheritance>,
    batch: &Batch,
    index: usize,
) -> DynResult<Value> {
    context.spec.validate()?;
    if index > 1
        || batch.side_ids[index] != side.side_id
        || context.attempt == 0
        || context.model.trim().is_empty()
        || context.generated_at.trim().is_empty()
        || !digest(context.source_model_sha256)
        || !digest(&identity.sha256)
        || !identity.path.is_absolute()
        || identity.path != side.binary
    {
        return Err(
            "manifest requires admitted binary/model identities, positive attempt and side index"
                .into(),
        );
    }
    let launched = batch.trials[index].iter().any(|trial| trial.launched);
    if launched && (environment.is_none() || inheritance.is_none()) {
        return Err(
            "launched side requires actual child environment and inheritance provenance".into(),
        );
    }
    if environment.is_some_and(|environment| {
        environment
            .get(trial_environment::GATE)
            .is_none_or(|entry| entry.redacted || entry.value != true)
            || environment
                .get(trial_environment::SELECTOR)
                .is_none_or(|entry| entry.redacted || entry.value != side.mode.label())
    }) {
        return Err(
            "manifest child environment does not carry the admitted benchmark gate and mode".into(),
        );
    }
    let progress = u64::from(side.mode == plan::Mode::EventDisabled && launched);
    let health = launched
        .then_some(batch.final_health[index].health.as_ref())
        .flatten();
    let p99 = launched
        .then_some(batch.final_health[index].ingress_p99_us)
        .flatten();
    Ok(json!({
        "schema_version":1,
        "metrics_schema":"streaming_v1",
        "trial_plan_algorithm":plan::ALGORITHM,
        "mode":side.mode.label(),
        "binary":identity,
        "model":context.model,
        "source_model_sha256":context.source_model_sha256,
        "seed":context.spec.seed,
        "pairs_primary":context.spec.pairs_primary,
        "pairs_scenario":context.spec.pairs_scenario,
        "scenarios":context.spec.scenarios,
        "attempt":context.attempt,
        "generated_at":context.generated_at,
        "host":context.host,
        "thermal_state":context.thermal_state,
        "environment":environment,
        "inherited_profile":inheritance,
        "trial_unit":trial_contract::unit(),
        "callback_ingress_p99_us":p99,
        "health":health,
        "health_observation_error":if launched { batch.trials[index].last().and_then(|trial| trial.health_observation_error.as_deref()) } else { None },
        "health_scope":"side_last_trial_final_observed_log",
        "expected_dropped_progress":progress,
        "expected_dropped_diagnostic":0,
        "trials":batch.trials[index],
        "executed_order":batch.executed_order,
        "execution_incomplete":batch.interrupted,
    }))
}

#[cfg(test)]
#[path = "manifest_output_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "producer_gaps_tests.rs"]
mod producer_gaps_tests;
