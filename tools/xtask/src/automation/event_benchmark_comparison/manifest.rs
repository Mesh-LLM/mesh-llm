//! Bounded typed event benchmark inputs and complete declared trial identity.
use std::collections::BTreeSet;

use serde::Deserialize;
use serde_json::Value;

use super::{
    callback_latency::Host,
    environment_identity::{self, Environment},
    health::Health,
    order_identity::Order,
    pairing::Trial,
};
use crate::command::DynResult;

pub(super) const PRIMARY: &str = "__primary__";
const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;

#[derive(Deserialize)]
pub(super) struct Manifest {
    pub schema_version: u32,
    pub metrics_schema: String,
    pub mode: String,
    pub seed: u64,
    #[serde(default = "first_attempt")]
    pub attempt: u64,
    pub environment: Environment,
    pub host: Host,
    #[serde(default)]
    pub health: Option<Health>,
    #[serde(default)]
    pub callback_ingress_p99_us: Option<f64>,
    #[serde(default)]
    pub expected_dropped_progress: Option<u64>,
    #[serde(default)]
    pub expected_dropped_diagnostic: Option<u64>,
    #[serde(default)]
    pub scenarios: Vec<String>,
    pub trials: Vec<Trial>,
    #[serde(default)]
    pub executed_order: Option<Vec<Order>>,
    #[serde(default)]
    pub binary: Value,
    #[serde(default)]
    pub thermal_state: Value,
}

fn first_attempt() -> u64 {
    1
}

impl Manifest {
    pub fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1 || self.metrics_schema != "streaming_v1" {
            return Err(
                "benchmark manifests require schema1 and exact streaming_v1 metrics".into(),
            );
        }
        if !["production", "event-disabled", "off"].contains(&self.mode.as_str())
            || self.attempt == 0
        {
            return Err("benchmark mode and positive attempt are required".into());
        }
        self.host.validate()?;
        environment_identity::validate(&self.environment)?;
        self.validate_scenarios()?;
        self.validate_trials()
    }

    fn validate_scenarios(&self) -> DynResult<()> {
        if self.scenarios.len() > 64 {
            return Err("benchmark scenario count exceeds 64".into());
        }
        let mut names = BTreeSet::new();
        for scenario in &self.scenarios {
            if scenario.trim().is_empty() || scenario == PRIMARY || !names.insert(scenario) {
                return Err(
                    "scenario names must be unique nonempty names distinct from primary".into(),
                );
            }
        }
        Ok(())
    }

    fn validate_trials(&self) -> DynResult<()> {
        if self.trials.len() > 1_000_000 {
            return Err("benchmark trial count exceeds one million".into());
        }
        let mut identities = BTreeSet::new();
        for trial in &self.trials {
            if trial.scenario != PRIMARY && !self.scenarios.contains(&trial.scenario) {
                return Err("trial belongs to an undeclared benchmark scenario".into());
            }
            if !identities.insert((&trial.scenario, trial.pair_index)) {
                return Err("duplicate trial scenario and pair index".into());
            }
        }
        Ok(())
    }
}

pub(super) fn decode(bytes: &[u8]) -> DynResult<Manifest> {
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Err("benchmark manifest exceeds 16 MiB".into());
    }
    let manifest: Manifest = serde_json::from_slice(bytes)?;
    manifest.validate()?;
    Ok(manifest)
}

#[cfg(test)]
#[path = "manifest_tests.rs"]
mod tests;
