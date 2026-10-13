//! Metric-specific pairing without replacing missing or unusable trials.
use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use serde_json::Value;

use crate::command::DynResult;

#[derive(Clone, Copy)]
pub(super) enum Metric {
    Decode,
    DecodeOnly,
    Ttft,
}

impl Metric {
    pub fn name(self) -> &'static str {
        match self {
            Self::Decode => "decode_tok_s",
            Self::DecodeOnly => "decode_only_tok_s",
            Self::Ttft => "ttft_ms",
        }
    }

    fn delta(self, baseline: f64, candidate: f64) -> f64 {
        match self {
            Self::Ttft => (candidate - baseline) / baseline,
            Self::Decode | Self::DecodeOnly => (baseline - candidate) / baseline,
        }
    }
}

#[derive(Clone, Deserialize)]
pub(super) struct Trial {
    pub scenario: String,
    pub pair_index: u64,
    pub status: String,
    #[serde(flatten)]
    pub measurements: BTreeMap<String, Value>,
}

pub(super) struct Paired {
    pub deltas: Vec<f64>,
    pub exclusions: Vec<String>,
}

type Key = (String, u64);

fn index(trials: &[Trial]) -> DynResult<BTreeMap<Key, &Trial>> {
    let mut result = BTreeMap::new();
    for trial in trials {
        if trial.scenario.trim().is_empty() {
            return Err("trial scenario must be nonempty".into());
        }
        let key = (trial.scenario.clone(), trial.pair_index);
        if result.insert(key, trial).is_some() {
            return Err("duplicate trial scenario and pair index".into());
        }
    }
    Ok(result)
}

fn values(baseline: &Trial, candidate: &Trial, metric: Metric) -> Result<(f64, f64), &'static str> {
    if baseline.status != "succeeded" || candidate.status != "succeeded" {
        return Err("non-succeeded trial");
    }
    let old = baseline
        .measurements
        .get(metric.name())
        .unwrap_or(&Value::Null);
    let new = candidate
        .measurements
        .get(metric.name())
        .unwrap_or(&Value::Null);
    if old.is_null() || new.is_null() {
        return Err("null measurement");
    }
    let numbers = old
        .as_f64()
        .filter(|v| v.is_finite())
        .zip(new.as_f64().filter(|v| v.is_finite()));
    let (old, new) = numbers.ok_or("non-finite or non-numeric measurement")?;
    if old <= 0.0 {
        return Err("non-positive baseline measurement");
    }
    Ok((old, new))
}

pub(super) fn pair(
    baseline: &[Trial],
    candidate: &[Trial],
    group: &str,
    metric: Metric,
) -> DynResult<Paired> {
    let old = index(baseline)?;
    let new = index(candidate)?;
    let keys = old
        .keys()
        .chain(new.keys())
        .filter(|key| key.0 == group)
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut result = Paired {
        deltas: Vec::new(),
        exclusions: Vec::new(),
    };
    for key in keys {
        let delta = match old.get(&key).zip(new.get(&key)) {
            None => Err("missing on one side"),
            Some((baseline, candidate)) => {
                values(baseline, candidate, metric).and_then(|(baseline, candidate)| {
                    let delta = metric.delta(baseline, candidate);
                    if delta.is_finite() {
                        Ok(delta)
                    } else {
                        Err("non-finite relative degradation")
                    }
                })
            }
        };
        match delta {
            Ok(value) => result.deltas.push(value),
            Err(reason) => {
                result
                    .exclusions
                    .push(format!("{}/{}: {reason} ({})", key.0, key.1, metric.name()))
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
#[path = "pairing_tests.rs"]
mod tests;
