//! Paired degradation evidence. Multiplicity adjustment never changes admission.
use serde::Serialize;

use crate::command::DynResult;

const POWER_QUANTILES: f64 = 1.959_963_985 + 0.841_621_234;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Status {
    Pass,
    Fail,
    Underpowered,
    InvalidInput,
}

#[derive(Debug, Serialize)]
pub(super) struct Evidence {
    pub mean_relative_degradation: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub minimal_detectable_degradation: f64,
    pub raw_p_value: f64,
}

impl Evidence {
    pub fn status(&self, degradation_limit: f64, detectable_limit: f64) -> DynResult<Status> {
        for limit in [degradation_limit, detectable_limit] {
            if !limit.is_finite() || limit < 0.0 {
                return Err(
                    "degradation and detectability limits must be finite and nonnegative".into(),
                );
            }
        }
        Ok(if self.ci_low > degradation_limit {
            Status::Fail
        } else if self.minimal_detectable_degradation > detectable_limit {
            Status::Underpowered
        } else {
            Status::Pass
        })
    }
}

/// Admit the original paired mean and complete bootstrap resample means.
/// The caller owns deterministic resampling and its complete declared count.
pub(super) fn summarize(mean: f64, resamples: &[f64]) -> DynResult<Evidence> {
    if !mean.is_finite() || resamples.len() < 2 || resamples.iter().any(|v| !v.is_finite()) {
        return Err(
            "bootstrap evidence requires a finite paired mean and two finite resamples".into(),
        );
    }
    let mut sorted = resamples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| {
        let position = (fraction * (sorted.len() - 1) as f64).round_ties_even() as usize;
        sorted[position.min(sorted.len() - 1)]
    };
    let error = standard_error(resamples)?;
    let detectability = POWER_QUANTILES * error;
    if !detectability.is_finite() {
        return Err("bootstrap detectability overflow".into());
    }
    let below = resamples.iter().filter(|v| **v <= 0.0).count() as f64;
    let above = resamples.iter().filter(|v| **v >= 0.0).count() as f64;
    Ok(Evidence {
        mean_relative_degradation: mean,
        ci_low: percentile(0.025),
        ci_high: percentile(0.975),
        minimal_detectable_degradation: detectability,
        raw_p_value: (2.0 * below.min(above) / resamples.len() as f64).min(1.0),
    })
}

fn standard_error(values: &[f64]) -> DynResult<f64> {
    // Welford accumulation keeps the variance independent of a large sum.
    let mut mean = 0.0;
    let mut squared_distance = 0.0;
    for (index, value) in values.iter().enumerate() {
        let difference = value - mean;
        mean += difference / (index + 1) as f64;
        squared_distance += difference * (value - mean);
    }
    let variance = squared_distance / (values.len() - 1) as f64;
    if !mean.is_finite() || !variance.is_finite() || variance < 0.0 {
        return Err("bootstrap variance is not finite and nonnegative".into());
    }
    Ok(variance.sqrt())
}

/// Return wording-only Holm values in the original metric/scenario order.
pub(super) fn holm(p_values: &[f64]) -> DynResult<Vec<f64>> {
    if p_values
        .iter()
        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err("Holm inputs must be finite probabilities".into());
    }
    let mut order = (0..p_values.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| p_values[*left].total_cmp(&p_values[*right]));
    let mut adjusted = vec![0.0; p_values.len()];
    let mut maximum: f64 = 0.0;
    for (rank, index) in order.into_iter().enumerate() {
        maximum = maximum.max(((p_values.len() - rank) as f64 * p_values[index]).min(1.0));
        adjusted[index] = maximum;
    }
    Ok(adjusted)
}

#[cfg(test)]
#[path = "statistics_tests.rs"]
mod tests;
