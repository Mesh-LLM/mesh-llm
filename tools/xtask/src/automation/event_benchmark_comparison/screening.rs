//! Complete paired-metric screening and wording-only report projection.
use serde_json::{Value, json};

use super::{
    pairing::{self, Metric, Trial},
    resampling,
    statistics::{self, Evidence, Status},
};
use crate::command::DynResult;

pub(super) struct Policy {
    pub required_pairs: usize,
    pub bootstrap_samples: usize,
    pub seed: u64,
    pub degradation_limit: f64,
    pub detectable_limit: f64,
}

impl Policy {
    fn validate(&self) -> DynResult<()> {
        if !(1..=10_000).contains(&self.required_pairs)
            || !(2..=1_000_000).contains(&self.bootstrap_samples)
            || [self.degradation_limit, self.detectable_limit]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(
                "metric screen requires bounded positive counts and finite nonnegative limits"
                    .into(),
            );
        }
        Ok(())
    }
}

pub(super) struct Screen {
    pub group: String,
    pub metric: &'static str,
    pub status: Status,
    pub valid_pairs: usize,
    pub required_pairs: usize,
    pub exclusions: Vec<String>,
    pub evidence: Option<Evidence>,
}

pub(super) fn screen(
    baseline: &[Trial],
    candidate: &[Trial],
    group: &str,
    metric: Metric,
    policy: &Policy,
) -> DynResult<Screen> {
    policy.validate()?;
    let pairs = pairing::pair(baseline, candidate, group, metric)?;
    let evidence = if pairs.deltas.len() < policy.required_pairs {
        None
    } else {
        Some(resampling::bootstrap(
            &pairs.deltas,
            policy.bootstrap_samples,
            policy.seed,
            group,
            metric.name(),
        )?)
    };
    let status = match &evidence {
        Some(evidence) => evidence.status(policy.degradation_limit, policy.detectable_limit)?,
        None => Status::InvalidInput,
    };
    Ok(Screen {
        group: group.into(),
        metric: metric.name(),
        status,
        valid_pairs: pairs.deltas.len(),
        required_pairs: policy.required_pairs,
        exclusions: pairs.exclusions,
        evidence,
    })
}

pub(super) fn overall(screens: &[Screen]) -> DynResult<Status> {
    if screens.is_empty() {
        return Err("an empty comparison has no statistical verdict".into());
    }
    for status in [Status::Fail, Status::InvalidInput, Status::Underpowered] {
        if screens.iter().any(|screen| screen.status == status) {
            return Ok(status);
        }
    }
    Ok(Status::Pass)
}

fn percentage(value: Option<f64>) -> DynResult<Option<f64>> {
    value
        .map(|value| {
            let result = value * 100.0;
            if !result.is_finite() {
                return Err("reported degradation percentage overflow".into());
            }
            Ok(result)
        })
        .transpose()
}

fn wording(status: Status) -> &'static str {
    match status {
        Status::Pass => "not proven worse by this screen",
        Status::Fail => {
            "this screen found a degradation whose 95% CI lies entirely above the threshold"
        }
        Status::Underpowered => {
            "UNDERPOWERED: minimal detectable degradation exceeds the configured bound; collect more pairs"
        }
        Status::InvalidInput => "invalid input: fewer valid pairs than required",
    }
}

pub(super) fn report(screens: &[Screen], report_holm: bool) -> DynResult<Value> {
    overall(screens)?;
    let probabilities = screens
        .iter()
        .filter_map(|screen| screen.evidence.as_ref().map(|e| e.raw_p_value))
        .collect::<Vec<_>>();
    let adjusted = if report_holm {
        statistics::holm(&probabilities)?
    } else {
        Vec::new()
    };
    let mut values = adjusted.into_iter();
    let mut entries = Vec::with_capacity(screens.len());
    for screen in screens {
        let evidence = screen.evidence.as_ref();
        let mut entry = json!({
            "group":screen.group,"metric":screen.metric,"status":screen.status,
            "valid_pairs":screen.valid_pairs,"required_pairs":screen.required_pairs,
            "mean_relative_degradation_pct":percentage(evidence.map(|e| e.mean_relative_degradation))?,
            "ci_low_pct":percentage(evidence.map(|e| e.ci_low))?,
            "ci_high_pct":percentage(evidence.map(|e| e.ci_high))?,
            "minimal_detectable_degradation_pct":percentage(evidence.map(|e| e.minimal_detectable_degradation))?,
            "exclusions":screen.exclusions,"wording":wording(screen.status)
        });
        if report_holm {
            entry["raw_p_value"] = json!(evidence.map(|e| e.raw_p_value));
            entry["holm_adjusted_p_value"] = json!(if evidence.is_some() {
                values.next()
            } else {
                None
            });
        }
        entries.push(entry);
    }
    Ok(Value::Array(entries))
}

#[cfg(test)]
#[path = "screening_tests.rs"]
mod tests;
