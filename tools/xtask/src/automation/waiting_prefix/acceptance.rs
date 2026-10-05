//! Typed acceptance of measured waiting-prefix A/B results.
use crate::command::DynResult;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub(super) enum Version {
    Old,
    New,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct Aggregate {
    pub version: Version,
    pub rounds: u64,
    pub requests: u64,
    pub successful: u64,
    pub capacity_rejections: u64,
    pub cache_hits_median: Option<f64>,
    pub suffix_prefill_tokens_median: Option<f64>,
    pub resident_evicted_tokens_median: Option<f64>,
    pub resident_evicted_entries_median: Option<f64>,
    pub predicted_recompute_cost_median: Option<f64>,
    pub ttft_ms_p50_median: Option<f64>,
    pub ttft_ms_p95_median: Option<f64>,
    pub makespan_ms_median: Option<f64>,
    pub output_tokens_per_second_median: Option<f64>,
    pub family_switches_median: Option<f64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Range {
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Contract {
    pub successful_requests_per_binary: u64,
    pub absolute_user_metric_delta_percent_max: Option<f64>,
    pub suffix_prefill_before_min: Option<f64>,
    pub family_switch_before_min: Option<f64>,
    pub capacity_rejections_after_max: Option<u64>,
    pub resident_evicted_tokens_after_min: Option<f64>,
    pub predicted_recompute_cost_after_min: Option<f64>,
    pub suffix_prefill_delta_percent: Option<Range>,
    pub suffix_prefill_delta_percent_max: Option<f64>,
    pub suffix_prefill_delta_percent_min: Option<f64>,
    pub family_switch_delta_percent: Option<Range>,
    pub family_switch_delta_percent_max: Option<f64>,
    pub family_switch_delta_percent_min: Option<f64>,
    pub ttft_p95_delta_percent: Option<Range>,
    pub ttft_p95_delta_percent_max: Option<f64>,
    pub ttft_p95_delta_percent_min: Option<f64>,
    pub makespan_delta_percent: Option<Range>,
    pub makespan_delta_percent_max: Option<f64>,
    pub makespan_delta_percent_min: Option<f64>,
    pub output_throughput_delta_percent: Option<Range>,
    pub output_throughput_delta_percent_max: Option<f64>,
    pub output_throughput_delta_percent_min: Option<f64>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "lowercase", tag = "relation", content = "threshold")]
pub(super) enum Requirement {
    Eq(u64),
    Max(f64),
    Min(f64),
    Range(Range),
}

#[derive(Debug, Serialize)]
pub(super) struct Check {
    pub label: String,
    pub actual: Option<f64>,
    #[serde(flatten)]
    pub requirement: Requirement,
    pub passed: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct Acceptance {
    pub passed: bool,
    pub checks: Vec<Check>,
}

pub(super) fn pair(rows: &[Aggregate]) -> DynResult<(&Aggregate, &Aggregate)> {
    let [a, b] = rows else {
        return Err("comparison requires exactly one before and one after aggregate".into());
    };
    match (a.version, b.version) {
        (Version::Old, Version::New) => Ok((a, b)),
        (Version::New, Version::Old) => Ok((b, a)),
        _ => Err("comparison has duplicate binary versions".into()),
    }
}

pub(super) fn delta(old: Option<f64>, new: Option<f64>) -> Option<f64> {
    let (old, new) = (old?, new?);
    if !old.is_finite() || !new.is_finite() || old < 0.0 || new < 0.0 {
        return None;
    }
    let value = if old == 0.0 {
        if new != 0.0 {
            return None;
        }
        0.0
    } else {
        (new - old) / old * 100.0
    };
    value.is_finite().then_some(value)
}

fn record(checks: &mut Vec<Check>, label: &str, actual: Option<f64>, requirement: Requirement) {
    let actual = actual.filter(|value| value.is_finite());
    let passed = actual.is_some_and(|value| match requirement {
        Requirement::Eq(expected) => value == expected as f64,
        Requirement::Max(max) => max.is_finite() && value <= max,
        Requirement::Min(min) => min.is_finite() && value >= min,
        Requirement::Range(range) => {
            range.min.is_finite()
                && range.max.is_finite()
                && range.min <= range.max
                && range.min <= value
                && value <= range.max
        }
    });
    checks.push(Check {
        label: label.into(),
        actual,
        requirement,
        passed,
    });
}

fn request_counts(checks: &mut Vec<Check>, before: &Aggregate, after: &Aggregate, expected: u64) {
    for (label, row) in [("before", before), ("after", after)] {
        for (suffix, count) in [
            ("successful requests", expected),
            ("complete requests", row.requests),
        ] {
            checks.push(Check {
                label: format!("{label} {suffix}"),
                actual: Some(row.successful as f64),
                requirement: Requirement::Eq(count),
                passed: row.successful == count,
            });
        }
        record(
            checks,
            &format!("{label} measured rounds"),
            Some(row.rounds as f64),
            Requirement::Min(1.0),
        );
    }
}

fn deltas(checks: &mut Vec<Check>, before: &Aggregate, after: &Aggregate, contract: &Contract) {
    for (label, old, new, range, max, min) in [
        (
            "suffix_prefill",
            before.suffix_prefill_tokens_median,
            after.suffix_prefill_tokens_median,
            contract.suffix_prefill_delta_percent,
            contract.suffix_prefill_delta_percent_max,
            contract.suffix_prefill_delta_percent_min,
        ),
        (
            "family_switch",
            before.family_switches_median,
            after.family_switches_median,
            contract.family_switch_delta_percent,
            contract.family_switch_delta_percent_max,
            contract.family_switch_delta_percent_min,
        ),
        (
            "ttft_p95",
            before.ttft_ms_p95_median,
            after.ttft_ms_p95_median,
            contract.ttft_p95_delta_percent,
            contract.ttft_p95_delta_percent_max,
            contract.ttft_p95_delta_percent_min,
        ),
        (
            "makespan",
            before.makespan_ms_median,
            after.makespan_ms_median,
            contract.makespan_delta_percent,
            contract.makespan_delta_percent_max,
            contract.makespan_delta_percent_min,
        ),
        (
            "output_throughput",
            before.output_tokens_per_second_median,
            after.output_tokens_per_second_median,
            contract.output_throughput_delta_percent,
            contract.output_throughput_delta_percent_max,
            contract.output_throughput_delta_percent_min,
        ),
    ] {
        let label = format!("{label} delta percent");
        let actual = delta(old, new);
        for requirement in [
            range.map(Requirement::Range),
            max.map(Requirement::Max),
            min.map(Requirement::Min),
        ]
        .into_iter()
        .flatten()
        {
            record(checks, &label, actual, requirement);
        }
    }
}

pub(super) fn evaluate(rows: &[Aggregate], contract: &Contract) -> DynResult<Acceptance> {
    if contract.successful_requests_per_binary == 0 {
        return Err("acceptance requires a positive successful request count".into());
    }
    let (before, after) = pair(rows)?;
    let mut checks = Vec::new();
    request_counts(
        &mut checks,
        before,
        after,
        contract.successful_requests_per_binary,
    );
    for (label, actual, threshold) in [
        (
            "before suffix prefill pressure",
            before.suffix_prefill_tokens_median,
            contract.suffix_prefill_before_min,
        ),
        (
            "before family-switch pressure",
            before.family_switches_median,
            contract.family_switch_before_min,
        ),
        (
            "after resident KV evicted tokens per round",
            after.resident_evicted_tokens_median,
            contract.resident_evicted_tokens_after_min,
        ),
        (
            "after predicted recompute cost per round",
            after.predicted_recompute_cost_median,
            contract.predicted_recompute_cost_after_min,
        ),
    ] {
        if let Some(threshold) = threshold {
            record(
                &mut checks,
                label,
                actual.filter(|v| *v >= 0.0),
                Requirement::Min(threshold),
            );
        }
    }
    if let Some(max) = contract.capacity_rejections_after_max {
        checks.push(Check {
            label: "after capacity rejections".into(),
            actual: Some(after.capacity_rejections as f64),
            requirement: Requirement::Max(max as f64),
            passed: after.capacity_rejections <= max,
        });
    }
    deltas(&mut checks, before, after, contract);
    if let Some(max) = contract.absolute_user_metric_delta_percent_max {
        for (label, old, new) in [
            (
                "TTFT p50",
                before.ttft_ms_p50_median,
                after.ttft_ms_p50_median,
            ),
            (
                "TTFT p95",
                before.ttft_ms_p95_median,
                after.ttft_ms_p95_median,
            ),
            (
                "makespan",
                before.makespan_ms_median,
                after.makespan_ms_median,
            ),
            (
                "output throughput",
                before.output_tokens_per_second_median,
                after.output_tokens_per_second_median,
            ),
        ] {
            record(
                &mut checks,
                &format!("absolute {label} delta percent"),
                delta(old, new).map(f64::abs),
                Requirement::Max(max),
            );
        }
    }
    Ok(Acceptance {
        passed: checks.iter().all(|check| check.passed),
        checks,
    })
}
