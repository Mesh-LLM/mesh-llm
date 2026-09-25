//! `compare_reports` of `collect-ci-metrics.py`: compares the baseline
//! report built from `--compare-input` with the candidate report without
//! conflating provider or capacity effects. Both reports are the port's own
//! `analyze` output, read back through the legacy fallbacks
//! (`jobs.comparison_cohort` falling back to `jobs`, `providers` and
//! `dimensions` falling back to `jobs.by_runner`, `job_names` to
//! `jobs.by_name`).

use crate::ci_operations::ci_metrics_markdown_format::{field, truthy};
use crate::ci_operations::ci_metrics_observe::{
    MIN_HEURISTIC_SAMPLES, QUEUE_CONTAMINATION_SECONDS, QUEUE_WARN_SECONDS,
};
use crate::ci_operations::ci_metrics_stats::round;
use crate::ci_operations::ci_metrics_value::{Value, display, object};
use std::collections::BTreeSet;

const DIMENSIONS: [&str; 3] = ["operating_system", "architecture", "runner_role"];

fn items(value: &Value) -> &[Value] {
    match value {
        Value::Array(items) => items,
        _ => &[],
    }
}

/// `_comparison_cohort(report)`.
fn cohort(report: &Value) -> &Value {
    let jobs = field(report, "jobs");
    jobs.get("comparison_cohort").unwrap_or(jobs)
}

fn by_runner(report: &Value) -> &[Value] {
    items(field(field(report, "jobs"), "by_runner"))
}

fn known_provider(value: &Value) -> bool {
    truthy(value) && !matches!(value, Value::Str(text) if text == "unknown")
}

/// `_providers(report)`.
fn providers(report: &Value) -> BTreeSet<String> {
    if let Some(list) = cohort(report).get("providers") {
        return items(list)
            .iter()
            .filter(|provider| known_provider(provider))
            .map(display)
            .collect();
    }
    by_runner(report)
        .iter()
        .filter_map(|item| item.get("provider"))
        .filter(|provider| known_provider(provider))
        .map(display)
        .collect()
}

/// `_runner_dimension_values(report, key)`.
fn dimension_values(report: &Value, key: &str) -> BTreeSet<String> {
    let values: Vec<&Value> = match cohort(report).get("dimensions") {
        Some(dimensions) => items(field(dimensions, key)).iter().collect(),
        None => by_runner(report)
            .iter()
            .filter_map(|item| item.get(key))
            .collect(),
    };
    values
        .into_iter()
        .filter(|value| truthy(value))
        .map(display)
        .collect()
}

/// `set(cohort.get("job_names", [])) or {item["name"] for item in by_name}`.
fn job_names(report: &Value) -> BTreeSet<String> {
    let listed: BTreeSet<String> = cohort(report)
        .get("job_names")
        .map(|names| items(names).iter().map(display).collect())
        .unwrap_or_default();
    if !listed.is_empty() {
        return listed;
    }
    items(field(field(report, "jobs"), "by_name"))
        .iter()
        .map(|item| display(field(item, "name")))
        .collect()
}

fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Float(float) => Some(*float),
        Value::Int(int) => int.to_string().parse().ok(),
        _ => None,
    }
}

/// `int(summary.get("count") or 0)`.
fn count(summary: &Value) -> i128 {
    match field(summary, "count") {
        Value::Int(int) => *int,
        Value::Float(float) if float.is_finite() => float.trunc() as i128,
        Value::Bool(flag) => i128::from(*flag),
        _ => 0,
    }
}

fn strings(values: impl IntoIterator<Item = String>) -> Value {
    Value::Array(values.into_iter().map(Value::Str).collect())
}

/// `{baseline, candidate, delta}` of one p95 series.
fn p95_delta(baseline: &Value, candidate: &Value) -> Value {
    let delta = match (number(baseline), number(candidate)) {
        (Some(base), Some(next)) => Some(round(next - base, 3)),
        _ => None,
    };
    object([
        ("baseline", baseline.clone()),
        ("candidate", candidate.clone()),
        ("delta", Value::opt_float(delta)),
    ])
}

struct Evidence {
    provider_separated: bool,
    sufficient: bool,
    common_names: bool,
    comparable_dimensions: bool,
    candidate_queue: Option<f64>,
}

fn recommendation(evidence: &Evidence) -> (&'static str, &'static str) {
    let queue = evidence.candidate_queue;
    if queue.is_some_and(|seconds| seconds >= QUEUE_CONTAMINATION_SECONDS) {
        return (
            "rollback",
            "candidate queue p95 crossed the deterministic contamination threshold",
        );
    }
    let comparable = evidence.sufficient
        && evidence.provider_separated
        && evidence.common_names
        && evidence.comparable_dimensions;
    if !comparable || queue.is_none_or(|seconds| seconds > QUEUE_WARN_SECONDS) {
        return (
            "hold",
            "collect more comparable evidence or resolve provider/capacity mismatch",
        );
    }
    (
        "eligible",
        "provider-separated comparable cohorts with candidate queue heuristics eligible",
    )
}

fn source_description(report: &Value) -> Value {
    field(field(report, "source"), "description").clone()
}

/// `compare_reports(baseline, candidate)`.
pub(crate) fn compare_reports(baseline: &Value, candidate: &Value) -> Value {
    let baseline_providers = providers(baseline);
    let candidate_providers = providers(candidate);
    let provider_separated = !baseline_providers.is_empty()
        && !candidate_providers.is_empty()
        && baseline_providers.is_disjoint(&candidate_providers);
    let common_names: Vec<String> = job_names(baseline)
        .intersection(&job_names(candidate))
        .cloned()
        .collect();
    let mut comparable_dimensions = true;
    let mut overlap = Vec::new();
    for key in DIMENSIONS {
        let (base, next) = (
            dimension_values(baseline, key),
            dimension_values(candidate, key),
        );
        let shared: Vec<String> = base.intersection(&next).cloned().collect();
        comparable_dimensions &= base.is_empty() || next.is_empty() || !shared.is_empty();
        overlap.push((key.to_owned(), strings(shared)));
    }
    let (base_cohort, next_cohort) = (cohort(baseline), cohort(candidate));
    let base_queue = field(base_cohort, "runner_queue_seconds");
    let next_queue = field(next_cohort, "runner_queue_seconds");
    let (baseline_count, candidate_count) = (count(base_queue), count(next_queue));
    let minimum = i128::try_from(MIN_HEURISTIC_SAMPLES).unwrap_or(i128::MAX);
    let sufficient = baseline_count >= minimum && candidate_count >= minimum;
    let (state, interpretation) = recommendation(&Evidence {
        provider_separated,
        sufficient,
        common_names: !common_names.is_empty(),
        comparable_dimensions,
        candidate_queue: number(field(next_queue, "p95")),
    });
    let execution = |cohort: &Value| field(field(cohort, "execution_seconds"), "p95").clone();
    object([
        ("baseline_source", source_description(baseline)),
        ("candidate_source", source_description(candidate)),
        (
            "provider_cohort_separation",
            object([
                ("baseline_providers", strings(baseline_providers)),
                ("candidate_providers", strings(candidate_providers)),
                ("disjoint", Value::Bool(provider_separated)),
                (
                    "status",
                    Value::text(if provider_separated { "pass" } else { "fail" }),
                ),
            ]),
        ),
        ("common_job_families", strings(common_names)),
        ("dimension_overlap", Value::Object(overlap)),
        ("comparable_dimensions", Value::Bool(comparable_dimensions)),
        (
            "sample_counts",
            object([
                ("baseline_jobs", Value::Int(baseline_count)),
                ("candidate_jobs", Value::Int(candidate_count)),
                ("minimum_each", Value::count(MIN_HEURISTIC_SAMPLES)),
                ("sufficient", Value::Bool(sufficient)),
            ]),
        ),
        (
            "p95_seconds",
            object([
                (
                    "queue",
                    p95_delta(field(base_queue, "p95"), field(next_queue, "p95")),
                ),
                (
                    "execution",
                    p95_delta(&execution(base_cohort), &execution(next_cohort)),
                ),
            ]),
        ),
        ("recommendation", Value::text(state)),
        ("interpretation", Value::text(interpretation)),
    ])
}
