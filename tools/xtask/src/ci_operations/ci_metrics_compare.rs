//! `compare_reports` of `collect-ci-metrics.py`: compares the baseline
//! report built from `--compare-input` with the candidate report without
//! conflating provider or capacity effects. Both reports are the port's own
//! `analyze` output, read back through the legacy fallbacks
//! (`jobs.comparison_cohort` falling back to `jobs`, `providers` and
//! `dimensions` falling back to `jobs.by_runner`, `job_names` to
//! `jobs.by_name`).

use crate::ci_operations::ci_metrics_markdown_format::field;
use crate::ci_operations::ci_metrics_observe::{
    MIN_HEURISTIC_SAMPLES, QUEUE_CONTAMINATION_SECONDS, QUEUE_WARN_SECONDS,
};
use crate::ci_operations::ci_metrics_stats::round_decimal;
use crate::ci_operations::ci_metrics_value::{Value, display, object};
use std::collections::BTreeSet;

const DIMENSIONS: [&str; 3] = ["operating_system", "architecture", "runner_role"];
const RUN_IDENTITY: [&str; 9] = [
    "workflow",
    "event",
    "head_sha",
    "plan_profile",
    "plan_digest",
    "change_class",
    "runner_image",
    "toolchain_epoch",
    "cache_mode",
];
const MIN_COMPARABLE_RUNS: usize = 3;

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
    matches!(value, Value::Str(text) if !text.trim().is_empty()
        && text == text.trim() && text != "unknown" && text != "n/a")
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
        .filter(|value| known_provider(value))
        .map(display)
        .collect()
}

/// `set(cohort.get("job_names", [])) or {item["name"] for item in by_name}`.
fn job_names(report: &Value) -> BTreeSet<String> {
    let listed: BTreeSet<String> = cohort(report)
        .get("job_names")
        .map(|names| items(names).iter().filter_map(text).collect())
        .unwrap_or_default();
    if !listed.is_empty() {
        return listed;
    }
    items(field(field(report, "jobs"), "by_name"))
        .iter()
        .filter_map(|item| text(field(item, "name")))
        .collect()
}

/// Aggregated dimension sets omit missing values, so inspect every runner
/// group containing an included job family before considering them complete.
fn runner_shapes(
    report: &Value,
    names: &BTreeSet<String>,
) -> Option<BTreeSet<(String, String, String, String)>> {
    let groups = by_runner(report);
    let mut covered = BTreeSet::new();
    let mut shapes = BTreeSet::new();
    for group in groups {
        let group_names: BTreeSet<String> = items(field(group, "job_names"))
            .iter()
            .filter_map(text)
            .collect();
        let selected: Vec<&String> = group_names.intersection(names).collect();
        if !selected.is_empty() {
            if !known_provider(field(group, "provider"))
                || DIMENSIONS
                    .iter()
                    .any(|key| !known_provider(field(group, key)))
            {
                return None;
            }
            for name in selected {
                covered.insert(name.clone());
                shapes.insert((
                    name.clone(),
                    display(field(group, "operating_system")),
                    display(field(group, "architecture")),
                    display(field(group, "runner_role")),
                ));
            }
        }
    }
    (covered == *names && !shapes.is_empty()).then_some(shapes)
}

fn number(value: &Value) -> Option<f64> {
    let number = match value {
        Value::Float(float) => Some(*float),
        Value::Int(int) => int.to_string().parse().ok(),
        _ => None,
    };
    number.filter(|number| number.is_finite())
}

fn text(value: &Value) -> Option<String> {
    match value {
        Value::Str(text) if !text.is_empty() => Some(text.clone()),
        _ => None,
    }
}

/// `int(summary.get("count") or 0)`.
fn count(summary: &Value) -> i128 {
    match field(summary, "count") {
        Value::Int(int) if *int >= 0 => *int,
        _ => 0,
    }
}

fn strings(values: impl IntoIterator<Item = String>) -> Value {
    Value::Array(values.into_iter().map(Value::Str).collect())
}

/// `{baseline, candidate, delta}` of one p95 series.
fn p95_delta(baseline: &Value, candidate: &Value) -> Value {
    let delta = match (number(baseline), number(candidate)) {
        (Some(base), Some(next)) => Some(round_decimal(next - base, 3)),
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
    comparable_runs: bool,
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
        && evidence.comparable_dimensions
        && evidence.comparable_runs;
    if !comparable || queue.is_none_or(|seconds| seconds > QUEUE_WARN_SECONDS) {
        return (
            "hold",
            "collect more comparable evidence or resolve provider/capacity mismatch",
        );
    }
    (
        "review",
        "matched recorded cohorts require independent review; this signal does not qualify a rollout",
    )
}

/// A conservative prerequisite for review, not proof of rollout readiness.
/// Reruns, duplicate IDs, excluded records and incomplete metadata must not
/// inflate the number of independent successful observations.
fn run_cohort(report: &Value) -> Option<(BTreeSet<i128>, Vec<String>)> {
    let selection = field(report, "selection");
    if !matches!(field(selection, "skipped_runs"), Value::Object(entries) if entries.is_empty()) {
        return None;
    }
    let runs = items(field(report, "runs"));
    if runs.len() < MIN_COMPARABLE_RUNS {
        return None;
    }
    let mut ids = BTreeSet::new();
    let mut identity: Option<Vec<String>> = None;
    for run in runs {
        let id = match field(run, "id") {
            Value::Int(id) if *id > 0 => *id,
            _ => return None,
        };
        if !ids.insert(id)
            || !matches!(field(run, "attempt"), Value::Int(1))
            || !matches!(field(run, "conclusion"), Value::Str(value) if value == "success")
        {
            return None;
        }
        let fields: Option<Vec<String>> = RUN_IDENTITY
            .iter()
            .map(|key| match field(run, key) {
                Value::Str(value)
                    if !value.trim().is_empty()
                        && value == value.trim()
                        && value != "unknown"
                        && value != "n/a" =>
                {
                    Some(value.clone())
                }
                _ => None,
            })
            .collect();
        let fields = fields?;
        let sha = &fields[2];
        if sha.len() != 40 || !sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return None;
        }
        if identity
            .as_ref()
            .is_some_and(|previous| previous != &fields)
        {
            return None;
        }
        identity = Some(fields);
    }
    Some((ids, identity?))
}

fn comparable_runs(baseline: &Value, candidate: &Value) -> bool {
    match (run_cohort(baseline), run_cohort(candidate)) {
        (Some((base_ids, base_identity)), Some((next_ids, next_identity))) => {
            base_ids.is_disjoint(&next_ids) && base_identity == next_identity
        }
        _ => false,
    }
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
    let (base_names, next_names) = (job_names(baseline), job_names(candidate));
    let common_names: Vec<String> = base_names.intersection(&next_names).cloned().collect();
    let (base_shapes, next_shapes) = (
        runner_shapes(baseline, &base_names),
        runner_shapes(candidate, &next_names),
    );
    let mut comparable_dimensions = base_shapes.is_some() && base_shapes == next_shapes;
    let mut overlap = Vec::new();
    for key in DIMENSIONS {
        let (base, next) = (
            dimension_values(baseline, key),
            dimension_values(candidate, key),
        );
        let shared: Vec<String> = base.intersection(&next).cloned().collect();
        comparable_dimensions &= !base.is_empty() && base == next;
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
        common_names: !base_names.is_empty() && base_names == next_names,
        comparable_dimensions,
        comparable_runs: comparable_runs(baseline, candidate),
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
            "comparable_runs",
            Value::Bool(comparable_runs(baseline, candidate)),
        ),
        (
            "matching_job_families",
            Value::Bool(!base_names.is_empty() && base_names == next_names),
        ),
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

#[cfg(test)]
#[path = "ci_metrics_compare_tests.rs"]
mod tests;
