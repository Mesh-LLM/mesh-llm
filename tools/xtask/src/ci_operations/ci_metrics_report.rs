//! `analyze` of `collect-ci-metrics.py`: assembles the schema-3 report
//! from the per-run pass and the aggregations, then adds `capacity` and
//! `heuristics`.

use crate::ci_operations::ci_metrics_aggregate::{by_name, by_runner, by_step, comparison_cohort};
use crate::ci_operations::ci_metrics_analyze::run_pass;
use crate::ci_operations::ci_metrics_normalize::{Outcome, Run};
use crate::ci_operations::ci_metrics_observe::{capacity_metrics, queue_heuristics};
use crate::ci_operations::ci_metrics_rollup::{
    critical, job_summaries, slowest, workflow_summaries,
};
use crate::ci_operations::ci_metrics_stats::Summary;
use crate::ci_operations::ci_metrics_time::{Instant, isoformat};
use crate::ci_operations::ci_metrics_value::{Value, object};

const DEFINITIONS: [(&str, &str); 12] = [
    (
        "workflow_wall_seconds",
        "first-attempt run created_at to updated_at; reruns excluded",
    ),
    (
        "workflow_queue_seconds",
        "first-attempt run created_at to started_at; reruns excluded",
    ),
    ("job_duration_seconds", "job started_at to completed_at"),
    (
        "job_execution_seconds",
        "same interval as job_duration_seconds",
    ),
    ("job_wall_seconds", "job created_at to completed_at"),
    (
        "job_queue_seconds",
        "job created_at to started_at; unavailable without job created_at",
    ),
    (
        "runner_queue_seconds",
        "job started_at minus dependency-ready timestamp when present; \
         otherwise falls back to creation-to-start for compatibility",
    ),
    (
        "dependency_wait_seconds",
        "dependency-ready timestamp to job created_at, or an explicit \
         instrumented duration; unavailable in standard jobs API data",
    ),
    (
        "job_start_delay_seconds",
        "first-attempt workflow created_at to job started_at; includes \
         dependency wait; reruns excluded",
    ),
    (
        "terminal_job",
        "last non-skipped job to finish; a critical-path candidate",
    ),
    (
        "runner_dimensions",
        "provider, operating system, architecture, semantic runner role, \
         and Depot runner size derived from labels or optional metadata",
    ),
    ("step_duration_seconds", "step started_at to completed_at"),
];

fn summaries(entries: &[(&'static str, Summary)]) -> Vec<(String, Value)> {
    entries
        .iter()
        .map(|(key, summary)| ((*key).to_owned(), summary.to_value()))
        .collect()
}

fn find<'s>(entries: &'s [(&'static str, Summary)], key: &str) -> Option<&'s Summary> {
    entries
        .iter()
        .find(|(name, _)| *name == key)
        .map(|(_, summary)| summary)
}

/// Inputs of `analyze` besides the runs.
pub(crate) struct Request<'r> {
    pub(crate) status: &'r str,
    pub(crate) top: usize,
    pub(crate) source: Value,
    pub(crate) labels: Vec<(String, Value)>,
}

pub(crate) fn analyze(runs: &[Run], request: Request<'_>) -> Outcome<Value> {
    let pass = run_pass(runs, request.status)?;
    let jobs = job_summaries(&pass);
    let workflow = workflow_summaries(&pass);
    let mut jobs_section = vec![(
        "sample_count".to_owned(),
        Value::count(pass.observations.len()),
    )];
    jobs_section.extend(summaries(&jobs));
    jobs_section.extend([
        ("by_name".to_owned(), by_name(&pass)),
        ("by_runner".to_owned(), by_runner(&pass)),
        ("comparison_cohort".to_owned(), comparison_cohort(&pass)),
        (
            "critical_finish_candidates".to_owned(),
            critical(&pass, request.top),
        ),
        (
            "slowest_observations".to_owned(),
            slowest(&pass, request.top),
        ),
    ]);
    let definitions = DEFINITIONS
        .iter()
        .map(|(key, text)| ((*key).to_owned(), Value::text(text)))
        .collect();
    let empty = Summary {
        count: 0,
        stats: None,
    };
    let job_queue = find(&jobs, "runner_queue_seconds").unwrap_or(&empty);
    let terminal_queue = find(&workflow, "terminal_job_runner_queue_seconds").unwrap_or(&empty);
    let heuristics = queue_heuristics(job_queue, terminal_queue);
    let generated_at = isoformat(Instant::now()).replace("+00:00", "Z");
    Ok(object([
        ("schema_version", Value::Int(3)),
        ("generated_at", Value::Str(generated_at)),
        ("source", request.source),
        ("benchmark_labels", Value::Object(request.labels)),
        ("definitions", Value::Object(definitions)),
        (
            "selection",
            object([
                ("requested_status", Value::text(request.status)),
                ("seen_run_count", Value::count(runs.len())),
                ("included_run_count", Value::count(pass.selected)),
                (
                    "workflow_timing_excluded_reruns",
                    Value::count(pass.excluded_reruns),
                ),
                ("skipped_runs", pass.skipped.sorted_value()),
            ]),
        ),
        ("workflow", Value::Object(summaries(&workflow))),
        ("jobs", Value::Object(jobs_section)),
        (
            "steps",
            object([
                ("sample_count", Value::count(pass.steps.len())),
                ("by_name", by_step(&pass)),
            ]),
        ),
        ("runs", Value::Array(pass.runs.clone())),
        ("capacity", capacity_metrics(&pass.observations)),
        ("heuristics", heuristics),
    ]))
}
