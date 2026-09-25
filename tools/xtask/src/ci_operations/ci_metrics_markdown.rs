//! `render_markdown(report, top)` of `collect-ci-metrics.py`: the Markdown
//! timing summary written to `--markdown-out` or, without any output path,
//! to stdout.

use crate::ci_operations::ci_metrics_markdown_format::{
    escape, escape_or_na, field, head, human, or_na, truthy,
};
use crate::ci_operations::ci_metrics_markdown_jobs;
use crate::ci_operations::ci_metrics_normalize::Outcome;
use crate::ci_operations::ci_metrics_value::{Value, display};

const WORKFLOW_TIMINGS: [(&str, &str); 4] = [
    ("Wall time", "wall_seconds"),
    ("Queue", "queue_seconds"),
    ("Dependency wait", "dependency_wait_seconds"),
    ("Execution", "execution_seconds"),
];

const FOOTNOTE: &str = "_Job queue uses GitHub's job creation-to-start interval. Offline \
`gh run view` data may omit creation times; those samples are n/a. \
Workflow and start-delay timing excludes rerun attempts because \
their run timestamps belong to the original attempt. Terminal jobs \
are candidates only; the DAG is not reconstructed._";

/// `human(mapping[key][stat])`.
pub(crate) fn timing(item: &Value, key: &str, stat: &str) -> Outcome<String> {
    human(field(field(item, key), stat))
}

pub(crate) fn render_markdown(report: &Value, top: usize) -> Outcome<String> {
    let mut lines: Vec<String> = Vec::new();
    header(&mut lines, report);
    workflow_timing(&mut lines, field(report, "workflow"))?;
    runner_dimensions(&mut lines, field(report, "jobs"), top)?;
    capacity(&mut lines, report)?;
    step_timing(&mut lines, field(report, "steps"), top)?;
    ci_metrics_markdown_jobs::job_families(&mut lines, field(report, "jobs"), top)?;
    ci_metrics_markdown_jobs::terminal_jobs(&mut lines, field(report, "jobs"), top)?;
    ci_metrics_markdown_jobs::slowest(&mut lines, field(report, "jobs"), top)?;
    comparison(&mut lines, report);
    lines.push(FOOTNOTE.to_owned());
    lines.push(String::new());
    Ok(lines.join("\n"))
}

fn header(lines: &mut Vec<String>, report: &Value) {
    let selection = field(report, "selection");
    lines.push("# CI timing summary".to_owned());
    lines.push(String::new());
    lines.push(format!(
        "Analyzed **{}** completed run(s) from `{}`.",
        display(field(selection, "included_run_count")),
        escape(field(field(report, "source"), "description")),
    ));
    lines.push(String::new());
    let excluded = field(selection, "workflow_timing_excluded_reruns");
    if truthy(excluded) {
        lines.push(format!(
            "> Excluded workflow wall, workflow queue, and job start-delay timing from \
             **{}** rerun attempt(s). GitHub retains the original run timestamps when \
             jobs are rerun.",
            display(excluded)
        ));
        lines.push(String::new());
    }
}

fn workflow_timing(lines: &mut Vec<String>, workflow: &Value) -> Outcome<()> {
    lines.extend(
        [
            "## Workflow timing",
            "",
            "| Timing | Samples | p50 | p90 | p95 | Max |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        .map(str::to_owned),
    );
    for (label, key) in WORKFLOW_TIMINGS {
        let stats = field(workflow, key);
        lines.push(format!(
            "| {label} | {} | {} | {} | {} | {} |",
            display(field(stats, "count")),
            human(field(stats, "p50"))?,
            human(field(stats, "p90"))?,
            human(field(stats, "p95"))?,
            human(field(stats, "max"))?,
        ));
    }
    Ok(())
}

fn runner_dimensions(lines: &mut Vec<String>, jobs: &Value, top: usize) -> Outcome<()> {
    lines.extend(
        [
            "",
            "## Runner dimensions",
            "",
            "| Provider | OS | Architecture | Role | Depot size | Jobs | Execution p50 | \
             Execution p95 | Runner queue p95 |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        .map(str::to_owned),
    );
    for item in head(jobs, "by_runner", top) {
        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            escape(field(item, "provider")),
            escape_or_na(field(item, "operating_system")),
            escape_or_na(field(item, "architecture")),
            escape_or_na(field(item, "runner_role")),
            escape_or_na(field(item, "runner_size")),
            display(field(item, "sample_count")),
            timing(item, "execution_seconds", "p50")?,
            timing(item, "execution_seconds", "p95")?,
            timing(item, "runner_queue_seconds", "p95")?,
        ));
    }
    Ok(())
}

fn capacity(lines: &mut Vec<String>, report: &Value) -> Outcome<()> {
    let capacity = field(report, "capacity");
    let heuristics = field(report, "heuristics");
    lines.extend(["", "## Capacity and rollout heuristics", ""].map(str::to_owned));
    lines.push(format!(
        "Runner-minutes: **{}**; cancelled runner-minutes: **{}**; peak workers: **{}**.",
        display(field(capacity, "runner_minutes")),
        display(field(capacity, "cancelled_runner_minutes")),
        or_na(field(field(capacity, "peak_workers"), "total")),
    ));
    lines.push(format!(
        "Queue p95: **{}**; terminal queue p95: **{}**; capacity contaminated: **{}**; \
         state: **{}**.",
        human(field(heuristics, "queue_p95_seconds"))?,
        human(field(heuristics, "terminal_job_queue_p95_seconds"))?,
        display(field(heuristics, "capacity_contaminated")),
        display(field(heuristics, "state")),
    ));
    lines.extend(["", "## Step timing", ""].map(str::to_owned));
    Ok(())
}

fn step_timing(lines: &mut Vec<String>, steps: &Value, top: usize) -> Outcome<()> {
    if !truthy(field(steps, "by_name")) {
        lines.push(
            "No job step timestamps were available; cache, context-upload, and \
             export/import phases are not inferred from logs."
                .to_owned(),
        );
        return Ok(());
    }
    lines.extend(
        [
            "| Provider | OS | Architecture | Role | Runner size | Job | Step | Samples | \
             Duration p50 | Duration p95 |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
        .map(str::to_owned),
    );
    for item in head(steps, "by_name", top) {
        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            escape_or_na(field(item, "provider")),
            escape_or_na(field(item, "operating_system")),
            escape_or_na(field(item, "architecture")),
            escape_or_na(field(item, "runner_role")),
            escape_or_na(field(item, "runner_size")),
            escape(field(item, "job_name")),
            escape(field(item, "step_name")),
            display(field(item, "sample_count")),
            timing(item, "duration_seconds", "p50")?,
            timing(item, "duration_seconds", "p95")?,
        ));
    }
    Ok(())
}

/// `', '.join(providers) or 'n/a'`.
fn providers(separation: &Value, key: &str) -> String {
    let joined = match field(separation, key) {
        Value::Array(items) => items.iter().map(display).collect::<Vec<_>>().join(", "),
        other => display(other),
    };
    if joined.is_empty() {
        "n/a".to_owned()
    } else {
        joined
    }
}

fn comparison(lines: &mut Vec<String>, report: &Value) {
    let comparison = field(report, "comparison");
    if !truthy(comparison) {
        return;
    }
    let separation = field(comparison, "provider_cohort_separation");
    lines.extend(["## Provider cohort comparison", ""].map(str::to_owned));
    lines.push(format!(
        "Provider separation: **{}** ({} \u{2192} {}); recommendation: **{}**.",
        display(field(separation, "status")),
        providers(separation, "baseline_providers"),
        providers(separation, "candidate_providers"),
        display(field(comparison, "recommendation")),
    ));
    lines.push(String::new());
}
