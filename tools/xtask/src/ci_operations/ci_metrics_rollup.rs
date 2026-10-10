//! Report rollups of `analyze` in `collect-ci-metrics.py`: slowest
//! observations, critical-finish candidates, and the `jobs` and `workflow`
//! summaries.

use crate::ci_operations::ci_metrics_aggregate::{
    DEPENDENCY_WAIT, DURATION, Field, QUEUE, RUNNER_QUEUE, START_DELAY, WALL, summary,
};
use crate::ci_operations::ci_metrics_analyze::Pass;
use crate::ci_operations::ci_metrics_observe::Observation;
use crate::ci_operations::ci_metrics_stats::{Summary, round, sort_descending, summarize};
use crate::ci_operations::ci_metrics_value::{Value, object};

/// `slowest_observations`: the `top` longest jobs, ties in input order.
pub(crate) fn slowest(pass: &Pass<'_>, top: usize) -> Value {
    let mut timed: Vec<&Observation<'_>> = pass
        .observations
        .iter()
        .filter(|sample| sample.duration.is_some())
        .collect();
    sort_descending(&mut timed, |sample| (sample.duration.unwrap_or(0.0), 0.0));
    timed.truncate(top);
    Value::Array(timed.into_iter().map(Observation::to_value).collect())
}

pub(crate) fn critical(pass: &Pass<'_>, top: usize) -> Value {
    let entries = pass
        .terminal_counts
        .most_common(top)
        .into_iter()
        .map(|(name, count)| {
            let share = count as f64 / pass.selected as f64;
            object([
                ("name", Value::text(&name)),
                ("terminal_count", Value::count(count)),
                ("share", Value::Float(round(share, 4))),
            ])
        });
    Value::Array(entries.collect())
}

/// The `jobs` rollup fields shared by every observation.
pub(crate) fn job_summaries(pass: &Pass<'_>) -> Vec<(&'static str, Summary)> {
    let samples: Vec<&Observation<'_>> = pass.observations.iter().collect();
    vec![
        ("duration_seconds", summary(&samples, DURATION)),
        ("execution_seconds", summary(&samples, DURATION)),
        ("wall_seconds", summary(&samples, WALL)),
        ("queue_seconds", summary(&samples, QUEUE)),
        ("runner_queue_seconds", summary(&samples, RUNNER_QUEUE)),
        (
            "dependency_wait_seconds",
            summary(&samples, DEPENDENCY_WAIT),
        ),
        ("start_delay_seconds", summary(&samples, START_DELAY)),
    ]
}

/// The `workflow` rollup, in report order.
pub(crate) fn workflow_summaries(pass: &Pass<'_>) -> Vec<(&'static str, Summary)> {
    let over = |field: Field| summarize(pass.observations.iter().map(field));
    vec![
        ("wall_seconds", summarize(pass.wall.iter().copied())),
        ("queue_seconds", summarize(pass.queue.iter().copied())),
        ("dependency_wait_seconds", over(DEPENDENCY_WAIT)),
        ("execution_seconds", over(DURATION)),
        ("job_wall_seconds", over(WALL)),
        (
            "terminal_job_queue_seconds",
            summarize(pass.terminal_queue.iter().copied()),
        ),
        (
            "terminal_job_runner_queue_seconds",
            summarize(pass.terminal_runner_queue.iter().copied()),
        ),
        (
            "terminal_job_dependency_wait_seconds",
            summarize(pass.terminal_dependency_wait.iter().copied()),
        ),
        (
            "terminal_job_execution_seconds",
            summarize(pass.terminal_execution.iter().copied()),
        ),
    ]
}
