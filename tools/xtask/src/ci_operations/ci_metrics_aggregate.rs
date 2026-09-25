//! The aggregation half of `analyze` in `collect-ci-metrics.py`: per-name,
//! per-runner and per-step summaries, the comparison cohort, slowest jobs,
//! critical-finish candidates and the workflow/job rollups.

use crate::ci_operations::ci_metrics_analyze::Pass;
use crate::ci_operations::ci_metrics_observe::Observation;
use crate::ci_operations::ci_metrics_runner::Dimensions;
use crate::ci_operations::ci_metrics_runner::is_comparison_executor;
use crate::ci_operations::ci_metrics_stats::{Counter, Summary, sort_descending, summarize};
use crate::ci_operations::ci_metrics_value::{Value, object};
use std::collections::BTreeSet;

pub(crate) type Field = fn(&Observation<'_>) -> Option<f64>;

pub(crate) const DURATION: Field = |sample| sample.duration;
pub(crate) const WALL: Field = |sample| sample.wall;
pub(crate) const QUEUE: Field = |sample| sample.queue;
pub(crate) const RUNNER_QUEUE: Field = |sample| sample.runner_queue;
pub(crate) const DEPENDENCY_WAIT: Field = |sample| sample.dependency_wait;
pub(crate) const START_DELAY: Field = |sample| sample.start_delay;

pub(crate) fn summary(samples: &[&Observation<'_>], field: Field) -> Summary {
    summarize(samples.iter().map(|sample| field(sample)))
}

/// Groups in first-seen order, like a `defaultdict(list)`.
fn group_by<'s, 'a, K: PartialEq>(
    samples: impl IntoIterator<Item = &'s Observation<'a>>,
    key: impl Fn(&Observation<'a>) -> K,
) -> Vec<(K, Vec<&'s Observation<'a>>)>
where
    'a: 's,
{
    let mut groups: Vec<(K, Vec<&'s Observation<'a>>)> = Vec::new();
    for sample in samples {
        let group_key = key(sample);
        match groups.iter_mut().find(|(seen, _)| *seen == group_key) {
            Some(group) => group.1.push(sample),
            None => groups.push((group_key, vec![sample])),
        }
    }
    groups
}

fn names(samples: &[&Observation<'_>]) -> Value {
    let names: BTreeSet<&str> = samples
        .iter()
        .map(|sample| sample.job.name.as_str())
        .collect();
    Value::Array(names.into_iter().map(Value::text).collect())
}

fn conclusions<'c>(values: impl IntoIterator<Item = &'c str>) -> Value {
    let mut counter = Counter::default();
    for value in values {
        counter.add(value);
    }
    counter.sorted_value()
}

fn with_sort_key(value: Value, duration: &Summary) -> ((f64, f64), Value) {
    (duration.sort_key(), value)
}

fn sorted_entries(mut entries: Vec<((f64, f64), Value)>) -> Value {
    sort_descending(&mut entries, |(key, _)| *key);
    Value::Array(entries.into_iter().map(|(_, value)| value).collect())
}

pub(crate) fn by_name(pass: &Pass<'_>) -> Value {
    let groups = group_by(&pass.observations, |sample| sample.job.name.clone());
    let entries = groups
        .into_iter()
        .map(|(name, samples)| {
            let duration = summary(&samples, DURATION);
            let value = object([
                ("name", Value::text(&name)),
                ("sample_count", Value::count(samples.len())),
                ("duration_seconds", duration.to_value()),
                ("execution_seconds", duration.to_value()),
                ("wall_seconds", summary(&samples, WALL).to_value()),
                ("queue_seconds", summary(&samples, QUEUE).to_value()),
                (
                    "runner_queue_seconds",
                    summary(&samples, RUNNER_QUEUE).to_value(),
                ),
                (
                    "dependency_wait_seconds",
                    summary(&samples, DEPENDENCY_WAIT).to_value(),
                ),
                (
                    "start_delay_seconds",
                    summary(&samples, START_DELAY).to_value(),
                ),
                (
                    "terminal_count",
                    Value::count(pass.terminal_counts.get(&name)),
                ),
                (
                    "conclusions",
                    conclusions(samples.iter().map(|s| s.job.conclusion.as_str())),
                ),
            ]);
            with_sort_key(value, &duration)
        })
        .collect();
    sorted_entries(entries)
}

pub(crate) fn by_runner(pass: &Pass<'_>) -> Value {
    let groups = group_by(&pass.observations, |sample| sample.dimensions.clone());
    let entries = groups
        .into_iter()
        .map(|(dimensions, samples)| {
            let duration = summary(&samples, DURATION);
            let value = object([
                ("provider", Value::text(&dimensions.provider)),
                (
                    "operating_system",
                    Value::opt_text(dimensions.operating_system.as_deref()),
                ),
                (
                    "architecture",
                    Value::opt_text(dimensions.architecture.as_deref()),
                ),
                (
                    "runner_role",
                    Value::opt_text(dimensions.runner_role.as_deref()),
                ),
                (
                    "runner_size",
                    Value::opt_text(dimensions.runner_size.as_deref()),
                ),
                ("sample_count", Value::count(samples.len())),
                ("job_names", names(&samples)),
                ("duration_seconds", duration.to_value()),
                ("execution_seconds", duration.to_value()),
                ("queue_seconds", summary(&samples, QUEUE).to_value()),
                (
                    "runner_queue_seconds",
                    summary(&samples, RUNNER_QUEUE).to_value(),
                ),
                (
                    "dependency_wait_seconds",
                    summary(&samples, DEPENDENCY_WAIT).to_value(),
                ),
            ]);
            with_sort_key(value, &duration)
        })
        .collect();
    sorted_entries(entries)
}

pub(crate) fn comparison_cohort(pass: &Pass<'_>) -> Value {
    let samples: Vec<&Observation<'_>> = pass
        .observations
        .iter()
        .filter(|sample| is_comparison_executor(&sample.job.name))
        .collect();
    let distinct = |key: &str| {
        let values: BTreeSet<&str> = samples
            .iter()
            .filter_map(|sample| sample.dimensions.field(key))
            .collect();
        Value::Array(values.into_iter().map(Value::text).collect())
    };
    let providers: BTreeSet<&str> = samples
        .iter()
        .map(|sample| sample.dimensions.provider.as_str())
        .filter(|provider| *provider != "unknown")
        .collect();
    object([
        (
            "scope",
            Value::text("build-test executors; orchestration and credentialed smoke jobs excluded"),
        ),
        ("sample_count", Value::count(samples.len())),
        ("job_names", names(&samples)),
        (
            "providers",
            Value::Array(providers.into_iter().map(Value::text).collect()),
        ),
        (
            "dimensions",
            object([
                ("operating_system", distinct("operating_system")),
                ("architecture", distinct("architecture")),
                ("runner_role", distinct("runner_role")),
            ]),
        ),
        (
            "runner_queue_seconds",
            summary(&samples, RUNNER_QUEUE).to_value(),
        ),
        ("execution_seconds", summary(&samples, DURATION).to_value()),
    ])
}

/// `(job name, step name, runner dimensions)`.
type StepKey<'p> = (&'p str, &'p str, &'p Dimensions);

pub(crate) fn by_step(pass: &Pass<'_>) -> Value {
    let mut groups: Vec<(StepKey<'_>, Vec<usize>)> = Vec::new();
    for (index, step) in pass.steps.iter().enumerate() {
        let dimensions = &pass.observations[step.observation].dimensions;
        let key = (step.job.name.as_str(), step.name, dimensions);
        match groups.iter_mut().find(|(seen, _)| *seen == key) {
            Some(group) => group.1.push(index),
            None => groups.push((key, vec![index])),
        }
    }
    let entries = groups
        .into_iter()
        .map(|((job_name, step_name, dimensions), members)| {
            let steps: Vec<_> = members.iter().map(|index| &pass.steps[*index]).collect();
            let duration = summarize(steps.iter().map(|step| step.duration));
            let value = object([
                ("job_name", Value::text(job_name)),
                ("step_name", Value::text(step_name)),
                ("provider", Value::text(&dimensions.provider)),
                (
                    "operating_system",
                    Value::opt_text(dimensions.operating_system.as_deref()),
                ),
                (
                    "architecture",
                    Value::opt_text(dimensions.architecture.as_deref()),
                ),
                (
                    "runner_role",
                    Value::opt_text(dimensions.runner_role.as_deref()),
                ),
                (
                    "runner_size",
                    Value::opt_text(dimensions.runner_size.as_deref()),
                ),
                ("sample_count", Value::count(steps.len())),
                ("duration_seconds", duration.to_value()),
                (
                    "conclusions",
                    conclusions(steps.iter().map(|step| step.conclusion)),
                ),
            ]);
            with_sort_key(value, &duration)
        })
        .collect();
    sorted_entries(entries)
}
