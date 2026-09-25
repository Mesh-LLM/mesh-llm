//! The per-run pass of `analyze` in `collect-ci-metrics.py`: run selection,
//! observations of non-skipped jobs, terminal and longest jobs, first-attempt
//! workflow timing, and the `runs` entries of the report.

use crate::ci_operations::ci_metrics_normalize::{Failure, Job, Outcome, Run};
use crate::ci_operations::ci_metrics_observe::{Observation, contaminated, included, observation};
use crate::ci_operations::ci_metrics_stats::Counter;
use crate::ci_operations::ci_metrics_time::elapsed;
use crate::ci_operations::ci_metrics_value::{Value, equal};

const SKIPPED: &str = "skipped";

pub(crate) struct StepSample<'a> {
    pub(crate) job: &'a Job,
    pub(crate) name: &'a str,
    pub(crate) conclusion: &'a str,
    pub(crate) duration: Option<f64>,
    /// Index of the owning job's observation, for its runner dimensions.
    pub(crate) observation: usize,
}

/// Everything the aggregation pass needs from the per-run pass.
#[derive(Default)]
pub(crate) struct Pass<'a> {
    pub(crate) selected: usize,
    pub(crate) skipped: Counter,
    pub(crate) observations: Vec<Observation<'a>>,
    pub(crate) steps: Vec<StepSample<'a>>,
    pub(crate) terminal_counts: Counter,
    pub(crate) runs: Vec<Value>,
    pub(crate) wall: Vec<Option<f64>>,
    pub(crate) queue: Vec<Option<f64>>,
    pub(crate) terminal_queue: Vec<Option<f64>>,
    pub(crate) terminal_runner_queue: Vec<Option<f64>>,
    pub(crate) terminal_dependency_wait: Vec<Option<f64>>,
    pub(crate) terminal_execution: Vec<Option<f64>>,
    pub(crate) excluded_reruns: usize,
}

pub(crate) fn run_pass<'a>(runs: &'a [Run], requested: &str) -> Outcome<Pass<'a>> {
    let mut pass = Pass::default();
    let selected: Vec<&Run> = runs
        .iter()
        .filter(|run| match included(run, requested) {
            Ok(()) => true,
            Err(reason) => {
                pass.skipped.add(&reason);
                false
            }
        })
        .collect();
    if selected.is_empty() {
        return Err(Failure::Reported(
            "no completed workflow runs matched the requested status".to_owned(),
        ));
    }
    pass.selected = selected.len();
    for run in selected {
        add_run(&mut pass, run)?;
    }
    Ok(pass)
}

/// Python's `hash()` rejects list and dict job ids in `sample_by_job_id`.
fn check_hashable(id: &Value) -> Outcome<()> {
    let kind = match id {
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
        _ => return Ok(()),
    };
    Err(Failure::Uncaught(format!(
        "TypeError: unhashable type: '{kind}'"
    )))
}

fn add_run<'a>(pass: &mut Pass<'a>, run: &'a Run) -> Outcome<()> {
    let first = pass.observations.len();
    let jobs: Vec<&Job> = run
        .jobs
        .iter()
        .filter(|job| job.conclusion != SKIPPED)
        .collect();
    for job in &jobs {
        let index = pass.observations.len();
        pass.observations.push(observation(run, job));
        pass.steps.extend(job.steps.iter().map(|step| StepSample {
            job,
            name: &step.name,
            conclusion: &step.conclusion,
            duration: step.duration,
            observation: index,
        }));
    }
    let samples = &pass.observations[first..];
    for sample in samples {
        check_hashable(&sample.job.id)?;
    }
    let mut terminal: Option<&Job> = None;
    for job in jobs.iter().filter(|job| job.completed.is_some()) {
        if terminal.is_none_or(|best| job.completed > best.completed) {
            terminal = Some(job);
        }
    }
    let terminal_sample = terminal.and_then(|terminal| {
        samples
            .iter()
            .rev()
            .find(|sample| equal(&sample.job.id, &terminal.id))
    });
    if let (Some(terminal), Some(sample)) = (terminal, terminal_sample) {
        pass.terminal_counts.add(&terminal.name);
        pass.terminal_queue.push(sample.queue);
        pass.terminal_runner_queue.push(sample.runner_queue);
        pass.terminal_dependency_wait.push(sample.dependency_wait);
        pass.terminal_execution.push(sample.duration);
    }
    let mut longest: Option<&Observation<'_>> = None;
    for sample in samples {
        if let Some(duration) = sample.duration
            && longest.is_none_or(|best| best.duration.is_some_and(|best| duration > best))
        {
            longest = Some(sample);
        }
    }
    let eligible = run.first_attempt();
    let (wall, queue) = if eligible {
        (
            elapsed(run.created, run.updated),
            elapsed(run.created, run.started),
        )
    } else {
        (None, None)
    };
    if eligible {
        pass.wall.push(wall);
        pass.queue.push(queue);
    } else {
        pass.excluded_reruns += 1;
    }
    let terminal_field = |field: fn(&Observation<'_>) -> Option<f64>| {
        Value::opt_float(terminal_sample.and_then(field))
    };
    let mut entry = vec![
        ("id", run.id.clone()),
        ("attempt", run.attempt.clone()),
        ("url", Value::text(&run.url)),
        ("workflow", Value::text(&run.workflow)),
        ("title", Value::text(&run.title)),
        ("event", Value::text(&run.event)),
        ("conclusion", Value::text(&run.conclusion)),
        ("head_sha", Value::text(&run.sha)),
        ("head_branch", Value::text(&run.branch)),
    ];
    entry.extend(
        run.metadata
            .iter()
            .map(|(key, value)| (*key, value.clone())),
    );
    entry.extend([
        ("wall_seconds", Value::opt_float(wall)),
        ("queue_seconds", Value::opt_float(queue)),
        ("workflow_timing_excluded", Value::Bool(!eligible)),
        ("executed_job_count", Value::count(jobs.len())),
        (
            "skipped_job_count",
            Value::count(run.jobs.len() - jobs.len()),
        ),
        (
            "terminal_job",
            Value::opt_text(terminal.map(|job| job.name.as_str())),
        ),
        (
            "longest_job",
            Value::opt_text(longest.map(|sample| sample.job.name.as_str())),
        ),
        (
            "longest_job_seconds",
            Value::opt_float(longest.and_then(|sample| sample.duration)),
        ),
        ("terminal_job_queue_seconds", terminal_field(|s| s.queue)),
        (
            "terminal_job_runner_queue_seconds",
            terminal_field(|s| s.runner_queue),
        ),
        (
            "terminal_job_dependency_wait_seconds",
            terminal_field(|s| s.dependency_wait),
        ),
        (
            "terminal_job_execution_seconds",
            terminal_field(|s| s.duration),
        ),
        (
            "capacity_contaminated",
            Value::Bool(
                samples
                    .iter()
                    .any(|sample| contaminated(sample.runner_queue)),
            ),
        ),
    ]);
    pass.runs.push(Value::Object(
        entry
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    ));
    Ok(())
}
