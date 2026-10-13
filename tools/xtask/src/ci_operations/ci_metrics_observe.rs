//! `observation`, `included`, `capacity_metrics` and `queue_heuristics` of
//! `collect-ci-metrics.py`.

use crate::ci_operations::ci_metrics_normalize::{Job, Run};
use crate::ci_operations::ci_metrics_runner::{Dimensions, runner_dimensions};
use crate::ci_operations::ci_metrics_stats::{Summary, float_sum, round};
use crate::ci_operations::ci_metrics_time::{elapsed, isoformat};
use crate::ci_operations::ci_metrics_value::{Value, object};

pub(crate) const QUEUE_WARN_SECONDS: f64 = 60.0;
pub(crate) const QUEUE_CONTAMINATION_SECONDS: f64 = 300.0;
pub(crate) const MIN_HEURISTIC_SAMPLES: usize = 3;

/// One executed job with its derived intervals.
pub(crate) struct Observation<'a> {
    pub(crate) run: &'a Run,
    pub(crate) job: &'a Job,
    pub(crate) duration: Option<f64>,
    pub(crate) wall: Option<f64>,
    pub(crate) queue: Option<f64>,
    pub(crate) runner_queue: Option<f64>,
    pub(crate) dependency_wait: Option<f64>,
    pub(crate) start_delay: Option<f64>,
    pub(crate) dimensions: Dimensions,
}

pub(crate) fn contaminated(queue: Option<f64>) -> bool {
    queue.is_some_and(|seconds| seconds >= QUEUE_CONTAMINATION_SECONDS)
}

pub(crate) fn observation<'a>(run: &'a Run, job: &'a Job) -> Observation<'a> {
    let dependency_wait = job
        .dependency_wait
        .or_else(|| elapsed(job.created, job.dependency_ready));
    Observation {
        run,
        job,
        duration: elapsed(job.started, job.completed),
        wall: elapsed(job.created, job.completed),
        queue: elapsed(job.created, job.started),
        runner_queue: elapsed(job.dependency_ready.or(job.created), job.started),
        dependency_wait,
        start_delay: if run.first_attempt() {
            elapsed(run.created, job.started)
        } else {
            None
        },
        dimensions: runner_dimensions(&job.labels, &job.runner_role, &job.operating_system),
    }
}

impl Observation<'_> {
    pub(crate) fn to_value(&self) -> Value {
        let (run, job) = (self.run, self.job);
        let time = |instant: Option<_>| instant.map_or(Value::Null, |at| Value::Str(isoformat(at)));
        let labels = job.labels.iter().map(|label| Value::text(label)).collect();
        object([
            ("run_id", run.id.clone()),
            ("run_attempt", run.attempt.clone()),
            ("run_url", Value::text(&run.url)),
            ("job_id", job.id.clone()),
            ("job_url", Value::text(&job.url)),
            ("name", Value::text(&job.name)),
            ("run_conclusion", Value::text(&run.conclusion)),
            ("conclusion", Value::text(&job.conclusion)),
            ("created_at", time(job.created)),
            ("started_at", time(job.started)),
            ("completed_at", time(job.completed)),
            ("dependency_ready_at", time(job.dependency_ready)),
            ("duration_seconds", Value::opt_float(self.duration)),
            ("execution_seconds", Value::opt_float(self.duration)),
            ("wall_seconds", Value::opt_float(self.wall)),
            ("queue_seconds", Value::opt_float(self.queue)),
            ("runner_queue_seconds", Value::opt_float(self.runner_queue)),
            (
                "dependency_wait_seconds",
                Value::opt_float(self.dependency_wait),
            ),
            (
                "dependency_wait_observed",
                Value::Bool(self.dependency_wait.is_some()),
            ),
            (
                "capacity_contaminated",
                Value::Bool(contaminated(self.runner_queue)),
            ),
            ("start_delay_seconds", Value::opt_float(self.start_delay)),
            ("runner_labels", Value::Array(labels)),
            ("runner_dimensions", self.dimensions.to_value()),
        ])
    }
}

/// `included(run, requested)`: `Err(reason)` when the run is skipped.
pub(crate) fn included(run: &Run, requested: &str) -> Result<(), String> {
    if run.status != "completed" {
        return Err("not_completed".to_owned());
    }
    if matches!(requested, "all" | "completed") || run.conclusion == requested {
        return Ok(());
    }
    let conclusion = if run.conclusion.is_empty() {
        "missing"
    } else {
        &run.conclusion
    };
    Err(format!("conclusion_{conclusion}"))
}

fn is_cancelled(conclusion: &str) -> bool {
    matches!(conclusion, "cancelled" | "canceled")
}

/// Peak concurrency of `(instant, delta)` points, ends before starts.
fn peak(mut points: Vec<Point>) -> usize {
    points.sort_unstable();
    let (mut active, mut best) = (0_i64, 0_i64);
    for (_, delta) in points {
        active += i64::from(delta);
        best = best.max(active);
    }
    usize::try_from(best).unwrap_or(0)
}

/// `(provider, operating system, runner role)`.
type PeakKey<'s> = [Option<&'s str>; 3];
/// A start (+1) or end (-1) at an instant in microseconds.
type Point = (i64, i8);

pub(crate) fn capacity_metrics(observations: &[Observation<'_>]) -> Value {
    let runner = float_sum(observations.iter().filter_map(|sample| sample.duration));
    let cancelled = float_sum(observations.iter().filter_map(|sample| {
        let stopped = is_cancelled(&sample.run.conclusion) || is_cancelled(&sample.job.conclusion);
        sample.duration.filter(|_| stopped)
    }));
    let mut groups: Vec<(PeakKey<'_>, Vec<Point>)> = Vec::new();
    let mut all_points = Vec::new();
    for sample in observations {
        let (Some(started), Some(completed)) = (sample.job.started, sample.job.completed) else {
            continue;
        };
        let dimensions = &sample.dimensions;
        let key = [
            Some(dimensions.provider.as_str()),
            dimensions.operating_system.as_deref(),
            dimensions.runner_role.as_deref(),
        ];
        let points = [(started.micros(), 1), (completed.micros(), -1)];
        match groups.iter_mut().find(|(seen, _)| *seen == key) {
            Some(group) => group.1.extend(points),
            None => groups.push((key, points.to_vec())),
        }
        all_points.extend(points);
    }
    let mut peaks: Vec<(String, Value)> = Vec::new();
    for (key, points) in groups {
        let name = key
            .map(|part| part.filter(|text| !text.is_empty()).unwrap_or("unknown"))
            .join("/");
        let value = Value::count(peak(points));
        match peaks.iter_mut().find(|(seen, _)| *seen == name) {
            Some(entry) => entry.1 = value,
            None => peaks.push((name, value)),
        }
    }
    peaks.sort_by(|(a, _), (b, _)| a.cmp(b));
    let total = if all_points.is_empty() {
        Value::Null
    } else {
        Value::count(peak(all_points.clone()))
    };
    object([
        ("runner_minutes", Value::Float(round(runner / 60.0, 3))),
        (
            "cancelled_runner_minutes",
            Value::Float(round(cancelled / 60.0, 3)),
        ),
        ("timestamped_job_count", Value::count(all_points.len() / 2)),
        (
            "peak_workers",
            object([
                ("total", total),
                ("by_provider_os_role", Value::Object(peaks)),
            ]),
        ),
    ])
}

pub(crate) fn queue_heuristics(job_queue: &Summary, terminal_queue: &Summary) -> Value {
    let (queue_p95, terminal_p95) = (job_queue.p95(), terminal_queue.p95());
    let capacity_contaminated = contaminated(queue_p95) || contaminated(terminal_p95);
    let (state, interpretation) = if job_queue.count < MIN_HEURISTIC_SAMPLES {
        (
            "insufficient_sample",
            "collect at least the minimum sample count",
        )
    } else if capacity_contaminated {
        (
            "rollback",
            "queue p95 reaches the capacity-contamination threshold",
        )
    } else if queue_p95.is_none_or(|p95| p95 > QUEUE_WARN_SECONDS) {
        ("hold", "queue p95 exceeds warn threshold or is unavailable")
    } else {
        (
            "eligible",
            "queue p95 <= warn threshold and no capacity contamination",
        )
    };
    object([
        (
            "thresholds_seconds",
            object([
                ("queue_warn", Value::Float(QUEUE_WARN_SECONDS)),
                (
                    "queue_capacity_contamination",
                    Value::Float(QUEUE_CONTAMINATION_SECONDS),
                ),
                ("minimum_samples", Value::count(MIN_HEURISTIC_SAMPLES)),
            ]),
        ),
        ("sample_count", Value::count(job_queue.count)),
        ("queue_p95_seconds", Value::opt_float(queue_p95)),
        (
            "terminal_job_queue_p95_seconds",
            Value::opt_float(terminal_p95),
        ),
        ("capacity_contaminated", Value::Bool(capacity_contaminated)),
        ("state", Value::text(state)),
        ("interpretation", Value::text(interpretation)),
    ])
}
