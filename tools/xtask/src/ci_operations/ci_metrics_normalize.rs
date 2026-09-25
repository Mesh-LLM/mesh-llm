//! `load_runs`, `normalize_run`, `normalize_job`, `normalize_step` and
//! `labels` of `collect-ci-metrics.py`. Fields are evaluated in the legacy
//! dict-literal order, so the first invalid timestamp reported matches.

use crate::ci_operations::ci_metrics_input::check_job_shape;
use crate::ci_operations::ci_metrics_int::python_int_text;
use crate::ci_operations::ci_metrics_time::{Instant, TimeError, elapsed, timestamp};
use crate::ci_operations::ci_metrics_value::{Value, display};
use crate::ci_operations::sccache_argv::python_float;

/// How a legacy exception surfaces.
#[derive(Debug)]
pub(crate) enum Failure {
    /// Caught by `main`: `ci metrics error: {message}`, status 2.
    Reported(String),
    /// An exception `main` does not catch (a Python traceback, status 1).
    Uncaught(String),
}

pub(crate) type Outcome<T> = Result<T, Failure>;

pub(crate) struct Step {
    pub(crate) name: String,
    pub(crate) conclusion: String,
    pub(crate) duration: Option<f64>,
}

pub(crate) struct Job {
    pub(crate) id: Value,
    pub(crate) name: String,
    pub(crate) conclusion: String,
    pub(crate) created: Option<Instant>,
    pub(crate) started: Option<Instant>,
    pub(crate) completed: Option<Instant>,
    pub(crate) dependency_ready: Option<Instant>,
    pub(crate) dependency_wait: Option<f64>,
    pub(crate) runner_role: Value,
    pub(crate) operating_system: Value,
    pub(crate) url: String,
    pub(crate) labels: Vec<String>,
    pub(crate) steps: Vec<Step>,
}

pub(crate) struct Run {
    pub(crate) id: Value,
    pub(crate) attempt: Value,
    pub(crate) workflow: String,
    pub(crate) title: String,
    pub(crate) event: String,
    pub(crate) status: String,
    pub(crate) conclusion: String,
    pub(crate) created: Option<Instant>,
    pub(crate) started: Option<Instant>,
    pub(crate) updated: Option<Instant>,
    pub(crate) url: String,
    pub(crate) sha: String,
    pub(crate) branch: String,
    /// `plan_profile` .. `cache_mode`, passed through unchanged.
    pub(crate) metadata: Vec<(&'static str, Value)>,
    pub(crate) jobs: Vec<Job>,
}

impl Run {
    pub(crate) fn first_attempt(&self) -> bool {
        matches!(self.attempt, Value::Int(1))
    }
}

/// `pick(data, *names)`: the first present key, even when its value is null.
fn pick<'a>(data: &'a Value, names: &[&str]) -> Option<&'a Value> {
    names.iter().find_map(|name| data.get(name))
}

fn text(data: &Value, names: &[&str], default: &str) -> String {
    pick(data, names).map_or_else(|| default.to_owned(), display)
}

fn raw(data: &Value, names: &[&str]) -> Value {
    pick(data, names).cloned().unwrap_or(Value::Null)
}

fn time(data: &Value, names: &[&str]) -> Outcome<Option<Instant>> {
    match pick(data, names) {
        Some(Value::Str(value)) if !value.is_empty() => {
            timestamp(value).map_err(|error| match error {
                TimeError::Invalid(message) => Failure::Reported(message),
                TimeError::Overflow => {
                    Failure::Uncaught("OverflowError: date value out of range".to_owned())
                }
            })
        }
        _ => Ok(None),
    }
}

fn normalize_step(raw_step: &Value) -> Outcome<Step> {
    let started = time(raw_step, &["started_at", "startedAt"])?;
    let completed = time(raw_step, &["completed_at", "completedAt"])?;
    Ok(Step {
        name: text(raw_step, &["name"], "unknown step"),
        conclusion: text(raw_step, &["conclusion"], ""),
        duration: elapsed(started, completed),
    })
}

/// `_number(value)`: a non-negative `float(value)`, bools excluded.
fn number(value: Option<&Value>) -> Outcome<Option<f64>> {
    let parsed = match value {
        Some(Value::Int(int)) => int.to_string().parse::<f64>().ok(),
        Some(Value::BigInt(text)) => match text.parse::<f64>() {
            Ok(float) if float.is_finite() => Some(float),
            _ => {
                return Err(Failure::Uncaught(
                    "OverflowError: int too large to convert to float".to_owned(),
                ));
            }
        },
        Some(Value::Float(float)) => Some(*float),
        Some(Value::Str(text)) => python_float(text),
        _ => None,
    };
    Ok(parsed.filter(|seconds| *seconds >= 0.0))
}

fn string_list(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(items)) => items.iter().map(display).collect(),
        _ => Vec::new(),
    }
}

fn normalize_job(raw_job: &Value) -> Outcome<Job> {
    check_job_shape(raw_job)?;
    let labels = string_list(pick(raw_job, &["labels", "runner_labels"]));
    let mut steps = Vec::new();
    if let Some(Value::Array(items)) = pick(raw_job, &["steps"]) {
        for item in items.iter().filter(|item| matches!(item, Value::Object(_))) {
            steps.push(normalize_step(item)?);
        }
    }
    Ok(Job {
        id: raw(raw_job, &["id", "databaseId", "database_id"]),
        name: text(raw_job, &["name"], "unknown job"),
        conclusion: text(raw_job, &["conclusion"], ""),
        created: time(raw_job, &["created_at", "createdAt"])?,
        started: time(raw_job, &["started_at", "startedAt"])?,
        completed: time(raw_job, &["completed_at", "completedAt"])?,
        dependency_ready: time(
            raw_job,
            &["dependency_ready_at", "needs_completed_at", "ready_at"],
        )?,
        dependency_wait: number(pick(
            raw_job,
            &["dependency_wait_seconds", "needs_wait_seconds"],
        ))?,
        runner_role: raw(raw_job, &["runner_role", "role"]),
        operating_system: raw(raw_job, &["operating_system", "runner_os", "os"]),
        url: text(raw_job, &["html_url", "url"], ""),
        labels,
        steps,
    })
}

/// `max(int(raw_attempt), 1)` with `TypeError`/`ValueError` meaning 1; the
/// result is exact however large (`int(1e30)` keeps every binary digit).
fn attempt(value: Option<&Value>) -> Outcome<Value> {
    let digits = match value {
        None => None,
        Some(Value::Int(int)) => Some(int.to_string()),
        Some(Value::BigInt(text)) => Some(text.clone()),
        Some(Value::Bool(flag)) => Some(u8::from(*flag).to_string()),
        Some(Value::Float(float)) if float.is_infinite() => {
            return Err(Failure::Uncaught(
                "OverflowError: cannot convert float infinity to integer".to_owned(),
            ));
        }
        Some(Value::Float(float)) if float.is_nan() => None,
        Some(Value::Float(float)) => Some(format!("{:.0}", float.trunc())),
        Some(Value::Str(text)) => python_int_text(text),
        Some(_) => None,
    };
    let positive =
        digits.filter(|text| !text.starts_with('-') && !text.trim_start_matches('0').is_empty());
    Ok(positive.map_or(Value::Int(1), |text| {
        text.parse::<i128>()
            .map_or_else(|_| Value::BigInt(text.clone()), Value::Int)
    }))
}

const METADATA: [(&str, [&str; 2]); 6] = [
    ("plan_profile", ["plan_profile", "profile"]),
    ("plan_digest", ["plan_digest", "planDigest"]),
    ("change_class", ["change_class", "changeClass"]),
    ("runner_image", ["runner_image", "runnerImage"]),
    ("toolchain_epoch", ["toolchain_epoch", "toolchainEpoch"]),
    ("cache_mode", ["cache_mode", "cacheMode"]),
];

pub(crate) fn normalize_run(raw_run: &Value) -> Outcome<Run> {
    let Some(Value::Array(raw_jobs)) = raw_run.get("jobs") else {
        let run_id = text(raw_run, &["id", "databaseId", "database_id"], "unknown");
        return Err(Failure::Reported(format!(
            "run {run_id} has no jobs array; use --raw-out or save \
             'gh run view --json ...,jobs' output"
        )));
    };
    let attempt = attempt(pick(raw_run, &["attempt", "run_attempt"]))?;
    let id = raw(raw_run, &["id", "databaseId", "database_id"]);
    let workflow = text(raw_run, &["workflow_name", "workflowName"], "");
    let title = text(raw_run, &["title", "displayTitle"], "");
    let event = text(raw_run, &["event"], "");
    let status = text(raw_run, &["status"], "");
    let conclusion = text(raw_run, &["conclusion"], "");
    let created = time(raw_run, &["created_at", "createdAt"])?;
    let started = time(raw_run, &["started_at", "startedAt"])?;
    let updated = time(raw_run, &["updated_at", "updatedAt"])?;
    let metadata = METADATA
        .iter()
        .map(|(name, keys)| (*name, raw(raw_run, keys)))
        .collect();
    let jobs = raw_jobs.iter().map(normalize_job).collect::<Outcome<_>>()?;
    Ok(Run {
        id,
        attempt,
        workflow,
        title,
        event,
        status,
        conclusion,
        created,
        started,
        updated,
        url: text(raw_run, &["html_url", "url"], ""),
        sha: text(raw_run, &["head_sha", "headSha"], ""),
        branch: text(raw_run, &["head_branch", "headBranch"], ""),
        metadata,
        jobs,
    })
}
