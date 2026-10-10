//! Normalize saved/GitHub CI records into run, job and step observations.
//! Alias selection is first-present; timestamp fields are checked in record order.

use crate::ci_operations::ci_metrics_input::check_job_shape;
use crate::ci_operations::ci_metrics_time::{Instant, TimeError, elapsed, timestamp};
use crate::ci_operations::ci_metrics_value::Value;

/// Metric collection failures retain the established CLI status categories.
#[derive(Debug)]
pub(crate) enum Failure {
    /// Caught by `main`: `ci metrics error: {message}`, status 2.
    Reported(String),
    /// A structural failure reported through the CLI status-1 category.
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
    match pick(data, names) {
        Some(Value::Str(text)) => text.clone(),
        _ => default.to_owned(),
    }
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
                    Failure::Uncaught("CI timestamp is outside the supported range".to_owned())
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

/// Admit finite nonnegative durations from numbers or saved decimal text.
fn number(value: Option<&Value>) -> Outcome<Option<f64>> {
    let parsed = match value {
        Some(Value::Int(int)) => int.to_string().parse::<f64>().ok(),
        Some(Value::BigInt(text)) => text.parse::<f64>().ok(),
        Some(Value::Float(float)) => Some(*float),
        Some(Value::Str(text)) => text.parse::<f64>().ok(),
        _ => None,
    };
    if parsed.is_some_and(|seconds| !seconds.is_finite()) {
        return Err(Failure::Reported("CI duration must be finite".into()));
    }
    Ok(parsed.filter(|seconds| *seconds >= 0.0))
}

fn string_list(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| match item {
                Value::Str(text) => Some(text.clone()),
                _ => None,
            })
            .collect(),
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

/// A missing attempt is the first attempt; explicit attempts are positive
/// integers or exact decimal strings from saved CLI snapshots.
fn attempt(value: Option<&Value>) -> Outcome<Value> {
    let digits = match value {
        None | Some(Value::Null) => return Ok(Value::Int(1)),
        Some(Value::Int(int)) if *int > 0 => int.to_string(),
        Some(Value::BigInt(text) | Value::Str(text))
            if !text.is_empty()
                && text.bytes().all(|byte| byte.is_ascii_digit())
                && text.bytes().any(|byte| byte != b'0') =>
        {
            text.trim_start_matches('0').to_owned()
        }
        Some(_) => {
            return Err(Failure::Reported(
                "CI run attempt must be a positive integer or decimal string".into(),
            ));
        }
    };
    Ok(digits
        .parse::<i128>()
        .map_or_else(|_| Value::BigInt(digits), Value::Int))
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
    if matches!(id, Value::Array(_) | Value::Object(_)) {
        return Err(Failure::Reported("CI run identity must be a scalar".into()));
    }

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

#[cfg(test)]
mod admission_tests {
    use super::*;

    #[test]
    fn saved_decimal_attempts_and_durations_remain_supported() {
        assert!(matches!(
            attempt(Some(&Value::text("01"))),
            Ok(Value::Int(1))
        ));
        assert_eq!(number(Some(&Value::text("4.5"))).ok().flatten(), Some(4.5));
        assert_eq!(number(Some(&Value::Int(-3))).ok().flatten(), None);
        for value in [
            Value::Bool(true),
            Value::Float(1.0),
            Value::text(" 1 "),
            Value::text("1_0"),
            Value::Int(0),
        ] {
            assert!(attempt(Some(&value)).is_err());
        }
        assert!(number(Some(&Value::Float(f64::INFINITY))).is_err());
        assert_eq!(
            string_list(Some(&Value::Array(vec![
                Value::text("linux"),
                Value::Bool(true)
            ]))),
            vec!["linux"]
        );
    }
}
