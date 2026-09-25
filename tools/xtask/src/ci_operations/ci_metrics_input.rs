//! Input-shape rules of `collect-ci-metrics.py`: `load_runs`, the
//! `TypeError` a non-dict job raises inside `normalize_job`, and `labels`.

use crate::ci_operations::ci_metrics_normalize::{Failure, Outcome};
use crate::ci_operations::ci_metrics_value::{Value, parse};
use crate::repository::python_text;

/// `load_runs` after the bytes are read: a run, a run array, or `.runs`.
pub(crate) fn load_runs(bytes: &[u8]) -> Outcome<Vec<Value>> {
    let data = parse(bytes).map_err(Failure::Reported)?;
    let runs = match data {
        Value::Object(_) => match data.get("runs") {
            Some(Value::Array(runs)) => runs.clone(),
            _ => vec![data],
        },
        Value::Array(runs) => runs,
        _ => return Err(shape_error()),
    };
    if runs.iter().all(|run| matches!(run, Value::Object(_))) {
        Ok(runs)
    } else {
        Err(shape_error())
    }
}

fn shape_error() -> Failure {
    Failure::Reported("input must be a run object, run array, or object.runs".to_owned())
}

/// Keys `normalize_job` probes with `name in raw` before indexing.
const JOB_KEYS: [&str; 27] = [
    "labels",
    "runner_labels",
    "steps",
    "id",
    "databaseId",
    "database_id",
    "name",
    "conclusion",
    "created_at",
    "createdAt",
    "started_at",
    "startedAt",
    "completed_at",
    "completedAt",
    "dependency_ready_at",
    "needs_completed_at",
    "ready_at",
    "dependency_wait_seconds",
    "needs_wait_seconds",
    "needs",
    "runner_role",
    "role",
    "operating_system",
    "runner_os",
    "os",
    "html_url",
    "url",
];

/// A non-dict job survives `pick` only when `in` works on it (a list or a
/// string) and matches no probed key; indexing a match raises `TypeError`.
pub(crate) fn check_job_shape(raw_job: &Value) -> Outcome<()> {
    let found = |key: &str| match raw_job {
        Value::Array(items) => items
            .iter()
            .any(|item| matches!(item, Value::Str(text) if text == key)),
        Value::Str(text) => text.contains(key),
        _ => true,
    };
    let message = match raw_job {
        Value::Object(_) => return Ok(()),
        Value::Str(_) => "string indices must be integers, not 'str'",
        Value::Array(_) => "list indices must be integers or slices, not str",
        Value::Null => "argument of type 'NoneType' is not iterable",
        Value::Bool(_) => "argument of type 'bool' is not iterable",
        Value::Int(_) | Value::BigInt(_) => "argument of type 'int' is not iterable",
        Value::Float(_) => "argument of type 'float' is not iterable",
    };
    if JOB_KEYS.iter().any(|key| found(key)) {
        return Err(Failure::Uncaught(format!("TypeError: {message}")));
    }
    Ok(())
}

/// `labels(values)`: `KEY=VALUE` pairs, later keys winning, sorted by key.
pub(crate) fn labels(values: &[String]) -> Outcome<Vec<(String, Value)>> {
    let mut result: Vec<(String, Value)> = Vec::new();
    for value in values {
        let (key, label) = match value.split_once('=') {
            Some((key, label)) if !key.is_empty() => (key, label),
            _ => {
                return Err(Failure::Reported(format!(
                    "benchmark label must be KEY=VALUE, got {}",
                    python_text::repr(value)
                )));
            }
        };
        match result.iter_mut().find(|(seen, _)| seen == key) {
            Some(entry) => entry.1 = Value::text(label),
            None => result.push((key.to_owned(), Value::text(label))),
        }
    }
    result.sort_by(|(a, _), (b, _)| a.cmp(b));
    Ok(result)
}
