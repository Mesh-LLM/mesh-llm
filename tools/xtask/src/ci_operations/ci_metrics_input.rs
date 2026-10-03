//! Saved CI run envelopes, object-shaped jobs and benchmark label admission.

use crate::ci_operations::ci_metrics_normalize::{Failure, Outcome};
use crate::ci_operations::ci_metrics_value::{Value, parse};
use crate::repository::text;

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

/// A job record is a JSON object; collections and scalars are not jobs.
pub(crate) fn check_job_shape(raw_job: &Value) -> Outcome<()> {
    if matches!(raw_job, Value::Object(_)) {
        Ok(())
    } else {
        Err(Failure::Uncaught("CI job record must be an object".into()))
    }
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
                    text::repr(value)
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
