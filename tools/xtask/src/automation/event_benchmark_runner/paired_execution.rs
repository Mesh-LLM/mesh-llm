//! Interleave actual trial attempts; preserve partial failures without inventing order.
use super::{health_log::Observation, plan, stream_metrics::Measurement};
use crate::command::DynResult;
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Default)]
pub(super) struct Outcome {
    pub launched: bool,
    pub measurement: Option<Measurement>,
    pub setup_ms: Option<f64>,
    pub readiness_ms: Option<f64>,
    pub shutdown_ms: Option<f64>,
    pub health: Observation,
    pub error: Option<String>,
    pub health_observation_error: Option<String>,
}

#[derive(Serialize)]
pub(super) struct Record {
    pub launched: bool,
    pub scenario: String,
    pub pair_index: u64,
    pub side_order_first: String,
    pub prompt_sha256: String,
    pub status: &'static str,
    pub completion_tokens: Option<u64>,
    pub elapsed_ms: Option<f64>,
    pub decode_tok_s: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub decode_only_tok_s: Option<f64>,
    pub setup_ms: Option<f64>,
    pub readiness_ms: Option<f64>,
    pub shutdown_ms: Option<f64>,
    pub error: Option<String>,
    pub health_observation_error: Option<String>,
}

fn nonnegative(value: Option<f64>) -> Option<f64> {
    value.filter(|n| n.is_finite() && *n >= 0.0)
}

impl Record {
    fn from_outcome(entry: &plan::Entry, outcome: Outcome) -> Self {
        let succeeded = outcome.launched
            && outcome.error.is_none()
            && outcome
                .measurement
                .as_ref()
                .is_some_and(|metrics| !metrics.malformed);
        let metrics = succeeded.then_some(outcome.measurement).flatten();
        Self {
            launched: outcome.launched,
            scenario: entry.scenario.clone(),
            pair_index: entry.pair_index,
            side_order_first: entry.side_order_first.clone(),
            prompt_sha256: entry.prompt_sha256(),
            status: if succeeded { "succeeded" } else { "failed" },
            completion_tokens: metrics.as_ref().and_then(|m| m.completion_tokens),
            elapsed_ms: metrics
                .as_ref()
                .and_then(|m| nonnegative(Some(m.elapsed_ms))),
            decode_tok_s: metrics.as_ref().and_then(|m| nonnegative(m.decode_tok_s)),
            ttft_ms: metrics.as_ref().and_then(|m| nonnegative(m.ttft_ms)),
            decode_only_tok_s: metrics
                .as_ref()
                .and_then(|m| nonnegative(m.decode_only_tok_s)),
            setup_ms: nonnegative(outcome.setup_ms),
            readiness_ms: nonnegative(outcome.readiness_ms),
            shutdown_ms: nonnegative(outcome.shutdown_ms),
            error: outcome.error,
            health_observation_error: outcome.health_observation_error,
        }
    }
}

#[derive(Serialize)]
pub(super) struct Order {
    pub scenario: String,
    pub pair_index: u64,
    pub order: [String; 2],
}

#[derive(Default)]
pub(super) struct Batch {
    pub side_ids: [String; 2],
    pub trials: [Vec<Record>; 2],
    pub final_health: [Observation; 2],
    pub executed_order: Vec<Order>,
    pub interrupted: Option<String>,
}

fn validate(entries: &[plan::Entry], sides: &[plan::Side; 2]) -> DynResult<()> {
    let mut keys = BTreeSet::new();
    if sides.iter().any(|side| side.side_id.is_empty()) || sides[0].side_id == sides[1].side_id {
        return Err("paired execution requires two distinct nonempty side identities".into());
    }
    for entry in entries {
        if !sides
            .iter()
            .any(|side| side.side_id == entry.side_order_first)
            || !keys.insert((&entry.scenario, entry.pair_index))
        {
            return Err("trial plan has an unknown first side or duplicate pair identity".into());
        }
    }
    Ok(())
}

pub(super) fn run(
    entries: &[plan::Entry],
    sides: &[plan::Side; 2],
    mut execute: impl FnMut(&plan::Side, &plan::Entry) -> DynResult<Outcome>,
) -> DynResult<Batch> {
    validate(entries, sides)?;
    let mut batch = Batch {
        side_ids: [sides[0].side_id.clone(), sides[1].side_id.clone()],
        ..Batch::default()
    };
    for entry in entries {
        let first = usize::from(sides[1].side_id == entry.side_order_first);
        for index in [first, 1 - first] {
            let outcome = match execute(&sides[index], entry) {
                Ok(outcome) => outcome,
                Err(error) => {
                    batch.interrupted = Some(error.to_string());
                    return Ok(batch);
                }
            };
            let launched = outcome.launched;
            batch.final_health[index] = outcome.health.clone();
            batch.trials[index].push(Record::from_outcome(entry, outcome));
            if !launched {
                batch.interrupted =
                    Some("trial could not launch; paired order is incomplete".into());
                return Ok(batch);
            }
        }
        batch.executed_order.push(Order {
            scenario: entry.scenario.clone(),
            pair_index: entry.pair_index,
            order: [
                sides[first].side_id.clone(),
                sides[1 - first].side_id.clone(),
            ],
        });
    }
    Ok(batch)
}

#[cfg(test)]
#[path = "paired_execution_tests.rs"]
mod tests;
