//! Positive causal execution evidence for the pinned contract fixture.
//! These checks establish readout mechanics, never answer quality.

use super::{Failure, Result, requests, validation};
use serde_json::{Value, json};

/// Construct a smoke request exercising both causal scoring methods.
pub(super) fn request(model: &str) -> Value {
    requests::request(
        model,
        json!({
            "single_token": {
                "type": "choice", "instructions": "Choose the responsible team.",
                "criteria": {"B": "Billing and refunds", "S": "Shipping and delivery"}
            },
            "multi_token": {
                "type": "choice", "instructions": "Choose the next action.",
                "criteria": {
                    "contact billing support": "Resolve a duplicate charge",
                    "track the shipment": "Locate a parcel"
                }
            }
        }),
        requests::DEFAULT_STATE,
    )
}

/// Validate ordered causal evidence, finite scores and zero generated tokens.
pub(super) fn validate(status: u16, body: &Value, request: &Value) -> Result<()> {
    validation::read(status, body, request)?;
    let readouts = body["readouts"]
        .as_object()
        .ok_or(Failure::Case("causal readout evidence missing"))?;
    if readouts.len() != 2 {
        return Err(Failure::Case("causal readout question count mismatch"));
    }
    for (key, method) in [
        ("single_token", "selected_token_logits"),
        ("multi_token", "candidate_log_likelihood"),
    ] {
        let readout = readouts
            .get(key)
            .ok_or(Failure::Case("causal question evidence missing"))?;
        validate_question(
            readout,
            &body["answers"][key],
            &request["questions"][key],
            method,
        )?;
    }
    let usage = &body["readout_usage"];
    let evaluated = count(&usage["evaluated_tokens"])?;
    let preparation = count(&usage["cache_preparation_tokens"])?;
    count(&usage["cached_tokens"])?;
    if evaluated == 0 || preparation > evaluated {
        return Err(Failure::Case("invalid causal physical token accounting"));
    }
    Ok(())
}

/// Read a required nonnegative integer from the smoke response.
fn count(value: &Value) -> Result<u64> {
    value.as_u64().ok_or(Failure::Case(
        "causal token count must be an unsigned integer",
    ))
}

/// Validate one question's candidates, score semantics and token counts.
fn validate_question(
    readout: &Value,
    answer: &Value,
    question: &Value,
    method: &str,
) -> Result<()> {
    if readout["method"] != method
        || readout["normalization"] != "sum_log_probabilities"
        || readout["probability_semantics"] != "relative_candidate_probability_not_calibrated_truth"
    {
        return Err(Failure::Case("causal scoring method or semantics mismatch"));
    }
    count(&readout["evaluated_tokens"])?;
    count(&readout["cached_tokens"])?;
    let scores = readout["candidate_log_probabilities"]
        .as_array()
        .ok_or(Failure::Case("causal scores missing"))?;
    let lengths = readout["candidate_token_counts"]
        .as_array()
        .ok_or(Failure::Case("causal token lengths missing"))?;
    let options = question["criteria"]
        .as_object()
        .ok_or(Failure::Case("causal options missing"))?;
    if scores.len() != options.len() || lengths.len() != options.len() {
        return Err(Failure::Case("causal candidate cardinality mismatch"));
    }
    let lengths = lengths.iter().map(count).collect::<Result<Vec<_>>>()?;
    if lengths.iter().any(|n| !(1..=64).contains(n))
        || (method == "selected_token_logits" && lengths.iter().any(|n| *n != 1))
        || (method == "candidate_log_likelihood" && lengths.iter().all(|n| *n == 1))
    {
        return Err(Failure::Case(
            "causal method disagrees with tokenizer evidence",
        ));
    }
    let scores = scores
        .iter()
        .map(|value| {
            value
                .as_f64()
                .filter(|n| n.is_finite() && *n <= 0.0)
                .ok_or(Failure::Case("invalid causal log probability"))
        })
        .collect::<Result<Vec<_>>>()?;
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let denominator: f64 = scores.iter().map(|score| (score - maximum).exp()).sum();
    for (key, score) in options.keys().zip(scores) {
        let probability = answer["probabilities"][key]
            .as_f64()
            .ok_or(Failure::Case("causal answer probability missing"))?;
        if (probability - (score - maximum).exp() / denominator).abs() > validation::TOLERANCE {
            return Err(Failure::Case(
                "causal probabilities disagree with ordered scores",
            ));
        }
    }
    Ok(())
}
