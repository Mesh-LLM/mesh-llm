//! Inert response corruption fixture; it supplies no native-model proof.
use serde_json::{Value, json};

/// Build valid causal evidence for smoke-validator regression tests.
pub(super) fn response(request: &Value) -> Value {
    let mut body = super::answer(request, 0, "");
    let mut readouts = serde_json::Map::new();
    for key in ["single_token", "multi_token"] {
        readouts.insert(key.into(), json!({
            "method": if key == "single_token" {"selected_token_logits"} else {"candidate_log_likelihood"},
            "normalization": "sum_log_probabilities",
            "probability_semantics": "relative_candidate_probability_not_calibrated_truth",
            "candidate_log_probabilities": [0.8_f64.ln(), 0.2_f64.ln()],
            "candidate_token_counts": if key == "single_token" {vec![1, 1]} else {vec![3, 3]},
            "evaluated_tokens": 100, "cached_tokens": 0
        }));
    }
    body["readouts"] = Value::Object(readouts);
    body["readout_usage"] =
        json!({"evaluated_tokens":200,"cached_tokens":0,"cache_preparation_tokens":0});
    body
}

/// Inject one malformed evidence case without changing unrelated fixture fields.
pub(super) fn mutate(body: &mut Value, mutation: &str) -> u16 {
    match mutation {
        "causal-refusal" => return 502,
        "causal-evidence-missing" => {
            body.as_object_mut().unwrap().remove("readouts");
        }
        "causal-question-missing" => {
            let evidence = body["readouts"]
                .as_object_mut()
                .unwrap()
                .remove("single_token")
                .unwrap();
            body["readouts"]["wrong_question"] = evidence;
        }
        "causal-method" => {
            body["readouts"]["single_token"]["method"] = json!("candidate_log_likelihood")
        }
        "causal-tokenizer" => {
            body["readouts"]["multi_token"]["candidate_token_counts"] = json!([1, 1])
        }
        "causal-token-fraction" => {
            body["readouts"]["single_token"]["candidate_token_counts"] = json!([1.5, 1])
        }
        "causal-positive-score" => {
            body["readouts"]["single_token"]["candidate_log_probabilities"] = json!([1, 0])
        }
        "causal-score-order" => {
            body["readouts"]["single_token"]["candidate_log_probabilities"] =
                json!([0.2_f64.ln(), 0.8_f64.ln()])
        }
        "causal-cardinality" => {
            body["readouts"]["multi_token"]["candidate_log_probabilities"] = json!([-1])
        }
        "causal-semantics" => {
            body["readouts"]["multi_token"]["probability_semantics"] = json!("calibrated")
        }
        "causal-accounting" => body["readout_usage"]["evaluated_tokens"] = json!(0),
        "causal-generated" => body["usage"]["output_tokens"] = json!(1),
        _ => (),
    }
    200
}
