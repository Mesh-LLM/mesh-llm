//! Laya equivalence unit tests: normalized System One and Decisions reads.
use super::equivalence;
use serde_json::{Value, json};

fn system_read() -> Value {
    json!({
        "model": "laya",
        "answers": {
            "question_0": {"noul": 0.75},
            "question_1": {"choice": "billing", "probabilities": {"billing": 0.6, "support": 0.4}},
            "question_2": {"score": 0.25, "probabilities": {"0": 0.75, "1": 0.25}},
        },
    })
}

fn decisions_read() -> Value {
    json!({
        "model": "laya",
        "answers": [
            {"type": "predicate", "name": "urgent", "probability": 0.75},
            {"type": "choice", "name": "team", "choice": "billing",
             "probabilities": [
                 {"value": "billing", "probability": 0.6},
                 {"value": "support", "probability": 0.4},
             ]},
            {"type": "score", "name": "frustration", "score": 0.25,
             "probabilities": [
                 {"value": 0, "label": "Calm", "probability": 0.75},
                 {"value": 1, "label": "Frustrated", "probability": 0.25},
             ]},
        ],
        "usage": {"input_tokens": 120, "output_tokens": 0},
    })
}

#[test]
fn equivalent_normalized_read_passes() {
    equivalence(&system_read(), &decisions_read(), "laya").expect("equivalent reads");
}

#[test]
fn rejects_probability_and_label_corruption() {
    let mut decisions = decisions_read();
    decisions["answers"][0]["probability"] = json!(0.5);
    let error = equivalence(&system_read(), &decisions, "laya").unwrap_err();
    assert!(error.contains("predicate"), "{error}");

    let mut decisions = decisions_read();
    decisions["answers"][2]["probabilities"][1]["label"] = json!("Angry");
    let error = equivalence(&system_read(), &decisions, "laya").unwrap_err();
    assert!(error.contains("score levels"), "{error}");

    let mut decisions = decisions_read();
    decisions["answers"][1]["choice"] = json!("support");
    assert!(equivalence(&system_read(), &decisions, "laya").is_err());

    let mut decisions = decisions_read();
    decisions["usage"]["output_tokens"] = json!(3);
    assert!(equivalence(&system_read(), &decisions, "laya").is_err());

    let mut system = system_read();
    system["model"] = json!("other");
    assert!(equivalence(&system, &decisions_read(), "laya").is_err());
}
