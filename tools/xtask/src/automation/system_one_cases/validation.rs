use super::{Failure, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
pub(super) const TOLERANCE: f64 = 1e-3;
pub(super) const DISTINCT: f64 = 1e-2;
fn number(value: &Value) -> Result<f64> {
    value
        .as_f64()
        .filter(|n| n.is_finite())
        .ok_or(Failure::Case("answer must contain finite numeric values"))
}
fn unit(value: &Value) -> Result<f64> {
    let n = number(value)?;
    if !(0.0..=1.0).contains(&n) {
        return Err(Failure::Case(
            "probability/confidence outside unit interval",
        ));
    }
    Ok(n)
}
fn probabilities(answer: &Value, keys: &BTreeSet<String>) -> Result<BTreeMap<String, f64>> {
    let values = answer["probabilities"]
        .as_object()
        .ok_or(Failure::Case("probabilities must be an object"))?;
    if values.keys().cloned().collect::<BTreeSet<_>>() != *keys {
        return Err(Failure::Case(
            "probability labels do not match requested options",
        ));
    }
    let values: BTreeMap<_, _> = values
        .iter()
        .map(|(k, v)| unit(v).map(|n| (k.clone(), n)))
        .collect::<Result<_>>()?;
    if (values.values().sum::<f64>() - 1.0).abs() > TOLERANCE {
        return Err(Failure::Case("probabilities do not sum to one"));
    }
    unit(&answer["confidence"])?;
    Ok(values)
}
pub(super) fn read(status: u16, body: &Value, request: &Value) -> Result<()> {
    if status != 200
        || body["model"].as_str().is_none_or(str::is_empty)
        || body["model"] != request["model"]
    {
        return Err(Failure::Case("read status/model mismatch"));
    }
    let usage = body["usage"]
        .as_object()
        .ok_or(Failure::Case("read usage missing"))?;
    if usage.len() != 2
        || !usage
            .get("input_tokens")
            .and_then(Value::as_u64)
            .is_some_and(|n| n > 0)
        || usage.get("output_tokens").and_then(Value::as_u64) != Some(0)
    {
        return Err(Failure::Case(
            "read usage must report positive input and no generated tokens",
        ));
    }
    let answers = body["answers"]
        .as_object()
        .ok_or(Failure::Case("read answers missing"))?;
    let questions = request["questions"]
        .as_object()
        .ok_or(Failure::Case("questions missing"))?;
    if answers.keys().collect::<BTreeSet<_>>() != questions.keys().collect::<BTreeSet<_>>() {
        return Err(Failure::Case("read question set mismatch"));
    }
    for (key, question) in questions {
        answer(&answers[key], question)?;
    }
    Ok(())
}
fn answer(answer: &Value, question: &Value) -> Result<()> {
    if answer["type"] != question["type"] {
        return Err(Failure::Case("answer type mismatch"));
    }
    match question["type"].as_str() {
        Some("noul") => {
            unit(&answer["noul"])?;
        }
        Some("choice") => {
            let keys = question["criteria"]
                .as_object()
                .ok_or(Failure::Case("choice criteria missing"))?
                .keys()
                .cloned()
                .collect();
            let distribution = probabilities(answer, &keys)?;
            let selected = answer["choice"]
                .as_str()
                .and_then(|k| distribution.get(k))
                .ok_or(Failure::Case("choice label outside distribution"))?;
            if distribution.values().any(|value| value > selected) {
                return Err(Failure::Case("choice is not a maximum probability option"));
            }
        }
        Some("score") => {
            let count = question["criteria"]
                .as_array()
                .ok_or(Failure::Case("score criteria missing"))?
                .len();
            let keys = (0..count).map(|i| i.to_string()).collect();
            let distribution = probabilities(answer, &keys)?;
            let score = number(&answer["score"])?;
            if score < 0.0 || score > (count - 1) as f64 {
                return Err(Failure::Case("score outside requested range"));
            }
            let legend = answer["legend"]
                .as_object()
                .ok_or(Failure::Case("score legend missing"))?;
            if legend.keys().cloned().collect::<BTreeSet<_>>() != keys {
                return Err(Failure::Case("score legend labels mismatch"));
            }
            let expected = (0..count)
                .map(|i| i as f64 * distribution[&i.to_string()])
                .sum::<f64>();
            if (score - expected).abs() > TOLERANCE {
                return Err(Failure::Case("score is not distribution expectation"));
            }
        }
        _ => return Err(Failure::Case("unsupported answer type")),
    }
    Ok(())
}
pub(super) fn differences(left: &Value, right: &Value) -> Result<Vec<f64>> {
    let a = left["answers"]
        .as_object()
        .ok_or(Failure::Case("answers missing"))?;
    let b = right["answers"]
        .as_object()
        .ok_or(Failure::Case("answers missing"))?;
    if a.keys().collect::<BTreeSet<_>>() != b.keys().collect::<BTreeSet<_>>() {
        return Err(Failure::Case("repeated question set changed"));
    }
    let mut differences = Vec::new();
    for (key, first) in a {
        let second = &b[key];
        if first["type"] != second["type"] || first["choice"] != second["choice"] {
            differences.push(f64::INFINITY);
        }
        for field in ["noul", "score"] {
            if first.get(field).is_some() {
                differences.push((number(&first[field])? - number(&second[field])?).abs());
            }
        }
        if let Some(values) = first["probabilities"].as_object() {
            for (label, value) in values {
                differences.push((number(value)? - number(&second["probabilities"][label])?).abs());
            }
        }
    }
    Ok(differences)
}
