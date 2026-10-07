use super::options::valid_id;
use crate::command::DynResult;
use serde_json::Value;
pub(super) fn select_model(models: &Value, requested: Option<&str>) -> DynResult<String> {
    let rows = models["data"]
        .as_array()
        .ok_or("Decisions discovery requires a model array")?;
    let capable = rows
        .iter()
        .filter_map(|row| {
            let id = row["id"].as_str()?;
            (valid_id(id)
                && !matches!(id, "mesh" | "auto")
                && row["capabilities"].as_array().is_some_and(|items| {
                    items.iter().any(|item| item.as_str() == Some("system_one"))
                }))
            .then_some(id)
        })
        .collect::<Vec<_>>();
    let selected = requested
        .or_else(|| capable.first().copied())
        .ok_or("no System One model advertised")?;
    if !capable.contains(&selected) {
        return Err("selected model does not advertise system_one".into());
    }
    Ok(selected.to_owned())
}
fn probability(value: &Value) -> bool {
    value
        .as_f64()
        .is_some_and(|v| v.is_finite() && (0.0..=1.0).contains(&v))
}
fn distributions(value: &Value, score: bool) -> bool {
    value.as_array().is_some_and(|rows| {
        rows.len() == 2
            && rows.iter().enumerate().all(|(index, row)| {
                let identity = if score {
                    row["value"].as_f64() == Some(if index == 0 { 0.0 } else { 1.0 })
                        && row["label"].as_str() == Some(if index == 0 { "0" } else { "1" })
                } else {
                    row["value"].as_str() == Some(if index == 0 { "billing" } else { "support" })
                };
                identity && probability(&row["probability"])
            })
    })
}
pub(super) fn validate(body: &Value, model: &str) -> DynResult<()> {
    let answers = body["answers"]
        .as_array()
        .ok_or("Decisions requires answers")?;
    let expected = [
        ("predicate", "urgent"),
        ("choice", "team"),
        ("score", "frustration"),
    ];
    if body["model"].as_str() != Some(model)
        || answers.len() != 3
        || answers.iter().zip(expected).any(|(row, (kind, name))| {
            row["type"].as_str() != Some(kind) || row["name"].as_str() != Some(name)
        })
    {
        return Err("unexpected Decisions response model or question order".into());
    }
    if !probability(&answers[0]["probability"]) {
        return Err("invalid Decisions predicate probability".into());
    }
    let choice = &answers[1];
    if !matches!(choice["choice"].as_str(), Some("billing" | "support"))
        || !probability(&choice["confidence"])
        || !distributions(&choice["probabilities"], false)
    {
        return Err("invalid Decisions choice answer".into());
    }
    let score = &answers[2];
    if !score["score"].as_f64().is_some_and(f64::is_finite)
        || !probability(&score["confidence"])
        || !distributions(&score["probabilities"], true)
    {
        return Err("invalid Decisions score answer".into());
    }
    if ["input_tokens", "output_tokens", "total_tokens"]
        .iter()
        .any(|key| body["usage"][*key].as_u64().is_none())
    {
        return Err("invalid Decisions token usage".into());
    }
    Ok(())
}
#[cfg(test)]
#[path = "response_tests.rs"]
mod tests;
