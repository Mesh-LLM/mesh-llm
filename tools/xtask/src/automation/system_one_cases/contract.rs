use super::{Failure, Result, requests as r, transport::Client};
use serde_json::{Value, json};
pub(super) async fn run(client: &Client<'_>, cases: &mut Vec<Value>) -> Result<()> {
    let model = &client.options.model;
    let base = r::request(model, r::billing(), r::DEFAULT_STATE);
    let mut matrix = vec![
        (
            r::request("definitely-not-loaded", r::billing(), r::DEFAULT_STATE),
            "invalid_value",
        ),
        (
            r::request(model, json!({}), r::DEFAULT_STATE),
            "invalid_value",
        ),
    ];
    for (field, value) in [
        ("sequential", json!(true)),
        ("steps", json!(2)),
        ("samples", json!(2)),
        ("think", json!(1)),
        (
            "images",
            json!([{"type":"image_url","image_url":{"url":"data:,"}}]),
        ),
    ] {
        let mut request = base.clone();
        request[field] = value;
        matrix.push((request, "unsupported_model_feature"));
    }
    for count in [1, 27] {
        matrix.push((
            r::request(
                model,
                json!({"team":r::choice(count,"Which team should handle it?")}),
                r::DEFAULT_STATE,
            ),
            "invalid_value",
        ));
    }
    for count in [1, 11] {
        matrix.push((
            r::request(model, json!({"urgency":r::score(count)}), r::DEFAULT_STATE),
            "invalid_value",
        ));
    }
    for (request, code) in matrix {
        let (status, body) = client.json(&request).await?;
        error(status, &body, 400, "invalid_request_error", code)?;
    }
    let (status, body) = client.send("POST", b"{".to_vec()).await?;
    error(status, &body, 400, "invalid_request_error", "invalid_value")?;
    let (status, body) = client.send("GET", Vec::new()).await?;
    error(
        status,
        &body,
        405,
        "invalid_request_error",
        "method_not_allowed",
    )?;
    let (status, body) = client.json(&base).await?;
    error(status, &body, 502, "server_error", "service_unavailable")?;
    let message = body["error"]["message"]
        .as_str()
        .ok_or(Failure::Case("architecture refusal missing message"))?;
    if !["DiffusionGemma", "System One", "system-one"]
        .iter()
        .any(|name| message.contains(name))
    {
        return Err(Failure::Case(
            "architecture refusal must name System One capability",
        ));
    }
    cases.push(json!({"name":"contract-matrix","status":"pass","cases":14}));
    Ok(())
}
fn error(status: u16, body: &Value, expected: u16, kind: &str, code: &str) -> Result<()> {
    let error = &body["error"];
    if status != expected
        || error["type"].as_str() != Some(kind)
        || error["code"].as_str() != Some(code)
        || !error["message"]
            .as_str()
            .is_some_and(|m| !m.trim().is_empty())
    {
        return Err(Failure::Case("System One refusal status/envelope mismatch"));
    }
    Ok(())
}
