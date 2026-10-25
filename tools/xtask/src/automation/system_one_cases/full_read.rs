use super::{Failure, Result, passed, requests as r, transport::Client, validation as v};
use serde_json::{Value, json};
async fn read(client: &Client<'_>, request: &Value) -> Result<Value> {
    let (status, body) = client.json(request).await?;
    v::read(status, &body, request)?;
    Ok(body)
}
pub(super) async fn run(client: &Client<'_>, cases: &mut Vec<Value>) -> Result<()> {
    let model = &client.options.model;
    let base = [
        ("noul-read", r::billing(), r::DEFAULT_STATE),
        (
            "choice-read",
            json!({"team":r::choice(4,"Which team should handle it?")}),
            r::DEFAULT_STATE,
        ),
        (
            "score-read",
            json!({"urgency":r::score(4)}),
            r::DEFAULT_STATE,
        ),
        (
            "mixed-read",
            json!({"billing":r::noul("Is this a billing issue?"),"team":r::choice(3,"Which team should handle it?"),"urgency":r::score(5)}),
            "The invoice shows two charges and the second one is unexplained.",
        ),
    ];
    for (name, questions, state) in base {
        read(client, &r::request(model, questions, state)).await?;
        passed(cases, name);
    }
    read(
        client,
        &r::request(&client.options.alias, r::billing(), r::DEFAULT_STATE),
    )
    .await?;
    passed(cases, "alias-read");
    let first_request = r::request(
        model,
        json!({"billing":r::noul("Is this a billing issue?"),"team":r::choice(3,"Which team should handle it?")}),
        r::DEFAULT_STATE,
    );
    let other_request = r::request(
        model,
        json!({"billing":r::noul("Is this a refund request?"),"team":r::choice(3,"Which queue should receive it?")}),
        "The deployment tool crashes with a segmentation fault on startup.",
    );
    let first = read(client, &first_request).await?;
    let other = read(client, &other_request).await?;
    let repeated = read(client, &first_request).await?;
    if v::differences(&first, &repeated)?
        .iter()
        .any(|delta| *delta > v::TOLERANCE)
    {
        return Err(Failure::Case("repeated interleaved read changed"));
    }
    if !v::differences(&first, &other)?
        .iter()
        .any(|delta| *delta > v::DISTINCT)
    {
        return Err(Failure::Case("different states produced a constant read"));
    }
    passed(cases, "interleaved-read-determinism");
    Ok(())
}
