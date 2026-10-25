use super::{cases::Case, options::Options, requests, responses, transport::Http};
use serde_json::Value;
use std::time::Instant;

pub(super) async fn run(http: &Http, options: &Options) -> Vec<Case> {
    let mut cases = vec![models(http).await];
    for model in &options.models {
        for attempt in 1..=options.attempts {
            if http.is_cancelled() {
                return cases;
            }
            cases.push(chat(http, model, attempt, false).await);
            if options.streaming && !http.is_cancelled() {
                cases.push(chat(http, model, attempt, true).await);
            }
        }
    }
    cases
}

async fn models(http: &Http) -> Case {
    let started = Instant::now();
    match http.models().await {
        Err(error) => Case::failure(None, None, "models", started, error),
        Ok(reply) => {
            let outcome = reply
                .json
                .as_ref()
                .and_then(|value| value.get("data"))
                .and_then(Value::as_array)
                .filter(|models| !models.is_empty())
                .map(|models| format!("{} models", models.len()))
                .ok_or_else(|| "models response did not contain any models".into());
            Case::reply(None, None, "models", started, &reply, outcome)
        }
    }
}

async fn chat(http: &Http, model: &str, attempt: u32, stream: bool) -> Case {
    let started = Instant::now();
    let phase = if stream { "stream_chat" } else { "chat" };
    let expected = if stream { "STREAM_OK" } else { "STABILITY_OK" };
    let payload = requests::surface(model, attempt, stream);
    match http.chat(&payload, stream).await {
        Err(error) => Case::failure(Some(model), Some(attempt), phase, started, error),
        Ok(reply) => {
            let content = if stream {
                responses::stream_content(&reply.events)
            } else {
                reply
                    .json
                    .as_ref()
                    .ok_or_else(|| "JSON reply missing".into())
                    .and_then(|value| responses::final_content(value).map(str::to_owned))
            };
            let outcome =
                content.and_then(|content| responses::validate_answer(&content, expected));
            Case::reply(Some(model), Some(attempt), phase, started, &reply, outcome)
        }
    }
}
