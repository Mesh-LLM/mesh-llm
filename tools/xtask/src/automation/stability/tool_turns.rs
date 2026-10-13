use super::{cases::Case, options::Options, requests, responses, tool_calls, transport::Http};
use std::time::Instant;

pub(super) async fn run(http: &Http, options: &Options) -> Vec<Case> {
    let mut cases = Vec::new();
    for model in &options.models {
        for attempt in 1..=options.attempts {
            if http.is_cancelled() {
                return cases;
            }
            cases.extend(turn(http, model, attempt, false).await);
            if options.streaming && !http.is_cancelled() {
                cases.extend(turn(http, model, attempt, true).await);
            }
        }
    }
    cases
}

async fn turn(http: &Http, model: &str, attempt: u32, stream: bool) -> Vec<Case> {
    let started = Instant::now();
    let phase = if stream {
        "stream_tool_call"
    } else {
        "tool_call"
    };
    let payload = requests::tool(model, attempt, stream);
    let reply = match http.chat(&payload, stream).await {
        Ok(reply) => reply,
        Err(error) => {
            return vec![Case::failure(
                Some(model),
                Some(attempt),
                phase,
                started,
                error,
            )];
        }
    };
    let call = if stream {
        tool_calls::extract_stream(&reply.events)
    } else {
        reply
            .json
            .as_ref()
            .ok_or_else(|| "JSON reply missing".into())
            .and_then(tool_calls::extract)
    };
    let call = match call {
        Ok(call) => call,
        Err(error) => {
            return vec![Case::reply(
                Some(model),
                Some(attempt),
                phase,
                started,
                &reply,
                Err(error),
            )];
        }
    };
    let mut cases = vec![Case::reply(
        Some(model),
        Some(attempt),
        phase,
        started,
        &reply,
        Ok(format!("{} key={}", tool_calls::TOOL_NAME, call.key.name())),
    )];
    if http.is_cancelled() {
        return cases;
    }
    let original = reply
        .json
        .as_ref()
        .and_then(|value| responses::message(value).ok());
    let payload = requests::continuation(model, attempt, &call, original, stream);
    cases.push(continuation(http, model, attempt, stream, &payload, call.key.fact()).await);
    cases
}

async fn continuation(
    http: &Http,
    model: &str,
    attempt: u32,
    stream: bool,
    payload: &serde_json::Value,
    expected: &str,
) -> Case {
    let started = Instant::now();
    let phase = if stream {
        "stream_tool_result"
    } else {
        "tool_result"
    };
    match http.chat(payload, stream).await {
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
