use super::transport::{Failure, Reply, millis};
use serde::Serialize;
use std::time::Instant;

#[derive(Serialize)]
pub(super) struct Case {
    pub model: Option<String>,
    pub attempt: Option<u32>,
    pub phase: &'static str,
    pub ok: bool,
    pub detail: String,
    pub elapsed_ms: u64,
    pub status_code: Option<u16>,
    pub ttft_ms: Option<u64>,
    pub actual_model: Option<String>,
    pub tok_per_sec: Option<f64>,
}

impl Case {
    pub fn reply(
        model: Option<&str>,
        attempt: Option<u32>,
        phase: &'static str,
        started: Instant,
        reply: &Reply,
        outcome: Result<String, String>,
    ) -> Self {
        let elapsed_ms = millis(started);
        let objects = reply
            .json
            .iter()
            .chain(reply.events.iter())
            .collect::<Vec<_>>();
        let actual_model = objects
            .iter()
            .find_map(|value| value.get("model").and_then(serde_json::Value::as_str))
            .map(str::to_owned);
        let tokens = objects.iter().rev().find_map(|value| {
            value
                .get("usage")
                .and_then(|usage| usage.get("completion_tokens"))
                .and_then(serde_json::Value::as_u64)
        });
        let (ok, detail) = match outcome {
            Ok(detail) => (true, detail),
            Err(detail) => (false, detail),
        };
        Self {
            model: model.map(str::to_owned),
            attempt,
            phase,
            ok,
            detail,
            elapsed_ms,
            status_code: Some(reply.status),
            ttft_ms: reply.first_event_ms,
            actual_model,
            tok_per_sec: tokens
                .filter(|tokens| *tokens > 0)
                .map(|tokens| tokens as f64 / (elapsed_ms.max(1) as f64 / 1000.0)),
        }
    }

    pub fn failure(
        model: Option<&str>,
        attempt: Option<u32>,
        phase: &'static str,
        started: Instant,
        failure: Failure,
    ) -> Self {
        Self {
            model: model.map(str::to_owned),
            attempt,
            phase,
            ok: false,
            detail: failure.detail,
            elapsed_ms: millis(started),
            status_code: failure.status,
            ttft_ms: None,
            actual_model: None,
            tok_per_sec: None,
        }
    }
}
