use super::{Attestation, Check, Rejection};
use crate::ci_operations::ci_metrics_value::{self as python, Value};

pub(crate) const RESPONSE_LIMIT: usize = 1024 * 1024;

pub(crate) struct Response<'a> {
    pub(crate) status: u16,
    pub(crate) body: &'a [u8],
}

pub(crate) enum Transfer<'a> {
    Complete(Response<'a>),
    Failed,
    TimedOut,
    Oversized,
}

pub(super) enum Evidence {
    Pending,
    Accepted,
    Model(String),
}

pub(super) fn classify(
    check: Check,
    transfer: Transfer<'_>,
    attestation: &Attestation,
) -> Result<Evidence, Rejection> {
    let body = match transfer {
        Transfer::Oversized => return Err(Rejection::ResponseLimit(check)),
        Transfer::TimedOut => return Err(Rejection::Deadline(check)),
        Transfer::Failed => return unavailable(check),
        Transfer::Complete(response) => {
            if response.body.len() > RESPONSE_LIMIT {
                return Err(Rejection::ResponseLimit(check));
            }
            if !(200..400).contains(&response.status) {
                return unavailable(check);
            }
            response.body
        }
    };
    let accepted = match check {
        Check::Runtime => return Ok(runtime(body)),
        Check::Models => return Ok(model(body)),
        Check::InspectAttestation | Check::RuntimeAttestation | Check::HeadlessAttestation => {
            attestation_matches(body, check, attestation)
        }
        Check::Chat => chat(body, true),
        Check::Auto => chat(body, false),
        Check::Stream => {
            body.windows(b"data: [DONE]".len())
                .any(|part| part == b"data: [DONE]")
                && body
                    .windows(b"\"role\":\"assistant\"".len())
                    .any(|part| part == b"\"role\":\"assistant\"")
        }
        Check::HeadlessModels | Check::HeadlessStatus => true,
    };
    if accepted {
        Ok(Evidence::Accepted)
    } else {
        Err(Rejection::Evidence(check))
    }
}

fn unavailable(check: Check) -> Result<Evidence, Rejection> {
    match check {
        Check::Runtime | Check::Models | Check::HeadlessModels | Check::HeadlessStatus => {
            Ok(Evidence::Pending)
        }
        Check::InspectAttestation
        | Check::RuntimeAttestation
        | Check::Chat
        | Check::Stream
        | Check::Auto
        | Check::HeadlessAttestation => Err(Rejection::Transfer(check)),
    }
}

fn runtime(body: &[u8]) -> Evidence {
    let Ok(value) = python::parse(body) else {
        return Evidence::Pending;
    };
    match value.get("llama_ready") {
        Some(Value::Bool(true)) => Evidence::Accepted,
        Some(Value::Str(text)) if text.trim_end_matches('\n') == "True" => Evidence::Accepted,
        _ => Evidence::Pending,
    }
}

fn model(body: &[u8]) -> Evidence {
    let Ok(value) = python::parse(body) else {
        return Evidence::Pending;
    };
    let Some(Value::Array(models)) = value.get("data") else {
        return Evidence::Pending;
    };
    let Some(Value::Object(first)) = models.first() else {
        return Evidence::Pending;
    };
    let Some((_, id)) = first.iter().find(|(key, _)| key == "id") else {
        return Evidence::Pending;
    };
    let rendered = python::display(id);
    let rendered = rendered.trim_end_matches('\n');
    if rendered.is_empty() {
        Evidence::Pending
    } else {
        Evidence::Model(rendered.to_owned())
    }
}

fn attestation_matches(body: &[u8], check: Check, requirement: &Attestation) -> bool {
    let expected = match requirement {
        Attestation::Required { expected } => expected,
        Attestation::Disabled => return false,
    };
    let Ok(value) = python::parse(body) else {
        return false;
    };
    let status = match check {
        Check::InspectAttestation => value.get("status"),
        Check::RuntimeAttestation | Check::HeadlessAttestation => value
            .get("release_attestation")
            .and_then(|value| value.get("status")),
        Check::Runtime
        | Check::Models
        | Check::Chat
        | Check::Stream
        | Check::Auto
        | Check::HeadlessModels
        | Check::HeadlessStatus => None,
    };
    status.is_some_and(|value| python::display(value).trim_end_matches('\n') == expected)
}

fn chat(body: &[u8], require_object: bool) -> bool {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) else {
        return false;
    };
    if require_object
        && value.get("object").and_then(serde_json::Value::as_str) != Some("chat.completion")
    {
        return false;
    }
    let content = value
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"));
    match content {
        Some(serde_json::Value::String(text)) => !text.is_empty(),
        Some(serde_json::Value::Array(items)) => !items.is_empty(),
        Some(serde_json::Value::Object(fields)) => !fields.is_empty(),
        Some(serde_json::Value::Number(number)) => {
            number.as_f64().is_some_and(|value| value.abs() > 0.0)
        }
        Some(serde_json::Value::Null | serde_json::Value::Bool(_)) | None => false,
    }
}
