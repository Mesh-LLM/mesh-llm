mod privacy;

use super::OutputEvent;
use super::formatting::OutputEventPresentation;
use mesh_llm_events::logging::{
    envelope::CanonicalEnvelope,
    events::{LifecycleEvent, TokenUsage},
};
use serde_json::{Map, Value, json};

use self::privacy::{
    json_scalar, native_category, safe_native_params, sanitize_map, sanitize_text,
};

pub(super) fn projected_message(event: &OutputEvent) -> String {
    match event {
        OutputEvent::CanonicalLog(envelope) => envelope.presentation_message(),
        OutputEvent::LlamaNativeLog { message, .. } => sanitize_text(message),
        OutputEvent::Info { message, .. }
        | OutputEvent::Warning { message, .. }
        | OutputEvent::Error { message, .. }
        | OutputEvent::Fatal { message, .. } => sanitize_text(message),
        _ => event.message(),
    }
}

pub(super) fn projected_pretty_text(event: &OutputEvent) -> String {
    match event {
        OutputEvent::CanonicalLog(envelope) => envelope.presentation_local_summary(),
        OutputEvent::LlamaNativeLog {
            message, params, ..
        } => {
            let mut rendered = sanitize_text(message);
            for (key, value) in safe_native_params(params) {
                rendered.push_str("\n  ↳ ");
                rendered.push_str(&key);
                rendered.push('=');
                rendered.push_str(&json_scalar(&value));
            }
            rendered
        }
        OutputEvent::Info { .. }
        | OutputEvent::Warning { .. }
        | OutputEvent::Error { .. }
        | OutputEvent::Fatal { .. } => sanitize_text(&event.pretty_text()),
        _ => event.pretty_text(),
    }
}

pub(super) fn projected_summary_line(event: &OutputEvent) -> String {
    match event {
        OutputEvent::CanonicalLog(envelope) => envelope.presentation_local_summary(),
        OutputEvent::LlamaNativeLog { category, .. } => {
            format!("native {} update", native_category(category))
        }
        OutputEvent::Info { .. }
        | OutputEvent::Warning { .. }
        | OutputEvent::Error { .. }
        | OutputEvent::Fatal { .. } => sanitize_text(&event.summary_line()),
        _ => sanitize_text(&event.summary_line()),
    }
}

pub(super) fn projected_json_fields(event: &OutputEvent) -> Map<String, Value> {
    match event {
        OutputEvent::CanonicalLog(envelope) => canonical_fields(envelope),
        OutputEvent::LlamaNativeLog {
            category, params, ..
        } => {
            let mut fields = safe_native_params(params);
            fields.insert(
                "native_category".to_string(),
                Value::String(native_category(category).to_string()),
            );
            fields
        }
        OutputEvent::Info { .. }
        | OutputEvent::Warning { .. }
        | OutputEvent::Error { .. }
        | OutputEvent::Fatal { .. } => sanitize_map(event.json_fields()),
        _ => event.json_fields(),
    }
}

pub(super) fn canonical_terminal_request_id(event: &OutputEvent) -> Option<String> {
    match event {
        OutputEvent::CanonicalLog(envelope) if envelope.presentation_outcome().is_some() => {
            Some(envelope.request_id.as_uuid().to_string())
        }
        _ => None,
    }
}

fn canonical_fields(envelope: &CanonicalEnvelope) -> Map<String, Value> {
    let mut fields = match &envelope.event {
        LifecycleEvent::AttemptCompleted { status_code, .. } => {
            json!({ "status_code": status_code })
        }
        LifecycleEvent::Completed {
            status_code,
            duration_ms,
            ..
        } => json!({ "status_code": status_code, "duration_ms": duration_ms }),
        LifecycleEvent::Failed { status_code, .. }
        | LifecycleEvent::Rejected { status_code, .. } => {
            json!({ "status_code": status_code })
        }
        LifecycleEvent::UsageRecorded {
            prompt_tokens,
            cached_prompt_tokens,
            completion_tokens,
            total_tokens,
        } => json!({
            "prompt_tokens": prompt_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }),
        LifecycleEvent::Admitted { .. }
        | LifecycleEvent::RouteSelected { .. }
        | LifecycleEvent::AttemptStarted { .. }
        | LifecycleEvent::AttemptFailed { .. }
        | LifecycleEvent::BackendStreamFirstItem
        | LifecycleEvent::StreamStarted { .. }
        | LifecycleEvent::StreamChunk { .. }
        | LifecycleEvent::StreamCompleted { .. }
        | LifecycleEvent::StreamError { .. }
        | LifecycleEvent::AuditError { .. }
        | LifecycleEvent::Cancelled { .. }
        | LifecycleEvent::Dropped { .. } => json!({}),
    }
    .as_object()
    .cloned()
    .unwrap_or_default();
    match &envelope.event {
        LifecycleEvent::Completed { usage, .. } | LifecycleEvent::StreamCompleted { usage, .. } => {
            insert_token_usage(&mut fields, usage.as_ref());
        }
        _ => {}
    }
    fields.retain(|_, value| !value.is_null());
    fields.insert(
        "schema_version".to_string(),
        json!(envelope.schema_version.as_u16()),
    );
    fields.insert(
        "channel".to_string(),
        Value::String(
            match envelope.channel {
                mesh_llm_events::logging::replay::ReplayChannel::Requests => "requests",
                mesh_llm_events::logging::replay::ReplayChannel::Operations => "operations",
                mesh_llm_events::logging::replay::ReplayChannel::System => "system",
            }
            .to_string(),
        ),
    );
    fields.insert("sequence".to_string(), json!(envelope.sequence));
    fields.insert(
        "request_id".to_string(),
        Value::String(envelope.request_id.as_uuid().to_string()),
    );
    fields.insert(
        "event_id".to_string(),
        Value::String(envelope.event_id.as_uuid().to_string()),
    );
    if !matches!(envelope.event, LifecycleEvent::AuditError { .. }) {
        fields.insert(
            "request_kind".to_string(),
            Value::String(envelope.presentation_request_kind().to_string()),
        );
        fields.insert(
            "source".to_string(),
            Value::String(envelope.presentation_source().to_string()),
        );
        for (key, value) in [
            ("route", envelope.presentation_route()),
            ("model", envelope.presentation_model()),
            ("provider", envelope.presentation_provider()),
            ("engine", envelope.presentation_engine()),
            ("method", envelope.presentation_method()),
        ] {
            if let Some(value) = value {
                fields.insert(key.to_string(), Value::String(value.to_owned()));
            }
        }
    }
    if let Some(tokens) = envelope.presentation_token_count() {
        fields.insert("tokens".to_string(), json!(tokens));
    }
    if let Some(outcome) = envelope.presentation_outcome() {
        fields.insert("outcome".to_string(), json!(outcome));
    }
    fields
}

fn insert_token_usage(fields: &mut Map<String, Value>, usage: Option<&TokenUsage>) {
    let Some(usage) = usage else {
        return;
    };
    if let Some(prompt_tokens) = usage.prompt_tokens {
        fields.insert("prompt_tokens".to_string(), json!(prompt_tokens));
    }
    if let Some(completion_tokens) = usage.completion_tokens {
        fields.insert("completion_tokens".to_string(), json!(completion_tokens));
    }
    if let Some(total_tokens) = usage.total_tokens {
        fields.insert("total_tokens".to_string(), json!(total_tokens));
    }
}
