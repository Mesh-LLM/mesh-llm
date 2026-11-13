//! Incrementally bounded replay of complete responses and quarantined stream
//! suffixes.
//!
//! Replay never exposes a partial lifecycle: accepted events stay in this
//! module's buffer until every event, including item-closing and terminal
//! events, fits. A rejection drops the whole buffer and its tentative sequence
//! numbers.

use std::io::{self, Write};

use serde_json::{Map, Value, json};

#[cfg(test)]
use super::obfuscation::RandomSource;
use super::{ResponsesStreamEvent, obfuscation::ObfuscationPolicy};
use crate::responses::{
    resource::{ResponseEnvelope, ResponseError, ResponseSnapshot, ResponseStatus},
    tools::MAX_TOOL_CALLS,
    turn::ResponsesTurn,
};

pub(super) const MAX_REPLAY_EVENT_BYTES: usize = 32 * 1024 * 1024;
pub(super) const MAX_REPLAY_EVENTS: usize = 8 * 1024;
const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub(super) struct ReplayLimits {
    pub(super) max_bytes: usize,
    pub(super) max_events: usize,
}

impl Default for ReplayLimits {
    fn default() -> Self {
        Self {
            max_bytes: MAX_REPLAY_EVENT_BYTES,
            max_events: MAX_REPLAY_EVENTS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReplayError {
    Budget,
    Obfuscation,
}

impl ReplayError {
    pub(super) const fn code(self) -> &'static str {
        match self {
            Self::Budget => "response_output_too_large",
            Self::Obfuscation => "stream_obfuscation_failed",
        }
    }

    pub(super) const fn message(self) -> &'static str {
        match self {
            Self::Budget => "upstream response exceeded the tool-call quarantine limit",
            Self::Obfuscation => {
                "stream event padding could not obtain operating-system randomness"
            }
        }
    }
}

/// An event batch whose accepted contents can never exceed its replay budget.
pub(super) struct EventBuffer {
    events: Vec<ResponsesStreamEvent>,
    next_sequence: u64,
    admission: Option<ReplayAdmission>,
    obfuscation: ObfuscationPolicy,
    #[cfg(test)]
    random_source: Option<RandomSource>,
}

impl EventBuffer {
    pub(super) fn unbounded(next_sequence: u64, obfuscation: ObfuscationPolicy) -> Self {
        Self {
            events: Vec::new(),
            next_sequence,
            admission: None,
            obfuscation,
            #[cfg(test)]
            random_source: None,
        }
    }

    pub(super) fn bounded(
        next_sequence: u64,
        limits: ReplayLimits,
        obfuscation: ObfuscationPolicy,
    ) -> Self {
        Self {
            events: Vec::new(),
            next_sequence,
            admission: Some(ReplayAdmission::new(limits)),
            obfuscation,
            #[cfg(test)]
            random_source: None,
        }
    }

    #[cfg(test)]
    pub(super) fn set_random_source(&mut self, random_source: RandomSource) {
        self.random_source = Some(random_source);
    }

    pub(super) fn emit(&mut self, event: &'static str, mut data: Value) -> Result<(), ReplayError> {
        data["type"] = Value::String(event.to_string());
        data["sequence_number"] = Value::from(self.next_sequence);
        #[cfg(test)]
        let obfuscation = match self.random_source {
            Some(random_source) => {
                self.obfuscation
                    .apply_with_random_source(event, &mut data, random_source)
            }
            None => self.obfuscation.apply(event, &mut data),
        };
        #[cfg(not(test))]
        let obfuscation = self.obfuscation.apply(event, &mut data);
        obfuscation.map_err(|_| ReplayError::Obfuscation)?;
        if let Some(admission) = &mut self.admission {
            admission.admit(&data)?;
        }
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.events.push(ResponsesStreamEvent { event, data });
        Ok(())
    }

    pub(super) fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    pub(super) fn events_mut(&mut self) -> &mut [ResponsesStreamEvent] {
        &mut self.events
    }

    pub(super) fn into_events(self) -> Vec<ResponsesStreamEvent> {
        self.events
    }
}

struct ReplayAdmission {
    limits: ReplayLimits,
    bytes: usize,
    events: usize,
}

impl ReplayAdmission {
    fn new(limits: ReplayLimits) -> Self {
        Self {
            limits,
            bytes: 0,
            events: 0,
        }
    }

    fn admit(&mut self, data: &Value) -> Result<(), ReplayError> {
        if self.events >= self.limits.max_events {
            return Err(ReplayError::Budget);
        }
        // The JSON value for one event necessarily exists before its exact
        // serialized size is known. That single event is the unavoidable
        // transient allocation bound; it is measured immediately and is never
        // retained when it would take the aggregate over budget.
        let event_bytes = serde_json::to_vec(data)
            .map_err(|_| ReplayError::Budget)?
            .len();
        let bytes = self
            .bytes
            .checked_add(event_bytes)
            .ok_or(ReplayError::Budget)?;
        if bytes > self.limits.max_bytes {
            return Err(ReplayError::Budget);
        }
        self.bytes = bytes;
        self.events += 1;
        Ok(())
    }
}

pub(super) fn stream_events_for_response(response: &Value) -> Vec<ResponsesStreamEvent> {
    stream_events_for_response_with_policy(
        response,
        ReplayLimits::default(),
        ObfuscationPolicy::enabled(),
    )
}

pub(super) fn stream_events_for_response_after_progress(
    response: &Value,
    next_sequence: u64,
) -> Vec<ResponsesStreamEvent> {
    match try_stream_events_for_response(
        response,
        ReplayLimits::default(),
        ObfuscationPolicy::enabled(),
        next_sequence,
        false,
    ) {
        Ok(events) => events,
        Err(error) => failed_replay_events(response, error, next_sequence, false),
    }
}

#[cfg(test)]
fn stream_events_for_response_with_limits(
    response: &Value,
    limits: ReplayLimits,
) -> Vec<ResponsesStreamEvent> {
    stream_events_for_response_with_policy(response, limits, ObfuscationPolicy::enabled())
}

fn stream_events_for_response_with_policy(
    response: &Value,
    limits: ReplayLimits,
    obfuscation: ObfuscationPolicy,
) -> Vec<ResponsesStreamEvent> {
    match try_stream_events_for_response(response, limits, obfuscation, 0, true) {
        Ok(events) => events,
        Err(error) => failed_replay_events(response, error, 0, true),
    }
}

pub(super) fn stream_events_for_chat_completion(
    turn: &ResponsesTurn,
    chat: &Value,
) -> Vec<ResponsesStreamEvent> {
    stream_events_for_chat_completion_from(turn, chat, ReplayLimits::default(), 0, true)
}

#[cfg(test)]
fn stream_events_for_chat_completion_with_limits(
    turn: &ResponsesTurn,
    chat: &Value,
    limits: ReplayLimits,
) -> Vec<ResponsesStreamEvent> {
    stream_events_for_chat_completion_from(turn, chat, limits, 0, true)
}

pub(super) fn stream_events_for_chat_completion_after_progress(
    turn: &ResponsesTurn,
    chat: &Value,
    next_sequence: u64,
) -> Vec<ResponsesStreamEvent> {
    stream_events_for_chat_completion_from(
        turn,
        chat,
        ReplayLimits::default(),
        next_sequence,
        false,
    )
}

fn stream_events_for_chat_completion_from(
    turn: &ResponsesTurn,
    chat: &Value,
    limits: ReplayLimits,
    next_sequence: u64,
    emit_created: bool,
) -> Vec<ResponsesStreamEvent> {
    let staged = turn.stage_chat_completion(chat);
    let mut events = match try_stream_events_for_response(
        staged.response(),
        limits,
        turn.obfuscation_policy(),
        next_sequence,
        emit_created,
    ) {
        Ok(events) => events,
        Err(error) => {
            return failed_replay_events(staged.response(), error, next_sequence, emit_created);
        }
    };
    let stored = staged.commit_storage();
    patch_response_store(&mut events, stored);
    events
}

fn try_stream_events_for_response(
    response: &Value,
    limits: ReplayLimits,
    obfuscation: ObfuscationPolicy,
    next_sequence: u64,
    emit_created: bool,
) -> Result<Vec<ResponsesStreamEvent>, ReplayError> {
    if !replay_response_is_valid(response) {
        return Err(ReplayError::Budget);
    }
    let mut events = EventBuffer::bounded(next_sequence, limits, obfuscation);
    let in_progress = in_progress_response(response).ok_or(ReplayError::Budget)?;
    if emit_created {
        events.emit("response.created", json!({"response": in_progress.clone()}))?;
    }
    events.emit("response.in_progress", json!({"response": in_progress}))?;
    let output = response
        .get("output")
        .and_then(Value::as_array)
        .ok_or(ReplayError::Budget)?;
    for (index, item) in output.iter().enumerate() {
        replay_item(&mut events, index, item)?;
    }
    let terminal = match response.get("status").and_then(Value::as_str) {
        Some("incomplete") => "response.incomplete",
        Some("failed") => "response.failed",
        _ => "response.completed",
    };
    events.emit(terminal, json!({"response": response}))?;
    Ok(events.into_events())
}

fn in_progress_response(response: &Value) -> Option<Value> {
    let response = response.as_object()?;
    let mut in_progress = Map::with_capacity(response.len());
    for (key, value) in response {
        if !matches!(
            key.as_str(),
            "output"
                | "completed_at"
                | "status"
                | "incomplete_details"
                | "error"
                | "usage"
                | "output_text"
        ) {
            in_progress.insert(key.clone(), value.clone());
        }
    }
    in_progress.insert("completed_at".into(), Value::Null);
    in_progress.insert("status".into(), Value::String("in_progress".into()));
    in_progress.insert("incomplete_details".into(), Value::Null);
    in_progress.insert("output".into(), Value::Array(Vec::new()));
    in_progress.insert("error".into(), Value::Null);
    in_progress.insert("usage".into(), Value::Null);
    in_progress.insert("output_text".into(), Value::String(String::new()));
    Some(Value::Object(in_progress))
}

fn replay_response_is_valid(response: &Value) -> bool {
    if !serialized_size_within(response, MAX_RESPONSE_BYTES) {
        return false;
    }
    let Some(output) = response.get("output").and_then(Value::as_array) else {
        return false;
    };
    let mut calls = 0usize;
    for item in output {
        if item.get("type").and_then(Value::as_str) != Some("function_call") {
            continue;
        }
        calls += 1;
        if calls > MAX_TOOL_CALLS
            || item
                .get("id")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            || item
                .get("call_id")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            || item
                .get("name")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            || !item.get("arguments").is_some_and(Value::is_string)
        {
            return false;
        }
    }
    true
}

fn serialized_size_within(value: &Value, max_bytes: usize) -> bool {
    struct LimitedWriter {
        remaining: usize,
    }

    impl Write for LimitedWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > self.remaining {
                return Err(io::Error::other("serialized value exceeds replay limit"));
            }
            self.remaining -= bytes.len();
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    serde_json::to_writer(
        LimitedWriter {
            remaining: max_bytes,
        },
        value,
    )
    .is_ok()
}

fn failed_replay_events(
    response: &Value,
    error: ReplayError,
    next_sequence: u64,
    emit_created: bool,
) -> Vec<ResponsesStreamEvent> {
    const MAX_IDENTITY_BYTES: usize = 256;
    let bounded_string = |field: &str, fallback: &str| {
        response
            .get(field)
            .and_then(Value::as_str)
            .filter(|value| value.len() <= MAX_IDENTITY_BYTES)
            .unwrap_or(fallback)
            .to_string()
    };
    let id = bounded_string("id", "resp_replay_failed");
    let model = bounded_string("model", "");
    let created_at = response
        .get("created_at")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let mut envelope = ResponseEnvelope::default();
    envelope.set_store(false);
    let in_progress = envelope.response(ResponseSnapshot {
        id: &id,
        created_at,
        model: &model,
        status: ResponseStatus::InProgress,
        output: &[],
        usage: None,
        error: None,
    });
    let failed = envelope.response(ResponseSnapshot {
        id: &id,
        created_at,
        model: &model,
        status: ResponseStatus::Failed,
        output: &[],
        usage: None,
        error: Some(ResponseError {
            code: match error {
                ReplayError::Budget => "upstream_response_invalid",
                ReplayError::Obfuscation => "stream_obfuscation_failed",
            },
            message: match error {
                ReplayError::Budget => "buffered response could not be replayed safely",
                ReplayError::Obfuscation => {
                    "stream event padding could not obtain operating-system randomness"
                }
            },
        }),
    });
    let mut events = EventBuffer::unbounded(next_sequence, ObfuscationPolicy::enabled());
    if emit_created {
        events
            .emit("response.created", json!({"response": in_progress}))
            .expect("unbounded event buffer accepts response.created");
    }
    events
        .emit(
            "error",
            json!({"error": {
                "type": "server_error",
                "code": match error {
                    ReplayError::Budget => "upstream_response_invalid",
                    ReplayError::Obfuscation => "stream_obfuscation_failed",
                },
                "message": match error {
                    ReplayError::Budget => "buffered response could not be replayed safely",
                    ReplayError::Obfuscation => "stream event padding could not obtain operating-system randomness",
                },
                "param": null
            }}),
        )
        .expect("unbounded event buffer accepts error");
    events
        .emit("response.failed", json!({"response": failed}))
        .expect("unbounded event buffer accepts response.failed");
    events.into_events()
}

fn patch_response_store(events: &mut [ResponsesStreamEvent], stored: bool) {
    for event in events {
        if let Some(response) = event.data.get_mut("response") {
            response["store"] = Value::Bool(stored);
        }
    }
}

fn replay_item(events: &mut EventBuffer, index: usize, item: &Value) -> Result<(), ReplayError> {
    let id = item.get("id").cloned().unwrap_or(Value::Null);
    let kind = item.get("type").and_then(Value::as_str);
    let mut added = clone_object_without(item, &["content", "arguments"]);
    match kind {
        Some("message") | Some("reasoning") => {
            added["content"] = Value::Array(Vec::new());
            if kind == Some("message") {
                added["status"] = Value::String("in_progress".into());
            }
        }
        Some("function_call") => {
            added["status"] = Value::String("in_progress".into());
            added["arguments"] = Value::String(String::new());
        }
        _ => {}
    }
    events.emit(
        "response.output_item.added",
        json!({"output_index": index, "item": added}),
    )?;
    let parts = item
        .get("content")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for (content_index, part) in parts.iter().enumerate() {
        replay_content_part(events, index, content_index, &id, part)?;
    }
    if kind == Some("function_call") {
        replay_function_call(events, index, &id, item)?;
    }
    events.emit(
        "response.output_item.done",
        json!({"output_index": index, "item": item}),
    )
}

fn replay_content_part(
    events: &mut EventBuffer,
    output_index: usize,
    content_index: usize,
    item_id: &Value,
    part: &Value,
) -> Result<(), ReplayError> {
    let text = part.get("text").and_then(Value::as_str).unwrap_or_default();
    let (delta, done) = match part.get("type").and_then(Value::as_str) {
        Some("reasoning_text") => ("response.reasoning.delta", "response.reasoning.done"),
        Some("output_text") => ("response.output_text.delta", "response.output_text.done"),
        _ => return Ok(()),
    };
    let mut empty = clone_object_without(part, &["text"]);
    empty["text"] = Value::String(String::new());
    let location = |extra: Value| {
        let mut body = json!({
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index
        });
        for (key, value) in extra.as_object().into_iter().flatten() {
            body[key] = value.clone();
        }
        body
    };
    events.emit(
        "response.content_part.added",
        location(json!({"part": empty})),
    )?;
    if !text.is_empty() {
        events.emit(delta, location(json!({"delta": text})))?;
    }
    events.emit(done, location(json!({"text": text})))?;
    events.emit(
        "response.content_part.done",
        location(json!({"part": part})),
    )
}

fn clone_object_without(value: &Value, excluded: &[&str]) -> Value {
    let Some(object) = value.as_object() else {
        return value.clone();
    };
    Value::Object(
        object
            .iter()
            .filter(|(key, _)| !excluded.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    )
}

fn replay_function_call(
    events: &mut EventBuffer,
    output_index: usize,
    item_id: &Value,
    item: &Value,
) -> Result<(), ReplayError> {
    let arguments = item.get("arguments").cloned().unwrap_or(Value::from(""));
    if arguments
        .as_str()
        .is_some_and(|arguments| !arguments.is_empty())
    {
        events.emit(
            "response.function_call_arguments.delta",
            json!({"item_id": item_id, "output_index": output_index, "delta": arguments}),
        )?;
    }
    events.emit(
        "response.function_call_arguments.done",
        json!({"item_id": item_id, "output_index": output_index, "arguments": arguments}),
    )
}

#[cfg(test)]
mod tests;
