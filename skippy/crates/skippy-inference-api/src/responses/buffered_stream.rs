//! Buffered response-to-stream adapters.
//!
//! Virtual and whole-response producers use these entry points so replay,
//! final sequence assignment, padding admission, and storage commit share the
//! same ordering as live Responses streams.

use serde_json::Value;

use super::{ResponsesTurn, stream};

/// Expand a complete `/v1/responses` body into the events that stream it.
pub fn events_for_response(response: &Value) -> Vec<Value> {
    stream::stream_events_for_response(response)
        .into_iter()
        .map(|event| event.data)
        .collect()
}

/// Expand a complete Responses body after an already committed virtual-model
/// progress prelude.
pub fn events_for_response_after_progress(response: &Value, next_sequence: u64) -> Vec<Value> {
    stream::stream_events_for_response_after_progress(response, next_sequence)
        .into_iter()
        .map(|event| event.data)
        .collect()
}

/// Translate a whole Chat Completions body and expand it into Responses
/// events, committing `turn` only after the complete replay is admitted.
pub fn events_for_chat_completion(turn: &ResponsesTurn, chat: &Value) -> Vec<Value> {
    stream::stream_events_for_chat_completion(turn, chat)
        .into_iter()
        .map(|event| event.data)
        .collect()
}

/// Translate and replay a buffered Chat Completions body after an already
/// committed virtual-model Responses progress prelude.
pub fn events_for_chat_completion_after_progress(
    turn: &ResponsesTurn,
    chat: &Value,
    next_sequence: u64,
) -> Vec<Value> {
    stream::stream_events_for_chat_completion_after_progress(turn, chat, next_sequence)
        .into_iter()
        .map(|event| event.data)
        .collect()
}
