//! Ordered lineage validation for typed Responses function-call items.

use std::collections::HashSet;

use serde_json::Value;

use crate::InferenceError;

pub(super) fn validate(items: &[Value]) -> Result<(), InferenceError> {
    let mut seen = HashSet::new();
    let mut pending = HashSet::new();
    for item in items {
        match item.get("type").and_then(Value::as_str) {
            Some("function_call") => {
                let call_id = required_call_id(item, "function_call")?;
                if !seen.insert(call_id) {
                    return Err(lineage_error(format!(
                        "function_call has duplicate call_id {call_id:?}"
                    )));
                }
                pending.insert(call_id);
            }
            Some("function_call_output") => {
                let call_id = required_call_id(item, "function_call_output")?;
                if pending.remove(call_id) {
                    continue;
                }
                let message = if seen.contains(call_id) {
                    format!("function_call_output duplicates output for call_id {call_id:?}")
                } else {
                    format!("function_call_output has no prior unresolved call_id {call_id:?}")
                };
                return Err(lineage_error(message));
            }
            _ => {}
        }
    }
    Ok(())
}

fn required_call_id<'a>(item: &'a Value, item_type: &str) -> Result<&'a str, InferenceError> {
    item.get("call_id")
        .and_then(Value::as_str)
        .filter(|call_id| !call_id.is_empty())
        .ok_or_else(|| lineage_error(format!("{item_type} requires a non-empty call_id")))
}

fn lineage_error(message: impl Into<String>) -> InferenceError {
    InferenceError::invalid_request(message).with_param("input")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::*;
    use crate::responses::{
        ResponseContinuation, ResponsesSession, ResponsesSessionConfig, ResponsesTurn,
        store::{CallerScope, ResponseStore},
    };

    fn error(items: Value) -> InferenceError {
        validate(items.as_array().unwrap()).unwrap_err()
    }

    fn assert_input_error(error: &InferenceError) {
        assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
        let body = error.body().error;
        assert_eq!(body.error_type, "invalid_request_error");
        assert_eq!(body.code.as_deref(), Some("invalid_value"));
        assert_eq!(body.param.as_deref(), Some("input"));
    }

    #[test]
    fn rejects_orphan_and_duplicate_outputs() {
        let orphan = error(json!([
            {"type":"function_call_output","call_id":"call_a","output":"orphan"}
        ]));
        assert_input_error(&orphan);
        assert!(orphan.message().contains("no prior unresolved"));

        let duplicate = error(json!([
            {"type":"function_call","call_id":"call_a","name":"same","arguments":"{}"},
            {"type":"function_call_output","call_id":"call_a","output":"first"},
            {"type":"function_call_output","call_id":"call_a","output":"second"}
        ]));
        assert_input_error(&duplicate);
        assert!(duplicate.message().contains("duplicates output"));
    }

    #[test]
    fn rejects_duplicate_call_ids_before_or_after_resolution() {
        for items in [
            json!([
                {"type":"function_call","call_id":"call_a"},
                {"type":"function_call","call_id":"call_a"}
            ]),
            json!([
                {"type":"function_call","call_id":"call_a"},
                {"type":"function_call_output","call_id":"call_a"},
                {"type":"function_call","call_id":"call_a"}
            ]),
        ] {
            let duplicate = error(items);
            assert_input_error(&duplicate);
            assert!(duplicate.message().contains("duplicate call_id"));
        }
    }

    #[test]
    fn requires_nonempty_typed_call_ids_without_treating_item_id_as_an_alias() {
        for items in [
            json!([{"type":"function_call","call_id":""}]),
            json!([{"type":"function_call_output","call_id":""}]),
            json!([{"type":"function_call","id":"call_item_only"}]),
            json!([{"type":"function_call_output","id":"output_item_only"}]),
        ] {
            let missing = error(items);
            assert_input_error(&missing);
            assert!(missing.message().contains("non-empty call_id"));
        }
    }

    #[test]
    fn accepts_parallel_interleaved_partial_calls_and_repeated_names() {
        let items = json!([
            {"type":"message","role":"user","content":"check both"},
            {"type":"function_call","call_id":"call_a","name":"same"},
            {"type":"function_call","call_id":"call_b","name":"same"},
            {"type":"message","role":"assistant","content":"one result follows"},
            {"type":"function_call_output","call_id":"call_b","output":"done"},
            {"type":"reasoning","summary":[]}
        ]);
        validate(items.as_array().unwrap()).unwrap();
    }

    #[test]
    fn stateless_and_restored_transcripts_are_validated_at_turn_preparation() {
        let mut stateless = json!({
            "model":"m",
            "input":[{"type":"function_call_output","call_id":"orphan","output":"x"}]
        });
        let error =
            ResponsesTurn::prepare(&mut stateless, ResponseContinuation::disabled()).unwrap_err();
        assert_input_error(&error);

        let store = Arc::new(ResponseStore::default());
        let continuation =
            || ResponseContinuation::new(Some(store.clone()), Some(CallerScope::LocalHost));
        let mut parent_body = json!({
            "model":"m",
            "input":[
                {"type":"function_call","call_id":"call_a","name":"same","arguments":"{}"},
                {"type":"function_call_output","call_id":"call_a","output":"first"}
            ]
        });
        let parent = ResponsesTurn::prepare(&mut parent_body, continuation())
            .unwrap()
            .finish_chat_completion(&json!({
                "choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}
            }));
        let mut child = json!({
            "previous_response_id":parent["id"],
            "input":[{"type":"function_call_output","call_id":"call_a","output":"second"}]
        });
        let error = ResponsesTurn::prepare(&mut child, continuation()).unwrap_err();
        assert_input_error(&error);
        assert!(error.message().contains("duplicates output"));
    }

    #[test]
    fn failed_session_preparation_evicts_local_state_but_not_shared_storage() {
        let store = Arc::new(ResponseStore::default());
        let session = ResponsesSession::new(
            ResponsesSessionConfig::default(),
            Some(store),
            Some(CallerScope::LocalHost),
        );
        let parent = session
            .prepare_create(
                json!({
                    "type":"response.create",
                    "model":"m",
                    "input":[
                        {"type":"message","role":"user","content":"use the tool"},
                        {"type":"function_call","call_id":"call_parent","name":"same","arguments":"{}"}
                    ]
                }),
                None,
            )
            .unwrap();
        let parent_id = parent.turn().id().to_string();
        let response = parent.turn().finish_chat_completion(&json!({
            "choices":[{"message":{"role":"assistant","content":"waiting"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}
        }));
        assert_eq!(response["store"], true);

        let orphan = session
            .prepare_create(
                json!({
                    "type":"response.create",
                    "previous_response_id":parent_id,
                    "input":[{"type":"function_call_output","call_id":"call_orphan","output":"x"}]
                }),
                None,
            )
            .unwrap_err();
        assert_input_error(&orphan);

        session
            .prepare_create(
                json!({
                    "type":"response.create",
                    "previous_response_id":parent_id,
                    "input":[{"type":"function_call_output","call_id":"call_parent","output":"ok"}]
                }),
                None,
            )
            .expect("the shared parent survives local eviction");
    }
}
