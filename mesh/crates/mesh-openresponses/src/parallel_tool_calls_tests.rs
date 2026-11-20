use std::sync::Arc;

use serde_json::{Value, json};

use super::{
    ResponseContinuation, ResponsesTurn, normalize_openai_compat_request,
    store::{CallerScope, ResponseStore},
};

fn continuation(store: &Arc<ResponseStore>) -> ResponseContinuation {
    ResponseContinuation::new(Some(store.clone()), Some(CallerScope::LocalHost))
}

fn tool_chat(turn: &str, call_count: usize) -> Value {
    let calls = (0..call_count)
        .map(|index| {
            json!({
                "id": format!("call_{turn}_{index}"),
                "type": "function",
                "function": {"name": "same", "arguments": "{}"},
            })
        })
        .collect::<Vec<_>>();
    json!({
        "choices": [{
            "message": {"role": "assistant", "content": null, "tool_calls": calls},
            "finish_reason": if call_count == 0 { "stop" } else { "tool_calls" },
        }],
    })
}

fn request(parallel_tool_calls: Value) -> Value {
    json!({
        "model": "m",
        "input": "call it",
        "tools": [{"type": "function", "name": "same"}],
        "parallel_tool_calls": parallel_tool_calls,
    })
}

#[test]
fn prepared_turn_preserves_backend_hint_and_echoes_effective_boolean() {
    for (requested, expected) in [
        (json!(false), false),
        (json!(true), true),
        (Value::Null, true),
    ] {
        let mut body = request(requested.clone());
        let turn = ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).unwrap();
        turn.normalize_backend_request("/v1/responses", &mut body)
            .unwrap();
        assert_eq!(body["parallel_tool_calls"], requested);
        let response = turn.response(super::ResponseStatus::InProgress, &[], None, None);
        assert_eq!(response["parallel_tool_calls"], expected);
    }
}

#[test]
fn prepared_normalization_requires_the_captured_serial_policy() {
    let mut body = request(json!(true));
    let turn = ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).unwrap();
    body["parallel_tool_calls"] = json!(false);
    let error = turn
        .normalize_backend_request("/v1/responses", &mut body)
        .unwrap_err();
    assert_eq!(
        error.body().error.param.as_deref(),
        Some("parallel_tool_calls")
    );

    let mut body = request(json!(false));
    let turn = ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).unwrap();
    body["parallel_tool_calls"] = json!(true);
    turn.normalize_backend_request("/v1/responses", &mut body)
        .unwrap();
    let response = turn.finish_chat_completion(&tool_chat("mutated", 2));
    assert_eq!(response["status"], "failed");
    assert_eq!(response["error"]["code"], "tool_call_limit_exceeded");
}

#[test]
fn serial_policy_accepts_zero_or_one_call_and_fails_two_without_storage() {
    let store = Arc::new(ResponseStore::default());
    for call_count in 0..=1 {
        let mut body = request(json!(false));
        let turn = ResponsesTurn::prepare(&mut body, continuation(&store)).unwrap();
        let response = turn.finish_chat_completion(&tool_chat("valid", call_count));
        assert_eq!(response["status"], "completed");
        assert_eq!(
            response["output"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|item| item["type"] == "function_call")
                .count(),
            call_count
        );
        assert_eq!(response["parallel_tool_calls"], false);
    }

    let mut body = request(json!(false));
    let turn = ResponsesTurn::prepare(&mut body, continuation(&store)).unwrap();
    let id = turn.id().to_string();
    let response = turn.finish_chat_completion(&tool_chat("excess", 2));
    assert_eq!(response["status"], "failed");
    assert_eq!(response["error"]["code"], "tool_call_limit_exceeded");
    assert_eq!(response["output"], json!([]));
    assert_eq!(response["store"], false);
    assert!(store.get(&CallerScope::LocalHost, &id).is_err());
}

#[test]
fn true_null_and_absent_allow_multiple_current_calls() {
    for requested in [None, Some(json!(true)), Some(Value::Null)] {
        let mut body = json!({
            "model": "m",
            "input": "call it",
            "tools": [{"type": "function", "name": "same"}],
        });
        if let Some(requested) = requested {
            body["parallel_tool_calls"] = requested;
        }
        let turn = ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).unwrap();
        let response = turn.finish_chat_completion(&tool_chat("parallel", 2));
        assert_eq!(response["status"], "completed");
        assert_eq!(response["output"].as_array().unwrap().len(), 2);
        assert_eq!(response["parallel_tool_calls"], true);
    }
}

#[test]
fn continuation_inherits_false_null_clears_it_and_each_turn_resets() {
    let store = Arc::new(ResponseStore::default());
    let mut parent_body = request(json!(false));
    let parent = ResponsesTurn::prepare(&mut parent_body, continuation(&store)).unwrap();
    let parent_response = parent.finish_chat_completion(&tool_chat("parent", 1));
    assert_eq!(parent_response["status"], "completed");

    let mut child_body = json!({
        "previous_response_id": parent_response["id"],
        "input": "call it again",
    });
    let child = ResponsesTurn::prepare(&mut child_body, continuation(&store)).unwrap();
    assert_eq!(child_body["parallel_tool_calls"], false);
    let child_response = child.finish_chat_completion(&tool_chat("child", 1));
    assert_eq!(child_response["status"], "completed");

    let mut cleared_body = json!({
        "previous_response_id": child_response["id"],
        "input": "two calls now",
        "parallel_tool_calls": null,
    });
    let cleared = ResponsesTurn::prepare(&mut cleared_body, continuation(&store)).unwrap();
    let cleared_response = cleared.finish_chat_completion(&tool_chat("cleared", 2));
    assert_eq!(cleared_response["status"], "completed");
    assert_eq!(cleared_response["parallel_tool_calls"], true);

    let mut inherited_null_body = json!({
        "previous_response_id": cleared_response["id"],
        "input": "still parallel",
    });
    let inherited_null =
        ResponsesTurn::prepare(&mut inherited_null_body, continuation(&store)).unwrap();
    assert!(inherited_null_body["parallel_tool_calls"].is_null());
    let inherited_null_response =
        inherited_null.finish_chat_completion(&tool_chat("inherited_null", 2));
    assert_eq!(inherited_null_response["status"], "completed");
    assert_eq!(inherited_null_response["parallel_tool_calls"], true);
}

#[test]
fn historical_input_calls_do_not_count_toward_current_response() {
    let mut body = json!({
        "model": "m",
        "parallel_tool_calls": false,
        "tools": [{"type": "function", "name": "same"}],
        "input": [
            {"type":"message","role":"user","content":"two earlier calls"},
            {"type":"function_call","call_id":"old_0","name":"same","arguments":"{}"},
            {"type":"function_call","call_id":"old_1","name":"same","arguments":"{}"},
            {"type":"function_call_output","call_id":"old_0","output":"a"},
            {"type":"function_call_output","call_id":"old_1","output":"b"},
            {"type":"message","role":"user","content":"one new call"}
        ],
    });
    let turn = ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).unwrap();
    let response = turn.finish_chat_completion(&tool_chat("current", 1));
    assert_eq!(response["status"], "completed");
    assert_eq!(response["output"].as_array().unwrap().len(), 1);
}

#[test]
fn malformed_values_are_bad_requests_and_policy_free_normalization_rejects_false() {
    for invalid in [json!(0), json!("false"), json!([]), json!({})] {
        let mut body = request(invalid);
        let error =
            ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).unwrap_err();
        assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(
            error.body().error.param.as_deref(),
            Some("parallel_tool_calls")
        );
        assert_eq!(error.body().error.code.as_deref(), Some("invalid_value"));
    }

    let mut unprotected = request(json!(false));
    let error = normalize_openai_compat_request("/v1/responses", &mut unprotected).unwrap_err();
    assert_eq!(
        error.body().error.param.as_deref(),
        Some("parallel_tool_calls")
    );
    assert_eq!(
        error.body().error.code.as_deref(),
        Some("unsupported_model_feature")
    );

    for safe in [json!(true), Value::Null] {
        let mut body = request(safe);
        normalize_openai_compat_request("/v1/responses", &mut body).unwrap();
    }
}
