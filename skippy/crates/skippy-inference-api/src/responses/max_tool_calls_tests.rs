use std::sync::Arc;

use serde_json::{Value, json};

use super::{
    ResponseContinuation, ResponsesTurn, normalize_openai_compat_request,
    store::{CallerScope, ResponseStore},
};

fn continuation(store: &Arc<ResponseStore>) -> ResponseContinuation {
    ResponseContinuation::new(Some(store.clone()), Some(CallerScope::LocalHost))
}

fn tool_chat(call_count: usize) -> Value {
    let calls = (0..call_count)
        .map(|index| {
            json!({
                "id": format!("call_{index}"),
                "type": "function",
                "function": {"name": "same", "arguments": "{}"},
            })
        })
        .collect::<Vec<_>>();
    json!({
        "choices": [{
            "message": {"role": "assistant", "content": null, "tool_calls": calls},
            "finish_reason": "tool_calls",
        }],
    })
}

fn request(max_tool_calls: Value) -> Value {
    json!({
        "model": "m",
        "input": "call it",
        "tools": [{"type": "function", "name": "same"}],
        "max_tool_calls": max_tool_calls,
    })
}

#[test]
fn prepared_turn_strips_backend_field_and_echoes_original_value() {
    for value in [json!(1), json!(u64::MAX), Value::Null] {
        let mut body = request(value.clone());
        let turn = ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).unwrap();
        assert!(body.get("max_tool_calls").is_none());
        let response = turn.response(super::ResponseStatus::InProgress, &[], None, None);
        assert_eq!(response["max_tool_calls"], value);
    }
}

#[test]
fn json_exact_cap_completes_but_excess_fails_without_output_or_storage() {
    let store = Arc::new(ResponseStore::default());

    let mut exact_body = request(json!(2));
    let exact = ResponsesTurn::prepare(&mut exact_body, continuation(&store)).unwrap();
    let exact_response = exact.finish_chat_completion(&tool_chat(2));
    assert_eq!(exact_response["status"], "completed");
    assert_eq!(exact_response["output"].as_array().unwrap().len(), 2);
    assert_eq!(exact_response["max_tool_calls"], 2);

    let mut excess_body = request(json!(1));
    let excess = ResponsesTurn::prepare(&mut excess_body, continuation(&store)).unwrap();
    let excess_id = excess.id().to_string();
    let excess_response = excess.finish_chat_completion(&tool_chat(2));
    assert_eq!(excess_response["status"], "failed");
    assert_eq!(excess_response["error"]["code"], "tool_call_limit_exceeded");
    assert_eq!(excess_response["output"], json!([]));
    assert_eq!(excess_response["store"], false);
    assert!(store.get(&CallerScope::LocalHost, &excess_id).is_err());
}

#[test]
fn continuation_inherits_cap_null_clears_it_and_each_turn_resets_the_count() {
    let store = Arc::new(ResponseStore::default());
    let mut parent_body = request(json!(1));
    let parent = ResponsesTurn::prepare(&mut parent_body, continuation(&store)).unwrap();
    let parent_response = parent.finish_chat_completion(&tool_chat(1));
    assert_eq!(parent_response["status"], "completed");

    let mut inherited_body = json!({
        "previous_response_id": parent_response["id"],
        "input": "call it again",
    });
    let inherited = ResponsesTurn::prepare(&mut inherited_body, continuation(&store)).unwrap();
    assert!(inherited_body.get("max_tool_calls").is_none());
    let inherited_response = inherited.finish_chat_completion(&tool_chat(1));
    assert_eq!(inherited_response["status"], "completed");
    assert_eq!(inherited_response["max_tool_calls"], 1);

    let mut cleared_body = json!({
        "previous_response_id": inherited_response["id"],
        "input": "two calls now",
        "max_tool_calls": null,
    });
    let cleared = ResponsesTurn::prepare(&mut cleared_body, continuation(&store)).unwrap();
    let cleared_response = cleared.finish_chat_completion(&tool_chat(2));
    assert_eq!(cleared_response["status"], "completed");
    assert!(cleared_response["max_tool_calls"].is_null());

    let mut null_inherited_body = json!({
        "previous_response_id": cleared_response["id"],
        "input": "still unlimited",
    });
    let null_inherited =
        ResponsesTurn::prepare(&mut null_inherited_body, continuation(&store)).unwrap();
    let null_inherited_response = null_inherited.finish_chat_completion(&tool_chat(2));
    assert_eq!(null_inherited_response["status"], "completed");
    assert!(null_inherited_response["max_tool_calls"].is_null());
}

#[test]
fn low_level_normalizer_rejects_nonnull_cap_and_strips_null() {
    let mut unsupported = request(json!(1));
    let error = normalize_openai_compat_request("/v1/responses", &mut unsupported).unwrap_err();
    assert_eq!(error.body().error.param.as_deref(), Some("max_tool_calls"));
    assert_eq!(
        error.body().error.code.as_deref(),
        Some("unsupported_model_feature")
    );

    let mut nullable = request(Value::Null);
    normalize_openai_compat_request("/v1/responses", &mut nullable).unwrap();
    assert!(nullable.get("max_tool_calls").is_none());
}
