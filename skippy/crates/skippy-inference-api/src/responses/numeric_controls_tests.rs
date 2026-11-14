use std::sync::Arc;

use serde_json::{Map, Value, json};

use super::store::StoredResponse;
use super::{
    CallerScope, ResponseContinuation, ResponseStatus, ResponseStore, ResponsesTurn,
    normalize_openai_compat_request,
};

fn continuation(store: &Arc<ResponseStore>) -> ResponseContinuation {
    ResponseContinuation::new(Some(store.clone()), Some(CallerScope::LocalHost))
}

fn completed_chat(text: &str) -> Value {
    json!({
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    })
}

fn in_progress(turn: &ResponsesTurn) -> Value {
    turn.response(ResponseStatus::InProgress, &[], None, None)
}

#[test]
fn numeric_boundaries_translate_and_echo_the_effective_responses_controls() {
    for (max_output_tokens, top_logprobs) in [(16_u64, 0_u64), (u32::MAX.into(), 20)] {
        let mut body = json!({
            "model":"m",
            "input":"hi",
            "max_output_tokens":max_output_tokens,
            "top_logprobs":top_logprobs,
        });
        let turn = ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).unwrap();
        let response = in_progress(&turn);
        assert_eq!(response["max_output_tokens"], max_output_tokens);
        assert_eq!(response["top_logprobs"], top_logprobs);

        normalize_openai_compat_request("/v1/responses", &mut body).unwrap();
        assert_eq!(body["max_tokens"], max_output_tokens);
        assert_eq!(body["top_logprobs"], top_logprobs);
        assert_eq!(body["logprobs"], true);
    }

    let mut cleared = json!({
        "model":"m",
        "input":"hi",
        "max_output_tokens":null,
        "top_logprobs":null,
    });
    let turn = ResponsesTurn::prepare(&mut cleared, ResponseContinuation::disabled()).unwrap();
    let response = in_progress(&turn);
    assert!(response["max_output_tokens"].is_null());
    assert_eq!(response["top_logprobs"], 0);
    normalize_openai_compat_request("/v1/responses", &mut cleared).unwrap();
    assert!(cleared["max_tokens"].is_null());
    assert!(cleared["top_logprobs"].is_null());
    assert!(cleared.get("logprobs").is_none());

    let mut included = json!({
        "model":"m",
        "input":"hi",
        "top_logprobs":null,
        "include":["message.output_text.logprobs"],
    });
    normalize_openai_compat_request("/v1/responses", &mut included).unwrap();
    assert!(included["top_logprobs"].is_null());
    assert_eq!(included["logprobs"], true);
}

#[test]
fn legacy_output_aliases_are_canonical_before_envelope_capture() {
    for alias in ["max_tokens", "max_completion_tokens", "n_predict"] {
        let mut body = json!({"model":"m", "input":"hi"});
        body[alias] = json!(32);
        let turn = ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).unwrap();

        assert_eq!(body["max_output_tokens"], 32);
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("max_completion_tokens").is_none());
        assert!(body.get("n_predict").is_none());
        assert_eq!(in_progress(&turn)["max_output_tokens"], 32);

        normalize_openai_compat_request("/v1/responses", &mut body).unwrap();
        assert_eq!(body["max_tokens"], 32);
        assert!(body.get("max_output_tokens").is_none());
        assert!(body.get("max_completion_tokens").is_none());
        assert!(body.get("n_predict").is_none());
    }
}

#[test]
fn canonical_output_limit_wins_by_presence_including_null() {
    let mut canonical = json!({
        "model":"m",
        "input":"hi",
        "max_output_tokens":16,
        "max_tokens":32,
        "max_completion_tokens":64,
        "n_predict":128,
    });
    let turn = ResponsesTurn::prepare(&mut canonical, ResponseContinuation::disabled()).unwrap();
    assert_eq!(in_progress(&turn)["max_output_tokens"], 16);
    normalize_openai_compat_request("/v1/responses", &mut canonical).unwrap();
    assert_eq!(canonical["max_tokens"], 16);

    let mut null = json!({
        "model":"m",
        "input":"hi",
        "max_output_tokens":null,
        "max_tokens":32,
        "max_completion_tokens":64,
        "n_predict":128,
    });
    let turn = ResponsesTurn::prepare(&mut null, ResponseContinuation::disabled()).unwrap();
    assert!(in_progress(&turn)["max_output_tokens"].is_null());
    normalize_openai_compat_request("/v1/responses", &mut null).unwrap();
    assert!(null["max_tokens"].is_null());

    let mut aliases = json!({
        "model":"m",
        "input":"hi",
        "max_tokens":32,
        "max_completion_tokens":64,
        "n_predict":128,
    });
    let turn = ResponsesTurn::prepare(&mut aliases, ResponseContinuation::disabled()).unwrap();
    assert_eq!(in_progress(&turn)["max_output_tokens"], 32);

    let mut completion_alias = json!({
        "model":"m",
        "input":"hi",
        "max_completion_tokens":64,
        "n_predict":128,
    });
    let turn =
        ResponsesTurn::prepare(&mut completion_alias, ResponseContinuation::disabled()).unwrap();
    assert_eq!(in_progress(&turn)["max_output_tokens"], 64);
}

#[test]
fn numeric_controls_inherit_and_children_override_or_clear_them() {
    let store = Arc::new(ResponseStore::default());
    let mut parent = json!({
        "model":"m",
        "input":"first",
        "max_tokens":32,
        "top_logprobs":20,
    });
    let parent_turn = ResponsesTurn::prepare(&mut parent, continuation(&store)).unwrap();
    let parent_response = parent_turn.finish_chat_completion(&completed_chat("answer"));

    let mut inherited = json!({
        "previous_response_id":parent_response["id"],
        "input":"next",
    });
    let inherited_turn = ResponsesTurn::prepare(&mut inherited, continuation(&store)).unwrap();
    let inherited_response = in_progress(&inherited_turn);
    assert_eq!(inherited_response["max_output_tokens"], 32);
    assert_eq!(inherited_response["top_logprobs"], 20);
    normalize_openai_compat_request("/v1/responses", &mut inherited).unwrap();
    assert_eq!(inherited["max_tokens"], 32);
    assert_eq!(inherited["top_logprobs"], 20);
    assert_eq!(inherited["logprobs"], true);

    let mut alias_override = json!({
        "previous_response_id":parent_response["id"],
        "input":"next",
        "max_completion_tokens":64,
    });
    let override_turn = ResponsesTurn::prepare(&mut alias_override, continuation(&store)).unwrap();
    assert_eq!(in_progress(&override_turn)["max_output_tokens"], 64);
    normalize_openai_compat_request("/v1/responses", &mut alias_override).unwrap();
    assert_eq!(alias_override["max_tokens"], 64);

    let mut alias_clear = json!({
        "previous_response_id":parent_response["id"],
        "input":"next",
        "max_tokens":null,
        "top_logprobs":null,
    });
    let cleared_turn = ResponsesTurn::prepare(&mut alias_clear, continuation(&store)).unwrap();
    let cleared_response = in_progress(&cleared_turn);
    assert!(cleared_response["max_output_tokens"].is_null());
    assert_eq!(cleared_response["top_logprobs"], 0);
    normalize_openai_compat_request("/v1/responses", &mut alias_clear).unwrap();
    assert!(alias_clear["max_tokens"].is_null());
    assert!(alias_clear["top_logprobs"].is_null());
    assert!(alias_clear.get("logprobs").is_none());

    let mut n_predict_override = json!({
        "previous_response_id":parent_response["id"],
        "input":"next",
        "n_predict":96,
    });
    let override_turn =
        ResponsesTurn::prepare(&mut n_predict_override, continuation(&store)).unwrap();
    assert_eq!(in_progress(&override_turn)["max_output_tokens"], 96);
    normalize_openai_compat_request("/v1/responses", &mut n_predict_override).unwrap();
    assert_eq!(n_predict_override["max_tokens"], 96);
    assert!(n_predict_override.get("n_predict").is_none());

    let mut n_predict_clear = json!({
        "previous_response_id":parent_response["id"],
        "input":"next",
        "n_predict":null,
    });
    let cleared_turn = ResponsesTurn::prepare(&mut n_predict_clear, continuation(&store)).unwrap();
    assert!(in_progress(&cleared_turn)["max_output_tokens"].is_null());
    normalize_openai_compat_request("/v1/responses", &mut n_predict_clear).unwrap();
    assert!(n_predict_clear["max_tokens"].is_null());
    assert!(n_predict_clear.get("n_predict").is_none());

    let mut canonical_clear_with_shadow = json!({
        "previous_response_id":parent_response["id"],
        "input":"next",
        "max_output_tokens":null,
        "max_tokens":128,
    });
    let cleared_turn =
        ResponsesTurn::prepare(&mut canonical_clear_with_shadow, continuation(&store)).unwrap();
    assert!(in_progress(&cleared_turn)["max_output_tokens"].is_null());
    normalize_openai_compat_request("/v1/responses", &mut canonical_clear_with_shadow).unwrap();
    assert!(canonical_clear_with_shadow["max_tokens"].is_null());
}

#[test]
fn malformed_n_predict_is_rejected_only_when_it_is_the_effective_alias() {
    for value in [json!(15), json!("16"), json!({})] {
        let mut body = json!({"model":"m", "input":"hi", "n_predict":value});
        let error =
            ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).unwrap_err();
        assert_eq!(
            error.body().error.param.as_deref(),
            Some("max_output_tokens")
        );
        assert_eq!(error.body().error.code.as_deref(), Some("invalid_value"));
        assert!(body.get("n_predict").is_none());
    }

    let mut losing = json!({
        "model":"m",
        "input":"hi",
        "max_output_tokens":16,
        "max_tokens":4294967296_u64,
        "max_completion_tokens":"invalid",
        "n_predict":{},
    });
    let turn = ResponsesTurn::prepare(&mut losing, ResponseContinuation::disabled()).unwrap();
    assert_eq!(in_progress(&turn)["max_output_tokens"], 16);
    for alias in ["max_tokens", "max_completion_tokens", "n_predict"] {
        assert!(losing.get(alias).is_none(), "losing alias {alias} survived");
    }
}

#[test]
fn ordinary_chat_keeps_n_predict_unchanged() {
    let mut body = json!({
        "model":"m",
        "messages":[{"role":"user", "content":"hi"}],
        "n_predict":{},
    });
    normalize_openai_compat_request("/v1/chat/completions", &mut body).unwrap();
    assert_eq!(body["n_predict"], json!({}));
}

#[test]
fn inherited_numeric_controls_are_revalidated_against_current_bounds() {
    let store = Arc::new(ResponseStore::default());
    for (field, value, expected_code) in [
        ("max_output_tokens", json!(15), "invalid_value"),
        (
            "max_output_tokens",
            json!(u64::from(u32::MAX) + 1),
            "unsupported_model_feature",
        ),
        ("top_logprobs", json!(21), "invalid_value"),
    ] {
        let configuration = Map::from_iter([(field.to_string(), value)]);
        let id = format!("resp_legacy_{field}_{expected_code}");
        assert!(store.insert(
            CallerScope::LocalHost,
            StoredResponse::new(
                id.clone(),
                "m".into(),
                None,
                None,
                configuration,
                Vec::new(),
                Vec::new(),
            ),
        ));
        let mut inherited = json!({"previous_response_id":id, "input":"next"});
        let error = ResponsesTurn::prepare(&mut inherited, continuation(&store)).unwrap_err();
        assert_eq!(error.body().error.param.as_deref(), Some(field));
        assert_eq!(error.body().error.code.as_deref(), Some(expected_code));

        let mut cleared = json!({"previous_response_id":id, "input":"next"});
        cleared[field] = Value::Null;
        ResponsesTurn::prepare(&mut cleared, continuation(&store)).unwrap();
    }
}
