//! Validation for Responses request controls consumed before Chat translation.

use serde_json::{Map, Value};

use crate::errors::InferenceError;

/// Validate controls the Responses ingress reads, rejecting unsupported values
/// rather than silently ignoring them.
pub(super) fn validate_request_fields(object: &Map<String, Value>) -> Result<(), InferenceError> {
    optional_boolean(object, "store")?;
    if optional_boolean(object, "background")? == Some(true) {
        return Err(
            InferenceError::unsupported("background responses are not supported")
                .with_param("background"),
        );
    }
    match object.get("truncation") {
        None | Some(Value::Null) => {}
        Some(Value::String(mode)) if mode == "disabled" => {}
        Some(Value::String(mode)) if mode == "auto" => {
            return Err(
                InferenceError::unsupported("automatic input truncation is not supported")
                    .with_param("truncation"),
            );
        }
        Some(_) => {
            return Err(InferenceError::invalid_request(
                "truncation must be \"disabled\" or \"auto\"",
            )
            .with_param("truncation"));
        }
    }
    if object
        .get("text")
        .and_then(|text| text.get("verbosity"))
        .is_some_and(|value| !value.is_null())
    {
        return Err(
            InferenceError::unsupported("text verbosity is not supported")
                .with_param("text.verbosity"),
        );
    }
    if let Some(tier) = object.get("service_tier").filter(|value| !value.is_null())
        && !matches!(tier.as_str(), Some("auto" | "default"))
    {
        return Err(
            InferenceError::unsupported("only the default service tier is supported")
                .with_param("service_tier"),
        );
    }
    Ok(())
}

fn optional_boolean(
    object: &Map<String, Value>,
    field: &'static str,
) -> Result<Option<bool>, InferenceError> {
    match object.get(field) {
        None => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(
            InferenceError::invalid_request(format!("{field} must be true or false"))
                .with_param(field),
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::{Value, json};

    use crate::responses::{
        CallerScope, ResponseContinuation, ResponseStore, ResponsesTurn,
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

    #[test]
    fn shared_turn_rejects_non_boole_before_continuation_lookup() {
        let store = Arc::new(ResponseStore::default());
        for field in ["store", "background"] {
            for invalid in [json!(null), json!(0), json!("false"), json!({}), json!([])] {
                let mut body = json!({
                    "model": "m",
                    "input": "hi",
                    "previous_response_id": "resp_missing"
                });
                body[field] = invalid;
                let error = ResponsesTurn::prepare(&mut body, continuation(&store)).unwrap_err();

                assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
                assert_eq!(error.body().error.param.as_deref(), Some(field));
                assert_eq!(error.body().error.code.as_deref(), Some("invalid_value"));
            }
        }
    }

    #[test]
    fn public_normalization_rejects_non_boolean_retention_controls() {
        for field in ["store", "background"] {
            for invalid in [json!(null), json!(0), json!("false"), json!({}), json!([])] {
                let mut body = json!({"model": "qwen", "input": "hello"});
                body[field] = invalid;
                let error =
                    normalize_openai_compat_request("/v1/responses", &mut body).unwrap_err();

                assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
                assert_eq!(error.body().error.param.as_deref(), Some(field));
                assert_eq!(error.body().error.code.as_deref(), Some("invalid_value"));
            }
        }
    }

    #[test]
    fn store_boole_and_absence_keep_their_retention_semantics() {
        for (store_value, expected_stored) in [
            (None, true),
            (Some(Value::Bool(true)), true),
            (Some(Value::Bool(false)), false),
        ] {
            let store = Arc::new(ResponseStore::default());
            let mut body = json!({"model": "m", "input": "hi", "background": false});
            if let Some(store_value) = store_value {
                body["store"] = store_value;
            }
            let turn = ResponsesTurn::prepare(&mut body, continuation(&store)).unwrap();
            let response = turn.finish_chat_completion(&completed_chat("answer"));

            assert_eq!(response["store"], expected_stored);
            assert_eq!(
                store
                    .get(&CallerScope::LocalHost, response["id"].as_str().unwrap())
                    .is_ok(),
                expected_stored
            );
        }
    }
}
