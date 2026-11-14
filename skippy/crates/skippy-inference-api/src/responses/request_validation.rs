//! Validation for Responses request controls consumed before Chat translation.

use serde_json::{Map, Value};

use crate::errors::InferenceError;

const MAX_METADATA_KEYS: usize = 16;
const MAX_METADATA_KEY_CHARS: usize = 64;
const MAX_METADATA_VALUE_CHARS: usize = 512;
const MAX_IDENTIFIER_CHARS: usize = 64;
const MIN_OUTPUT_TOKENS: u64 = 16;
const MAX_TOP_LOGPROBS: u64 = 20;

/// Validate controls the Responses ingress reads, rejecting unsupported values
/// rather than silently ignoring them.
pub(super) fn validate_request_fields(object: &Map<String, Value>) -> Result<(), InferenceError> {
    max_output_tokens(object)?;
    top_logprobs(object)?;
    nullable_string(object, "instructions")?;
    metadata(object)?;
    nullable_bounded_string(object, "safety_identifier", MAX_IDENTIFIER_CHARS)?;
    nullable_bounded_string(object, "prompt_cache_key", MAX_IDENTIFIER_CHARS)?;
    text(object)?;
    reasoning(object)?;
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

pub(super) fn canonicalize_and_validate_request_fields(
    object: &mut Map<String, Value>,
) -> Result<(), InferenceError> {
    canonicalize_max_output_tokens(object);
    validate_request_fields(object)
}

/// Normalize legacy Chat token-limit aliases onto the Responses field before
/// continuation inheritance and response-envelope capture.
///
/// Presence decides precedence, including an explicit `null`: the canonical
/// Responses field wins, followed by `max_tokens`, then
/// `max_completion_tokens`, then the legacy llama.cpp `n_predict` alias.
/// Removing the losing aliases prevents a shadow value from overriding or
/// resurrecting the effective limit later, and keeps released peers from
/// trying to deserialize a malformed losing alias before dispatch.
pub(super) fn canonicalize_max_output_tokens(object: &mut Map<String, Value>) -> bool {
    let canonical_present = object.contains_key("max_output_tokens");
    let max_tokens = object.remove("max_tokens");
    let max_completion_tokens = object.remove("max_completion_tokens");
    let n_predict = object.remove("n_predict");
    if canonical_present {
        return max_tokens.is_some() || max_completion_tokens.is_some() || n_predict.is_some();
    }
    if let Some(value) = max_tokens.or(max_completion_tokens).or(n_predict) {
        object.insert("max_output_tokens".to_string(), value);
        return true;
    }
    false
}

/// Translate the already-effective Responses limit to the shared Chat field.
pub(super) fn translate_max_output_tokens(object: &mut Map<String, Value>) -> bool {
    let Some(value) = object.remove("max_output_tokens") else {
        return false;
    };
    object.insert("max_tokens".to_string(), value);
    true
}

fn max_output_tokens(object: &Map<String, Value>) -> Result<(), InferenceError> {
    let Some(value) = object
        .get("max_output_tokens")
        .filter(|value| !value.is_null())
    else {
        return Ok(());
    };
    let Some(value) = value.as_u64() else {
        if value.as_f64().is_some_and(|value| {
            value.is_finite() && value.fract() == 0.0 && value > f64::from(u32::MAX)
        }) {
            return Err(unsupported_output_token_limit());
        }
        return Err(InferenceError::invalid_request(
            "max_output_tokens must be an integer greater than or equal to 16, or null",
        )
        .with_param("max_output_tokens"));
    };
    if value < MIN_OUTPUT_TOKENS {
        return Err(InferenceError::invalid_request(
            "max_output_tokens must be an integer greater than or equal to 16, or null",
        )
        .with_param("max_output_tokens"));
    }
    if value > u64::from(u32::MAX) {
        return Err(unsupported_output_token_limit());
    }
    Ok(())
}

fn unsupported_output_token_limit() -> InferenceError {
    InferenceError::unsupported(format!(
        "max_output_tokens values greater than {} are not supported by the generation backend",
        u32::MAX
    ))
    .with_param("max_output_tokens")
}

fn top_logprobs(object: &Map<String, Value>) -> Result<(), InferenceError> {
    let Some(value) = object.get("top_logprobs").filter(|value| !value.is_null()) else {
        return Ok(());
    };
    if value
        .as_u64()
        .is_some_and(|value| value <= MAX_TOP_LOGPROBS)
    {
        return Ok(());
    }
    Err(InferenceError::invalid_request(
        "top_logprobs must be an integer between 0 and 20, or null",
    )
    .with_param("top_logprobs"))
}

fn nullable_string(object: &Map<String, Value>, field: &'static str) -> Result<(), InferenceError> {
    match object.get(field) {
        None | Some(Value::Null | Value::String(_)) => Ok(()),
        Some(_) => Err(InferenceError::invalid_request(format!(
            "{field} must be a string or null"
        ))
        .with_param(field)),
    }
}

fn nullable_bounded_string(
    object: &Map<String, Value>,
    field: &'static str,
    max_chars: usize,
) -> Result<(), InferenceError> {
    nullable_string(object, field)?;
    if object
        .get(field)
        .and_then(Value::as_str)
        .is_some_and(|value| value.chars().count() > max_chars)
    {
        return Err(InferenceError::invalid_request(format!(
            "{field} must be at most {max_chars} characters"
        ))
        .with_param(field));
    }
    Ok(())
}

fn metadata(object: &Map<String, Value>) -> Result<(), InferenceError> {
    let Some(value) = object.get("metadata") else {
        return Ok(());
    };
    let Some(metadata) = value.as_object() else {
        return if value.is_null() {
            Ok(())
        } else {
            Err(
                InferenceError::invalid_request("metadata must be an object or null")
                    .with_param("metadata"),
            )
        };
    };
    if metadata.len() > MAX_METADATA_KEYS {
        return Err(InferenceError::invalid_request(format!(
            "metadata must contain at most {MAX_METADATA_KEYS} keys"
        ))
        .with_param("metadata"));
    }
    for (key, value) in metadata {
        if key.chars().count() > MAX_METADATA_KEY_CHARS {
            return Err(InferenceError::invalid_request(format!(
                "metadata keys must be at most {MAX_METADATA_KEY_CHARS} characters"
            ))
            .with_param("metadata"));
        }
        let Some(value) = value.as_str() else {
            return Err(
                InferenceError::invalid_request("metadata values must be strings")
                    .with_param("metadata"),
            );
        };
        if value.chars().count() > MAX_METADATA_VALUE_CHARS {
            return Err(InferenceError::invalid_request(format!(
                "metadata values must be at most {MAX_METADATA_VALUE_CHARS} characters"
            ))
            .with_param("metadata"));
        }
    }
    Ok(())
}

fn nullable_object<'a>(
    object: &'a Map<String, Value>,
    field: &'static str,
) -> Result<Option<&'a Map<String, Value>>, InferenceError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(value)) => Ok(Some(value)),
        Some(_) => Err(InferenceError::invalid_request(format!(
            "{field} must be an object or null"
        ))
        .with_param(field)),
    }
}

fn text(object: &Map<String, Value>) -> Result<(), InferenceError> {
    let Some(text) = nullable_object(object, "text")? else {
        return Ok(());
    };
    if text
        .get("format")
        .is_some_and(|value| !value.is_null() && !value.is_object())
    {
        return Err(
            InferenceError::invalid_request("text.format must be an object or null")
                .with_param("text.format"),
        );
    }
    if text
        .get("format")
        .and_then(Value::as_object)
        .and_then(|format| format.get("type"))
        .is_some_and(|kind| !kind.is_string())
    {
        return Err(
            InferenceError::invalid_request("text.format.type must be a string")
                .with_param("text.format.type"),
        );
    }
    if text.get("verbosity").is_some_and(|value| !value.is_null()) {
        return Err(
            InferenceError::unsupported("text verbosity is not supported")
                .with_param("text.verbosity"),
        );
    }
    Ok(())
}

fn reasoning(object: &Map<String, Value>) -> Result<(), InferenceError> {
    let Some(reasoning) = nullable_object(object, "reasoning")? else {
        return Ok(());
    };
    optional_reasoning_effort(reasoning)?;
    match reasoning.get("summary") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(summary))
            if matches!(summary.as_str(), "auto" | "concise" | "detailed") =>
        {
            Err(
                InferenceError::unsupported("reasoning summaries are not supported")
                    .with_param("reasoning.summary"),
            )
        }
        Some(_) => Err(InferenceError::invalid_request(
            "reasoning.summary must be one of auto, concise, detailed, or null",
        )
        .with_param("reasoning.summary")),
    }
}

fn optional_reasoning_effort(reasoning: &Map<String, Value>) -> Result<(), InferenceError> {
    match reasoning.get("effort") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(effort))
            if matches!(
                effort.as_str(),
                "none" | "low" | "medium" | "high" | "xhigh"
            ) =>
        {
            Ok(())
        }
        Some(_) => Err(InferenceError::invalid_request(
            "reasoning.effort must be one of none, low, medium, high, xhigh, or null",
        )
        .with_param("reasoning.effort")),
    }
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

    use serde_json::{Map, Value, json};

    use super::{MAX_METADATA_KEY_CHARS, MAX_METADATA_KEYS};
    use crate::responses::{
        CallerScope, ResponseContinuation, ResponseStore, ResponsesTurn,
        normalize_openai_compat_request, store::StoredResponse,
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

    fn assert_invalid(body: &mut Value, expected_param: &str) {
        let error = ResponsesTurn::prepare(body, ResponseContinuation::disabled()).unwrap_err();
        assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(error.body().error.param.as_deref(), Some(expected_param));
        assert_eq!(error.body().error.code.as_deref(), Some("invalid_value"));
    }

    #[test]
    fn nullable_shapes_and_schema_boundaries_are_accepted_and_echoed() {
        let unicode_64 = "🦀".repeat(64);
        let unicode_512 = "界".repeat(512);
        let mut metadata = Map::new();
        metadata.insert(unicode_64.clone(), Value::String(unicode_512));
        for index in 1..MAX_METADATA_KEYS {
            metadata.insert(
                format!("key-{index}"),
                Value::String(format!("value-{index}")),
            );
        }
        let mut body = json!({
            "model": "m",
            "input": "hi",
            "instructions": null,
            "metadata": metadata,
            "safety_identifier": unicode_64,
            "prompt_cache_key": "界".repeat(64),
            "text": {"format": null, "future_extension": true},
            "reasoning": {"effort": "xhigh", "summary": null, "future_extension": true},
            "unknown_extension": {"kept": true}
        });
        let turn = ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).unwrap();
        let response = turn.finish_chat_completion(&completed_chat("answer"));

        assert_eq!(
            response["metadata"].as_object().unwrap().len(),
            MAX_METADATA_KEYS
        );
        assert_eq!(
            response["safety_identifier"]
                .as_str()
                .unwrap()
                .chars()
                .count(),
            64
        );
        assert_eq!(
            response["prompt_cache_key"]
                .as_str()
                .unwrap()
                .chars()
                .count(),
            64
        );
        assert_eq!(response["reasoning"]["effort"], "xhigh");

        for field in [
            "instructions",
            "metadata",
            "safety_identifier",
            "prompt_cache_key",
            "text",
            "reasoning",
        ] {
            let mut nullable = json!({"model": "m", "input": "hi"});
            nullable[field] = Value::Null;
            ResponsesTurn::prepare(&mut nullable, ResponseContinuation::disabled()).unwrap();
        }
    }

    #[test]
    fn request_shape_and_character_bounds_fail_before_continuation_lookup() {
        let mut too_many_metadata = Map::new();
        for index in 0..=MAX_METADATA_KEYS {
            too_many_metadata.insert(format!("key-{index}"), json!("value"));
        }
        let overlong_key_metadata =
            Map::from_iter([("k".repeat(MAX_METADATA_KEY_CHARS + 1), json!("value"))]);
        let cases = [
            (json!({"instructions": []}), "instructions"),
            (json!({"metadata": []}), "metadata"),
            (json!({"metadata": {"key": 7}}), "metadata"),
            (json!({"metadata": overlong_key_metadata}), "metadata"),
            (json!({"metadata": {"key": "界".repeat(513)}}), "metadata"),
            (json!({"metadata": too_many_metadata}), "metadata"),
            (json!({"safety_identifier": 7}), "safety_identifier"),
            (
                json!({"safety_identifier": "🦀".repeat(65)}),
                "safety_identifier",
            ),
            (json!({"prompt_cache_key": []}), "prompt_cache_key"),
            (
                json!({"prompt_cache_key": "界".repeat(65)}),
                "prompt_cache_key",
            ),
            (json!({"text": "plain"}), "text"),
            (json!({"text": {"format": "plain"}}), "text.format"),
            (json!({"text": {"format": []}}), "text.format"),
            (json!({"text": {"format": 1}}), "text.format"),
            (json!({"text": {"format": {"type": 7}}}), "text.format.type"),
            (
                json!({"text": {"format": {"type": null}}}),
                "text.format.type",
            ),
            (
                json!({"text": {"format": {"type": []}}}),
                "text.format.type",
            ),
            (json!({"reasoning": []}), "reasoning"),
            (
                json!({"reasoning": {"effort": "minimal"}}),
                "reasoning.effort",
            ),
            (json!({"reasoning": {"effort": 7}}), "reasoning.effort"),
            (
                json!({"reasoning": {"summary": "brief"}}),
                "reasoning.summary",
            ),
            (json!({"reasoning": {"summary": 7}}), "reasoning.summary"),
        ];
        for (fields, expected_param) in cases {
            let mut body = json!({
                "model": "m",
                "input": "hi",
                "previous_response_id": "resp_missing"
            });
            body.as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            assert_invalid(&mut body, expected_param);
        }
    }

    #[test]
    fn schema_reasoning_efforts_are_accepted_and_summaries_are_explicitly_unsupported() {
        for effort in ["none", "low", "medium", "high", "xhigh"] {
            let mut body = json!({"model":"m", "input":"hi", "reasoning":{"effort":effort}});
            ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).unwrap();
        }
        for summary in ["auto", "concise", "detailed"] {
            let mut body = json!({"model":"m", "input":"hi", "reasoning":{"summary":summary}});
            let error =
                ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).unwrap_err();
            assert_eq!(
                error.body().error.param.as_deref(),
                Some("reasoning.summary")
            );
            assert_eq!(
                error.body().error.code.as_deref(),
                Some("unsupported_model_feature")
            );
        }
    }

    #[test]
    fn public_normalization_validates_raw_responses_but_not_ordinary_chat() {
        let mut responses = json!({"model":"m", "input":"hi", "instructions":[]});
        let error = normalize_openai_compat_request("/v1/responses", &mut responses).unwrap_err();
        assert_eq!(error.body().error.param.as_deref(), Some("instructions"));

        let mut chat = json!({
            "model":"m",
            "messages":[{"role":"user","content":"hi"}],
            "instructions":[]
        });
        normalize_openai_compat_request("/v1/chat/completions", &mut chat).unwrap();
        assert_eq!(chat["instructions"], json!([]));
    }

    #[test]
    fn stream_obfuscation_control_is_validated_and_stripped_only_for_responses() {
        for stream_options in [
            Value::Null,
            json!({}),
            json!({"include_obfuscation":true,"include_usage":false,"extension":"kept"}),
            json!({"include_obfuscation":false,"include_usage":true}),
        ] {
            let mut responses = json!({
                "model":"m",
                "input":"hi",
                "stream":true,
                "stream_options":stream_options,
            });
            normalize_openai_compat_request("/v1/responses", &mut responses).unwrap();
            let options = responses["stream_options"].as_object().unwrap();
            assert!(!options.contains_key("include_obfuscation"));
            assert_eq!(options.get("include_usage"), Some(&Value::Bool(true)));
            if options.get("extension").is_some() {
                assert_eq!(options["extension"], "kept");
            }
        }

        for (stream_options, param) in [
            (json!(false), "stream_options"),
            (json!([]), "stream_options"),
            (
                json!({"include_obfuscation":null}),
                "stream_options.include_obfuscation",
            ),
            (
                json!({"include_obfuscation":"false"}),
                "stream_options.include_obfuscation",
            ),
        ] {
            let mut responses = json!({
                "model":"m",
                "input":"hi",
                "previous_response_id":"resp_missing",
                "stream_options":stream_options,
            });
            let response_store = Arc::new(ResponseStore::default());
            let error =
                ResponsesTurn::prepare(&mut responses, continuation(&response_store)).unwrap_err();
            assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
            assert_eq!(error.body().error.param.as_deref(), Some(param));
        }

        let original = json!({"include_obfuscation":false,"extension":"chat"});
        let mut chat = json!({
            "model":"m",
            "messages":[{"role":"user","content":"hi"}],
            "stream_options":original,
        });
        normalize_openai_compat_request("/v1/chat/completions", &mut chat).unwrap();
        assert_eq!(chat["stream_options"], original);
    }

    #[test]
    fn inherited_configuration_is_revalidated_and_an_explicit_null_can_clear_it() {
        let store = Arc::new(ResponseStore::default());
        let configuration = Map::from_iter([("reasoning".to_string(), json!({"summary":"auto"}))]);
        assert!(store.insert(
            CallerScope::LocalHost,
            StoredResponse::new(
                "resp_legacy".into(),
                "m".into(),
                None,
                None,
                configuration,
                Vec::new(),
                Vec::new(),
            ),
        ));

        let mut inherited = json!({"previous_response_id":"resp_legacy", "input":"next"});
        let error = ResponsesTurn::prepare(&mut inherited, continuation(&store)).unwrap_err();
        assert_eq!(
            error.body().error.param.as_deref(),
            Some("reasoning.summary")
        );

        let mut cleared = json!({
            "previous_response_id":"resp_legacy",
            "input":"next",
            "reasoning":null
        });
        ResponsesTurn::prepare(&mut cleared, continuation(&store)).unwrap();
        assert!(cleared["reasoning"].is_null());
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
