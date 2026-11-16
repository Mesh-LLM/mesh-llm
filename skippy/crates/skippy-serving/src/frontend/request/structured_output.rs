use super::*;

mod schema;

fn structured_output_string(
    request: &GenerationRequest,
    name: &str,
) -> InferenceResult<Option<String>> {
    let value = bounded_optional_string_extra(&request.extra, name)?;
    if value.is_some()
        && request
            .extra
            .get("json_schema")
            .is_some_and(|value| !value.is_null())
    {
        return Err(InferenceError::invalid_request(
            "grammar and json_schema cannot both be set",
        ));
    }
    if let Some(grammar) = value.as_deref() {
        crate::grammar_bounds::check_grammar(grammar).map_err(InferenceError::invalid_request)?;
    }
    Ok(value)
}

fn structured_output_json(
    request: &GenerationRequest,
    name: &str,
) -> InferenceResult<Option<String>> {
    request
        .extra
        .get(name)
        .filter(|value| !value.is_null())
        .map(|value| {
            if !value.is_object() {
                return Err(InferenceError::invalid_request(
                    "json_schema must be an object",
                ));
            }
            serialize_bounded_native_parser_json("json_schema", value)
        })
        .transpose()
}

pub(super) fn resolve(
    request: &GenerationRequest,
) -> InferenceResult<(Option<String>, Option<String>)> {
    let grammar = structured_output_string(request, "grammar")?;
    let legacy_schema = structured_output_json(request, "json_schema")?;
    let Some(format) = request.response_format.as_ref().filter(|v| !v.is_null()) else {
        return Ok((grammar, legacy_schema));
    };
    let invalid = |message: &str| {
        InferenceError::invalid_request(message.to_string()).with_param("response_format")
    };
    let schema = match format.get("type").and_then(Value::as_str) {
        Some("text") => return Ok((grammar, legacy_schema)),
        Some("json_object") => serde_json::json!({"type": "object"}),
        Some("json_schema") => {
            let wrapper = format
                .get("json_schema")
                .filter(|v| v.is_object())
                .ok_or_else(|| invalid("json_schema format requires a json_schema object"))?;
            if wrapper
                .get("name")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            {
                return Err(invalid("json_schema format requires a nonempty name"));
            }
            if wrapper
                .get("strict")
                .is_some_and(|v| !v.is_null() && !v.is_boolean())
            {
                return Err(invalid("json_schema strict must be a boolean"));
            }
            match wrapper.get("schema") {
                None => serde_json::json!({}),
                Some(schema) if schema.is_object() => schema.clone(),
                _ => return Err(invalid("json_schema format requires a schema object")),
            }
        }
        _ => return Err(invalid("unsupported response_format type")),
    };
    if grammar.is_some() || legacy_schema.is_some() {
        return Err(invalid(
            "response_format cannot be combined with grammar or json_schema",
        ));
    }
    let serialized = serialize_bounded_native_parser_json("response_format", &schema)
        .map_err(|error| error.with_param("response_format"))?;
    if format.get("type").and_then(Value::as_str) == Some("json_schema") {
        schema::validate(&schema)?;
        if schema.get("type").is_none() {
            // The native chat path treats an empty root schema as no constraint.
            return Ok((
                None,
                Some(
                    serde_json::json!({
                        "type":["object","array","string","number","boolean","null"]
                    })
                    .to_string(),
                ),
            ));
        }
    }
    Ok((None, Some(serialized)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use skippy_inference_api::chat::ChatCompletionRequest;

    fn request(extra: Value) -> GenerationRequest {
        let mut value = json!({"model":"test", "messages":[{"role":"user","content":"hi"}]});
        value
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value::<ChatCompletionRequest>(value)
            .unwrap()
            .into()
    }

    #[test]
    fn standard_formats_reach_native_schema_options() {
        let schema = json!({"type":"object", "properties":{"answer":{"type":"integer","const":7}}, "additionalProperties":false});
        let request = request(
            json!({"response_format":{"type":"json_schema","json_schema":{
                "name":"answer", "strict":true, "schema":schema
            }}}),
        );
        let options = super::super::chat_template_options(
            &request,
            &EmbeddedOpenAiRequestDefaults::default(),
        )
        .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(options.json_schema.as_ref().unwrap()).unwrap(),
            schema
        );
        assert!(options.grammar.is_none());
        let (_, schema) = resolve(&self::request(
            json!({"response_format":{"type":"json_object"}}),
        ))
        .unwrap();
        assert_eq!(schema.as_deref(), Some("{\"type\":\"object\"}"));
    }

    #[test]
    fn rejects_ambiguous_and_malformed_formats() {
        for extra in [
            json!({"response_format":{"type":"json_object"}, "grammar":"root ::= \"a\""}),
            json!({"response_format":{"type":"json_object"}, "json_schema":{"type":"object"}}),
            json!({"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":null}}}),
            json!({"response_format":{"type":"json_schema","json_schema":{"schema":{}}}}),
            json!({"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{},"strict":1}}}),
            json!({"response_format":{"type":"unknown"}}),
        ] {
            assert!(resolve(&request(extra.clone())).is_err(), "{extra}");
        }
    }

    #[test]
    fn explicit_format_suppresses_deployment_grammar_defaults() {
        let defaults = EmbeddedOpenAiRequestDefaults {
            grammar: Some(json!("root ::= \"default\"")),
            ..EmbeddedOpenAiRequestDefaults::default()
        };
        for format in [json!({"type":"text"}), json!({"type":"json_object"})] {
            let mut request = request(json!({"response_format":format}));
            super::super::apply_chat_request_defaults(&mut request, &defaults).unwrap();
            assert!(!request.extra.contains_key("grammar"));
            resolve(&request).unwrap();
        }
    }

    #[test]
    fn responses_text_format_suppresses_deployment_defaults_after_translation() {
        let mut body = json!({
            "model":"test", "input":"hi", "text":{"format":{"type":"text"}}
        });
        skippy_inference_api::responses::normalize_openai_compat_request(
            "/v1/responses",
            &mut body,
        )
        .unwrap();
        let mut request: GenerationRequest = serde_json::from_value::<ChatCompletionRequest>(body)
            .unwrap()
            .into();
        let defaults = EmbeddedOpenAiRequestDefaults {
            grammar: Some(json!("root ::= \"default\"")),
            ..EmbeddedOpenAiRequestDefaults::default()
        };
        super::super::apply_chat_request_defaults(&mut request, &defaults).unwrap();
        let options = super::super::chat_template_options(&request, &defaults).unwrap();
        assert!(options.grammar.is_none());
        assert!(options.json_schema.is_none());
    }

    #[test]
    fn responses_nullable_strict_reaches_native_options() {
        let mut body = json!({"model":"test", "input":"hi", "text":{"format":{
            "type":"json_schema", "name":"answer", "strict":null,
            "schema":{"type":"integer","const":7}
        }}});
        skippy_inference_api::responses::normalize_openai_compat_request(
            "/v1/responses",
            &mut body,
        )
        .unwrap();
        let request: GenerationRequest = serde_json::from_value::<ChatCompletionRequest>(body)
            .unwrap()
            .into();
        assert!(resolve(&request).unwrap().1.is_some());
    }

    #[test]
    fn responses_omitted_schema_and_name_reach_unconstrained_native_json() {
        for format in [
            json!({"type":"json_schema"}),
            json!({"type":"json_schema","schema":{}}),
            json!({"type":"json_schema","schema":{"title":"Any JSON","description":"Any root"}}),
        ] {
            let mut body = json!({"model":"test","input":"hi","text":{"format":format}});
            skippy_inference_api::responses::normalize_openai_compat_request(
                "/v1/responses",
                &mut body,
            )
            .unwrap();
            let request: GenerationRequest = serde_json::from_value::<ChatCompletionRequest>(body)
                .unwrap()
                .into();
            let schema: Value =
                serde_json::from_str(&resolve(&request).unwrap().1.unwrap()).unwrap();
            assert_eq!(
                schema["type"],
                json!(["object", "array", "string", "number", "boolean", "null"])
            );
        }
    }

    #[test]
    fn preserves_legacy_and_bounds_standard_schema() {
        let legacy =
            request(json!({"json_schema":{"type":"object"}, "response_format":{"type":"text"}}));
        assert!(resolve(&legacy).unwrap().1.is_some());
        let oversized = request(
            json!({"response_format":{"type":"json_schema", "json_schema":{
                "name":"x", "schema":{"description":"x".repeat(MAX_NATIVE_PARSER_INPUT_BYTES)}
            }}}),
        );
        assert_eq!(
            resolve(&oversized)
                .unwrap_err()
                .body()
                .error
                .param
                .as_deref(),
            Some("response_format")
        );
    }
}
