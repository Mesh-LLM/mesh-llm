use super::*;
use tokio::io::AsyncWriteExt;

async fn parse_responses_http_request(
    body: serde_json::Value,
    store: std::sync::Arc<skippy_inference_api::responses::ResponseStore>,
) -> std::result::Result<BufferedHttpRequest, OpenAiRequestReadError> {
    let body = body.to_string();
    let request = format!(
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len(),
    );
    parse_raw_http_request(request.as_bytes(), store).await
}

async fn parse_raw_http_request(
    request: &[u8],
    store: std::sync::Arc<skippy_inference_api::responses::ResponseStore>,
) -> std::result::Result<BufferedHttpRequest, OpenAiRequestReadError> {
    let (mut client, mut server) = tokio::io::duplex(request.len() + 1);
    client.write_all(request).await.unwrap();
    client.shutdown().await.unwrap();

    read_http_request_with_limits_with_context(
        &mut server,
        HTTP_READ_LIMITS,
        None,
        RequestTransforms::Apply,
        Some(store),
        Some(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)),
    )
    .await
}

#[tokio::test]
async fn mesh_http_and_sse_reject_malformed_responses_controls_before_lookup_or_forwarding() {
    let cases = [
        (serde_json::json!({"instructions": []}), "instructions"),
        (serde_json::json!({"metadata": []}), "metadata"),
        (serde_json::json!({"metadata": {"key": 7}}), "metadata"),
        (
            serde_json::json!({"safety_identifier": "🦀".repeat(65)}),
            "safety_identifier",
        ),
        (
            serde_json::json!({"prompt_cache_key": []}),
            "prompt_cache_key",
        ),
        (serde_json::json!({"text": "plain"}), "text"),
        (serde_json::json!({"reasoning": []}), "reasoning"),
        (
            serde_json::json!({"reasoning": {"summary": "auto"}}),
            "reasoning.summary",
        ),
        (
            serde_json::json!({"max_output_tokens": 15}),
            "max_output_tokens",
        ),
        (
            serde_json::json!({"max_output_tokens": "16"}),
            "max_output_tokens",
        ),
        (serde_json::json!({"max_tokens": 15}), "max_output_tokens"),
        (
            serde_json::json!({"max_completion_tokens": 15}),
            "max_output_tokens",
        ),
        (serde_json::json!({"n_predict": 15}), "max_output_tokens"),
        (serde_json::json!({"n_predict": {}}), "max_output_tokens"),
        (serde_json::json!({"top_logprobs": 21}), "top_logprobs"),
        (serde_json::json!({"top_logprobs": false}), "top_logprobs"),
    ];
    for stream in [false, true] {
        for (fields, expected_param) in &cases {
            let mut request = serde_json::json!({
                "model":"m",
                "input":"hi",
                "stream":stream,
                "previous_response_id":"resp_missing"
            });
            request
                .as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            let body = serde_json::to_vec(&request).unwrap();
            let store =
                std::sync::Arc::new(skippy_inference_api::responses::ResponseStore::default());
            let error = match rewrite_request_body_for_forwarding(
                "/v1/responses",
                &body,
                None,
                Transformer {
                    plugins: None,
                    private_hint: false,
                    node_limit: None,
                },
                true,
                Some(skippy_inference_api::responses::ResponseContinuation::new(
                    Some(store),
                    Some(skippy_inference_api::responses::CallerScope::LocalHost),
                )),
            )
            .await
            {
                Err(error) => error,
                Ok(_) => panic!("malformed Responses controls should be rejected"),
            };
            let inference = error
                .downcast_ref::<skippy_inference_api::InferenceError>()
                .expect("typed Responses validation error");

            assert_eq!(inference.status(), http::StatusCode::BAD_REQUEST);
            assert_eq!(
                inference.body().error.param.as_deref(),
                Some(*expected_param)
            );
        }
    }
}

#[tokio::test]
async fn mesh_ingress_preserves_numeric_boundaries_and_maps_responses_logprobs() {
    for (max_output_tokens, top_logprobs) in [(16_u64, 0_u64), (u32::MAX.into(), 20)] {
        let body = serde_json::to_vec(&serde_json::json!({
            "model":"m",
            "input":"hi",
            "max_output_tokens":max_output_tokens,
            "top_logprobs":top_logprobs,
        }))
        .unwrap();
        let outcome = rewrite_request_body_for_forwarding(
            "/v1/responses",
            &body,
            None,
            Transformer {
                plugins: None,
                private_hint: false,
                node_limit: None,
            },
            true,
            Some(skippy_inference_api::responses::ResponseContinuation::disabled()),
        )
        .await
        .unwrap();
        let normalized = outcome.body_json.unwrap();

        assert_eq!(normalized["max_tokens"], max_output_tokens);
        assert_eq!(normalized["top_logprobs"], top_logprobs);
        assert_eq!(normalized["logprobs"], true);
        let response = outcome.responses_turn.unwrap().response(
            skippy_inference_api::responses::ResponseStatus::InProgress,
            &[],
            None,
            None,
        );
        assert_eq!(response["max_output_tokens"], max_output_tokens);
        assert_eq!(response["top_logprobs"], top_logprobs);
    }
}

#[tokio::test]
async fn mesh_ingress_rejects_unrepresentable_output_limit_before_forwarding() {
    let body = serde_json::to_vec(&serde_json::json!({
        "model":"m",
        "input":"hi",
        "previous_response_id":"resp_missing",
        "max_output_tokens":u64::from(u32::MAX) + 1,
    }))
    .unwrap();
    let error = match rewrite_request_body_for_forwarding(
        "/v1/responses",
        &body,
        None,
        Transformer {
            plugins: None,
            private_hint: false,
            node_limit: None,
        },
        true,
        Some(skippy_inference_api::responses::ResponseContinuation::disabled()),
    )
    .await
    {
        Err(error) => error,
        Ok(_) => panic!("unrepresentable max_output_tokens should be rejected"),
    };
    let inference = error
        .downcast_ref::<skippy_inference_api::InferenceError>()
        .expect("typed Responses validation error");

    assert_eq!(inference.status(), http::StatusCode::BAD_REQUEST);
    assert_eq!(
        inference.body().error.param.as_deref(),
        Some("max_output_tokens")
    );
    assert_eq!(
        inference.body().error.code.as_deref(),
        Some("unsupported_model_feature")
    );
}

#[tokio::test]
async fn full_http_parser_budgets_from_the_effective_responses_limit() {
    let cases = [
        (
            serde_json::json!({
                "model":"m",
                "input":"hi",
                "max_output_tokens":16,
                "max_tokens":32,
                "max_completion_tokens":64,
            }),
            Some(16),
        ),
        (
            serde_json::json!({"model":"m", "input":"hi", "max_tokens":32}),
            Some(32),
        ),
        (
            serde_json::json!({"model":"m", "input":"hi", "max_completion_tokens":48}),
            Some(48),
        ),
        (
            serde_json::json!({"model":"m", "input":"hi", "n_predict":96}),
            Some(96),
        ),
        (
            serde_json::json!({
                "model":"m",
                "input":"hi",
                "max_output_tokens":null,
                "max_tokens":32,
                "max_completion_tokens":64,
            }),
            None,
        ),
    ];

    for (body, expected) in cases {
        let store = std::sync::Arc::new(skippy_inference_api::responses::ResponseStore::default());
        let request = parse_responses_http_request(body, store).await.unwrap();
        assert_eq!(request.completion_tokens, expected);
        let forwarded = request.body_json.as_ref().unwrap();
        assert_eq!(
            forwarded
                .get("max_tokens")
                .and_then(serde_json::Value::as_u64),
            expected.map(u64::from)
        );
        for alias in ["max_output_tokens", "max_completion_tokens", "n_predict"] {
            assert!(forwarded.get(alias).is_none(), "forwarded {alias}");
        }
        let response = request.responses_turn.as_ref().unwrap().response(
            skippy_inference_api::responses::ResponseStatus::InProgress,
            &[],
            None,
            None,
        );
        assert_eq!(
            response["max_output_tokens"].as_u64(),
            expected.map(u64::from)
        );
    }
}

#[tokio::test]
async fn full_http_parser_routes_from_effective_body_when_losing_aliases_are_malformed() {
    let cases = [
        (
            "overflowing max_tokens",
            serde_json::json!({"max_tokens":4294967296_u64}),
        ),
        (
            "string max_tokens",
            serde_json::json!({"max_tokens":"invalid"}),
        ),
        (
            "object max_tokens",
            serde_json::json!({"max_tokens":{"invalid":true}}),
        ),
        (
            "invalid max_completion_tokens",
            serde_json::json!({"max_completion_tokens":"invalid"}),
        ),
        (
            "invalid n_predict extension",
            serde_json::json!({"n_predict":{"invalid":true}}),
        ),
    ];

    for (case, fields) in cases {
        let mut body = serde_json::json!({
            "model":"m",
            "input":"hi",
            "stream":true,
            "max_output_tokens":16,
        });
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        let store = std::sync::Arc::new(skippy_inference_api::responses::ResponseStore::default());
        let request = parse_responses_http_request(body, store)
            .await
            .unwrap_or_else(|error| panic!("{case}: {error}"));

        assert_eq!(request.model_name.as_deref(), Some("m"), "{case}");
        assert_eq!(request.completion_tokens, Some(16), "{case}");
        assert_eq!(request.stream, Some(true), "{case}");
        let backend_body = parse_json_body_from_http_request(&request.raw)
            .unwrap_or_else(|| panic!("{case}: forwarded backend body must be valid JSON"));
        let backend: skippy_inference_api::ChatCompletionRequest =
            serde_json::from_value(backend_body.clone())
                .unwrap_or_else(|error| panic!("{case}: invalid backend body: {error}"));
        assert_eq!(backend.model, "m", "{case}");
        assert_eq!(backend.effective_max_tokens(), Some(16), "{case}");
        assert!(backend.stream, "{case}");
        assert_eq!(request.body_json.as_ref(), Some(&backend_body), "{case}");
        for alias in ["max_output_tokens", "max_completion_tokens", "n_predict"] {
            assert!(
                backend_body.get(alias).is_none(),
                "{case}: forwarded {alias}"
            );
        }
    }
}

#[tokio::test]
async fn full_http_parser_budgets_inherited_alias_overrides_and_null_resets() {
    let store = std::sync::Arc::new(skippy_inference_api::responses::ResponseStore::default());
    let parent = parse_responses_http_request(
        serde_json::json!({
            "model":"m",
            "input":"first",
            "max_output_tokens":32,
        }),
        store.clone(),
    )
    .await
    .unwrap();
    assert_eq!(parent.completion_tokens, Some(32));
    let parent_response =
        parent
            .responses_turn
            .unwrap()
            .finish_chat_completion(&serde_json::json!({
                "choices":[{
                    "message":{"role":"assistant", "content":"answer"},
                    "finish_reason":"stop"
                }],
                "usage":{"prompt_tokens":3, "completion_tokens":2, "total_tokens":5}
            }));
    let parent_id = parent_response["id"].as_str().unwrap();

    for (fields, expected) in [
        (serde_json::json!({}), Some(32)),
        (serde_json::json!({"max_completion_tokens":64}), Some(64)),
        (serde_json::json!({"n_predict":96}), Some(96)),
        (serde_json::json!({"max_tokens":null}), None),
        (serde_json::json!({"n_predict":null}), None),
        (
            serde_json::json!({"max_output_tokens":null, "max_tokens":128}),
            None,
        ),
    ] {
        let mut body = serde_json::json!({
            "previous_response_id":parent_id,
            "input":"next",
        });
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        let request = parse_responses_http_request(body, store.clone())
            .await
            .unwrap();

        assert_eq!(request.completion_tokens, expected);
        let response = request.responses_turn.as_ref().unwrap().response(
            skippy_inference_api::responses::ResponseStatus::InProgress,
            &[],
            None,
            None,
        );
        assert_eq!(
            response["max_output_tokens"].as_u64(),
            expected.map(u64::from)
        );
    }
}

#[tokio::test]
async fn full_http_parser_routes_with_inherited_model_and_actual_stream() {
    let store = std::sync::Arc::new(skippy_inference_api::responses::ResponseStore::default());
    let parent = parse_responses_http_request(
        serde_json::json!({
            "model":"m",
            "input":"first",
            "max_output_tokens":16,
        }),
        store.clone(),
    )
    .await
    .unwrap();
    let parent_response =
        parent
            .responses_turn
            .unwrap()
            .finish_chat_completion(&serde_json::json!({
                "choices":[{
                    "message":{"role":"assistant", "content":"answer"},
                    "finish_reason":"stop"
                }],
                "usage":{"prompt_tokens":3, "completion_tokens":2, "total_tokens":5}
            }));
    let request = parse_responses_http_request(
        serde_json::json!({
            "previous_response_id":parent_response["id"],
            "input":"next",
            "stream":true,
            "max_output_tokens":16,
            "n_predict":"ignored extension",
        }),
        store,
    )
    .await
    .unwrap();

    assert_eq!(request.model_name.as_deref(), Some("m"));
    assert_eq!(request.completion_tokens, Some(16));
    assert_eq!(request.stream, Some(true));
    let backend_body = parse_json_body_from_http_request(&request.raw).unwrap();
    let backend: skippy_inference_api::ChatCompletionRequest =
        serde_json::from_value(backend_body.clone()).unwrap();
    assert_eq!(backend.model, "m");
    assert_eq!(backend.effective_max_tokens(), Some(16));
    assert!(backend.stream);
    assert_eq!(request.body_json.as_ref(), Some(&backend_body));
    assert!(backend_body.get("n_predict").is_none());
}

#[tokio::test]
async fn current_peer_recovers_concrete_model_when_an_unknown_responses_extension_breaks_metadata()
{
    let body = serde_json::json!({
        "model":"auto",
        "input":"Reply only OK",
        "max_output_tokens":16,
        "expected_identity":[],
    });
    let store = std::sync::Arc::new(skippy_inference_api::responses::ResponseStore::default());
    let mut request = parse_responses_http_request(body, store).await.unwrap();
    rewrite_model_field(&mut request, "selected-local-model");
    inject_skippy_hooks_flag(&mut request.raw, true);

    let peer_request = parse_raw_http_request(
        &request.raw,
        std::sync::Arc::new(skippy_inference_api::responses::ResponseStore::default()),
    )
    .await
    .unwrap();

    assert_eq!(peer_request.client_path, "/v1/chat/completions");
    assert_eq!(
        peer_request.model_name.as_deref(),
        Some("selected-local-model")
    );
    assert_eq!(peer_request.completion_tokens, Some(16));
    assert_eq!(peer_request.stream, None);
    assert_eq!(peer_request.body_json, None);
    assert_eq!(
        parse_json_body_from_http_request(&peer_request.raw).unwrap()["expected_identity"],
        serde_json::json!([])
    );
}
