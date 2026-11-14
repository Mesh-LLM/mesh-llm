use super::*;

#[tokio::test]
async fn responses_http_and_sse_reject_malformed_numeric_controls_before_backend_execution() {
    let backend = Arc::new(SessionCaptureBackend::default());
    let app = router_for(backend.clone());
    let cases = [
        (json!({"max_output_tokens": 15}), "max_output_tokens"),
        (json!({"max_output_tokens": "16"}), "max_output_tokens"),
        (json!({"max_output_tokens": 16.5}), "max_output_tokens"),
        (json!({"max_tokens": 15}), "max_output_tokens"),
        (json!({"max_completion_tokens": 15}), "max_output_tokens"),
        (json!({"n_predict": 15}), "max_output_tokens"),
        (json!({"n_predict": {}}), "max_output_tokens"),
        (json!({"top_logprobs": 21}), "top_logprobs"),
        (json!({"top_logprobs": "20"}), "top_logprobs"),
        (json!({"top_logprobs": 0.5}), "top_logprobs"),
    ];
    for stream in [false, true] {
        for (fields, expected_param) in &cases {
            let mut request = json!({"model":"capture-model", "input":"hi", "stream":stream});
            request
                .as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            let response =
                post_json_with_app_and_request_id(app.clone(), "/v1/responses", request, None)
                    .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = response_body_json(response).await;
            assert_eq!(body["error"]["param"], *expected_param);
        }
    }
    assert!(backend.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn responses_http_numeric_boundaries_reach_the_backend_without_chat_narrowing() {
    let backend = Arc::new(SessionCaptureBackend::default());
    let app = router_for(backend.clone());
    for (max_output_tokens, top_logprobs) in [(16_u64, 0_u64), (u32::MAX.into(), 20)] {
        let response = post_json_with_app_and_request_id(
            app.clone(),
            "/v1/responses",
            json!({
                "model":"capture-model",
                "input":"hi",
                "max_output_tokens":max_output_tokens,
                "top_logprobs":top_logprobs,
            }),
            None,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body_json(response).await;
        assert_eq!(body["max_output_tokens"], max_output_tokens);
        assert_eq!(body["top_logprobs"], top_logprobs);
    }

    let requests = backend.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].max_tokens, Some(16));
    assert_eq!(requests[0].top_logprobs, Some(0));
    assert_eq!(requests[0].logprobs, Some(true));
    assert_eq!(requests[1].max_tokens, Some(u32::MAX));
    assert_eq!(requests[1].top_logprobs, Some(20));
    assert_eq!(requests[1].logprobs, Some(true));
}

#[tokio::test]
async fn responses_http_legacy_limit_aliases_are_echoed_and_enforced() {
    for alias in ["max_tokens", "max_completion_tokens", "n_predict"] {
        let backend = Arc::new(SessionCaptureBackend::default());
        let app = router_for(backend.clone());
        let mut request = json!({"model":"capture-model", "input":"hi"});
        request[alias] = json!(32);

        let response = post_json_with_app_and_request_id(app, "/v1/responses", request, None).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_body_json(response).await;
        assert_eq!(body["max_output_tokens"], 32);

        let requests = backend.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].max_tokens, Some(32));
    }
}

#[tokio::test]
async fn responses_http_rejects_unrepresentable_output_limit_before_backend_execution() {
    let backend = Arc::new(SessionCaptureBackend::default());
    let app = router_for(backend.clone());
    let response = post_json_with_app_and_request_id(
        app,
        "/v1/responses",
        json!({
            "model":"capture-model",
            "input":"hi",
            "max_output_tokens":u64::from(u32::MAX) + 1,
        }),
        None,
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_body_json(response).await;
    assert_eq!(body["error"]["param"], "max_output_tokens");
    assert_eq!(body["error"]["code"], "unsupported_model_feature");
    assert!(backend.requests.lock().unwrap().is_empty());
}
