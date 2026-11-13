use super::*;

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
