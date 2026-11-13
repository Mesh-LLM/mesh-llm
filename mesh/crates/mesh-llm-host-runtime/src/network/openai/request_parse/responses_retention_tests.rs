use super::*;

#[tokio::test]
async fn http_and_sse_reject_non_boolean_retention_before_forwarding() {
    for stream in [false, true] {
        for (field, invalid) in [
            ("store", serde_json::json!(null)),
            ("store", serde_json::json!({})),
            ("background", serde_json::json!(null)),
            ("background", serde_json::json!([])),
        ] {
            let store =
                std::sync::Arc::new(skippy_inference_api::responses::ResponseStore::default());
            let mut body = serde_json::json!({"model": "m", "input": "hi", "stream": stream});
            body[field] = invalid;
            let error = read_loopback_responses_request(body, "", store)
                .await
                .unwrap_err();
            let inference = error
                .inference_error()
                .expect("typed request validation error");

            assert_eq!(inference.status(), http::StatusCode::BAD_REQUEST);
            assert_eq!(
                inference.body().error.code.as_deref(),
                Some("invalid_value")
            );
            assert_eq!(inference.body().error.param.as_deref(), Some(field));
        }
    }
}
