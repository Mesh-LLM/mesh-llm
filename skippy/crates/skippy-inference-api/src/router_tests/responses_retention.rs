use super::*;

#[tokio::test]
async fn http_and_sse_reject_non_boolean_retention_before_backend_execution() {
    let backend = Arc::new(SessionCaptureBackend::default());
    let app = router_for(backend.clone());
    for stream in [false, true] {
        for (field, invalid) in [
            ("store", json!(null)),
            ("store", json!("false")),
            ("background", json!(null)),
            ("background", json!(0)),
        ] {
            let mut body = json!({"model": "capture-model", "input": "hi", "stream": stream});
            body[field] = invalid;
            let response =
                post_json_with_app_and_request_id(app.clone(), "/v1/responses", body, None).await;

            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = response_body_json(response).await;
            assert_eq!(body["error"]["type"], "invalid_request_error");
            assert_eq!(body["error"]["code"], "invalid_value");
            assert_eq!(body["error"]["param"], field);
        }
    }
    assert!(backend.requests.lock().unwrap().is_empty());
}
