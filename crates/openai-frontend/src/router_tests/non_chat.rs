use super::*;

#[tokio::test]
async fn audio_upload_rejects_each_duplicate_field() {
    let boundary = "duplicate-audio-field";
    for path in ["/v1/audio/transcriptions", "/v1/audio/translations"] {
        for (field, value) in [
            ("model", "audio-model"),
            ("file", "WAVE"),
            ("language", "en"),
            ("prompt", "words"),
            ("response_format", "json"),
            ("temperature", "0"),
        ] {
            let part = format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"\r\n\r\n{value}\r\n"
            );
            let mut body = part.repeat(2).into_bytes();
            body.extend_from_slice(&audio_multipart(boundary, "json"));
            let response = post_audio_multipart(path, boundary, body).await;
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "{path}: {field}"
            );
            let body = response_body_json(response).await;
            assert_eq!(
                body["error"]["message"],
                format!("duplicate multipart {field} field")
            );
        }
    }
}

#[tokio::test]
async fn audio_upload_enforces_temperature_range_at_both_endpoints() {
    let boundary = "audio-temperature";
    for path in ["/v1/audio/transcriptions", "/v1/audio/translations"] {
        for (value, accepted) in [
            ("0", true),
            ("1", true),
            ("0.5", true),
            ("1.1", false),
            ("-0.1", false),
            ("NaN", false),
            ("inf", false),
            ("-inf", false),
        ] {
            let mut body = format!("--{boundary}\r\nContent-Disposition: form-data; name=\"temperature\"\r\n\r\n{value}\r\n").into_bytes();
            body.extend_from_slice(&audio_multipart(boundary, "json"));
            let response = post_audio_multipart(path, boundary, body).await;
            assert_eq!(
                response.status(),
                if accepted {
                    StatusCode::OK
                } else {
                    StatusCode::BAD_REQUEST
                },
                "{path}: {value}"
            );
        }
    }
}

#[tokio::test]
async fn embeddings_route_preserves_batch_order_and_usage() {
    let response = post_json(
        "/v1/embeddings",
        json!({
            "model": "embed-model",
            "input": ["first", "second"],
            "encoding_format": "float"
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "embed-model");
    assert_eq!(body["data"][0]["index"], 0);
    assert_eq!(body["data"][0]["embedding"], json!([1.0, -0.5]));
    assert_eq!(body["data"][1]["index"], 1);
    assert_eq!(body["data"][1]["embedding"], json!([2.0, -0.5]));
    assert_eq!(body["usage"]["prompt_tokens"], 7);
    assert_eq!(body["usage"]["total_tokens"], 7);
}

#[tokio::test]
async fn embeddings_route_returns_openai_error_for_invalid_format() {
    let response = post_json(
        "/v1/embeddings",
        json!({
            "model": "embed-model",
            "input": "hello",
            "encoding_format": "hex"
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_body_json(response).await;
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["code"], "invalid_value");
}

#[tokio::test]
async fn rerank_route_sorts_limits_and_optionally_returns_documents() {
    let response = post_json(
        "/v1/rerank",
        json!({
            "model": "rerank-model",
            "query": "query",
            "documents": ["one", {"text": "two", "title": "Two"}],
            "top_n": 1,
            "return_documents": true
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["id"], "rerank_test");
    assert_eq!(body["results"].as_array().unwrap().len(), 1);
    assert_eq!(body["results"][0]["index"], 1);
    assert_eq!(body["results"][0]["relevance_score"], 0.2);
    assert_eq!(body["results"][0]["document"]["text"], "two");
    assert_eq!(body["usage"]["prompt_tokens"], 11);
}

#[tokio::test]
async fn audio_speech_route_returns_backend_bytes_and_content_type() {
    let response = post_json(
        "/v1/audio/speech",
        json!({
            "model": "tts-model",
            "input": "Hello",
            "voice": "default",
            "response_format": "wav"
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "audio/wav");
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(bytes.as_ref(), b"RIFF");
}

fn audio_multipart(boundary: &str, response_format: &str) -> Vec<u8> {
    format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\naudio-model\r\n\
         --{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\n{response_format}\r\n\
         --{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n\
         Content-Type: audio/wav\r\n\r\nWAVE\r\n--{boundary}--\r\n"
    )
    .into_bytes()
}

async fn post_audio_multipart(path: &str, boundary: &str, body: Vec<u8>) -> Response {
    router_for(Arc::new(FakeBackend))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn audio_transcription_supports_json_and_text_responses() {
    let boundary = "mesh-audio-boundary";
    let json_response = post_audio_multipart(
        "/v1/audio/transcriptions",
        boundary,
        audio_multipart(boundary, "json"),
    )
    .await;
    assert_eq!(json_response.status(), StatusCode::OK);
    let body = response_body_json(json_response).await;
    assert_eq!(body["text"], "transcribed 4 bytes");

    let text_response = post_audio_multipart(
        "/v1/audio/transcriptions",
        boundary,
        audio_multipart(boundary, "text"),
    )
    .await;
    assert_eq!(text_response.status(), StatusCode::OK);
    assert_eq!(
        text_response.headers()["content-type"],
        "text/plain; charset=utf-8"
    );
    assert_eq!(
        response_body_text(text_response).await,
        "transcribed 4 bytes"
    );
}

#[tokio::test]
async fn audio_translation_uses_translation_backend() {
    let boundary = "mesh-audio-boundary";
    let response = post_audio_multipart(
        "/v1/audio/translations",
        boundary,
        audio_multipart(boundary, "json"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["text"], "translated 4 bytes");
}

#[tokio::test]
async fn audio_upload_uses_its_dedicated_body_limit() {
    let boundary = "mesh-large-audio-boundary";
    let file_bytes = vec![0x2a; OpenAiFrontendConfig::default().max_request_body_bytes + 1];
    let mut body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\naudio-model\r\n\
         --{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n\
         Content-Type: audio/wav\r\n\r\n"
    )
    .into_bytes();
    body.extend_from_slice(&file_bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let response = post_audio_multipart("/v1/audio/transcriptions", boundary, body).await;

    assert_eq!(response.status(), StatusCode::OK);
    let response = response_body_json(response).await;
    assert_eq!(
        response["text"],
        format!("transcribed {} bytes", file_bytes.len())
    );
}

#[tokio::test]
async fn malformed_audio_multipart_uses_openai_error_envelope() {
    let response = router_for(Arc::new(FakeBackend))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/transcriptions")
                .header("content-type", "multipart/form-data")
                .body(Body::from("not multipart"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_body_json(response).await;
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn duplicate_audio_model_field_is_rejected() {
    let boundary = "mesh-audio-boundary";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\naudio-model\r\n\
         --{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nother-model\r\n\
         --{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n\
         Content-Type: audio/wav\r\n\r\nWAVE\r\n--{boundary}--\r\n"
    );
    for path in ["/v1/audio/transcriptions", "/v1/audio/translations"] {
        let response = post_audio_multipart(path, boundary, body.as_bytes().to_vec()).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let payload = response_body_json(response).await;
        assert_eq!(payload["error"]["type"], "invalid_request_error");
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap()
                .contains("duplicate multipart model field")
        );
    }
}
