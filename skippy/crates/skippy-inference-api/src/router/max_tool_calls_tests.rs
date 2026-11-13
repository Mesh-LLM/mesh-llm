use std::sync::Arc;

use async_trait::async_trait;
use axum::{body::Body, http::Request};
use futures_util::stream;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use super::*;
use crate::{
    FinishReason, ModelObject,
    generation::{
        GeneratedMessage, GenerationChoice, GenerationDelta, GenerationEvent,
        GenerationEventChoice, GenerationEventStream, GenerationRequest, GenerationResponse,
    },
};

struct PairToolBackend;

#[async_trait]
impl InferenceBackend for PairToolBackend {
    async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("tool-call-pair")])
    }

    async fn chat_completion(
        &self,
        request: GenerationRequest,
    ) -> InferenceResult<GenerationResponse> {
        Ok(GenerationResponse {
            id: "chatcmpl_tool_pair".to_string(),
            created: 123,
            model: request.model,
            choices: vec![GenerationChoice {
                index: 0,
                message: GeneratedMessage {
                    role: "assistant",
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(tool_calls()),
                },
                logprobs: None,
                finish_reason: Some(FinishReason::ToolCalls),
            }],
            usage: Usage::new(3, 2),
            timings: None,
            capsule_marker: None,
            exchange_id: None,
        })
    }

    async fn chat_completion_stream(
        &self,
        request: GenerationRequest,
        _context: InferenceRequestContext,
    ) -> InferenceResult<GenerationEventStream> {
        let model = request.model;
        Ok(Box::pin(stream::iter([
            Ok(GenerationEvent {
                id: "chatcmpl_stream_tool_pair".to_string(),
                created: 123,
                model: model.clone(),
                choices: vec![GenerationEventChoice {
                    index: 0,
                    delta: GenerationDelta {
                        role: None,
                        content: None,
                        reasoning_content: None,
                        tool_calls: Some(stream_tool_calls()),
                    },
                    logprobs: None,
                    finish_reason: None,
                }],
                usage: None,
                timings: None,
            }),
            Ok(GenerationEvent::done_with_reason(
                model,
                FinishReason::ToolCalls,
            )),
        ])))
    }
}

fn tool_calls() -> Value {
    json!([
        {"id":"call_0","type":"function","function":{"name":"same","arguments":"{}"}},
        {"id":"call_1","type":"function","function":{"name":"same","arguments":"{}"}}
    ])
}

fn stream_tool_calls() -> Value {
    json!([
        {"index":0,"id":"call_0","type":"function","function":{"name":"same","arguments":"{}"}},
        {"index":1,"id":"call_1","type":"function","function":{"name":"same","arguments":"{}"}}
    ])
}

async fn post(body: Value) -> axum::response::Response {
    router_for(Arc::new(PairToolBackend))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
}

async fn body_json(response: axum::response::Response) -> Value {
    serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap()
}

async fn body_text(response: axum::response::Response) -> String {
    String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap()
}

#[tokio::test]
async fn responses_json_enforces_max_tool_calls_at_the_http_ingress() {
    for (cap, status, output_len) in [(2, "completed", 2), (1, "failed", 0)] {
        let response = post(json!({
            "model": "tool-call-pair",
            "input": "hi",
            "max_tool_calls": cap,
            "tools": [{"type": "function", "name": "same"}],
        }))
        .await;
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["status"], status);
        assert_eq!(body["output"].as_array().unwrap().len(), output_len);
        assert_eq!(body["max_tool_calls"], cap);
        if cap == 1 {
            assert_eq!(body["error"]["code"], "tool_call_limit_exceeded");
            assert_eq!(body["store"], false);
        }
    }
}

#[tokio::test]
async fn responses_sse_quarantines_calls_over_max_tool_calls() {
    let response = post(json!({
        "model": "tool-call-pair",
        "input": "hi",
        "stream": true,
        "max_tool_calls": 1,
        "tools": [{"type": "function", "name": "same"}],
    }))
    .await;
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let body = body_text(response).await;
    assert!(body.contains("event: response.failed"));
    assert!(body.contains(r#""code":"tool_call_limit_exceeded""#));
    assert!(!body.contains("event: response.output_item.added"));
    assert!(!body.contains("event: response.function_call_arguments"));
}

#[tokio::test]
async fn responses_json_enforces_parallel_tool_calls_at_the_http_ingress() {
    for (parallel, status, output_len) in [(true, "completed", 2), (false, "failed", 0)] {
        let response = post(json!({
            "model": "tool-call-pair",
            "input": "hi",
            "parallel_tool_calls": parallel,
            "tools": [{"type": "function", "name": "same"}],
        }))
        .await;
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["status"], status);
        assert_eq!(body["output"].as_array().unwrap().len(), output_len);
        assert_eq!(body["parallel_tool_calls"], parallel);
        if !parallel {
            assert_eq!(body["error"]["code"], "tool_call_limit_exceeded");
            assert_eq!(body["store"], false);
        }
    }
}

#[tokio::test]
async fn responses_sse_quarantines_calls_when_parallel_tools_are_disabled() {
    let response = post(json!({
        "model": "tool-call-pair",
        "input": "hi",
        "stream": true,
        "parallel_tool_calls": false,
        "tools": [{"type": "function", "name": "same"}],
    }))
    .await;
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let body = body_text(response).await;
    assert!(body.contains("event: response.failed"));
    assert!(body.contains(r#""code":"tool_call_limit_exceeded""#));
    assert!(!body.contains("event: response.output_item.added"));
    assert!(!body.contains("event: response.function_call_arguments"));
}

#[tokio::test]
async fn responses_rejects_malformed_parallel_tool_calls_with_parameter() {
    let response = post(json!({
        "model": "tool-call-pair",
        "input": "hi",
        "parallel_tool_calls": "false",
        "tools": [{"type": "function", "name": "same"}],
    }))
    .await;
    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["error"]["param"], "parallel_tool_calls");
}
