//! HTTP compatibility for ingress-owned conversation state, across requests.
//! Context follows the pinned 2026-04-24 spec's input + output + input rule.
use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use skippy_inference_api::{
    InferenceBackend, InferenceFrontendConfig, InferenceRequestContext, InferenceResult,
    ModelObject, Usage,
    generation::{GenerationEventStream, GenerationRequest, GenerationResponse},
    messages_to_plain_prompt,
    responses::ResponseStore,
    router_for_with_config,
};
use tower::ServiceExt;

#[derive(Default)]
struct RecordingBackend(Mutex<Vec<GenerationRequest>>);

#[async_trait]
impl InferenceBackend for RecordingBackend {
    async fn chat_completion_stream(
        &self,
        _request: GenerationRequest,
        _context: InferenceRequestContext,
    ) -> InferenceResult<GenerationEventStream> {
        unreachable!("these contracts use non-streaming HTTP requests")
    }

    async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("echo")])
    }

    async fn chat_completion(
        &self,
        request: GenerationRequest,
    ) -> InferenceResult<GenerationResponse> {
        let needs_tool = request.model == "tools"
            && request.tools.is_some()
            && !request
                .messages
                .iter()
                .any(|message| message.role == "tool");
        self.0.lock().unwrap().push(request);
        let mut response = GenerationResponse::new("echo", "saved answer", Usage::new(4, 2));
        if needs_tool {
            response.choices[0].message.content = None;
            response.choices[0].message.tool_calls = Some(json!([{
                "id": "call_weather", "type": "function",
                "function": {"name": "weather", "arguments": "{\"city\":\"Melbourne\"}"}
            }]));
            response.choices[0].finish_reason = Some(skippy_inference_api::FinishReason::ToolCalls);
        }
        Ok(response)
    }
}

fn router(backend: Arc<RecordingBackend>) -> axum::Router {
    router_for_with_config(
        backend,
        InferenceFrontendConfig::default().with_response_store(Arc::new(ResponseStore::default())),
    )
}

async fn post(router: &axum::Router, body: Value, forwarded: Option<&str>) -> (StatusCode, Value) {
    post_at(router, "/v1/responses", body, forwarded).await
}

async fn post_at(
    router: &axum::Router,
    path: &str,
    body: Value,
    forwarded: Option<&str>,
) -> (StatusCode, Value) {
    let mut builder = Request::post(path).header("content-type", "application/json");
    if let Some(header) = forwarded {
        builder = builder.header(header, "");
    }
    let mut request = builder.body(Body::from(body.to_string())).unwrap();
    request.extensions_mut().insert(ConnectInfo(
        "127.0.0.1:12345".parse::<SocketAddr>().unwrap(),
    ));
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn separate_http_requests_restore_input_and_output_but_not_instructions() {
    let backend = Arc::new(RecordingBackend::default());
    let router = router(backend.clone());
    let (status, first) = post(
        &router,
        json!({
            "model": "echo", "input": "remember slate", "instructions": "old instruction"
        }),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(first["store"], true);
    let (status, _) = post(&router, json!({
        "previous_response_id": first["id"], "input": "what word?", "instructions": "new instruction"
    }), None).await;
    assert_eq!(status, StatusCode::OK);
    let requests = backend.0.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].model, "echo");
    let prompt = messages_to_plain_prompt(&requests[1].messages);
    for text in [
        "remember slate",
        "saved answer",
        "what word?",
        "new instruction",
    ] {
        assert!(prompt.contains(text), "missing {text}: {prompt}");
    }
    assert!(!prompt.contains("old instruction"));
}

#[tokio::test]
async fn store_false_cannot_be_continued_over_http() {
    let backend = Arc::new(RecordingBackend::default());
    let router = router(backend.clone());
    let (status, first) = post(
        &router,
        json!({"model": "echo", "input": "private", "store": false}),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(first["store"], false);
    let (status, error) = post(
        &router,
        json!({"model": "echo", "previous_response_id": first["id"], "input": "continue"}),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error["error"]["code"], "previous_response_not_found");
    assert_eq!(backend.0.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn forwarding_header_presence_denies_reading_and_writing_local_state() {
    let backend = Arc::new(RecordingBackend::default());
    let router = router(backend.clone());
    let (_, local) = post(
        &router,
        json!({"model": "echo", "input": "host secret"}),
        None,
    )
    .await;
    for header in ["forwarded", "x-forwarded-for", "x-real-ip"] {
        let (status, error) = post(
            &router,
            json!({"model": "echo", "previous_response_id": local["id"], "input": "continue"}),
            Some(header),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(error["error"]["code"], "previous_response_not_found");
        let (status, proxied) = post(
            &router,
            json!({"model": "echo", "input": "proxied"}),
            Some(header),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(proxied["store"], false);
        let (status, error) = post(
            &router,
            json!({"model": "echo", "previous_response_id": proxied["id"], "input": "continue"}),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(error["error"]["code"], "previous_response_not_found");
    }
    assert_eq!(backend.0.lock().unwrap().len(), 4);
}

#[tokio::test]
async fn tool_output_only_continuation_restores_the_original_call_and_tools() {
    let backend = Arc::new(RecordingBackend::default());
    let router = router(backend.clone());
    let (status, first) = post(
        &router,
        json!({
            "model": "tools", "input": "weather in Melbourne?",
            "tools": [{"type": "function", "name": "weather", "parameters": {
                "type": "object", "properties": {"city": {"type": "string"}}
            }}]
        }),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(first["output"][0]["type"], "function_call");
    assert_eq!(first["output"][0]["call_id"], "call_weather");
    let (status, second) = post(
        &router,
        json!({
            "previous_response_id": first["id"],
            "input": [{"type": "function_call_output", "call_id": "call_weather", "output": "18 C"}]
        }),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(second["status"], "completed");
    let requests = backend.0.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].tools, requests[0].tools);
    let history = serde_json::to_value(&requests[1].messages).unwrap();
    assert_eq!(history[0]["role"], "user");
    assert_eq!(history[1]["tool_calls"][0]["id"], "call_weather");
    assert_eq!(
        history[1]["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Melbourne\"}"
    );
    assert_eq!(history[2]["role"], "tool");
    assert_eq!(history[2]["tool_call_id"], "call_weather");
    assert_eq!(history[2]["content"], "18 C");
}

#[tokio::test]
async fn compacted_pending_call_starts_a_new_http_chain_with_its_tool_result() {
    let backend = Arc::new(RecordingBackend::default());
    let router = router(backend.clone());
    let tools = json!([{"type": "function", "name": "weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}
    }}]);
    let (_, first) = post(
        &router,
        json!({
            "model": "tools", "input": "weather in Melbourne?", "tools": tools
        }),
        None,
    )
    .await;
    let (status, compacted) = post_at(
        &router,
        "/v1/responses/compact",
        json!({
            "model": "tools", "previous_response_id": first["id"]
        }),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{compacted}");
    assert_eq!(compacted["object"], "response.compaction");
    let mut input = compacted["output"].as_array().unwrap().clone();
    assert!(input.iter().any(|item| item["type"] == "compaction"));
    input
        .push(json!({"type": "function_call_output", "call_id": "call_weather", "output": "18 C"}));
    let (status, response) = post(
        &router,
        json!({
            "model": "tools", "tools": tools, "input": input
        }),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["status"], "completed");
    assert_eq!(response["previous_response_id"], Value::Null);
    let requests = backend.0.lock().unwrap();
    assert_eq!(requests.len(), 3);
    let history = serde_json::to_value(&requests[2].messages).unwrap();
    assert!(
        history
            .as_array()
            .unwrap()
            .iter()
            .any(|message| message["tool_calls"][0]["id"] == "call_weather")
    );
    assert!(
        history
            .as_array()
            .unwrap()
            .iter()
            .any(|message| message["role"] == "tool" && message["content"] == "18 C")
    );
    let prompt = messages_to_plain_prompt(&requests[2].messages);
    assert!(prompt.contains("weather in Melbourne?"));
    assert!(prompt.contains("saved answer"));
}

#[tokio::test]
async fn orphan_tool_output_is_rejected_before_backend_dispatch() {
    let backend = Arc::new(RecordingBackend::default());
    let router = router(backend.clone());
    let (status, error) = post(&router, json!({
        "model": "echo", "input": [{"type": "function_call_output", "call_id": "missing", "output": "untrusted result"}]
    }), None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error["error"]["param"], "input");
    assert!(backend.0.lock().unwrap().is_empty());
}
