//! Contract tests for the Anthropic `/v1/messages` surface.
//!
//! Same shape as `lifecycle_observer.rs` / `benchy_contract.rs`: a mock
//! [`OpenAiBackend`] behind the real router, exercised via
//! `tower::ServiceExt::oneshot`, asserting the wire protocol end to end
//! (request translation, response envelope, SSE event sequence, tool
//! round-trips, and error bodies).

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use futures_util::stream;
use http_body_util::BodyExt;
use openai_frontend::{
    ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionDelta, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionStream, FinishReason, ModelObject, OpenAiBackend,
    OpenAiFrontendConfig, OpenAiRequestContext, OpenAiResult, Usage, router_for_with_config,
};
use serde_json::{Value, json};
use tower::ServiceExt;

const MODEL_ID: &str = "org/repo:Q4_K_M";

#[derive(Default)]
struct RecordingBackend {
    seen_requests: Mutex<Vec<ChatCompletionRequest>>,
    stream_responses: Mutex<Vec<Vec<OpenAiResult<ChatCompletionChunk>>>>,
}

impl RecordingBackend {
    /// Queue a chunk script for the next streaming call (LIFO, mirroring the
    /// mock style used in `benchy_contract.rs`).
    fn queue_stream(&self, chunks: Vec<OpenAiResult<ChatCompletionChunk>>) {
        self.stream_responses.lock().expect("lock").push(chunks);
    }
}

#[async_trait]
impl OpenAiBackend for RecordingBackend {
    async fn count_chat_tokens(&self, request: ChatCompletionRequest) -> OpenAiResult<u32> {
        self.seen_requests.lock().expect("lock").push(request);
        Ok(42)
    }

    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new(MODEL_ID)])
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        self.seen_requests.lock().expect("lock").push(request);
        let mut response = ChatCompletionResponse::new(
            MODEL_ID,
            "The secret word is OBSIDIAN-4.",
            Usage::new(15_033, 9),
        );
        let _ = &mut response;
        Ok(response)
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        self.seen_requests.lock().expect("lock").push(request);
        let scripted = self
            .stream_responses
            .lock()
            .expect("lock")
            .pop()
            .unwrap_or_else(|| {
                vec![
                    Ok(ChatCompletionChunk::delta(MODEL_ID, "Hel")),
                    Ok(ChatCompletionChunk::delta(MODEL_ID, "lo")),
                ]
            });
        Ok(Box::pin(stream::iter(scripted)))
    }
}

fn app() -> axum::Router {
    router_for_with_config(
        Arc::new(RecordingBackend::default()),
        OpenAiFrontendConfig::default(),
    )
}

fn app_with(backend: RecordingBackend) -> axum::Router {
    router_for_with_config(Arc::new(backend), OpenAiFrontendConfig::default())
}

async fn post_json(uri: &str, body: Value) -> (StatusCode, Value) {
    let response = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request builds"),
        )
        .await
        .expect("response");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value)
}

/// Post a JSON body and get the raw response body text back — for SSE.
async fn post_stream_with(uri: &str, body: Value, app: axum::Router) -> (StatusCode, String) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request builds"),
        )
        .await
        .expect("response");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

fn messages_body() -> Value {
    json!({
        "model": MODEL_ID,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "What is the secret word?"}]
    })
}

#[tokio::test]
async fn non_streaming_response_matches_anthropic_envelope() {
    let (status, body) = post_json("/v1/messages", messages_body()).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["type"], json!("message"));
    assert_eq!(body["role"], json!("assistant"));
    assert_eq!(body["model"], json!(MODEL_ID));
    assert_eq!(body["content"][0]["type"], json!("text"));
    assert_eq!(
        body["content"][0]["text"],
        json!("The secret word is OBSIDIAN-4.")
    );
    assert_eq!(body["stop_reason"], json!("end_turn"));
    assert!(body["stop_sequence"].is_null());
    assert_eq!(body["usage"]["input_tokens"], json!(15_033));
    assert_eq!(body["usage"]["output_tokens"], json!(9));
    let id = body["id"].as_str().expect("id");
    assert!(id.starts_with("msg_"), "id should be msg_-prefixed: {id}");
}

#[tokio::test]
async fn streaming_response_emits_anthropic_sse_sequence() {
    let backend = RecordingBackend::default();
    let (status, body) = {
        let app = app_with(backend);
        // Streamed through the same router; read the SSE body raw.
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": MODEL_ID,
                            "max_tokens": 16,
                            "stream": true,
                            "messages": [{"role": "user", "content": "hi"}]
                        })
                        .to_string(),
                    ))
                    .expect("request builds"),
            )
            .await
            .expect("response");
        let status = response.status();
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("body")
            .to_bytes();
        (status, String::from_utf8_lossy(&bytes).to_string())
    };
    assert_eq!(status, StatusCode::OK, "body: {body}");

    let event_blocks: Vec<&str> = body
        .split("\n\n")
        .filter(|block| block.contains("event:"))
        .collect();
    let event_names: Vec<&str> = event_blocks
        .iter()
        .filter_map(|block| {
            block.lines().find_map(|line| {
                line.strip_prefix("event: ")
                    .map(str::trim)
                    .map(str::to_string)
                    .map(|v| -> &'static str {
                        match v.as_str() {
                            "message_start" => "message_start",
                            "content_block_start" => "content_block_start",
                            "content_block_delta" => "content_block_delta",
                            "content_block_stop" => "content_block_stop",
                            "message_delta" => "message_delta",
                            "message_stop" => "message_stop",
                            "ping" => "ping",
                            "error" => "error",
                            _ => "other",
                        }
                    })
            })
        })
        .collect();

    // Canonical ordering: message_start first, message_stop last, and no
    // deltas outside an open block.
    assert_eq!(event_names.first(), Some(&"message_start"));
    assert_eq!(event_names.last(), Some(&"message_stop"));
    assert!(event_names.contains(&"content_block_start"));
    assert!(event_names.contains(&"content_block_delta"));
    assert!(event_names.contains(&"content_block_stop"));
    assert!(event_names.contains(&"message_delta"));

    // The text deltas carry the streamed content.
    assert!(body.contains("text_delta"), "deltas present: {body}");
    assert!(body.contains("Hel"), "first fragment present: {body}");
    assert!(body.contains("lo"), "second fragment present: {body}");

    // message_delta carries a stop_reason.
    let delta_block = event_blocks
        .iter()
        .find(|block| block.contains("message_delta"))
        .expect("message_delta block");
    assert!(delta_block.contains("end_turn"), "{delta_block}");
}

#[tokio::test]
async fn tool_result_conversation_round_trips_through_chat_pipeline() {
    let body = json!({
        "model": MODEL_ID,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "weather in oslo?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"city": "oslo"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": [{"type": "text", "text": "sunny"}]}
            ]}
        ]
    });
    let (status, body) = post_json("/v1/messages", body).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["content"][0]["type"], json!("text"));
}

#[tokio::test]
async fn missing_max_tokens_is_an_anthropic_error() {
    let body = json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let (status, body) = post_json("/v1/messages", body).await;
    // max_tokens is required by the Anthropic contract; the serde type makes
    // it a 400 before the backend sees the request (matches api.anthropic.com,
    // which answers missing max_tokens with 400 invalid_request_error).
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
    assert_eq!(body["type"], json!("error"));
    assert_eq!(body["error"]["type"], json!("invalid_request_error"));
}

#[tokio::test]
async fn unknown_route_and_method_still_render_anthropic_errors() {
    let response = app()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/messages")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let error: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(error["type"], "error");
    let (status, error) = post_json("/v1/messages/nope", messages_body()).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error["type"], "error");
    assert_eq!(error["error"]["type"], "not_found_error");
}

#[tokio::test]
async fn count_tokens_uses_backend_without_generation_budget() {
    let mut request = messages_body();
    request.as_object_mut().unwrap().remove("max_tokens");
    let (status, body) = post_json("/v1/messages/count_tokens", request).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    let tokens = body["input_tokens"].as_u64().expect("input_tokens");
    assert_eq!(tokens, 42);
}

/// Build a chat chunk the way a provider emits streaming tool calls: an
/// optional id/name header on the first fragment, argument JSON split across
/// fragments afterwards.
fn tool_call_chunk(
    model: &str,
    id: Option<&str>,
    name: Option<&str>,
    arguments: &str,
) -> ChatCompletionChunk {
    let mut function = serde_json::Map::new();
    if let Some(name) = name {
        function.insert("name".to_string(), json!(name));
    }
    function.insert("arguments".to_string(), json!(arguments));
    let mut fragment = serde_json::Map::new();
    fragment.insert("index".to_string(), json!(0));
    if let Some(id) = id {
        fragment.insert("id".to_string(), json!(id));
    }
    fragment.insert("type".to_string(), json!("function"));
    fragment.insert("function".to_string(), Value::Object(function));
    ChatCompletionChunk {
        id: "chatcmpl-agent".to_string(),
        object: "chat.completion.chunk",
        created: 0,
        model: model.to_string(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: ChatCompletionDelta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: Some(Value::Array(vec![Value::Object(fragment)])),
            },
            logprobs: None,
            finish_reason: None,
        }],
        usage: None,
        timings: None,
    }
}

/// Collect the `event:` names of an SSE body in order.
fn sse_event_names(body: &str) -> Vec<&str> {
    body.split("\n\n")
        .filter(|block| block.contains("event:"))
        .filter_map(|block| block.lines().find_map(|line| line.strip_prefix("event: ")))
        .collect()
}

/// Simulated tool executor: the claude-code-style agent loop answers a
/// `tool_use` with a `tool_result` in the next user turn.
#[tokio::test]
async fn simulated_agent_tool_use_loop_round_trips_over_streaming_messages() {
    let backend = RecordingBackend::default();
    // LIFO queue: the turn-2 script goes in first so turn 1 pops first.
    backend.queue_stream(vec![
        Ok(ChatCompletionChunk::delta(MODEL_ID, "It is sunny in Oslo.")),
        Ok(ChatCompletionChunk::usage(MODEL_ID, Usage::new(34, 6))),
        Ok(ChatCompletionChunk::done_with_reason(
            MODEL_ID,
            FinishReason::Stop,
        )),
    ]);
    backend.queue_stream(vec![
        // Turn 1: the model reasons briefly, then requests the tool.
        Ok(ChatCompletionChunk::delta(MODEL_ID, "Checking ")),
        Ok(ChatCompletionChunk::delta(MODEL_ID, "the weather.")),
        Ok(tool_call_chunk(
            MODEL_ID,
            Some("toolu_01"),
            Some("get_weather"),
            "{\"ci",
        )),
        Ok(tool_call_chunk(MODEL_ID, None, None, "ty\":\"oslo\"}")),
        Ok(ChatCompletionChunk::usage(MODEL_ID, Usage::new(21, 14))),
        Ok(ChatCompletionChunk::done_with_reason(
            MODEL_ID,
            FinishReason::ToolCalls,
        )),
    ]);

    let app = app_with(backend);
    let agent_tools = json!([
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    ]);

    // --- Turn 1: agent sends system prompt + user task; model streams a
    // tool_use request. The stream must terminate with message_stop and
    // stop_reason `tool_use`.
    let turn_1 = json!({
        "model": MODEL_ID,
        "max_tokens": 1024,
        "system": "You are a careful weather assistant.",
        "tools": agent_tools,
        "tool_choice": {"type": "auto"},
        "stream": true,
        "messages": [
            {"role": "user", "content": "What is the weather in Oslo? Use the tool."}
        ]
    });
    let (status, body) = post_stream_with("/v1/messages", turn_1, app.clone()).await;
    assert_eq!(status, StatusCode::OK, "turn 1 body: {body}");

    let names = sse_event_names(&body);
    assert_eq!(names.first(), Some(&"message_start"), "{names:?}");
    assert_eq!(names.last(), Some(&"message_stop"), "{names:?}");
    let block_stops: Vec<usize> = names
        .iter()
        .enumerate()
        .filter(|(_, name)| **name == "content_block_stop")
        .map(|(position, _)| position)
        .collect();
    let message_delta = names
        .iter()
        .position(|name| *name == "message_delta")
        .expect("message_delta present");
    assert!(
        block_stops.iter().all(|stop| stop < &message_delta),
        "all blocks close before message_delta: {names:?}"
    );
    assert!(body.contains("\"stop_reason\":\"tool_use\""), "{body}");

    // Exactly one tool_use block whose assembled input JSON is complete.
    let tool_start = body
        .split("\n\n")
        .find(|block| block.contains("event: content_block_start") && block.contains("tool_use"))
        .expect("tool_use content_block_start")
        .to_string();
    assert!(tool_start.contains("\"id\":\"toolu_01\""), "{tool_start}");
    assert!(
        tool_start.contains("\"name\":\"get_weather\""),
        "{tool_start}"
    );
    let json_deltas: Vec<String> = body
        .split("\n\n")
        .filter(|block| block.contains("input_json_delta"))
        .map(str::to_string)
        .collect();
    assert_eq!(json_deltas.len(), 2, "both argument fragments: {body}");
    // Assemble the streamed fragments the way an SDK would and confirm the
    // reconstructed tool input is complete, valid JSON.
    let mut partial_json = String::new();
    for block in json_deltas {
        let data = block
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .expect("delta data line");
        let value: Value = serde_json::from_str(data).expect("delta is valid JSON");
        partial_json.push_str(
            value["delta"]["partial_json"]
                .as_str()
                .expect("partial_json"),
        );
    }
    assert_eq!(partial_json, r#"{"city":"oslo"}"#);

    // Usage lands in message_delta: prompt 21, completion 14.
    let message_delta_block = body
        .split("\n\n")
        .find(|block| block.contains("event: message_delta"))
        .expect("message_delta block");
    assert!(
        message_delta_block.contains("\"input_tokens\":21"),
        "{message_delta_block}"
    );
    assert!(
        message_delta_block.contains("\"output_tokens\":14"),
        "{message_delta_block}"
    );

    // --- Turn 2: the agent executes the tool and replays the conversation:
    // assistant tool_use turn + user tool_result turn, against the same app
    // (same backend, next queued script).
    let turn_2 = json!({
        "model": MODEL_ID,
        "max_tokens": 1024,
        "system": "You are a careful weather assistant.",
        "tools": agent_tools,
        "stream": true,
        "messages": [
            {"role": "user", "content": "What is the weather in Oslo? Use the tool."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Checking the weather."},
                {"type": "tool_use", "id": "toolu_01", "name": "get_weather",
                 "input": {"city": "oslo"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01",
                 "cache_control": {"type": "ephemeral"},
                 "content": [{"type": "text", "text": "sunny, 12C"}]}
            ]}
        ]
    });
    let (status, body) = post_stream_with("/v1/messages", turn_2, app).await;
    assert_eq!(status, StatusCode::OK, "turn 2 body: {body}");
    let names = sse_event_names(&body);
    assert_eq!(names.first(), Some(&"message_start"), "{names:?}");
    assert_eq!(names.last(), Some(&"message_stop"), "{names:?}");
    assert!(body.contains("\"stop_reason\":\"end_turn\""), "{body}");
    assert!(body.contains("It is sunny in Oslo."), "{body}");
}

#[test]
fn review_preserves_mesh_hooks() {
    let mut body = messages_body();
    body["mesh_hooks"] = json!(true);
    let req = openai_frontend::anthropic::messages_request_to_chat_request(
        serde_json::from_value(body).unwrap(),
    )
    .unwrap();
    assert!(
        openai_frontend::chat_mesh_hooks_enabled(&req),
        "mesh_hooks flag discarded"
    );
}
#[test]
fn review_preserves_image_content() {
    let mut body = messages_body();
    body["messages"][0]["content"] = json!([{"type":"text","text":"describe"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]);
    let req = openai_frontend::anthropic::messages_request_to_chat_request(
        serde_json::from_value(body).unwrap(),
    )
    .unwrap();
    let wire = format!("{:?}", req.messages);
    assert!(wire.to_string().contains("AAAA"), "image discarded: {wire}");
}
#[tokio::test]
async fn review_count_tokens_without_max_tokens() {
    let mut body = messages_body();
    body.as_object_mut().unwrap().remove("max_tokens");
    let (status, body) = post_json("/v1/messages/count_tokens", body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}
#[tokio::test]
async fn review_two_streamed_tools_remain_distinct() {
    let backend = RecordingBackend::default();
    let first = tool_call_chunk(MODEL_ID, Some("toolu_a"), Some("alpha"), "{}");
    let mut second = tool_call_chunk(MODEL_ID, Some("toolu_b"), Some("beta"), "{}");
    second.choices[0].delta.tool_calls.as_mut().unwrap()[0]["index"] = json!(1);
    backend.queue_stream(vec![
        Ok(first),
        Ok(second),
        Ok(ChatCompletionChunk::done_with_reason(
            MODEL_ID,
            FinishReason::ToolCalls,
        )),
    ]);
    let mut body = messages_body();
    body["stream"] = json!(true);
    let (_, wire) = post_stream_with("/v1/messages", body, app_with(backend)).await;
    let starts = wire
        .split("\n\n")
        .filter(|b| b.contains("event: content_block_start") && b.contains("tool_use"))
        .count();
    assert_eq!(starts, 2, "{wire}");
}
#[tokio::test]
async fn review_error_does_not_emit_success_end_turn() {
    let backend = RecordingBackend::default();
    backend.queue_stream(vec![
        Ok(ChatCompletionChunk::delta(MODEL_ID, "partial")),
        Err(openai_frontend::OpenAiError::internal("backend failed")),
        Err(openai_frontend::OpenAiError::internal("second failure")),
        Ok(ChatCompletionChunk::delta(MODEL_ID, "after failure")),
    ]);
    let mut body = messages_body();
    body["stream"] = json!(true);
    let (_, wire) = post_stream_with("/v1/messages", body, app_with(backend)).await;
    assert!(!wire.contains("\"stop_reason\":\"end_turn\""), "{wire}");
    assert_eq!(wire.matches("event: error").count(), 1, "{wire}");
    assert!(!wire.contains("after failure"), "{wire}");
}
#[tokio::test]
async fn review_usage_after_finish_is_reported() {
    let backend = RecordingBackend::default();
    backend.queue_stream(vec![
        Ok(ChatCompletionChunk::delta(MODEL_ID, "ok")),
        Ok(ChatCompletionChunk::done_with_reason(
            MODEL_ID,
            FinishReason::Stop,
        )),
        Ok(ChatCompletionChunk::usage(MODEL_ID, Usage::new(21, 14))),
    ]);
    let mut body = messages_body();
    body["stream"] = json!(true);
    let (_, wire) = post_stream_with("/v1/messages", body, app_with(backend)).await;
    assert!(wire.contains("\"output_tokens\":14"), "{wire}");
}

#[derive(Default)]
struct AnthropicObserver(Mutex<Vec<openai_frontend::OpenAiLifecycleEvent>>);
impl openai_frontend::OpenAiLifecycleObserver for AnthropicObserver {
    fn observe(&self, event: &openai_frontend::OpenAiLifecycleEvent) {
        self.0.lock().unwrap().push(event.clone());
    }
}
struct InjectingHook;
#[async_trait]
impl openai_frontend::OpenAiHookPolicy for InjectingHook {
    async fn before_chat_completion(
        &self,
        request: &mut ChatCompletionRequest,
    ) -> OpenAiResult<openai_frontend::ChatHookOutcome> {
        assert_eq!(request.extra.get("mesh_hooks"), Some(&json!(true)));
        Ok(openai_frontend::ChatHookOutcome::injected("hook-marker"))
    }
}

#[tokio::test]
async fn anthropic_and_chat_share_hooks_and_terminal_usage() {
    use openai_frontend::{HookedOpenAiBackend, OpenAiLifecycleEvent};
    let backend = Arc::new(RecordingBackend::default());
    let observer = Arc::new(AnthropicObserver::default());
    let app = router_for_with_config(
        Arc::new(HookedOpenAiBackend::new(
            backend.clone(),
            Arc::new(InjectingHook),
        )),
        OpenAiFrontendConfig::default().with_lifecycle_observer(observer.clone()),
    );
    for path in ["/v1/messages", "/v1/chat/completions"] {
        let body = json!({"model":MODEL_ID,"max_tokens":32,"mesh_hooks":true,"messages":[{"role":"user","content":"hello"}]});
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        response.into_body().collect().await.unwrap();
    }
    let requests = backend.seen_requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let messages: Vec<Value> = requests
        .iter()
        .map(|request| serde_json::to_value(&request.messages).unwrap())
        .collect();
    assert_eq!(messages[0], messages[1]);
    assert!(messages[0].to_string().contains("hook-marker"));
    let events = observer.0.lock().unwrap();
    let usages: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            OpenAiLifecycleEvent::ResponseCompleted { usage, .. } => Some(*usage),
            _ => None,
        })
        .collect();
    assert_eq!(usages.len(), 2);
    assert_eq!(usages[0], usages[1]);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, OpenAiLifecycleEvent::NonStreamTerminal { .. }))
            .count(),
        2
    );
}

#[tokio::test]
async fn unsupported_semantics_are_rejected_before_backend_dispatch() {
    for (key, value) in [("service_tier", json!("priority"))] {
        let mut body = json!({"model":MODEL_ID,"max_tokens":32,"messages":[{"role":"user","content":"hello"}]});
        body[key] = value;
        let (status, error) = post_json("/v1/messages", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
        assert_eq!(error["error"]["type"], "invalid_request_error");
        assert!(error["error"]["message"].as_str().unwrap().contains(key));
    }
}

#[tokio::test]
async fn claude_code_thinking_cache_and_context_defaults_are_accepted() {
    let body = json!({
        "model": MODEL_ID,
        "max_tokens": 32000,
        "stream": true,
        "thinking": {"type": "adaptive", "display": "omitted"},
        "context_management": {
            "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
        },
        "output_config": {"effort": "high"},
        "system": [{
            "type": "text",
            "text": "You are Claude Code.",
            "cache_control": {"type": "ephemeral"}
        }],
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "hello",
                "cache_control": {"type": "ephemeral"}
            }]
        }]
    });
    let (status, wire) = post_stream_with("/v1/messages?beta=true", body, app()).await;
    assert_eq!(status, StatusCode::OK, "{wire}");
    assert!(wire.contains("event: message_start"), "{wire}");
    assert!(wire.contains("event: message_stop"), "{wire}");
}

#[test]
fn translated_cached_usage_preserves_prompt_token_accounting() {
    let response = ChatCompletionResponse::new(
        MODEL_ID,
        "cached answer",
        Usage::new(20, 5).with_cached_tokens(12),
    );
    let translated =
        openai_frontend::anthropic::messages_response_from_chat_response(&response).unwrap();
    let usage = serde_json::to_value(translated.usage).unwrap();
    assert_eq!(
        usage,
        json!({"input_tokens":8,"cache_read_input_tokens":12,"output_tokens":5})
    );
}
