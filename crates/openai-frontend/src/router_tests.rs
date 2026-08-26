use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
};
use futures_util::stream;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use super::*;

#[test]
fn trusted_agent_session_header_parser_accepts_valid_names() {
    assert_eq!(
        parse_agent_session_header("x-litellm-session-id"),
        Some(HeaderName::from_static("x-litellm-session-id"))
    );
}

#[test]
fn trusted_agent_session_header_parser_rejects_invalid_names() {
    assert!(parse_agent_session_header("not a header").is_none());
}

#[test]
fn backend_timeout_parser_accepts_whole_seconds() {
    assert_eq!(
        parse_backend_timeout_secs("900"),
        Some(Some(Duration::from_secs(900)))
    );
    assert_eq!(
        parse_backend_timeout_secs(" 900 "),
        Some(Some(Duration::from_secs(900)))
    );
}

#[test]
fn backend_timeout_parser_treats_zero_as_disabled() {
    assert_eq!(parse_backend_timeout_secs("0"), Some(None));
}

#[test]
fn backend_timeout_parser_rejects_non_numeric_values() {
    assert!(parse_backend_timeout_secs("10m").is_none());
    assert!(parse_backend_timeout_secs("-1").is_none());
    assert!(parse_backend_timeout_secs("").is_none());
}

#[test]
fn default_backend_timeout_exceeds_a_cold_large_prompt_prefill() {
    // A 60k-token cold prefill on a single Apple-silicon host measures ~250s.
    // The default must leave headroom above that, not abort it.
    assert!(OpenAiFrontendConfig::DEFAULT_BACKEND_TIMEOUT >= Duration::from_secs(600));
}
use crate::{
    FinishReason,
    backend::{
        CancellationToken, ChatCompletionStream, CompletionStream, OpenAiRequestContext,
        OpenAiResult,
    },
    chat::{
        AssistantMessage, ChatCompletionChoice, ChatCompletionResponse, ChatMessage,
        MessageContent, MessageContentPart, messages_to_plain_prompt,
    },
    common::Usage,
    completions::{CompletionPrompt, CompletionResponse},
    errors::{OpenAiErrorKind, already_openai_error, map_upstream_error_body},
    guardrails::{GuardedOpenAiBackend, GuardrailMode, GuardrailPolicy},
    lifecycle::{OpenAiFailure, OpenAiTerminalResult},
    models::ModelObject,
};

struct FakeBackend;

#[derive(Default)]
struct GuardrailRescueBackend {
    seen_chat_requests: Mutex<Vec<ChatCompletionRequest>>,
}

#[async_trait]
impl OpenAiBackend for GuardrailRescueBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("Qwen3-8B-Q4_K_M")])
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        self.seen_chat_requests
            .lock()
            .unwrap()
            .push(request.clone());
        Ok(ChatCompletionResponse {
            id: "chatcmpl_guarded_tool".to_string(),
            object: "chat.completion",
            created: 123,
            model: request.model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(json!([{
                        "id": "call_lookup",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"city\":\"Sydney\"}"
                        }
                    }])),
                },
                logprobs: None,
                finish_reason: Some(FinishReason::ToolCalls),
            }],
            usage: Usage::new(3, 2),
            timings: None,
        })
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        unreachable!("guardrail rescue test uses non-streaming requests")
    }

    async fn completion(&self, _request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        unreachable!("guardrail rescue test only calls chat")
    }

    async fn completion_stream(
        &self,
        _request: CompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        unreachable!("guardrail rescue test only calls chat")
    }
}

fn guarded_test_app(backend: Arc<GuardrailRescueBackend>) -> Router {
    let guarded = Arc::new(GuardedOpenAiBackend::new(
        backend,
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            ..GuardrailPolicy::default()
        },
    ));
    router_for_with_config(
        guarded,
        OpenAiFrontendConfig::default().without_backend_timeout(),
    )
}

#[derive(Default)]
struct RecordingLifecycleObserver {
    events: Mutex<Vec<OpenAiLifecycleEvent>>,
}

impl RecordingLifecycleObserver {
    fn events(&self) -> Vec<OpenAiLifecycleEvent> {
        self.events
            .lock()
            .expect("lifecycle observer lock poisoned")
            .clone()
    }
}

impl OpenAiLifecycleObserver for RecordingLifecycleObserver {
    fn observe(&self, event: &OpenAiLifecycleEvent) {
        self.events
            .lock()
            .expect("lifecycle observer lock poisoned")
            .push(event.clone());
    }
}

fn assert_stream_terminal_usage(
    observer: &RecordingLifecycleObserver,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
) {
    assert!(observer.events().iter().any(|event| matches!(
        event,
        OpenAiLifecycleEvent::StreamTerminal {
            result: OpenAiTerminalResult::CompletedWithUsage {
                status_code: 200,
                usage: TokenUsage {
                    prompt_tokens: Some(prompt),
                    cached_prompt_tokens: _,
                    completion_tokens: Some(completion),
                    total_tokens: Some(total),
                },
            },
            ..
        } if *prompt == prompt_tokens
            && *completion == completion_tokens
            && *total == total_tokens
    )));
}

#[async_trait]
impl OpenAiBackend for FakeBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("org/repo:Q4_K_M")])
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        if request.model == "missing" {
            return Err(OpenAiError::model_not_found(request.model));
        }
        if request.model == "unsupported-feature" {
            return Err(OpenAiError::unsupported(
                "structured output is parsed but not yet implemented by skippy runtime",
            ));
        }
        if request.model == "tool-call" {
            return Ok(ChatCompletionResponse {
                id: "chatcmpl_tool".to_string(),
                object: "chat.completion",
                created: 123,
                model: request.model,
                choices: vec![ChatCompletionChoice {
                    index: 0,
                    message: AssistantMessage {
                        role: "assistant",
                        content: Some("calling lookup".to_string()),
                        reasoning_content: None,
                        tool_calls: Some(json!([{
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"city\":\"Sydney\"}"
                            }
                        }])),
                    },
                    logprobs: Some(json!({
                        "content": [{
                            "token": "calling",
                            "logprob": -0.2
                        }]
                    })),
                    finish_reason: Some(FinishReason::ToolCalls),
                }],
                usage: Usage::new(3, 2),
                timings: None,
            });
        }
        Ok(ChatCompletionResponse::new(
            request.model,
            format!("echo: {}", messages_to_plain_prompt(&request.messages)),
            Usage::new(3, 2),
        ))
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        if request.model == "missing" {
            return Err(OpenAiError::model_not_found(request.model));
        }
        if request.model == "context-overflow" {
            return Err(OpenAiError::context_length_exceeded(
                "prompt tokens plus requested completion exceed context window",
            ));
        }
        if request.model == "stream-error" {
            return Ok(Box::pin(stream::iter(vec![Err(OpenAiError::backend(
                "stream backend failed",
            ))])));
        }
        if request.model == "stream-logprobs" {
            let model = request.model;
            return Ok(Box::pin(stream::iter(vec![
                Ok(ChatCompletionChunk {
                    id: "chatcmpl_stream_logprobs".to_string(),
                    object: "chat.completion.chunk",
                    created: 123,
                    model: model.clone(),
                    choices: vec![crate::chat::ChatCompletionChunkChoice {
                        index: 0,
                        delta: crate::chat::ChatCompletionDelta {
                            role: None,
                            content: Some("tok".to_string()),
                            reasoning_content: None,
                            tool_calls: None,
                        },
                        logprobs: Some(json!({
                            "content": [{
                                "token": "tok",
                                "logprob": -0.1
                            }]
                        })),
                        finish_reason: None,
                    }],
                    usage: None,
                }),
                Ok(ChatCompletionChunk::done(model)),
            ])));
        }
        let model = request.model;
        Ok(Box::pin(stream::iter(vec![
            Ok(ChatCompletionChunk::delta(model.clone(), "hel")),
            Ok(ChatCompletionChunk::delta(model.clone(), "lo")),
            Ok(ChatCompletionChunk::usage(model.clone(), Usage::new(3, 2))),
            Ok(ChatCompletionChunk::done_with_reason(
                model,
                FinishReason::Length,
            )),
        ])))
    }

    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        Ok(CompletionResponse::new(
            request.model,
            format!("echo: {}", request.prompt.text_lossy()),
            Usage::new(2, 1),
        ))
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        let model = request.model;
        Ok(Box::pin(stream::iter(vec![
            Ok(crate::CompletionChunk::delta(model.clone(), "a")),
            Ok(crate::CompletionChunk::usage(
                model.clone(),
                Usage::new(2, 1),
            )),
            Ok(crate::CompletionChunk::done_with_reason(
                model,
                FinishReason::Length,
            )),
        ])))
    }
}

struct SlowBackend;

#[async_trait]
impl OpenAiBackend for SlowBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        tokio::time::sleep(Duration::from_millis(50)).await;
        Ok(vec![ModelObject::new("slow-model")])
    }

    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        unreachable!("slow backend test only calls readiness")
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        unreachable!("slow backend test only calls readiness")
    }

    async fn completion(&self, _request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        unreachable!("slow backend test only calls readiness")
    }

    async fn completion_stream(
        &self,
        _request: CompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        unreachable!("slow backend test only calls readiness")
    }
}

struct CancellationBackend {
    token: Arc<Mutex<Option<CancellationToken>>>,
}

#[derive(Default)]
struct SessionCaptureBackend {
    requests: Arc<Mutex<Vec<ChatCompletionRequest>>>,
}

#[async_trait]
impl OpenAiBackend for SessionCaptureBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("capture-model")])
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        self.requests.lock().unwrap().push(request.clone());
        Ok(ChatCompletionResponse::new(
            request.model,
            "ok",
            Usage::new(1, 1),
        ))
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        unreachable!("agent-session tests use non-streaming requests")
    }
}
#[async_trait]
impl OpenAiBackend for CancellationBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("cancel-model")])
    }

    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        unreachable!("cancellation backend test only calls streaming")
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        *self.token.lock().expect("token lock poisoned") = Some(context.cancellation_token());
        Ok(Box::pin(stream::pending()))
    }
}

#[test]
fn messages_to_plain_prompt_extracts_text_parts() {
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: Some(MessageContent::Text("system text".to_string())),
            extra: Default::default(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Parts(vec![MessageContentPart {
                content_type: "text".to_string(),
                text: Some("part text".to_string()),
                extra: Default::default(),
            }])),
            extra: Default::default(),
        },
    ];
    assert_eq!(
        messages_to_plain_prompt(&messages),
        "system text\npart text"
    );
}

#[test]
fn max_completion_tokens_takes_precedence() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
        "max_completion_tokens": 3
    }))
    .unwrap();
    assert_eq!(request.effective_max_tokens(), Some(3));
}

#[test]
fn completion_prompt_text_lossy_for_string_arrays() {
    assert_eq!(
        CompletionPrompt::ManyText(vec!["one".to_string(), "two".to_string()]).text_lossy(),
        "one\ntwo"
    );
}

#[test]
fn strict_error_body_uses_openai_shape() {
    let error = OpenAiError::from_kind(
        StatusCode::SERVICE_UNAVAILABLE,
        OpenAiErrorKind::ServiceUnavailable,
        "upstream down",
    );
    let value = serde_json::to_value(error.body()).unwrap();
    assert_eq!(value["error"]["message"], "upstream down");
    assert_eq!(value["error"]["type"], "server_error");
    assert_eq!(value["error"]["code"], "service_unavailable");
}

#[test]
fn upstream_error_body_maps_llama_error_shape() {
    let body = br#"{"type":"exceed_context_size_error","message":"too long"}"#;
    let mapped = map_upstream_error_body(400, body).unwrap();
    let value: Value = serde_json::from_slice(&mapped).unwrap();
    assert_eq!(value["error"]["message"], "too long");
    assert_eq!(value["error"]["type"], "invalid_request_error");
    assert_eq!(value["error"]["code"], "context_length_exceeded");
}

#[test]
fn upstream_error_body_maps_legacy_string_error_shape() {
    let body = br#"{"error":"skippy ABI call failed: Unsupported"}"#;
    let mapped = map_upstream_error_body(503, body).unwrap();
    let value: Value = serde_json::from_slice(&mapped).unwrap();
    assert_eq!(
        value["error"]["message"],
        "skippy ABI call failed: Unsupported"
    );
    assert_eq!(value["error"]["type"], "server_error");
    assert_eq!(value["error"]["code"], "service_unavailable");
}

#[test]
fn already_openai_error_passthrough_is_detected() {
    let value = json!({
        "error": {
            "message": "bad request",
            "type": "invalid_request_error",
            "param": null,
            "code": "invalid_value"
        }
    });
    assert!(already_openai_error(&value));
    let body = serde_json::to_vec(&value).unwrap();
    assert_eq!(map_upstream_error_body(400, &body), None);
}

#[tokio::test]
async fn models_route_returns_model_list() {
    let app = router_for(Arc::new(FakeBackend));
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "org/repo:Q4_K_M");
}

#[tokio::test]
async fn health_route_returns_liveness_probe() {
    let app = router_for(Arc::new(FakeBackend));
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn healthz_observer_records_completed_terminal() {
    let observer = Arc::new(RecordingLifecycleObserver::default());
    let app = router_for_with_config(
        Arc::new(FakeBackend),
        OpenAiFrontendConfig::default().with_lifecycle_observer(observer.clone()),
    );
    let response = app
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .header("x-request-id", "00000000-0000-4000-8000-000000000021")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let events = observer.events();
    assert!(matches!(
        events.as_slice(),
        [
            OpenAiLifecycleEvent::Admitted { context },
            OpenAiLifecycleEvent::NonStreamTerminal { context: terminal_context, result: OpenAiTerminalResult::Completed { status_code: 200 } },
        ] if context.route == OpenAiFrontendRoute::Healthz
            && context.method == OpenAiRequestMethod::Get
            && context == terminal_context
    ));
}

#[tokio::test]
async fn readiness_route_checks_backend_models() {
    let app = router_for(Arc::new(FakeBackend));
    let response = app
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["status"], "ready");
}

#[tokio::test]
async fn backend_timeout_returns_openai_error_shape() {
    let app = router_for_with_config(
        Arc::new(SlowBackend),
        OpenAiFrontendConfig::default().with_backend_timeout(Duration::from_millis(1)),
    );
    let response = app
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    let body = response_body_json(response).await;
    assert_eq!(body["error"]["type"], "server_error");
    assert_eq!(body["error"]["code"], "timeout");
}

#[tokio::test]
async fn timeout_observer_records_failed_terminal() {
    let observer = Arc::new(RecordingLifecycleObserver::default());
    let app = router_for_with_config(
        Arc::new(SlowBackend),
        OpenAiFrontendConfig::default()
            .with_backend_timeout(Duration::from_millis(1))
            .with_lifecycle_observer(observer.clone()),
    );
    let response = app
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .header("x-request-id", "00000000-0000-4000-8000-000000000022")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    let events = observer.events();
    assert!(matches!(
        events.as_slice(),
        [
            OpenAiLifecycleEvent::Admitted { context },
            OpenAiLifecycleEvent::BackendDispatched { context: dispatched_context, operation: OpenAiBackendOperation::Models },
            OpenAiLifecycleEvent::BackendTerminal {
                context: backend_terminal_context,
                operation: OpenAiBackendOperation::Models,
                result: OpenAiTerminalResult::Failed {
                    status_code: 504,
                    failure: OpenAiFailure::Timeout,
                },
            },
            OpenAiLifecycleEvent::NonStreamTerminal { context: terminal_context, result: OpenAiTerminalResult::Failed { status_code: 504, failure: OpenAiFailure::Timeout } },
        ] if context.route == OpenAiFrontendRoute::Readyz
            && context.method == OpenAiRequestMethod::Get
            && context == dispatched_context
            && context == backend_terminal_context
            && context == terminal_context
    ));
}

#[tokio::test]

async fn configured_trusted_header_reaches_backend_as_agent_session_identity() {
    let backend = Arc::new(SessionCaptureBackend::default());
    let app = router_for_with_config(
        backend.clone(),
        OpenAiFrontendConfig::default()
            .with_agent_session_header(HeaderName::from_static("x-litellm-session-id")),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("x-litellm-session-id", "agent-thread-42")
                .body(Body::from(
                    json!({
                        "model": "capture-model",
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let requests = backend.requests.lock().unwrap();
    assert_eq!(requests[0].agent_session(), Some("agent-thread-42"));
    assert_eq!(
        requests[0].agent_session_source(),
        Some("x-litellm-session-id")
    );
}

#[tokio::test]
async fn unconfigured_session_header_is_ignored() {
    let backend = Arc::new(SessionCaptureBackend::default());
    let app = router_for(backend.clone());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("x-litellm-session-id", "untrusted-thread")
                .body(Body::from(
                    json!({
                        "model": "capture-model",
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        backend.requests.lock().unwrap()[0]
            .agent_session()
            .is_none()
    );
}

#[tokio::test]
async fn responses_conversation_is_normalized_without_leaking_into_chat_body() {
    let backend = Arc::new(SessionCaptureBackend::default());
    let app = router_for(backend.clone());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "capture-model",
                        "conversation": {"id": "conversation-7"},
                        "input": "hello"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let requests = backend.requests.lock().unwrap();
    assert_eq!(requests[0].agent_session(), Some("conversation-7"));
    assert_eq!(
        requests[0].agent_session_source(),
        Some("responses.conversation")
    );
    assert!(!requests[0].extra.contains_key("conversation"));
}

#[tokio::test]
async fn conflicting_header_and_responses_conversation_fail_closed() {
    let backend = Arc::new(SessionCaptureBackend::default());
    let app = router_for_with_config(
        backend.clone(),
        OpenAiFrontendConfig::default()
            .with_agent_session_header(HeaderName::from_static("x-litellm-session-id")),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .header("x-litellm-session-id", "header-session")
                .body(Body::from(
                    json!({
                        "model": "capture-model",
                        "conversation": {"id": "body-session"},
                        "input": "hello"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(backend.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn request_id_is_returned_on_success_and_errors() {
    let app = router_for(Arc::new(FakeBackend));
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("x-request-id", "2a36d783-d345-4a23-87a6-302b3a6896e1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.headers()["x-request-id"],
        "2a36d783-d345-4a23-87a6-302b3a6896e1"
    );

    let app = router_for(Arc::new(FakeBackend));
    let response = app
        .oneshot(
            Request::builder()
                .uri("/not-here")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(response.headers().get("x-request-id").is_some());
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn unknown_routes_return_openai_error_shape() {
    let app = router_for(Arc::new(FakeBackend));
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/unknown")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = response_body_json(response).await;
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["code"], "not_found");
}

#[tokio::test]
async fn unsupported_methods_return_openai_error_shape() {
    for (method, path) in [
        ("POST", "/v1/models"),
        ("GET", "/v1/chat/completions"),
        ("GET", "/v1/completions"),
    ] {
        let app = router_for(Arc::new(FakeBackend));
        let response = app
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        let body = response_body_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "method_not_allowed");
    }
}

#[tokio::test]
async fn chat_completion_route_returns_openai_shape() {
    let response = post_json(
        "/v1/chat/completions",
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], "echo: hi");
    assert_eq!(body["usage"]["total_tokens"], 5);
}

#[tokio::test]
async fn chat_completion_lifecycle_reports_authoritative_usage() {
    let observer = Arc::new(RecordingLifecycleObserver::default());
    let app = router_for_with_config(
        Arc::new(FakeBackend),
        OpenAiFrontendConfig::default().with_lifecycle_observer(observer.clone()),
    );
    let response = post_json_with_app_and_request_id(
        app,
        "/v1/chat/completions",
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}]
        }),
        Some("00000000-0000-4000-8000-000000000031"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    assert!(observer.events().iter().any(|event| matches!(
        event,
        OpenAiLifecycleEvent::NonStreamTerminal {
            result: OpenAiTerminalResult::CompletedWithUsage {
                status_code: 200,
                usage: TokenUsage {
                    prompt_tokens: Some(3),
                    cached_prompt_tokens: None,
                    completion_tokens: Some(2),
                    total_tokens: Some(5),
                },
            },
            ..
        }
    )));
}

#[tokio::test]
async fn chat_completion_stream_route_returns_sse() {
    let observer = Arc::new(RecordingLifecycleObserver::default());
    let app = router_for_with_config(
        Arc::new(FakeBackend),
        OpenAiFrontendConfig::default().with_lifecycle_observer(observer.clone()),
    );
    let response = post_json_with_app_and_request_id(
        app,
        "/v1/chat/completions",
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        }),
        Some("00000000-0000-4000-8000-000000000034"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_text(response).await;
    assert!(body.contains(r#""role":"assistant""#));
    assert!(body.contains(r#""content":"hel""#));
    assert!(body.contains(r#""finish_reason":"length""#));
    assert!(body.contains(r#""total_tokens":5"#));
    assert!(body.contains("data: [DONE]"));
    assert_stream_terminal_usage(&observer, 3, 2, 5);
}

#[tokio::test]
async fn chat_completion_stream_suppresses_usage_unless_requested() {
    let observer = Arc::new(RecordingLifecycleObserver::default());
    let app = router_for_with_config(
        Arc::new(FakeBackend),
        OpenAiFrontendConfig::default().with_lifecycle_observer(observer.clone()),
    );
    let response = post_json_with_app_and_request_id(
        app,
        "/v1/chat/completions",
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }),
        Some("00000000-0000-4000-8000-000000000032"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_text(response).await;
    assert!(!body.contains(r#""total_tokens":5"#));
    assert!(body.contains("data: [DONE]"));
    assert_stream_terminal_usage(&observer, 3, 2, 5);
}

#[tokio::test]
async fn chat_completion_stream_maps_pre_stream_context_overflow() {
    let response = post_json(
        "/v1/chat/completions",
        json!({
            "model": "context-overflow",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 768,
            "stream": true
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_body_json(response).await;
    assert_eq!(body["error"]["code"], "context_length_exceeded");
}

#[tokio::test]
async fn chat_completion_stream_frames_backend_errors() {
    let response = post_json(
        "/v1/chat/completions",
        json!({
            "model": "stream-error",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_text(response).await;
    assert!(body.contains(r#""error":{"#));
    assert!(body.contains(r#""code":"service_unavailable""#));
    assert!(body.contains("data: [DONE]"));
}

#[tokio::test]
async fn responses_stream_frames_backend_errors_without_completed_tail() {
    let response = post_json(
        "/v1/responses",
        json!({
            "model": "stream-error",
            "input": "hi",
            "stream": true
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_text(response).await;
    assert!(body.contains("event: error"));
    assert!(body.contains(r#""message":"stream backend failed""#));
    assert!(body.contains(r#""code":"service_unavailable""#));
    assert!(!body.contains("event: response.completed"));
    assert!(body.contains("data: [DONE]"));
}

#[tokio::test]
async fn dropping_stream_response_cancels_request_context() {
    let token = Arc::new(Mutex::new(None));
    let app = router(Arc::new(CancellationBackend {
        token: token.clone(),
    }));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "cancel-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let cancellation = token
        .lock()
        .expect("token lock poisoned")
        .clone()
        .expect("backend saw request context");
    assert!(!cancellation.is_cancelled());

    drop(response);
    tokio::task::yield_now().await;

    assert!(cancellation.is_cancelled());
}

#[tokio::test]
async fn chat_completion_route_maps_backend_errors() {
    let response = post_json(
        "/v1/chat/completions",
        json!({
            "model": "missing",
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = response_body_json(response).await;
    assert_eq!(body["error"]["code"], "model_not_found");
}

#[tokio::test]
async fn chat_completion_route_rejects_empty_messages() {
    let response = post_json(
        "/v1/chat/completions",
        json!({
            "model": "test-model",
            "messages": []
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_body_json(response).await;
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn chat_completion_route_rejects_multiple_choices_until_supported() {
    let response = post_json(
        "/v1/chat/completions",
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "n": 2
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_body_json(response).await;
    assert_eq!(body["error"]["code"], "unsupported_model_feature");
}

#[tokio::test]
async fn chat_completion_route_accepts_tools_structured_output_and_logprobs() {
    let response = post_json(
            "/v1/chat/completions",
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "tool_choice": "auto",
                "parallel_tool_calls": true,
                "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}},
                "logprobs": true,
                "top_logprobs": 2
            }),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn chat_completion_route_accepts_noop_parity_fields() {
    let response = post_json(
        "/v1/chat/completions",
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "n": 1,
            "tools": [],
            "response_format": {"type": "text"}
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn responses_route_translates_to_chat_and_back() {
    let response = post_json(
            "/v1/responses",
            json!({
                "model": "test-model",
                "instructions": "be concise",
                "input": "hi",
                "max_output_tokens": 12,
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}},
                "logprobs": true,
                "top_logprobs": 1
            }),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["object"], "response");
    assert_eq!(body["output_text"], "echo: be concise\nhi");
    assert_eq!(body["usage"]["input_tokens"], 3);
    assert_eq!(body["usage"]["output_tokens"], 2);
}

#[tokio::test]
async fn responses_route_preserves_tool_calls_and_logprobs() {
    let response = post_json(
        "/v1/responses",
        json!({
            "model": "tool-call",
            "input": "hi",
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "logprobs": true,
            "top_logprobs": 1
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["output_text"], "calling lookup");
    assert_eq!(
        body["output"][0]["content"][0]["logprobs"]["content"][0]["token"],
        "calling"
    );
    assert_eq!(body["output"][1]["type"], "function_call");
    assert_eq!(body["output"][1]["call_id"], "call_123");
    assert_eq!(body["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn responses_route_preserves_backend_unsupported_errors() {
    let response = post_json(
        "/v1/responses",
        json!({
            "model": "unsupported-feature",
            "input": "hi",
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {"type": "object"}}
            }
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_body_json(response).await;
    assert_eq!(body["error"]["code"], "unsupported_model_feature");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("structured output")
    );
}

#[tokio::test]
async fn guarded_chat_rescues_tool_call_text() {
    let backend = Arc::new(GuardrailRescueBackend::default());
    let app = guarded_test_app(backend.clone());

    let response = post_json_with_app_and_request_id(
        app,
        "/v1/chat/completions",
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "auto"
        }),
        Some("a35bb624-5c07-431f-92a9-9c884472ca95"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()["x-request-id"],
        "a35bb624-5c07-431f-92a9-9c884472ca95"
    );
    let body = response_body_json(response).await;
    assert_eq!(body["object"], "chat.completion");
    assert!(body["choices"][0]["message"]["content"].is_null());
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "lookup"
    );
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Sydney\"}"
    );
    assert!(!serde_json::to_string(&body).unwrap().contains("_mesh_"));

    let seen_requests = backend.seen_chat_requests.lock().unwrap();
    assert_eq!(seen_requests.len(), 1);
    let seen_tools = seen_requests[0]
        .tools
        .as_ref()
        .and_then(Value::as_array)
        .cloned()
        .expect("guarded backend should receive tools");
    assert_eq!(seen_tools.len(), 2);
    assert_eq!(seen_tools[0]["function"]["name"], "lookup");
    assert_eq!(seen_tools[1]["function"]["name"], "_mesh_respond");
}

#[tokio::test]
async fn responses_stream_route_returns_responses_sse() {
    let response = post_json(
        "/v1/responses",
        json!({
            "model": "test-model",
            "input": "hi",
            "stream": true,
            "stream_options": {"include_usage": true}
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_text(response).await;
    assert!(body.contains("event: response.created"));
    assert!(body.contains("event: response.output_item.added"));
    assert!(body.contains("event: response.content_part.added"));
    assert!(body.contains("event: response.output_text.delta"));
    assert!(body.contains("event: response.output_text.done"));
    assert!(body.contains("event: response.content_part.done"));
    assert!(body.contains("event: response.output_item.done"));
    assert!(body.contains("event: response.completed"));
    assert!(body.contains(r#""sequence_number":1"#));
    assert!(body.contains(r#""output_text":"hello""#));
    assert!(body.contains("data: [DONE]"));
}

#[tokio::test]
async fn responses_stream_route_preserves_logprobs() {
    let response = post_json(
        "/v1/responses",
        json!({
            "model": "stream-logprobs",
            "input": "hi",
            "stream": true,
            "logprobs": true
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_text(response).await;
    assert!(body.contains("event: response.output_text.delta"));
    assert!(body.contains(r#""logprobs":{"content":[{"logprob":-0.1,"token":"tok"}]}"#));
}

#[tokio::test]
async fn completion_route_returns_openai_shape() {
    let response = post_json(
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": "hi"
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["choices"][0]["text"], "echo: hi");
    assert_eq!(body["usage"]["total_tokens"], 3);
}

#[tokio::test]
async fn completion_stream_route_returns_sse() {
    let observer = Arc::new(RecordingLifecycleObserver::default());
    let app = router_for_with_config(
        Arc::new(FakeBackend),
        OpenAiFrontendConfig::default().with_lifecycle_observer(observer.clone()),
    );
    let response = post_json_with_app_and_request_id(
        app,
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": "hi",
            "stream": true,
            "stream_options": {"include_usage": true}
        }),
        Some("00000000-0000-4000-8000-000000000035"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_text(response).await;
    assert!(body.contains(r#""text":"a""#));
    assert!(body.contains(r#""finish_reason":"length""#));
    assert!(body.contains(r#""total_tokens":3"#));
    assert!(body.contains("data: [DONE]"));
    assert_stream_terminal_usage(&observer, 2, 1, 3);
}

#[tokio::test]
async fn completion_stream_suppresses_usage_unless_requested() {
    let observer = Arc::new(RecordingLifecycleObserver::default());
    let app = router_for_with_config(
        Arc::new(FakeBackend),
        OpenAiFrontendConfig::default().with_lifecycle_observer(observer.clone()),
    );
    let response = post_json_with_app_and_request_id(
        app,
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": "hi",
            "stream": true
        }),
        Some("00000000-0000-4000-8000-000000000033"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_text(response).await;
    assert!(!body.contains(r#""total_tokens":3"#));
    assert!(body.contains("data: [DONE]"));
    assert_stream_terminal_usage(&observer, 2, 1, 3);
}

#[tokio::test]
async fn completion_route_rejects_empty_prompt() {
    let response = post_json(
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": ""
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_body_json(response).await;
    assert_eq!(body["error"]["code"], "invalid_value");
}

#[tokio::test]
async fn completion_route_accepts_token_prompts() {
    let response = post_json(
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": [1, 2, 3]
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn completion_route_accepts_logprobs() {
    let response = post_json(
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": "hi",
            "logprobs": 2
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn completion_route_accepts_single_choice_controls() {
    let response = post_json(
        "/v1/completions",
        json!({
            "model": "test-model",
            "prompt": "hi",
            "n": 1,
            "best_of": 1
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn missing_content_type_maps_to_strict_error_shape() {
    let app = router_for(Arc::new(FakeBackend));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_body_json(response).await;
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn invalid_json_maps_to_strict_error_shape() {
    let app = router_for(Arc::new(FakeBackend));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from("{"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_body_json(response).await;
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn oversized_json_maps_to_strict_error_shape() {
    let app = router_for_with_config(
        Arc::new(FakeBackend),
        OpenAiFrontendConfig::default().with_max_request_body_bytes(64),
    );
    let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "this body is intentionally much larger than sixty four bytes"}]
                    }).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let body = response_body_json(response).await;
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["code"], "payload_too_large");
}

async fn post_json(path: &str, value: Value) -> axum::response::Response {
    post_json_with_app_and_request_id(router_for(Arc::new(FakeBackend)), path, value, None).await
}

async fn post_json_with_app_and_request_id(
    app: Router,
    path: &str,
    value: Value,
    request_id: Option<&str>,
) -> axum::response::Response {
    let mut builder = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json");
    if let Some(request_id) = request_id {
        builder = builder.header("x-request-id", request_id);
    }
    app.oneshot(builder.body(Body::from(value.to_string())).unwrap())
        .await
        .unwrap()
}

async fn response_body_json(response: axum::response::Response) -> Value {
    let body = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&body).unwrap()
}

async fn response_body_text(response: axum::response::Response) -> String {
    let body = response.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(body.to_vec()).unwrap()
}
