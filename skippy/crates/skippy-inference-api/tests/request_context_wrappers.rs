use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::stream;
use serde_json::json;
use skippy_inference_api::{
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStream,
    CompactingInferenceBackend, CompactionConfig, GuardedInferenceBackend, GuardrailPolicy,
    HookedInferenceBackend, InferenceBackend, InferenceHookPolicy, InferenceRequestContext,
    InferenceResult, ModelObject, Usage, parse_request_id,
};

#[derive(Default)]
struct ContextCaptureBackend {
    contexts: Mutex<Vec<InferenceRequestContext>>,
}

#[async_trait]
impl InferenceBackend for ContextCaptureBackend {
    async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("context-model")])
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> InferenceResult<ChatCompletionResponse> {
        Ok(ChatCompletionResponse::new(
            request.model,
            "ok",
            Usage::new(1, 1),
        ))
    }

    async fn chat_completion_with_context(
        &self,
        request: ChatCompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<ChatCompletionResponse> {
        self.contexts.lock().expect("context lock").push(context);
        self.chat_completion(request).await
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<ChatCompletionStream> {
        self.contexts.lock().expect("context lock").push(context);
        Ok(Box::pin(stream::empty()))
    }
}

struct NoopHook;

#[async_trait]
impl InferenceHookPolicy for NoopHook {}

#[tokio::test]
async fn backend_wrapper_stack_preserves_authoritative_request_context() {
    let capture = Arc::new(ContextCaptureBackend::default());
    let compacting = Arc::new(CompactingInferenceBackend::new(
        capture.clone(),
        CompactionConfig::default(),
    ));
    let guarded = Arc::new(GuardedInferenceBackend::new(
        compacting,
        GuardrailPolicy::default(),
    ));
    let hooked = HookedInferenceBackend::new(guarded, Arc::new(NoopHook));
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "context-model",
        "messages": [{"role": "user", "content": "private prompt"}]
    }))
    .expect("request fixture");
    let unary_id = parse_request_id("cc752a7e-40ce-4708-be8c-ad2ca885d28a").expect("request ID");
    let stream_id = parse_request_id("f876529e-9800-445e-8f67-46326af341e6").expect("request ID");

    hooked
        .chat_completion_with_context(
            request.clone(),
            InferenceRequestContext::with_request_id(unary_id),
        )
        .await
        .expect("unary wrapper call");
    let _stream = hooked
        .chat_completion_stream(
            request,
            InferenceRequestContext::with_request_id(stream_id).with_stream_usage_observation(),
        )
        .await
        .expect("stream wrapper call");

    let contexts = capture.contexts.lock().expect("context lock");
    assert_eq!(contexts.len(), 2);
    assert_eq!(contexts[0].request_id(), Some(unary_id));
    assert!(!contexts[0].observes_stream_usage());
    assert_eq!(contexts[1].request_id(), Some(stream_id));
    assert!(contexts[1].observes_stream_usage());
}
