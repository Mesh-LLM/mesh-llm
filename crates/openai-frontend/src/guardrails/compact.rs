use std::sync::Arc;

use async_trait::async_trait;
use mesh_llm_guardrails::{
    CompactionConfig, CompactionOverride, CompactionRequest, MESH_COMPACT_FIELD, compact_messages,
};

use crate::{
    backend::{
        ChatCompletionStream, CompletionStream, OpenAiBackend, OpenAiRequestContext, OpenAiResult,
    },
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    completions::{CompletionRequest, CompletionResponse},
    errors::OpenAiError,
    models::ModelObject,
};

pub struct CompactingOpenAiBackend {
    backend: Arc<dyn OpenAiBackend>,
    config: CompactionConfig,
}

impl CompactingOpenAiBackend {
    pub fn new(backend: Arc<dyn OpenAiBackend>, config: CompactionConfig) -> Self {
        Self { backend, config }
    }

    fn compact_request(
        &self,
        mut request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionRequest> {
        let messages = request
            .messages
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| OpenAiError::internal(format!("serialize chat messages: {error}")))?;
        let override_value = CompactionOverride::from_value(request.extra.get(MESH_COMPACT_FIELD));
        let (messages, _report) = compact_messages(
            CompactionRequest {
                messages,
                override_value,
            },
            self.config,
        );
        request.messages = messages
            .into_iter()
            .map(serde_json::from_value)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                OpenAiError::internal(format!("deserialize compacted messages: {error}"))
            })?;
        Ok(request)
    }
}

#[async_trait]
impl OpenAiBackend for CompactingOpenAiBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        self.backend.models().await
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        self.backend
            .chat_completion(self.compact_request(request)?)
            .await
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        self.backend
            .chat_completion_stream(self.compact_request(request)?, context)
            .await
    }

    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        self.backend.completion(request).await
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        self.backend.completion_stream(request, context).await
    }
}
