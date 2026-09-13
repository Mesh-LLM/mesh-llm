use std::sync::Arc;

use async_trait::async_trait;
use mesh_llm_guardrails::{
    CompactionConfig, CompactionOverride, CompactionRequest, MESH_COMPACT_FIELD, compact_messages,
};

use crate::{
    audio::{
        AudioResponse, AudioSpeechRequest, AudioTranscriptionRequest, AudioTranscriptionResponse,
    },
    backend::{
        ChatCompletionStream, CompletionStream, OpenAiBackend, OpenAiRequestContext, OpenAiResult,
    },
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    completions::{CompletionRequest, CompletionResponse},
    embeddings::{EmbeddingResponse, EmbeddingsRequest},
    errors::OpenAiError,
    models::ModelObject,
    rerank::{RerankRequest, RerankResponse},
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
        self.chat_completion_with_context(request, OpenAiRequestContext::new())
            .await
    }

    async fn chat_completion_with_context(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionResponse> {
        self.backend
            .chat_completion_with_context(self.compact_request(request)?, context)
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
        self.completion_with_context(request, OpenAiRequestContext::new())
            .await
    }

    async fn completion_with_context(
        &self,
        request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionResponse> {
        self.backend.completion_with_context(request, context).await
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        self.backend.completion_stream(request, context).await
    }

    async fn embeddings(
        &self,
        request: EmbeddingsRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<EmbeddingResponse> {
        self.backend.embeddings(request, context).await
    }

    async fn rerank(
        &self,
        request: RerankRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<RerankResponse> {
        self.backend.rerank(request, context).await
    }

    async fn audio_speech(
        &self,
        request: AudioSpeechRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<AudioResponse> {
        self.backend.audio_speech(request, context).await
    }

    async fn audio_transcription(
        &self,
        request: AudioTranscriptionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<AudioTranscriptionResponse> {
        self.backend.audio_transcription(request, context).await
    }

    async fn audio_translation(
        &self,
        request: AudioTranscriptionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<AudioTranscriptionResponse> {
        self.backend.audio_translation(request, context).await
    }
}
