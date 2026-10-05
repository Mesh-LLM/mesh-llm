use std::sync::Arc;

use async_trait::async_trait;
use skippy_guardrails::{
    CompactionConfig, CompactionOverride, CompactionRequest, MESH_COMPACT_FIELD, compact_messages,
};

use crate::{
    audio::{
        AudioResponse, AudioSpeechRequest, AudioTranscriptionRequest, AudioTranscriptionResponse,
    },
    backend::{
        ChatCompletionStream, CompletionStream, InferenceBackend, InferenceRequestContext,
        InferenceResult,
    },
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    completions::{CompletionRequest, CompletionResponse},
    embeddings::{EmbeddingResponse, EmbeddingsRequest},
    errors::InferenceError,
    models::ModelObject,
    rerank::{RerankRequest, RerankResponse},
    system_one::{SystemOneRequest, SystemOneResponse},
};

pub struct CompactingInferenceBackend {
    backend: Arc<dyn InferenceBackend>,
    config: CompactionConfig,
}

impl CompactingInferenceBackend {
    pub fn new(backend: Arc<dyn InferenceBackend>, config: CompactionConfig) -> Self {
        Self { backend, config }
    }

    fn compact_request(
        &self,
        mut request: ChatCompletionRequest,
    ) -> InferenceResult<ChatCompletionRequest> {
        let messages = request
            .messages
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                InferenceError::internal(format!("serialize chat messages: {error}"))
            })?;
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
                InferenceError::internal(format!("deserialize compacted messages: {error}"))
            })?;
        Ok(request)
    }
}

#[async_trait]
impl InferenceBackend for CompactingInferenceBackend {
    async fn count_chat_tokens(&self, request: ChatCompletionRequest) -> InferenceResult<u32> {
        self.backend.count_chat_tokens(request).await
    }

    async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
        self.backend.models().await
    }

    async fn system_one(&self, request: SystemOneRequest) -> InferenceResult<SystemOneResponse> {
        self.backend.system_one(request).await
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> InferenceResult<ChatCompletionResponse> {
        self.chat_completion_with_context(request, InferenceRequestContext::new())
            .await
    }

    async fn chat_completion_with_context(
        &self,
        request: ChatCompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<ChatCompletionResponse> {
        self.backend
            .chat_completion_with_context(self.compact_request(request)?, context)
            .await
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<ChatCompletionStream> {
        self.backend
            .chat_completion_stream(self.compact_request(request)?, context)
            .await
    }

    async fn completion(&self, request: CompletionRequest) -> InferenceResult<CompletionResponse> {
        self.completion_with_context(request, InferenceRequestContext::new())
            .await
    }

    async fn completion_with_context(
        &self,
        request: CompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<CompletionResponse> {
        self.backend.completion_with_context(request, context).await
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<CompletionStream> {
        self.backend.completion_stream(request, context).await
    }

    /// Forward embeddings and request context without chat processing.
    async fn embeddings(
        &self,
        request: EmbeddingsRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<EmbeddingResponse> {
        self.backend.embeddings(request, context).await
    }

    /// Forward reranking and request context without chat processing.
    async fn rerank(
        &self,
        request: RerankRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<RerankResponse> {
        self.backend.rerank(request, context).await
    }

    /// Forward speech generation and request context unchanged.
    async fn audio_speech(
        &self,
        request: AudioSpeechRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<AudioResponse> {
        self.backend.audio_speech(request, context).await
    }

    /// Forward multipart transcription and request context unchanged.
    async fn audio_transcription(
        &self,
        request: AudioTranscriptionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<AudioTranscriptionResponse> {
        self.backend.audio_transcription(request, context).await
    }

    /// Forward multipart translation and request context unchanged.
    async fn audio_translation(
        &self,
        request: AudioTranscriptionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<AudioTranscriptionResponse> {
        self.backend.audio_translation(request, context).await
    }
}
