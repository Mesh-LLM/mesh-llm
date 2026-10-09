use std::{
    pin::Pin,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use async_trait::async_trait;
use futures_core::Stream;
use tokio::sync::Notify;

use crate::{
    audio::{
        AudioResponse, AudioSpeechRequest, AudioTranscriptionRequest, AudioTranscriptionResponse,
    },
    chat::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
    completions::{CompletionChunk, CompletionRequest, CompletionResponse},
    embeddings::{EmbeddingResponse, EmbeddingsRequest},
    errors::InferenceError,
    lifecycle::RequestId,
    models::ModelObject,
    rerank::{RerankRequest, RerankResponse},
    system_one::{SystemOneRequest, SystemOneResponse},
};

pub type ChatCompletionStream =
    Pin<Box<dyn Stream<Item = InferenceResult<ChatCompletionChunk>> + Send + 'static>>;
pub type CompletionStream =
    Pin<Box<dyn Stream<Item = InferenceResult<CompletionChunk>> + Send + 'static>>;

pub type InferenceResult<T> = Result<T, InferenceError>;

#[derive(Debug, Default)]
struct CancellationState {
    cancelled: AtomicBool,
    notify: Notify,
}

#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    state: Arc<CancellationState>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        if !self.state.cancelled.swap(true, Ordering::AcqRel) {
            self.state.notify.notify_waiters();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Acquire)
    }

    pub async fn cancelled(&self) {
        loop {
            let notified = self.state.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceRequestContext {
    cancellation: CancellationToken,
    request_id: Option<RequestId>,
    exchange_id: Arc<OnceLock<String>>,
    stream_usage_observation: bool,
    trusted_agent_session: bool,
}

impl InferenceRequestContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a backend context correlated to a frontend request identifier.
    pub fn with_request_id(request_id: RequestId) -> Self {
        Self {
            request_id: Some(request_id),
            ..Self::default()
        }
    }

    /// Enable internal usage observation for an HTTP streaming adapter.
    ///
    /// This is deliberately independent of the client's `include_usage` wire
    /// option: a backend may provide usage to the frontend for lifecycle
    /// accounting while the frontend still suppresses it on the client wire.
    pub fn with_stream_usage_observation(mut self) -> Self {
        self.stream_usage_observation = true;
        self
    }

    /// Return whether an HTTP adapter requested internal stream usage.
    pub fn observes_stream_usage(&self) -> bool {
        self.stream_usage_observation
    }

    pub(crate) fn with_trusted_agent_session(mut self) -> Self {
        self.trusted_agent_session = true;
        self
    }

    /// Return whether the endpoint's configured trusted header supplied the
    /// agent-session identity. Request body metadata cannot set this marker.
    pub fn has_trusted_agent_session(&self) -> bool {
        self.trusted_agent_session
    }

    /// Return the frontend request identifier when the caller supplied one.
    pub fn request_id(&self) -> Option<RequestId> {
        self.request_id
    }

    /// Publish the join key minted by the backend for a hook-tracked exchange.
    pub fn publish_exchange_id(&self, exchange_id: String) {
        let _ = self.exchange_id.set(exchange_id);
    }

    /// Return the join key published by a hook-tracked backend exchange.
    pub fn exchange_id(&self) -> Option<String> {
        self.exchange_id.get().cloned()
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
}

impl Default for InferenceRequestContext {
    fn default() -> Self {
        Self {
            cancellation: CancellationToken::new(),
            request_id: None,
            exchange_id: Arc::new(OnceLock::new()),
            stream_usage_observation: false,
            trusted_agent_session: false,
        }
    }
}

#[async_trait]
pub trait InferenceBackend: Send + Sync + 'static {
    /// HTTP lifecycle contract inherited by routers and transparent wrappers.
    fn http_exchange_policy(&self) -> Option<Arc<dyn crate::http_exchange::HttpExchangePolicy>> {
        None
    }
    async fn models(&self) -> InferenceResult<Vec<ModelObject>>;

    /// Count the model-rendered chat prompt without generating or running hooks.
    async fn count_chat_tokens(&self, _request: ChatCompletionRequest) -> InferenceResult<u32> {
        Err(InferenceError::unsupported(
            "token counting is unavailable for this backend",
        ))
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> InferenceResult<ChatCompletionResponse>;

    async fn chat_completion_with_context(
        &self,
        request: ChatCompletionRequest,
        _context: InferenceRequestContext,
    ) -> InferenceResult<ChatCompletionResponse> {
        self.chat_completion(request).await
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<ChatCompletionStream>;

    async fn completion(&self, _request: CompletionRequest) -> InferenceResult<CompletionResponse> {
        Err(InferenceError::unsupported(
            "/v1/completions is not supported by this backend",
        ))
    }

    async fn completion_with_context(
        &self,
        request: CompletionRequest,
        _context: InferenceRequestContext,
    ) -> InferenceResult<CompletionResponse> {
        self.completion(request).await
    }

    async fn completion_stream(
        &self,
        _request: CompletionRequest,
        _context: InferenceRequestContext,
    ) -> InferenceResult<CompletionStream> {
        Err(InferenceError::unsupported(
            "/v1/completions streaming is not supported by this backend",
        ))
    }

    async fn system_one(&self, _request: SystemOneRequest) -> InferenceResult<SystemOneResponse> {
        Err(InferenceError::unsupported(
            "/systemone is not supported by this backend",
        ))
    }

    /// Embed text or token batches. The default rejects unsupported backends.
    async fn embeddings(
        &self,
        _request: EmbeddingsRequest,
        _context: InferenceRequestContext,
    ) -> InferenceResult<EmbeddingResponse> {
        Err(InferenceError::unsupported(
            "/v1/embeddings is not supported by this backend",
        ))
    }

    /// Rank documents against a query, preserving their original indexes.
    async fn rerank(
        &self,
        _request: RerankRequest,
        _context: InferenceRequestContext,
    ) -> InferenceResult<RerankResponse> {
        Err(InferenceError::unsupported(
            "/v1/rerank is not supported by this backend",
        ))
    }

    /// Synthesize binary audio; unsupported codecs and voices must return errors.
    async fn audio_speech(
        &self,
        _request: AudioSpeechRequest,
        _context: InferenceRequestContext,
    ) -> InferenceResult<AudioResponse> {
        Err(InferenceError::unsupported(
            "/v1/audio/speech is not supported by this backend",
        ))
    }

    /// Transcribe uploaded audio in its source language using the request context.
    async fn audio_transcription(
        &self,
        _request: AudioTranscriptionRequest,
        _context: InferenceRequestContext,
    ) -> InferenceResult<AudioTranscriptionResponse> {
        Err(InferenceError::unsupported(
            "/v1/audio/transcriptions is not supported by this backend",
        ))
    }

    /// Translate uploaded audio into English; unsupported backends fail explicitly.
    async fn audio_translation(
        &self,
        _request: AudioTranscriptionRequest,
        _context: InferenceRequestContext,
    ) -> InferenceResult<AudioTranscriptionResponse> {
        Err(InferenceError::unsupported(
            "/v1/audio/translations is not supported by this backend",
        ))
    }
}

pub(crate) type SharedBackend = Arc<dyn InferenceBackend>;
