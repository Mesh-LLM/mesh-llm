use std::{
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use async_trait::async_trait;
use futures_core::Stream;
use tokio::sync::Notify;

use crate::{
    chat::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
    completions::{CompletionChunk, CompletionRequest, CompletionResponse},
    errors::OpenAiError,
    lifecycle::RequestId,
    models::ModelObject,
};

pub type ChatCompletionStream =
    Pin<Box<dyn Stream<Item = OpenAiResult<ChatCompletionChunk>> + Send + 'static>>;
pub type CompletionStream =
    Pin<Box<dyn Stream<Item = OpenAiResult<CompletionChunk>> + Send + 'static>>;

pub type OpenAiResult<T> = Result<T, OpenAiError>;

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

#[derive(Debug, Clone, Default)]
pub struct OpenAiRequestContext {
    cancellation: CancellationToken,
    request_id: Option<RequestId>,
}

impl OpenAiRequestContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a backend context correlated to a frontend request identifier.
    pub fn with_request_id(request_id: RequestId) -> Self {
        Self {
            cancellation: CancellationToken::new(),
            request_id: Some(request_id),
        }
    }

    /// Return the frontend request identifier when the caller supplied one.
    pub fn request_id(&self) -> Option<RequestId> {
        self.request_id
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

#[async_trait]
pub trait OpenAiBackend: Send + Sync + 'static {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>>;

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse>;

    async fn chat_completion_with_context(
        &self,
        request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionResponse> {
        self.chat_completion(request).await
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream>;

    async fn completion(&self, _request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        Err(OpenAiError::unsupported(
            "/v1/completions is not supported by this backend",
        ))
    }

    async fn completion_with_context(
        &self,
        request: CompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionResponse> {
        self.completion(request).await
    }

    async fn completion_stream(
        &self,
        _request: CompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        Err(OpenAiError::unsupported(
            "/v1/completions streaming is not supported by this backend",
        ))
    }
}

pub(crate) type SharedBackend = Arc<dyn OpenAiBackend>;
