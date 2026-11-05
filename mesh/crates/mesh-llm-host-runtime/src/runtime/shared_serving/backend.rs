//! Membership-bound access to the host's one inference backend.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures_util::Stream;
use skippy_inference_api::*;

use super::admission::{Admission, MemberId};

#[cfg(test)]
mod tests;

pub(super) struct MembershipBackend {
    inner: Arc<dyn InferenceBackend>,
    admission: Arc<Admission>,
    member: MemberId,
    scope: InferenceExecutionScope,
}

struct CancelOnDrop(InferenceRequestContext);

impl Drop for CancelOnDrop {
    /// Signal request cancellation when its HTTP future or stream is dropped.
    /// Worker-held context clones continue to retain the execution lease.
    fn drop(&mut self) {
        self.0.cancel();
    }
}

struct MembershipStream<T> {
    inner: Pin<Box<dyn Stream<Item = InferenceResult<T>> + Send>>,
    // Also retains the execution lease while a slow consumer holds the stream.
    _execution: CancelOnDrop,
}

impl<T> Stream for MembershipStream<T> {
    type Item = InferenceResult<T>;

    /// Poll the backend stream while retaining this consumer's execution lease.
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

impl MembershipBackend {
    /// Bind a shared backend to one admission identity and a fresh cache/session scope.
    pub(super) fn new(
        inner: Arc<dyn InferenceBackend>,
        admission: Arc<Admission>,
        member: MemberId,
    ) -> Self {
        Self {
            inner,
            admission,
            member,
            scope: InferenceExecutionScope::new(),
        }
    }

    /// Acquire shared capacity and install this membership's trusted execution scope.
    async fn admitted(
        &self,
        context: InferenceRequestContext,
    ) -> InferenceResult<InferenceRequestContext> {
        let lease = self
            .admission
            .acquire(self.member, context.cancellation_token())
            .await?;
        Ok(context.with_execution_scope(self.scope, lease))
    }

    /// Run one admitted request, signalling cancellation if its future is abandoned.
    /// The backend must retain the supplied context until its actual worker finishes.
    async fn execute<T, F, Fut>(
        &self,
        context: InferenceRequestContext,
        work: F,
    ) -> InferenceResult<T>
    where
        F: FnOnce(InferenceRequestContext) -> Fut,
        Fut: Future<Output = InferenceResult<T>>,
    {
        let context = self.admitted(context).await?;
        let _cancel = CancelOnDrop(context.clone());
        work(context).await
    }
}

#[async_trait]
impl InferenceBackend for MembershipBackend {
    // Do not inherit hooks bound to a different Node. The shared backend is
    // loaded without node/plugin hooks; memberships do not bridge callbacks.
    /// Expose the shared model catalog without consuming an execution slot.
    async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
        self.inner.models().await
    }

    /// Create a cancellable context and use the membership-scoped chat path.
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> InferenceResult<ChatCompletionResponse> {
        self.chat_completion_with_context(request, InferenceRequestContext::new())
            .await
    }

    /// Execute chat under this membership's admission budget and isolation scope.
    async fn chat_completion_with_context(
        &self,
        request: ChatCompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<ChatCompletionResponse> {
        self.execute(context, |context| {
            self.inner.chat_completion_with_context(request, context)
        })
        .await
    }

    /// Retain admission through stream consumption and cancel when the stream drops.
    /// Any worker-held context continues to own capacity after consumer disconnect.
    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<ChatCompletionStream> {
        let context = self.admitted(context).await?;
        let cancel = CancelOnDrop(context.clone());
        let inner = self.inner.chat_completion_stream(request, context).await?;
        Ok(Box::pin(MembershipStream {
            inner,
            _execution: cancel,
        }))
    }

    /// Create a cancellable context and use the membership-scoped completion path.
    async fn completion(&self, request: CompletionRequest) -> InferenceResult<CompletionResponse> {
        self.completion_with_context(request, InferenceRequestContext::new())
            .await
    }

    /// Execute a text completion under the shared budget and membership scope.
    async fn completion_with_context(
        &self,
        request: CompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<CompletionResponse> {
        self.execute(context, |context| {
            self.inner.completion_with_context(request, context)
        })
        .await
    }

    /// Wrap a completion stream with cancellation and execution-lease ownership.
    async fn completion_stream(
        &self,
        request: CompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<CompletionStream> {
        let context = self.admitted(context).await?;
        let cancel = CancelOnDrop(context.clone());
        let inner = self.inner.completion_stream(request, context).await?;
        Ok(Box::pin(MembershipStream {
            inner,
            _execution: cancel,
        }))
    }

    /// Compute embeddings through the same bounded, scoped admission as generation.
    async fn embeddings(
        &self,
        request: EmbeddingsRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<EmbeddingResponse> {
        self.execute(context, |context| self.inner.embeddings(request, context))
            .await
    }

    /// Run reranking under this membership's shared execution lease.
    async fn rerank(
        &self,
        request: RerankRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<RerankResponse> {
        self.execute(context, |context| self.inner.rerank(request, context))
            .await
    }

    /// Synthesize audio with membership-scoped admission and cancellation.
    async fn audio_speech(
        &self,
        request: AudioSpeechRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<AudioResponse> {
        self.execute(context, |context| self.inner.audio_speech(request, context))
            .await
    }

    /// Transcribe audio through the shared budget using this membership's context.
    async fn audio_transcription(
        &self,
        request: AudioTranscriptionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<AudioTranscriptionResponse> {
        self.execute(context, |context| {
            self.inner.audio_transcription(request, context)
        })
        .await
    }

    /// Translate audio through the shared budget using this membership's context.
    async fn audio_translation(
        &self,
        request: AudioTranscriptionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<AudioTranscriptionResponse> {
        self.execute(context, |context| {
            self.inner.audio_translation(request, context)
        })
        .await
    }
}
