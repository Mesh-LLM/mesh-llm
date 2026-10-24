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

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

impl MembershipBackend {
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
    async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
        self.inner.models().await
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
        self.execute(context, |context| {
            self.inner.chat_completion_with_context(request, context)
        })
        .await
    }

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

    async fn completion(&self, request: CompletionRequest) -> InferenceResult<CompletionResponse> {
        self.completion_with_context(request, InferenceRequestContext::new())
            .await
    }

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

    async fn embeddings(
        &self,
        request: EmbeddingsRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<EmbeddingResponse> {
        self.execute(context, |context| self.inner.embeddings(request, context))
            .await
    }

    async fn rerank(
        &self,
        request: RerankRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<RerankResponse> {
        self.execute(context, |context| self.inner.rerank(request, context))
            .await
    }

    async fn audio_speech(
        &self,
        request: AudioSpeechRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<AudioResponse> {
        self.execute(context, |context| self.inner.audio_speech(request, context))
            .await
    }

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
