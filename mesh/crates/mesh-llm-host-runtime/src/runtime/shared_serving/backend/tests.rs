use super::*;
use std::sync::Mutex;
use std::time::Duration;

#[derive(Default)]
struct HoldingBackend {
    workers: Mutex<Vec<InferenceRequestContext>>,
}

#[async_trait]
impl InferenceBackend for HoldingBackend {
    /// Expose one fake model for admission tests.
    async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("shared")])
    }

    /// Reject non-stream chat so this fixture cannot accidentally execute untracked work.
    async fn chat_completion(
        &self,
        _: ChatCompletionRequest,
    ) -> InferenceResult<ChatCompletionResponse> {
        Err(InferenceError::unsupported("stream-only test backend"))
    }

    /// Keep the worker context alive behind a permanently pending response stream.
    async fn chat_completion_stream(
        &self,
        _: ChatCompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<ChatCompletionStream> {
        self.workers.lock().unwrap().push(context);
        Ok(Box::pin(futures_util::stream::pending()))
    }

    /// Retain a worker context after the non-streaming caller's future is abandoned.
    async fn completion_with_context(
        &self,
        _: CompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<CompletionResponse> {
        self.workers.lock().unwrap().push(context);
        std::future::pending().await
    }

    /// Record an invalid delegation so the test detects any realtime admission bypass.
    async fn realtime_transcription(
        &self,
        _: realtime::RealtimeTranscriptionConfig,
        context: InferenceRequestContext,
    ) -> InferenceResult<realtime::RealtimeTranscriptionSession> {
        self.workers.lock().unwrap().push(context);
        Err(InferenceError::backend(
            "unscoped realtime reached the backend",
        ))
    }
}

/// Register a fresh membership wrapper around the same holding backend.
fn membership(inner: Arc<HoldingBackend>, admission: &Arc<Admission>) -> Arc<MembershipBackend> {
    let id = admission.register(2);
    Arc::new(MembershipBackend::new(inner, admission.clone(), id))
}

#[tokio::test]
/// Unsupported realtime must fail before reaching the underlying shared backend.
async fn unsupported_realtime_cannot_bypass_membership_admission() {
    let inner = Arc::new(HoldingBackend::default());
    let admission = Admission::new(1, 4).unwrap();
    let member = membership(inner.clone(), &admission);
    let result = member
        .realtime_transcription(
            realtime::RealtimeTranscriptionConfig::new("shared"),
            InferenceRequestContext::new(),
        )
        .await;
    let Err(error) = result else {
        panic!("this shared-serving milestone must not open an unscoped live session")
    };
    assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
    assert!(error.message().contains("/v1/realtime"));
    assert!(inner.workers.lock().unwrap().is_empty());
    assert_eq!(admission.totals(), (0, 0));
}

#[tokio::test]
/// Dropping one stream must isolate cancellation and preserve worker-owned capacity.
async fn disconnect_cancels_only_its_stream_and_does_not_release_a_running_worker() {
    let inner = Arc::new(HoldingBackend::default());
    let admission = Admission::new(2, 4).unwrap();
    let a = membership(inner.clone(), &admission);
    let b = membership(inner.clone(), &admission);
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "shared", "messages": [{"role":"user", "content":"hello"}], "stream": true,
    }))
    .unwrap();
    let context_a = InferenceRequestContext::new();
    let context_b = InferenceRequestContext::new();
    let stream_a = a
        .chat_completion_stream(request.clone(), context_a.clone())
        .await
        .unwrap();
    let stream_b = b
        .chat_completion_stream(request, context_b.clone())
        .await
        .unwrap();
    assert_eq!(admission.active(), 2);
    let workers = inner.workers.lock().unwrap();
    assert_ne!(workers[0].execution_scope(), workers[1].execution_scope());
    drop(workers);
    drop(stream_a);
    assert!(context_a.is_cancelled());
    assert!(!context_b.is_cancelled());
    assert_eq!(
        admission.active(),
        2,
        "native work outlives the HTTP consumer"
    );
    inner.workers.lock().unwrap().remove(0);
    assert_eq!(admission.active(), 1);
    drop(stream_b);
    assert!(context_b.is_cancelled());
    inner.workers.lock().unwrap().clear();
    assert_eq!(admission.active(), 0);
}

#[tokio::test]
/// Aborting a non-stream request must cancel its worker without releasing its held lease.
async fn dropping_a_nonstream_future_cancels_but_waits_for_worker_cleanup() {
    let inner = Arc::new(HoldingBackend::default());
    let admission = Admission::new(1, 4).unwrap();
    let a = membership(inner.clone(), &admission);
    let context = InferenceRequestContext::new();
    let observed = context.clone();
    let task = tokio::spawn(async move {
        a.completion_with_context(
            serde_json::from_value(serde_json::json!({
                "model": "shared", "prompt": "hello",
            }))
            .unwrap(),
            context,
        )
        .await
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while inner.workers.lock().unwrap().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert!(observed.is_cancelled());
    assert_eq!(admission.active(), 1);
    inner.workers.lock().unwrap().clear();
    assert_eq!(admission.active(), 0);
}
