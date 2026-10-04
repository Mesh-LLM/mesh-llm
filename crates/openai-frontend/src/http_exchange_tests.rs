use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use async_trait::async_trait;
use axum::{
    body::{Body, to_bytes},
    http::{HeaderMap, Request, StatusCode},
};
use tower::ServiceExt;

use crate::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStream,
    CompletionChunk, CompletionRequest, CompletionResponse, CompletionStream, ModelObject,
    OpenAiBackend, OpenAiError, OpenAiRequestContext, OpenAiResult, RequestId, Usage,
    http_exchange::{HttpExchangeAdmission, HttpExchangePolicy},
    wire_bytes::{WireBytesCommitment, WireBytesIncomplete, WireBytesObserver, commit_wire_bytes},
};

#[derive(Default)]
struct Recorder {
    original: Mutex<Vec<Vec<u8>>>,
    response: Mutex<Vec<u8>>,
    terminals: Mutex<Vec<WireBytesCommitment>>,
    deny: bool,
}

#[async_trait]
impl HttpExchangePolicy for Arc<Recorder> {
    async fn received(
        &self,
        _method: &str,
        _path: &str,
        _headers: &HeaderMap,
        body: &[u8],
        _id: RequestId,
    ) -> HttpExchangeAdmission {
        self.original.lock().unwrap().push(body.to_vec());
        HttpExchangeAdmission {
            observation_id: None,
            observer: Some(self.clone()),
            response_headers: Vec::new(),
            denial: self.deny.then(|| {
                OpenAiError::from_kind(
                    StatusCode::FORBIDDEN,
                    crate::OpenAiErrorKind::Permission,
                    "denied by observer",
                )
            }),
        }
    }
}

impl WireBytesObserver for Recorder {
    fn try_chunk(&self, offset: u64, bytes: &[u8]) -> bool {
        let mut response = self.response.lock().unwrap();
        assert_eq!(offset, response.len() as u64);
        response.extend_from_slice(bytes);
        true
    }
    fn finish(&self, commitment: WireBytesCommitment) {
        self.terminals.lock().unwrap().push(commitment);
    }
}

struct Backend {
    recorder: Arc<Recorder>,
    calls: AtomicUsize,
}

impl Backend {
    fn count_call(&self, model: &str) -> OpenAiResult<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match model {
            "error" => Err(OpenAiError::backend("failed backend")),
            "timeout" => Err(OpenAiError::timeout("backend deadline")),
            _ => Ok(()),
        }
    }
}

#[async_trait]
impl OpenAiBackend for Backend {
    fn http_exchange_policy(&self) -> Option<Arc<dyn HttpExchangePolicy>> {
        Some(Arc::new(self.recorder.clone()))
    }
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(Vec::new())
    }
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        self.count_call(&request.model)?;
        let mut response = ChatCompletionResponse::new(request.model, "hello", Usage::new(2, 1));
        response.id = "fixed".into();
        response.created = 1;
        Ok(response)
    }
    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        self.count_call(&request.model)?;
        if request.model == "pending" {
            return Ok(Box::pin(futures_util::stream::pending()));
        }
        let mut chunk = ChatCompletionChunk::done(request.model);
        chunk.id = "fixed".into();
        chunk.created = 1;
        Ok(Box::pin(futures_util::stream::iter(vec![Ok(chunk)])))
    }
    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        self.count_call(&request.model)?;
        let mut response = CompletionResponse::new(request.model, "hello", Usage::new(2, 1));
        response.id = "fixed".into();
        response.created = 1;
        Ok(response)
    }
    async fn completion_stream(
        &self,
        request: CompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        self.count_call(&request.model)?;
        let mut chunk = CompletionChunk::done(request.model);
        chunk.id = "fixed".into();
        chunk.created = 1;
        Ok(Box::pin(futures_util::stream::iter(vec![Ok(chunk)])))
    }
}

fn backend(deny: bool) -> Arc<Backend> {
    Arc::new(Backend {
        recorder: Arc::new(Recorder {
            deny,
            ..Recorder::default()
        }),
        calls: AtomicUsize::new(0),
    })
}

fn request(path: &str, body: &[u8]) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_vec()))
        .unwrap()
}

fn request_body(path: &str, stream: bool, model: &str) -> Vec<u8> {
    match path {
        "/v1/completions" => format!("{{ \"model\":\"{model}\", \"prompt\":\"hi\", \"stream\":{stream} }}\n"),
        "/v1/responses" => format!("{{ \"model\":\"{model}\", \"input\":\"hi\", \"stream\":{stream} }}\n"),
        _ => format!("{{ \"model\":\"{model}\", \"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}], \"stream\":{stream} }}\n"),
    }.into_bytes()
}

#[tokio::test]
async fn all_three_endpoints_observe_original_and_full_final_bytes() {
    for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
        for stream in [false, true] {
            let backend = backend(false);
            let body = request_body(path, stream, "tiny");
            let response = crate::router(backend.clone())
                .oneshot(request(path, &body))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
            assert_eq!(*backend.recorder.original.lock().unwrap(), vec![body]);
            assert_eq!(*backend.recorder.response.lock().unwrap(), bytes);
            assert_eq!(
                *backend.recorder.terminals.lock().unwrap(),
                vec![commit_wire_bytes(&bytes)]
            );
            assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
            if stream {
                assert!(std::str::from_utf8(&bytes).unwrap().contains("data: "));
            }
        }
    }
}

#[tokio::test]
async fn admission_denial_never_reaches_backend_and_observes_emitted_error() {
    for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
        let backend = backend(true);
        let body = request_body(path, false, "tiny");
        let response = crate::router(backend.clone())
            .oneshot(request(path, &body))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        let bytes = to_bytes(response.into_body(), 1024).await.unwrap();
        assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            *backend.recorder.terminals.lock().unwrap(),
            vec![commit_wire_bytes(&bytes)]
        );
    }
}

#[tokio::test]
async fn invalid_request_backend_failure_and_timeout_still_hash_final_error() {
    for (body, status, calls) in [
        (b"{invalid".to_vec(), StatusCode::BAD_REQUEST, 0),
        (
            request_body("/v1/chat/completions", false, "error"),
            StatusCode::BAD_GATEWAY,
            1,
        ),
        (
            request_body("/v1/chat/completions", false, "timeout"),
            StatusCode::GATEWAY_TIMEOUT,
            1,
        ),
    ] {
        let backend = backend(false);
        let response = crate::router(backend.clone())
            .oneshot(request("/v1/chat/completions", &body))
            .await
            .unwrap();
        assert_eq!(response.status(), status);
        let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
        assert_eq!(*backend.recorder.original.lock().unwrap(), vec![body]);
        assert_eq!(
            *backend.recorder.terminals.lock().unwrap(),
            vec![commit_wire_bytes(&bytes)]
        );
        assert_eq!(backend.calls.load(Ordering::SeqCst), calls);
    }
}

#[tokio::test]
async fn dropped_pending_stream_has_one_incomplete_terminal() {
    let backend = backend(false);
    let body = request_body("/v1/chat/completions", true, "pending");
    let response = crate::router(backend.clone())
        .oneshot(request("/v1/chat/completions", &body))
        .await
        .unwrap();
    drop(response);
    let terminals = backend.recorder.terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert_eq!(
        terminals[0].incomplete,
        Some(WireBytesIncomplete::Cancelled)
    );
}
