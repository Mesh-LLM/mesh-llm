//! Installed policy over the real typed HTTP router and production hook composition.
use super::lifecycle_live_tests::LiveHost;
use async_trait::async_trait;
use openai_frontend::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStream,
    CompletionRequest, CompletionResponse, CompletionStream, HookedOpenAiBackend, ModelObject,
    OpenAiBackend, OpenAiHookPolicy, OpenAiRequestContext, OpenAiResult, Usage,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct TypedBackend {
    calls: Arc<AtomicUsize>,
    hooks: Arc<dyn OpenAiHookPolicy>,
}

#[async_trait]
impl OpenAiBackend for TypedBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("allowed-model")])
    }
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ChatCompletionResponse::new(
            request.model,
            "typed",
            Usage::new(2, 1),
        ))
    }
    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Box::pin(futures_util::stream::iter(vec![
            Ok(ChatCompletionChunk::delta(request.model.clone(), "typed")),
            Ok(ChatCompletionChunk::done(request.model)),
        ])))
    }
    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        let id = uuid::Uuid::new_v4().to_string();
        self.hooks.admit_effective_completion(&request, &id).await?;
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.hooks.on_completion_terminal(&id, "completed").await;
        Ok(CompletionResponse::new(
            request.model,
            "typed",
            Usage::new(2, 1),
        ))
    }
    async fn completion_stream(
        &self,
        request: CompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        // Mirror the native completion seam: admit the prepared request, then
        // publish backend failure before the frontend exposes the SSE error.
        let id = uuid::Uuid::new_v4().to_string();
        self.hooks.admit_effective_completion(&request, &id).await?;
        self.calls.fetch_add(1, Ordering::SeqCst);
        let hooks = self.hooks.clone();
        Ok(Box::pin(futures_util::stream::once(async move {
            hooks.on_completion_terminal(&id, "backend_error").await;
            Err(openai_frontend::OpenAiError::backend(
                "fixture completion failure",
            ))
        })))
    }
}

async fn terminal_events(host: &LiveHost, minimum: usize) -> Vec<Value> {
    // Wait for a concrete terminal count, since HTTP Body::finish schedules
    // its terminal service asynchronously. No timing-based settling delay.
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let events = host.events();
            if events
                .iter()
                .filter(|event| event["phase"] == "exchange_finished")
                .count()
                >= minimum
            {
                return events;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("installed terminal callbacks")
}

fn assert_commitment(events: &[Value], request_body: &Value, bytes: &[u8], outcome: &str) {
    let received = events
        .iter()
        .rev()
        .find(|event| event["phase"] == "request_received" && event["body"] == *request_body)
        .unwrap();
    let id = &received["exchange_id"];
    let terminal = events
        .iter()
        .find(|event| event["phase"] == "exchange_finished" && event["exchange_id"] == *id)
        .unwrap();
    assert_eq!(terminal["execution_outcome"], outcome);
    assert_eq!(
        terminal["response_wire_commitment"]["sha256"],
        hex::encode(Sha256::digest(bytes))
    );
    assert_eq!(
        terminal["response_wire_commitment"]["byte_count"],
        bytes.len()
    );
    assert_eq!(
        terminal["response_wire_commitment"]["side_stream_complete"],
        true
    );
}

#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_lifecycle_typed_router_denies_all_endpoints_and_commits_stream_outcomes() {
    let host = LiveHost::start(true, true).await;
    let calls = Arc::new(AtomicUsize::new(0));
    let hooks = crate::plugin::exchange_policy::compose_node_hooks(host.node.clone());
    let backend = Arc::new(HookedOpenAiBackend::new(
        Arc::new(TypedBackend {
            calls: calls.clone(),
            hooks: hooks.clone(),
        }),
        hooks,
    ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, openai_frontend::router(backend))
            .await
            .unwrap();
    });
    let client = reqwest::Client::new();
    let denied = [
        (
            "/v1/chat/completions",
            json!({"model":"blocked-model","messages":[{"role":"user","content":"deny"}]}),
        ),
        (
            "/v1/completions",
            json!({"model":"blocked-model","prompt":"deny"}),
        ),
        (
            "/v1/responses",
            json!({"model":"blocked-model","input":"deny"}),
        ),
    ];
    for (path, body) in denied {
        let response = client
            .post(format!("{base}{path}"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::FORBIDDEN);
        let _ = response.bytes().await.unwrap();
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "received denial prevents every typed backend call"
    );
    let cases = [
        (
            "/v1/chat/completions",
            json!({"model":"allowed-model","messages":[{"role":"user","content":"allow chat"}],"stream":true}),
            "completed",
        ),
        (
            "/v1/responses",
            json!({"model":"allowed-model","input":"allow responses","stream":true}),
            "completed",
        ),
        (
            "/v1/completions",
            json!({"model":"allowed-model","prompt":"allow completion","stream":true}),
            "backend_error",
        ),
    ];
    for (index, (path, body, outcome)) in cases.into_iter().enumerate() {
        let response = client
            .post(format!("{base}{path}"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let bytes = response.bytes().await.unwrap();
        assert!(!bytes.is_empty());
        let events = terminal_events(&host, 4 + index).await;
        assert_commitment(&events, &body, &bytes, outcome);
        let selected = events
            .iter()
            .rev()
            .find(|event| event["phase"] == "backend_selected")
            .unwrap();
        assert_eq!(
            selected["effective_request_encoding"],
            "typed_json_serialization"
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    server.abort();
    host.stop().await;
}
