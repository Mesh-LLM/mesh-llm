use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use serde_json::json;

use super::*;
use crate::Usage;

struct RecordingBackend {
    seen: Mutex<Option<ChatCompletionRequest>>,
}

#[async_trait]
impl OpenAiBackend for RecordingBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("auto")])
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        *self.seen.lock().unwrap() = Some(request.clone());
        Ok(ChatCompletionResponse::new(
            request.model,
            "ok",
            Usage::new(0, 0),
        ))
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        *self.seen.lock().unwrap() = Some(request);
        Ok(Box::pin(futures_util::stream::empty()))
    }
}

struct InjectingHook;

#[async_trait]
impl OpenAiHookPolicy for InjectingHook {
    async fn before_chat_completion(
        &self,
        _request: &mut ChatCompletionRequest,
    ) -> OpenAiResult<ChatHookOutcome> {
        Ok(ChatHookOutcome::injected("[hint]\n"))
    }
}

struct FailingBackend;

#[async_trait]
impl OpenAiBackend for FailingBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(Vec::new())
    }

    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        Err(crate::errors::OpenAiError::backend("upstream exploded"))
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        Err(crate::errors::OpenAiError::backend("upstream exploded"))
    }
}

#[derive(Debug, Clone, PartialEq)]
enum TerminalRecord {
    Success { model: String },
    Error { status: u16, message: String },
    Denied { status: u16, reason: String },
    Cancelled,
    StreamCompleted,
}

#[derive(Default)]
struct RecordingPolicy {
    deny: bool,
    effective: Mutex<Vec<(ChatCompletionRequest, ChatExchangeRoute)>>,
    terminals: Mutex<Vec<TerminalRecord>>,
}

#[async_trait]
impl OpenAiHookPolicy for RecordingPolicy {
    async fn before_chat_completion(
        &self,
        _request: &mut ChatCompletionRequest,
    ) -> OpenAiResult<ChatHookOutcome> {
        if self.deny {
            return Err(crate::errors::OpenAiError::invalid_request(
                "denied by policy",
            ));
        }
        Ok(ChatHookOutcome::injected("[hint]\n"))
    }

    async fn on_effective_chat_completion(
        &self,
        request: &ChatCompletionRequest,
        route: &ChatExchangeRoute,
    ) {
        self.effective
            .lock()
            .unwrap()
            .push((request.clone(), route.clone()));
    }

    async fn on_chat_completion_terminal(
        &self,
        _request: &ChatCompletionRequest,
        _exchange_id: &str,
        outcome: &ChatCompletionOutcome<'_>,
    ) {
        let record = match outcome {
            ChatCompletionOutcome::Success { response } => TerminalRecord::Success {
                model: response.model.clone(),
            },
            ChatCompletionOutcome::Error { status, message } => TerminalRecord::Error {
                status: *status,
                message: (*message).to_string(),
            },
            ChatCompletionOutcome::Denied { status, reason } => TerminalRecord::Denied {
                status: *status,
                reason: (*reason).to_string(),
            },
            ChatCompletionOutcome::Cancelled => TerminalRecord::Cancelled,
            ChatCompletionOutcome::StreamCompleted => TerminalRecord::StreamCompleted,
        };
        self.terminals.lock().unwrap().push(record);
    }
}

struct MediaRescueHook;

#[async_trait]
impl OpenAiHookPolicy for MediaRescueHook {
    async fn before_chat_completion(
        &self,
        request: &mut ChatCompletionRequest,
    ) -> OpenAiResult<ChatHookOutcome> {
        let media = first_chat_media(&request.messages).expect("media");
        Ok(ChatHookOutcome::injected_with_consumed_media(
            "[Audio context: hello]\n\n",
            media,
        ))
    }
}

#[test]
fn chat_mesh_hooks_enabled_reads_extra_flag() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "hello"}],
        "mesh_hooks": true
    }))
    .unwrap();

    assert!(chat_mesh_hooks_enabled(&request));

    set_chat_mesh_hooks_enabled(&mut request, false);

    assert!(!chat_mesh_hooks_enabled(&request));
}

#[test]
fn first_chat_media_extracts_image_url_and_user_text() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
            ]
        }]
    }))
    .unwrap();

    let media = first_chat_media(&request.messages).expect("media");

    assert_eq!(media.kind, ChatMediaKind::Image);
    assert_eq!(media.url, "data:image/png;base64,abc");
    assert_eq!(media.user_text, "what is this?");
    assert_eq!(media.message_index, 0);
    assert_eq!(media.part_index, 1);
}

#[test]
fn first_chat_media_extracts_audio_url_and_user_text() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "please transcribe this"},
                {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,abc"}}
            ]
        }]
    }))
    .unwrap();

    let media = first_chat_media(&request.messages).expect("media");

    assert_eq!(media.kind, ChatMediaKind::Audio);
    assert_eq!(media.url, "data:audio/wav;base64,abc");
    assert_eq!(media.user_text, "please transcribe this");
    assert_eq!(media.message_index, 0);
    assert_eq!(media.part_index, 1);
}

#[test]
fn first_chat_media_extracts_inline_input_audio_data() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what does this say?"},
                {"type": "input_audio", "input_audio": {
                    "data": "YWJj",
                    "format": "wav"
                }}
            ]
        }]
    }))
    .unwrap();

    let media = first_chat_media(&request.messages).expect("media");

    assert_eq!(media.kind, ChatMediaKind::Audio);
    assert_eq!(media.url, "data:audio/wav;base64,YWJj");
    assert_eq!(media.user_text, "what does this say?");
    assert_eq!(media.message_index, 0);
    assert_eq!(media.part_index, 1);
}

#[test]
fn image_only_message_with_mesh_hooks_is_valid_before_hook_injection() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
            ]
        }],
        "mesh_hooks": true
    }))
    .unwrap();

    request.validate().unwrap();
}

#[test]
fn image_only_message_without_mesh_hooks_is_valid_for_native_multimodal_backend() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
            ]
        }]
    }))
    .unwrap();

    request.validate().unwrap();
}

#[test]
fn inject_text_into_chat_messages_prepends_last_user_text() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "original"}]
    }))
    .unwrap();

    inject_text_into_chat_messages(&mut request.messages, "[hint]\n");

    assert_eq!(
        request.messages[0].content,
        Some(MessageContent::Text("[hint]\noriginal".to_string()))
    );
}

#[tokio::test]
async fn hooked_backend_applies_injection_once_before_forwarding() {
    let backend = Arc::new(RecordingBackend {
        seen: Mutex::new(None),
    });
    let hooked = HookedOpenAiBackend::new(backend.clone(), Arc::new(InjectingHook));
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "original"}],
        "mesh_hooks": true
    }))
    .unwrap();

    hooked.chat_completion(request).await.unwrap();

    let seen = backend.seen.lock().unwrap().clone().unwrap();
    assert_eq!(
        seen.messages[0].content,
        Some(MessageContent::Text("[hint]\noriginal".to_string()))
    );
}

#[tokio::test]
async fn hooked_backend_consumes_rescued_audio_media_before_forwarding() {
    let backend = Arc::new(RecordingBackend {
        seen: Mutex::new(None),
    });
    let hooked = HookedOpenAiBackend::new(backend.clone(), Arc::new(MediaRescueHook));
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "please transcribe this"},
                {"type": "input_audio", "input_audio": {
                    "data": "YWJj",
                    "format": "wav"
                }}
            ]
        }],
        "mesh_hooks": true
    }))
    .unwrap();

    hooked.chat_completion(request).await.unwrap();

    let seen = backend.seen.lock().unwrap().clone().unwrap();
    assert_eq!(first_chat_media(&seen.messages), None);
    assert_eq!(
        seen.messages[0].content,
        Some(MessageContent::Parts(vec![
            MessageContentPart {
                content_type: "text".to_string(),
                text: Some("[Audio context: hello]\n\n".to_string()),
                extra: Default::default(),
            },
            MessageContentPart {
                content_type: "text".to_string(),
                text: Some("please transcribe this".to_string()),
                extra: Default::default(),
            },
        ]))
    );
}

#[test]
fn consumed_media_action_removes_only_matching_media_part() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what is here?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                {"type": "input_audio", "input_audio": {"url": "data:audio/wav;base64,def"}}
            ]
        }],
        "mesh_hooks": true
    }))
    .unwrap();
    let media = ChatMediaRef {
        kind: ChatMediaKind::Audio,
        url: "data:audio/wav;base64,def".to_string(),
        user_text: "what is here?".to_string(),
        message_index: 0,
        part_index: 2,
    };

    apply_chat_hook_outcome(
        &mut request,
        &ChatHookOutcome::injected_with_consumed_media("[Audio context: beep]\n\n", media),
    );

    let Some(MessageContent::Parts(parts)) = &request.messages[0].content else {
        panic!("expected multipart content");
    };
    assert_eq!(
        parts
            .iter()
            .filter(|part| part.content_type == "input_audio")
            .count(),
        0
    );
    assert_eq!(
        parts
            .iter()
            .filter(|part| part.content_type == "image_url")
            .count(),
        1
    );
}

fn request_for(model: &str) -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": model,
        "messages": [{"role": "user", "content": "original"}]
    }))
    .unwrap()
}

#[tokio::test]
async fn effective_request_is_observed_after_mutation_and_terminal_reports_success() {
    let backend = Arc::new(RecordingBackend {
        seen: Mutex::new(None),
    });
    let policy = Arc::new(RecordingPolicy::default());
    let hooked = HookedOpenAiBackend::new(backend.clone(), policy.clone());

    let response = hooked
        .chat_completion(request_for("gpt-mesh"))
        .await
        .expect("backend call succeeds");
    assert_eq!(response.model, "gpt-mesh");

    // The backend actually ran: the extension must not short-circuit dispatch.
    let seen = backend.seen.lock().unwrap().clone().expect("dispatched");
    assert_eq!(
        seen.messages[0].content,
        Some(MessageContent::Text("[hint]\noriginal".to_string()))
    );

    let effective = policy.effective.lock().unwrap();
    assert_eq!(effective.len(), 1);
    let (effective_request, route) = &effective[0];
    assert_eq!(route.model, "gpt-mesh");
    assert_eq!(
        effective_request.messages[0].content,
        Some(MessageContent::Text("[hint]\noriginal".to_string())),
        "the effective request must reflect before_chat_completion's mutation"
    );

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(
        terminals.as_slice(),
        [TerminalRecord::Success {
            model: "gpt-mesh".to_string()
        }]
    );
}

#[tokio::test]
async fn backend_failure_reports_terminal_error_after_observing_effective_request() {
    let backend = Arc::new(FailingBackend);
    let policy = Arc::new(RecordingPolicy::default());
    let hooked = HookedOpenAiBackend::new(backend, policy.clone());

    let error = hooked
        .chat_completion(request_for("gpt-mesh"))
        .await
        .expect_err("backend fails");
    assert_eq!(error.status().as_u16(), 502);

    assert_eq!(policy.effective.lock().unwrap().len(), 1);
    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert!(matches!(
        &terminals[0],
        TerminalRecord::Error { status: 502, message }
            if message.contains("upstream exploded")
    ));
}

struct HangingBackend;

#[async_trait]
impl OpenAiBackend for HangingBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(Vec::new())
    }

    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        std::future::pending::<()>().await;
        unreachable!("this backend never returns")
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        Ok(Box::pin(futures_util::stream::empty()))
    }
}

/// Reproduces the bug this guards against: an outer timeout or client
/// disconnect drops the future driving `backend.await` before it can
/// return, so without `TerminalGuard` the exchange would never get a
/// terminal event at all.
#[tokio::test]
async fn dropping_the_backend_future_still_fires_exactly_one_terminal_event() {
    let backend = Arc::new(HangingBackend);
    let policy = Arc::new(RecordingPolicy::default());
    let hooked = Arc::new(HookedOpenAiBackend::new(backend, policy.clone()));

    let hooked_for_task = hooked.clone();
    let handle = tokio::spawn(async move {
        hooked_for_task
            .chat_completion(request_for("gpt-mesh"))
            .await
    });

    // Let the task run until it's parked on `backend.await`, then cancel
    // it the way an outer timeout or client disconnect would.
    tokio::task::yield_now().await;
    handle.abort();
    let _ = handle.await;

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(1);
    loop {
        if !policy.terminals.lock().unwrap().is_empty() {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "terminal event never fired after the backend future was dropped"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.as_slice(), [TerminalRecord::Cancelled]);
}

/// A policy whose terminal hook hangs forever on a `Success` outcome
/// (after signalling `started`), but records `Cancelled` immediately.
/// This lets a test park the exchange future *inside* `TerminalGuard::fire`'s
/// await — the window Finding A's fix targets — rather than only before
/// `fire` is ever called.
#[derive(Default)]
struct HangOnTerminalPolicy {
    started: tokio::sync::Notify,
    terminals: Mutex<Vec<TerminalRecord>>,
}

#[async_trait]
impl OpenAiHookPolicy for HangOnTerminalPolicy {
    async fn on_chat_completion_terminal(
        &self,
        _request: &ChatCompletionRequest,
        _exchange_id: &str,
        outcome: &ChatCompletionOutcome<'_>,
    ) {
        match outcome {
            ChatCompletionOutcome::Success { .. } => {
                self.started.notify_one();
                std::future::pending::<()>().await;
            }
            ChatCompletionOutcome::Cancelled => {
                self.terminals
                    .lock()
                    .unwrap()
                    .push(TerminalRecord::Cancelled);
            }
            _ => {}
        }
    }
}

/// Reproduces Finding A: `TerminalGuard::fire` used to set `fired = true`
/// *before* awaiting the terminal hook. If the exchange future is
/// dropped while that await is still pending (client disconnects right
/// as the backend returns), the in-flight `fire` call is interrupted
/// *and* `fired` was already `true`, so `Drop` no-ops too — the exchange
/// gets zero terminal events. With `fired` set only after the await
/// completes, `Drop` still sees `fired == false` in this window and
/// fires the `Cancelled` fallback, so the exchange still gets exactly
/// one.
#[tokio::test]
async fn cancelling_during_the_terminal_hook_await_still_fires_exactly_one_terminal_event() {
    let backend = Arc::new(RecordingBackend {
        seen: Mutex::new(None),
    });
    let policy = Arc::new(HangOnTerminalPolicy::default());
    let hooked = Arc::new(HookedOpenAiBackend::new(backend, policy.clone()));

    let hooked_for_task = hooked.clone();
    let handle = tokio::spawn(async move {
        hooked_for_task
            .chat_completion(request_for("gpt-mesh"))
            .await
    });

    // Wait until the backend has returned and `fire()` has started
    // awaiting the terminal hook (which then hangs), then cancel the
    // exchange the way an outer timeout or client disconnect would —
    // this is the mid-`fire`-await drop Finding A is about.
    policy.started.notified().await;
    handle.abort();
    let _ = handle.await;

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(1);
    loop {
        if !policy.terminals.lock().unwrap().is_empty() {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "terminal event never fired after cancelling mid-terminal-hook-await"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.as_slice(), [TerminalRecord::Cancelled]);
}

/// A policy whose `on_effective_chat_completion` hangs forever (after
/// signalling `started`). This lets a test park the exchange future
/// *inside* that pre-backend await — the window Gap B's fix targets —
/// rather than only during the backend call itself.
#[derive(Default)]
struct HangOnEffectivePolicy {
    started: tokio::sync::Notify,
    terminals: Mutex<Vec<TerminalRecord>>,
}

#[async_trait]
impl OpenAiHookPolicy for HangOnEffectivePolicy {
    async fn on_effective_chat_completion(
        &self,
        _request: &ChatCompletionRequest,
        _route: &ChatExchangeRoute,
    ) {
        self.started.notify_one();
        std::future::pending::<()>().await;
    }

    async fn on_chat_completion_terminal(
        &self,
        _request: &ChatCompletionRequest,
        _exchange_id: &str,
        outcome: &ChatCompletionOutcome<'_>,
    ) {
        if let ChatCompletionOutcome::Cancelled = outcome {
            self.terminals
                .lock()
                .unwrap()
                .push(TerminalRecord::Cancelled);
        }
    }
}

/// Reproduces Gap B: the guard used to be armed right before the
/// backend call, *after* `before_chat_completion` and
/// `on_effective_chat_completion` had already run. A future dropped
/// during either of those pre-backend awaits got no terminal event at
/// all. Arming the guard immediately after the exchange id is minted
/// closes that window.
#[tokio::test]
async fn dropping_the_future_during_on_effective_chat_completion_still_fires_exactly_one_terminal_event()
 {
    let backend = Arc::new(RecordingBackend {
        seen: Mutex::new(None),
    });
    let policy = Arc::new(HangOnEffectivePolicy::default());
    let hooked = Arc::new(HookedOpenAiBackend::new(backend, policy.clone()));

    let hooked_for_task = hooked.clone();
    let handle = tokio::spawn(async move {
        hooked_for_task
            .chat_completion(request_for("gpt-mesh"))
            .await
    });

    // Wait until the exchange is parked inside `on_effective_chat_completion`,
    // then cancel it the way an outer timeout or client disconnect would.
    policy.started.notified().await;
    handle.abort();
    let _ = handle.await;

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(1);
    loop {
        if !policy.terminals.lock().unwrap().is_empty() {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "terminal event never fired after cancelling during on_effective_chat_completion"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.as_slice(), [TerminalRecord::Cancelled]);
}

/// A policy that denies every request and whose terminal hook hangs
/// forever on a `Denied` outcome (after signalling `started`), but
/// records `Cancelled` immediately. This lets a test park the exchange
/// future *inside* the denial path's `TerminalGuard::fire` await.
#[derive(Default)]
struct DenyingHangOnTerminalPolicy {
    started: tokio::sync::Notify,
    terminals: Mutex<Vec<TerminalRecord>>,
}

#[async_trait]
impl OpenAiHookPolicy for DenyingHangOnTerminalPolicy {
    async fn before_chat_completion(
        &self,
        _request: &mut ChatCompletionRequest,
    ) -> OpenAiResult<ChatHookOutcome> {
        Err(crate::errors::OpenAiError::invalid_request(
            "denied by policy",
        ))
    }

    async fn on_chat_completion_terminal(
        &self,
        _request: &ChatCompletionRequest,
        _exchange_id: &str,
        outcome: &ChatCompletionOutcome<'_>,
    ) {
        match outcome {
            ChatCompletionOutcome::Denied { .. } => {
                self.started.notify_one();
                std::future::pending::<()>().await;
            }
            ChatCompletionOutcome::Cancelled => {
                self.terminals
                    .lock()
                    .unwrap()
                    .push(TerminalRecord::Cancelled);
            }
            _ => {}
        }
    }
}

/// Reproduces Gap A: the denial path used to call
/// `on_chat_completion_terminal` directly, bypassing `TerminalGuard`
/// entirely. If that direct call's await was cancelled mid-flight, the
/// denied exchange got zero terminal events. Routing the denial through
/// `guard.fire` gives it the same exactly-once + `Drop`-fallback
/// guarantee as the admitted path.
#[tokio::test]
async fn cancelling_a_denied_requests_terminal_delivery_still_fires_exactly_one_terminal_event()
{
    let backend = Arc::new(RecordingBackend {
        seen: Mutex::new(None),
    });
    let policy = Arc::new(DenyingHangOnTerminalPolicy::default());
    let hooked = Arc::new(HookedOpenAiBackend::new(backend, policy.clone()));

    let hooked_for_task = hooked.clone();
    let handle = tokio::spawn(async move {
        hooked_for_task
            .chat_completion(request_for("gpt-mesh"))
            .await
    });

    // Wait until the denial's `guard.fire` call has started awaiting the
    // terminal hook (which then hangs), then cancel the exchange the way
    // an outer timeout or client disconnect would.
    policy.started.notified().await;
    handle.abort();
    let _ = handle.await;

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(1);
    loop {
        if !policy.terminals.lock().unwrap().is_empty() {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "terminal event never fired after cancelling mid-denial-terminal-hook-await"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.as_slice(), [TerminalRecord::Cancelled]);
}

struct CapsuleMintingPolicy;

#[async_trait]
impl OpenAiHookPolicy for CapsuleMintingPolicy {
    async fn capsule_marker_for_response(
        &self,
        _request: &ChatCompletionRequest,
        response: &ChatCompletionResponse,
    ) -> Option<CapsuleMarker> {
        Some(CapsuleMarker {
            capsule_id: format!("capsule-{}", response.id),
            nonce: "test-nonce".to_string(),
        })
    }
}

#[tokio::test]
async fn capsule_marker_from_hook_is_attached_to_response_before_terminal_fires() {
    let backend = Arc::new(RecordingBackend {
        seen: Mutex::new(None),
    });
    let hooked = HookedOpenAiBackend::new(backend, Arc::new(CapsuleMintingPolicy));

    let response = hooked
        .chat_completion(request_for("gpt-mesh"))
        .await
        .expect("backend call succeeds");

    // The response returned to the router carries the marker (this is
    // what lets router.rs promote it to an `X-Capsule-Id` header).
    let marker = response.capsule_marker.expect("marker attached");
    assert_eq!(marker.capsule_id, format!("capsule-{}", response.id));
    assert_eq!(marker.nonce, "test-nonce");
}

#[derive(Default)]
struct TerminalSnapshotPolicy {
    marker_seen_at_terminal: Mutex<Option<Option<CapsuleMarker>>>,
}

#[async_trait]
impl OpenAiHookPolicy for TerminalSnapshotPolicy {
    async fn capsule_marker_for_response(
        &self,
        _request: &ChatCompletionRequest,
        _response: &ChatCompletionResponse,
    ) -> Option<CapsuleMarker> {
        Some(CapsuleMarker {
            capsule_id: "capsule-fixed".to_string(),
            nonce: "n".to_string(),
        })
    }

    async fn on_chat_completion_terminal(
        &self,
        _request: &ChatCompletionRequest,
        _exchange_id: &str,
        outcome: &ChatCompletionOutcome<'_>,
    ) {
        let marker = match outcome {
            ChatCompletionOutcome::Success { response } => response.capsule_marker.clone(),
            _ => None,
        };
        *self.marker_seen_at_terminal.lock().unwrap() = Some(marker);
    }
}

#[tokio::test]
async fn terminal_hook_observes_the_minted_marker_so_a_plugin_can_correlate_the_ack() {
    let backend = Arc::new(RecordingBackend {
        seen: Mutex::new(None),
    });
    let policy = Arc::new(TerminalSnapshotPolicy::default());
    let hooked = HookedOpenAiBackend::new(backend, policy.clone());

    hooked
        .chat_completion(request_for("gpt-mesh"))
        .await
        .expect("backend call succeeds");

    let seen = policy
        .marker_seen_at_terminal
        .lock()
        .unwrap()
        .clone()
        .expect("terminal fired");
    let marker = seen.expect("marker visible inside on_chat_completion_terminal");
    assert_eq!(marker.capsule_id, "capsule-fixed");
}

#[derive(Default)]
struct RequestSnapshotPolicy {
    consumes: bool,
    model_seen_at_terminal: Mutex<Option<String>>,
}

#[async_trait]
impl OpenAiHookPolicy for RequestSnapshotPolicy {
    async fn on_chat_completion_terminal(
        &self,
        request: &ChatCompletionRequest,
        _exchange_id: &str,
        _outcome: &ChatCompletionOutcome<'_>,
    ) {
        *self.model_seen_at_terminal.lock().unwrap() = Some(request.model.clone());
    }

    fn observes_dispatched_request(&self) -> bool {
        self.consumes
    }
}

#[tokio::test]
async fn observes_dispatched_request_true_gets_the_real_post_dispatch_request() {
    let backend = Arc::new(RecordingBackend {
        seen: Mutex::new(None),
    });
    let policy = Arc::new(RequestSnapshotPolicy {
        consumes: true,
        ..Default::default()
    });
    let hooked = HookedOpenAiBackend::new(backend, policy.clone());

    let response = hooked
        .chat_completion(request_for("gpt-mesh"))
        .await
        .expect("backend call succeeds");

    assert_eq!(response.model, "gpt-mesh");
    assert_eq!(
        policy.model_seen_at_terminal.lock().unwrap().clone(),
        Some("gpt-mesh".to_string())
    );
}

#[tokio::test]
async fn observes_dispatched_request_false_skips_the_clone_and_sees_a_default_request() {
    let backend = Arc::new(RecordingBackend {
        seen: Mutex::new(None),
    });
    let policy = Arc::new(RequestSnapshotPolicy {
        consumes: false,
        ..Default::default()
    });
    let hooked = HookedOpenAiBackend::new(backend, policy.clone());

    // The backend still dispatches the real request (the response model
    // proves it); only the post-dispatch hook snapshot is skipped.
    let response = hooked
        .chat_completion(request_for("gpt-mesh"))
        .await
        .expect("backend call succeeds");
    assert_eq!(response.model, "gpt-mesh");

    // An empty model here (not "gpt-mesh") proves HookedOpenAiBackend
    // handed the hook a default placeholder instead of cloning the real
    // request, matching the `observes_dispatched_request = false` contract.
    assert_eq!(
        policy.model_seen_at_terminal.lock().unwrap().clone(),
        Some(String::new())
    );
}

#[tokio::test]
async fn default_hook_policy_mints_no_capsule_marker() {
    let backend = Arc::new(RecordingBackend {
        seen: Mutex::new(None),
    });
    let hooked = HookedOpenAiBackend::new(backend, Arc::new(RecordingPolicy::default()));

    let response = hooked
        .chat_completion(request_for("gpt-mesh"))
        .await
        .expect("backend call succeeds");

    assert!(response.capsule_marker.is_none());
}

#[tokio::test]
async fn denial_by_before_hook_skips_dispatch_and_effective_request_but_reports_terminal() {
    let backend = Arc::new(RecordingBackend {
        seen: Mutex::new(None),
    });
    let policy = Arc::new(RecordingPolicy {
        deny: true,
        ..RecordingPolicy::default()
    });
    let hooked = HookedOpenAiBackend::new(backend.clone(), policy.clone());

    let error = hooked
        .chat_completion(request_for("gpt-mesh"))
        .await
        .expect_err("policy denies the request");
    assert_eq!(error.status().as_u16(), 400);

    // A denied request must never reach the backend or be reported as dispatched.
    assert!(backend.seen.lock().unwrap().is_none());
    assert!(policy.effective.lock().unwrap().is_empty());

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert!(matches!(
        &terminals[0],
        TerminalRecord::Denied { status: 400, reason }
            if reason.contains("denied by policy")
    ));
}

struct StreamingBackend {
    chunks: Mutex<Option<Vec<OpenAiResult<ChatCompletionChunk>>>>,
}

impl StreamingBackend {
    fn new(chunks: Vec<OpenAiResult<ChatCompletionChunk>>) -> Self {
        Self {
            chunks: Mutex::new(Some(chunks)),
        }
    }
}

#[async_trait]
impl OpenAiBackend for StreamingBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(Vec::new())
    }

    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        unreachable!("streaming tests only call chat_completion_stream")
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        let chunks = self.chunks.lock().unwrap().take().expect("chunks");
        Ok(Box::pin(futures_util::stream::iter(chunks)))
    }
}

/// A backend whose stream yields one real chunk, then never resolves
/// again — long enough for a test to observe a chunk before dropping the
/// stream, mirroring a client that disconnects mid-stream rather than
/// before receiving anything at all.
struct HangingStreamBackend;

#[async_trait]
impl OpenAiBackend for HangingStreamBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(Vec::new())
    }

    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        unreachable!("streaming tests only call chat_completion_stream")
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        let first = ChatCompletionChunk::delta(request.model, "partial");
        Ok(Box::pin(
            futures_util::stream::once(async move { Ok(first) })
                .chain(futures_util::stream::pending()),
        ))
    }
}

/// Terminal delivery for a stream fires via
/// [`TerminalGuard::fire_detached`] — a spawned, detached task, since
/// `Stream::poll_next` can't `.await` it inline — so it lands sometime
/// after `poll_next` returns rather than before. Tests must wait for it
/// rather than asserting synchronously the instant the stream stops
/// yielding items.
async fn wait_for_terminal(policy: &RecordingPolicy) {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(1);
    loop {
        if !policy.terminals.lock().unwrap().is_empty() {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "terminal event never fired"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
}

#[tokio::test]
async fn streaming_exchange_that_ends_normally_fires_stream_completed_terminal_exactly_once() {
    let backend = Arc::new(StreamingBackend::new(vec![
        Ok(ChatCompletionChunk::delta("gpt-mesh", "hi")),
        Ok(ChatCompletionChunk::done("gpt-mesh")),
    ]));
    let policy = Arc::new(RecordingPolicy::default());
    let hooked = HookedOpenAiBackend::new(backend, policy.clone());

    let mut stream = hooked
        .chat_completion_stream(request_for("gpt-mesh"), OpenAiRequestContext::new())
        .await
        .expect("stream created");
    while stream
        .next()
        .await
        .transpose()
        .expect("no chunk errors")
        .is_some()
    {}
    wait_for_terminal(&policy).await;

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.as_slice(), [TerminalRecord::StreamCompleted]);
}

#[tokio::test]
async fn streaming_exchange_with_an_error_chunk_fires_error_terminal_exactly_once() {
    let backend = Arc::new(StreamingBackend::new(vec![
        Ok(ChatCompletionChunk::delta("gpt-mesh", "hi")),
        Err(crate::errors::OpenAiError::backend("upstream exploded")),
    ]));
    let policy = Arc::new(RecordingPolicy::default());
    let hooked = HookedOpenAiBackend::new(backend, policy.clone());

    let mut stream = hooked
        .chat_completion_stream(request_for("gpt-mesh"), OpenAiRequestContext::new())
        .await
        .expect("stream created");
    while let Some(item) = stream.next().await {
        let _ = item;
    }
    wait_for_terminal(&policy).await;

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert!(matches!(
        &terminals[0],
        TerminalRecord::Error { status: 502, message }
            if message.contains("upstream exploded")
    ));
}

/// Reproduces the streaming counterpart of
/// `dropping_the_backend_future_still_fires_exactly_one_terminal_event`:
/// an outer timeout or client disconnect drops the stream — after it has
/// already delivered a chunk — before it ends on its own, so without
/// `TerminalGuardedChatStream` the exchange would never get a terminal
/// event at all.
#[tokio::test]
async fn streamed_exchange_dropped_mid_stream_fires_exactly_one_cancelled_terminal() {
    let backend = Arc::new(HangingStreamBackend);
    let policy = Arc::new(RecordingPolicy::default());
    let hooked = HookedOpenAiBackend::new(backend, policy.clone());

    let mut stream = hooked
        .chat_completion_stream(request_for("gpt-mesh"), OpenAiRequestContext::new())
        .await
        .expect("stream created");
    let first = stream.next().await;
    assert!(matches!(first, Some(Ok(_))), "first chunk should flow");
    drop(stream);
    wait_for_terminal(&policy).await;

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.as_slice(), [TerminalRecord::Cancelled]);
}

#[tokio::test]
async fn streaming_denial_by_before_hook_never_creates_a_stream_but_reports_terminal() {
    let backend = Arc::new(StreamingBackend::new(Vec::new()));
    let policy = Arc::new(RecordingPolicy {
        deny: true,
        ..RecordingPolicy::default()
    });
    let hooked = HookedOpenAiBackend::new(backend, policy.clone());

    let error = match hooked
        .chat_completion_stream(request_for("gpt-mesh"), OpenAiRequestContext::new())
        .await
    {
        Ok(_) => panic!("policy denies the request"),
        Err(error) => error,
    };
    assert_eq!(error.status().as_u16(), 400);

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert!(matches!(
        &terminals[0],
        TerminalRecord::Denied { status: 400, reason }
            if reason.contains("denied by policy")
    ));
}

#[tokio::test]
async fn streaming_backend_failure_before_any_chunk_reports_terminal_error_exactly_once() {
    let backend = Arc::new(FailingBackend);
    let policy = Arc::new(RecordingPolicy::default());
    let hooked = HookedOpenAiBackend::new(backend, policy.clone());

    let error = match hooked
        .chat_completion_stream(request_for("gpt-mesh"), OpenAiRequestContext::new())
        .await
    {
        Ok(_) => panic!("backend fails before yielding a stream"),
        Err(error) => error,
    };
    assert_eq!(error.status().as_u16(), 502);

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert!(matches!(
        &terminals[0],
        TerminalRecord::Error { status: 502, message }
            if message.contains("upstream exploded")
    ));
}
