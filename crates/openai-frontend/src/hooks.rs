use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::{
    backend::{
        ChatCompletionStream, CompletionStream, OpenAiBackend, OpenAiRequestContext, OpenAiResult,
    },
    chat::{
        CapsuleMarker, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
        ChatMessage, MessageContent, MessageContentPart, capsule_id_is_valid,
    },
    completions::{CompletionRequest, CompletionResponse},
    models::ModelObject,
};

pub const MESH_HOOKS_FIELD: &str = "mesh_hooks";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatHookOutcome {
    pub actions: Vec<ChatHookAction>,
}

impl ChatHookOutcome {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn injected(text: impl Into<String>) -> Self {
        Self {
            actions: vec![ChatHookAction::InjectText { text: text.into() }],
        }
    }

    pub fn injected_with_consumed_media(text: impl Into<String>, media: ChatMediaRef) -> Self {
        Self {
            actions: vec![
                ChatHookAction::ConsumeMedia { media },
                ChatHookAction::InjectText { text: text.into() },
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatHookAction {
    InjectText { text: String },
    ConsumeMedia { media: ChatMediaRef },
    None,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrefillHookSignals {
    pub first_token_entropy: f64,
    pub first_token_margin: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationHookSignals {
    pub n_decoded: i64,
    pub window_tokens: u32,
    pub mean_entropy: f64,
    pub max_entropy: f64,
    pub mean_margin: f64,
    pub min_margin: f64,
    pub high_entropy_count: u32,
    pub repetition_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMediaRef {
    pub kind: ChatMediaKind,
    pub url: String,
    pub user_text: String,
    pub message_index: usize,
    pub part_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatMediaKind {
    Image,
    Audio,
    Video,
}

#[async_trait]
pub trait OpenAiHookPolicy: Send + Sync + 'static {
    async fn before_chat_completion(
        &self,
        _request: &mut ChatCompletionRequest,
    ) -> OpenAiResult<ChatHookOutcome> {
        Ok(ChatHookOutcome::none())
    }

    async fn after_prefill(
        &self,
        _request: &mut ChatCompletionRequest,
        _signals: PrefillHookSignals,
    ) -> OpenAiResult<ChatHookOutcome> {
        Ok(ChatHookOutcome::none())
    }

    async fn mid_generation(
        &self,
        _request: &mut ChatCompletionRequest,
        _signals: GenerationHookSignals,
    ) -> OpenAiResult<ChatHookOutcome> {
        Ok(ChatHookOutcome::none())
    }

    /// Observe the effective (post-mutation) request immediately before it is
    /// dispatched to the backend for a non-streaming chat completion.
    ///
    /// This fires after [`Self::before_chat_completion`] has run and its
    /// outcome has been applied, so `request` reflects what will actually be
    /// sent. The route carries only what this layer knows about backend
    /// selection: the frontend dispatches every request to one already-chosen
    /// [`crate::backend::OpenAiBackend`], so there is no per-request backend
    /// identity to report here.
    async fn on_effective_chat_completion(
        &self,
        _request: &ChatCompletionRequest,
        _route: &ChatExchangeRoute,
    ) {
    }

    /// Observe the terminal outcome of a non-streaming chat completion:
    /// success, a backend error, or denial by an earlier hook.
    ///
    /// `exchange_id` is the same value [`ChatExchangeRoute::exchange_id`]
    /// carried on this exchange's [`Self::on_effective_chat_completion`]
    /// call (or, for a denied request, the id minted for an exchange that
    /// never reached admission) — a plugin observing both events can pair
    /// them without guessing from timing on a model shared by concurrent
    /// requests.
    async fn on_chat_completion_terminal(
        &self,
        _request: &ChatCompletionRequest,
        _exchange_id: &str,
        _outcome: &ChatCompletionOutcome<'_>,
    ) {
    }

    /// Mint an optional rung-ladder response-leg marker for a successful
    /// non-streaming chat completion.
    ///
    /// Fires once, after the backend has returned a response and before
    /// [`Self::on_chat_completion_terminal`] and the HTTP response are
    /// produced. A `Some` return is attached to
    /// [`ChatCompletionResponse::capsule_marker`], which the router turns
    /// into an `X-Capsule-Id` response header (see
    /// `frontend_lifecycle_middleware` in `router.rs`) — the write-capable
    /// half of the response leg that a plain observer method cannot provide,
    /// since every other hook method here takes `&ChatCompletionResponse`.
    /// Default: no marker (unchanged behavior for existing implementors).
    async fn capsule_marker_for_response(
        &self,
        _request: &ChatCompletionRequest,
        _response: &ChatCompletionResponse,
    ) -> Option<CapsuleMarker> {
        None
    }

    /// Whether this policy reads the `request` argument passed to
    /// [`Self::on_chat_completion_terminal`] or
    /// [`Self::capsule_marker_for_response`]. Both fire after the backend
    /// has already taken the effective request by value, so
    /// `HookedOpenAiBackend` must clone it up front to still have one to
    /// hand them — a clone that copies real bytes (message content, inline
    /// media) on every non-streaming completion regardless of whether
    /// either hook looks at it. [`Self::on_effective_chat_completion`]
    /// doesn't need this: it runs before that move, on `&request` directly.
    ///
    /// Defaults to `true` (always clone) so a policy that starts reading
    /// the post-dispatch request keeps getting the real one without also
    /// having to remember to flip this. A policy that only implements the
    /// pre-dispatch hooks — or ignores `request` in the post-dispatch ones —
    /// can override this to `false` to skip the clone; in that case the two
    /// post-dispatch hooks still fire, but with a default/empty
    /// `ChatCompletionRequest` in place of the real one.
    fn observes_dispatched_request(&self) -> bool {
        true
    }
}

/// The route information available to a hook at dispatch time.
///
/// Deliberately narrow: see [`OpenAiHookPolicy::on_effective_chat_completion`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatExchangeRoute {
    pub model: String,
    /// Stable id for this exchange, shared with the terminal event's
    /// `exchange_id` — see [`OpenAiHookPolicy::on_chat_completion_terminal`].
    pub exchange_id: String,
}

impl ChatExchangeRoute {
    pub fn for_request(request: &ChatCompletionRequest, exchange_id: impl Into<String>) -> Self {
        Self {
            model: request.model.clone(),
            exchange_id: exchange_id.into(),
        }
    }
}

/// The terminal outcome of a non-streaming chat completion, as seen by
/// [`OpenAiHookPolicy::on_chat_completion_terminal`].
///
/// `#[non_exhaustive]`: [`Self::Cancelled`] was added after this type
/// shipped, precisely so a downstream `match` without a wildcard arm fails
/// loudly instead of silently missing the case it needs to handle.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ChatCompletionOutcome<'a> {
    /// The backend returned a response.
    Success {
        response: &'a ChatCompletionResponse,
    },
    /// The backend call failed or timed out.
    Error { status: u16, message: &'a str },
    /// An earlier hook (`before_chat_completion`) denied the request before
    /// it reached the backend.
    Denied { status: u16, reason: &'a str },
    /// The exchange's future was dropped (an outer timeout, or the client
    /// disconnecting) before the backend call returned — see
    /// [`TerminalGuard`]. There is no HTTP status or response to report;
    /// this variant exists so every admitted exchange still gets exactly
    /// one terminal callback instead of none.
    Cancelled,
    /// A streaming chat completion sent every chunk to the client and the
    /// underlying stream ended on its own (as opposed to being dropped
    /// mid-stream, which reports [`Self::Cancelled`] instead). Streaming
    /// dispatches whole [`crate::chat::ChatCompletionChunk`]s rather than
    /// one assembled [`ChatCompletionResponse`], so unlike [`Self::Success`]
    /// there is no response to report here.
    StreamCompleted,
}

/// Guarantees exactly one [`OpenAiHookPolicy::on_chat_completion_terminal`]
/// call per admitted exchange, even if the future driving the backend call
/// is dropped mid-flight (an outer request timeout, or the client
/// disconnecting) before it can report success/error itself.
///
/// [`Self::fire`] is the normal path: it awaits the hook, then marks the
/// guard fired. If `fire`'s future is itself dropped mid-await (the
/// enclosing future was cancelled while the hook call was in flight) the
/// guard is still unfired, so [`Drop::drop`] below still fires the terminal
/// callback — exactly one call either way. If the caller instead drops the
/// guard without calling `fire` at all — because the enclosing future was
/// cancelled before `fire` was even invoked — [`Drop::drop`] spawns the
/// terminal call with
/// [`ChatCompletionOutcome::Cancelled`] so it still happens, just detached
/// from (and unable to block) whatever cancelled the original future.
pub struct TerminalGuard {
    hooks: Arc<dyn OpenAiHookPolicy>,
    request: ChatCompletionRequest,
    exchange_id: String,
    fired: bool,
}

/// A [`ChatCompletionOutcome`] with no borrowed fields, for the streaming
/// terminal path: [`TerminalGuard::fire_detached`] hands the outcome to a
/// spawned task that outlives the caller's stack frame, so it needs data it
/// owns rather than a reference into a local that's about to go away.
/// Deliberately narrower than [`ChatCompletionOutcome`] — it omits
/// [`ChatCompletionOutcome::Success`], which streaming never has a
/// [`ChatCompletionResponse`] to report; [`ChatCompletionOutcome::Denied`],
/// which is always fired inline (before any stream exists to detach from);
/// and [`ChatCompletionOutcome::Cancelled`], which [`Drop`] below fires
/// directly without going through `fire_detached` at all.
enum OwnedChatCompletionOutcome {
    Error { status: u16, message: String },
    StreamCompleted,
}

impl OwnedChatCompletionOutcome {
    fn as_ref(&self) -> ChatCompletionOutcome<'_> {
        match self {
            Self::Error { status, message } => ChatCompletionOutcome::Error {
                status: *status,
                message,
            },
            Self::StreamCompleted => ChatCompletionOutcome::StreamCompleted,
        }
    }
}

impl TerminalGuard {
    pub fn new(
        hooks: Arc<dyn OpenAiHookPolicy>,
        request: ChatCompletionRequest,
        exchange_id: String,
    ) -> Self {
        Self {
            hooks,
            request,
            exchange_id,
            fired: false,
        }
    }

    pub fn set_request(&mut self, request: ChatCompletionRequest) {
        self.request = request;
    }

    pub async fn fire(mut self, outcome: &ChatCompletionOutcome<'_>) {
        self.hooks
            .on_chat_completion_terminal(&self.request, &self.exchange_id, outcome)
            .await;
        self.fired = true;
    }

    /// Fire the terminal callback from a context that cannot `.await` — a
    /// `Stream::poll_next` implementation, specifically — by handing it to a
    /// detached task on the current Tokio runtime, exactly like [`Drop`]'s
    /// own fallback below. Consumes `self` (after marking it fired) so the
    /// guard's own `Drop` can never also fire once this returns.
    fn fire_detached(mut self, outcome: OwnedChatCompletionOutcome) {
        self.fired = true;
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::debug!(
                exchange_id = %self.exchange_id,
                "TerminalGuard dropped outside a Tokio runtime; skipping terminal callback"
            );
            return;
        };
        let hooks = self.hooks.clone();
        let request = std::mem::take(&mut self.request);
        let exchange_id = std::mem::take(&mut self.exchange_id);
        handle.spawn(async move {
            hooks
                .on_chat_completion_terminal(&request, &exchange_id, &outcome.as_ref())
                .await;
        });
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if self.fired {
            return;
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            // No runtime to spawn onto — this only happens at process/runtime
            // teardown, where a detached terminal callback would be moot anyway.
            tracing::debug!(
                exchange_id = %self.exchange_id,
                "TerminalGuard dropped outside a Tokio runtime; skipping terminal callback"
            );
            return;
        };
        let hooks = self.hooks.clone();
        let request = std::mem::take(&mut self.request);
        let exchange_id = std::mem::take(&mut self.exchange_id);
        handle.spawn(async move {
            hooks
                .on_chat_completion_terminal(
                    &request,
                    &exchange_id,
                    &ChatCompletionOutcome::Cancelled,
                )
                .await;
        });
    }
}

/// Wraps a [`ChatCompletionStream`] so its admitted exchange still gets
/// exactly one terminal callback, the same guarantee
/// [`HookedOpenAiBackend::chat_completion_with_context`] gives a
/// non-streaming exchange via [`TerminalGuard`] — just adapted for a type
/// that can outlive the call that created it and whose `Stream::poll_next`
/// cannot `.await`.
///
/// - The stream ending on its own (`poll_next` returns `Ready(None)`) fires
///   [`ChatCompletionOutcome::StreamCompleted`].
/// - A chunk carrying an error fires [`ChatCompletionOutcome::Error`]
///   immediately — matching the non-streaming path, which never waits for a
///   graceful end once the backend has already reported failure.
/// - Both fire via [`TerminalGuard::fire_detached`], since neither can
///   `.await` inside `poll_next`.
/// - Dropping this wrapper before either of the above happens — an outer
///   timeout, or the client disconnecting mid-stream — drops the
///   still-armed [`TerminalGuard`], whose own `Drop` fires
///   [`ChatCompletionOutcome::Cancelled`]. Exactly one of
///   {`StreamCompleted`, `Error`, `Cancelled`} can ever happen, because each
///   path takes the guard out of `self.guard` (an `Option`) before firing,
///   and a `None` guard fires nothing on drop.
pub struct TerminalGuardedChatStream {
    inner: ChatCompletionStream,
    guard: Option<TerminalGuard>,
}

impl TerminalGuardedChatStream {
    pub fn new(inner: ChatCompletionStream, guard: TerminalGuard) -> ChatCompletionStream {
        Box::pin(Self {
            inner,
            guard: Some(guard),
        })
    }
}

impl futures_core::Stream for TerminalGuardedChatStream {
    type Item = OpenAiResult<ChatCompletionChunk>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let poll = this.inner.as_mut().poll_next(cx);
        match &poll {
            std::task::Poll::Ready(Some(Err(error))) => {
                if let Some(guard) = this.guard.take() {
                    guard.fire_detached(OwnedChatCompletionOutcome::Error {
                        status: error.status().as_u16(),
                        message: error.to_string(),
                    });
                }
            }
            std::task::Poll::Ready(None) => {
                if let Some(guard) = this.guard.take() {
                    guard.fire_detached(OwnedChatCompletionOutcome::StreamCompleted);
                }
            }
            std::task::Poll::Ready(Some(Ok(_))) | std::task::Poll::Pending => {}
        }
        poll
    }
}

pub struct HookedOpenAiBackend {
    backend: Arc<dyn OpenAiBackend>,
    hooks: Arc<dyn OpenAiHookPolicy>,
}

impl HookedOpenAiBackend {
    pub fn new(backend: Arc<dyn OpenAiBackend>, hooks: Arc<dyn OpenAiHookPolicy>) -> Self {
        Self { backend, hooks }
    }
}

#[async_trait]
impl OpenAiBackend for HookedOpenAiBackend {
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
        mut request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionResponse> {
        let exchange_id = uuid::Uuid::new_v4().to_string();
        // Armed immediately after minting the exchange id — before
        // `before_chat_completion` and `on_effective_chat_completion` run —
        // so a future dropped during either of those pre-backend awaits
        // still gets exactly one terminal callback via the guard's `Drop`.
        // The pre-mutation `request` clone is fine for that fallback: it
        // only ever surfaces on the `Cancelled` path, which doesn't need the
        // post-dispatch copy.
        let mut guard =
            TerminalGuard::new(self.hooks.clone(), request.clone(), exchange_id.clone());
        let outcome = match self.hooks.before_chat_completion(&mut request).await {
            Ok(outcome) => outcome,
            Err(error) => {
                let reason = error.to_string();
                let denial = ChatCompletionOutcome::Denied {
                    status: error.status().as_u16(),
                    reason: &reason,
                };
                guard.request = request.clone();
                guard.fire(&denial).await;
                return Err(error);
            }
        };
        apply_chat_hook_outcome(&mut request, &outcome);
        let route = ChatExchangeRoute::for_request(&request, exchange_id.clone());
        self.hooks
            .on_effective_chat_completion(&request, &route)
            .await;
        // Only clone the effective request when a hook actually observes it
        // after dispatch — `chat_completion_with_context` below takes
        // `request` by value, so without a hook that needs the post-dispatch
        // copy, moving the original straight in (no clone) is enough.
        let dispatched_request = if self.hooks.observes_dispatched_request() {
            request.clone()
        } else {
            ChatCompletionRequest::default()
        };
        // Swap in the post-mutation request now that dispatch is imminent,
        // so the backend-call `Cancelled` fallback and the final
        // success/error terminal both report what was actually sent.
        guard.request = dispatched_request.clone();
        let mut result = self
            .backend
            .chat_completion_with_context(request, context)
            .await;
        if let Ok(response) = &mut result
            && let Some(marker) = self
                .hooks
                .capsule_marker_for_response(&dispatched_request, &*response)
                .await
        {
            // A marker whose capsule id can't become a valid `X-Capsule-Id`
            // header must not be attached at all — otherwise a plugin
            // observing the terminal event below would see a capsule id the
            // client's own response never carried.
            if capsule_id_is_valid(&marker.capsule_id) {
                response.capsule_marker = Some(marker);
            } else {
                tracing::warn!(
                    capsule_id = %marker.capsule_id,
                    "dropping capsule marker: invalid capsule id"
                );
            }
        }
        let error_message;
        let terminal = match &result {
            Ok(response) => ChatCompletionOutcome::Success { response },
            Err(error) => {
                error_message = error.to_string();
                ChatCompletionOutcome::Error {
                    status: error.status().as_u16(),
                    message: &error_message,
                }
            }
        };
        guard.fire(&terminal).await;
        result
    }

    async fn chat_completion_stream(
        &self,
        mut request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        let exchange_id = uuid::Uuid::new_v4().to_string();
        // Same admission-time arming as `chat_completion_with_context` above
        // — see its comment. A future dropped while `before_chat_completion`
        // is still running still gets exactly one terminal callback.
        let mut guard =
            TerminalGuard::new(self.hooks.clone(), request.clone(), exchange_id.clone());
        let outcome = match self.hooks.before_chat_completion(&mut request).await {
            Ok(outcome) => outcome,
            Err(error) => {
                let reason = error.to_string();
                let denial = ChatCompletionOutcome::Denied {
                    status: error.status().as_u16(),
                    reason: &reason,
                };
                guard.set_request(request.clone());
                guard.fire(&denial).await;
                return Err(error);
            }
        };
        apply_chat_hook_outcome(&mut request, &outcome);
        let route = ChatExchangeRoute::for_request(&request, exchange_id.clone());
        self.hooks
            .on_effective_chat_completion(&request, &route)
            .await;
        // Post-mutation copy for the guard, mirroring the non-streaming path
        // — the stream itself will own `request` from here.
        guard.set_request(request.clone());
        match self.backend.chat_completion_stream(request, context).await {
            Ok(stream) => Ok(TerminalGuardedChatStream::new(stream, guard)),
            Err(error) => {
                // The backend failed before yielding a stream at all — an
                // `Error` terminal, exactly like a non-streaming backend
                // failure, not the `Cancelled` the guard's `Drop` would
                // report if left to fire on its own.
                let message = error.to_string();
                guard
                    .fire(&ChatCompletionOutcome::Error {
                        status: error.status().as_u16(),
                        message: &message,
                    })
                    .await;
                Err(error)
            }
        }
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
}

pub fn chat_mesh_hooks_enabled(request: &ChatCompletionRequest) -> bool {
    request
        .extra
        .get(MESH_HOOKS_FIELD)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

pub fn set_chat_mesh_hooks_enabled(request: &mut ChatCompletionRequest, enabled: bool) {
    request
        .extra
        .insert(MESH_HOOKS_FIELD.to_string(), Value::Bool(enabled));
}

pub fn inject_text_into_chat_messages(messages: &mut Vec<ChatMessage>, text: impl Into<String>) {
    let text = text.into();
    if text.is_empty() {
        return;
    }

    if let Some(message) = messages
        .iter_mut()
        .rev()
        .find(|message| message.role == "user")
    {
        inject_text_into_message(message, text);
    } else {
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text(text)),
            extra: Default::default(),
        });
    }
}

pub fn apply_chat_hook_outcome(request: &mut ChatCompletionRequest, outcome: &ChatHookOutcome) {
    for action in &outcome.actions {
        match action {
            ChatHookAction::InjectText { text } => {
                inject_text_into_chat_messages(&mut request.messages, text.clone());
            }
            ChatHookAction::ConsumeMedia { media } => {
                consume_chat_media(&mut request.messages, media);
            }
            ChatHookAction::None => {}
        }
    }
}

pub fn first_chat_media(messages: &[ChatMessage]) -> Option<ChatMediaRef> {
    messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, message)| message.role == "user")
        .and_then(|(message_index, message)| media_from_message(message_index, message))
}

fn inject_text_into_message(message: &mut ChatMessage, text: String) {
    match message.content.take() {
        Some(MessageContent::Text(existing)) => {
            message.content = Some(MessageContent::Text(format!("{text}{existing}")));
        }
        Some(MessageContent::Parts(mut parts)) => {
            parts.insert(
                0,
                MessageContentPart {
                    content_type: "text".to_string(),
                    text: Some(text),
                    extra: Default::default(),
                },
            );
            message.content = Some(MessageContent::Parts(parts));
        }
        Some(MessageContent::Other(_)) | None => {
            message.content = Some(MessageContent::Text(text));
        }
    }
}

fn media_from_message(message_index: usize, message: &ChatMessage) -> Option<ChatMediaRef> {
    let parts = match message.content.as_ref()? {
        MessageContent::Parts(parts) => parts,
        MessageContent::Text(_) | MessageContent::Other(_) => return None,
    };
    let user_text = parts
        .iter()
        .filter(|part| part.content_type == "text")
        .filter_map(|part| part.text.as_deref())
        .collect::<Vec<_>>()
        .join("\n");
    for (part_index, part) in parts.iter().enumerate() {
        if let Some(media) = media_from_part(message_index, part_index, part, &user_text) {
            return Some(media);
        }
    }
    None
}

fn media_from_part(
    message_index: usize,
    part_index: usize,
    part: &MessageContentPart,
    user_text: &str,
) -> Option<ChatMediaRef> {
    let kind = match part.content_type.as_str() {
        "image_url" | "input_image" | "image" => ChatMediaKind::Image,
        "input_audio" | "audio" | "audio_url" => ChatMediaKind::Audio,
        "input_video" | "video" | "video_url" => ChatMediaKind::Video,
        _ => return None,
    };
    let url = media_url(part)?;
    Some(ChatMediaRef {
        kind,
        url,
        user_text: user_text.to_string(),
        message_index,
        part_index,
    })
}

fn consume_chat_media(messages: &mut [ChatMessage], media: &ChatMediaRef) -> bool {
    let Some(message) = messages.get_mut(media.message_index) else {
        return false;
    };
    consume_message_media(message, media)
}

fn consume_message_media(message: &mut ChatMessage, media: &ChatMediaRef) -> bool {
    if message.role != "user" {
        return false;
    }
    let Some(MessageContent::Parts(parts)) = message.content.as_mut() else {
        return false;
    };
    let Some(part) = parts.get(media.part_index) else {
        return false;
    };
    if !media_part_matches(part, media) {
        return false;
    }
    parts.remove(media.part_index);
    true
}

fn media_part_matches(part: &MessageContentPart, media: &ChatMediaRef) -> bool {
    media_from_part(media.message_index, media.part_index, part, "")
        .is_some_and(|candidate| candidate.kind == media.kind && candidate.url == media.url)
}

fn media_url(part: &MessageContentPart) -> Option<String> {
    for key in [
        "image_url",
        "input_image",
        "image",
        "input_audio",
        "audio",
        "audio_url",
        "input_video",
        "video",
        "video_url",
        "url",
    ] {
        if let Some(value) = part.extra.get(key) {
            if let Some(url) = value.as_str() {
                return Some(url.to_string());
            }
            if let Some(url) = value.get("url").and_then(Value::as_str) {
                return Some(url.to_string());
            }
            if let Some(data_url) = inline_media_data_url(key, value) {
                return Some(data_url);
            }
        }
    }
    None
}

fn inline_media_data_url(container_key: &str, value: &Value) -> Option<String> {
    let data = value.get("data").and_then(Value::as_str)?;
    if data.trim_start().starts_with("data:") {
        return Some(data.to_string());
    }
    let mime_type = value
        .get("mime_type")
        .or_else(|| value.get("media_type"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .or_else(|| {
            value
                .get("format")
                .and_then(Value::as_str)
                .and_then(|format| mime_type_from_format(container_key, format))
                .map(ToString::to_string)
        })
        .unwrap_or_else(|| default_media_mime_type(container_key).to_string());
    Some(format!("data:{mime_type};base64,{data}"))
}

fn mime_type_from_format(container_key: &str, format: &str) -> Option<&'static str> {
    let format = format.trim().trim_start_matches('.').to_ascii_lowercase();
    match format.as_str() {
        "wav" => Some("audio/wav"),
        "mp3" => Some("audio/mpeg"),
        "flac" => Some("audio/flac"),
        "ogg" | "opus" => Some("audio/ogg"),
        "webm" if is_audio_container(container_key) => Some("audio/webm"),
        "webm" => Some("video/webm"),
        "m4a" | "mp4" if is_audio_container(container_key) => Some("audio/mp4"),
        "mp4" => Some("video/mp4"),
        "mpeg" | "mpga" if is_audio_container(container_key) => Some("audio/mpeg"),
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

fn default_media_mime_type(container_key: &str) -> &'static str {
    if is_audio_container(container_key) {
        "audio/wav"
    } else if is_video_container(container_key) {
        "video/mp4"
    } else {
        "image/png"
    }
}

fn is_audio_container(container_key: &str) -> bool {
    matches!(container_key, "input_audio" | "audio" | "audio_url")
}

fn is_video_container(container_key: &str) -> bool {
    matches!(container_key, "input_video" | "video" | "video_url")
}

#[cfg(test)]
#[path = "hooks_tests.rs"]
mod tests;
