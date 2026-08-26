use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::{
    backend::{
        ChatCompletionStream, CompletionStream, OpenAiBackend, OpenAiRequestContext, OpenAiResult,
    },
    chat::{
        CapsuleMarker, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent,
        MessageContentPart, capsule_id_is_valid,
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
struct TerminalGuard {
    hooks: Arc<dyn OpenAiHookPolicy>,
    request: ChatCompletionRequest,
    exchange_id: String,
    fired: bool,
}

impl TerminalGuard {
    fn new(
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

    async fn fire(mut self, outcome: &ChatCompletionOutcome<'_>) {
        self.hooks
            .on_chat_completion_terminal(&self.request, &self.exchange_id, outcome)
            .await;
        self.fired = true;
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
        let outcome = self.hooks.before_chat_completion(&mut request).await?;
        apply_chat_hook_outcome(&mut request, &outcome);
        self.backend.chat_completion_stream(request, context).await
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
mod tests {
    use std::sync::{Arc, Mutex};

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
}
