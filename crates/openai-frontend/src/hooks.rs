use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::{
    backend::{
        ChatCompletionStream, CompletionStream, OpenAiBackend, OpenAiRequestContext, OpenAiResult,
    },
    chat::{
        CapsuleMarker, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent,
        MessageContentPart,
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
    async fn on_chat_completion_terminal(
        &self,
        _request: &ChatCompletionRequest,
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
}

/// The route information available to a hook at dispatch time.
///
/// Deliberately narrow: see [`OpenAiHookPolicy::on_effective_chat_completion`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatExchangeRoute {
    pub model: String,
}

impl ChatExchangeRoute {
    pub fn for_request(request: &ChatCompletionRequest) -> Self {
        Self {
            model: request.model.clone(),
        }
    }
}

/// The terminal outcome of a non-streaming chat completion, as seen by
/// [`OpenAiHookPolicy::on_chat_completion_terminal`].
#[derive(Debug, Clone, Copy)]
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
        let outcome = match self.hooks.before_chat_completion(&mut request).await {
            Ok(outcome) => outcome,
            Err(error) => {
                let reason = error.to_string();
                let denial = ChatCompletionOutcome::Denied {
                    status: error.status().as_u16(),
                    reason: &reason,
                };
                self.hooks
                    .on_chat_completion_terminal(&request, &denial)
                    .await;
                return Err(error);
            }
        };
        apply_chat_hook_outcome(&mut request, &outcome);
        let route = ChatExchangeRoute::for_request(&request);
        self.hooks
            .on_effective_chat_completion(&request, &route)
            .await;
        let mut result = self
            .backend
            .chat_completion_with_context(request.clone(), context)
            .await;
        if let Ok(response) = &mut result
            && let Some(marker) = self
                .hooks
                .capsule_marker_for_response(&request, &*response)
                .await
        {
            response.capsule_marker = Some(marker);
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
        self.hooks
            .on_chat_completion_terminal(&request, &terminal)
            .await;
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
