//! Bridges the two real OpenAI-exchange dispatch paths (see
//! `docs/plugins/openai-exchange-lifecycle-design-note.md`, #1331 M1/M2) to
//! an out-of-process plugin over the existing `PluginMeshEvent::Channel`
//! transport, so a plugin sees one unified stream regardless of which
//! in-process Rust hook interface produced an event.

use std::sync::Arc;

use async_trait::async_trait;
use openai_frontend::{
    CapsuleMarker, ChatCompletionOutcome, ChatCompletionRequest, ChatCompletionResponse,
    ChatExchangeRoute, OpenAiHookPolicy,
};
use serde::Serialize;

use super::PluginManager;

/// The single mesh channel both dispatch paths publish to.
pub const OPENAI_EXCHANGE_CHANNEL: &str = "openai.exchange.v1";

/// Which real dispatch path produced an [`OpenAiExchangeEnvelope`] — the two
/// paths M1 found are disjoint and don't share a request type, so the
/// envelope carries this instead of assuming one shape fits both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiExchangeDispatchPath {
    /// `openai-frontend`'s typed `OpenAiHookPolicy`/`HookedOpenAiBackend` seam.
    TypedFrontend,
    /// The raw-proxy ingress (`network/openai/ingress.rs`), used for
    /// plugin-served models; never sees a typed `ChatCompletionRequest`.
    RawProxy,
}

/// Which moment in an exchange's lifecycle an [`OpenAiExchangeEnvelope`]
/// reports — the same two moments [`OpenAiHookPolicy::on_effective_chat_completion`]
/// and [`OpenAiHookPolicy::on_chat_completion_terminal`] already observe for
/// path 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiExchangePhase {
    EffectiveRequest,
    Terminal,
}

/// The wire shape both dispatch paths publish on [`OPENAI_EXCHANGE_CHANNEL`].
/// Deliberately independent of `openai_frontend`'s typed request/response —
/// the raw-proxy path never has one — so one shape covers both paths without
/// either being forced into the other's type.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpenAiExchangeEnvelope {
    pub dispatch_path: OpenAiExchangeDispatchPath,
    pub phase: OpenAiExchangePhase,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    /// Present only on a `Terminal` envelope carrying a rung-ladder response
    /// marker (see [`CapsuleMarker`]) — the `capsule_id` already written into
    /// the client's response as `X-Capsule-Id`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capsule_id: Option<String>,
    /// The nonce the marker is correlated against, so a plugin observing
    /// this event knows what a later client ack must sign over.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nonce: Option<String>,
}

impl OpenAiExchangeEnvelope {
    pub fn effective(dispatch_path: OpenAiExchangeDispatchPath, model: impl Into<String>) -> Self {
        Self {
            dispatch_path,
            phase: OpenAiExchangePhase::EffectiveRequest,
            model: model.into(),
            status: None,
            capsule_id: None,
            nonce: None,
        }
    }

    pub fn terminal(
        dispatch_path: OpenAiExchangeDispatchPath,
        model: impl Into<String>,
        status: Option<u16>,
        marker: Option<CapsuleMarker>,
    ) -> Self {
        Self {
            dispatch_path,
            phase: OpenAiExchangePhase::Terminal,
            model: model.into(),
            status,
            capsule_id: marker.as_ref().map(|marker| marker.capsule_id.clone()),
            nonce: marker.as_ref().map(|marker| marker.nonce.clone()),
        }
    }
}

/// Publishes [`OpenAiExchangeEnvelope`]s to whatever is subscribed on
/// [`OPENAI_EXCHANGE_CHANNEL`] — an out-of-process plugin in production, a
/// recording double in tests. Fire-and-forget by design, mirroring
/// [`OpenAiHookPolicy`]'s own observer methods: exchange delivery to a
/// plugin must never affect whether the client's own request succeeds.
#[async_trait]
pub trait OpenAiExchangeChannel: Send + Sync + 'static {
    async fn publish(&self, event: &OpenAiExchangeEnvelope);
}

#[async_trait]
impl OpenAiExchangeChannel for PluginManager {
    async fn publish(&self, event: &OpenAiExchangeEnvelope) {
        let body = match serde_json::to_vec(event) {
            Ok(body) => body,
            Err(error) => {
                tracing::warn!(%error, "failed to serialize openai exchange event");
                return;
            }
        };
        if let Err(error) = self
            .broadcast_channel_message(OPENAI_EXCHANGE_CHANNEL, "application/json", body)
            .await
        {
            tracing::warn!(%error, "failed to publish openai exchange event to plugins");
        }
    }
}

/// Bridges path 1 (`openai-frontend`'s typed hook seam) to
/// [`OpenAiExchangeChannel`], so an out-of-process plugin observes the same
/// effective-request/terminal events this crate's `MeshAutoHookPolicy`
/// already sees in-process. Compose alongside other [`OpenAiHookPolicy`]
/// implementors rather than in place of them — this bridge only observes and
/// mints capsule markers, it never mutates or denies a request.
pub struct OpenAiExchangeHookBridge {
    channel: Arc<dyn OpenAiExchangeChannel>,
}

impl OpenAiExchangeHookBridge {
    pub fn new(channel: Arc<dyn OpenAiExchangeChannel>) -> Self {
        Self { channel }
    }
}

#[async_trait]
impl OpenAiHookPolicy for OpenAiExchangeHookBridge {
    async fn on_effective_chat_completion(
        &self,
        _request: &ChatCompletionRequest,
        route: &ChatExchangeRoute,
    ) {
        self.channel
            .publish(&OpenAiExchangeEnvelope::effective(
                OpenAiExchangeDispatchPath::TypedFrontend,
                route.model.clone(),
            ))
            .await;
    }

    async fn on_chat_completion_terminal(
        &self,
        request: &ChatCompletionRequest,
        outcome: &ChatCompletionOutcome<'_>,
    ) {
        let (status, marker) = match outcome {
            ChatCompletionOutcome::Success { response } => (200, response.capsule_marker.clone()),
            ChatCompletionOutcome::Error { status, .. } => (*status, None),
            ChatCompletionOutcome::Denied { status, .. } => (*status, None),
        };
        self.channel
            .publish(&OpenAiExchangeEnvelope::terminal(
                OpenAiExchangeDispatchPath::TypedFrontend,
                request.model.clone(),
                Some(status),
                marker,
            ))
            .await;
    }

    /// Reference nonce sourcing for the rung-ladder response leg: a
    /// client-contributed `client_nonce` (landing in `request.extra` via
    /// `ChatCompletionRequest`'s `#[serde(flatten)]` bag, the same mechanism
    /// `mesh_hooks` already uses) wins; absent that, mint a fallback rather
    /// than silently mislabeling it as client-supplied — mirroring
    /// `capsule-emit-mesh`'s own `client_nonce_source` tri-state
    /// (`client_supplied` / `sidecar_generated_fallback`).
    async fn capsule_marker_for_response(
        &self,
        request: &ChatCompletionRequest,
        response: &ChatCompletionResponse,
    ) -> Option<CapsuleMarker> {
        let nonce = request
            .extra
            .get("client_nonce")
            .and_then(|value| value.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| format!("fallback-{}", response.id));
        Some(CapsuleMarker {
            capsule_id: format!("capsule-{}", response.id),
            nonce,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use openai_frontend::{ChatCompletionOutcome, HookedOpenAiBackend, OpenAiBackend, Usage};

    use super::*;

    #[derive(Default)]
    struct RecordingChannel {
        events: Mutex<Vec<OpenAiExchangeEnvelope>>,
    }

    #[async_trait]
    impl OpenAiExchangeChannel for RecordingChannel {
        async fn publish(&self, event: &OpenAiExchangeEnvelope) {
            self.events.lock().unwrap().push(event.clone());
        }
    }

    struct EchoBackend;

    #[async_trait]
    impl OpenAiBackend for EchoBackend {
        async fn models(&self) -> openai_frontend::OpenAiResult<Vec<openai_frontend::ModelObject>> {
            Ok(Vec::new())
        }

        async fn chat_completion(
            &self,
            request: ChatCompletionRequest,
        ) -> openai_frontend::OpenAiResult<ChatCompletionResponse> {
            Ok(ChatCompletionResponse::new(
                request.model,
                "ok",
                Usage::new(1, 1),
            ))
        }

        async fn chat_completion_stream(
            &self,
            _request: ChatCompletionRequest,
            _context: openai_frontend::OpenAiRequestContext,
        ) -> openai_frontend::OpenAiResult<openai_frontend::ChatCompletionStream> {
            Ok(Box::pin(futures_util::stream::empty()))
        }
    }

    fn chat_request(model: &str) -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap()
    }

    /// Reference: a full request through `HookedOpenAiBackend` wired with
    /// this bridge publishes both the effective-request and terminal events
    /// on the typed-frontend path, and the terminal event carries the same
    /// capsule marker that (per the openai-frontend-crate tests) also became
    /// the client-visible `X-Capsule-Id` header — proving the plugin sees
    /// exactly what the client's response leg exposed, not a divergent copy.
    #[tokio::test]
    async fn typed_frontend_path_publishes_effective_and_terminal_with_capsule_marker() {
        let channel = Arc::new(RecordingChannel::default());
        let bridge = Arc::new(OpenAiExchangeHookBridge::new(channel.clone()));
        let hooked = HookedOpenAiBackend::new(Arc::new(EchoBackend), bridge);

        let response = hooked
            .chat_completion(chat_request("gpt-mesh"))
            .await
            .expect("backend call succeeds");

        let events = channel.events.lock().unwrap();
        assert_eq!(events.len(), 2, "one effective-request, one terminal");

        assert_eq!(
            events[0].dispatch_path,
            OpenAiExchangeDispatchPath::TypedFrontend
        );
        assert_eq!(events[0].phase, OpenAiExchangePhase::EffectiveRequest);
        assert_eq!(events[0].model, "gpt-mesh");

        assert_eq!(events[1].phase, OpenAiExchangePhase::Terminal);
        assert_eq!(events[1].status, Some(200));
        let capsule_id = events[1]
            .capsule_id
            .as_deref()
            .expect("terminal event carries the capsule id");
        assert_eq!(
            capsule_id,
            response
                .capsule_marker
                .as_ref()
                .expect("router-visible marker")
                .capsule_id
        );
    }

    #[tokio::test]
    async fn client_supplied_nonce_survives_into_the_terminal_event() {
        let channel = Arc::new(RecordingChannel::default());
        let bridge = Arc::new(OpenAiExchangeHookBridge::new(channel.clone()));
        let hooked = HookedOpenAiBackend::new(Arc::new(EchoBackend), bridge);

        let mut request = chat_request("gpt-mesh");
        request
            .extra
            .insert("client_nonce".to_string(), serde_json::json!("abc123"));

        hooked
            .chat_completion(request)
            .await
            .expect("backend call succeeds");

        let events = channel.events.lock().unwrap();
        assert_eq!(events[1].nonce.as_deref(), Some("abc123"));
    }

    /// A denial never reaches the backend, so there is no response to mint a
    /// marker from — the bridge's own terminal handling (not a stand-in) must
    /// publish a status-only event with no capsule id.
    #[tokio::test]
    async fn denied_outcome_publishes_terminal_without_a_capsule_marker() {
        let channel = Arc::new(RecordingChannel::default());
        let bridge = OpenAiExchangeHookBridge::new(channel.clone());
        let request = chat_request("gpt-mesh");
        let denial = ChatCompletionOutcome::Denied {
            status: 400,
            reason: "denied by policy",
        };

        bridge.on_chat_completion_terminal(&request, &denial).await;

        let events = channel.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].status, Some(400));
        assert!(events[0].capsule_id.is_none());
        assert!(events[0].nonce.is_none());
    }
}
