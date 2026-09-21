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
    /// The raw-proxy ingress routes this exchange to a peer on the mesh
    /// rather than serving it locally (`route_missing_local_model`'s
    /// remote-mesh branch). This node is the requester/router, not the
    /// server, for the exchange this envelope describes — a downstream
    /// plugin must not treat it as the served-side event.
    RemoteMesh,
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

/// Which side actually contributed a terminal event's `nonce` — the
/// authoritative signal for the same tri-state `capsule-emit-mesh`'s own
/// sidecar tracks as `client_nonce_source` (`client_supplied` /
/// `sidecar_generated_fallback`; the implicit third state is "no marker was
/// minted at all," carried by `nonce_source` itself being `None`). A
/// downstream plugin (M3) must use this field rather than sniffing the
/// `nonce`'s `fallback-` prefix, which stays only for human-readability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientNonceSource {
    ClientSupplied,
    SidecarGeneratedFallback,
}

/// What the host actually knows, at serve time, about *what ran, at what
/// fidelity, on whose hardware* for one exchange — the proof-of-inference
/// provenance a downstream capsule attests over (advances #1233's digest
/// advertisement). Every field is either a real value the host holds for the
/// served model/node, or omitted (serialized only `if Some`) when the host
/// genuinely does not know it for this exchange — never a fabricated string.
///
/// Sourced entirely from state the local [`mesh::Node`](crate::mesh::Node)
/// already holds for the served model and this host's hardware survey (see
/// the raw-proxy dispatch callsite in `network/openai/ingress.rs`): model
/// metadata comes from the served-model descriptor (`ServedModelMetadata`:
/// `quant`, `architecture`, `native_context_length`, `identity_hash`,
/// revision/repository), and hardware comes from the node's startup hardware
/// survey (`gpu_name`, `hostname`, `is_soc`, `advertised_memory`). No raw
/// prompt or response text is carried — provenance only.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ServingProvenance {
    /// The node that actually served the inference — this host's own mesh
    /// endpoint id.
    pub served_by_node_id: String,
    /// Serving host name, when the hardware survey resolved one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
    /// Model quantization format as the served-model descriptor reports it
    /// (e.g. `"Q4_K_M"`), from `ServedModelMetadata.quant`. Omitted when the
    /// descriptor carries no quant (unquantized weights, or metadata absent).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    /// Model architecture / family (e.g. `"llama"`), from
    /// `ServedModelMetadata.architecture`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    /// Native context length (n_ctx) the served weights advertise, from
    /// `ServedModelMetadata.native_context_length`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    /// Human-readable parameter size (e.g. `"7B"`), from
    /// `ServedModelMetadata.parameter_size`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameter_size: Option<String>,
    /// Transformer layer count, from `ServedModelMetadata.layer_count`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_count: Option<u32>,
    /// Content-addressed identity hash of the served model artifact, from
    /// `ServedModelIdentity.identity_hash` — a digest of the actual model
    /// identity (not a hash of the model *name* string). Omitted when the
    /// descriptor did not resolve one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_identity_hash: Option<String>,
    /// Canonical model reference (e.g. `repo@rev/file`), from
    /// `ServedModelIdentity.canonical_ref`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_canonical_ref: Option<String>,
    /// Source revision (git commit / tag) of the served model, from
    /// `ServedModelIdentity.revision`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_revision: Option<String>,
    /// From `ServedModelIdentity.weights_digest` — see that field's doc
    /// comment for what it is a digest over. Omitted exactly when the
    /// descriptor carries no digest; never a fabricated or zeroed value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights_digest: Option<String>,
    /// GPU display name from this host's startup hardware survey
    /// (`Node.gpu_name`). Omitted on CPU-only hosts or where no accelerator
    /// was enumerated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    /// Enumerated accelerator VRAM capacity in bytes this host advertised
    /// (`Node.advertised_memory.total_bytes`) — the sum of device VRAM, or
    /// the unified working set on SoCs. Omitted when nothing was enumerated
    /// (a bare CPU host advertising only via an explicit cap has no real
    /// enumerated figure to report).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_bytes: Option<u64>,
    /// Whether the serving host is a unified-memory SoC (Apple Silicon and
    /// similar), from the hardware survey (`Node.is_soc`) — the honest
    /// device signal this host has (it does not carry a separate cpu/cuda/
    /// metal enum on the served-model path).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_soc: Option<bool>,
}

/// The real token accounting the host observed for a served exchange, from
/// the dispatch outcome's
/// [`RespondedWithUsage`](crate::network::openai::transport::RouteDispatchOutcome::RespondedWithUsage)
/// (the served backend's own OpenAI-shaped `usage` object). Present on a
/// terminal envelope only when the served response actually carried usage;
/// omitted (never zeroed) when the dispatch produced no usage — so a
/// downstream plugin can seal the REAL token counts of a host-served
/// real-weights exchange rather than a stub's zeros. Every field is a real
/// count the host read off the wire; nothing is fabricated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ExchangeUsage {
    pub prompt_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_prompt_tokens: Option<u64>,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// How a terminal envelope's `capsule_id` was obtained.
///
/// `SelfMinted` is the capsule_id this node minted for its own served
/// response — the same value already written into the client's response as
/// `X-Capsule-Id`. `PeerAsserted` (via
/// [`OpenAiExchangeEnvelope::terminal_remote_mesh`]) is a value this node
/// merely OBSERVED on a peer's raw response header while routing (not
/// serving) the exchange. `X-Capsule-Id` is an unauthenticated,
/// relay-injectable header — a `PeerAsserted` value is never elevated to
/// verified here. It becomes verified only when a puller dereferences it via
/// a later out-of-band fetch and the fetched capsule's digest matches; that check lives in
/// the puller, not at this producer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapsuleIdProvenance {
    SelfMinted,
    PeerAsserted,
}

/// The wire shape both dispatch paths publish on [`OPENAI_EXCHANGE_CHANNEL`].
/// Deliberately independent of `openai_frontend`'s typed request/response —
/// the raw-proxy path never has one — so one shape covers both paths without
/// either being forced into the other's type.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpenAiExchangeEnvelope {
    /// Stable per-exchange id, minted when the dispatch path admits the
    /// request. Shared by an exchange's `EffectiveRequest` and `Terminal`
    /// envelopes (and mirrored into the transport `correlation_id`), so a
    /// plugin can pair the two events for one exchange even when concurrent
    /// requests on the same model are in flight.
    pub exchange_id: String,
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
    /// How `capsule_id` was obtained — see [`CapsuleIdProvenance`]. `None`
    /// exactly when `capsule_id` is `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capsule_id_provenance: Option<CapsuleIdProvenance>,
    /// The nonce the marker is correlated against, so a plugin observing
    /// this event knows what a later client ack must sign over.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nonce: Option<String>,
    /// Which side contributed `nonce` — see [`ClientNonceSource`]. `None`
    /// exactly when `nonce` is `None` (no marker minted).
    ///
    /// **Asymmetry across routing-node pairs:** when node A minted the
    /// fallback nonce (the client sent none), A reads its own
    /// `x-capsule-nonce-origin` header and reports
    /// `SidecarGeneratedFallback`. Node B strips that header deliberately
    /// (anti-smuggling, `request_parse.rs:582`) so it sees a
    /// well-formed nonce with no origin marker and reports `ClientSupplied`
    /// for the same nonce. Both are locally correct: A reports what it
    /// minted; B cannot trust the origin claim. A consumer joining both
    /// halves on the same nonce will observe two different `nonce_source`
    /// values — this is NOT a bug. Use the routing node's own envelope to
    /// judge whether the nonce was client-supplied or sidecar-generated;
    /// do not compare across nodes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nonce_source: Option<ClientNonceSource>,
    /// What ran, at what fidelity, on whose hardware — see [`ServingProvenance`].
    /// Present on a `Terminal` envelope only when the exchange was served
    /// locally (this node's own weights, not a plugin endpoint) AND the
    /// dispatch outcome was an actual 2xx response (`Responded`/
    /// `RespondedWithUsage`); `None` on effective-request envelopes, on the
    /// plugin-served path regardless of status (a plugin endpoint can proxy
    /// anywhere — none of this node's own hardware/weights identity is
    /// honest to attach to it), and on any non-2xx terminal envelope (a
    /// denial/error before dispatch, a 503, or a dropped/failed connection) —
    /// those served nothing, so there is nothing this field can honestly
    /// report.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub serving_provenance: Option<ServingProvenance>,
    /// The real token usage the served backend reported for this exchange (see
    /// [`ExchangeUsage`]). Present on a terminal envelope on either dispatch
    /// path whenever the dispatch outcome carried the backend's real token
    /// counts (`RespondedWithUsage`), regardless of status; `None` on
    /// effective-request envelopes and wherever the dispatch produced no usage
    /// (a plugin-served stub, a denial, or a non-usage-bearing backend).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ExchangeUsage>,
    /// The canonical JSON-DIGEST (`HEX(SHA-256(JCS(stringify_floats(body))))`
    /// — see [`request_body_digest`]) of the REAL request body this host
    /// actually dispatched. This is the one fact a downstream capsule needs
    /// to bind its `agent_input_digest` to the real bytes: the terminal event
    /// otherwise carries provenance and usage but nothing tying the sealed
    /// capsule to *what was asked*. Present on a terminal envelope on either
    /// dispatch path (host-served or plugin-served) whenever the host held a
    /// parsed JSON request body; `None` when it did not (never a fabricated
    /// digest). No raw prompt text is carried — only its digest.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_digest: Option<String>,
}

impl OpenAiExchangeEnvelope {
    pub fn effective(
        exchange_id: impl Into<String>,
        dispatch_path: OpenAiExchangeDispatchPath,
        model: impl Into<String>,
    ) -> Self {
        Self {
            exchange_id: exchange_id.into(),
            dispatch_path,
            phase: OpenAiExchangePhase::EffectiveRequest,
            model: model.into(),
            status: None,
            capsule_id: None,
            capsule_id_provenance: None,
            nonce: None,
            nonce_source: None,
            serving_provenance: None,
            usage: None,
            request_digest: None,
        }
    }

    pub fn terminal(
        exchange_id: impl Into<String>,
        dispatch_path: OpenAiExchangeDispatchPath,
        model: impl Into<String>,
        status: Option<u16>,
        marker: Option<CapsuleMarker>,
        nonce_source: Option<ClientNonceSource>,
    ) -> Self {
        Self {
            exchange_id: exchange_id.into(),
            dispatch_path,
            phase: OpenAiExchangePhase::Terminal,
            model: model.into(),
            status,
            capsule_id: marker.as_ref().map(|marker| marker.capsule_id.clone()),
            capsule_id_provenance: marker.as_ref().map(|_| CapsuleIdProvenance::SelfMinted),
            nonce: marker.as_ref().map(|marker| marker.nonce.clone()),
            nonce_source,
            serving_provenance: None,
            usage: None,
            request_digest: None,
        }
    }

    /// Attach the serving provenance the host resolved for this exchange. A
    /// small builder rather than a wider constructor so the two existing
    /// callsites that already pass six positional args aren't churned, and so
    /// the raw-proxy path can add provenance in one readable line after it has
    /// gathered it from the node.
    #[must_use]
    pub fn with_serving_provenance(mut self, provenance: ServingProvenance) -> Self {
        self.serving_provenance = Some(provenance);
        self
    }

    /// Attach the real token usage the served backend reported. Mirrors
    /// [`Self::with_serving_provenance`] — a small builder so the host-served
    /// raw-proxy path can add the REAL counts it read off the dispatch outcome
    /// in one readable line, without churning the positional `terminal`
    /// constructor. Only ever called with real usage; the field stays `None`
    /// when the dispatch produced none.
    #[must_use]
    pub fn with_usage(mut self, usage: ExchangeUsage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Attach the canonical JSON-DIGEST of the REAL request body this host
    /// dispatched, so a downstream capsule can bind its `agent_input_digest` to
    /// the real bytes. Mirrors the other builders — a small one-liner the
    /// raw-proxy host-served path calls after it has the request body in hand.
    /// Only ever called with a real digest computed by [`request_body_digest`];
    /// the field stays `None` when the host held no parsed body.
    #[must_use]
    pub fn with_request_digest(mut self, digest: String) -> Self {
        self.request_digest = Some(digest);
        self
    }

    /// Effective-request envelope for the `RemoteMesh` dispatch path,
    /// carrying the nonce this node is about to forward to the peer
    /// unchanged — so a plugin observing only the effective event already
    /// knows what a later client ack must sign over, rather than having to
    /// wait for the terminal event. `capsule_id` stays absent: this node
    /// mints nothing on this path.
    ///
    /// **`nonce_source` asymmetry:** when the routing node (node A) minted
    /// the fallback nonce, it reports `SidecarGeneratedFallback` here.
    /// The receiving peer (node B) strips the `x-capsule-nonce-origin`
    /// header (anti-smuggling) and therefore reports `ClientSupplied` for
    /// the same nonce on its own envelope. Both are locally correct; a
    /// consumer joining both envelopes will see two different `nonce_source`
    /// values for the same nonce — see the field-level doc on
    /// [`OpenAiExchangeEnvelope::nonce_source`] for the full explanation.
    pub fn effective_remote_mesh(
        exchange_id: impl Into<String>,
        model: impl Into<String>,
        nonce: Option<String>,
        nonce_source: Option<ClientNonceSource>,
    ) -> Self {
        Self {
            exchange_id: exchange_id.into(),
            dispatch_path: OpenAiExchangeDispatchPath::RemoteMesh,
            phase: OpenAiExchangePhase::EffectiveRequest,
            model: model.into(),
            status: None,
            capsule_id: None,
            capsule_id_provenance: None,
            nonce,
            nonce_source,
            serving_provenance: None,
            usage: None,
            request_digest: None,
        }
    }

    /// Terminal envelope for the `RemoteMesh` dispatch path — a routing node
    /// observing (not serving) an exchange it forwarded to a peer.
    ///
    /// Unlike [`Self::terminal`]'s `marker`, which bundles a capsule_id this
    /// node minted together with the nonce that capsule is correlated
    /// against, a routing node mints nothing here: `nonce` is the same
    /// client-contributed value forwarded to the peer unchanged (present
    /// only when the request already carries a stabilized nonce), and
    /// `peer_capsule_id` is the peer's own `X-Capsule-Id` response header,
    /// read back off the raw-proxy return. That value is always recorded as
    /// [`CapsuleIdProvenance::PeerAsserted`] — see its doc for why it is
    /// never elevated to verified here. `X-Capsule-Id` is an unauthenticated,
    /// relay-injectable header, so this producer does not attempt to
    /// distinguish a genuine peer value from an injected one; it can't, over
    /// an unauthenticated header. It never invents a value: `None` when the
    /// peer's response carried no such header.
    ///
    /// **`nonce_source` asymmetry:** same as [`Self::effective_remote_mesh`]
    /// — node A reports `SidecarGeneratedFallback` when it minted the nonce;
    /// node B strips the origin header (anti-smuggling) and reports
    /// `ClientSupplied` for the identical nonce. See
    /// [`OpenAiExchangeEnvelope::nonce_source`] for the full explanation.
    ///
    /// `serving_provenance`/`usage`/`request_digest` are never attached on
    /// this constructor — the raw-proxy host-served callsite that resolves
    /// those (`network/openai/ingress.rs::publish_raw_proxy_terminal`) is a
    /// different dispatch path (`RawProxy`); a routing node forwarding to a
    /// peer never resolves them for itself.
    pub fn terminal_remote_mesh(
        exchange_id: impl Into<String>,
        model: impl Into<String>,
        status: Option<u16>,
        nonce: Option<String>,
        nonce_source: Option<ClientNonceSource>,
        peer_capsule_id: Option<String>,
    ) -> Self {
        let capsule_id_provenance = peer_capsule_id
            .as_ref()
            .map(|_| CapsuleIdProvenance::PeerAsserted);
        Self {
            exchange_id: exchange_id.into(),
            dispatch_path: OpenAiExchangeDispatchPath::RemoteMesh,
            phase: OpenAiExchangePhase::Terminal,
            model: model.into(),
            status,
            capsule_id: peer_capsule_id,
            capsule_id_provenance,
            nonce,
            nonce_source,
            serving_provenance: None,
            usage: None,
            request_digest: None,
        }
    }
}

mod canonical_digest;
pub use canonical_digest::request_body_digest;

/// Publishes [`OpenAiExchangeEnvelope`]s to whatever is subscribed on
/// [`OPENAI_EXCHANGE_CHANNEL`] — an out-of-process plugin in production, a
/// recording double in tests. Fire-and-forget by design, mirroring
/// [`OpenAiHookPolicy`]'s own observer methods: exchange delivery to a
/// plugin must never affect whether the client's own request succeeds.
#[async_trait]
pub trait OpenAiExchangeChannel: Send + Sync + 'static {
    async fn publish(&self, event: &OpenAiExchangeEnvelope);

    /// Whether anything is actually listening on [`OPENAI_EXCHANGE_CHANNEL`]
    /// right now. Lets a caller skip the work that only exists to build an
    /// event (canonicalizing and hashing a request body, cloning a
    /// served-model descriptor) before finding out `publish` had nowhere to
    /// send it. Defaults to `true` — a test double with no subscriber
    /// concept (e.g. a recording channel) should behave as it always has.
    async fn has_subscriber(&self) -> bool {
        true
    }
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
            .broadcast_channel_message(
                OPENAI_EXCHANGE_CHANNEL,
                "application/json",
                body,
                &event.exchange_id,
            )
            .await
        {
            tracing::warn!(%error, "failed to publish openai exchange event to plugins");
        }
    }

    async fn has_subscriber(&self) -> bool {
        self.any_plugin_declares_mesh_channel(OPENAI_EXCHANGE_CHANNEL)
            .await
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
                route.exchange_id.clone(),
                OpenAiExchangeDispatchPath::TypedFrontend,
                route.model.clone(),
            ))
            .await;
    }

    async fn on_chat_completion_terminal(
        &self,
        request: &ChatCompletionRequest,
        exchange_id: &str,
        outcome: &ChatCompletionOutcome<'_>,
    ) {
        let (status, marker): (Option<u16>, Option<CapsuleMarker>) = match outcome {
            ChatCompletionOutcome::Success { response } => {
                (Some(200), response.capsule_marker.clone())
            }
            ChatCompletionOutcome::Error { status, .. } => (Some(*status), None),
            ChatCompletionOutcome::Denied { status, .. } => (Some(*status), None),
            // `ChatCompletionOutcome::Cancelled` and any future variant:
            // no HTTP response was produced, so there's nothing to report
            // beyond a status-free terminal event.
            _ => (None, None),
        };
        // Recomputed from `request` rather than threaded through
        // `CapsuleMarker` (an `openai-frontend` public type this crate
        // doesn't own): both this and `capsule_marker_for_response` below
        // read the same `client_nonce` field, so they always agree on which
        // branch was taken.
        let nonce_source = marker.as_ref().map(|_| client_nonce_source(request));
        self.channel
            .publish(&OpenAiExchangeEnvelope::terminal(
                exchange_id,
                OpenAiExchangeDispatchPath::TypedFrontend,
                request.model.clone(),
                status,
                marker,
                nonce_source,
            ))
            .await;
    }

    /// Reference nonce sourcing for the rung-ladder response leg: a
    /// client-contributed `client_nonce` (landing in `request.extra` via
    /// `ChatCompletionRequest`'s `#[serde(flatten)]` bag, the same mechanism
    /// `mesh_hooks` already uses) wins; absent that, mint a fallback rather
    /// than silently mislabeling it as client-supplied — mirroring
    /// `capsule-emit-mesh`'s own `client_nonce_source` tri-state
    /// (`client_supplied` / `sidecar_generated_fallback`). The `fallback-`
    /// prefix stays for readability, but [`ClientNonceSource`] (see
    /// `on_chat_completion_terminal`) is the authoritative signal — a plugin
    /// must not infer sourcing by sniffing this string.
    async fn capsule_marker_for_response(
        &self,
        request: &ChatCompletionRequest,
        response: &ChatCompletionResponse,
    ) -> Option<CapsuleMarker> {
        let nonce = match client_nonce_source(request) {
            ClientNonceSource::ClientSupplied => request
                .extra
                .get("client_nonce")
                .and_then(|value| value.as_str())
                .expect("client_nonce_source() confirmed a client_nonce string is present")
                .to_string(),
            ClientNonceSource::SidecarGeneratedFallback => format!("fallback-{}", response.id),
        };
        Some(CapsuleMarker {
            capsule_id: format!("capsule-{}", response.id),
            nonce,
        })
    }
}

/// The single place that decides client-supplied vs. sidecar-minted, used by
/// both [`OpenAiExchangeHookBridge::capsule_marker_for_response`] (to choose
/// the nonce value) and [`OpenAiExchangeHookBridge::on_chat_completion_terminal`]
/// (to label it on the envelope) so the two can never disagree.
fn client_nonce_source(request: &ChatCompletionRequest) -> ClientNonceSource {
    if request
        .extra
        .get("client_nonce")
        .and_then(|value| value.as_str())
        .is_some()
    {
        ClientNonceSource::ClientSupplied
    } else {
        ClientNonceSource::SidecarGeneratedFallback
    }
}

/// A publish sink that records every envelope it receives instead of
/// delivering it anywhere — shared by this module's own tests (path 1, the
/// typed frontend hook bridge) and `network::openai::ingress`'s tests (path
/// 2, the raw-proxy terminal builder), so both dispatch paths can assert on
/// exactly what a subscribing plugin would have seen without spinning one
/// up.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Mutex;

    use async_trait::async_trait;

    use super::{OpenAiExchangeChannel, OpenAiExchangeEnvelope};

    #[derive(Default)]
    pub(crate) struct RecordingChannel {
        events: Mutex<Vec<OpenAiExchangeEnvelope>>,
    }

    impl RecordingChannel {
        pub(crate) fn events(&self) -> Vec<OpenAiExchangeEnvelope> {
            self.events.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl OpenAiExchangeChannel for RecordingChannel {
        async fn publish(&self, event: &OpenAiExchangeEnvelope) {
            self.events.lock().unwrap().push(event.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use openai_frontend::{ChatCompletionOutcome, HookedOpenAiBackend, OpenAiBackend, Usage};

    use super::test_support::RecordingChannel;
    use super::*;

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

        let events = channel.events();
        assert_eq!(events.len(), 2, "one effective-request, one terminal");

        assert_eq!(
            events[0].dispatch_path,
            OpenAiExchangeDispatchPath::TypedFrontend
        );
        assert_eq!(events[0].phase, OpenAiExchangePhase::EffectiveRequest);
        assert_eq!(events[0].model, "gpt-mesh");

        assert_eq!(events[1].phase, OpenAiExchangePhase::Terminal);
        assert_eq!(events[1].status, Some(200));
        assert!(!events[0].exchange_id.is_empty());
        assert_eq!(events[0].exchange_id, events[1].exchange_id);
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

        let events = channel.events();
        assert_eq!(events[1].nonce.as_deref(), Some("abc123"));
        assert_eq!(
            events[1].nonce_source,
            Some(ClientNonceSource::ClientSupplied),
            "a plugin must be able to trust nonce_source over sniffing the nonce string"
        );
    }

    /// When the client contributes no nonce, the mint still labels it
    /// `sidecar_generated_fallback` via `nonce_source` — not just the
    /// human-readable `fallback-` prefix on the nonce string itself.
    #[tokio::test]
    async fn absent_client_nonce_is_labeled_sidecar_generated_fallback() {
        let channel = Arc::new(RecordingChannel::default());
        let bridge = Arc::new(OpenAiExchangeHookBridge::new(channel.clone()));
        let hooked = HookedOpenAiBackend::new(Arc::new(EchoBackend), bridge);

        hooked
            .chat_completion(chat_request("gpt-mesh"))
            .await
            .expect("backend call succeeds");

        let events = channel.events();
        assert!(
            events[1]
                .nonce
                .as_deref()
                .is_some_and(|n| n.starts_with("fallback-"))
        );
        assert_eq!(
            events[1].nonce_source,
            Some(ClientNonceSource::SidecarGeneratedFallback)
        );
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

        bridge
            .on_chat_completion_terminal(&request, "exchange-1", &denial)
            .await;

        let events = channel.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].exchange_id, "exchange-1");
        assert_eq!(events[0].status, Some(400));
        assert!(events[0].capsule_id.is_none());
        assert!(events[0].nonce.is_none());
        assert!(events[0].nonce_source.is_none());
    }

    /// The exact scenario `TerminalGuard` (in `openai-frontend`) exists to
    /// close: the backend future never returns, so `HookedOpenAiBackend`
    /// reports `ChatCompletionOutcome::Cancelled` instead of nothing — this
    /// bridge must still publish a terminal event for it, with no status,
    /// capsule id, nonce, or nonce_source to report.
    #[tokio::test]
    async fn cancelled_outcome_publishes_a_status_free_terminal_event() {
        let channel = Arc::new(RecordingChannel::default());
        let bridge = OpenAiExchangeHookBridge::new(channel.clone());
        let request = chat_request("gpt-mesh");

        bridge
            .on_chat_completion_terminal(&request, "exchange-1", &ChatCompletionOutcome::Cancelled)
            .await;

        let events = channel.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].phase, OpenAiExchangePhase::Terminal);
        assert!(events[0].status.is_none());
        assert!(events[0].capsule_id.is_none());
        assert!(events[0].nonce.is_none());
        assert!(events[0].nonce_source.is_none());
    }

    struct DelayedBackend {
        delay: std::time::Duration,
    }

    #[async_trait]
    impl OpenAiBackend for DelayedBackend {
        async fn models(&self) -> openai_frontend::OpenAiResult<Vec<openai_frontend::ModelObject>> {
            Ok(Vec::new())
        }

        async fn chat_completion(
            &self,
            request: ChatCompletionRequest,
        ) -> openai_frontend::OpenAiResult<ChatCompletionResponse> {
            tokio::time::sleep(self.delay).await;
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

    /// Two concurrent exchanges on the same model must not be pairable by
    /// mere arrival order — the terminal event for the exchange with the
    /// shorter backend delay lands before the effective event of neither
    /// exchange lines up with it positionally. Only matching `exchange_id`
    /// correctly recovers each exchange's own effective/terminal pair.
    #[tokio::test(start_paused = true)]
    async fn concurrent_exchanges_on_the_same_model_pair_by_exchange_id_not_by_arrival_order() {
        let channel = Arc::new(RecordingChannel::default());
        let bridge = Arc::new(OpenAiExchangeHookBridge::new(channel.clone()));
        let slow = HookedOpenAiBackend::new(
            Arc::new(DelayedBackend {
                delay: std::time::Duration::from_millis(50),
            }),
            bridge.clone(),
        );
        let fast = HookedOpenAiBackend::new(
            Arc::new(DelayedBackend {
                delay: std::time::Duration::from_millis(1),
            }),
            bridge,
        );

        let (slow_result, fast_result) = tokio::join!(
            slow.chat_completion(chat_request("gpt-mesh")),
            fast.chat_completion(chat_request("gpt-mesh")),
        );
        slow_result.expect("slow exchange succeeds");
        fast_result.expect("fast exchange succeeds");

        let events = channel.events();
        assert_eq!(events.len(), 4, "two effective + two terminal events");
        assert_eq!(events[0].phase, OpenAiExchangePhase::EffectiveRequest);
        assert_eq!(events[1].phase, OpenAiExchangePhase::EffectiveRequest);
        assert_eq!(events[2].phase, OpenAiExchangePhase::Terminal);
        assert_eq!(events[3].phase, OpenAiExchangePhase::Terminal);
        assert_ne!(
            events[0].exchange_id, events[1].exchange_id,
            "each exchange mints its own id"
        );

        // The fast exchange finishes first, so its terminal event (index 2)
        // is adjacent to the slow exchange's effective event (index 0) by
        // position — but it must still pair with the fast effective event
        // (index 1) by id, and the slow terminal (index 3) with the slow
        // effective (index 0).
        assert_eq!(
            events[2].exchange_id, events[1].exchange_id,
            "fast exchange's terminal event pairs with its own effective event"
        );
        assert_eq!(
            events[3].exchange_id, events[0].exchange_id,
            "slow exchange's terminal event pairs with its own effective event"
        );
    }

    /// A terminal envelope with serving provenance serializes the known fields
    /// and OMITS the unknown ones (never a fabricated `null` or empty string) —
    /// this is the honesty contract a downstream capsule relies on: a field
    /// that is present is a real host fact, a field that is absent is genuinely
    /// unknown, not zeroed.
    #[test]
    fn terminal_carries_serving_provenance_and_omits_unknown_fields() {
        let envelope = OpenAiExchangeEnvelope::terminal(
            "exch-1",
            OpenAiExchangeDispatchPath::RawProxy,
            "hermes-2-pro-mistral-7b",
            Some(200),
            None,
            None,
        )
        .with_serving_provenance(ServingProvenance {
            served_by_node_id: "node-abc".to_string(),
            hostname: Some("host-1".to_string()),
            quantization: Some("Q4_K_M".to_string()),
            architecture: Some("llama".to_string()),
            context_length: Some(8192),
            parameter_size: Some("7B".to_string()),
            layer_count: Some(32),
            model_identity_hash: Some("abc123".to_string()),
            model_canonical_ref: None,
            model_revision: None,
            weights_digest: None,
            gpu: None,
            vram_bytes: None,
            is_soc: Some(true),
        });

        let value = serde_json::to_value(&envelope).expect("serialize");
        let prov = &value["serving_provenance"];
        assert_eq!(prov["served_by_node_id"], "node-abc");
        assert_eq!(prov["quantization"], "Q4_K_M");
        assert_eq!(prov["architecture"], "llama");
        assert_eq!(prov["context_length"], 8192);
        assert_eq!(prov["layer_count"], 32);
        assert_eq!(prov["is_soc"], true);
        // Unknown facts are ABSENT (omitted), not fabricated as null/empty.
        assert!(prov.get("model_canonical_ref").is_none());
        assert!(prov.get("model_revision").is_none());
        assert!(prov.get("weights_digest").is_none());
        assert!(prov.get("gpu").is_none());
        assert!(prov.get("vram_bytes").is_none());
    }

    /// A terminal envelope whose serving provenance resolved a real load-time
    /// weights digest carries it on the wire — the field this consumer exists
    /// to thread onto the exchange (see `mesh::weights_digest_for_file`'s
    /// module doc for what the digest is over).
    #[test]
    fn terminal_carries_weights_digest_when_present() {
        let envelope = OpenAiExchangeEnvelope::terminal(
            "exch-1",
            OpenAiExchangeDispatchPath::RawProxy,
            "hermes-2-pro-mistral-7b",
            Some(200),
            None,
            None,
        )
        .with_serving_provenance(ServingProvenance {
            served_by_node_id: "node-abc".to_string(),
            hostname: None,
            quantization: None,
            architecture: None,
            context_length: None,
            parameter_size: None,
            layer_count: None,
            model_identity_hash: None,
            model_canonical_ref: None,
            model_revision: None,
            weights_digest: Some(
                "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                    .to_string(),
            ),
            gpu: None,
            vram_bytes: None,
            is_soc: None,
        });

        let value = serde_json::to_value(&envelope).expect("serialize");
        assert_eq!(
            value["serving_provenance"]["weights_digest"],
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// An effective-request envelope carries NO serving provenance (the field
    /// is omitted entirely), so the block is a terminal-only, served-exchange
    /// fact — never claimed before the exchange actually ran.
    #[test]
    fn effective_envelope_has_no_serving_provenance() {
        let envelope =
            OpenAiExchangeEnvelope::effective("exch-1", OpenAiExchangeDispatchPath::RawProxy, "m");
        assert!(envelope.serving_provenance.is_none());
        let value = serde_json::to_value(&envelope).expect("serialize");
        assert!(value.get("serving_provenance").is_none());
    }

    /// The real token usage the host-served path reads off its dispatch outcome
    /// rides the terminal envelope, so a downstream plugin can seal the REAL
    /// counts of a host-served real-weights exchange instead of a stub's zeros.
    #[test]
    fn terminal_carries_real_usage_when_attached() {
        let envelope = OpenAiExchangeEnvelope::terminal(
            "exch-usage",
            OpenAiExchangeDispatchPath::RawProxy,
            "llama-3.2-3b-instruct",
            Some(200),
            None,
            None,
        )
        .with_usage(ExchangeUsage {
            prompt_tokens: 42,
            cached_prompt_tokens: Some(10),
            completion_tokens: 6,
            total_tokens: 48,
        });

        let value = serde_json::to_value(&envelope).expect("serialize");
        assert_eq!(value["usage"]["prompt_tokens"], 42);
        assert_eq!(value["usage"]["cached_prompt_tokens"], 10);
        assert_eq!(value["usage"]["completion_tokens"], 6);
        assert_eq!(value["usage"]["total_tokens"], 48);
    }

    /// A terminal envelope with no usage attached OMITS the `usage` key entirely
    /// (never a fabricated all-zero object) — the same honesty contract the
    /// serving-provenance fields hold: absent means genuinely unknown.
    #[test]
    fn terminal_omits_usage_when_none_attached() {
        let envelope = OpenAiExchangeEnvelope::terminal(
            "exch-no-usage",
            OpenAiExchangeDispatchPath::RawProxy,
            "some-plugin-model",
            Some(200),
            None,
            None,
        );
        let value = serde_json::to_value(&envelope).expect("serialize");
        assert!(value.get("usage").is_none());
    }

    /// A terminal envelope carrying a real request digest serializes it, and it
    /// survives a round-trip — the one fact a downstream capsule binds its
    /// `agent_input_digest` to.
    #[test]
    fn terminal_carries_request_digest_when_attached() {
        let envelope = OpenAiExchangeEnvelope::terminal(
            "exch-rd",
            OpenAiExchangeDispatchPath::RawProxy,
            "llama-3.2-3b-instruct",
            Some(200),
            None,
            None,
        )
        .with_request_digest("deadbeef".to_string());
        let value = serde_json::to_value(&envelope).expect("serialize");
        assert_eq!(value["request_digest"], "deadbeef");
    }

    /// No request digest attached -> the key is omitted entirely (never a
    /// fabricated empty digest), same honesty contract as usage/provenance.
    #[test]
    fn terminal_omits_request_digest_when_none_attached() {
        let envelope = OpenAiExchangeEnvelope::terminal(
            "exch-no-rd",
            OpenAiExchangeDispatchPath::RawProxy,
            "m",
            Some(200),
            None,
            None,
        );
        let value = serde_json::to_value(&envelope).expect("serialize");
        assert!(value.get("request_digest").is_none());
    }

    // --- #1668 review round: RemoteMesh / RawProxy envelope shapes ---
    //
    // Shape tests only — these call envelope constructors directly and assert
    // on the fields they set. They do NOT invoke `route_missing_local_model`
    // or `try_route_plugin_model`, so they would still pass if the publish
    // calls inside those routing functions were deleted. Real end-to-end
    // publish coverage (including both envelopes being emitted and their
    // nonce_source values) lives in `ingress_tests::tests`.

    /// Verifies the envelope constructor shape for the RemoteMesh effective +
    /// terminal pair: both envelopes carry `RemoteMesh` dispatch path, the
    /// nonce and nonce_source are threaded onto both, and `capsule_id` is
    /// absent on both (this node mints nothing on the remote-mesh path).
    ///
    /// // Shape test only — does not invoke the routing function.
    /// // Real publish coverage is in ingress_tests::tests.
    #[tokio::test]
    async fn envelope_shape_remote_mesh_effective_and_terminal_carry_nonce_fields() {
        let channel = RecordingChannel::default();
        let nonce = Some("6d7d8d2e-3f4a-4b5c-8d9e-0a1b2c3d4e5f".to_string());
        let nonce_source = Some(ClientNonceSource::ClientSupplied);

        channel
            .publish(&OpenAiExchangeEnvelope::effective_remote_mesh(
                "exch-rm-1",
                "hermes-2-pro-mistral-7b",
                nonce.clone(),
                nonce_source,
            ))
            .await;
        channel
            .publish(&OpenAiExchangeEnvelope::terminal_remote_mesh(
                "exch-rm-1",
                "hermes-2-pro-mistral-7b",
                Some(200),
                nonce.clone(),
                nonce_source,
                None,
            ))
            .await;

        let events = channel.events();
        assert_eq!(events.len(), 2, "one effective-request, one terminal");

        assert_eq!(
            events[0].dispatch_path,
            OpenAiExchangeDispatchPath::RemoteMesh
        );
        assert_eq!(events[0].phase, OpenAiExchangePhase::EffectiveRequest);
        assert_eq!(events[0].nonce, nonce);
        assert_eq!(events[0].nonce_source, nonce_source);
        assert!(events[0].capsule_id.is_none());

        assert_eq!(
            events[1].dispatch_path,
            OpenAiExchangeDispatchPath::RemoteMesh
        );
        assert_eq!(events[1].phase, OpenAiExchangePhase::Terminal);
        assert_eq!(events[1].exchange_id, events[0].exchange_id);
        assert_eq!(events[1].status, Some(200));
        assert_eq!(events[1].nonce, nonce);
        assert_eq!(events[1].nonce_source, nonce_source);
        assert!(events[1].capsule_id.is_none());
        assert!(events[1].capsule_id_provenance.is_none());
    }

    /// A `RemoteMesh` terminal envelope
    /// carries the peer's `X-Capsule-Id` as `capsule_id`, always labeled
    /// `PeerAsserted` — never `SelfMinted` (this node minted nothing), and
    /// never silently upgraded to verified.
    #[tokio::test]
    async fn terminal_remote_mesh_labels_a_peer_capsule_id_as_peer_asserted() {
        let envelope = OpenAiExchangeEnvelope::terminal_remote_mesh(
            "exch-rm-cap",
            "hermes-2-pro-mistral-7b",
            Some(200),
            Some("6d7d8d2e-3f4a-4b5c-8d9e-0a1b2c3d4e5f".to_string()),
            Some(ClientNonceSource::ClientSupplied),
            Some("capsule-peer-1".to_string()),
        );
        assert_eq!(
            envelope.dispatch_path,
            OpenAiExchangeDispatchPath::RemoteMesh
        );
        assert_eq!(envelope.capsule_id.as_deref(), Some("capsule-peer-1"));
        assert_eq!(
            envelope.capsule_id_provenance,
            Some(CapsuleIdProvenance::PeerAsserted)
        );
        let value = serde_json::to_value(&envelope).expect("serialize");
        assert_eq!(value["capsule_id_provenance"], "peer_asserted");
    }

    /// Mutant: the peer's response carried no `X-Capsule-Id` header —
    /// `capsule_id` and its provenance both stay honestly absent, never
    /// invented.
    #[tokio::test]
    async fn terminal_remote_mesh_without_a_peer_capsule_id_omits_both_fields() {
        let envelope = OpenAiExchangeEnvelope::terminal_remote_mesh(
            "exch-rm-cap-2",
            "hermes-2-pro-mistral-7b",
            Some(200),
            None,
            None,
            None,
        );
        assert!(envelope.capsule_id.is_none());
        assert!(envelope.capsule_id_provenance.is_none());
        let value = serde_json::to_value(&envelope).expect("serialize");
        assert!(value.get("capsule_id").is_none());
        assert!(value.get("capsule_id_provenance").is_none());
    }

    /// Mutant: a substitute/injected `X-Capsule-Id` is still recorded — this
    /// producer does not attempt to distinguish a genuine peer value from an
    /// injected one (it cannot, over an unauthenticated header) — but it is
    /// NEVER sealed as anything other than `PeerAsserted`. Would-be
    /// verification is out of scope here by design (see the puller).
    #[tokio::test]
    async fn terminal_remote_mesh_records_an_injected_value_as_peer_asserted_never_verified() {
        let envelope = OpenAiExchangeEnvelope::terminal_remote_mesh(
            "exch-rm-cap-3",
            "hermes-2-pro-mistral-7b",
            Some(200),
            None,
            None,
            Some("injected-not-really-the-peers".to_string()),
        );
        assert_eq!(
            envelope.capsule_id.as_deref(),
            Some("injected-not-really-the-peers")
        );
        assert_eq!(
            envelope.capsule_id_provenance,
            Some(CapsuleIdProvenance::PeerAsserted)
        );
    }

    /// A `RemoteMesh` terminal envelope mints no marker of its own: a
    /// self-minted capsule_id (`SelfMinted`) can never appear on this
    /// dispatch path.
    #[tokio::test]
    async fn terminal_remote_mesh_never_produces_self_minted_provenance() {
        let envelope = OpenAiExchangeEnvelope::terminal_remote_mesh(
            "exch-rm-cap-4",
            "m",
            Some(200),
            Some("nonce-1".to_string()),
            Some(ClientNonceSource::ClientSupplied),
            Some("capsule-peer-2".to_string()),
        );
        assert_ne!(
            envelope.capsule_id_provenance,
            Some(CapsuleIdProvenance::SelfMinted)
        );
    }

    /// Verifies the envelope constructor shape for the RawProxy effective +
    /// terminal pair: both envelopes carry `RawProxy` dispatch path, and
    /// nonce/nonce_source/capsule_id are all absent (the raw-proxy path never
    /// runs through `openai-frontend`'s `OpenAiHookPolicy`, so no marker is
    /// minted).
    ///
    /// // Shape test only — does not invoke the routing function.
    /// // Real publish coverage is in ingress_tests::tests.
    #[tokio::test]
    async fn envelope_shape_raw_proxy_effective_and_terminal_have_no_marker() {
        let channel = RecordingChannel::default();

        channel
            .publish(&OpenAiExchangeEnvelope::effective(
                "exch-rp-1",
                OpenAiExchangeDispatchPath::RawProxy,
                "acme/plugin-model",
            ))
            .await;
        channel
            .publish(&OpenAiExchangeEnvelope::terminal(
                "exch-rp-1",
                OpenAiExchangeDispatchPath::RawProxy,
                "acme/plugin-model",
                Some(200),
                None,
                None,
            ))
            .await;

        let events = channel.events();
        assert_eq!(events.len(), 2, "one effective-request, one terminal");

        assert_eq!(
            events[0].dispatch_path,
            OpenAiExchangeDispatchPath::RawProxy
        );
        assert_eq!(events[0].phase, OpenAiExchangePhase::EffectiveRequest);
        assert!(events[0].nonce.is_none());
        assert!(events[0].capsule_id.is_none());

        assert_eq!(
            events[1].dispatch_path,
            OpenAiExchangeDispatchPath::RawProxy
        );
        assert_eq!(events[1].phase, OpenAiExchangePhase::Terminal);
        assert_eq!(events[1].exchange_id, events[0].exchange_id);
        assert_eq!(events[1].status, Some(200));
        assert!(events[1].nonce.is_none());
        assert!(events[1].nonce_source.is_none());
        assert!(events[1].capsule_id.is_none());
    }
}
