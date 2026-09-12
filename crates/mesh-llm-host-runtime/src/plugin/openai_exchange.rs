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
    /// endpoint id. On a plugin-served (raw-proxy) exchange this is the node
    /// whose plugin endpoint produced the response.
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
    pub completion_tokens: u64,
    pub total_tokens: u64,
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
    /// The nonce the marker is correlated against, so a plugin observing
    /// this event knows what a later client ack must sign over.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nonce: Option<String>,
    /// Which side contributed `nonce` — see [`ClientNonceSource`]. `None`
    /// exactly when `nonce` is `None` (no marker minted).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nonce_source: Option<ClientNonceSource>,
    /// What ran, at what fidelity, on whose hardware — see [`ServingProvenance`].
    /// Present on a `Terminal` envelope for a served exchange; `None` on
    /// effective-request envelopes and on terminal envelopes where nothing was
    /// served (a denial/error before dispatch).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub serving_provenance: Option<ServingProvenance>,
    /// The real token usage the served backend reported for this exchange (see
    /// [`ExchangeUsage`]). Present on a terminal envelope for a host-served
    /// exchange whose response carried a `usage` object; `None` on
    /// effective-request envelopes and wherever the dispatch produced no usage
    /// (a plugin-served stub, a denial, or a non-usage-bearing backend).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ExchangeUsage>,
    /// The canonical JSON-DIGEST (`HEX(SHA-256(JCS(stringify_floats(body))))`
    /// — see [`request_body_digest`]) of the REAL request body this host
    /// actually dispatched. This is the one fact a downstream capsule needs
    /// to bind its `agent_input_digest` to the real bytes: the terminal event
    /// otherwise carries provenance and usage but nothing tying the sealed
    /// capsule to *what was asked*. Present on a host-served terminal
    /// envelope whose request carried a JSON body; `None` when the host held
    /// no parsed body to digest (never a fabricated digest). No raw prompt
    /// text is carried — only its digest.
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
}

/// The canonical JSON-DIGEST of a request body: `HEX(SHA-256(JCS(v)))` over
/// the float-stringified body. Byte-for-byte identical to the current
/// `agent_action_capsule.canonical.json_digest` reference and the Python
/// `capsule_sidecar.digest_json` sidecar wrapper it backs
/// (`digest_json(v) = json_digest(_stringify_floats(v))`) — cross-verified
/// against a live run of that reference, not just read off its source.
///
/// Deliberately does **not** apply the profile's absent-field `normalize`
/// step: `agent_action_capsule.canonical` reserves normalization for the
/// vintage format-2 Capsule-ID path only — its current (non-vintage)
/// `json_digest` is plain `JCS(v)`, no normalize — so an OpenAI request body
/// that explicitly carries `null` for an unset optional field (common OpenAI
/// client behavior, e.g. `"stop": null`) must still digest to a DIFFERENT
/// value than the same body with that field omitted; normalizing here would
/// silently collapse the two. (Note for the record: `capsule-emit-mesh`'s own
/// Rust `canonical_body_digest` currently still routes through
/// `capsule_producer::jcs::json_digest`, which normalizes — a drift from the
/// current Python reference that this host digest does not replicate. Flagged
/// separately; out of scope for this crate to fix.)
///
/// It is `HEX(SHA-256(JCS(stringify_floats(body))))`:
///  1. `stringify_floats` — every JSON float becomes its exact decimal string
///     (JCS refuses floats in a digest-bearing value; OpenAI chat bodies are
///     full of them: temperature, top_p, penalties);
///  2. JCS — RFC 8785 canonical serialization (sorted keys, minimal form);
///  3. SHA-256, lowercase hex.
///
/// A self-contained port kept in this crate (the host cannot depend on the
/// plugin's `capsule-producer`), verified against the Python reference on the
/// frozen fixture in the tests below.
pub fn request_body_digest(body: &serde_json::Value) -> String {
    use sha2::{Digest, Sha256};
    let canonical = jcs_bytes(&stringify_floats(body));
    hex::encode(Sha256::digest(&canonical))
}

/// Replace every JSON float with its exact decimal-string form (mirrors the
/// Python reference's `_stringify_floats`).
fn stringify_floats(value: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match value {
        Value::Number(n) if n.is_f64() && !(n.is_i64() || n.is_u64()) => {
            let f = n.as_f64().expect("n.is_f64() confirmed a f64 is present");
            let s = format!("{f}");
            let s = if s.contains('.') || s.contains('e') || s.contains('E') {
                s
            } else {
                format!("{s}.0")
            };
            Value::String(s)
        }
        Value::Number(_) => value.clone(),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), stringify_floats(v)))
                .collect(),
        ),
        Value::Array(arr) => Value::Array(arr.iter().map(stringify_floats).collect()),
        other => other.clone(),
    }
}

/// RFC 8785 JCS serialization (mirror of `agent_action_capsule.canonical.jcs`).
/// Floats are already stringified before this runs, so a bare float here is a
/// programmer error, serialized via serde's default rather than panicking.
fn jcs_bytes(v: &serde_json::Value) -> Vec<u8> {
    let mut out = String::new();
    jcs_value(v, &mut out);
    out.into_bytes()
}

fn jcs_value(v: &serde_json::Value, out: &mut String) {
    use serde_json::Value;
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::String(s) => jcs_string(s, out),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::Array(arr) => {
            out.push('[');
            for (i, x) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                jcs_value(x, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            // RFC 8785 §3.2.3: object members sorted by UTF-16 code-unit sequence.
            let mut items: Vec<(&String, &Value)> = map.iter().collect();
            items.sort_by(|(a, _), (b, _)| {
                let au: Vec<u16> = a.encode_utf16().collect();
                let bu: Vec<u16> = b.encode_utf16().collect();
                au.cmp(&bu).then_with(|| a.cmp(b))
            });
            out.push('{');
            for (i, (k, val)) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                jcs_string(k, out);
                out.push(':');
                jcs_value(val, out);
            }
            out.push('}');
        }
    }
}

fn jcs_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        let o = ch as u32;
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ if o == 0x08 => out.push_str("\\b"),
            _ if o == 0x09 => out.push_str("\\t"),
            _ if o == 0x0A => out.push_str("\\n"),
            _ if o == 0x0C => out.push_str("\\f"),
            _ if o == 0x0D => out.push_str("\\r"),
            _ if o < 0x20 => out.push_str(&format!("\\u{o:04x}")),
            _ => out.push(ch),
        }
    }
    out.push('"');
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

        let events = channel.events.lock().unwrap();
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

        let events = channel.events.lock().unwrap();
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

        let events = channel.events.lock().unwrap();
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

        let events = channel.events.lock().unwrap();
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

        let events = channel.events.lock().unwrap();
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
        assert!(prov.get("gpu").is_none());
        assert!(prov.get("vram_bytes").is_none());
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
            completion_tokens: 6,
            total_tokens: 48,
        });

        let value = serde_json::to_value(&envelope).expect("serialize");
        assert_eq!(value["usage"]["prompt_tokens"], 42);
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

    /// The host's `request_body_digest` is byte-for-byte the value a live run
    /// of the Python reference, `capsule_sidecar.digest_json`, produces over
    /// the identical JSON value:
    ///
    ///   python3 -c "
    ///   from capsule_sidecar import digest_json
    ///   print(digest_json({
    ///       'model': 'hermes-2-pro-mistral-7b',
    ///       'messages': [{'role': 'user', 'content': 'hello'}],
    ///       'temperature': 0.7,
    ///       'top_p': 1.0,
    ///       'max_tokens': 512,
    ///   }))"
    ///
    /// `top_p: 1.0` exercises the whole-number-float edge case
    /// (`stringify_floats` must emit "1.0", not "1"). This value has no
    /// null/absent fields, so it does not by itself distinguish a
    /// normalizing digest from a non-normalizing one — see the next test for
    /// that.
    #[test]
    fn request_body_digest_matches_python_reference() {
        let body = serde_json::json!({
            "model": "hermes-2-pro-mistral-7b",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 512
        });
        let expected = "a6329c5ebb66562f38a8136a8d8511b6aeed166e4c7d889b9133ac96fc49a9d5";
        assert_eq!(request_body_digest(&body), expected);
    }

    /// The same body, plus two explicit-`null` optional fields (as real
    /// OpenAI clients routinely send, e.g. `"stop": null`), must digest to a
    /// DIFFERENT value than the null-free body above — proving this digest
    /// does NOT apply the profile's absent-field `normalize` step. Expected
    /// value from the same live Python reference invocation with the two
    /// extra `None` fields added to the dict. Pins the current
    /// `agent_action_capsule.canonical.json_digest` contract (normalize is
    /// vintage-format-2-only) against a future accidental reintroduction of
    /// normalization here.
    #[test]
    fn request_body_digest_does_not_normalize_absent_fields() {
        let body = serde_json::json!({
            "model": "hermes-2-pro-mistral-7b",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 512,
            "stop": null,
            "user": null
        });
        let expected = "ee8aeb450ccf8c8017caae0d3733d3dcd62ec88752053894118d28cea0d176fe";
        assert_eq!(request_body_digest(&body), expected);
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
}
