//! Consumes the #1331 lifecycle-hook terminal-event broadcast on the
//! `openai.exchange.v1` mesh channel (`mesh-llm-host-runtime`'s
//! `plugin::openai_exchange` module). The host originally wired only the
//! *plugin-dispatch* raw-proxy path into production
//! (`network/openai/ingress.rs`'s `try_route_plugin_model`, the path this
//! plugin's own `inference::provider()` registration is routed through), so
//! early testing only proved the plugin observed exchanges it was itself the
//! HTTP backend for.
//!
//! **Routing-companion finding (2026-09-06): this is broader than that.** The
//! host-served (real-weights) local-routing branch in `ingress.rs` — the one
//! that serves a node's own loaded GGUF model directly, used whether the
//! caller is this node's sidecar or a mesh-routed peer hitting the node's
//! advertised API port directly — now publishes the same effective/terminal
//! pair on this channel (`feat/serving-provenance-host-served-terminal`;
//! previously that branch published nothing). Proven empirically in
//! `REAL-HOST-VERIFICATION.md`'s real-GGUF test: the captured exchange's
//! model (`local-gguf/sha256-...`) is not a plugin-advertised name, yet the
//! plugin's own lifecycle-events log recorded it. So a mesh-routed peer that
//! bypasses this node's sidecar entirely is still sealed by this plugin
//! (real hardware citation) via this channel — the routing gap that remains
//! is sidecar-side only: that traffic has no sidecar I/O capsule to join
//! against, since the sidecar's reverse proxy never saw it. This module is
//! independent proof, either way, that the plugin observed the same exchange
//! the host itself terminal-logged, not just its own view of the call.
//!
//! `OpenAiExchangeEnvelope` here is a hand-mirrored copy of the host-side
//! struct, not a shared dependency -- the host lives in a different repo
//! (`mesh-llm`) that this plugin cannot depend on. Field shape verified
//! against `crates/mesh-llm-host-runtime/src/plugin/openai_exchange.rs` on
//! that branch (`dispatch_path`/`phase`/`model`/`status`/`capsule_id`/`nonce`/
//! `serving_provenance`/`twin_bracket_id`).
//!
//! The `serving_provenance` block is the host's proof-of-inference metadata
//! (#1233 digest advertisement): what ran (model identity hash, revision), at
//! what fidelity (quantization, architecture, context length), on whose
//! hardware (gpu, vram, soc). Every field is `Option` and `#[serde(default)]`
//! so this mirror stays forward/backward compatible with a host that predates
//! the block (it simply arrives absent) -- and, critically, so a fact the host
//! genuinely does not know stays `None` here rather than being fabricated.

use serde::Deserialize;
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const OPENAI_EXCHANGE_CHANNEL: &str = "openai.exchange.v1";

#[derive(Debug, Clone, Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DispatchPath {
    TypedFrontend,
    RawProxy,
    /// The raw-proxy ingress routed this exchange to a peer on the mesh
    /// rather than serving it locally (`route_missing_local_model`'s
    /// remote-mesh branch, `mesh-llm-host-runtime`'s
    /// `OpenAiExchangeDispatchPath::RemoteMesh`). This node is the
    /// requester/router for the exchange, never the server -- the
    /// authoritative signal `is_sealable_requester_side`-style role
    /// derivation keys on (the 2026-09-06 role ruling). Present on the
    /// fork's enrichment-aware host today; ships on bare upstream main the
    /// day mesh-llm#1668 merges, so keying role on this field (not on the
    /// fork-only `served_by_node_id` enrichment) labels correctly either way.
    RemoteMesh,
    /// Catch-all for any dispatch_path value this mirror predates. `#[serde(other)]`
    /// is required here -- `#[serde(default)]` on the *field* does not apply to
    /// an unrecognized *variant*; without this, an old plugin build fails to
    /// parse the WHOLE envelope the moment a newer host emits a value this
    /// enum doesn't know, silently dropping every field, not just this one.
    /// Same "unknown != trusted" discipline as the `tpm_measured` registry
    /// values: an unrecognized dispatch path is observed, never guessed at --
    /// and, per the 2026-09-06 role ruling, never silently defaulted to
    /// `served` either (see `role_for_dispatch_path`).
    #[serde(other)]
    Unknown,
}

/// The top-level `x-mesh-poc-v1.role` value derived from an observed
/// exchange's `dispatch_path` -- the AUTHORITATIVE signal (the 2026-09-06
/// role ruling): `RemoteMesh` means this node routed the exchange to a
/// peer, so it is the REQUESTER, never the server, no matter what a heuristic
/// might otherwise guess. `Unknown` is labeled `"unknown"`, never silently
/// defaulted to `"served"` -- a missing/unrecognized field must never become
/// a claim. The `served_by_node_id`-vs-self CONSISTENCY CHECK (fork-only
/// enrichment, applied where that field is present) can still upgrade this to
/// `"conflict"` -- see `capsule_emit::role_and_observation_point`.
pub fn role_for_dispatch_path(dispatch_path: &DispatchPath) -> &'static str {
    match dispatch_path {
        DispatchPath::RemoteMesh => "requested",
        DispatchPath::TypedFrontend | DispatchPath::RawProxy => "served",
        DispatchPath::Unknown => "unknown",
    }
}

/// The raw wire value a `DispatchPath` was (or would be) serialized as --
/// e.g. for `ServingProvenance::dispatch_path`, so a sealed record carries
/// the actual observed signal `role_for_dispatch_path` derived from, not just
/// the resolved label. Matches the `#[serde(rename_all = "snake_case")]` wire
/// form exactly (kept as a hand-written match, not a serde round-trip,
/// because `Unknown` has no single canonical wire string -- it is a
/// catch-all for whatever unrecognized value the host actually sent, which
/// this mirror never retains).
pub fn dispatch_path_wire_value(dispatch_path: &DispatchPath) -> &'static str {
    match dispatch_path {
        DispatchPath::TypedFrontend => "typed_frontend",
        DispatchPath::RawProxy => "raw_proxy",
        DispatchPath::RemoteMesh => "remote_mesh",
        DispatchPath::Unknown => "unknown",
    }
}

/// Mirror of the host's `CapsuleIdProvenance`
/// (`mesh-llm-host-runtime::plugin::openai_exchange::CapsuleIdProvenance`).
/// `PeerAsserted` (the only variant a `RemoteMesh` terminal envelope ever
/// carries) means this node merely OBSERVED the value on a peer's raw
/// response header while ROUTING the exchange -- an unauthenticated,
/// relay-injectable claim, never elevated to verified here. `#[serde(other)]`
/// for the same forward-compat reason as `DispatchPath::Unknown` above: an
/// unrecognized provenance value is observed, never guessed at, and this
/// mirror must not fail to parse the whole envelope over a value it predates.
#[derive(Debug, Clone, Copy, Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CapsuleIdProvenance {
    SelfMinted,
    PeerAsserted,
    #[serde(other)]
    Unknown,
}

/// The raw wire value a `CapsuleIdProvenance` was serialized as -- same
/// hand-written-match discipline as `dispatch_path_wire_value` (`Unknown` has
/// no single canonical host-side wire string, so a round-trip through serde
/// would be lossy about which unrecognized value this mirror actually saw).
pub fn capsule_id_provenance_wire_value(provenance: &CapsuleIdProvenance) -> &'static str {
    match provenance {
        CapsuleIdProvenance::SelfMinted => "self_minted",
        CapsuleIdProvenance::PeerAsserted => "peer_asserted",
        CapsuleIdProvenance::Unknown => "unknown",
    }
}

/// THE MISLABELING GUARD: `capsule_id`/`capsule_id_provenance` also carries a
/// `SelfMinted` value on a locally-served envelope (this node's OWN
/// `X-Capsule-Id` marker -- nothing to do with a peer). Only a `PeerAsserted`
/// value names a peer's half of the exchange; every other case (`SelfMinted`,
/// `Unknown`, or no value at all) must seal `peer_capsule_id: None`, never
/// surface this node's own marker (or an unrecognized value) as though a peer
/// had asserted it. `seal_observed_host_exchange` (`main.rs`) calls this
/// directly rather than re-deriving the match inline, so the guard is unit-
/// testable independent of the axum/CapsuleState wiring around it.
pub fn peer_capsule_id_for_seal(envelope: &OpenAiExchangeEnvelope) -> Option<(&str, &'static str)> {
    match envelope.capsule_id_provenance {
        Some(CapsuleIdProvenance::PeerAsserted) => envelope.capsule_id.as_deref().map(|id| {
            (
                id,
                capsule_id_provenance_wire_value(&CapsuleIdProvenance::PeerAsserted),
            )
        }),
        _ => None,
    }
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    EffectiveRequest,
    Terminal,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiExchangeEnvelope {
    /// Stable per-exchange id the host mints for this raw-proxy exchange
    /// (`OpenAiExchangeEnvelope::exchange_id` host-side). Carried through so a
    /// host-served capsule can attest the host's own correlation id rather than
    /// invent one. `#[serde(default)]` for forward-compat with a host that
    /// predates the field (it then stays `None` -> "unknown", never faked).
    #[serde(default)]
    pub exchange_id: Option<String>,
    pub dispatch_path: DispatchPath,
    pub phase: Phase,
    pub model: String,
    #[serde(default)]
    pub status: Option<u16>,
    #[serde(default)]
    pub capsule_id: Option<String>,
    /// How `capsule_id` was obtained -- see [`CapsuleIdProvenance`]. On a
    /// `RemoteMesh` terminal envelope this is the PEER's asserted capsule id
    /// for its own half of the exchange (host-side
    /// `CapsuleIdProvenance::PeerAsserted`) -- the lookup key a later
    /// ledger fetch dereferences, never itself verified here. `None`
    /// exactly when `capsule_id` is `None`, and always `None` on a host that
    /// predates this field -- never fabricated.
    #[serde(default)]
    pub capsule_id_provenance: Option<CapsuleIdProvenance>,
    #[serde(default)]
    pub nonce: Option<String>,
    /// The host's serving-provenance block for this terminal event -- what
    /// ran, at what fidelity, on whose hardware. Absent on a host that
    /// predates the block, and on non-terminal / non-served envelopes.
    #[serde(default)]
    pub serving_provenance: Option<HostServingProvenance>,
    /// The REAL token usage the host-served backend reported for this exchange
    /// (host-side `ExchangeUsage`). Previously DROPPED by this mirror -- carried
    /// through now so a host-served capsule seals the backend's real counts, not
    /// a zeroed stub. `None` on effective-request envelopes and wherever the
    /// dispatch produced no usage (plugin-served stub, denial) -- never zeroed.
    #[serde(default)]
    pub usage: Option<MirrorUsage>,
    /// The canonical JSON-DIGEST of the REAL request body the host dispatched
    /// (host-side `request_digest`, computed the same way as this plugin's
    /// `canonical_body_digest`). This is the one fact that lets a host-served
    /// capsule bind its `agent_input_digest` to the real request bytes. `None`
    /// on a host predating the field / a non-JSON-body exchange -- never faked.
    #[serde(default)]
    pub request_digest: Option<String>,
    /// The canonical JSON-DIGEST of the REAL response body the host served
    /// (host-side `response_digest`). Lets a host-served capsule bind its
    /// `agent_output_digest` to the real response bytes, not just the terminal
    /// accounting facts. `None` on a host predating the field / a streamed body
    /// -- never faked.
    #[serde(default)]
    pub response_digest: Option<String>,
    /// The canonical JSON-DIGEST of the flattened `tool_calls` the model emitted
    /// (host-side `tool_calls_digest`, byte-for-byte the Python reference
    /// `json_digest(tool_calls)`). This is the fact that lets a host-served
    /// capsule seal a REAL `tool_calls_digest`. `None` -- and then absent from
    /// the capsule -- when the model emitted none, never a digest over `[]`.
    #[serde(default)]
    pub tool_calls_digest: Option<String>,
    /// The canonical JSON-DIGEST of the model's `reasoning_content` (host-side
    /// `reasoning_digest`). `None` for a non-reasoning model (honest null),
    /// never fabricated.
    #[serde(default)]
    pub reasoning_digest: Option<String>,
    /// The id shared by BOTH halves of a client-marked twin pair, minted
    /// host-side (`mesh-llm-host-runtime`'s
    /// `runtime::twin_sample::mint_twin_bracket_id`). `#[serde(default)]` for
    /// forward-compat with a host that predates the field (it then stays
    /// `None`, never fabricated) -- mirrors host-side
    /// `OpenAiExchangeEnvelope::twin_bracket_id`, which the host itself omits
    /// (`skip_serializing_if`) on every exchange that wasn't ambiently
    /// twinned.
    #[serde(default)]
    pub twin_bracket_id: Option<String>,
    /// SHA-256 hex of the answer text (`choices[0].message.content`), set by a
    /// host that computes it; `None` from a host that predates the field.
    #[serde(default)]
    pub response_text_digest: Option<String>,
    /// The exchange's request and response bodies. mesh-llm does not send
    /// them today, so this is always absent there and the plugin sees digests
    /// only.
    /// Kept on disk only with this plugin's own opt-in (`exchange_text`).
    #[serde(default)]
    pub exchange_bodies: Option<ExchangeBodies>,
}

/// See [`OpenAiExchangeEnvelope::exchange_bodies`]. Either side may be absent.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct ExchangeBodies {
    #[serde(default)]
    pub request: Option<serde_json::Value>,
    #[serde(default)]
    pub response: Option<serde_json::Value>,
}

/// Mirror of the host's `ExchangeUsage` (real token counts). Every field is a
/// real count the host read off the served backend's `usage` object; this
/// plugin never fabricates one.
#[derive(Debug, Clone, Copy, Deserialize, serde::Serialize, PartialEq, Eq)]
pub struct MirrorUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// Hand-mirror of the host's `ServingProvenance` (same repo/module note as
/// [`OpenAiExchangeEnvelope`]). Every field is `Option` + `#[serde(default)]`:
/// the host omits (via `skip_serializing_if`) any fact it does not know, so an
/// absent field deserializes to `None` here -- an honest "the host did not
/// tell us", never a fabricated value. These are the real fields that fill the
/// capsule's own `serving_provenance` quantization/hardware/model-digest slots.
#[derive(Debug, Clone, Deserialize, serde::Serialize, PartialEq, Eq)]
pub struct HostServingProvenance {
    #[serde(default)]
    pub served_by_node_id: Option<String>,
    #[serde(default)]
    pub hostname: Option<String>,
    #[serde(default)]
    pub quantization: Option<String>,
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub parameter_size: Option<String>,
    #[serde(default)]
    pub layer_count: Option<u32>,
    #[serde(default)]
    pub model_identity_hash: Option<String>,
    /// SHA-256 of the served GGUF's file BYTES, from the host's load-time
    /// hash of the file it actually opened for serving (`mesh-llm`
    /// `ServedModelIdentity::weights_digest`) -- a different fact from
    /// `model_identity_hash` (a hash of a reference STRING, absent for a bare
    /// local path). `#[serde(default)]` so a host that predates this field
    /// simply arrives without it, and `None` here is never fabricated.
    #[serde(default)]
    pub weights_digest: Option<String>,
    #[serde(default)]
    pub model_canonical_ref: Option<String>,
    #[serde(default)]
    pub model_revision: Option<String>,
    #[serde(default)]
    pub gpu: Option<String>,
    #[serde(default)]
    pub vram_bytes: Option<u64>,
    #[serde(default)]
    pub is_soc: Option<bool>,
    /// The mesh node that asked this host to serve the exchange -- set by the
    /// host only on a host-served exchange that arrived over the mesh HTTP
    /// tunnel, from the tunnel's QUIC-authenticated remote id (never from the
    /// request). `#[serde(default)]` so a host that predates it reads as
    /// "requester unknown", and nothing is pushed.
    #[serde(default)]
    pub requested_by_node_id: Option<String>,
}

/// Cap on the in-memory lifecycle-event mirror: a long-lived plugin on a
/// busy node observes an unbounded stream of envelopes, and the old
/// unbounded `Vec` grew for the life of the process (a slow memory leak by
/// design). The COMPLETE record is the on-disk JSONL log this module also
/// appends; in memory only the most recent window is needed (its one reader,
/// `latest_provenance_for_model`, scans most-recent-first anyway).
const MAX_IN_MEMORY_EVENTS: usize = 1024;

/// The lifecycle envelopes this plugin has recently observed on the mesh
/// channel, most-recent-last, bounded to [`MAX_IN_MEMORY_EVENTS`] (older
/// entries are dropped from memory; the JSONL log keeps everything) --
/// inspectable by the e2e test (a separate process) via the JSONL file this
/// also appends to under `data_dir`, so cross-process correlation against
/// the capsule this plugin's own handler produced for the same exchange
/// doesn't depend on shared memory.
pub struct ObservedLifecycleEvents {
    events: Mutex<VecDeque<OpenAiExchangeEnvelope>>,
    log_path: PathBuf,
}

impl ObservedLifecycleEvents {
    pub fn open(data_dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(data_dir)?;
        Ok(Self {
            events: Mutex::new(VecDeque::new()),
            log_path: data_dir.join("lifecycle-events.jsonl"),
        })
    }

    /// Whether this observed envelope is a HOST-SERVED terminal exchange this
    /// plugin should seal a capsule for -- the gap this closes. A host-served
    /// real-weights exchange (a loaded GGUF, routed host->native-runtime) never
    /// reaches this plugin's own HTTP handler, so nothing else seals it.
    ///
    /// It is distinguished from this plugin's OWN plugin-served stub (which the
    /// handler already seals, and which must NOT be double-sealed here) by a
    /// real served-model descriptor: a host-loaded GGUF carries `architecture`
    /// and/or `model_identity_hash` in its serving provenance, while the
    /// synthetic plugin-advertised endpoint has neither (no loaded weights ->
    /// those fields are `null`). That is the honest discriminator -- a real
    /// model-identity fact only a real served model has -- not a heuristic.
    ///
    /// Requires: a `Terminal` phase, a 2xx status (a served success), a
    /// serving-provenance block, real model identity in it, and a
    /// NON-`RemoteMesh`
    /// dispatch path -- a `RemoteMesh` terminal event is this node's
    /// REQUESTER-side half (see `is_sealable_requester_side`), never the
    /// served half, no matter what a forwarded serving-provenance block might
    /// otherwise look like. Mutually exclusive with `is_sealable_requester_side`
    /// by construction (one requires `dispatch_path != RemoteMesh`, the other
    /// requires `== RemoteMesh`), so a single terminal event is never
    /// double-sealed under two roles.
    pub fn is_sealable_host_served(envelope: &OpenAiExchangeEnvelope) -> bool {
        if envelope.phase != Phase::Terminal {
            return false;
        }
        if !matches!(envelope.status, Some(200..=299)) {
            return false;
        }
        if matches!(envelope.dispatch_path, DispatchPath::RemoteMesh) {
            return false;
        }
        match envelope.serving_provenance.as_ref() {
            Some(prov) => {
                // This is a DISCRIMINATOR against double-sealing, not a
                // "worth sealing" test: it separates a real host-served
                // exchange from this plugin's OWN synthetic plugin-served stub
                // (already sealed by the HTTP handler). It keys on real
                // served-MODEL IDENTITY, which only a real loaded model has.
                // `weights_digest` IS such identity -- the digest of the bytes
                // actually loaded -- and the strongest of the three: on a local
                // `--gguf` serve the host leaves `architecture` and
                // `model_identity_hash` null but still reports `weights_digest`.
                // Accepting it is a refinement of the same criterion, not a
                // loosening. Safety rests on the invariant that a plugin-served
                // stub (no loaded GGUF) never carries a `weights_digest`; the
                // `plugin_served_stub_*` tests below are the guard that goes red
                // if that invariant is ever violated.
                //
                // NEVER widen this to non-identity fields. `served_by_node_id`,
                // `hostname`, `dispatch_path` are WHO/WHERE served, not WHAT
                // model; they must never qualify an exchange on their own, or
                // the stub discriminator breaks. If the stub ever *can* carry a
                // digest, this needs a different discriminator, not a 4th field.
                prov.architecture.is_some()
                    || prov.model_identity_hash.is_some()
                    || prov.weights_digest.is_some()
            }
            None => false,
        }
    }

    /// Whether this observed envelope is the REQUESTER'S half of a proxied
    /// exchange this plugin should seal a capsule for
    /// (requester-side sealing on a proxied exchange -- the accountability
    /// gap the 2026-09-07 live twin run surfaced: when this node proxies a chat
    /// completion to a peer via the `RemoteMesh` dispatch path, the peer seals a
    /// `served`-role capsule for its half, but the routing/requesting node
    /// sealed NOTHING for its own half, directly against the manifesto's
    /// "build the requester's half first").
    ///
    /// Unlike `is_sealable_host_served`, this does NOT require a populated
    /// `serving_provenance` model-identity block: the router legitimately may
    /// not know the peer's hardware/model fidelity. `role_and_observation_point`
    /// (`capsule_emit.rs`) already derives `role: "requested"` from
    /// `dispatch_path` alone, honestly defaulting every unreported enrichment
    /// field to "unknown" rather than fabricating one.
    ///
    /// Requires: a `Terminal` phase, a 2xx status (the peer answered
    /// successfully), and `dispatch_path == RemoteMesh`.
    pub fn is_sealable_requester_side(envelope: &OpenAiExchangeEnvelope) -> bool {
        envelope.phase == Phase::Terminal
            && matches!(envelope.status, Some(200..=299))
            && matches!(envelope.dispatch_path, DispatchPath::RemoteMesh)
    }

    pub fn record(&self, envelope: OpenAiExchangeEnvelope) {
        tracing::info!(
            dispatch_path = ?envelope.dispatch_path,
            phase = ?envelope.phase,
            model = %envelope.model,
            status = ?envelope.status,
            "observed openai.exchange.v1 lifecycle event"
        );
        if let Ok(line) = serde_json::to_string(&LoggedEnvelope::from(&envelope)) {
            if let Ok(mut file) = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.log_path)
            {
                let _ = writeln!(file, "{line}");
            }
        }
        let mut events = self
            .events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        events.push_back(envelope);
        // Ring-buffer semantics: drop the oldest once past the cap.
        while events.len() > MAX_IN_MEMORY_EVENTS {
            events.pop_front();
        }
    }

    /// In-process accessor; the e2e test instead reads the JSONL log this
    /// also writes (it runs the plugin as a separate process, so in-memory
    /// state isn't visible to it) -- kept for a same-process caller (e.g. a
    /// future unit test of the handler itself). Returns at most the
    /// in-memory window ([`MAX_IN_MEMORY_EVENTS`]).
    #[allow(dead_code)]
    pub fn snapshot(&self) -> Vec<OpenAiExchangeEnvelope> {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .cloned()
            .collect()
    }

    /// The most recently observed host serving-provenance for `model` (from a
    /// `Terminal` envelope that carried one). This is how the host's real
    /// quantization/hardware/model-digest reach the capsule: the host publishes
    /// it on the exchange channel, the plugin captures it here, and the next
    /// exchange this plugin seals for the same model reads it back.
    ///
    /// Correlation is by MODEL, not by exchange id, and deliberately so: the
    /// host mints its own random raw-proxy `exchange_id` for the terminal
    /// event, which is disjoint from the deterministic id this plugin's own
    /// direct-serve handler mints -- so the two cannot be paired by id. Model
    /// is the honest join key here: serving fidelity/hardware for a given model
    /// on a given node is stable across that node's exchanges, so "the latest
    /// host-reported provenance for this model" is the correct fact to attest.
    /// Returns `None` when the host has published no provenance for the model
    /// yet, leaving the capsule's slots at their honest `unknown`/`null`.
    pub fn latest_provenance_for_model(&self, model: &str) -> Option<HostServingProvenance> {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .rev()
            .find(|e| e.model == model && e.serving_provenance.is_some())
            .and_then(|e| e.serving_provenance.clone())
    }
}

/// `OpenAiExchangeEnvelope` only derives `Deserialize` (it's the wire shape
/// we receive, never one we send) -- this sibling carries `Serialize` so the
/// observed-events log can be written without adding an unused derive to the
/// wire type itself.
#[derive(serde::Serialize)]
struct LoggedEnvelope {
    #[serde(skip_serializing_if = "Option::is_none")]
    exchange_id: Option<String>,
    dispatch_path: DispatchPath,
    phase: Phase,
    model: String,
    status: Option<u16>,
    capsule_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    capsule_id_provenance: Option<CapsuleIdProvenance>,
    nonce: Option<String>,
    /// Persisted so the out-of-process e2e test can confirm the host's real
    /// serving provenance was received (in-memory state isn't visible to it).
    #[serde(skip_serializing_if = "Option::is_none")]
    serving_provenance: Option<HostServingProvenance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<MirrorUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls_digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    twin_bracket_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_text_digest: Option<String>,
}

impl From<&OpenAiExchangeEnvelope> for LoggedEnvelope {
    fn from(e: &OpenAiExchangeEnvelope) -> Self {
        Self {
            exchange_id: e.exchange_id.clone(),
            dispatch_path: e.dispatch_path.clone(),
            phase: e.phase.clone(),
            model: e.model.clone(),
            status: e.status,
            capsule_id: e.capsule_id.clone(),
            capsule_id_provenance: e.capsule_id_provenance,
            nonce: e.nonce.clone(),
            serving_provenance: e.serving_provenance.clone(),
            usage: e.usage,
            request_digest: e.request_digest.clone(),
            response_digest: e.response_digest.clone(),
            tool_calls_digest: e.tool_calls_digest.clone(),
            reasoning_digest: e.reasoning_digest.clone(),
            twin_bracket_id: e.twin_bracket_id.clone(),
            response_text_digest: e.response_text_digest.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stock upstream host's terminal event, as its serializer writes it
    /// (`plugin::openai_exchange` on mesh-llm main): no `requested_by_node_id`
    /// or `twin_bracket_id` (fork-only), and a `nonce_source` this plugin does
    /// not read. It must parse, or a stock node would seal nothing.
    #[test]
    fn a_stock_upstream_terminal_event_parses() {
        let body = serde_json::json!({
            "exchange_id": "exch-1",
            "dispatch_path": "remote_mesh",
            "phase": "terminal",
            "model": "qwen",
            "status": 200,
            "capsule_id": "capsule-chatcmpl-1",
            "capsule_id_provenance": "peer_asserted",
            "nonce": "n-1",
            "nonce_source": "client_supplied",
            "serving_provenance": {
                "served_by_node_id": "c1f5",
                "quantization": "Q4_K_M",
                "architecture": "llama"
            },
            "usage": { "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8 },
            "request_digest": "a".repeat(64),
            "response_digest": "b".repeat(64)
        });
        let envelope: OpenAiExchangeEnvelope =
            serde_json::from_value(body).expect("stock event parses");
        assert_eq!(envelope.dispatch_path, DispatchPath::RemoteMesh);
        let provenance = envelope.serving_provenance.expect("serving provenance");
        assert_eq!(provenance.served_by_node_id.as_deref(), Some("c1f5"));
        assert!(provenance.requested_by_node_id.is_none());
        assert!(envelope.twin_bracket_id.is_none());
    }

    /// A REAL host terminal event (verbatim wire bytes captured from a live
    /// `mesh-llm serve` on this branch) deserializes into the mirror, and the
    /// serving-provenance hardware facts survive intact -- while the fields the
    /// host reported as `null` (synthetic plugin-served model has no GGUF
    /// metadata) stay `None`, never fabricated.
    /// A host that hands exchange bodies to plugins sends them on the
    /// terminal event; any other host sends none, and the field stays `None`.
    #[test]
    fn exchange_bodies_are_read_when_sent_and_absent_otherwise() {
        let with = r#"{"dispatch_path":"raw_proxy","phase":"terminal","model":"m","exchange_id":"ex-1","exchange_bodies":{"request":{"model":"m"},"response":{"choices":[{"message":{"content":"hi"}}]}}}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(with).unwrap();
        let bodies = env.exchange_bodies.expect("bodies");
        assert_eq!(bodies.request.unwrap()["model"], "m");
        assert_eq!(
            bodies.response.unwrap()["choices"][0]["message"]["content"],
            "hi"
        );

        let without = r#"{"dispatch_path":"raw_proxy","phase":"terminal","model":"m"}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(without).unwrap();
        assert!(env.exchange_bodies.is_none());
    }

    #[test]
    fn real_host_terminal_event_deserializes_serving_provenance() {
        let wire = r#"{"dispatch_path":"raw_proxy","phase":"terminal","model":"allowed-test-model","status":200,"capsule_id":null,"nonce":null,"serving_provenance":{"served_by_node_id":"fa28d0dfe5f0b2c4a8f0fcb15838075e4e5f0b32d6dd5df029588e8992fad5ac","hostname":"node-a.local","quantization":null,"architecture":null,"context_length":null,"parameter_size":null,"layer_count":null,"model_identity_hash":null,"model_canonical_ref":null,"model_revision":null,"gpu":"Apple M4 Max","vram_bytes":28991029248,"is_soc":true}}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire).expect("parse real event");
        let prov = env.serving_provenance.expect("provenance present");
        // Real hardware facts the host survey knows.
        assert_eq!(prov.gpu.as_deref(), Some("Apple M4 Max"));
        assert_eq!(prov.vram_bytes, Some(28_991_029_248));
        assert_eq!(prov.is_soc, Some(true));
        assert_eq!(prov.hostname.as_deref(), Some("node-a.local"));
        assert_eq!(
            prov.served_by_node_id.as_deref(),
            Some("fa28d0dfe5f0b2c4a8f0fcb15838075e4e5f0b32d6dd5df029588e8992fad5ac")
        );
        // Synthetic plugin-served model has no GGUF metadata -> honest None.
        assert!(prov.quantization.is_none());
        assert!(prov.architecture.is_none());
        assert!(prov.model_identity_hash.is_none());
        // A host that predates `weights_digest` simply omits the field --
        // this mirror stays forward/backward compatible, and it deserializes
        // to `None`, never a fabricated placeholder.
        assert!(prov.weights_digest.is_none());
    }

    /// A host reporting BOTH the name-hash (`model_identity_hash`) and the
    /// bytes-hash (`weights_digest`) survives the deserialize with both
    /// distinct facts intact -- neither collapses into or replaces the other.
    #[test]
    fn serving_provenance_carries_weights_digest_alongside_model_identity_hash() {
        let wire = r#"{"dispatch_path":"raw_proxy","phase":"terminal","model":"local-gguf/sha256-4ff195f73917d9c2","status":200,"capsule_id":null,"nonce":null,"serving_provenance":{"served_by_node_id":"143f4d9f8cd9a9","hostname":"node-a.local","architecture":"llama","model_identity_hash":null,"weights_digest":"904548955b8a6478df029588e8992fad5ac4ff195f73917d9c2e5f0b32d6dd5","gpu":"Apple M4 Max","vram_bytes":28991029248,"is_soc":true}}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire).expect("parse");
        assert!(ObservedLifecycleEvents::is_sealable_host_served(&env));
        let prov = env.serving_provenance.expect("provenance present");
        // A local GGUF has no reference-string identity hash (honest None)...
        assert!(prov.model_identity_hash.is_none());
        // ...but the load-time bytes digest is a real, present fact.
        assert_eq!(
            prov.weights_digest.as_deref(),
            Some("904548955b8a6478df029588e8992fad5ac4ff195f73917d9c2e5f0b32d6dd5")
        );
    }

    /// A REAL host-served terminal event now carries the backend's real `usage`
    /// AND the host-forwarded canonical `request_digest` — previously the mirror
    /// dropped both. Both survive the deserialize, and the event is recognized
    /// as a sealable host-served exchange (real model identity present).
    #[test]
    fn real_host_served_terminal_carries_usage_and_request_digest_and_is_sealable() {
        let wire = r#"{"exchange_id":"exch-7","dispatch_path":"raw_proxy","phase":"terminal","model":"local-gguf/sha256-4ff195f73917d9c2","status":200,"capsule_id":null,"nonce":null,"usage":{"prompt_tokens":41,"completion_tokens":2,"total_tokens":43},"request_digest":"a6329c5ebb66562f38a8136a8d8511b6aeed166e4c7d889b9133ac96fc49a9d5","response_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","tool_calls_digest":"f294be8a53bb9c29cd94472721f0857591f34b23fe010882de79b9fb210b1395","serving_provenance":{"served_by_node_id":"143f4d9f8cd9a9","hostname":"node-a.local","architecture":"llama","context_length":131072,"parameter_size":"3B","layer_count":28,"model_identity_hash":"904548955b8a6478","gpu":"Apple M4 Max","vram_bytes":28991029248,"is_soc":true}}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire).expect("parse");
        let usage = env.usage.expect("real usage carried through");
        assert_eq!(usage.prompt_tokens, 41);
        assert_eq!(usage.completion_tokens, 2);
        assert_eq!(usage.total_tokens, 43);
        assert_eq!(
            env.request_digest.as_deref(),
            Some("a6329c5ebb66562f38a8136a8d8511b6aeed166e4c7d889b9133ac96fc49a9d5")
        );
        // The host-forwarded response-body / tool_calls digests survive the
        // deserialize -- the real tool_calls_digest is the Python-reference value.
        assert_eq!(
            env.response_digest.as_deref(),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );
        assert_eq!(
            env.tool_calls_digest.as_deref(),
            Some("f294be8a53bb9c29cd94472721f0857591f34b23fe010882de79b9fb210b1395")
        );
        // A non-reasoning model forwarded no reasoning digest -> honest None.
        assert!(env.reasoning_digest.is_none());
        assert_eq!(env.exchange_id.as_deref(), Some("exch-7"));
        assert!(ObservedLifecycleEvents::is_sealable_host_served(&env));
    }

    /// A terminal event that carries the host-minted `twin_bracket_id`
    /// (a client-marked twin pair) survives the deserialize; an ordinary
    /// terminal event that omits the field (the overwhelming majority)
    /// parses with it `None` -- never a fabricated bracket.
    #[test]
    fn twin_bracket_id_round_trips_when_present_and_is_none_when_omitted() {
        let wire_with_id = r#"{"exchange_id":"exch-7","dispatch_path":"raw_proxy","phase":"terminal","model":"local-gguf/sha256-4ff195f73917d9c2","status":200,"capsule_id":null,"nonce":null,"twin_bracket_id":"twin-abc123"}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire_with_id).expect("parse");
        assert_eq!(env.twin_bracket_id.as_deref(), Some("twin-abc123"));

        let wire_without_id = r#"{"exchange_id":"exch-7","dispatch_path":"raw_proxy","phase":"terminal","model":"local-gguf/sha256-4ff195f73917d9c2","status":200,"capsule_id":null,"nonce":null}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire_without_id).expect("parse");
        assert!(env.twin_bracket_id.is_none());
    }

    #[test]
    fn response_text_digest_parses_when_present_and_is_none_when_omitted() {
        let with = format!(
            r#"{{"exchange_id":"e","dispatch_path":"raw_proxy","phase":"terminal","model":"m","status":200,"capsule_id":null,"nonce":null,"response_text_digest":"{}"}}"#,
            "cd".repeat(32)
        );
        let env: OpenAiExchangeEnvelope = serde_json::from_str(&with).expect("parse");
        assert_eq!(
            env.response_text_digest.as_deref(),
            Some("cd".repeat(32).as_str())
        );
        let without = r#"{"exchange_id":"e","dispatch_path":"raw_proxy","phase":"terminal","model":"m","status":200,"capsule_id":null,"nonce":null}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(without).expect("parse");
        assert!(env.response_text_digest.is_none());
    }

    /// The plugin's OWN plugin-served stub terminal event (a synthetic endpoint
    /// with no loaded GGUF -> null architecture / model_identity_hash) is NOT
    /// recognized as sealable-on-observe: its capsule is already produced by the
    /// plugin's own HTTP handler, and sealing it here too would double-seal.
    #[test]
    fn plugin_served_stub_terminal_is_not_sealed_on_observe() {
        let wire = r#"{"dispatch_path":"raw_proxy","phase":"terminal","model":"allowed-test-model","status":200,"capsule_id":null,"nonce":null,"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0},"serving_provenance":{"served_by_node_id":"node","hostname":"h","architecture":null,"model_identity_hash":null,"gpu":"Apple M4 Max","vram_bytes":28991029248,"is_soc":true}}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire).expect("parse");
        assert!(!ObservedLifecycleEvents::is_sealable_host_served(&env));
    }

    /// A real local-`--gguf` host-served terminal event: the host leaves
    /// `architecture`/`model_identity_hash` null but reports `served_by_node_id`
    /// and `weights_digest` (the raw served-weights hash). That IS real
    /// served-model identity, so the exchange is sealable -- otherwise a
    /// single-node `--gguf` serve seals nothing and the Evidence ledger stays
    /// empty despite honest provenance being present.
    #[test]
    fn host_served_gguf_terminal_with_weights_digest_only_is_sealable() {
        let wire = r#"{"dispatch_path":"raw_proxy","phase":"terminal","model":"local-gguf/sha256-6c1a2b41","status":200,"capsule_id":null,"nonce":null,"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12},"serving_provenance":{"served_by_node_id":"e988a4a64c","hostname":"node-a.local","architecture":null,"model_identity_hash":null,"weights_digest":"6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff","gpu":"Apple M4 Max","vram_bytes":28991029248,"is_soc":true}}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire).expect("parse");
        assert!(ObservedLifecycleEvents::is_sealable_host_served(&env));
    }

    /// MUTANT / hazard documentation for the `weights_digest` disjunction.
    /// A plugin-served STUB shape (`allowed-test-model`, no loaded GGUF, so
    /// `architecture`/`model_identity_hash` null) that is mutated to ALSO carry
    /// a `weights_digest` DOES evaluate sealable -- a double-seal hazard, since
    /// the HTTP handler already sealed the plugin-served exchange. This is safe
    /// in production ONLY because a real plugin-served stub has no loaded
    /// weights and therefore never carries a `weights_digest`; that invariant is
    /// guarded by `plugin_served_stub_terminal_is_not_sealed_on_observe` (which
    /// goes red the moment a stub gains one). If a stub ever legitimately grows
    /// a digest, fix the discriminator (e.g. gate on `dispatch_path`/served-role
    /// too) -- do NOT paper over it by removing this field.
    #[test]
    fn mutant_stub_with_weights_digest_seals_documenting_the_double_seal_hazard() {
        let wire = r#"{"dispatch_path":"raw_proxy","phase":"terminal","model":"allowed-test-model","status":200,"capsule_id":null,"nonce":null,"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0},"serving_provenance":{"served_by_node_id":"node","hostname":"h","architecture":null,"model_identity_hash":null,"weights_digest":"deadbeef","gpu":"Apple M4 Max","vram_bytes":28991029248,"is_soc":true}}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire).expect("parse");
        // Passes today only because real stubs never reach this shape -- see the
        // guard test above, which is what actually protects the invariant.
        assert!(ObservedLifecycleEvents::is_sealable_host_served(&env));
    }

    /// An effective-request (non-terminal) envelope is never sealed on observe.
    #[test]
    fn effective_request_is_not_sealable() {
        let wire = r#"{"dispatch_path":"raw_proxy","phase":"effective_request","model":"local-gguf/x","status":null}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire).expect("parse");
        assert!(!ObservedLifecycleEvents::is_sealable_host_served(&env));
    }

    /// A host predating the serving_provenance block (its terminal event omits
    /// the field entirely) still deserializes -- the mirror is forward/backward
    /// compatible, and the block is simply `None`.
    #[test]
    fn terminal_event_without_serving_provenance_still_parses() {
        let wire = r#"{"dispatch_path":"raw_proxy","phase":"terminal","model":"m","status":200}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire).expect("parse legacy event");
        assert!(env.serving_provenance.is_none());
    }

    /// A host newer than this plugin build emitting a `dispatch_path` this
    /// mirror predates (e.g. some future dispatch path not yet named here --
    /// `remote_mesh` itself graduated out of this role once the plugin
    /// learned to recognize it) must not fail to parse the WHOLE envelope --
    /// `#[serde(other)]` catches it as `Unknown` rather than rejecting every
    /// field in the record. Never treated as sealable, and (2026-09-06 role
    /// ruling) never silently labeled `served`: an unrecognized dispatch path
    /// is observed and labeled `role: "unknown"`, never guessed at.
    #[test]
    fn unknown_dispatch_path_parses_as_unknown_not_a_parse_failure() {
        let wire = r#"{"dispatch_path":"some_future_dispatch_path","phase":"terminal","model":"m","status":200}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire)
            .expect("a forward host value must not break parsing of the whole envelope");
        assert_eq!(env.dispatch_path, DispatchPath::Unknown);
    }

    /// `remote_mesh` is now a KNOWN variant (2026-09-06 role ruling), not the
    /// `Unknown` catch-all -- this is the precondition for
    /// `role_for_dispatch_path` to tell "this node routed the exchange" apart
    /// from "this mirror predates whatever the host just sent".
    #[test]
    fn remote_mesh_dispatch_path_parses_as_its_own_known_variant() {
        let wire = r#"{"dispatch_path":"remote_mesh","phase":"terminal","model":"m","status":200}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire).expect("parse remote_mesh");
        assert_eq!(env.dispatch_path, DispatchPath::RemoteMesh);
    }

    /// A `RemoteMesh` terminal event's `capsule_id`/`capsule_id_provenance`
    /// -- the peer's self-asserted capsule id for its own half of the
    /// exchange -- survives the wire round trip byte-for-byte (real shape
    /// verified against `CapsuleIdProvenance::PeerAsserted`'s host-side wire
    /// value, `mesh-llm-host-runtime`'s `#[serde(rename_all =
    /// "snake_case")]`).
    #[test]
    fn peer_asserted_capsule_id_survives_the_wire_round_trip() {
        let wire = r#"{"dispatch_path":"remote_mesh","phase":"terminal","model":"m","status":200,"capsule_id":"peer-cap-abc123","capsule_id_provenance":"peer_asserted"}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire).expect("parse");
        assert_eq!(env.capsule_id.as_deref(), Some("peer-cap-abc123"));
        assert_eq!(
            env.capsule_id_provenance,
            Some(CapsuleIdProvenance::PeerAsserted)
        );
    }

    /// A locally-served envelope's `capsule_id` is this node's OWN
    /// `X-Capsule-Id` marker (`SelfMinted`) -- a distinct fact from a peer's
    /// asserted id, and this mirror must keep the two distinguishable so a
    /// caller never mislabels this node's own marker as a peer's claim (see
    /// `main.rs::seal_observed_host_exchange`'s guard).
    #[test]
    fn self_minted_capsule_id_parses_as_its_own_distinct_variant() {
        let wire = r#"{"dispatch_path":"typed_frontend","phase":"terminal","model":"m","status":200,"capsule_id":"self-cap-xyz","capsule_id_provenance":"self_minted"}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire).expect("parse");
        assert_eq!(env.capsule_id.as_deref(), Some("self-cap-xyz"));
        assert_eq!(
            env.capsule_id_provenance,
            Some(CapsuleIdProvenance::SelfMinted)
        );
        assert_ne!(
            env.capsule_id_provenance,
            Some(CapsuleIdProvenance::PeerAsserted)
        );
    }

    /// Same forward-compat discipline as `DispatchPath::Unknown`: a
    /// provenance value this mirror predates must not fail parsing of the
    /// whole envelope, and must never be silently promoted to
    /// `PeerAsserted`.
    #[test]
    fn unknown_capsule_id_provenance_parses_as_unknown_not_a_parse_failure() {
        let wire = r#"{"dispatch_path":"remote_mesh","phase":"terminal","model":"m","status":200,"capsule_id":"c","capsule_id_provenance":"some_future_provenance"}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire)
            .expect("a forward host value must not break parsing of the whole envelope");
        assert_eq!(
            env.capsule_id_provenance,
            Some(CapsuleIdProvenance::Unknown)
        );
    }

    /// A host predating this field omits it entirely -- `#[serde(default)]`
    /// keeps it `None`, never a fabricated provenance.
    #[test]
    fn missing_capsule_id_provenance_defaults_to_none() {
        let wire = r#"{"dispatch_path":"remote_mesh","phase":"terminal","model":"m","status":200,"capsule_id":"c"}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire).expect("parse");
        assert_eq!(env.capsule_id_provenance, None);
    }

    /// `capsule_id_provenance_wire_value` matches the host's own
    /// `#[serde(rename_all = "snake_case")]` wire form exactly -- this is
    /// the value `main.rs` re-serializes onto `ServingProvenance::
    /// peer_capsule_id_provenance`, so a mismatch here would silently
    /// diverge the sealed record from what the host actually sent.
    #[test]
    fn capsule_id_provenance_wire_value_matches_the_snake_case_host_wire_form() {
        assert_eq!(
            capsule_id_provenance_wire_value(&CapsuleIdProvenance::SelfMinted),
            "self_minted"
        );
        assert_eq!(
            capsule_id_provenance_wire_value(&CapsuleIdProvenance::PeerAsserted),
            "peer_asserted"
        );
        assert_eq!(
            capsule_id_provenance_wire_value(&CapsuleIdProvenance::Unknown),
            "unknown"
        );
    }

    /// A `PeerAsserted` value is the only case `peer_capsule_id_for_seal`
    /// surfaces -- and it returns the exact `(id, wire_value)` pair a caller
    /// threads onto `ServingProvenance::peer_capsule_id{,_provenance}`.
    #[test]
    fn peer_capsule_id_for_seal_surfaces_peer_asserted() {
        let wire = r#"{"dispatch_path":"remote_mesh","phase":"terminal","model":"m","status":200,"capsule_id":"peer-cap-1","capsule_id_provenance":"peer_asserted"}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire).expect("parse");
        assert_eq!(
            peer_capsule_id_for_seal(&env),
            Some(("peer-cap-1", "peer_asserted"))
        );
    }

    /// THE MISLABELING BUG this guard exists to prevent: a `SelfMinted`
    /// capsule id (this node's own `X-Capsule-Id` marker on a locally-served
    /// envelope) must NEVER be surfaced as a peer's claim, even though the
    /// wire shape is otherwise identical to the `PeerAsserted` case.
    #[test]
    fn peer_capsule_id_for_seal_never_surfaces_self_minted() {
        let wire = r#"{"dispatch_path":"typed_frontend","phase":"terminal","model":"m","status":200,"capsule_id":"self-cap-1","capsule_id_provenance":"self_minted"}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire).expect("parse");
        assert_eq!(peer_capsule_id_for_seal(&env), None);
    }

    /// An unrecognized provenance value (a host that emits a variant this
    /// mirror predates) is observed, never guessed at -- it must not be
    /// treated as `PeerAsserted` by default.
    #[test]
    fn peer_capsule_id_for_seal_never_surfaces_unknown_provenance() {
        let wire = r#"{"dispatch_path":"remote_mesh","phase":"terminal","model":"m","status":200,"capsule_id":"c","capsule_id_provenance":"some_future_value"}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire).expect("parse");
        assert_eq!(peer_capsule_id_for_seal(&env), None);
    }

    /// No `capsule_id` at all (a host that predates the field, or a
    /// `RemoteMesh` event where nothing was observed) stays `None` -- an
    /// honest absence, never fabricated.
    #[test]
    fn peer_capsule_id_for_seal_is_none_when_no_capsule_id_was_observed() {
        let wire = r#"{"dispatch_path":"remote_mesh","phase":"terminal","model":"m","status":200}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(wire).expect("parse");
        assert_eq!(peer_capsule_id_for_seal(&env), None);
    }

    /// THE GAP REQUESTER-SIDE SEALING CLOSES: a `RemoteMesh`
    /// terminal event (this node routed the exchange to a peer) is sealable as
    /// the REQUESTER'S half, even with no `serving_provenance` at all (the
    /// router legitimately may not know the peer's hardware) -- and it is
    /// mutually exclusive with `is_sealable_host_served`, even when a
    /// `RemoteMesh` event DOES carry a full served-model provenance block
    /// (e.g. forwarded from the peer), so a single terminal event is never
    /// double-sealed under two roles.
    #[test]
    fn remote_mesh_terminal_is_sealable_requester_side_never_host_served() {
        let bare = r#"{"dispatch_path":"remote_mesh","phase":"terminal","model":"m","status":200}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(bare).expect("parse");
        assert!(ObservedLifecycleEvents::is_sealable_requester_side(&env));
        assert!(!ObservedLifecycleEvents::is_sealable_host_served(&env));

        let enriched = r#"{"dispatch_path":"remote_mesh","phase":"terminal","model":"m","status":200,"serving_provenance":{"architecture":"llama","model_identity_hash":"abc"}}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(enriched).expect("parse");
        assert!(ObservedLifecycleEvents::is_sealable_requester_side(&env));
        assert!(
            !ObservedLifecycleEvents::is_sealable_host_served(&env),
            "a RemoteMesh event must never be sealed as host-served, even with a \
             served-model provenance block forwarded onto it"
        );
    }

    /// A non-terminal or non-2xx `RemoteMesh` event is not sealable as the
    /// requester's half -- same discipline as `is_sealable_host_served`.
    #[test]
    fn remote_mesh_requester_side_requires_terminal_and_2xx() {
        let effective = r#"{"dispatch_path":"remote_mesh","phase":"effective_request","model":"m","status":null}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(effective).expect("parse");
        assert!(!ObservedLifecycleEvents::is_sealable_requester_side(&env));

        let error =
            r#"{"dispatch_path":"remote_mesh","phase":"terminal","model":"m","status":502}"#;
        let env: OpenAiExchangeEnvelope = serde_json::from_str(error).expect("parse");
        assert!(!ObservedLifecycleEvents::is_sealable_requester_side(&env));
    }

    /// A locally-served (`typed_frontend`/`raw_proxy`) terminal event is never
    /// sealable as the requester's half -- regression guard so a local-served
    /// exchange keeps sealing exactly one capsule (via `is_sealable_host_served`
    /// or the plugin's own handler), never two.
    #[test]
    fn locally_served_dispatch_paths_are_never_requester_side_sealable() {
        for wire in [
            r#"{"dispatch_path":"typed_frontend","phase":"terminal","model":"m","status":200}"#,
            r#"{"dispatch_path":"raw_proxy","phase":"terminal","model":"m","status":200}"#,
        ] {
            let env: OpenAiExchangeEnvelope = serde_json::from_str(wire).expect("parse");
            assert!(!ObservedLifecycleEvents::is_sealable_requester_side(&env));
        }
    }

    #[test]
    fn role_for_dispatch_path_matches_the_2026_09_06_ruling() {
        assert_eq!(
            role_for_dispatch_path(&DispatchPath::RemoteMesh),
            "requested"
        );
        assert_eq!(
            role_for_dispatch_path(&DispatchPath::TypedFrontend),
            "served"
        );
        assert_eq!(role_for_dispatch_path(&DispatchPath::RawProxy), "served");
        assert_eq!(role_for_dispatch_path(&DispatchPath::Unknown), "unknown");
    }

    /// `latest_provenance_for_model` returns the MOST RECENT provenance for the
    /// asked-for model, ignores other models, and returns `None` for a model
    /// the host never reported provenance for (leaving capsule slots honest).
    #[test]
    fn latest_provenance_for_model_picks_most_recent_and_scopes_by_model() {
        let dir = std::env::temp_dir().join(format!("lc-test-{}", std::process::id()));
        let store = ObservedLifecycleEvents::open(&dir).expect("open store");

        let event = |model: &str, gpu: Option<&str>| OpenAiExchangeEnvelope {
            exchange_id: None,
            dispatch_path: DispatchPath::RawProxy,
            phase: Phase::Terminal,
            model: model.to_string(),
            status: Some(200),
            capsule_id: None,
            capsule_id_provenance: None,
            nonce: None,
            usage: None,
            request_digest: None,
            response_digest: None,
            tool_calls_digest: None,
            reasoning_digest: None,
            serving_provenance: gpu.map(|g| HostServingProvenance {
                served_by_node_id: Some("node-1".to_string()),
                hostname: None,
                quantization: None,
                architecture: None,
                context_length: None,
                parameter_size: None,
                layer_count: None,
                model_identity_hash: None,
                weights_digest: None,
                model_canonical_ref: None,
                model_revision: None,
                gpu: Some(g.to_string()),
                vram_bytes: None,
                is_soc: None,
                requested_by_node_id: None,
            }),
            twin_bracket_id: None,
            response_text_digest: None,
            exchange_bodies: None,
        };

        store.record(event("model-a", Some("gpu-old")));
        store.record(event("model-b", Some("gpu-other")));
        store.record(event("model-a", Some("gpu-new")));

        // Most recent for model-a wins.
        let a = store
            .latest_provenance_for_model("model-a")
            .expect("model-a provenance");
        assert_eq!(a.gpu.as_deref(), Some("gpu-new"));
        // Model scoping: model-b is untouched by model-a's events.
        let b = store
            .latest_provenance_for_model("model-b")
            .expect("model-b provenance");
        assert_eq!(b.gpu.as_deref(), Some("gpu-other"));
        // A model the host never reported provenance for -> None (honest).
        assert!(store.latest_provenance_for_model("model-unseen").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
