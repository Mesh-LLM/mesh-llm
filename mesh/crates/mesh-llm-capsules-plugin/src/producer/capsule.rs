//! The mesh record kinds, sealed through capsule-emit's public extension
//! points: an exchange record is capsule-emit's `seal_body` with the mesh
//! members (`x-mesh-poc-v1` and the runtime/model extension fields) in its
//! `compute_attestation_extensions` slot, and every record with no served
//! exchange (citing, settlement, routing, maintenance, adjudication) is
//! capsule-emit's `seal_local_record` under this node's mesh header. The
//! generic pieces (chain links, host binding, envelopes, references) are
//! capsule-emit's own, re-exported here.

use crate::producer::jcs::compute_capsule_id;
use capsule_emit_lib::capsule as emit;
pub use capsule_emit_lib::capsule::{
    attach_producer_envelope, capsule_reference, finish_seal, fresh_store_nonce, payload_bytes,
    ChainLink, HostBinding, LocalRecordHeader, SealError, CANONICALIZATION_ID,
    CHAIN_RELATION_FOLLOWS, CITATION_PURPOSE_COUNTERPARTY_HALF,
    CITATION_PURPOSE_COUNTERPARTY_INCLUSION, FORMAT_VERSION, REFERENCE_DIGEST_ALG,
    REFERENCE_TYPE_CAPSULE, SPEC_VERSION, STORE_NONCE_FIELD,
};
use serde_json::{json, Map, Value};

/// Granularity of a committed `latency_ms`, in milliseconds.
pub const LATENCY_BUCKET_MS: f64 = 100.0;

/// The `x-mesh-poc-v1.latency_ms` a record commits to: the measured latency
/// rounded UP to the next [`LATENCY_BUCKET_MS`], as an exact decimal string
/// (§5.1 forbids floats in digest-bearing fields). Latency is a timing side
/// channel like a timestamp (Evidence Layer -00 §12.1), so it is coarsened
/// here, when the bytes are produced. Rounding up keeps it an honest upper
/// bound: a 40 ms reply reads "100.000", never "0.000".
pub fn committed_latency_ms(measured_ms: f64) -> String {
    let bucketed = if measured_ms.is_finite() && measured_ms > 0.0 {
        (measured_ms / LATENCY_BUCKET_MS).ceil() * LATENCY_BUCKET_MS
    } else {
        0.0
    };
    format!("{bucketed:.3}")
}

/// Token accounting for one exchange, sourced verbatim from the OpenAI-shaped
/// response body's `usage` object (`openai-frontend`'s `Usage`:
/// `prompt_tokens`/`completion_tokens`/`total_tokens`). `None` when the served
/// response carried no `usage` (e.g. a stub or error body) — never fabricated.
#[derive(Clone)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// Everything we can HONESTLY attest about *what ran where* for one exchange.
/// Every field here is either (a) a real value the host/response actually
/// exposed, or (b) an explicit `"unknown"`/`null` for a fact the host's
/// `openai.exchange.v1` event and OpenAI-shaped response body genuinely do NOT
/// carry. It is deliberately truthful about the second class: we do not invent
/// a quantization or GPU string the mesh-llm host never told us.
///
/// What the host DOES expose to this plugin (verified against
/// `mesh-llm/…/plugin/openai_exchange.rs` on the `mesh1331-lifecycle-hooks` +
/// `feat/serving-provenance` branch and `openai-frontend`'s
/// `ChatCompletionResponse`):
///
/// - `model` (the served model NAME) — already in `model_attestation.model_id`
/// - `usage` (prompt/completion/total tokens) — in the response body
/// - `exchange_id` / a per-exchange correlation id, `nonce`, `nonce_source`,
///   `capsule_id` marker, `status`, `dispatch_path` — on the lifecycle event
/// - the emitting node's own id — this plugin's `node_id`
/// - **NEW (host `serving_provenance` block):** quantization, architecture,
///   context length, parameter size, layer count, model identity hash /
///   canonical ref / revision, serving GPU, VRAM bytes, SoC flag, and the
///   serving node id + hostname — read straight from the host's terminal
///   event when it carries them.
///
/// Fields the host STILL does not expose stay honest defaults
/// (`"unknown"`/`null`), and any host field this plugin has not yet observed
/// for the served model likewise stays at its default — never fabricated:
///
/// - `hardware_device` — the host carries an `is_soc` flag, not a cpu/cuda/
///   metal device enum, so this stays `null` unless a future event adds one.
/// - a serving-peer node id DISTINCT from the requester — in the single-node
///   PoC the host's `served_by_node_id` and the requester are the same node.
#[derive(Clone)]
pub struct ServingProvenance {
    /// The node that actually served the inference. Prefer the host event's
    /// `served_by_node_id` when observed; otherwise the emitting plugin's own
    /// `node_id` (single-node PoC).
    pub served_by_node_id: String,
    /// The raw `dispatch_path` wire value this record was derived from (e.g.
    /// `"remote_mesh"`, `"typed_frontend"`, `"raw_proxy"`, `"unknown"`), or
    /// `None` on a path that never received one (this plugin's own directly
    /// admitted `/v1` exchanges, which have no host envelope at all). Carried
    /// alongside `MeshPocV1::role` (a sibling, top-level field on the
    /// enclosing block -- see its doc comment for why it lives there and not
    /// here) even when they agree -- on a `role: "conflict"` this is one of
    /// the two raw facts a reader needs to see why, not just a resolved label.
    pub dispatch_path: Option<String>,
    /// The requesting party / client identity for this exchange. `"unknown"`
    /// when the caller supplied no identity beyond the (optional) client nonce.
    pub requesting_party: String,
    /// Stable per-exchange correlation id (the host's `exchange_id` /
    /// response `id` / `x-request-id` lineage), so this record can be tied back
    /// to the host's own terminal-event log for the same exchange.
    pub exchange_id: String,
    /// Model QUANTIZATION (e.g. "Q4_K_M"), read from the host's serving
    /// provenance when present; `"unknown"` when the host reported none
    /// (unquantized weights, or a host predating the block) — never guessed.
    pub quantization: String,
    /// Serving hardware, populated from the host's serving-provenance block
    /// when it carries them; `null` otherwise. `hardware_device` stays `null`
    /// (host carries `is_soc`, not a device enum) — never fabricated.
    pub hardware_gpu: Option<String>,
    pub hardware_vram_bytes: Option<u64>,
    pub hardware_device: Option<String>,
    /// Whether the serving host is a unified-memory SoC, from the host event.
    pub hardware_is_soc: Option<bool>,
    /// Serving host name, from the host event -- `Some` only when the
    /// operator opted in to sealing it; `None` omits the key entirely.
    pub hostname: Option<String>,
    /// Model IDENTITY / fidelity from the host serving-provenance block. Each
    /// is `None` when the host did not report it — never fabricated.
    pub architecture: Option<String>,
    pub context_length: Option<u32>,
    pub parameter_size: Option<String>,
    pub layer_count: Option<u32>,
    /// Content-addressed identity hash of the served model artifact (a digest
    /// of the model identity, NOT of the model *name* string). `None` when the
    /// host did not resolve one.
    pub model_identity_hash: Option<String>,
    /// SHA-256 of the served GGUF's file BYTES, from the host's load-time
    /// hash of the file it actually opened for serving. A different fact
    /// from `model_identity_hash` (a hash of a reference STRING, or absent
    /// for a bare local path) -- this never replaces it. `None` when the
    /// host has not resolved one (unreadable file, or no load-time hash
    /// computed yet for this model).
    pub weights_digest: Option<String>,
    pub model_canonical_ref: Option<String>,
    pub model_revision: Option<String>,
    /// Token accounting from the response `usage`, or `None` if absent.
    pub usage: Option<TokenUsage>,
    /// This record's position in the monotone per-`(self, counterparty)`
    /// sequence (history proposal §1 -- continuity is bilateral only), from
    /// [`crate::producer::sequence::SequenceCounterStore`]. `self` is this node
    /// (`served_by_node_id`); `counterparty` is `requesting_party`.
    pub seq: u64,
    /// The pair's previous `seq`, or `None` for this pair's first-ever
    /// record. A verifier walking the ledger checks `prev_seq` against the
    /// prior record's `seq` for the same pair -- see
    /// [`crate::producer::sequence::verify_pair_continuity`].
    pub prev_seq: Option<u64>,
    /// On a `role: "requested"` record: the capsule id the PEER asserted for
    /// its own (served-side) half of this exchange -- the lookup key an
    /// evidence request dereferences to populate the two-sided ledger's
    /// "theirs" column. Forwarded verbatim off the host's `RemoteMesh`
    /// terminal envelope (`mesh-llm-host-runtime`'s
    /// `CapsuleIdProvenance::PeerAsserted`) -- an unauthenticated,
    /// relay-injectable value observed on the peer's raw response header
    /// while ROUTING, never itself verified or signed by anyone here. `None`
    /// on every `role: "served"` record (there is no peer half to name) and
    /// on a `requested` record where no value was observed.
    pub peer_capsule_id: Option<String>,
    /// How `peer_capsule_id` was obtained -- the wire value of the host's
    /// `CapsuleIdProvenance` (`"peer_asserted"` today; `"unknown"` for a
    /// value this mirror predates). `None` exactly when `peer_capsule_id`
    /// is `None`. Graded self-attested/peer-asserted, never promoted to a
    /// verified or signed claim by this field's mere presence.
    pub peer_capsule_id_provenance: Option<String>,
    /// The id shared by BOTH halves of a client-marked twin pair, forwarded
    /// verbatim from the terminal `openai.exchange.v1` envelope's own
    /// `twin_bracket_id` (host-minted; this plugin never mints or derives
    /// one). `None` -- and then ABSENT from the sealed capsule, never a
    /// fabricated default -- on every exchange the envelope reports as not
    /// twinned (the overwhelming majority). Both rows of a real twin carry
    /// the identical string because both plugin instances forward the same
    /// host-minted value off their own terminal envelope for that exchange.
    pub twin_bracket_id: Option<String>,
    /// SHA-256 (lowercase hex) of the answer TEXT (`choices[0].message.content`,
    /// UTF-8; a stream's assembled text), forwarded from the host's terminal
    /// envelope. Unlike `response_digest`, which covers the whole body (ids,
    /// timestamps), two halves of a twin that gave the same answer carry the
    /// same value. `None` -- and then ABSENT, never fabricated -- when the
    /// host sent none.
    pub response_text_digest: Option<String>,
}

impl ServingProvenance {
    fn to_value(&self) -> Value {
        let usage = match &self.usage {
            Some(u) => json!({
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            }),
            None => Value::Null,
        };
        let mut value = json!({
            "served_by_node_id": self.served_by_node_id,
            "dispatch_path": self.dispatch_path,
            "requesting_party": self.requesting_party,
            "exchange_id": self.exchange_id,
            // Quantization: real value from the host event, or "unknown".
            "quantization": self.quantization,
            // Model identity / fidelity from the host serving-provenance block.
            "model": {
                "architecture": self.architecture,
                "context_length": self.context_length,
                "parameter_size": self.parameter_size,
                "layer_count": self.layer_count,
                "identity_hash": self.model_identity_hash,
                "weights_digest": self.weights_digest,
                "canonical_ref": self.model_canonical_ref,
                "revision": self.model_revision,
            },
            // Serving hardware from the host serving-provenance block.
            "hardware": {
                "gpu": self.hardware_gpu,
                "vram_bytes": self.hardware_vram_bytes,
                // Host carries is_soc, not a cpu/cuda/metal device enum.
                "device": self.hardware_device,
                "is_soc": self.hardware_is_soc,
            },
            "usage": usage,
            "seq": self.seq,
            "prev_seq": self.prev_seq,
            "peer_capsule_id": self.peer_capsule_id,
            "peer_capsule_id_provenance": self.peer_capsule_id_provenance,
        });
        // hostname: OMITTED entirely unless the operator opted in upstream --
        // the sealed body travels to every counterparty, and a machine name
        // is not theirs to receive by default.
        if let Some(hostname) = &self.hostname {
            value
                .as_object_mut()
                .expect("serving_provenance value is always a JSON object")
                .insert("hostname".into(), json!(hostname));
        }
        // twin_bracket_id: OMITTED entirely when the envelope carried none --
        // never a null placeholder -- so an untwinned row (the overwhelming
        // majority) never reads as a half-bracket.
        if let Some(twin_bracket_id) = &self.twin_bracket_id {
            value
                .as_object_mut()
                .expect("serving_provenance value is always a JSON object")
                .insert("twin_bracket_id".into(), json!(twin_bracket_id));
        }
        if let Some(digest) = &self.response_text_digest {
            value
                .as_object_mut()
                .expect("serving_provenance value is always a JSON object")
                .insert("response_text_digest".into(), json!(digest));
        }
        value
    }
}

/// The PoC-only `x-mesh-poc-v1` extension namespace inside
/// `model_attestation.compute_attestation` (see capsule-emit-mesh README
/// "Field mapping"). Not a registered spec field — rides inside
/// `compute_attestation`, which is documented as a free-form best-effort dict.
pub struct MeshPocV1 {
    pub client_nonce: String,
    pub client_nonce_source: String,
    /// SHA-256 of the model NAME string only — NOT a digest of the model
    /// weights or package. Named truthfully (`model_name_digest`) so it never
    /// implies weight/artifact binding it does not provide. The real
    /// package-digest path (issue #1233's `sha256(manifest+artifacts+abi)`)
    /// is the Python `model_identity.py` producer; the live Rust plugin only
    /// has the request's model name, so it attests exactly that and no more.
    pub model_name_digest: String,
    /// Detailed serving provenance: what ran where, for which exchange.
    pub serving_provenance: ServingProvenance,
    /// Which half of this exchange this record is: `"requested"` (this node
    /// routed the exchange to a peer, never served it), `"served"` (this node
    /// served it), `"conflict"` (the `dispatch_path`-derived expectation and
    /// the `served_by_node_id`-vs-self consistency check disagree -- both raw
    /// signals still ride alongside in `serving_provenance` so a reader never
    /// has to take this label's word for it), or `"unknown"` (an
    /// unrecognized `dispatch_path`). A reader takes this top-level field
    /// FIRST, as the authoritative signal. It deliberately lives HERE, a
    /// sibling of `serving_provenance`, not nested inside it: other producers
    /// use a `serving_provenance.role` key for an unrelated vocabulary
    /// ("provider"/"requester"), a different axis this field must never
    /// collide with.
    /// NEVER silently defaulted to `"served"` -- a missing/unrecognized
    /// signal must never become a claim; that silent default was the actual
    /// bug this field exists to close.
    pub role: String,
    /// Complementary vantage provenance:
    /// `Some("client_egress")` on the `RemoteMesh` dispatch path only -- this
    /// node observed the exchange at its own outbound/client-facing vantage
    /// point, not a serving vantage. `None` on every other path. A top-level
    /// sibling of `role`, where other producers place their own
    /// (provisional) `observation_point` field. Independent of `role`: one
    /// more honestly-scoped fact, not a restatement of it.
    pub observation_point: Option<String>,
    /// Generation parameters as exact decimal STRINGS (§5.1 forbids floats in
    /// digest-bearing fields) — e.g. `{"temperature": "0.7"}`.
    pub generation_parameters: Map<String, Value>,
    /// Build with [`committed_latency_ms`]: 100 ms buckets, never the raw
    /// measurement (a timing side channel).
    pub latency_ms: String,
    /// The runtime/binary attestation rung: a signed, `self_measured`
    /// reference to the serving binary the node actually runs, or `None` when
    /// the binary could not be measured (graceful degradation — the
    /// `binary_attestation` evidence slot is then recorded empty, never
    /// fabricated). See [`crate::producer::runtime_attest`] for the honesty grade: this is
    /// SELF-measured (the node hashed its own binary) and is only trustworthy up
    /// to an OS/TEE that independently measures it.
    pub binary_attestation: Option<crate::producer::runtime_attest::BinaryAttestation>,
    /// The `tee_measured` rung's slot, as [`crate::producer::tee_attest::slot_for`]
    /// built it for this record. `None` records the empty slot with
    /// [`crate::producer::tee_attest::names::NO_TEE`].
    pub tee_attestation: Option<Value>,
}

impl MeshPocV1 {
    fn to_value(&self) -> Value {
        json!({
            "client_nonce": self.client_nonce,
            "client_nonce_source": self.client_nonce_source,
            "model_name_digest": self.model_name_digest,
            "serving_provenance": self.serving_provenance.to_value(),
            "role": self.role,
            "observation_point": self.observation_point,
            "generation_parameters": self.generation_parameters,
            "latency_ms": self.latency_ms,
            // Typed reference fields, present-but-empty (issue #1233 step 4,
            // statistical fingerprint, is future work that upgrades this slot
            // without changing the record shape).
            //
            // `binary_attestation` is the runtime/binary-attestation
            // rung: a SIGNED, `self_measured` reference to the serving binary the
            // node runs. When the binary could not be measured (unresolvable
            // path / unreadable file) the slot degrades to the same empty shape
            // as the other future slots — recorded ABSENT, never fabricated.
            //
            // `tee_attestation` is the `tee_measured` rung's slot: a TDX
            // quote bound to this record when one was taken, else empty with
            // the reason (`crate::producer::tee_attest`).
            "evidence_refs": {
                "statistical_fingerprint": {"type": "statistical_fingerprint", "digest": null, "context": null},
                "tee_attestation": self.tee_attestation.clone().unwrap_or_else(|| {
                    crate::producer::tee_attest::empty(crate::producer::tee_attest::names::NO_TEE)
                }),
                "binary_attestation": match &self.binary_attestation {
                    Some(att) => att.to_value(),
                    // Honest empty slot: no binary was measured, so no
                    // measurement is claimed. `measurement_class` stays null so a
                    // reader never mistakes an unmeasured node for a self_measured
                    // one.
                    None => json!({
                        "type": "binary_attestation",
                        "measurement_class": null,
                        "digest": null,
                        "context": "binary not measured (path unresolvable or file unreadable); recorded absent, never fabricated",
                    }),
                },
            },
        })
    }
}

/// `epistemic_type` (the fabric's shared
/// record-header vocabulary convention) for the SAME `x-mesh-poc-v1.role` value this record carries.
/// `None` for `"conflict"`/`"unknown"`/anything else unrecognized -- an
/// ambiguous or unlabelable role must never be rounded up to either claim.
fn epistemic_type_for_role(role: &str) -> Option<&'static str> {
    match role {
        "served" => Some("producer_claim"),
        "requested" => Some("observed_event"),
        _ => None,
    }
}

/// The registered label for mesh-llm's own request-body digest, stated AS
/// DATA: the exact byte source is the HTTP request body as mesh-llm's host
/// runtime transmitted it (before any canonicalization on our side), the
/// hash is SHA-256, and the representation is unprefixed lowercase hex —
/// mesh-llm's own scheme, as mesh-llm defines it, never re-derived here.
pub const MESH_LLM_REQUEST_BODY_SHA256_V1: &str = "mesh-llm/request-body-sha256/v1";

/// A registered `host_binding.construction` label. Each entry states, AS
/// DATA, the exact byte source, hash, and representation the label pins —
/// this module does not normalize, re-encode, or re-derive those bytes. The
/// closed set exists so [`validate_host_binding`] can reject an unregistered
/// or missing label rather than accept any string.
pub const REGISTERED_HOST_BINDING_CONSTRUCTIONS: &[&str] = &[MESH_LLM_REQUEST_BODY_SHA256_V1];

/// The registered `host_binding.purpose` label for joining this record into
/// mesh-llm's own operational log.
pub const HOST_LOG_JOIN: &str = "host-log-join";

/// A registered `host_binding.purpose` label (mirrors the entry's
/// purpose-label list convention). `host-log-join` is the first member.
pub const REGISTERED_HOST_BINDING_PURPOSES: &[&str] = &[HOST_LOG_JOIN];

/// Structural validation ONLY: presence and shape of `host_binding`. This
/// function intentionally does NOT read or compare against
/// `agent_input_digest` (or anything else on the record) — see the
/// INDEPENDENCE RULE on [`HostBinding`]. It exists so a verifier can reject a
/// malformed group (missing/unregistered `construction`, a `null` group)
/// without ever inferring equality between two independent bindings.
pub fn validate_host_binding(capsule: &Value) -> Result<(), String> {
    let compute_attestation = capsule
        .get("model_attestation")
        .and_then(|m| m.get("compute_attestation"));
    let Some(hb) = compute_attestation.and_then(|ca| ca.get("host_binding")) else {
        // Absent key entirely -- OPTIONAL group, nothing to check.
        return Ok(());
    };
    if hb.is_null() {
        return Err("host_binding present but null -- must be absent, never null".to_string());
    }
    let obj = hb
        .as_object()
        .ok_or_else(|| "host_binding is not an object".to_string())?;
    let digest = obj
        .get("digest")
        .and_then(Value::as_str)
        .ok_or_else(|| "host_binding.digest missing or not a string".to_string())?;
    if digest.len() != 64
        || !digest
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    {
        return Err("host_binding.digest is not lowercase-hex-64".to_string());
    }
    let construction = obj
        .get("construction")
        .and_then(Value::as_str)
        .ok_or_else(|| "host_binding.construction missing or not a string".to_string())?;
    if !REGISTERED_HOST_BINDING_CONSTRUCTIONS.contains(&construction) {
        return Err(format!(
            "host_binding.construction {construction:?} is not a registered label"
        ));
    }
    let purpose = obj
        .get("purpose")
        .and_then(Value::as_str)
        .ok_or_else(|| "host_binding.purpose missing or not a string".to_string())?;
    if !REGISTERED_HOST_BINDING_PURPOSES.contains(&purpose) {
        return Err(format!(
            "host_binding.purpose {purpose:?} is not a registered label"
        ));
    }
    Ok(())
}

pub struct CapsuleInput {
    pub action_id: String,
    pub action_type: String,
    pub operator: String,
    pub developer: String,
    pub timestamp: String,
    pub domain: Option<String>,
    pub provenance: Option<String>,
    pub model_id: String,
    pub provider: String,
    pub agent_input_digest: String,
    /// JSON-DIGEST of the output the agent produced. `None` omits the member
    /// when the producer never saw the output body -- never a digest of
    /// something else that would read as one.
    pub agent_output_digest: Option<String>,
    /// OPTIONAL labeled sub-digest over the flattened `tool_calls` the model
    /// emitted for this exchange, mirroring the Python reference
    /// `capsule_ledger/conversation/exchange.py`'s `digest_conversation_exchange`
    /// (`tool_calls_digest = json_digest(tool_calls) if tool_calls else None`).
    /// `None` — and then ABSENT from the sealed capsule, never a fabricated
    /// digest over `[]` — when the model emitted no tool call, so a reader can
    /// never misread the record as asserting "zero tool calls". Rides inside
    /// `model_attestation.compute_attestation` alongside `agent_output_digest`.
    pub tool_calls_digest: Option<String>,
    /// OPTIONAL labeled sub-digest over the model's `reasoning_content` chunk(s)
    /// (same reference/shape as `tool_calls_digest`). `None` — and ABSENT from
    /// the capsule — when the model surfaced no reasoning (the honest null for a
    /// non-reasoning model like Llama-3.2), never fabricated.
    pub reasoning_digest: Option<String>,
    /// OPTIONAL reverse-direction composition binding — mesh-llm's own digest
    /// for this exchange, under its own construction. `None` — and ABSENT
    /// from the capsule, never null — when the host exposed no digest for
    /// this exchange. See [`HostBinding`] for the independence rule.
    pub host_binding: Option<HostBinding>,
    /// `compute_attestation.runtime` per the runtime/model extension draft:
    /// `{name, runtime_digest?, measurement_class?, platform_integrity?}`.
    /// Build with [`crate::producer::runtime_attest::BinaryAttestation::runtime_value`]
    /// when a real measurement exists, else `json!({"name": ...})` alone --
    /// never a fabricated digest/class for an unmeasured binary.
    pub runtime: Value,
    pub mesh_poc: MeshPocV1,
    pub effect_status: String,
    pub effect_type: String,
    /// `None` omits the member. Present only as a 64-hex JSON-DIGEST of the
    /// request body; never a sentinel string.
    pub effect_request_digest: Option<String>,
    /// `None` omits the member. `confirmed` REQUIRES it (a 64-hex digest of
    /// the effect's actual output); `dispatched`/`planned` REQUIRE it absent
    /// (AAC-05 §5.2). [`seal`] refuses any other combination.
    pub effect_response_digest: Option<String>,
    pub effect_attestation: String,
    pub disposition_decision: String,
    pub disposition_approver: String,
    pub disposition_human_disposed: bool,
    pub disposition_verdict_class: String,
    /// `None` for the first capsule in a chain — mirrors `emit()`'s
    /// `prior_capsule_id=None` (no `chain` key at all, not a null value).
    pub chain: Option<ChainLink>,
    /// The record's [`STORE_NONCE_FIELD`]: 64 lowercase hex from
    /// [`fresh_store_nonce`], drawn for this record alone. A caller-supplied
    /// field (not drawn inside [`seal`]) only so conformance tests can pin
    /// it; the live seal paths always pass a fresh one. [`seal`] refuses
    /// anything that is not 64 lowercase hex.
    pub store_nonce: String,
}

/// Build and seal a Capsule (mirrors `emit.emit()` + `parse.Capsule.seal()`):
/// returns the full capsule dict with `capsule_id` computed over the canonical
/// capsule form (§5.1) — standalone (no `chain` block; chaining is milestone 2).
pub fn seal(input: &CapsuleInput) -> Result<Value, SealError> {
    finish_seal(seal_body(input)?)
}

/// Everything [`seal`] commits to, before `capsule_id` is computed. Split out
/// so a record kind that adds blocks to an exchange record (the split-main
/// record, `crate::producer::stage::seal_split_main_record`) commits them under the
/// same id computation, never by editing a sealed capsule.
pub(crate) fn seal_body(input: &CapsuleInput) -> Result<Map<String, Value>, SealError> {
    emit::seal_body(&emit::CapsuleInput {
        action_id: input.action_id.clone(),
        action_type: input.action_type.clone(),
        operator: input.operator.clone(),
        developer: input.developer.clone(),
        timestamp: input.timestamp.clone(),
        domain: input.domain.clone(),
        provenance: input.provenance.clone(),
        model_id: input.model_id.clone(),
        provider: input.provider.clone(),
        agent_input_digest: input.agent_input_digest.clone(),
        agent_output_digest: input.agent_output_digest.clone(),
        tool_calls_digest: input.tool_calls_digest.clone(),
        reasoning_digest: input.reasoning_digest.clone(),
        host_binding: input.host_binding.clone(),
        runtime: input.runtime.clone(),
        compute_attestation_extensions: mesh_extensions(input),
        effect_status: input.effect_status.clone(),
        effect_type: input.effect_type.clone(),
        effect_request_digest: input.effect_request_digest.clone(),
        effect_response_digest: input.effect_response_digest.clone(),
        effect_attestation: input.effect_attestation.clone(),
        disposition_decision: input.disposition_decision.clone(),
        disposition_approver: input.disposition_approver.clone(),
        disposition_human_disposed: input.disposition_human_disposed,
        disposition_verdict_class: input.disposition_verdict_class.clone(),
        chain: input.chain.clone(),
        store_nonce: input.store_nonce.clone(),
    })
}

/// The mesh members of an exchange record's `compute_attestation`, in the
/// order they are committed: they ride capsule-emit's extension slot, after
/// `attestation_refs` and before the store nonce.
fn mesh_extensions(input: &CapsuleInput) -> Map<String, Value> {
    let mut extensions = Map::new();
    // model_attestation fields from the
    // runtime/model extension draft. agent_action_capsule.ModelAttestation
    // (the Python contracts dataclass this producer stays byte-compatible
    // with) has no top-level slot for model_revision/weights_digest/
    // quantization/decoding/source -- only model_id/provider/
    // compute_attestation -- so this producer's only free-form extension
    // point is compute_attestation itself; these fields ride there, same as
    // x-mesh-poc-v1. Promoting them to true model_attestation siblings needs
    // a matching agent_action_capsule change, out of this repo's scope.
    let sp = &input.mesh_poc.serving_provenance;
    let mut model_source = Map::new();
    if let Some(rev) = &sp.model_revision {
        extensions.insert("model_revision".into(), json!(rev));
        model_source.insert("model_revision".into(), json!("self_reported"));
    }
    if let Some(wd) = &sp.weights_digest {
        // A real SHA-256 of the served GGUF's file bytes (see
        // ServingProvenance::weights_digest's own doc comment) -- `computed`,
        // `scope: "file"` per the draft's definition, never `"tensors"`
        // (only achievable under `attested`).
        extensions.insert(
            "weights_digest".into(),
            json!({"digest_alg": "SHA-256", "digest": wd, "scope": "file"}),
        );
        model_source.insert("weights_digest".into(), json!("computed"));
    }
    if sp.quantization != "unknown" {
        extensions.insert("quantization".into(), json!(sp.quantization));
        model_source.insert("quantization".into(), json!("self_reported"));
    }
    let mut decoding = Map::new();
    for key in ["temperature", "seed"] {
        if let Some(v) = input.mesh_poc.generation_parameters.get(key) {
            decoding.insert(key.to_string(), v.clone());
        }
    }
    if !decoding.is_empty() {
        extensions.insert("decoding".into(), Value::Object(decoding));
        model_source.insert("decoding".into(), json!("self_reported"));
    }
    if !model_source.is_empty() {
        extensions.insert("source".into(), Value::Object(model_source));
    }

    // compute_attestation.hardware -- only the two fields this plugin's host
    // event actually carries (accelerator/memory_bytes); `platform` stays
    // absent (an `is_soc` flag is not a platform enum -- never guessed).
    let mut hardware = Map::new();
    let mut hardware_source = Map::new();
    if let Some(gpu) = &sp.hardware_gpu {
        hardware.insert("accelerator".into(), json!(gpu));
        hardware_source.insert("accelerator".into(), json!("os_reported"));
    }
    if let Some(vram) = sp.hardware_vram_bytes {
        hardware.insert("memory_bytes".into(), json!(vram));
        hardware_source.insert("memory_bytes".into(), json!("os_reported"));
    }
    if !hardware.is_empty() {
        hardware.insert("source".into(), Value::Object(hardware_source));
        extensions.insert("hardware".into(), Value::Object(hardware));
    }

    // compute_attestation.invocation -- this producer serves the model
    // itself, so requested/resolved are the same self-reported value; usage
    // is the real response `usage`, provider_reported.
    let mut invocation = Map::new();
    let mut invocation_source = Map::new();
    invocation.insert("requested_model_id".into(), json!(input.model_id));
    invocation_source.insert("requested_model_id".into(), json!("self_reported"));
    invocation.insert("resolved_model_id".into(), json!(input.model_id));
    invocation_source.insert("resolved_model_id".into(), json!("self_reported"));
    if let Some(usage) = &sp.usage {
        invocation.insert(
            "usage".into(),
            json!({
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }),
        );
        invocation_source.insert("usage".into(), json!("provider_reported"));
    }
    invocation.insert("source".into(), Value::Object(invocation_source));
    extensions.insert("invocation".into(), Value::Object(invocation));

    // An additive record-header field, a
    // top-level sibling of x-mesh-poc-v1 (never nested inside it --
    // epistemic_type is fabric vocabulary, not a PoC extension). Derived
    // from the SAME x-mesh-poc-v1.role signal, never re-guessed: "served"
    // is this node's own claim about what it served (`producer_claim`),
    // "requested" is this node's own observation of what it received
    // (`observed_event`). `role: "conflict"`/`"unknown"` never map to
    // either -- omitted, same "absent, never fabricated" discipline as
    // `role` itself (see MeshPocV1::role's own doc comment).
    if let Some(epistemic_type) = epistemic_type_for_role(&input.mesh_poc.role) {
        extensions.insert("epistemic_type".into(), json!(epistemic_type));
    }
    extensions.insert("x-mesh-poc-v1".into(), input.mesh_poc.to_value());
    extensions
}

// ---------------------------------------------------------------------------
// The LOCAL citing record for a
// received counterparty half.
// ---------------------------------------------------------------------------

/// The provenance facts a citing record carries about the received half it
/// cites -- the receiver-verified triple plus the structural `digest_match` grade
/// (the pane's CLOSED-gate input). Every field is a real value the receiver
/// established or the responder computed; nothing is fabricated.
pub struct ReceivedHalfProvenance<'a> {
    /// The foreign half's own `capsule_id` -- the citation target
    /// (`references[0].digest`) AND the key it is stored under in the
    /// held-artifact store `received-capsules.jsonl`.
    pub foreign_capsule_id: &'a str,
    /// The claimed sender's mesh peer id, as the receiver verified it against
    /// the announced key.
    pub received_from: &'a str,
    /// The carrier the half arrived over -- `"push"` for record-push.
    pub via: &'a str,
    /// When this node received the half (the receiver's `received_at`).
    pub received_at: &'a str,
    /// The receiver's recorded signature verdict -- always `true` here, since
    /// the receiver refuses (and this seal never runs) when the signature did not
    /// verify.
    pub signature_ok: bool,
    /// The structural `digest_match` grade the responder computed between the
    /// foreign half and this node's own correlated half, if one was found:
    /// `"verified"` / `"failed"` / `"present-unverified"`. `None` when this
    /// node held no correlated half to compare against (a received half with
    /// no local counterpart) -- honest absence, never a fabricated match.
    pub digest_match: Option<&'a str>,
    /// The foreign half's own `agent_input_digest`, carried so the pane's
    /// `exchange_key_for` correlator (digest-first) can group this citing
    /// record with our own half of the same exchange. `None` when the foreign
    /// half carried none.
    pub foreign_agent_input_digest: Option<&'a str>,
    /// The foreign half's own `agent_output_digest`, carried for the same
    /// correlation/`digest_match` reason. `None` when absent.
    pub foreign_agent_output_digest: Option<&'a str>,
}

/// Seal the LOCAL CITING record for a
/// received counterparty half: OUR OWN record of the RECEIVING event, chained
/// onto `chain_head` (`chain.relation = "follows"` -- the citation is the
/// `references[]` entry, never a relation value) with a
/// top-level `references` entry citing the foreign `capsule_id` by CPB typed
/// digest (`citation_purpose = "counterparty_half"`), and the inline producer
/// envelope attached under `signing_key`. Returns the fully sealed +
/// enveloped capsule; the caller appends it (and its detached `.cose`) through
/// `Ledger::append`, the SAME single-writer path every other local capsule
/// uses. The foreign body itself NEVER enters the chain -- it stays in the
/// held-artifact store `received-capsules.jsonl`. "cite, never mutate."
///
/// The reference participates in the citing record's `capsule_id` exactly as
/// the Python `served_request_join._with_references` does: `references` is part
/// of the body `compute_capsule_id` digests (see `jcs.rs`).
pub fn seal_citing_record(
    prov: &ReceivedHalfProvenance,
    chain_head: Option<&str>,
    signing_key: &ed25519_dalek::SigningKey,
) -> Result<Value, SealError> {
    let chain = chain_head.map(|parent| ChainLink {
        parent_capsule_id: parent.to_string(),
        relation: CHAIN_RELATION_FOLLOWS.to_string(),
    });

    // The citing record's own compute_attestation. It carries NO x-mesh-poc-v1
    // serving block (this node served nothing here -- it RECEIVED a half),
    // only the honest facts of the receiving event: the receiver's provenance
    // triple + verdict, the structural digest_match grade, and the foreign
    // half's own digests (so the pane's digest-first `exchange_key_for`
    // correlates this citing record with our own half of the same exchange).
    // Absent facts are omitted, never fabricated.
    let mut received_half = Map::new();
    received_half.insert("cited_capsule_id".into(), json!(prov.foreign_capsule_id));
    received_half.insert("received_from".into(), json!(prov.received_from));
    received_half.insert("via".into(), json!(prov.via));
    received_half.insert(
        "received_at".into(),
        json!(committed_time(prov.received_at)),
    );
    received_half.insert("signature_ok".into(), json!(prov.signature_ok));
    if let Some(dm) = prov.digest_match {
        received_half.insert("digest_match".into(), json!(dm));
    }
    if let Some(rd) = prov.foreign_agent_input_digest {
        received_half.insert("agent_input_digest".into(), json!(rd));
    }
    if let Some(rd) = prov.foreign_agent_output_digest {
        received_half.insert("agent_output_digest".into(), json!(rd));
    }

    seal_local_citation(
        format!(
            "mesh-poc/received-half-citation/{}",
            prov.foreign_capsule_id
        ),
        "n/a-received-half-citation",
        "received_half",
        received_half,
        json!([{
            "type": REFERENCE_TYPE_CAPSULE,
            "digest_alg": REFERENCE_DIGEST_ALG,
            "digest": prov.foreign_capsule_id,
            "citation_purpose": CITATION_PURPOSE_COUNTERPARTY_HALF,
        }]),
        chain,
        signing_key,
    )
}
// ---------------------------------------------------------------------------
// Settlement records: the payer node's sealed observation of one
// `payment.lifecycle.v1` event.
// ---------------------------------------------------------------------------

/// The host mesh channel a settlement record's observation came from. The
/// host broadcasts payer-side payment lifecycle events on it; a settlement
/// record names it verbatim in `x-mesh-settlement-v1.channel`.
pub const SETTLEMENT_CHANNEL: &str = "payment.lifecycle.v1";

/// The `model_attestation.compute_attestation` key a settlement record's
/// observation lives under. A reader identifies the settlement record KIND by
/// this key's presence; `Ledger` keys its event_ref dedup set on it.
pub const SETTLEMENT_EXTENSION_KEY: &str = "x-mesh-settlement-v1";

/// One observed payment lifecycle event, borrowed from the caller's parsed
/// event. Every value is copied verbatim into the sealed record by
/// [`seal_settlement_record`]; none is computed, summed or normalized here.
/// The caller is responsible for having checked the event (the plugin's
/// channel handler recomputes `event_ref` before it ever builds one of these).
pub struct SettlementObservation<'a> {
    /// The host's exchange id -- the join key to the exchange's own records.
    pub exchange_id: &'a str,
    /// The host's content reference for the event (lowercase-hex SHA-256 of
    /// the JCS event with `event_ref` blanked). Unique per distinct event, so
    /// it is also the dedup key.
    pub event_ref: &'a str,
    /// The host's digest of the public request terms.
    pub terms_digest: &'a str,
    /// The lifecycle phase wire string (e.g. `"input_settlement_observed"`).
    pub phase: &'a str,
    /// Who asserted this event's values: `"payer_asserted"`,
    /// `"provider_asserted"` or `"wallet_reported"`.
    pub source: &'a str,
    /// `Some("terminal")` for a wallet-reported settlement; `None` otherwise.
    pub settlement: Option<&'a str>,
    /// The payment segment (0 = input, non-zero = output) when the event has one.
    pub segment: Option<u32>,
    /// The Lightning payment hash when the event carries one.
    pub payment_hash: Option<&'a str>,
    /// The amount the event states, in millisatoshis -- a recorded value.
    pub amount_msat: u64,
    /// Which side of the exchange emitted the event: `"payer"` or
    /// `"provider"` (a host that names no role emitted the payer side only).
    pub observed_by: &'a str,
    /// The provider's delivered-token watermark, on a `delivered` event only.
    pub tokens: Option<u64>,
    /// A settlement's amounts as the wallet reported them: what it credited
    /// (provider side) and the fee it charged (payer: paid on top; provider:
    /// deducted). Copied when the host's event carries them and omitted
    /// otherwise, so a record sealed from an event without them is unchanged.
    pub credited_msat: Option<u64>,
    pub fee_msat: Option<u64>,
}

/// Seal a SETTLEMENT record: this payer node's own signed, chained record that
/// it observed one payment lifecycle event the host broadcast on
/// [`SETTLEMENT_CHANNEL`]. Sealed through the same local-record path as
/// [`seal_citing_record`] (same header fields, minute-granular timestamp, a
/// fresh store nonce, `chain.relation = "follows"` onto `chain_head`, inline
/// producer envelope) but with NO `references[]` entry -- it cites no capsule. The observation rides under
/// `model_attestation.compute_attestation["x-mesh-settlement-v1"]`; `null`
/// optional fields are omitted rather than written as `null`.
///
/// What the record claims, and what it does not. It is the payer node's
/// sealed observation of what its host broadcast, nothing more. The `source`
/// field says who asserted each value: the payer (`payer_asserted`), the
/// provider's invoice (`provider_asserted`), or the payer's wallet
/// (`wallet_reported`). It is not a claim that money moved beyond what a
/// `wallet_reported` event says, and never a claim about the provider's
/// books. The absence of settlement records for an exchange means "no payment
/// lifecycle observed" (a free exchange, payments off, or a failure before
/// authorization), never "unpaid".
///
/// Errors only when the observation cannot be canonicalized -- in practice an
/// `amount_msat` above 2^53-1, which [`compute_capsule_id`]'s JCS refuses.
pub fn seal_settlement_record(
    ev: &SettlementObservation,
    chain_head: Option<&str>,
    signing_key: &ed25519_dalek::SigningKey,
) -> Result<Value, SealError> {
    let chain = chain_head.map(|parent| ChainLink {
        parent_capsule_id: parent.to_string(),
        relation: CHAIN_RELATION_FOLLOWS.to_string(),
    });

    let mut observation = Map::new();
    observation.insert("v".into(), json!(1));
    observation.insert("observed_by".into(), json!(ev.observed_by));
    observation.insert("channel".into(), json!(SETTLEMENT_CHANNEL));
    observation.insert("exchange_id".into(), json!(ev.exchange_id));
    observation.insert("event_ref".into(), json!(ev.event_ref));
    observation.insert("terms_digest".into(), json!(ev.terms_digest));
    observation.insert("phase".into(), json!(ev.phase));
    observation.insert("source".into(), json!(ev.source));
    if let Some(settlement) = ev.settlement {
        observation.insert("settlement".into(), json!(settlement));
    }
    if let Some(segment) = ev.segment {
        observation.insert("segment".into(), json!(segment));
    }
    if let Some(payment_hash) = ev.payment_hash {
        observation.insert("payment_hash".into(), json!(payment_hash));
    }
    observation.insert("amount_msat".into(), json!(ev.amount_msat));
    if let Some(tokens) = ev.tokens {
        observation.insert("tokens".into(), json!(tokens));
    }
    if let Some(credited) = ev.credited_msat {
        observation.insert("credited_msat".into(), json!(credited));
    }
    if let Some(fee) = ev.fee_msat {
        observation.insert("fee_msat".into(), json!(fee));
    }

    // The same local-record path as every other record with no served
    // exchange: minute-granular committed time and a fresh store nonce
    // (Evidence Layer -00 §12.1), so a settlement record can't be confirmed
    // by guessing its content or timed to the millisecond.
    let mut blocks = Map::new();
    blocks.insert(SETTLEMENT_EXTENSION_KEY.into(), Value::Object(observation));
    seal_local_record(
        format!("mesh-poc/settlement/{}/{}", ev.exchange_id, ev.event_ref),
        "n/a-settlement-observation",
        blocks,
        None,
        chain,
        signing_key,
    )
}

/// A time a citing record commits to, truncated to the minute
/// (Evidence Layer -00 §12.1; see `crate::producer::timestamp`'s module doc). A value
/// that is not RFC3339 is carried as given -- never replaced with a time
/// nobody observed.
fn committed_time(raw: &str) -> String {
    crate::producer::timestamp::coarsen_to_minute(raw).unwrap_or_else(|| raw.to_string())
}

/// The shared body of every LOCAL citing record kind (a received half, a
/// received half's inclusion evidence): an `fyi` capsule with no served
/// exchange, carrying one `compute_attestation.<block_name>` block of the
/// receiving event's facts and the given top-level `references[]`, chained
/// onto this node's own head with `follows`, sealed, and enveloped.
fn seal_local_citation(
    action_id: String,
    model_id: &str,
    block_name: &str,
    block: Map<String, Value>,
    references: Value,
    chain: Option<ChainLink>,
    signing_key: &ed25519_dalek::SigningKey,
) -> Result<Value, SealError> {
    let mut blocks = Map::new();
    blocks.insert(block_name.into(), Value::Object(block));
    seal_local_record(
        action_id,
        model_id,
        blocks,
        Some(references),
        chain,
        signing_key,
    )
}

/// The producer identity every mesh local record carries in its header.
const LOCAL_RECORD_OPERATOR: &str = "capsule-emit-mesh-poc-rust";
/// The producer this plugin names in every record it seals: its own name and
/// version.
pub(crate) const LOCAL_RECORD_DEVELOPER: &str =
    concat!("capsule-emit-mesh/", env!("CARGO_PKG_VERSION"));
const LOCAL_RECORD_PROVIDER: &str = "mesh-llm";

/// The shared body of every LOCAL record with no served exchange (the citing
/// kinds above, and the split-stage records in `crate::producer::stage`):
/// capsule-emit's local record under this node's mesh header.
pub(crate) fn seal_local_record(
    action_id: String,
    model_id: &str,
    blocks: Map<String, Value>,
    references: Option<Value>,
    chain: Option<ChainLink>,
    signing_key: &ed25519_dalek::SigningKey,
) -> Result<Value, SealError> {
    let header = LocalRecordHeader {
        operator: LOCAL_RECORD_OPERATOR,
        developer: LOCAL_RECORD_DEVELOPER,
        provider: LOCAL_RECORD_PROVIDER,
        model_id,
    };
    emit::seal_local_record(&header, action_id, blocks, references, chain, signing_key)
}

/// Seal one settlement-records leg (draft-mih-agent-settlement-records-00)
/// under this node's mesh header: a local `fyi` record whose top-level
/// `settlement` member is `member`, chained onto this node's own head. The
/// member is checked by capsule-emit's `settlement` module first.
pub fn seal_settlement_leg(
    action_id: String,
    member: Value,
    references: Option<Value>,
    chain: Option<ChainLink>,
    signing_key: &ed25519_dalek::SigningKey,
) -> Result<Value, capsule_emit_lib::settlement::SealLegError> {
    let header = LocalRecordHeader {
        operator: LOCAL_RECORD_OPERATOR,
        developer: LOCAL_RECORD_DEVELOPER,
        provider: LOCAL_RECORD_PROVIDER,
        model_id: SETTLEMENT_LEG_MODEL_ID,
    };
    capsule_emit_lib::settlement::seal_leg(
        &header,
        action_id,
        member,
        references,
        chain,
        signing_key,
    )
}

/// A settlement leg is about a payment, not a model run.
const SETTLEMENT_LEG_MODEL_ID: &str = "n/a-settlement-leg";

// ---------------------------------------------------------------------------
// The LOCAL citing record for a received half's inclusion evidence
// (`counterparty_inclusion`, AAC-05 `citation_purpose` registry).
// ---------------------------------------------------------------------------

/// `references[].type` for a cited inclusion proof. PROVISIONAL: no CPB
/// Artifact Type is registered for a CLL inclusion proof yet; hard-coded
/// ahead of registration, like `REFERENCE_TYPE_CAPSULE`.
pub const REFERENCE_TYPE_INCLUSION_PROOF: &str = "cll-inclusion-proof";
/// `references[].type` for a cited checkpoint. PROVISIONAL, same reason; the
/// token is the CLL checkpoint wire kind.
pub const REFERENCE_TYPE_CHECKPOINT: &str = "cll-checkpoint";

/// The facts a `counterparty_inclusion` citing record carries: which held
/// half the evidence is about, where it came from, the leaf position and
/// covering size, and the digests of the two held artifacts it cites. Every
/// value is what the receiver verified and stored; nothing is fabricated.
pub struct InclusionCitation<'a> {
    /// The held half the proof is for -- already cited by this node's earlier
    /// `counterparty_half` record, never re-cited here.
    pub half_capsule_id: &'a str,
    pub received_from: &'a str,
    pub via: &'a str,
    pub received_at: &'a str,
    /// The half's leaf index in the counterparty's log.
    pub leaf_index: u64,
    /// The covering checkpoint's `mmr_size`.
    pub mmr_size: u64,
    /// SHA-256 of the checkpoint's signing body (`CheckpointRecord::digest`)
    /// -- the checkpoint's own identity, recomputable from the held copy.
    pub checkpoint_digest: &'a str,
    /// SHA-256 of the inclusion proof's sorted-key compact JSON (its JCS form:
    /// the proof holds only integers and strings).
    pub inclusion_proof_digest: &'a str,
}

/// Seal the LOCAL citing record for a received half's inclusion evidence:
/// one `references[]` entry per cited artifact (the inclusion proof, the
/// covering checkpoint), `citation_purpose = "counterparty_inclusion"`,
/// chained onto `chain_head` with `follows`. The earlier `counterparty_half`
/// record is never touched -- later evidence is a later record (AAC-05).
/// The cited artifacts stay in the held-artifact store, never in the chain.
pub fn seal_inclusion_citing_record(
    citation: &InclusionCitation,
    chain_head: Option<&str>,
    signing_key: &ed25519_dalek::SigningKey,
) -> Result<Value, SealError> {
    let chain = chain_head.map(|parent| ChainLink {
        parent_capsule_id: parent.to_string(),
        relation: CHAIN_RELATION_FOLLOWS.to_string(),
    });
    let mut block = Map::new();
    block.insert("half_capsule_id".into(), json!(citation.half_capsule_id));
    block.insert("received_from".into(), json!(citation.received_from));
    block.insert("via".into(), json!(citation.via));
    block.insert(
        "received_at".into(),
        json!(committed_time(citation.received_at)),
    );
    block.insert("leaf_index".into(), json!(citation.leaf_index));
    block.insert("mmr_size".into(), json!(citation.mmr_size));
    seal_local_citation(
        format!(
            "mesh-poc/counterparty-inclusion-citation/{}",
            citation.half_capsule_id
        ),
        "n/a-counterparty-inclusion-citation",
        "counterparty_inclusion",
        block,
        json!([
            {
                "type": REFERENCE_TYPE_INCLUSION_PROOF,
                "digest_alg": REFERENCE_DIGEST_ALG,
                "digest": citation.inclusion_proof_digest,
                "citation_purpose": CITATION_PURPOSE_COUNTERPARTY_INCLUSION,
            },
            {
                "type": REFERENCE_TYPE_CHECKPOINT,
                "digest_alg": REFERENCE_DIGEST_ALG,
                "digest": citation.checkpoint_digest,
                "citation_purpose": CITATION_PURPOSE_COUNTERPARTY_INCLUSION,
            },
        ]),
        chain,
        signing_key,
    )
}

/// Which way a local routing choice went: the person stopped routing to a
/// peer, or undid that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingChoiceChange {
    Block,
    Unblock,
}

impl RoutingChoiceChange {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::Unblock => "unblock",
        }
    }
}

/// The facts of one local routing choice. The peer id is an input to the
/// commitment only; it is never written into the sealed record.
pub struct LocalRoutingChoice<'a> {
    pub change: RoutingChoiceChange,
    /// The peer's endpoint id, as the host keys its local block store.
    pub peer_id: &'a str,
    /// Fresh per record, chosen and kept by the host's local block store;
    /// never sealed.
    pub salt: &'a [u8; 32],
    /// When a block lapses (RFC 3339), or `None` for "until I undo" and for
    /// every unblock.
    pub until: Option<&'a str>,
    /// Set when an operator rule, not a person at the console, asked for this
    /// block: the rule and the verdicts that met it.
    pub rule: Option<&'a RoutingRuleCitation<'a>>,
}

/// The operator rule behind a block, and the referee-signed verdicts that met
/// it. The record cites each verdict by `sha256(salt || verdict capsule id)`
/// under the same salt as the peer: the verdicts name the peer, so citing
/// them in clear would tell any asker who was blocked.
#[derive(Debug, Clone, Copy)]
pub struct RoutingRuleCitation<'a> {
    /// The rule's name, e.g. `stop_routing_after_contradictions`.
    pub rule: &'a str,
    /// N: contradictions that fire the rule.
    pub after: u32,
    /// D: the window, in days, they must fall in.
    pub window_days: u32,
    pub verdict_capsule_ids: &'a [String],
}

/// `sha256(salt || peer_id)`, lowercase hex, where `peer_id` is the endpoint id
/// as hex text (the bytes of the string, not the decoded key). Only a holder of
/// the salt (this node's local block store) can say which peer a routing-choice
/// record names. The host computes the same value to confirm a seal; both pin
/// one vector (`peer_commitment_matches_the_host_vector`).
pub fn peer_commitment(peer_id: &str, salt: &[u8; 32]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(salt);
    hasher.update(peer_id.as_bytes());
    hex::encode(hasher.finalize())
}

/// Seal the record of a local routing choice (block or unblock), chained onto
/// `chain_head` like every other local record.
///
/// The record names the peer only by [`peer_commitment`]. A `range` answer
/// from this node's evidence responder returns whole records to any asker, so a
/// record that named the peer in clear would tell the peer, and everyone else,
/// who was blocked; the commitment keeps "nobody else is told" true for the
/// record itself (`routing_choice_names_the_peer_only_by_commitment`).
pub fn seal_local_routing_choice(
    choice: &LocalRoutingChoice<'_>,
    chain_head: Option<&str>,
    signing_key: &ed25519_dalek::SigningKey,
) -> Result<Value, SealError> {
    let chain = chain_head.map(|parent| ChainLink {
        parent_capsule_id: parent.to_string(),
        relation: CHAIN_RELATION_FOLLOWS.to_string(),
    });
    let commitment = peer_commitment(choice.peer_id, choice.salt);

    let mut body = Map::new();
    body.insert("spec_version".into(), json!(SPEC_VERSION));
    body.insert("format_version".into(), json!(FORMAT_VERSION));
    body.insert("canonicalization_id".into(), json!(CANONICALIZATION_ID));
    body.insert(
        "action_id".into(),
        json!(format!(
            "mesh-poc/local-routing-choice/{}/{commitment}",
            choice.change.as_str()
        )),
    );
    body.insert("action_type".into(), json!("decide"));
    body.insert("operator".into(), json!("capsule-emit-mesh-poc-rust"));
    body.insert("developer".into(), json!(LOCAL_RECORD_DEVELOPER));
    // The same seal path as every other record (Evidence Layer -00 §12.1):
    // committed times truncated to the minute, and a fresh store nonce.
    body.insert(
        "timestamp".into(),
        json!(crate::producer::timestamp::utc_now_minute()),
    );
    body.insert("domain".into(), json!("action"));
    body.insert("provenance".into(), json!("collector"));
    body.insert(
        "model_attestation".into(),
        json!({
            "model_id": "n/a-local-routing-choice",
            "provider": "mesh-llm",
            "compute_attestation": {
                "local_routing_choice": {
                    "change": choice.change.as_str(),
                    "peer_commitment": {"alg": "SHA-256", "digest": commitment},
                    "requested_via": "host_local_api",
                    "until": choice.until.map(committed_time),
                    "scope": "this_node_only",
                    "rule": choice.rule.map(|rule| json!({
                        "rule": rule.rule,
                        "after": rule.after,
                        "window_days": rule.window_days,
                        "verdict_commitments": rule
                            .verdict_capsule_ids
                            .iter()
                            .map(|id| json!({"alg": "SHA-256", "digest": peer_commitment(id, choice.salt)}))
                            .collect::<Vec<_>>(),
                    })),
                },
                STORE_NONCE_FIELD: fresh_store_nonce(),
            },
        }),
    );
    body.insert(
        "assurance".into(),
        json!({
            "attestation_mode": "self_attested",
            // The record is the decision; the host's router enforces it.
            "effect_mode": "not_applicable",
            "ledger_mode": if chain.is_some() { "chained" } else { "standalone" },
        }),
    );
    // The host's local operator API asked for this. A local API call does not
    // prove a person made it, so the record does not claim one did.
    body.insert(
        "disposition".into(),
        json!({
            "decision": "accept",
            "approver": "policy",
            "human_disposed": false,
            "verdict_class": "executed",
        }),
    );
    if let Some(chain) = &chain {
        body.insert("chain".into(), chain.to_value());
    }

    let capsule_id = compute_capsule_id(&Value::Object(body.clone()))?;
    let mut sealed = Map::new();
    sealed.insert("capsule_id".into(), json!(capsule_id));
    for (k, v) in body {
        sealed.entry(k).or_insert(v);
    }
    let mut capsule = Value::Object(sealed);
    attach_producer_envelope(&mut capsule, signing_key)
        .expect("routing-choice record always carries a hex capsule_id");
    Ok(capsule)
}

// ---------------------------------------------------------------------------
// The OWNER-MAINTENANCE record:
// a cleanup the node's owner ran on their own records, sealed onto the chain
// so the cleanup is itself on the record.
// ---------------------------------------------------------------------------

/// `compute_attestation` key an owner-maintenance record carries its facts
/// under. PROVISIONAL (not a registered AAC field): a reader identifies this
/// record KIND by the presence of this block alone.
pub const OWNER_MAINTENANCE_BLOCK: &str = "owner_maintenance";

/// `references[].citation_purpose` for a new history's first record citing the
/// last record of the history it replaced. PROVISIONAL, same status as
/// [`CITATION_PURPOSE_COUNTERPARTY_HALF`]: flagged ahead of registry promotion.
pub const CITATION_PURPOSE_PRIOR_HISTORY: &str = "prior_history";

/// One owner cleanup to seal. `kind` names the action (`stored_text_deleted`,
/// `index_rebuilt`, `history_closing`, `history_started`); `facts` are the
/// action's own measured facts (counts, ranges, digests) -- never a path, and
/// never text the cleanup removed. `prior_history_head` is set only on a new
/// history's first record and becomes its single `references[]` entry.
pub struct OwnerMaintenance<'a> {
    pub kind: &'a str,
    pub facts: Map<String, Value>,
    pub prior_history_head: Option<&'a str>,
}

/// Seal an owner-maintenance record chained onto `chain_head` (`"follows"`,
/// like every other local record; `None` only for the first record of a new
/// history), with the inline producer envelope attached. Same body shape as
/// [`seal_citing_record`]: a capsule KIND with no served exchange, so
/// `effect_mode` is `not_applicable` and there is no `x-mesh-poc-v1` block.
/// The caller appends it through `Ledger::append`, the single-writer path.
pub fn seal_owner_maintenance_record(
    action: &OwnerMaintenance,
    chain_head: Option<&str>,
    signing_key: &ed25519_dalek::SigningKey,
) -> Result<Value, SealError> {
    let chain = chain_head.map(|parent| ChainLink {
        parent_capsule_id: parent.to_string(),
        relation: CHAIN_RELATION_FOLLOWS.to_string(),
    });
    // The same seal path as every other record (Evidence Layer -00 §12.1):
    // committed times truncated to the minute, and a fresh store nonce.
    let timestamp = crate::producer::timestamp::utc_now_minute();

    let mut block = Map::new();
    block.insert("kind".into(), json!(action.kind));
    for (k, v) in &action.facts {
        block.entry(k.clone()).or_insert_with(|| v.clone());
    }

    let mut body = Map::new();
    body.insert("spec_version".into(), json!(SPEC_VERSION));
    body.insert("format_version".into(), json!(FORMAT_VERSION));
    body.insert("canonicalization_id".into(), json!(CANONICALIZATION_ID));
    body.insert(
        "action_id".into(),
        json!(format!(
            "mesh-poc/owner-maintenance/{}/{timestamp}",
            action.kind
        )),
    );
    body.insert("action_type".into(), json!("fyi"));
    body.insert("operator".into(), json!("capsule-emit-mesh-poc-rust"));
    body.insert("developer".into(), json!(LOCAL_RECORD_DEVELOPER));
    body.insert("timestamp".into(), json!(timestamp));
    body.insert("domain".into(), json!("action"));
    body.insert("provenance".into(), json!("collector"));

    let mut compute_attestation = Map::new();
    compute_attestation.insert(OWNER_MAINTENANCE_BLOCK.into(), Value::Object(block));
    compute_attestation.insert(STORE_NONCE_FIELD.into(), json!(fresh_store_nonce()));
    body.insert(
        "model_attestation".into(),
        json!({
            "model_id": "n/a-owner-maintenance",
            "provider": "mesh-llm",
            "compute_attestation": Value::Object(compute_attestation),
        }),
    );
    body.insert(
        "assurance".into(),
        json!({
            "attestation_mode": "self_attested",
            "effect_mode": "not_applicable",
            "ledger_mode": if chain.is_some() { "chained" } else { "standalone" },
        }),
    );
    body.insert(
        "disposition".into(),
        json!({
            "decision": "accept",
            "approver": "owner",
            "human_disposed": true,
            "verdict_class": "executed",
        }),
    );
    if let Some(chain) = &chain {
        body.insert("chain".into(), chain.to_value());
    }
    if let Some(prior) = action.prior_history_head {
        body.insert(
            "references".into(),
            json!([{
                "type": REFERENCE_TYPE_CAPSULE,
                "digest_alg": REFERENCE_DIGEST_ALG,
                "digest": prior,
                "citation_purpose": CITATION_PURPOSE_PRIOR_HISTORY,
            }]),
        );
    }

    let capsule_id = compute_capsule_id(&Value::Object(body.clone()))?;
    let mut sealed = Map::new();
    sealed.insert("capsule_id".into(), json!(capsule_id));
    for (k, v) in body {
        sealed.entry(k).or_insert(v);
    }
    let mut capsule = Value::Object(sealed);
    attach_producer_envelope(&mut capsule, signing_key)
        .expect("owner-maintenance record always carries a hex capsule_id");
    Ok(capsule)
}

// ---------------------------------------------------------------------------
// Twin adjudication: the referee's and the judged nodes' own records of a
// signed verdict. The verdict itself is held beside the ledger, never on it;
// these records cite it by its capsule id.
// ---------------------------------------------------------------------------

/// `references[].citation_purpose` for a record's reference to a referee's
/// signed verdict.
pub const CITATION_PURPOSE_ADJUDICATION_VERDICT: &str = "adjudication_verdict";
/// `references[].citation_purpose` for the referee's own served record of the
/// answer that decided the verdict.
pub const CITATION_PURPOSE_REFEREE_ANSWER: &str = "referee_answer";
/// `references[].citation_purpose` for a twin half a verdict judged.
pub const CITATION_PURPOSE_ADJUDICATED_HALF: &str = "adjudicated_half";
/// The block a referee's record of a verdict it issued carries.
pub const ADJUDICATION_ISSUED_BLOCK: &str = "adjudication_issued";
/// The block a node's record of a verdict delivered to it carries.
pub const ADJUDICATION_RECEIVED_BLOCK: &str = "adjudication_received";
/// The block a courier's record of a refused delivery carries.
pub const ADJUDICATION_ACK_REFUSED_BLOCK: &str = "adjudication_ack_refused";

/// The facts of one verdict, as the referee signed them.
pub struct VerdictFacts<'a> {
    /// `"corroborated"` or `"contradicted:<node id>"`.
    pub verdict: &'a str,
    pub verdict_capsule_id: &'a str,
    pub referee_node_id: &'a str,
    /// The two judged halves' capsule ids, and the nodes that served them.
    pub halves: [&'a str; 2],
    pub half_node_ids: [&'a str; 2],
    pub twin_bracket_id: Option<&'a str>,
    /// The model the twins served, as the verdict seals it: counts and the
    /// bar window are kept per model.
    pub model_hash: Option<&'a str>,
}

fn verdict_block(facts: &VerdictFacts) -> Map<String, Value> {
    let mut block = Map::new();
    block.insert("verdict".into(), json!(facts.verdict));
    block.insert("verdict_capsule_id".into(), json!(facts.verdict_capsule_id));
    block.insert("referee_node_id".into(), json!(facts.referee_node_id));
    block.insert("halves".into(), json!(facts.halves));
    block.insert("half_node_ids".into(), json!(facts.half_node_ids));
    if let Some(bracket) = facts.twin_bracket_id {
        block.insert("twin_bracket_id".into(), json!(bracket));
    }
    if let Some(model) = facts.model_hash {
        block.insert("model_hash".into(), json!(model));
    }
    block
}

/// Seal the REFEREE's own record of a verdict it issued: it cites the
/// verdict, its own served record of the answer that decided it (when one
/// did: twins that agree need no referee answer), and both judged halves.
pub fn seal_adjudication_issued_record(
    facts: &VerdictFacts,
    referee_capsule_id: Option<&str>,
    issued_at: &str,
    chain_head: Option<&str>,
    signing_key: &ed25519_dalek::SigningKey,
) -> Result<Value, SealError> {
    let chain = chain_head.map(|parent| ChainLink {
        parent_capsule_id: parent.to_string(),
        relation: CHAIN_RELATION_FOLLOWS.to_string(),
    });
    let mut block = verdict_block(facts);
    if let Some(referee) = referee_capsule_id {
        block.insert("referee_capsule_id".into(), json!(referee));
    }
    block.insert("issued_at".into(), json!(committed_time(issued_at)));
    let mut references = vec![capsule_reference(
        facts.verdict_capsule_id,
        CITATION_PURPOSE_ADJUDICATION_VERDICT,
    )];
    if let Some(referee) = referee_capsule_id {
        references.push(capsule_reference(referee, CITATION_PURPOSE_REFEREE_ANSWER));
    }
    for half in facts.halves {
        references.push(capsule_reference(half, CITATION_PURPOSE_ADJUDICATED_HALF));
    }
    seal_local_citation(
        format!("mesh-poc/adjudication-issued/{}", facts.verdict_capsule_id),
        "n/a-adjudication-issued",
        ADJUDICATION_ISSUED_BLOCK,
        block,
        Value::Array(references),
        chain,
        signing_key,
    )
}

/// Seal a node's own record of a verdict delivered to it: it cites the
/// verdict, and names the half this node holds that the verdict is about.
pub fn seal_adjudication_received_record(
    facts: &VerdictFacts,
    held_half_capsule_id: &str,
    received_from: &str,
    received_at: &str,
    chain_head: Option<&str>,
    signing_key: &ed25519_dalek::SigningKey,
) -> Result<Value, SealError> {
    let chain = chain_head.map(|parent| ChainLink {
        parent_capsule_id: parent.to_string(),
        relation: CHAIN_RELATION_FOLLOWS.to_string(),
    });
    let mut block = verdict_block(facts);
    block.insert("held_half_capsule_id".into(), json!(held_half_capsule_id));
    block.insert("received_from".into(), json!(received_from));
    block.insert("received_at".into(), json!(committed_time(received_at)));
    seal_local_citation(
        format!(
            "mesh-poc/adjudication-received/{}",
            facts.verdict_capsule_id
        ),
        "n/a-adjudication-received",
        ADJUDICATION_RECEIVED_BLOCK,
        block,
        json!([capsule_reference(
            facts.verdict_capsule_id,
            CITATION_PURPOSE_ADJUDICATION_VERDICT
        )]),
        chain,
        signing_key,
    )
}

/// `chain.relation` of a referee's verdict record, chained to the pair's first
/// half: the registered non-terminal `confirms` (it observes that half; the
/// half's own state remains), as capsule-emit's adjudication writes. What
/// marks the record as a verdict is its `compute_attestation` (the
/// `adjudication` epistemic type and block), not the relation: see
/// [`is_adjudication`].
pub const CHAIN_RELATION_ADJUDICATES: &str = "confirms";
/// The unregistered token earlier releases wrote (the AAC registry lists it as
/// a legacy alias of `confirms`). Read by [`is_adjudication`], never written.
pub const CHAIN_RELATION_ADJUDICATES_LEGACY: &str = "adjudicates";
/// The registered `chain.relation` values (AAC REGISTRY section 6). Every
/// relation this crate writes is one of them.
pub const REGISTERED_CHAIN_RELATIONS: [&str; 5] = [
    "follows",
    "confirms",
    "supersedes",
    "epoch_opens",
    "duplicates",
];
/// `compute_attestation.epistemic_type` of a verdict record: the referee's own
/// judgment over the two halves.
pub const EPISTEMIC_TYPE_ADJUDICATION: &str = "adjudication";
/// The ruling block a referee's verdict record carries.
pub const ADJUDICATION_BLOCK: &str = "adjudication";

/// Whether `record` is a referee's verdict record: its `compute_attestation`
/// carries the `adjudication` epistemic type and an `adjudication` block, or it
/// is a record from an earlier release with the legacy `chain.relation ==
/// "adjudicates"`. The relation alone no longer marks one: verdicts now write
/// the registered `confirms`, which other records write too.
pub fn is_adjudication(record: &Value) -> bool {
    if record.pointer("/chain/relation").and_then(Value::as_str)
        == Some(CHAIN_RELATION_ADJUDICATES_LEGACY)
    {
        return true;
    }
    let Some(ca) = record.pointer("/model_attestation/compute_attestation") else {
        return false;
    };
    ca.get("epistemic_type").and_then(Value::as_str) == Some(EPISTEMIC_TYPE_ADJUDICATION)
        && ca.get(ADJUDICATION_BLOCK).is_some_and(Value::is_object)
}

/// Seal a REFEREE's signed verdict: the ruling `block`, chained to the pair's
/// first half (`confirms`), signed with the referee's own key. It is held
/// beside the ledger and delivered as it is, never written into a chain;
/// each chain cites it by its capsule id.
pub fn seal_verdict_record(
    block: Map<String, Value>,
    half_a_capsule_id: &str,
    signing_key: &ed25519_dalek::SigningKey,
) -> Result<Value, SealError> {
    let timestamp = crate::producer::timestamp::utc_now_minute();
    let mut body = Map::new();
    body.insert("spec_version".into(), json!(SPEC_VERSION));
    body.insert("format_version".into(), json!(FORMAT_VERSION));
    body.insert("canonicalization_id".into(), json!(CANONICALIZATION_ID));
    body.insert(
        "action_id".into(),
        json!(format!("mesh-poc/adjudicate/{half_a_capsule_id}")),
    );
    body.insert("action_type".into(), json!("decide"));
    body.insert("operator".into(), json!(LOCAL_RECORD_OPERATOR));
    body.insert("developer".into(), json!(LOCAL_RECORD_DEVELOPER));
    body.insert("timestamp".into(), json!(timestamp));
    body.insert("domain".into(), json!("action"));
    body.insert("provenance".into(), json!("referee"));

    let mut compute_attestation = Map::new();
    compute_attestation.insert("epistemic_type".into(), json!(EPISTEMIC_TYPE_ADJUDICATION));
    compute_attestation.insert(ADJUDICATION_BLOCK.into(), Value::Object(block));
    compute_attestation.insert(STORE_NONCE_FIELD.into(), json!(fresh_store_nonce()));
    body.insert(
        "model_attestation".into(),
        json!({
            "model_id": "n/a-adjudication",
            "provider": LOCAL_RECORD_PROVIDER,
            "compute_attestation": Value::Object(compute_attestation),
        }),
    );
    body.insert(
        "assurance".into(),
        json!({
            "attestation_mode": "self_attested",
            "effect_mode": "not_applicable",
            "ledger_mode": "chained",
        }),
    );
    body.insert(
        "disposition".into(),
        json!({
            "decision": "accept",
            "approver": "policy",
            "human_disposed": false,
            "verdict_class": "assessed",
        }),
    );
    body.insert(
        "chain".into(),
        ChainLink {
            parent_capsule_id: half_a_capsule_id.to_string(),
            relation: CHAIN_RELATION_ADJUDICATES.to_string(),
        }
        .to_value(),
    );
    let capsule_id = compute_capsule_id(&Value::Object(body.clone()))?;
    let mut sealed = Map::new();
    sealed.insert("capsule_id".into(), json!(capsule_id));
    for (k, v) in body {
        sealed.entry(k).or_insert(v);
    }
    let mut capsule = Value::Object(sealed);
    attach_producer_envelope(&mut capsule, signing_key)
        .expect("a verdict record always carries a hex capsule_id");
    Ok(capsule)
}

/// A delivery of a verdict that its receiver refused, with a signed refusal.
pub struct RefusedDelivery<'a> {
    pub verdict_capsule_id: &'a str,
    pub refused_by: &'a str,
    pub reason: &'a str,
    /// SHA-256 of the refusal as received (compact JSON).
    pub refusal_digest: &'a str,
    /// The key the refusal says signed it; not checked here.
    pub refusal_key_id: Option<&'a str>,
    pub refused_at: &'a str,
}

/// Seal the COURIER's own record of a refused delivery: the receiver's chain
/// shows nothing, so this node's chain holds the decline. It cites the verdict
/// and names who refused and why.
pub fn seal_adjudication_ack_refused_record(
    refused: &RefusedDelivery,
    chain_head: Option<&str>,
    signing_key: &ed25519_dalek::SigningKey,
) -> Result<Value, SealError> {
    let chain = chain_head.map(|parent| ChainLink {
        parent_capsule_id: parent.to_string(),
        relation: CHAIN_RELATION_FOLLOWS.to_string(),
    });
    let mut block = Map::new();
    block.insert(
        "verdict_capsule_id".into(),
        json!(refused.verdict_capsule_id),
    );
    block.insert("refused_by".into(), json!(refused.refused_by));
    block.insert("reason".into(), json!(refused.reason));
    block.insert("refusal_digest".into(), json!(refused.refusal_digest));
    if let Some(key) = refused.refusal_key_id {
        block.insert("refusal_key_id".into(), json!(key));
    }
    block.insert(
        "refused_at".into(),
        json!(committed_time(refused.refused_at)),
    );
    seal_local_citation(
        format!(
            "mesh-poc/adjudication-ack-refused/{}/{}",
            refused.verdict_capsule_id, refused.refused_by
        ),
        "n/a-adjudication-ack-refused",
        ADJUDICATION_ACK_REFUSED_BLOCK,
        block,
        json!([capsule_reference(
            refused.verdict_capsule_id,
            CITATION_PURPOSE_ADJUDICATION_VERDICT
        )]),
        chain,
        signing_key,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_verdict_record_writes_the_registered_confirms_and_reads_as_an_adjudication() {
        let key = ed25519_dalek::SigningKey::from_bytes(&[7u8; 32]);
        let mut block = Map::new();
        block.insert("verdict".into(), json!("corroborated"));
        let record = seal_verdict_record(block, &"a".repeat(64), &key).unwrap();
        assert_eq!(record["chain"]["relation"], json!("confirms"));
        assert!(is_adjudication(&record));
    }

    #[test]
    fn is_adjudication_reads_legacy_records_and_not_a_plain_confirms() {
        let legacy = json!({"chain": {"relation": "adjudicates"}});
        assert!(is_adjudication(&legacy));
        let plain = json!({"chain": {"relation": "confirms"},
            "model_attestation": {"compute_attestation": {"epistemic_type": "producer_claim"}}});
        assert!(!is_adjudication(&plain));
        let no_block = json!({"chain": {"relation": "confirms"},
            "model_attestation": {"compute_attestation": {"epistemic_type": "adjudication"}}});
        assert!(!is_adjudication(&no_block));
    }

    #[test]
    fn every_relation_this_crate_writes_is_registered() {
        for relation in [CHAIN_RELATION_FOLLOWS, CHAIN_RELATION_ADJUDICATES] {
            assert!(
                REGISTERED_CHAIN_RELATIONS.contains(&relation),
                "{relation} is not registered"
            );
        }
        assert!(!REGISTERED_CHAIN_RELATIONS.contains(&CHAIN_RELATION_ADJUDICATES_LEGACY));
    }

    #[test]
    fn committed_latency_rounds_up_to_the_bucket() {
        assert_eq!(committed_latency_ms(0.0), "0.000");
        assert_eq!(committed_latency_ms(40.2), "100.000");
        assert_eq!(committed_latency_ms(100.0), "100.000");
        assert_eq!(committed_latency_ms(1234.567), "1300.000");
        assert_eq!(committed_latency_ms(f64::NAN), "0.000");
    }

    fn base_input(chain: Option<ChainLink>) -> CapsuleInput {
        let mut generation_parameters = Map::new();
        generation_parameters.insert("temperature".into(), json!("0.7"));
        CapsuleInput {
            action_id: "mesh-poc/chain-test/1".to_string(),
            action_type: "decide".to_string(),
            operator: "op".to_string(),
            developer: "dev".to_string(),
            timestamp: "2026-08-23T00:00:00Z".to_string(),
            domain: Some("action".to_string()),
            provenance: Some("collector".to_string()),
            model_id: "m".to_string(),
            provider: "p".to_string(),
            agent_input_digest: "a".repeat(64),
            agent_output_digest: Some("b".repeat(64)),
            tool_calls_digest: None,
            reasoning_digest: None,
            host_binding: None,
            runtime: json!({"name": "runtime"}),
            mesh_poc: MeshPocV1 {
                client_nonce: "c".repeat(32),
                client_nonce_source: "client_supplied".to_string(),
                model_name_digest: "d".repeat(64),
                serving_provenance: ServingProvenance {
                    served_by_node_id: "node-under-test".to_string(),
                    dispatch_path: None,
                    requesting_party: "client-under-test".to_string(),
                    exchange_id: "exch-under-test".to_string(),
                    quantization: "Q4_K_M".to_string(),
                    hardware_gpu: Some("Apple M3 Max".to_string()),
                    hardware_vram_bytes: Some(38_654_705_664),
                    hardware_device: None,
                    hardware_is_soc: Some(true),
                    hostname: Some("host-under-test".to_string()),
                    architecture: Some("llama".to_string()),
                    context_length: Some(8192),
                    parameter_size: Some("7B".to_string()),
                    layer_count: Some(32),
                    model_identity_hash: Some("a".repeat(64)),
                    weights_digest: Some("9".repeat(64)),
                    model_canonical_ref: Some("repo@rev/model.gguf".to_string()),
                    model_revision: Some("rev".to_string()),
                    usage: Some(TokenUsage {
                        prompt_tokens: 11,
                        completion_tokens: 22,
                        total_tokens: 33,
                    }),
                    seq: 1,
                    prev_seq: None,
                    peer_capsule_id: None,
                    peer_capsule_id_provenance: None,
                    twin_bracket_id: None,
                    response_text_digest: None,
                },
                role: "served".to_string(),
                observation_point: None,
                generation_parameters,
                latency_ms: "1.0".to_string(),
                binary_attestation: None,
                tee_attestation: None,
            },
            effect_status: "confirmed".to_string(),
            effect_type: "inference_completion".to_string(),
            effect_request_digest: Some("a".repeat(64)),
            effect_response_digest: Some("b".repeat(64)),
            effect_attestation: "gate_executed".to_string(),
            disposition_decision: "accept".to_string(),
            disposition_approver: "policy".to_string(),
            disposition_human_disposed: false,
            disposition_verdict_class: "executed".to_string(),
            chain,
            store_nonce: "5".repeat(64),
        }
    }

    /// CAPSULE_ID INVARIANCE PIN (inline signature envelope):
    /// the `capsule_id` for a FIXED input MUST NOT change when the inline
    /// producer-signature envelope is added. `signature`/`key_id` are excluded
    /// from the `capsule_id` preimage (see `crate::producer::jcs` /
    /// `capsule_emit.canonicalization._LOCAL_ONLY_FIELDS`), so attaching the
    /// envelope leaves the id byte-for-byte unchanged and the tour-fixture
    /// capsule ids stay stable. These two literals were captured from
    /// `seal()`'s output BEFORE this task's envelope work (the `seal()`
    /// function body is byte-identical to that commit); if either changes, the
    /// preimage was perturbed and the fix is wrong. Re-pinned once for the
    /// AAC-05 `spec_version`; before:
    /// standalone f22a917852446fe83865ad3a247bad9471d3b1bc464c1e193401804e07c44550,
    /// chained 050b194efccf9e000d19f3d7f965b82861a7217389ed5e91888f3f5f9c04f8e3
    /// (both still reproduce with `SPEC_VERSION` set back to -02).
    /// Re-pinned again for the store nonce (Evidence Layer -00 §12.1):
    /// `compute_attestation.store_nonce` now rides inside
    /// the preimage, so every id moves. Before:
    /// standalone bff097741877ff783911eb6a2771c0fcca9e6e4862d0126b368f20cee1ebb7c8,
    /// chained 4ba7516fbae648e0bf8c868a3ea59c1f1884a892dde85ef1c40c82b816a19231.
    #[test]
    fn capsule_id_is_unchanged_by_attaching_the_producer_envelope() {
        const STANDALONE_ID: &str =
            "8180ac62b30a6a3ba7fbfc2888ff8ffb6bdc6d2fc23bf9d6b02e62f434aa74f0";
        const CHAINED_ID: &str = "17b5330e0b8dbacf18306f345a08c8d3a6e77fd9a9c1d67f546964dea545a94e";

        // 1. The pinned fixture ids are what `seal()` computes today.
        let mut standalone = seal(&base_input(None)).unwrap();
        assert_eq!(standalone["capsule_id"], STANDALONE_ID);
        let mut chained = seal(&base_input(Some(ChainLink {
            parent_capsule_id: "f".repeat(64),
            relation: "confirms".to_string(),
        })))
        .unwrap();
        assert_eq!(chained["capsule_id"], CHAINED_ID);

        // 2. Attaching the inline envelope adds signature/key_id but does NOT
        //    move capsule_id -- with two DIFFERENT keys, to prove the id is
        //    signer-independent.
        let key_a = crate::producer::keys::KeyPair::generate();
        let key_b = crate::producer::keys::KeyPair::generate();
        attach_producer_envelope(&mut standalone, &key_a.signing_key).unwrap();
        attach_producer_envelope(&mut chained, &key_b.signing_key).unwrap();
        assert_eq!(
            standalone["capsule_id"], STANDALONE_ID,
            "attaching the envelope must never change capsule_id"
        );
        assert_eq!(chained["capsule_id"], CHAINED_ID);
        assert!(standalone.get("signature").is_some());
        assert!(standalone.get("key_id").is_some());
    }

    /// The inline `key_id` is the RAW 32-byte Ed25519 public key, hex (64
    /// chars) -- exactly `capsule_emit.seal()`'s `capsule["key_id"]` and what
    /// the announced-key registry (`CAPSULES_PEER_KEYS`) keys off. NOT this crate's own short
    /// SHA-256-based `keys::key_id` (16 chars).
    #[test]
    fn attached_key_id_is_the_raw_public_key_hex_not_the_short_key_id() {
        let key = crate::producer::keys::KeyPair::generate();
        let mut capsule = seal(&base_input(None)).unwrap();
        attach_producer_envelope(&mut capsule, &key.signing_key).unwrap();
        let key_id = capsule["key_id"].as_str().unwrap();
        assert_eq!(
            key_id.len(),
            64,
            "key_id must be the raw 32-byte pubkey hex"
        );
        assert_eq!(
            key_id,
            hex::encode(key.signing_key.verifying_key().to_bytes())
        );
        assert_ne!(key_id, key.key_id(), "must NOT be the short SHA-256 key_id");
    }

    /// The `x-mesh-poc-v1` extension's serving-provenance sub-object of one
    /// sealed capsule.
    fn provenance(capsule: &Value) -> &Value {
        &capsule["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"]["serving_provenance"]
    }

    /// The enriched capsule carries EVERY provenance field the host exposes,
    /// under `x-mesh-poc-v1.serving_provenance`, plus the truthfully-renamed
    /// `model_name_digest`. This is the record that must answer "which model,
    /// at what quantization, on whose hardware, for which exchange".
    #[test]
    fn capsule_carries_full_serving_provenance() {
        let capsule = seal(&base_input(None)).unwrap();
        let poc = &capsule["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"];

        // Truthful name: NOT model_package_digest (weights), it's a name hash.
        assert_eq!(poc["model_name_digest"], "d".repeat(64));
        assert!(
            poc.get("model_package_digest").is_none(),
            "the overclaiming name must be gone entirely"
        );

        let prov = provenance(&capsule);
        assert_eq!(prov["served_by_node_id"], "node-under-test");
        assert_eq!(prov["requesting_party"], "client-under-test");
        assert_eq!(prov["exchange_id"], "exch-under-test");
        assert_eq!(prov["hostname"], "host-under-test");
        // Quantization from the host serving-provenance block (real value).
        assert_eq!(prov["quantization"], "Q4_K_M");
        // Model identity / fidelity from the host serving-provenance block.
        assert_eq!(prov["model"]["architecture"], "llama");
        assert_eq!(prov["model"]["context_length"], 8192);
        assert_eq!(prov["model"]["parameter_size"], "7B");
        assert_eq!(prov["model"]["layer_count"], 32);
        assert_eq!(prov["model"]["identity_hash"], "a".repeat(64));
        // The bytes-hash rides alongside the name-hash as a distinct fact.
        assert_eq!(prov["model"]["weights_digest"], "9".repeat(64));
        assert_eq!(prov["model"]["canonical_ref"], "repo@rev/model.gguf");
        assert_eq!(prov["model"]["revision"], "rev");
        // Hardware from the host serving-provenance block (real values).
        assert_eq!(prov["hardware"]["gpu"], "Apple M3 Max");
        assert_eq!(prov["hardware"]["vram_bytes"], 38_654_705_664u64);
        assert_eq!(prov["hardware"]["is_soc"], true);
        // Host carries is_soc, not a device enum -> device stays null.
        assert!(prov["hardware"]["device"].is_null());
        // Usage IS real when the response carried it.
        assert_eq!(prov["usage"]["prompt_tokens"], 11);
        assert_eq!(prov["usage"]["completion_tokens"], 22);
        assert_eq!(prov["usage"]["total_tokens"], 33);
    }

    /// `epistemic_type` -- additive, a
    /// top-level sibling of `x-mesh-poc-v1`, derived from the SAME `role`
    /// signal `capsule_mesh_view.label_role()` already treats as
    /// authoritative. `served` -> `producer_claim` (this node's own claim
    /// about what it served); `requested` -> `observed_event` (this node's
    /// own observation of what it received).
    #[test]
    fn epistemic_type_for_served_role_is_producer_claim() {
        let mut input = base_input(None);
        input.mesh_poc.role = "served".to_string();
        let capsule = seal(&input).unwrap();
        assert_eq!(
            capsule["model_attestation"]["compute_attestation"]["epistemic_type"],
            "producer_claim"
        );
    }

    #[test]
    fn epistemic_type_for_requested_role_is_observed_event() {
        let mut input = base_input(None);
        input.mesh_poc.role = "requested".to_string();
        let capsule = seal(&input).unwrap();
        assert_eq!(
            capsule["model_attestation"]["compute_attestation"]["epistemic_type"],
            "observed_event"
        );
    }

    /// The mutant this guards against: rounding an ambiguous/unrecognized
    /// role up to either epistemic claim. `conflict`/`unknown` must never
    /// silently become `producer_claim` or `observed_event` -- the field is
    /// absent, same "absent, never fabricated" discipline `role` itself
    /// documents.
    #[test]
    fn epistemic_type_absent_for_conflict_and_unknown_roles() {
        for role in ["conflict", "unknown"] {
            let mut input = base_input(None);
            input.mesh_poc.role = role.to_string();
            let capsule = seal(&input).unwrap();
            assert!(
                capsule["model_attestation"]["compute_attestation"]
                    .get("epistemic_type")
                    .is_none(),
                "role={role:?} must never carry an epistemic_type claim"
            );
        }
    }

    /// The OPTIONAL `tool_calls_digest`/`reasoning_digest` sub-digests ride
    /// inside `compute_attestation` beside `agent_output_digest` when present,
    /// mirroring the Python reference shape. This is the slot a real
    /// tool_calls_digest lands in.
    #[test]
    fn capsule_carries_optional_tool_calls_and_reasoning_digests_when_present() {
        let mut input = base_input(None);
        input.tool_calls_digest = Some("f".repeat(64));
        input.reasoning_digest = Some("e".repeat(64));
        let capsule = seal(&input).unwrap();
        let ca = &capsule["model_attestation"]["compute_attestation"];
        assert_eq!(ca["tool_calls_digest"], "f".repeat(64));
        assert_eq!(ca["reasoning_digest"], "e".repeat(64));
        // The mandatory digests are still present alongside them.
        assert_eq!(ca["agent_output_digest"], "b".repeat(64));
    }

    /// ABSENT, not null: when the model had no tool calls / no reasoning, the
    /// keys are omitted entirely from the sealed capsule — never a fabricated
    /// digest over an empty list — exactly as the Python reference
    /// `build_conversation_exchange_capsule` omits them.
    #[test]
    fn capsule_omits_optional_sub_digests_when_absent() {
        let capsule = seal(&base_input(None)).unwrap();
        let ca = &capsule["model_attestation"]["compute_attestation"];
        assert!(
            ca.get("tool_calls_digest").is_none(),
            "an absent tool_calls_digest must be omitted, not null"
        );
        assert!(
            ca.get("reasoning_digest").is_none(),
            "an absent reasoning_digest must be omitted, not null"
        );
    }

    /// The sub-digests are digest-bound: setting a real `tool_calls_digest`
    /// changes `capsule_id`, so an attester cannot swap the tool calls out of a
    /// sealed record without breaking the content address.
    #[test]
    fn setting_tool_calls_digest_changes_capsule_id() {
        let baseline = seal(&base_input(None)).unwrap();
        let baseline_id = baseline["capsule_id"].as_str().unwrap().to_string();
        let mut input = base_input(None);
        input.tool_calls_digest = Some("f".repeat(64));
        assert_ne!(seal(&input).unwrap()["capsule_id"], baseline_id.as_str());
    }

    /// Mutating ANY provenance field changes `capsule_id` — the record is
    /// content-addressed over its provenance, so an attester cannot swap the
    /// served node, exchange id, or token counts without breaking the digest.
    #[test]
    fn mutating_a_provenance_field_changes_capsule_id() {
        let baseline = seal(&base_input(None)).unwrap();
        let baseline_id = baseline["capsule_id"].as_str().unwrap().to_string();

        // (1) served_by_node_id
        let mut input = base_input(None);
        input.mesh_poc.serving_provenance.served_by_node_id = "other-node".to_string();
        assert_ne!(seal(&input).unwrap()["capsule_id"], baseline_id.as_str());

        // (2) exchange_id
        let mut input = base_input(None);
        input.mesh_poc.serving_provenance.exchange_id = "other-exch".to_string();
        assert_ne!(seal(&input).unwrap()["capsule_id"], baseline_id.as_str());

        // (3) quantization (baseline is "Q4_K_M"; mutate to a different quant)
        let mut input = base_input(None);
        input.mesh_poc.serving_provenance.quantization = "Q8_0".to_string();
        assert_ne!(seal(&input).unwrap()["capsule_id"], baseline_id.as_str());

        // (3b) a model-fidelity field (architecture) is also digest-bound.
        let mut input = base_input(None);
        input.mesh_poc.serving_provenance.architecture = Some("mistral".to_string());
        assert_ne!(seal(&input).unwrap()["capsule_id"], baseline_id.as_str());

        // (3c) a hardware field (gpu) is also digest-bound.
        let mut input = base_input(None);
        input.mesh_poc.serving_provenance.hardware_gpu = Some("NVIDIA H100".to_string());
        assert_ne!(seal(&input).unwrap()["capsule_id"], baseline_id.as_str());

        // (3d) the weights digest is also digest-bound -- a quant swap under
        // the same model name records as a changed capsule (it does not
        // prevent the swap; it makes it a signed, non-repudiable fact).
        let mut input = base_input(None);
        input.mesh_poc.serving_provenance.weights_digest = Some("8".repeat(64));
        assert_ne!(seal(&input).unwrap()["capsule_id"], baseline_id.as_str());

        // (4) usage token counts
        let mut input = base_input(None);
        input.mesh_poc.serving_provenance.usage = Some(TokenUsage {
            prompt_tokens: 999,
            completion_tokens: 1,
            total_tokens: 1000,
        });
        assert_ne!(seal(&input).unwrap()["capsule_id"], baseline_id.as_str());

        // (5) model_name_digest
        let mut input = base_input(None);
        input.mesh_poc.model_name_digest = "e".repeat(64);
        assert_ne!(seal(&input).unwrap()["capsule_id"], baseline_id.as_str());
    }

    // =======================================================================
    // twin_bracket_id -- forwarded verbatim from the terminal envelope,
    // never minted or defaulted here (see [`ServingProvenance::twin_bracket_id`]).
    // =======================================================================

    /// Positive vector: when the envelope carried a `twin_bracket_id`, the
    /// sealed record carries it verbatim under `serving_provenance`.
    /// MUTANT: drop the forward (stop copying the field into `ServingProvenance`
    /// before calling `seal`) and this assertion goes red.
    #[test]
    fn twin_bracket_id_present_is_carried_verbatim() {
        let mut input = base_input(None);
        input.mesh_poc.serving_provenance.twin_bracket_id = Some("twin-abc123".to_string());
        let capsule = seal(&input).unwrap();
        assert_eq!(provenance(&capsule)["twin_bracket_id"], "twin-abc123");
    }

    #[test]
    fn response_text_digest_is_carried_when_present_and_absent_otherwise() {
        let mut input = base_input(None);
        let capsule = seal(&input).unwrap();
        assert!(
            provenance(&capsule).get("response_text_digest").is_none(),
            "never a null placeholder"
        );
        input.mesh_poc.serving_provenance.response_text_digest = Some("ab".repeat(32));
        let capsule = seal(&input).unwrap();
        assert_eq!(
            provenance(&capsule)["response_text_digest"],
            "ab".repeat(32)
        );
    }

    /// Absence rule: when the envelope carried no `twin_bracket_id` (the
    /// overwhelming majority of exchanges), the key is OMITTED entirely --
    /// never `null` -- so a reader can never mistake an untwinned row for a
    /// half-formed bracket.
    #[test]
    fn twin_bracket_id_absent_when_envelope_carried_none() {
        let capsule = seal(&base_input(None)).unwrap();
        assert!(
            provenance(&capsule).get("twin_bracket_id").is_none(),
            "an absent twin_bracket_id must be omitted, not null"
        );
    }

    /// AAC-05 §5.2: `confirmed` without a digest of the actual output is a
    /// false record, so `seal` refuses it rather than emitting one.
    #[test]
    fn seal_refuses_confirmed_without_a_response_digest() {
        let mut input = base_input(None);
        input.effect_response_digest = None;
        assert!(matches!(seal(&input), Err(SealError::EffectInvariant(_))));
    }

    #[test]
    fn seal_refuses_a_non_digest_in_an_effect_digest_slot() {
        let mut input = base_input(None);
        input.effect_request_digest = Some("unknown-request:m".to_string());
        assert!(matches!(seal(&input), Err(SealError::EffectInvariant(_))));
    }

    #[test]
    fn seal_refuses_dispatched_with_a_response_digest() {
        let mut input = base_input(None);
        input.effect_status = "dispatched".to_string();
        assert!(matches!(seal(&input), Err(SealError::EffectInvariant(_))));
    }

    /// `dispatched` with no response digest seals, omits the member (never
    /// null), and derives `dispatched_unconfirmed`.
    #[test]
    fn dispatched_effect_omits_response_digest_and_is_unconfirmed() {
        let mut input = base_input(None);
        input.effect_status = "dispatched".to_string();
        input.effect_response_digest = None;
        let capsule = seal(&input).unwrap();
        assert_eq!(capsule["effect"]["status"], "dispatched");
        assert!(capsule["effect"].get("response_digest").is_none());
        assert_eq!(capsule["effect"]["request_digest"], "a".repeat(64));
        assert_eq!(
            capsule["assurance"]["effect_mode"],
            "dispatched_unconfirmed"
        );
    }

    #[test]
    fn sealed_capsule_declares_aac_05() {
        let capsule = seal(&base_input(None)).unwrap();
        assert_eq!(
            capsule["spec_version"],
            "draft-mih-scitt-agent-action-capsule-05"
        );
    }

    /// A record sealed without an opted-in hostname carries no `hostname`
    /// key at all -- not a null that invites a reader to ask what was hidden.
    #[test]
    fn hostname_absent_when_not_supplied() {
        let mut input = base_input(None);
        input.mesh_poc.serving_provenance.hostname = None;
        let capsule = seal(&input).unwrap();
        assert!(
            provenance(&capsule).get("hostname").is_none(),
            "an unsupplied hostname must be omitted, not null"
        );
    }

    /// Digest-bound: a real `twin_bracket_id` is committed into `capsule_id`
    /// like every other provenance fact, so an attester cannot swap a row
    /// into (or out of) a bracket without changing the sealed record's
    /// content address.
    #[test]
    fn twin_bracket_id_is_digest_bound() {
        let baseline = seal(&base_input(None)).unwrap();
        let baseline_id = baseline["capsule_id"].as_str().unwrap().to_string();
        let mut input = base_input(None);
        input.mesh_poc.serving_provenance.twin_bracket_id = Some("twin-abc123".to_string());
        assert_ne!(seal(&input).unwrap()["capsule_id"], baseline_id.as_str());
    }

    /// Two rows of one real twin: both are sealed from their own node's copy
    /// of the SAME host-minted id, so both sealed records carry the
    /// identical `twin_bracket_id` even though everything else about them
    /// (which node served, the exchange id) differs -- this is what lets a
    /// reader bracket the two rows together.
    #[test]
    fn two_rows_sharing_a_twin_bracket_id_carry_the_identical_value() {
        let mut requester = base_input(None);
        requester.mesh_poc.serving_provenance.served_by_node_id = "node-a".to_string();
        requester.mesh_poc.serving_provenance.exchange_id = "exch-a".to_string();
        requester.mesh_poc.serving_provenance.twin_bracket_id = Some("twin-shared".to_string());

        let mut served = base_input(None);
        served.mesh_poc.serving_provenance.served_by_node_id = "node-b".to_string();
        served.mesh_poc.serving_provenance.exchange_id = "exch-b".to_string();
        served.mesh_poc.serving_provenance.twin_bracket_id = Some("twin-shared".to_string());

        let requester_capsule = seal(&requester).unwrap();
        let served_capsule = seal(&served).unwrap();
        assert_eq!(
            provenance(&requester_capsule)["twin_bracket_id"],
            provenance(&served_capsule)["twin_bracket_id"]
        );
        assert_eq!(
            provenance(&requester_capsule)["twin_bracket_id"],
            "twin-shared"
        );
        // Different exchanges, same bracket id -- distinct capsule_ids.
        assert_ne!(
            requester_capsule["capsule_id"],
            served_capsule["capsule_id"]
        );
    }

    /// the producer's own acceptance
    /// centerpiece: every sealed capsule declares format_version "4" +
    /// canonicalization_id "jcs" (§5.1) -- without both, the unmodified
    /// draft-04 Python `verify()` rejects the record with
    /// `canonicalization_id_missing` (bumping the number alone is not enough).
    #[test]
    fn sealed_capsule_declares_format_4_and_jcs_canonicalization() {
        let capsule = seal(&base_input(None)).unwrap();
        assert_eq!(capsule["format_version"], "4");
        assert_eq!(capsule["canonicalization_id"], "jcs");
    }

    #[test]
    fn standalone_capsule_has_no_chain_block_and_standalone_ledger_mode() {
        let capsule = seal(&base_input(None)).unwrap();
        assert!(capsule.get("chain").is_none());
        assert_eq!(capsule["assurance"]["ledger_mode"], "standalone");
    }

    #[test]
    fn chained_capsule_carries_chain_block_and_chained_ledger_mode() {
        let parent = "f".repeat(64);
        let input = base_input(Some(ChainLink {
            parent_capsule_id: parent.clone(),
            relation: "follows".to_string(),
        }));
        let capsule = seal(&input).unwrap();
        assert_eq!(capsule["chain"]["parent_capsule_id"], parent);
        assert_eq!(capsule["chain"]["relation"], "follows");
        assert_eq!(capsule["assurance"]["ledger_mode"], "chained");
    }

    #[test]
    fn capsule_id_is_bound_to_the_chain_blocks_content() {
        // Format 4 / plain JCS (the draft-04 reversal) commits `chain` into
        // the `capsule_id` digest -- unlike the withdrawn vintage `jcs-n`
        // profile, which excluded it. Among capsules that are ALREADY chained
        // (same ledger_mode), varying parent_capsule_id/relation now DOES
        // perturb capsule_id, closing the prior unauthenticated-chain gap: an
        // attester can no longer splice a sealed record onto a different
        // parent without changing its content address.
        let chained_a = seal(&base_input(Some(ChainLink {
            parent_capsule_id: "f".repeat(64),
            relation: "follows".to_string(),
        })))
        .unwrap();
        let chained_b = seal(&base_input(Some(ChainLink {
            parent_capsule_id: "0".repeat(64),
            relation: "confirms".to_string(),
        })))
        .unwrap();
        assert_ne!(chained_a["capsule_id"], chained_b["capsule_id"]);
        assert_eq!(chained_a["chain"]["parent_capsule_id"], "f".repeat(64));
        assert_eq!(chained_b["chain"]["parent_capsule_id"], "0".repeat(64));
    }

    // =======================================================================
    // host_binding — reverse-direction composition binding
    // =======================================================================

    /// Positive vector: a `host_binding` group carries mesh-llm's own digest
    /// under its own construction, structurally valid per
    /// `validate_host_binding`, and digest-bound into `capsule_id`.
    #[test]
    fn host_binding_present_is_carried_and_valid() {
        let mut input = base_input(None);
        input.host_binding = Some(HostBinding {
            digest: "a6329c5ebb66562f38a8136a8d8511b6aeed166e4c7d889b9133ac96fc49a9d5".to_string(),
            construction: MESH_LLM_REQUEST_BODY_SHA256_V1.to_string(),
            purpose: HOST_LOG_JOIN.to_string(),
        });
        let capsule = seal(&input).unwrap();
        let hb = &capsule["model_attestation"]["compute_attestation"]["host_binding"];
        assert_eq!(
            hb["digest"],
            "a6329c5ebb66562f38a8136a8d8511b6aeed166e4c7d889b9133ac96fc49a9d5"
        );
        assert_eq!(hb["construction"], MESH_LLM_REQUEST_BODY_SHA256_V1);
        assert_eq!(hb["purpose"], HOST_LOG_JOIN);
        assert!(validate_host_binding(&capsule).is_ok());

        // digest-bound: a different host digest changes capsule_id.
        let baseline_id = capsule["capsule_id"].as_str().unwrap().to_string();
        let mut other = base_input(None);
        other.host_binding = Some(HostBinding {
            digest: "0".repeat(64),
            construction: MESH_LLM_REQUEST_BODY_SHA256_V1.to_string(),
            purpose: HOST_LOG_JOIN.to_string(),
        });
        assert_ne!(seal(&other).unwrap()["capsule_id"], baseline_id.as_str());
    }

    /// Absence rule: when no host digest exists, the key is OMITTED entirely
    /// — never present as `null`.
    #[test]
    fn host_binding_absent_when_not_supplied() {
        let capsule = seal(&base_input(None)).unwrap();
        let ca = &capsule["model_attestation"]["compute_attestation"];
        assert!(
            ca.get("host_binding").is_none(),
            "an absent host_binding must be omitted, not null"
        );
        assert!(validate_host_binding(&capsule).is_ok());
    }

    /// MUST-FAIL (a): `construction` missing or unregistered.
    #[test]
    fn must_fail_construction_missing_or_unregistered() {
        let mut capsule = seal(&base_input(None)).unwrap();
        // (a1) construction missing entirely.
        capsule["model_attestation"]["compute_attestation"]["host_binding"] = json!({
            "digest": "a".repeat(64),
            "purpose": HOST_LOG_JOIN,
        });
        assert!(validate_host_binding(&capsule).is_err());

        // (a2) construction present but not a registered label.
        capsule["model_attestation"]["compute_attestation"]["host_binding"] = json!({
            "digest": "a".repeat(64),
            "construction": "some-unregistered-scheme/v1",
            "purpose": HOST_LOG_JOIN,
        });
        assert!(validate_host_binding(&capsule).is_err());
    }

    /// MUST-FAIL (b): a `null` group — the absence rule requires the key be
    /// OMITTED, never carried as `null`.
    #[test]
    fn must_fail_null_group() {
        let mut capsule = seal(&base_input(None)).unwrap();
        capsule["model_attestation"]["compute_attestation"]["host_binding"] = Value::Null;
        assert!(validate_host_binding(&capsule).is_err());
    }

    /// MUST-FAIL (c) — THE ACCEPTANCE CENTERPIECE: the equality-inference
    /// mutant. `host_binding.digest` and `agent_input_digest` are two
    /// independent, co-signed claims (the INDEPENDENCE RULE on
    /// [`HostBinding`]) — a record where they legitimately DIFFER (mesh-llm's
    /// format drifting from ours) is still a perfectly valid record.
    ///
    /// `validate_host_binding` (the real structural check) agrees: it never
    /// reads `agent_input_digest`, so a divergent-but-well-formed
    /// `host_binding` still passes.
    ///
    /// The MUTANT below is a verifier that additionally infers equality
    /// between the two bindings — exactly the inference the independence
    /// rule prohibits. Run against the SAME divergent-but-valid record, the
    /// mutant WRONGLY rejects it. That the mutant fails red here is the
    /// point: it demonstrates precisely the false rejection the rule exists
    /// to prevent, and pins that this crate's real verifier contains no such
    /// inference.
    #[test]
    fn equality_inference_mutant_wrongly_rejects_legitimate_format_drift() {
        let mut input = base_input(None);
        input.agent_input_digest = "1".repeat(64); // our construction's value
        input.host_binding = Some(HostBinding {
            digest: "2".repeat(64), // mesh-llm's construction's value -- DIFFERENT
            construction: MESH_LLM_REQUEST_BODY_SHA256_V1.to_string(),
            purpose: HOST_LOG_JOIN.to_string(),
        });
        let capsule = seal(&input).unwrap();

        // The REAL verifier: independence respected, passes.
        assert!(
            validate_host_binding(&capsule).is_ok(),
            "the real validator must not infer equality and must accept divergent bindings"
        );

        // The MUTANT verifier: infers equality, wrongly fails.
        assert!(
            equality_inference_mutant(&capsule).is_err(),
            "MUTANT FAILS RED as expected: inferring equality between two \
             independent bindings wrongly rejects a legitimate record"
        );
    }

    /// The equality-inference mutant also wrongly rejects on the very
    /// PLUMBING every other test in this module uses (`host_binding` absent):
    /// with no `host_binding` present there is nothing to compare, so both
    /// the real validator and this mutant agree — Ok. Included so the mutant
    /// is not vacuously "always Err" and only fails on the case that matters.
    #[test]
    fn equality_inference_mutant_agrees_when_host_binding_absent() {
        let capsule = seal(&base_input(None)).unwrap();
        assert!(validate_host_binding(&capsule).is_ok());
        assert!(equality_inference_mutant(&capsule).is_ok());
    }

    /// A MUTANT verifier, deliberately wrong: asserts record validity BECAUSE
    /// `host_binding.digest == agent_input_digest`. This function exists ONLY
    /// to be exercised by the tests above — production code (`seal`,
    /// `validate_host_binding`) contains no such comparison anywhere.
    fn equality_inference_mutant(capsule: &Value) -> Result<(), String> {
        let ca = capsule
            .get("model_attestation")
            .and_then(|m| m.get("compute_attestation"))
            .ok_or_else(|| "missing compute_attestation".to_string())?;
        let Some(hb) = ca.get("host_binding").filter(|v| !v.is_null()) else {
            return Ok(()); // nothing to (wrongly) compare
        };
        let hb_digest = hb
            .get("digest")
            .and_then(Value::as_str)
            .ok_or_else(|| "host_binding.digest missing".to_string())?;
        let agent_input_digest = ca
            .get("agent_input_digest")
            .and_then(Value::as_str)
            .ok_or_else(|| "agent_input_digest missing".to_string())?;
        if hb_digest != agent_input_digest {
            return Err(format!(
                "MUTANT: host_binding.digest {hb_digest:?} != agent_input_digest \
                 {agent_input_digest:?} -- treated as invalid (WRONG: these are \
                 two independent bindings, not required to match)"
            ));
        }
        Ok(())
    }

    // -------------------------------------------------------------------
    // seal_citing_record
    // -------------------------------------------------------------------

    #[test]
    fn routing_choice_names_the_peer_only_by_commitment() {
        let key = crate::producer::keys::KeyPair::generate();
        let peer = "a70d3967bea3b22f".repeat(4);
        let salt = [7u8; 32];
        let choice = LocalRoutingChoice {
            change: RoutingChoiceChange::Block,
            peer_id: &peer,
            salt: &salt,
            until: Some("2026-10-04T00:00:00Z"),
            rule: None,
        };
        let head = "c".repeat(64);
        let capsule = seal_local_routing_choice(&choice, Some(&head), &key.signing_key).unwrap();

        assert_eq!(
            capsule["capsule_id"].as_str().unwrap(),
            compute_capsule_id(&capsule).unwrap()
        );
        assert_eq!(capsule["chain"]["parent_capsule_id"], json!(head));
        let fact = &capsule["model_attestation"]["compute_attestation"]["local_routing_choice"];
        assert_eq!(fact["change"], json!("block"));
        // `until` is committed at minute granularity, like every sealed time.
        assert_eq!(fact["until"], json!("2026-10-04T00:00:00Z"));
        assert_eq!(
            fact["peer_commitment"]["digest"],
            json!(peer_commitment(&peer, &salt))
        );
        // A local API call is not proof a person made it: no human claim.
        assert_eq!(capsule["disposition"]["approver"], json!("policy"));
        assert_eq!(capsule["disposition"]["human_disposed"], json!(false));
        // The peer id itself appears nowhere in the sealed bytes.
        assert!(!serde_json::to_string(&capsule).unwrap().contains(&peer));
        assert!(capsule.get("signature").is_some());
    }

    /// A block an operator rule asked for names the rule and cites its
    /// verdicts, but only by commitment under the record's salt: the verdicts
    /// name the peer, so no verdict id appears in the sealed bytes either.
    #[test]
    fn a_rule_block_cites_its_verdicts_only_by_commitment() {
        let key = crate::producer::keys::KeyPair::generate();
        let peer = "a70d3967bea3b22f".repeat(4);
        let salt = [7u8; 32];
        let verdicts = vec!["1".repeat(64), "2".repeat(64)];
        let rule = RoutingRuleCitation {
            rule: "stop_routing_after_contradictions",
            after: 2,
            window_days: 30,
            verdict_capsule_ids: &verdicts,
        };
        let choice = LocalRoutingChoice {
            change: RoutingChoiceChange::Block,
            peer_id: &peer,
            salt: &salt,
            until: None,
            rule: Some(&rule),
        };
        let capsule = seal_local_routing_choice(&choice, None, &key.signing_key).unwrap();
        let fact = &capsule["model_attestation"]["compute_attestation"]["local_routing_choice"];
        assert_eq!(
            fact["rule"]["rule"],
            json!("stop_routing_after_contradictions")
        );
        assert_eq!(fact["rule"]["after"], json!(2));
        assert_eq!(fact["rule"]["window_days"], json!(30));
        assert_eq!(
            fact["rule"]["verdict_commitments"][1]["digest"],
            json!(peer_commitment(&verdicts[1], &salt))
        );
        let sealed = serde_json::to_string(&capsule).unwrap();
        assert!(verdicts.iter().all(|id| !sealed.contains(id.as_str())));
        assert!(!sealed.contains(&peer));

        let manual = LocalRoutingChoice {
            rule: None,
            ..choice
        };
        let capsule = seal_local_routing_choice(&manual, None, &key.signing_key).unwrap();
        assert_eq!(
            capsule["model_attestation"]["compute_attestation"]["local_routing_choice"]["rule"],
            Value::Null
        );
    }

    /// A routing-choice record takes the same seal path as every other record
    /// (Evidence Layer -00 §12.1): a fresh 256-bit store nonce, and committed
    /// times (the record timestamp and the block's `until`) truncated to the
    /// minute.
    #[test]
    fn routing_choice_carries_a_store_nonce_and_minute_times() {
        let key = crate::producer::keys::KeyPair::generate();
        let peer = "a70d3967bea3b22f".repeat(4);
        let salt = [7u8; 32];
        let choice = LocalRoutingChoice {
            change: RoutingChoiceChange::Block,
            peer_id: &peer,
            salt: &salt,
            until: Some("2026-10-04T00:00:37.250Z"),
            rule: None,
        };
        let first = seal_local_routing_choice(&choice, None, &key.signing_key).unwrap();
        let second = seal_local_routing_choice(&choice, None, &key.signing_key).unwrap();

        let compute = &first["model_attestation"]["compute_attestation"];
        let nonce = compute[STORE_NONCE_FIELD]
            .as_str()
            .expect("store_nonce is sealed");
        assert_eq!(nonce.len(), 64);
        assert!(nonce
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert_ne!(
            second["model_attestation"]["compute_attestation"][STORE_NONCE_FIELD],
            compute[STORE_NONCE_FIELD],
            "a fresh nonce per record"
        );
        assert_ne!(first["capsule_id"], second["capsule_id"]);

        assert_eq!(
            compute["local_routing_choice"]["until"],
            json!("2026-10-04T00:00:00Z")
        );
        let timestamp = first["timestamp"].as_str().unwrap();
        assert_eq!(
            crate::producer::timestamp::coarsen_to_minute(timestamp).as_deref(),
            Some(timestamp),
            "the record timestamp is already whole minutes"
        );
        let unblock = LocalRoutingChoice {
            change: RoutingChoiceChange::Unblock,
            until: None,
            ..choice
        };
        let sealed = seal_local_routing_choice(&unblock, None, &key.signing_key).unwrap();
        assert_eq!(
            sealed["model_attestation"]["compute_attestation"]["local_routing_choice"]["until"],
            Value::Null
        );
    }

    /// Cross-implementation vector: the mesh-llm host's
    /// `network::peer_blocks::peer_commitment` pins the same inputs and output
    /// (`commitment_matches_the_plugin_vector`); computed independently with
    /// Python's hashlib.
    #[test]
    fn peer_commitment_matches_the_host_vector() {
        let peer = "a70d3967bea3b22fa48a28f77c5d2b3764fc8bd5204a82c09ff8430f3f2a0a00";
        assert_eq!(
            peer_commitment(peer, &[7u8; 32]),
            "265aff057ebdeba261253f6ce9d8eea274ef65bab0aadfbf99a5ddb2facd7e74"
        );
    }

    #[test]
    fn peer_commitment_binds_salt_and_peer() {
        let peer = "a".repeat(64);
        let base = peer_commitment(&peer, &[1u8; 32]);
        assert_ne!(base, peer_commitment(&peer, &[2u8; 32]));
        assert_ne!(base, peer_commitment(&"b".repeat(64), &[1u8; 32]));
        assert_eq!(base, peer_commitment(&peer, &[1u8; 32]));
    }

    #[test]
    fn unblock_record_is_standalone_without_head_and_has_no_until() {
        let key = crate::producer::keys::KeyPair::generate();
        let peer = "b".repeat(64);
        let choice = LocalRoutingChoice {
            change: RoutingChoiceChange::Unblock,
            peer_id: &peer,
            salt: &[3u8; 32],
            until: None,
            rule: None,
        };
        let capsule = seal_local_routing_choice(&choice, None, &key.signing_key).unwrap();
        assert!(capsule.get("chain").is_none());
        assert_eq!(capsule["assurance"]["ledger_mode"], json!("standalone"));
        let fact = &capsule["model_attestation"]["compute_attestation"]["local_routing_choice"];
        assert_eq!(fact["change"], json!("unblock"));
        assert_eq!(fact["until"], Value::Null);
    }

    fn sample_prov<'a>(foreign_capsule_id: &'a str) -> ReceivedHalfProvenance<'a> {
        ReceivedHalfProvenance {
            foreign_capsule_id,
            received_from: "node-b",
            via: "push",
            received_at: "2026-09-25T00:00:00Z",
            signature_ok: true,
            digest_match: None,
            foreign_agent_input_digest: Some("a".repeat(64).leak()),
            foreign_agent_output_digest: Some("b".repeat(64).leak()),
        }
    }

    /// A citing record is a well-formed format-4 capsule whose recomputed
    /// `capsule_id` covers the `references` array (so a reader who tampers the
    /// cited digest is caught), carries `chain.relation == "follows"` onto the
    /// supplied head (the citation is the `references[]` entry, never a
    /// relation value), and cites the foreign half by CPB
    /// typed digest with `citation_purpose == "counterparty_half"`.
    #[test]
    fn seal_citing_record_is_well_formed_and_cites_the_foreign_half() {
        let key = crate::producer::keys::KeyPair::generate();
        let head = "c".repeat(64);
        let foreign = "d".repeat(64);
        let prov = sample_prov(&foreign);
        let capsule = seal_citing_record(&prov, Some(&head), &key.signing_key).unwrap();

        // capsule_id recomputes over the whole body (incl. chain + references).
        let stored = capsule["capsule_id"].as_str().unwrap();
        assert_eq!(stored, compute_capsule_id(&capsule).unwrap());

        assert_eq!(capsule["chain"]["parent_capsule_id"], json!(head));
        assert_eq!(capsule["chain"]["relation"], json!(CHAIN_RELATION_FOLLOWS));
        assert_eq!(capsule["chain"]["relation"], json!("follows"));
        let reference = &capsule["references"][0];
        assert_eq!(reference["type"], json!(REFERENCE_TYPE_CAPSULE));
        assert_eq!(reference["digest_alg"], json!(REFERENCE_DIGEST_ALG));
        assert_eq!(reference["digest"], json!(foreign));
        assert_eq!(
            reference["citation_purpose"],
            json!(CITATION_PURPOSE_COUNTERPARTY_HALF)
        );
        // The receiving-event facts ride in compute_attestation.received_half.
        let rh = &capsule["model_attestation"]["compute_attestation"]["received_half"];
        assert_eq!(rh["received_from"], json!("node-b"));
        assert_eq!(rh["signature_ok"], json!(true));
        assert_eq!(rh["cited_capsule_id"], json!(foreign));
        // Inline producer envelope attached (one-capsule-shape), excluded from id.
        assert!(capsule.get("signature").is_some());
        assert!(capsule.get("key_id").is_some());
    }

    /// The FIRST record in a chain: a citing record with no head is standalone
    /// (no `chain` block) and still cites the foreign half.
    #[test]
    fn seal_citing_record_standalone_when_no_head() {
        let key = crate::producer::keys::KeyPair::generate();
        let foreign = "e".repeat(64);
        let prov = sample_prov(&foreign);
        let capsule = seal_citing_record(&prov, None, &key.signing_key).unwrap();
        assert!(capsule.get("chain").is_none());
        assert_eq!(capsule["assurance"]["ledger_mode"], json!("standalone"));
        assert_eq!(capsule["references"][0]["digest"], json!(foreign));
        assert_eq!(
            capsule["capsule_id"].as_str().unwrap(),
            compute_capsule_id(&capsule).unwrap()
        );
    }

    /// Tampering the cited digest post-seal breaks the recomputed capsule_id --
    /// the reference genuinely participates in the id (the {{xref}} rule).
    #[test]
    fn tampering_the_cited_digest_breaks_the_capsule_id() {
        let key = crate::producer::keys::KeyPair::generate();
        let foreign = "f".repeat(64);
        let prov = sample_prov(&foreign);
        let mut capsule = seal_citing_record(&prov, None, &key.signing_key).unwrap();
        let stored = capsule["capsule_id"].as_str().unwrap().to_string();
        capsule["references"][0]["digest"] = json!("0".repeat(64));
        assert_ne!(stored, compute_capsule_id(&capsule).unwrap());
    }

    fn sample_inclusion<'a>(half: &'a str) -> InclusionCitation<'a> {
        InclusionCitation {
            half_capsule_id: half,
            received_from: "node-b",
            via: "push",
            received_at: "2026-09-27T00:00:00Z",
            leaf_index: 6,
            mmr_size: 11,
            checkpoint_digest: "1".repeat(64).leak(),
            inclusion_proof_digest: "2".repeat(64).leak(),
        }
    }

    /// The inclusion citing record: one `references[]` entry per cited
    /// artifact (proof, checkpoint), both `counterparty_inclusion`, chained
    /// with `follows`, and it names the held half in its own block -- never
    /// by re-citing the half (that is the earlier `counterparty_half` record).
    #[test]
    fn seal_inclusion_citing_record_cites_proof_and_checkpoint() {
        let key = crate::producer::keys::KeyPair::generate();
        let head = "c".repeat(64);
        let half = "d".repeat(64);
        let capsule =
            seal_inclusion_citing_record(&sample_inclusion(&half), Some(&head), &key.signing_key)
                .unwrap();

        assert_eq!(
            capsule["capsule_id"].as_str().unwrap(),
            compute_capsule_id(&capsule).unwrap()
        );
        assert_eq!(capsule["chain"]["parent_capsule_id"], json!(head));
        assert_eq!(capsule["chain"]["relation"], json!("follows"));
        let refs = capsule["references"].as_array().unwrap();
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0]["type"], json!(REFERENCE_TYPE_INCLUSION_PROOF));
        assert_eq!(refs[0]["digest"], json!("2".repeat(64)));
        assert_eq!(refs[1]["type"], json!(REFERENCE_TYPE_CHECKPOINT));
        assert_eq!(refs[1]["digest"], json!("1".repeat(64)));
        for reference in refs {
            assert_eq!(
                reference["citation_purpose"],
                json!("counterparty_inclusion")
            );
            assert_ne!(
                reference["digest"],
                json!(half),
                "the half itself is never re-cited"
            );
        }
        let block = &capsule["model_attestation"]["compute_attestation"]["counterparty_inclusion"];
        assert_eq!(block["half_capsule_id"], json!(half));
        assert_eq!(block["leaf_index"], json!(6));
        assert_eq!(block["mmr_size"], json!(11));
        assert_eq!(capsule["action_type"], json!("fyi"));
    }

    /// The half-citation refactor onto `seal_local_citation` kept the
    /// `counterparty_half` record's shape: same keys at every level.
    #[test]
    fn half_and_inclusion_citations_share_one_header_shape() {
        let key = crate::producer::keys::KeyPair::generate();
        let half = "d".repeat(64);
        let a = seal_citing_record(&sample_prov(&half), None, &key.signing_key).unwrap();
        let b =
            seal_inclusion_citing_record(&sample_inclusion(&half), None, &key.signing_key).unwrap();
        let keys = |v: &Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
        assert_eq!(keys(&a), keys(&b));
        assert_eq!(a["assurance"], b["assurance"]);
        assert_eq!(a["disposition"], b["disposition"]);
    }

    /// An owner-maintenance record is a well-formed format-4 capsule: chained
    /// `"follows"` onto the head, its facts under
    /// `compute_attestation.owner_maintenance` (covered by the id -- tampering
    /// a count breaks it), and no `references` unless it starts a new history.
    #[test]
    fn seal_owner_maintenance_record_chains_and_covers_its_facts() {
        let key = crate::producer::keys::KeyPair::generate();
        let head = "c".repeat(64);
        let mut facts = Map::new();
        facts.insert("deleted_count".into(), json!(3));
        facts.insert("kind".into(), json!("an attempt to override the kind"));
        let action = OwnerMaintenance {
            kind: "stored_text_deleted",
            facts,
            prior_history_head: None,
        };
        let mut capsule =
            seal_owner_maintenance_record(&action, Some(&head), &key.signing_key).unwrap();

        let stored = capsule["capsule_id"].as_str().unwrap().to_string();
        assert_eq!(stored, compute_capsule_id(&capsule).unwrap());
        assert_eq!(capsule["chain"]["parent_capsule_id"], json!(head));
        assert_eq!(capsule["chain"]["relation"], json!("follows"));
        let block = &capsule["model_attestation"]["compute_attestation"][OWNER_MAINTENANCE_BLOCK];
        assert_eq!(
            block["kind"],
            json!("stored_text_deleted"),
            "facts never override the kind"
        );
        assert_eq!(block["deleted_count"], json!(3));
        assert_eq!(capsule["assurance"]["effect_mode"], json!("not_applicable"));
        assert!(capsule.get("references").is_none());
        assert!(capsule.get("signature").is_some());
        assert!(crate::producer::timestamp::is_minute_granular(
            capsule["timestamp"].as_str().unwrap()
        ));
        let nonce = capsule["model_attestation"]["compute_attestation"][STORE_NONCE_FIELD]
            .as_str()
            .unwrap();
        assert!(
            nonce.len() == 64
                && nonce
                    .chars()
                    .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
        );

        capsule["model_attestation"]["compute_attestation"][OWNER_MAINTENANCE_BLOCK]
            ["deleted_count"] = json!(0);
        assert_ne!(stored, compute_capsule_id(&capsule).unwrap());
    }

    /// A new history's first record is standalone and cites the prior
    /// history's last record by CPB typed digest (`prior_history`).
    #[test]
    fn history_started_record_is_standalone_and_cites_the_prior_head() {
        let key = crate::producer::keys::KeyPair::generate();
        let prior = "a".repeat(64);
        let action = OwnerMaintenance {
            kind: "history_started",
            facts: Map::new(),
            prior_history_head: Some(&prior),
        };
        let capsule = seal_owner_maintenance_record(&action, None, &key.signing_key).unwrap();
        assert!(capsule.get("chain").is_none());
        assert_eq!(capsule["assurance"]["ledger_mode"], json!("standalone"));
        assert_eq!(capsule["references"][0]["digest"], json!(prior));
        assert_eq!(
            capsule["references"][0]["citation_purpose"],
            json!(CITATION_PURPOSE_PRIOR_HISTORY)
        );
        assert_eq!(
            capsule["capsule_id"].as_str().unwrap(),
            compute_capsule_id(&capsule).unwrap()
        );
    }

    // -------------------------------------------------------------------
    // seal_settlement_record
    // -------------------------------------------------------------------

    fn sample_settlement<'a>(payment_hash: Option<&'a str>) -> SettlementObservation<'a> {
        SettlementObservation {
            exchange_id: "ex-paid-1",
            event_ref: "0f".repeat(32).leak(),
            terms_digest: "1e".repeat(32).leak(),
            phase: "output_settlement_observed",
            source: "wallet_reported",
            settlement: Some("terminal"),
            segment: Some(1),
            payment_hash,
            amount_msat: 123457,
            observed_by: "payer",
            tokens: None,
            credited_msat: None,
            fee_msat: None,
        }
    }

    /// A provider's record names its side and carries the delivered-token
    /// watermark; a payer's names the payer and carries no `tokens`.
    #[test]
    fn seal_settlement_record_names_the_observing_side_and_carries_tokens() {
        let key = crate::producer::keys::KeyPair::generate();
        let block = |ev: &SettlementObservation| {
            seal_settlement_record(ev, None, &key.signing_key).unwrap()["model_attestation"]
                ["compute_attestation"]["x-mesh-settlement-v1"]
                .clone()
        };
        let delivered = SettlementObservation {
            phase: "delivered",
            source: "provider_asserted",
            settlement: None,
            segment: None,
            amount_msat: 0,
            observed_by: "provider",
            tokens: Some(42),
            ..sample_settlement(None)
        };
        let provider = block(&delivered);
        assert_eq!(provider["observed_by"], json!("provider"));
        assert_eq!(provider["tokens"], json!(42));
        let payer = block(&sample_settlement(None));
        assert_eq!(payer["observed_by"], json!("payer"));
        assert!(payer.get("tokens").is_none());
    }

    /// A settlement's amounts as the wallet reported them ride on the record
    /// when the host's event carries them; a record from an event without
    /// them is exactly what it was before.
    #[test]
    fn seal_settlement_record_carries_the_wallets_credited_and_fee_when_given() {
        let key = crate::producer::keys::KeyPair::generate();
        let block = |ev: &SettlementObservation| {
            seal_settlement_record(ev, None, &key.signing_key).unwrap()["model_attestation"]
                ["compute_attestation"]["x-mesh-settlement-v1"]
                .clone()
        };
        let with_wallet = SettlementObservation {
            observed_by: "provider",
            credited_msat: Some(123450),
            fee_msat: Some(7),
            ..sample_settlement(Some("ab"))
        };
        let sealed = block(&with_wallet);
        assert_eq!(sealed["credited_msat"], json!(123450));
        assert_eq!(sealed["fee_msat"], json!(7));
        let plain = block(&sample_settlement(Some("ab")));
        assert!(plain.get("credited_msat").is_none());
        assert!(plain.get("fee_msat").is_none());
        let mut keys: Vec<&String> = plain.as_object().unwrap().keys().collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "amount_msat",
                "channel",
                "event_ref",
                "exchange_id",
                "observed_by",
                "payment_hash",
                "phase",
                "segment",
                "settlement",
                "source",
                "terms_digest",
                "v"
            ]
        );
    }

    /// Every observed value lands verbatim under
    /// `compute_attestation["x-mesh-settlement-v1"]`, the record chains onto
    /// the head with `"follows"`, carries no `references`, and its id recomputes.
    #[test]
    fn seal_settlement_record_copies_every_field_verbatim_and_chains() {
        let key = crate::producer::keys::KeyPair::generate();
        let head = "c".repeat(64);
        let hash = "bb".repeat(32);
        let ev = sample_settlement(Some(&hash));
        let capsule = seal_settlement_record(&ev, Some(&head), &key.signing_key).unwrap();

        assert_eq!(
            capsule["capsule_id"].as_str().unwrap(),
            compute_capsule_id(&capsule).unwrap()
        );
        assert_eq!(capsule["chain"]["parent_capsule_id"], json!(head));
        assert_eq!(capsule["chain"]["relation"], json!("follows"));
        assert!(capsule.get("references").is_none());
        assert_eq!(capsule["action_type"], json!("fyi"));
        assert_eq!(
            capsule["action_id"],
            json!(format!("mesh-poc/settlement/ex-paid-1/{}", "0f".repeat(32)))
        );
        assert_eq!(
            capsule["model_attestation"]["model_id"],
            json!("n/a-settlement-observation")
        );
        let obs = &capsule["model_attestation"]["compute_attestation"]["x-mesh-settlement-v1"];
        assert_eq!(
            obs,
            &json!({
                "v": 1,
                "observed_by": "payer",
                "channel": "payment.lifecycle.v1",
                "exchange_id": "ex-paid-1",
                "event_ref": "0f".repeat(32),
                "terms_digest": "1e".repeat(32),
                "phase": "output_settlement_observed",
                "source": "wallet_reported",
                "settlement": "terminal",
                "segment": 1,
                "payment_hash": hash,
                "amount_msat": 123457,
            })
        );
        assert!(capsule.get("signature").is_some());
        assert!(capsule.get("key_id").is_some());
    }

    /// Null optional fields are omitted, never written as `null`; no head ->
    /// standalone.
    #[test]
    fn seal_settlement_record_omits_absent_optionals_and_is_standalone_without_head() {
        let key = crate::producer::keys::KeyPair::generate();
        let ev = SettlementObservation {
            phase: "terms_accepted",
            source: "payer_asserted",
            settlement: None,
            segment: None,
            ..sample_settlement(None)
        };
        let capsule = seal_settlement_record(&ev, None, &key.signing_key).unwrap();
        let obs = capsule["model_attestation"]["compute_attestation"]["x-mesh-settlement-v1"]
            .as_object()
            .unwrap();
        for absent in ["settlement", "segment", "payment_hash", "tokens"] {
            assert!(
                !obs.contains_key(absent),
                "{absent} must be omitted when null"
            );
        }
        assert!(capsule.get("chain").is_none());
        assert_eq!(capsule["assurance"]["ledger_mode"], json!("standalone"));
    }

    /// A settlement record takes the same seal path as every other local
    /// record (Evidence Layer -00 §12.1): a fresh 256-bit store nonce beside
    /// the observation, and a minute-granular timestamp. Sealing the same
    /// observation twice gives two different records.
    #[test]
    fn seal_settlement_record_carries_a_store_nonce_and_a_minute_timestamp() {
        let key = crate::producer::keys::KeyPair::generate();
        let payment_hash = "ab".repeat(32);
        let ev = sample_settlement(Some(&payment_hash));
        let first = seal_settlement_record(&ev, None, &key.signing_key).unwrap();
        let second = seal_settlement_record(&ev, None, &key.signing_key).unwrap();

        let compute = &first["model_attestation"]["compute_attestation"];
        let nonce = compute[STORE_NONCE_FIELD]
            .as_str()
            .expect("store_nonce is sealed");
        assert_eq!(nonce.len(), 64);
        assert!(nonce
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert_ne!(
            second["model_attestation"]["compute_attestation"][STORE_NONCE_FIELD],
            compute[STORE_NONCE_FIELD],
            "a fresh nonce per record"
        );
        assert_ne!(first["capsule_id"], second["capsule_id"]);
        assert!(
            compute[SETTLEMENT_EXTENSION_KEY].is_object(),
            "the observation is still there"
        );

        let timestamp = first["timestamp"].as_str().unwrap();
        assert_eq!(
            crate::producer::timestamp::coarsen_to_minute(timestamp).as_deref(),
            Some(timestamp),
            "the record timestamp is already whole minutes"
        );
    }

    /// An amount the JCS cannot represent losslessly is refused, never sealed.
    #[test]
    fn seal_settlement_record_refuses_an_unsafe_amount() {
        let key = crate::producer::keys::KeyPair::generate();
        let ev = SettlementObservation {
            amount_msat: 1u64 << 53,
            observed_by: "payer",
            tokens: None,
            ..sample_settlement(None)
        };
        assert!(matches!(
            seal_settlement_record(&ev, None, &key.signing_key),
            Err(SealError::Jcs(
                crate::producer::jcs::JcsError::UnsafeInteger(_)
            ))
        ));
    }
}
