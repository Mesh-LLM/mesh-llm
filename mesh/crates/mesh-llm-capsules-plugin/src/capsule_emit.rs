//! Seals the plugin's records (COSE-sign -> chain -> ledger). Every ALLOWED
//! chat-completion exchange this plugin itself serves is turned into a
//! signed, chained, ledgered AAC (`x-mesh-poc-v1` mapping) --
//! `effect_request_digest`/`effect_response_digest` are the canonical
//! JSON-DIGEST (RFC 8785 JCS, spec §5.1) of the parsed request/response
//! body, the same digest the Agent Action Capsule reference implementation
//! computes -- NOT a raw hash of the wire bytes, so reserializing the
//! identical semantic content (key order, whitespace) does not change the
//! digest, and it stays comparable across implementations. Mutating the
//! actual content still changes `capsule_id`.

use crate::lifecycle_channel::{dispatch_path_wire_value, role_for_dispatch_path, DispatchPath};
use crate::producer::capsule::{
    seal, seal_owner_maintenance_record, CapsuleInput, ChainLink, HostBinding, MeshPocV1,
    OwnerMaintenance, ServingProvenance, TokenUsage,
};
use crate::producer::cose::{build_signed_statement, SignedStatementInput};
use crate::producer::jcs;
use crate::producer::keys::{self, KeyPair};
use crate::producer::ledger::{Ledger, LedgerEntry};
use crate::producer::sequence::SequenceCounterStore;
use crate::producer::timestamp::utc_now_minute;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

/// Derive `(role, observation_point)` for an exchange this plugin only
/// OBSERVED on the `openai.exchange.v1` channel (2026-09-06 role ruling).
///
/// `dispatch_path` is the AUTHORITATIVE signal: `RemoteMesh` means this node
/// routed the exchange to a peer (role `"requested"`), `TypedFrontend`/
/// `RawProxy` mean this node served it (`"served"`), and an unrecognized
/// value is labeled `"unknown"` -- NEVER silently defaulted to `"served"`,
/// which was the actual bug (a missing/unrecognized field silently became a
/// claim). Keying on `dispatch_path` rather than the fork-only
/// `served_by_node_id` enrichment means this plugin labels correctly against
/// bare upstream `main` the day mesh-llm#1668 merges, not only against the
/// fork.
///
/// `served_by_node_id` (fork-only enrichment; `None` on bare upstream) is
/// then a CONSISTENCY CHECK, applied only when BOTH it and `self_node_id`
/// are present -- agreement seals the `dispatch_path`-derived role
/// unchanged; disagreement seals `role: "conflict"` instead of guessing
/// which signal is right. Both raw facts (`dispatch_path` and
/// `served_by_node_id`) still ride the sealed record (see
/// `ServingProvenance::dispatch_path`/`served_by_node_id`) so a reader never
/// has to take the resolved label's word for it.
///
/// `self_node_id` (2026-09-06 domain fix) is this node's own mesh identity
/// in the SAME domain as `served_by_node_id` -- LEARNED from a prior
/// locally-served terminal event (see [`LearnedSelfNodeId`]), never a
/// compile-time plugin-type label like `PLUGIN_ID`. `None` until learned
/// (fresh node, or no locally-served event observed yet): the check is then
/// SKIPPED (same as the `served_by_node_id: None` arm) rather than compared
/// against a placeholder that can never legitimately match, which was the
/// bug (`PLUGIN_ID` vs mesh node id domain confusion sealed `role:
/// "conflict"` on every locally-served exchange).
///
/// `observation_point` is complementary vantage provenance (ruling option
/// C): `Some("client_egress")` on the `RemoteMesh` path only, `None`
/// elsewhere -- independent of `role`, never a restatement of it.
fn role_and_observation_point(
    dispatch_path: &DispatchPath,
    served_by_node_id: Option<&str>,
    self_node_id: Option<&str>,
) -> (String, Option<String>) {
    let dispatch_role = role_for_dispatch_path(dispatch_path);
    let observation_point =
        matches!(dispatch_path, DispatchPath::RemoteMesh).then(|| "client_egress".to_string());

    let role = if dispatch_role == "unknown" {
        "unknown".to_string()
    } else {
        match (served_by_node_id, self_node_id) {
            // Both signals present -- the actual consistency check.
            (Some(served_by), Some(self_id)) => {
                let dispatch_implies_this_node_served = dispatch_role == "served";
                let node_id_implies_this_node_served = served_by == self_id;
                if dispatch_implies_this_node_served == node_id_implies_this_node_served {
                    dispatch_role.to_string()
                } else {
                    "conflict".to_string()
                }
            }
            // No fork enrichment on this event, or this node's own mesh
            // identity is not yet learned -- the dispatch_path-derived role
            // is the only signal available; never compared against a
            // placeholder.
            _ => dispatch_role.to_string(),
        }
    };
    (role, observation_point)
}

/// Recursively replace JSON floats with their exact decimal-string form,
/// as the Agent Action Capsule reference implementation does:
/// the JSON-DIGEST (spec §5.1) refuses any float in a digest-bearing value
/// (float serialization isn't cross-implementation deterministic), and
/// OpenAI-shaped chat request/response bodies are full of floats
/// (temperature, top_p, penalties, ...). `python_repr_f64` is not a claim of
/// byte-parity with Python's `repr()` for every float (neither the Python
/// docstring it mirrors makes that claim) -- it is deterministic for this
/// plugin's own digest and matches Python for the ordinary decimal range
/// chat-completion parameters actually use.
fn stringify_floats(value: Value) -> Value {
    match value {
        Value::Number(n) => match n.as_f64() {
            Some(f) if n.is_f64() && !(n.is_i64() || n.is_u64()) => {
                Value::String(python_repr_f64(f))
            }
            _ => Value::Number(n),
        },
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(k, v)| (k, stringify_floats(v)))
                .collect(),
        ),
        Value::Array(arr) => Value::Array(arr.into_iter().map(stringify_floats).collect()),
        other => other,
    }
}

fn python_repr_f64(f: f64) -> String {
    let s = format!("{f}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

/// The JSON-DIGEST of a value, floats stringified first: the digest a signed
/// record's `request_digest` / `response_digest` holds for a body, and the
/// referee's digests over its own inputs (`crate::referee`).
pub(crate) fn digest_json(value: &Value) -> Result<String, jcs::JcsError> {
    jcs::json_digest(&stringify_floats(value.clone()))
}

/// The canonical JSON-DIGEST of a request/response body: parse as JSON,
/// stringify floats, then `jcs::json_digest`. See the module docs for why
/// this replaces a raw hash of the wire bytes.
fn canonical_body_digest(bytes: &[u8]) -> anyhow::Result<String> {
    let value: Value = serde_json::from_slice(bytes)?;
    Ok(jcs::json_digest(&stringify_floats(value))?)
}

/// Extract real token accounting from an OpenAI-shaped response body's `usage`
/// object (`openai-frontend`'s `Usage`: `prompt_tokens` / `completion_tokens` /
/// `total_tokens`). Returns `None` — never a fabricated zero — when the served
/// body carried no well-formed `usage` (e.g. an allow-stub or error body). This
/// is the only honest source of usage the plugin has: the counts come from the
/// response the host actually produced, not from anything this plugin invents.
fn parse_usage(response_bytes: &[u8]) -> Option<TokenUsage> {
    let value: Value = serde_json::from_slice(response_bytes).ok()?;
    let usage = value.get("usage")?;
    let prompt_tokens = usage.get("prompt_tokens")?.as_u64()?;
    let completion_tokens = usage.get("completion_tokens")?.as_u64()?;
    // total_tokens: prefer the server-reported value; fall back to the sum only
    // when the body omitted it (still a real derivation, not an invention).
    let total_tokens = usage
        .get("total_tokens")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| prompt_tokens.saturating_add(completion_tokens));
    Some(TokenUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
    })
}

/// The generation-parameter (sampling knob) keys carried verbatim in the
/// capsule, in step with the reference implementation's
/// `GENERATION_PARAM_KEYS`. These are the settings the CLIENT asked for in the
/// request -- requested, not proven-effective -- and are legible policy values
/// (not prompt content), so they ride as-is rather than digested. Kept
/// byte-identical to the Python list so both capture paths seal the SAME param
/// set.
const GENERATION_PARAM_KEYS: &[&str] = &[
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "max_tokens",
    "max_completion_tokens",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "repeat_penalty",
    "stop",
];

/// Lift the generation parameters the client ACTUALLY sent in this request body
/// into a map for the capsule, by the allowlist above. Honest-by-absence: a
/// key that was
/// not present in the request (or was JSON `null`) is OMITTED, never defaulted
/// to a fabricated value -- so a request that carried only `temperature` seals
/// exactly `temperature`, and the old hardcoded `temperature=0.0` is gone.
/// Values are stringified through the same `stringify_floats` used for the
/// digest path, so `0.7` -> `"0.7"` (float) while integers like `seed`/`n`
/// stay JSON numbers -- the reference implementation's convention, keeping
/// the sealed shape stable across implementations.
/// A non-JSON or non-object body yields an empty map (no params claimed).
fn parse_generation_parameters(request_bytes: &[u8]) -> Map<String, Value> {
    let mut out = Map::new();
    let Ok(Value::Object(request)) = serde_json::from_slice::<Value>(request_bytes) else {
        return out;
    };
    for &key in GENERATION_PARAM_KEYS {
        match request.get(key) {
            Some(value) if !value.is_null() => {
                out.insert(key.to_string(), stringify_floats(value.clone()));
            }
            _ => {}
        }
    }
    out
}

/// The OPTIONAL labeled sub-digests over one response body: `tool_calls_digest`
/// and `reasoning_digest`, each the canonical `jcs::json_digest` (plain JCS,
/// matching the capsule-ledger package's `digest_conversation_exchange`) of
/// the flattened `tool_calls` /
/// `reasoning_content` across the response's assistant message(s). Each is
/// `None` -- and then ABSENT from the sealed capsule, never a fabricated digest
/// over `[]` -- when the response carried none. Used by the plugin-served path,
/// which holds the response bytes directly (the host-served observe path instead
/// binds the digests the host forwarded). No float-stringification: tool_calls /
/// reasoning are strings/structural JSON with no floats, so plain JCS matches
/// the Python reference exactly.
fn output_sub_digests(response_bytes: &[u8]) -> (Option<String>, Option<String>) {
    let Ok(value) = serde_json::from_slice::<Value>(response_bytes) else {
        return (None, None);
    };
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut reasoning: Vec<Value> = Vec::new();
    if let Some(choices) = value.get("choices").and_then(Value::as_array) {
        for choice in choices {
            let Some(message) = choice.get("message") else {
                continue;
            };
            if let Some(tcs) = message.get("tool_calls").and_then(Value::as_array) {
                tool_calls.extend(tcs.iter().cloned());
            }
            if let Some(r) = message.get("reasoning_content").filter(|r| !r.is_null()) {
                if !matches!(r, Value::String(s) if s.is_empty()) {
                    reasoning.push(r.clone());
                }
            }
        }
    }
    let tool_calls_digest = (!tool_calls.is_empty())
        .then(|| jcs::json_digest(&Value::Array(tool_calls)))
        .transpose()
        .ok()
        .flatten();
    let reasoning_digest = (!reasoning.is_empty())
        .then(|| jcs::json_digest(&Value::Array(reasoning)))
        .transpose()
        .ok()
        .flatten();
    (tool_calls_digest, reasoning_digest)
}

/// Operator opt-in for sealing the serving host's name. The sealed body is
/// pushed at completion to every counterparty, so the machine name is
/// withheld unless the operator sets this to `1`, `true` or `on`.
pub const ENV_SEAL_HOSTNAME: &str = "CAPSULES_SEAL_HOSTNAME";

fn hostname_opt_in() -> bool {
    hostname_opt_in_for(crate::settings::var(ENV_SEAL_HOSTNAME).ok().as_deref())
}

fn hostname_opt_in_for(raw: Option<&str>) -> bool {
    matches!(raw, Some("1" | "true" | "on"))
}

/// The hostname to seal: the host-reported value only when the operator
/// opted in; otherwise `None`, which the producer omits from the record.
fn sealed_hostname(host: &HostProvenance, opted_in: bool) -> Option<String> {
    if opted_in {
        host.hostname.clone()
    } else {
        None
    }
}

/// The signed statement's content type. Its `profile` is the spec version the
/// sealed record itself declares, so the two cannot drift apart.
static CAPSULE_CONTENT_TYPE: LazyLock<String> = LazyLock::new(|| {
    format!(
        "application/vnd.agent-action-capsule+json; profile={}",
        crate::producer::capsule::SPEC_VERSION
    )
});

/// Persisted cache of this node's own mesh identity (2026-09-06 self-id
/// domain fix), beside the ledger at `<data_dir>/learned_self_node_id.json`
/// (same restart-safe, best-effort convention as `SequenceCounterStore`).
///
/// `PLUGIN_ID` (`"capsules"`) is a compile-time plugin-TYPE label,
/// not a mesh node id, and this plugin has no other source for its own real
/// mesh identity -- the host never sends one directly. But the host DOES
/// tell us, on every locally-served terminal event
/// (`dispatch_path` in {`TypedFrontend`, `RawProxy`}) that carries the
/// fork's `served_by_node_id` enrichment, exactly who served it -- and since
/// this plugin observed that event as locally-served, that value IS this
/// node's own asserted mesh identity. So: LEARN it from the first such
/// observation; a later locally-served event naming a DIFFERENT id is a
/// real identity ROTATION (replace + log), never a conflict -- conflict is
/// reserved for a REMOTE-dispatched event that claims our own learned id
/// (see `role_and_observation_point`).
struct LearnedSelfNodeId {
    path: PathBuf,
    value: Option<String>,
}

impl LearnedSelfNodeId {
    fn open(path: PathBuf) -> Self {
        let value = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Option<String>>(&bytes).ok())
            .flatten();
        Self { path, value }
    }

    fn get(&self) -> Option<&str> {
        self.value.as_deref()
    }

    /// Learn, confirm, or rotate this node's self id from a locally-served
    /// terminal event's `served_by_node_id`. No-op when unchanged; logs and
    /// replaces on rotation; persists on any change. A persist failure is
    /// logged, never panics -- the in-memory value still updates so the
    /// current process behaves correctly even if the disk write failed.
    fn observe(&mut self, served_by_node_id: &str) {
        match self.value.as_deref() {
            Some(current) if current == served_by_node_id => {}
            Some(current) => {
                tracing::warn!(
                    previous_self_node_id = current,
                    new_self_node_id = served_by_node_id,
                    "learned self node id rotated"
                );
                self.value = Some(served_by_node_id.to_string());
                self.save();
            }
            None => {
                tracing::info!(self_node_id = served_by_node_id, "learned self node id");
                self.value = Some(served_by_node_id.to_string());
                self.save();
            }
        }
    }

    fn save(&self) {
        let Ok(bytes) = serde_json::to_vec(&self.value) else {
            return;
        };
        let tmp = self.path.with_extension("tmp");
        if let Err(err) = fs::write(&tmp, bytes) {
            tracing::warn!(%err, "failed to write learned_self_node_id.json.tmp");
            return;
        }
        if let Err(err) = fs::rename(&tmp, &self.path) {
            tracing::warn!(%err, "failed to persist learned_self_node_id.json");
        }
    }
}

/// LOCK POISONING NOTE: every mutex here is taken with
/// `unwrap_or_else(PoisonError::into_inner)` rather than a panicking
/// `expect` -- capsule production is best-effort OBSERVABILITY on the
/// serving path, so one panicked seal (its own exchange already failed) must
/// not poison the lock and turn every later seal into a panic cascade. The
/// guarded state stays consistent across a mid-seal panic: `Ledger::append`
/// orders its writes (statement fsync before the jsonl line) exactly so a
/// torn stop is recoverable, and reload re-validates everything on open.
/// Beside the ledger: one line per cut of a torn last line.
const REPAIRS_FILE: &str = "repairs.jsonl";

pub struct CapsuleState {
    keys: KeyPair,
    ledger: Mutex<Ledger>,
    /// Where `ledger` lives, kept so an owner cleanup can re-open it
    /// (`rebuild_index`) without a second writer ever existing.
    ledger_dir: PathBuf,
    node_id: String,
    /// Per-`(self, counterparty)` monotone `seq`/`prev_seq` cache, persisted
    /// beside the ledger at `<data_dir>/sequence_counters.json` (see
    /// `crate::producer::sequence` for why this cache is never the source
    /// of truth for continuity).
    sequence_counters: Mutex<SequenceCounterStore>,
    /// This node's own mesh identity, LEARNED from observed traffic -- see
    /// [`LearnedSelfNodeId`]. Distinct from `node_id` above (a plugin-type
    /// label used for the ledger issuer / sequence-counter key), which is
    /// NEVER used as a stand-in for the mesh node id domain.
    learned_self_node_id: Mutex<LearnedSelfNodeId>,
    /// The byte offset the ledger was cut back to when it was opened, when
    /// its last line was torn (a write cut short, e.g. by a crash): that line
    /// is gone. Logged, and shown in the records status.
    torn_write_cut_at: Option<u64>,
    /// Settlement-records legs waiting on the other side (in memory only).
    settlement_legs: crate::settlement_legs::Legs,
    /// See [`CapsuleState::observed_not_sealed`].
    observed_not_sealed: std::sync::atomic::AtomicU64,
    /// Where a TDX quote for a record comes from (`producer::tee_attest`).
    quote_source: Box<dyn crate::producer::tee_attest::QuoteSource>,
}

pub struct EmittedCapsule {
    pub capsule_id: String,
    pub capsule: Value,
}

/// The host's serving-provenance facts for the served model, captured from the
/// `openai.exchange.v1` terminal event (mirror `lifecycle_channel::
/// HostServingProvenance`). Every field is `Option`: a fact the host did not
/// report stays `None` and lands as an honest `unknown`/`null` in the capsule —
/// never fabricated. Passed as owned data (not a borrow of the channel state)
/// so the emit path holds no lock on the lifecycle store.
#[derive(Clone, Default)]
pub struct HostProvenance {
    pub served_by_node_id: Option<String>,
    pub hostname: Option<String>,
    pub quantization: Option<String>,
    pub architecture: Option<String>,
    pub context_length: Option<u32>,
    pub parameter_size: Option<String>,
    pub layer_count: Option<u32>,
    pub model_identity_hash: Option<String>,
    /// SHA-256 of the served GGUF's file BYTES (mirror `lifecycle_channel::
    /// HostServingProvenance::weights_digest`) -- a different fact from
    /// `model_identity_hash`, never a replacement for it.
    pub weights_digest: Option<String>,
    pub model_canonical_ref: Option<String>,
    pub model_revision: Option<String>,
    pub gpu: Option<String>,
    pub vram_bytes: Option<u64>,
    pub is_soc: Option<bool>,
    /// The mesh node that asked this node to serve the exchange (mirror
    /// `lifecycle_channel::HostServingProvenance::requested_by_node_id`):
    /// the host's QUIC-authenticated remote id, never the request's word.
    pub requested_by_node_id: Option<String>,
}

/// One admitted exchange to seal into a capsule — the raw request/response
/// bytes (digested, never stored raw) plus the provenance metadata the host
/// exposed for it. Grouped into a struct so the provenance surface can grow
/// without churning the `emit_for_exchange` signature.
pub struct ExchangeRecord<'a> {
    pub model: &'a str,
    pub client_nonce: Option<&'a str>,
    pub request_bytes: &'a [u8],
    pub response_bytes: &'a [u8],
    pub latency_ms: f64,
    /// Stable per-exchange correlation id — the host's `exchange_id` /
    /// response `id` / `x-request-id` lineage, so the record ties back to the
    /// host's own terminal-event log. `None` -> `"unknown"`, never faked.
    pub exchange_id: Option<&'a str>,
    /// Requesting party / client identity beyond the (optional) client nonce.
    /// `None` -> `"unknown"`, never invented.
    pub requesting_party: Option<&'a str>,
    /// The host's serving provenance for this model, captured from the
    /// `openai.exchange.v1` terminal event. `None` when the host has published
    /// none yet — quantization/hardware/model-digest then stay honest defaults.
    pub host_provenance: Option<HostProvenance>,
}

impl CapsuleState {
    /// Loads (or creates, on first run) a persistent Ed25519 signing key and
    /// opens the durable local ledger under `data_dir` -- both restart-safe,
    /// mirroring the acceptance already proven in isolation by
    /// `mesh-rust-capsule-production-m2`'s `chain_ledger_conformance` test.
    pub fn open(data_dir: &Path, node_id: impl Into<String>) -> anyhow::Result<Self> {
        let keys = keys::load_or_create(&data_dir.join("keys"))?;
        let ledger_dir = data_dir.join("ledger");
        let (ledger, report) = crate::producer::index::open_ledger(&ledger_dir)?;
        tracing::info!(recovered_entries = report.valid_entries, "ledger opened");
        if let Some(offset) = report.truncated_torn_write_at {
            tracing::warn!(
                offset,
                "the ledger's last line was torn (a write cut short); it was cut back to the last whole record"
            );
            // Kept beside the ledger, so the page still says so after the
            // next restart.
            let line = serde_json::json!({
                "at": crate::producer::timestamp::utc_now_iso8601(),
                "torn_write_cut_at": offset,
            });
            if let Err(error) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(ledger_dir.join(REPAIRS_FILE))
                .and_then(|mut f| std::io::Write::write_all(&mut f, format!("{line}\n").as_bytes()))
            {
                tracing::warn!(%error, "the ledger repair was not noted");
            }
        }
        let sequence_counters = SequenceCounterStore::open(data_dir.join("sequence_counters.json"));
        let learned_self_node_id =
            LearnedSelfNodeId::open(data_dir.join("learned_self_node_id.json"));
        Ok(Self {
            keys,
            ledger: Mutex::new(ledger),
            ledger_dir,
            node_id: node_id.into(),
            sequence_counters: Mutex::new(sequence_counters),
            learned_self_node_id: Mutex::new(learned_self_node_id),
            settlement_legs: crate::settlement_legs::Legs::default(),
            observed_not_sealed: std::sync::atomic::AtomicU64::new(0),
            torn_write_cut_at: report.truncated_torn_write_at,
            quote_source: Box::new(crate::producer::tee_attest::ConfigfsTsm::default()),
        })
    }

    /// Take TDX quotes from `source` instead of the kernel's configfs-tsm.
    #[cfg(test)]
    pub fn set_quote_source(&mut self, source: Box<dyn crate::producer::tee_attest::QuoteSource>) {
        self.quote_source = source;
    }

    /// The `tee_attestation` slot for a record about to be sealed with these
    /// values. Called before the ledger lock: a quote can take a while.
    fn tee_slot(
        &self,
        agent_input_digest: Option<&str>,
        agent_output_digest: Option<&str>,
        client_nonce: Option<&str>,
        client_nonce_source: &str,
    ) -> serde_json::Value {
        crate::producer::tee_attest::slot_for(
            self.quote_source.as_ref(),
            agent_input_digest,
            agent_output_digest,
            client_nonce.map(|value| crate::producer::tee_attest::ClientNonce {
                value,
                source: client_nonce_source,
            }),
            &self.keys.signing_key.verifying_key().to_bytes(),
        )
    }

    /// Where the ledger was cut back to when it was opened, if its last line
    /// was torn.
    pub fn torn_write_cut_at(&self) -> Option<u64> {
        self.torn_write_cut_at
    }

    /// Every cut of a torn last line this ledger has had, oldest first:
    /// `{at, torn_write_cut_at}`.
    pub fn ledger_repairs(&self) -> Vec<serde_json::Value> {
        std::fs::read_to_string(self.ledger_dir.join(REPAIRS_FILE))
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    }

    /// This node's settlement-records state ([`crate::settlement_legs`]).
    pub fn settlement_legs(&self) -> &crate::settlement_legs::Legs {
        &self.settlement_legs
    }

    /// Observed host-served exchanges this node failed to seal since start.
    /// A refusal is counted and logged, never dropped silently.
    pub fn observed_not_sealed(&self) -> u64 {
        self.observed_not_sealed
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// This node's own mesh identity as LEARNED so far (see
    /// [`LearnedSelfNodeId`]) -- `None` until a locally-served terminal
    /// event has carried the fork's `served_by_node_id` enrichment.
    fn learned_self_node_id(&self) -> Option<String> {
        self.learned_self_node_id
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get()
            .map(str::to_string)
    }

    /// Read by the `ledger-fetch/1` responder (`ledger_fetch_bridge`) so a
    /// peer's fetch response carries the key its own COSE_Sign1 verifies
    /// against, without a second attestation wrapper.
    pub fn public_key_pem(&self) -> String {
        self.keys.public_key_pem()
    }

    /// This node's own already-loaded signing key -- for a caller (the
    /// checkpoint cadence task) that needs to sign something ELSE with the
    /// SAME node identity that signs capsules, through the `&dyn
    /// CheckpointSigner` trait indirection (`ed25519_dalek::SigningKey`
    /// implements it), never a second, purpose-minted key.
    pub fn signing_key(&self) -> &ed25519_dalek::SigningKey {
        &self.keys.signing_key
    }

    /// The raw Ed25519 public key every record this node seals verifies
    /// against (its `key_id`): the key it announces (`crate::peer_keys::own`).
    pub fn public_key(&self) -> [u8; 32] {
        self.keys.signing_key.verifying_key().to_bytes()
    }

    /// Pad the ledger to a multiple of `bucket` lines under the SAME writer
    /// lock every seal takes, so a real record never lands between two
    /// padding records (Evidence Layer -00 §12.1; see
    /// `crate::producer::padding`). Returns the padded line count.
    pub fn pad_ledger_to_bucket(&self, bucket: u64) -> anyhow::Result<u64> {
        let mut ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Ok(ledger.pad_to_bucket(bucket, &self.keys.signing_key, &self.node_id)?)
    }

    /// Looks up one previously-sealed capsule by id -- the `ledger-fetch/1`
    /// responder's only data source. A thin passthrough: whatever `Ledger::
    /// lookup` returns (or doesn't) is exactly what goes out, nothing
    /// re-derived or fabricated here.
    pub fn lookup(&self, capsule_id: &str) -> anyhow::Result<Option<LedgerEntry>> {
        Ok(self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .lookup(capsule_id)?)
    }

    pub fn chain_head(&self) -> Option<String> {
        self.ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .chain_head()
            .map(str::to_string)
    }

    /// Seal, sign, chain, and ledger one admitted exchange this plugin's own
    /// `/v1/chat/completions` handler just served.
    pub fn emit_for_exchange(&self, exchange: &ExchangeRecord) -> anyhow::Result<EmittedCapsule> {
        let ExchangeRecord {
            model,
            client_nonce,
            request_bytes,
            response_bytes,
            latency_ms,
            exchange_id,
            requesting_party,
            host_provenance,
        } = exchange;
        let (
            model,
            client_nonce,
            request_bytes,
            response_bytes,
            latency_ms,
            exchange_id,
            requesting_party,
        ) = (
            *model,
            *client_nonce,
            *request_bytes,
            *response_bytes,
            *latency_ms,
            *exchange_id,
            *requesting_party,
        );
        // Honest defaults for every host-provenance fact; each is overwritten
        // only if the host actually reported it (never fabricated).
        let host = host_provenance.clone().unwrap_or_default();
        let agent_input_digest = canonical_body_digest(request_bytes)?;
        let agent_output_digest = canonical_body_digest(response_bytes)?;
        // The nonce source as this record carries it (`client_nonce_source`
        // below): a nonce is only ever the client's own on this path.
        let tee_attestation = self.tee_slot(
            Some(&agent_input_digest),
            Some(&agent_output_digest),
            client_nonce,
            "client_supplied",
        );
        let usage = parse_usage(response_bytes);
        // This path serves the response itself, so it holds the bytes -- compute
        // the OPTIONAL tool_calls/reasoning sub-digests directly (absent when the
        // model emitted none). Real digests over the real served response.
        let (tool_calls_digest, reasoning_digest) = output_sub_digests(response_bytes);

        let mut ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let chain = ledger.chain_head().map(|parent| ChainLink {
            parent_capsule_id: parent.to_string(),
            relation: "follows".to_string(),
        });

        // The REAL sampling settings the client requested, parsed straight from
        // the request body this path holds -- only the keys actually present,
        // never a fabricated default.
        let generation_parameters = parse_generation_parameters(request_bytes);

        // Runtime/binary attestation rung (task B3, extended by rung 3a): measure
        // the serving binary this node runs and sign the hash with the node key.
        // On macOS this upgrades to `os_measured` (the kernel independently
        // reports the running process's cdhash via csops(2)); elsewhere it stays
        // `self_measured`. `None` when the binary path is unresolvable/unreadable
        // -- the rung then degrades gracefully (empty evidence slot + placeholder
        // runtime field), never a fabricated hash. See
        // `crate::producer::runtime_attest` for the honesty grades and their
        // trust ceilings.
        let binary_attestation =
            crate::producer::runtime_attest::measure_self(&self.keys, utc_now_minute());

        // Provider-side pair key (history proposal §1): self = this node
        // (`served_by_node_id` below); counterparty = the requesting party,
        // from a forwarded identity header when present, else the honest
        // "unknown" bucket -- never invented.
        let requesting_party_id = requesting_party.unwrap_or("unknown").to_string();
        // Peeked, not consumed: committed only after the ledger write, so a
        // record that fails to seal leaves no `seq` gap.
        let mut sequence_counters = self
            .sequence_counters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let sequence = sequence_counters.peek(&self.node_id, &requesting_party_id);

        let input = CapsuleInput {
            action_id: format!("mesh-poc/capsule-emit-mesh-integration/{agent_input_digest}"),
            action_type: "decide".to_string(),
            operator: "capsule-emit-mesh-poc-rust".to_string(),
            developer: crate::producer::capsule::LOCAL_RECORD_DEVELOPER.to_string(),
            timestamp: utc_now_minute(),
            domain: Some("action".to_string()),
            provenance: Some("collector".to_string()),
            model_id: model.to_string(),
            provider: "mesh-llm".to_string(),
            agent_input_digest: agent_input_digest.clone(),
            agent_output_digest: Some(agent_output_digest.clone()),
            // OPTIONAL labeled sub-digests over the served response body; absent
            // when the model emitted none (never fabricated).
            tool_calls_digest,
            reasoning_digest,
            // This path serves the exchange itself -- there is no SEPARATE
            // host-native digest to bind (this plugin IS the one computing
            // `agent_input_digest`, over the same bytes, under our own
            // construction) -- so `host_binding` is not applicable here.
            host_binding: None,
            // The REAL measured serving-binary hash + its honesty grade
            // (`os_measured` on macOS, else `self_measured`) when the binary
            // was measurable; the old `0*64` placeholder ONLY when it was not
            // (graceful degradation, never a fabricated hash).
            runtime: runtime_field(&binary_attestation),
            mesh_poc: MeshPocV1 {
                client_nonce: client_nonce.unwrap_or("plugin_generated").to_string(),
                client_nonce_source: if client_nonce.is_some() {
                    "client_supplied"
                } else {
                    "plugin_generated_fallback"
                }
                .to_string(),
                // Renamed from the overclaiming `model_package_digest`: this is
                // SHA-256 of the model NAME only, not the weights/package: the
                // live plugin only has the request's model name.
                model_name_digest: hex_sha256(model.as_bytes()),
                serving_provenance: ServingProvenance {
                    // Prefer the host event's serving node id; fall back to
                    // this node's LEARNED mesh identity (real domain, never
                    // the `PLUGIN_ID` plugin-type label) when the host
                    // reported none; honest "unknown" when nothing has been
                    // learned yet -- never a fabricated identity claim.
                    served_by_node_id: host.served_by_node_id.clone().unwrap_or_else(|| {
                        self.learned_self_node_id()
                            .unwrap_or_else(|| "unknown".to_string())
                    }),
                    // This admitted exchange came through this plugin's own
                    // `/v1` handler directly -- no host envelope, so no
                    // dispatch_path signal exists on this path at all.
                    dispatch_path: None,
                    requesting_party: requesting_party_id.clone(),
                    exchange_id: exchange_id.unwrap_or("unknown").to_string(),
                    hostname: sealed_hostname(&host, hostname_opt_in()),
                    // Quantization from the host serving-provenance block when it
                    // carried one; else "unknown" — never guessed.
                    quantization: host
                        .quantization
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string()),
                    // Serving hardware from the host serving-provenance block.
                    // `hardware_device` stays None (host carries is_soc, not a
                    // cpu/cuda/metal device enum) — never fabricated.
                    hardware_gpu: host.gpu.clone(),
                    hardware_vram_bytes: host.vram_bytes,
                    hardware_device: None,
                    hardware_is_soc: host.is_soc,
                    // Model identity / fidelity from the host block.
                    architecture: host.architecture.clone(),
                    context_length: host.context_length,
                    parameter_size: host.parameter_size.clone(),
                    layer_count: host.layer_count,
                    model_identity_hash: host.model_identity_hash.clone(),
                    weights_digest: host.weights_digest.clone(),
                    model_canonical_ref: host.model_canonical_ref.clone(),
                    model_revision: host.model_revision.clone(),
                    // Real token counts from the response body's `usage`, if any.
                    usage,
                    seq: sequence.seq,
                    prev_seq: sequence.prev_seq,
                    // This path serves the exchange itself -- there is no
                    // peer half to name (this node IS the server), never a
                    // fabricated self-reference.
                    peer_capsule_id: None,
                    peer_capsule_id_provenance: None,
                    // This path is admitted directly by this plugin's own
                    // `/v1` handler -- there is no mesh terminal envelope to
                    // read a twin bracket off of, so this is never
                    // twinned on this path.
                    twin_bracket_id: None,
                    response_text_digest: None,
                },
                // This exchange was admitted and served by THIS plugin's own
                // `/v1` handler -- unambiguously "served", not a guess: there
                // is no dispatch_path signal on this path because there is no
                // ambiguity to resolve on it.
                role: "served".to_string(),
                observation_point: None,
                generation_parameters,
                latency_ms: crate::producer::capsule::committed_latency_ms(latency_ms),
                binary_attestation,
                tee_attestation: Some(tee_attestation),
            },
            effect_status: "confirmed".to_string(),
            effect_type: "inference_completion".to_string(),
            effect_request_digest: Some(agent_input_digest),
            effect_response_digest: Some(agent_output_digest),
            effect_attestation: "gate_executed".to_string(),
            disposition_decision: "accept".to_string(),
            // §5.4: disposition.approver MUST be `human` or `policy` -- the
            // plugin is the latter (an automated policy
            // engine, not a human disposer), never its own plugin name.
            disposition_approver: "policy".to_string(),
            disposition_human_disposed: false,
            disposition_verdict_class: "executed".to_string(),
            chain,
            store_nonce: crate::producer::capsule::fresh_store_nonce(),
        };

        let mut capsule = seal(&input)?;
        let capsule_id = capsule["capsule_id"]
            .as_str()
            .expect("seal() always sets capsule_id")
            .to_string();
        // Attach the inline producer-signature envelope BEFORE building the
        // detached `.cose` statement, so the ledgered capsule, the `.cose`
        // payload, and any pushed half are ONE shape (`signature`/`key_id`
        // present) -- a peer that receives just the pushed half verifies it in
        // isolation, exactly as a `capsule_emit.seal()` capsule does. Excluded
        // from the `capsule_id` preimage, so `capsule_id` is unchanged.
        crate::producer::capsule::attach_producer_envelope(&mut capsule, &self.keys.signing_key)
            .expect("seal() always sets a hex capsule_id");
        let payload = crate::producer::capsule::payload_bytes(&capsule);
        let statement = build_signed_statement(
            &SignedStatementInput {
                payload: &payload,
                issuer: &self.node_id,
                subject: &capsule_id,
                content_type: &CAPSULE_CONTENT_TYPE,
            },
            &self.keys.signing_key,
        );
        ledger.append(&capsule, &statement)?;
        sequence_counters.commit(&self.node_id, &requesting_party_id, sequence)?;

        Ok(EmittedCapsule {
            capsule_id,
            capsule,
        })
    }
}

/// A host-forwarded digest as a 64-lowercase-hex JSON-DIGEST: lowercased,
/// with a `sha256:` prefix stripped; `None` for anything else.
fn normalise_host_digest(raw: Option<&str>) -> Option<String> {
    let raw = raw?;
    let hex = raw
        .strip_prefix("sha256:")
        .unwrap_or(raw)
        .to_ascii_lowercase();
    (hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit())).then_some(hex)
}

fn hex_sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// The `compute_attestation.runtime` value for the SERVED path: the runtime/
/// model extension draft's `{name, runtime_digest, measurement_class,
/// platform_integrity}` object, from the real measured serving-binary hash +
/// its honesty grade (`os_measured` on macOS, else `self_measured`) when the
/// binary was measurable, else `name` alone (graceful degradation — never a
/// fabricated digest/class). Runtime name identifies the serving runtime
/// this plugin fronts.
fn runtime_field(att: &Option<crate::producer::runtime_attest::BinaryAttestation>) -> Value {
    const NAME: &str = "capsule-emit-mesh/mesh-llm-host-runtime";
    match att {
        Some(a) => a.runtime_value(NAME),
        None => json!({"name": NAME}),
    }
}

/// The `runtime` value for the OBSERVE path. HONESTY: the measured binary is the
/// OBSERVING plugin, NOT the host runtime that actually served the inference —
/// so the runtime name says `observer/...` to keep that distinction legible in
/// the sealed capsule. Same graceful-degradation shape on `None`.
fn observer_runtime_field(
    att: &Option<crate::producer::runtime_attest::BinaryAttestation>,
) -> Value {
    const NAME: &str = "observer/capsule-emit-mesh";
    match att {
        Some(a) => a.runtime_value(NAME),
        None => json!({"name": NAME}),
    }
}

/// One host-served exchange this plugin only OBSERVED (never served itself),
/// reconstructed from its `openai.exchange.v1` terminal event. Unlike
/// [`ExchangeRecord`], the plugin holds no request/response *bytes* here -- a
/// host-served GGUF exchange routes host->native-runtime and never reaches this
/// plugin's HTTP handler. What it does hold, off the terminal event, is:
///   - the host's real `serving_provenance` (model identity + hardware),
///   - the backend's real `usage` (token counts),
///   - the canonical digest of the REAL request body (`request_digest`),
///     forwarded by the host so the capsule's `agent_input_digest` binds the
///     real bytes without the plugin ever seeing the prompt.
///
/// Every field is real-or-honest-default; nothing is fabricated.
pub struct ObservedHostExchange<'a> {
    pub model: &'a str,
    /// Host-minted per-exchange correlation id from the terminal event.
    pub exchange_id: Option<&'a str>,
    /// Canonical JSON-DIGEST of the REAL request body, forwarded by the host.
    /// `None` when the host did not forward one (older host / non-JSON body) --
    /// the capsule then records an honest "unknown-request" sentinel, never a
    /// fabricated digest.
    pub request_digest: Option<&'a str>,
    /// Canonical JSON-DIGEST of the REAL response body, forwarded by the host
    /// (computed at its JSON-relay delivery point). When present, the capsule's
    /// `agent_output_digest` binds the REAL response bytes rather than merely
    /// the terminal accounting facts. `None` on a host that did not forward one
    /// (older host / streamed body) -- the capsule then falls back to the
    /// observed-terminal-facts digest, documented as such, never fabricated.
    pub response_digest: Option<&'a str>,
    /// Canonical JSON-DIGEST of the flattened `tool_calls` the model emitted,
    /// forwarded by the host (byte-for-byte the Python reference
    /// `json_digest(tool_calls)`). `None` -- and then ABSENT from the capsule,
    /// never a fabricated digest over `[]` -- when the model emitted none.
    pub tool_calls_digest: Option<&'a str>,
    /// Canonical JSON-DIGEST of the model's `reasoning_content`, forwarded by
    /// the host. `None` -- and ABSENT from the capsule -- for a non-reasoning
    /// model (honest null), never fabricated.
    pub reasoning_digest: Option<&'a str>,
    /// The backend's real token usage from the terminal event.
    pub usage: Option<TokenUsage>,
    /// The host's serving provenance for this served model.
    pub host_provenance: HostProvenance,
    /// Which real dispatch path produced this terminal event -- the
    /// AUTHORITATIVE signal `role_and_observation_point` derives `role` from
    /// (2026-09-06 role ruling). See `lifecycle_channel::DispatchPath`.
    pub dispatch_path: DispatchPath,
    /// The client-forwarded nonce off the terminal event, when the host
    /// reported one (requester-side sealing on a proxied exchange) --
    /// this is the JOIN KEY between a proxied exchange's two sealed halves
    /// (this node's requester-role capsule and the peer's served-role
    /// capsule): both sides forward the SAME client nonce, while
    /// `exchange_id` is minted per-node and cannot be relied on to match.
    /// `None` when the host did not forward one -- the capsule then falls
    /// back to the pre-existing honest "no nonce observed" default, never a
    /// fabricated value.
    pub nonce: Option<&'a str>,
    /// On a `dispatch_path == RemoteMesh` (requester-side) terminal event:
    /// the capsule id the PEER asserted for its own half, off the host's
    /// `capsule_id` field. `None` on every other dispatch path (there is no
    /// peer half to name) and when the host observed none.
    pub peer_capsule_id: Option<&'a str>,
    /// The wire value of the host's `CapsuleIdProvenance` for
    /// `peer_capsule_id` (`lifecycle_channel::capsule_id_provenance_wire_value`)
    /// -- `None` exactly when `peer_capsule_id` is `None`.
    pub peer_capsule_id_provenance: Option<&'a str>,
    /// The id shared by BOTH halves of a client-marked twin pair, forwarded
    /// verbatim off the terminal envelope's own `twin_bracket_id` -- this
    /// plugin never mints or derives one, only relays what the host already
    /// minted. `None` on every exchange the envelope reports as not twinned
    /// (the overwhelming majority) -- the capsule then omits the field
    /// entirely, never a fabricated bracket.
    pub twin_bracket_id: Option<&'a str>,
    /// The host's digest of the answer text, forwarded verbatim (see
    /// `ServingProvenance::response_text_digest`). `None` when the host sent none.
    pub response_text_digest: Option<&'a str>,
}

impl CapsuleState {
    /// Seal, sign, chain, and ledger one host-served exchange this plugin only
    /// OBSERVED on the `openai.exchange.v1` channel -- closing the gap where a
    /// host-served GGUF (routed host->native-runtime, never through this
    /// plugin's own handler) produced NO capsule at all. The three real facts
    /// come straight off the host's terminal event (serving provenance, usage,
    /// request digest); no bytes are handled here.
    ///
    /// DIGEST BINDING (honest, precise):
    ///   * `agent_input_digest` = the host-forwarded `request_digest`, the
    ///     canonical JSON-DIGEST of the REAL request body (computed host-side
    ///     the same way this plugin's `canonical_body_digest` does). When the
    ///     host forwarded none, an explicit `unknown-request:<model>` sentinel.
    ///   * `agent_output_digest` = the host-forwarded response-body digest, or
    ///     absent when the host forwarded none.
    ///   * `effect` (AAC-05 §5.2) carries ONLY digests of real bodies:
    ///     `request_digest` is the host-forwarded request digest or absent;
    ///     `status` is `confirmed` with `response_digest` = the host-forwarded
    ///     response-body digest, and `dispatched` with no `response_digest`
    ///     when the host forwarded none. The sentinel never enters the effect.
    ///   * `action_type` is `fyi`: the plugin decided nothing on this path.
    pub fn emit_for_observed_host_exchange(
        &self,
        observed: &ObservedHostExchange,
    ) -> anyhow::Result<EmittedCapsule> {
        let result = self.seal_observed_host_exchange(observed);
        if let Err(error) = &result {
            let count = self
                .observed_not_sealed
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            tracing::warn!(%error, observed_not_sealed = count, "observed exchange NOT sealed");
        }
        result
    }

    /// The coordinator's main record of a split request: the same observed
    /// exchange record [`Self::emit_for_observed_host_exchange`] seals, with
    /// the split added. Its stage-exchange records are appended first, under
    /// the same ledger lock, so the main record chains straight after them
    /// and cites each one `split_stage`.
    pub fn emit_for_observed_split_exchange(
        &self,
        observed: &ObservedHostExchange,
        plan: &crate::split_stage::SplitPlan,
    ) -> anyhow::Result<EmittedCapsule> {
        let result = self.seal_observed_host_exchange_with(observed, Some(plan));
        if let Err(error) = &result {
            let count = self
                .observed_not_sealed
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            tracing::warn!(%error, observed_not_sealed = count, "observed split exchange NOT sealed");
        }
        result
    }

    /// A stage node's own record of one request (`side: stage`), through the
    /// single-writer path every local record uses.
    pub fn emit_stage_record(
        &self,
        block: &crate::producer::stage::StageBlock,
    ) -> anyhow::Result<EmittedCapsule> {
        let mut ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.append_stage_record(&mut ledger, block, None)
    }

    fn append_stage_record(
        &self,
        ledger: &mut Ledger,
        block: &crate::producer::stage::StageBlock,
        stage_record_id: Option<&str>,
    ) -> anyhow::Result<EmittedCapsule> {
        let capsule = crate::producer::stage::seal_stage_record(
            block,
            stage_record_id,
            ledger.chain_head(),
            &self.keys.signing_key,
        )?;
        let capsule_id = capsule["capsule_id"]
            .as_str()
            .expect("seal_stage_record always sets capsule_id")
            .to_string();
        self.append_local(ledger, &capsule, &capsule_id)?;
        Ok(EmittedCapsule {
            capsule_id,
            capsule,
        })
    }

    fn append_local(
        &self,
        ledger: &mut Ledger,
        capsule: &Value,
        capsule_id: &str,
    ) -> anyhow::Result<()> {
        let payload = crate::producer::capsule::payload_bytes(capsule);
        let statement = build_signed_statement(
            &SignedStatementInput {
                payload: &payload,
                issuer: &self.node_id,
                subject: capsule_id,
                content_type: &CAPSULE_CONTENT_TYPE,
            },
            &self.keys.signing_key,
        );
        ledger.append(capsule, &statement)?;
        Ok(())
    }

    fn seal_observed_host_exchange(
        &self,
        observed: &ObservedHostExchange,
    ) -> anyhow::Result<EmittedCapsule> {
        self.seal_observed_host_exchange_with(observed, None)
    }

    fn seal_observed_host_exchange_with(
        &self,
        observed: &ObservedHostExchange,
        split: Option<&crate::split_stage::SplitPlan>,
    ) -> anyhow::Result<EmittedCapsule> {
        let ObservedHostExchange {
            model,
            exchange_id,
            request_digest,
            response_digest,
            tool_calls_digest,
            reasoning_digest,
            usage,
            host_provenance,
            dispatch_path,
            nonce,
            peer_capsule_id,
            peer_capsule_id_provenance,
            twin_bracket_id,
            response_text_digest,
        } = observed;
        let response_text_digest = response_text_digest
            .map(|d| d.trim().to_ascii_lowercase())
            .filter(|d| d.len() == 64 && d.bytes().all(|b| b.is_ascii_hexdigit()));
        let (
            model,
            exchange_id,
            request_digest,
            response_digest,
            tool_calls_digest,
            reasoning_digest,
            usage,
            nonce,
            peer_capsule_id,
            peer_capsule_id_provenance,
            twin_bracket_id,
        ) = (
            *model,
            *exchange_id,
            *request_digest,
            *response_digest,
            *tool_calls_digest,
            *reasoning_digest,
            usage.clone(),
            *nonce,
            *peer_capsule_id,
            *peer_capsule_id_provenance,
            *twin_bracket_id,
        );
        let host = host_provenance.clone();
        // Host digests in a recoverable spelling are normalised; anything
        // else is treated as absent -- never passed on to be refused at seal.
        let request_digest_owned = normalise_host_digest(request_digest);
        let response_digest_owned = normalise_host_digest(response_digest);
        let request_digest = request_digest_owned.as_deref();
        let response_digest = response_digest_owned.as_deref();

        // LEARN (or confirm, or rotate) this node's own mesh identity: a
        // locally-served event (`TypedFrontend`/`RawProxy`) that carries the
        // fork's `served_by_node_id` enrichment IS this node telling us who
        // it is -- see `LearnedSelfNodeId`. A `RemoteMesh` event never
        // learns (this node is the REQUESTER there, not the server).
        if role_for_dispatch_path(dispatch_path) == "served" {
            if let Some(served_by) = host.served_by_node_id.as_deref() {
                self.learned_self_node_id
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .observe(served_by);
            }
        }
        let learned_self = self.learned_self_node_id();

        let (role, observation_point) = role_and_observation_point(
            dispatch_path,
            host.served_by_node_id.as_deref(),
            learned_self.as_deref(),
        );
        if split.is_some() && role != "served" {
            anyhow::bail!("a split's main record is the coordinator's served record, not {role:?}");
        }

        // agent_input_digest: the host-forwarded canonical request-body digest
        // when present; an explicit honest sentinel otherwise (never fabricated).
        let agent_input_digest = request_digest
            .map(str::to_string)
            .unwrap_or_else(|| format!("unknown-request:{model}"));

        // host_binding: the SAME host-forwarded value, ALSO carried as an
        // independent reverse-direction composition binding under mesh-llm's
        // OWN construction -- not a re-derivation of `agent_input_digest`
        // above, and never presumed byte-equal to it (see
        // `crate::producer::capsule::HostBinding`'s INDEPENDENCE RULE). This
        // is what lets the record join mesh-llm's own ops log even if a
        // future mesh-llm canonicalization change makes the two values
        // diverge: the record already carries mesh-llm's digest labeled AS
        // mesh-llm's own scheme, not smuggled in under ours. Absent -- never
        // null -- when the host forwarded none.
        let host_binding = request_digest.map(|rd| HostBinding {
            digest: rd.to_string(),
            construction: crate::producer::capsule::MESH_LLM_REQUEST_BODY_SHA256_V1.to_string(),
            purpose: crate::producer::capsule::HOST_LOG_JOIN.to_string(),
        });

        // agent_output_digest: the host-forwarded canonical digest of the REAL
        // response body, or ABSENT when the host forwarded none (older host, or
        // a streamed body it could not buffer). Never a digest of the terminal
        // facts in its place: a bare 64-hex there reads as a body digest.
        let agent_output_digest = response_digest.map(str::to_string);
        // As this record carries it: a nonce here is always host-forwarded.
        let tee_attestation = self.tee_slot(
            request_digest,
            agent_output_digest.as_deref(),
            nonce,
            "host_forwarded_nonce",
        );

        let mut ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // A split's stage-exchange records go on the chain first, so the main
        // record below chains after them and can cite them.
        let split = match split {
            Some(plan) => {
                let mut stage_exchange_records = Vec::new();
                for (block, stage_record_id) in &plan.exchanges {
                    let emitted =
                        self.append_stage_record(&mut ledger, block, stage_record_id.as_deref())?;
                    stage_exchange_records.push((block.stage_index, emitted.capsule_id));
                }
                Some(crate::producer::stage::SplitMainExtension {
                    own_slice: plan.own_slice.clone(),
                    receipt: plan.receipt.clone(),
                    stage_exchange_records,
                })
            }
            None => None,
        };
        let chain = ledger.chain_head().map(|parent| ChainLink {
            parent_capsule_id: parent.to_string(),
            relation: "follows".to_string(),
        });

        // Honest-by-absence: a host-served exchange routes host->native-runtime
        // and never reaches this plugin's handler, so it holds NO request bytes
        // here (only the host-forwarded request DIGEST) -- the client's sampling
        // settings are simply not observable on this path. We therefore seal an
        // EMPTY generation-parameter set rather than the old fabricated
        // `temperature=0.0`; absent facts are recorded as absent, never invented.
        // (Full generation-parameter capture on the observe path would require
        // the host to forward them on the terminal event; see PROTOCOL-NOTE.md.)
        let generation_parameters = Map::new();

        // Runtime/binary attestation rung (task B3), OBSERVE path. IMPORTANT
        // HONESTY BOUND: on this path the binary that actually SERVED the
        // inference is the mesh-llm HOST's native runtime, which this plugin
        // never runs and cannot hash -- so this attestation is of the OBSERVING/
        // EMITTING binary (this plugin), NOT the serving one.
        // It is still self_measured/os_measured OF THE EMITTING NODE'S OWN
        // BINARY (never the host's), and is labeled as the observer's binary
        // via the runtime name, so no reader can mistake it for a measurement
        // of the host serving runtime.
        // Degrades to `None` (empty slot) when unmeasurable, never fabricated.
        let binary_attestation =
            crate::producer::runtime_attest::measure_self(&self.keys, utc_now_minute());

        // The node that asked this node to serve the exchange: the host's
        // `requested_by_node_id`, which it sets only on a host-served
        // exchange that came over the mesh, from that connection's
        // QUIC-authenticated remote id (never from the request). It is what
        // makes the asker this record's counterparty, so a signed request
        // from that node can get the record back. Any other exchange (one
        // that reached the local API, a requester-side half, an older host)
        // keeps the honest "unknown", never a fabricated party. The pair is
        // sequenced under the same id.
        let requesting_party = if role == "served" {
            host.requested_by_node_id
                .as_deref()
                .map(str::trim)
                .filter(|id| !id.is_empty() && *id != "unknown")
        } else {
            None
        }
        .unwrap_or("unknown")
        .to_string();
        let mut sequence_counters = self
            .sequence_counters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let sequence = sequence_counters.peek(&self.node_id, &requesting_party);

        let input = CapsuleInput {
            action_id: format!("mesh-poc/capsule-emit-mesh-host-served/{agent_input_digest}"),
            // Observed, not admitted: this plugin decided nothing here.
            action_type: "fyi".to_string(),
            operator: "capsule-emit-mesh-poc-rust".to_string(),
            developer: crate::producer::capsule::LOCAL_RECORD_DEVELOPER.to_string(),
            timestamp: utc_now_minute(),
            domain: Some("action".to_string()),
            provenance: Some("collector".to_string()),
            model_id: model.to_string(),
            provider: "mesh-llm".to_string(),
            agent_input_digest: agent_input_digest.clone(),
            agent_output_digest: agent_output_digest.clone(),
            // The REAL labeled sub-digests the host forwarded off the response
            // body -- OPTIONAL, absent when the model emitted none (never
            // fabricated). This is where the real `tool_calls_digest` lands.
            tool_calls_digest: tool_calls_digest.map(str::to_string),
            reasoning_digest: reasoning_digest.map(str::to_string),
            host_binding: host_binding.clone(),
            // Observe path: this measures the OBSERVING plugin binary, not the
            // host serving runtime (see the note above) -- labeled as such in the
            // runtime name. Real self-measured hash when measurable; `0*64`
            // placeholder only on graceful degradation.
            runtime: observer_runtime_field(&binary_attestation),
            mesh_poc: MeshPocV1 {
                // The host-forwarded terminal-event nonce, when present
                // (requester-side sealing on a proxied exchange) -- THE JOIN
                // KEY between a proxied exchange's two sealed halves (see
                // `ObservedHostExchange::nonce`). A host that did not forward one
                // (predates the forwarding, or a locally-served exchange whose
                // client request never carried this plugin's own handler) keeps
                // the pre-existing honest default -- never a fabricated nonce.
                client_nonce: nonce
                    .map(str::to_string)
                    .unwrap_or_else(|| crate::evidence_log::HOST_SERVED_NO_NONCE.to_string()),
                client_nonce_source: if nonce.is_some() {
                    "host_forwarded_nonce"
                } else {
                    "host_served_observed"
                }
                .to_string(),
                model_name_digest: hex_sha256(model.as_bytes()),
                serving_provenance: ServingProvenance {
                    // Fall back to this node's LEARNED mesh identity ONLY
                    // when dispatch_path itself says this node served the
                    // exchange -- for a RemoteMesh event with no
                    // served_by_node_id (bare upstream, pre-enrichment), the
                    // honest value is "unknown", never a fabricated claim
                    // that this node served an exchange it just labeled
                    // `role: "requested"`. And when this IS a served event
                    // but nothing has been learned yet (no prior enrichment
                    // observed), "unknown" is still the honest answer --
                    // never the `PLUGIN_ID` plugin-type label.
                    served_by_node_id: host.served_by_node_id.clone().unwrap_or_else(|| {
                        if role_for_dispatch_path(dispatch_path) == "served" {
                            learned_self
                                .clone()
                                .unwrap_or_else(|| "unknown".to_string())
                        } else {
                            "unknown".to_string()
                        }
                    }),
                    dispatch_path: Some(dispatch_path_wire_value(dispatch_path).to_string()),
                    requesting_party: requesting_party.clone(),
                    exchange_id: exchange_id.unwrap_or("unknown").to_string(),
                    hostname: sealed_hostname(&host, hostname_opt_in()),
                    quantization: host
                        .quantization
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string()),
                    hardware_gpu: host.gpu.clone(),
                    hardware_vram_bytes: host.vram_bytes,
                    hardware_device: None,
                    hardware_is_soc: host.is_soc,
                    architecture: host.architecture.clone(),
                    context_length: host.context_length,
                    parameter_size: host.parameter_size.clone(),
                    layer_count: host.layer_count,
                    model_identity_hash: host.model_identity_hash.clone(),
                    weights_digest: host.weights_digest.clone(),
                    model_canonical_ref: host.model_canonical_ref.clone(),
                    model_revision: host.model_revision.clone(),
                    // The REAL token counts from the host terminal event.
                    usage,
                    seq: sequence.seq,
                    prev_seq: sequence.prev_seq,
                    // The peer's self-asserted capsule id for its own half of
                    // this exchange, forwarded verbatim off the host's
                    // RemoteMesh terminal envelope -- present only on the
                    // requester side (see `ObservedHostExchange::
                    // peer_capsule_id` doc), never fabricated when absent.
                    peer_capsule_id: peer_capsule_id.map(str::to_string),
                    peer_capsule_id_provenance: peer_capsule_id_provenance.map(str::to_string),
                    // Forwarded verbatim off the terminal envelope -- this
                    // plugin never mints or derives one. `None` on every
                    // exchange that wasn't ambiently twinned.
                    twin_bracket_id: twin_bracket_id.map(str::to_string),
                    response_text_digest: response_text_digest.clone(),
                },
                role,
                observation_point,
                generation_parameters,
                // Latency is not carried on the observe path (the plugin did not
                // time the host's own dispatch) -- honest zero-marker, not faked.
                latency_ms: "0.000".to_string(),
                binary_attestation,
                tee_attestation: Some(tee_attestation),
            },
            // Confirmed only over the host's digest of the REAL response body;
            // without one the completion was dispatched but its output is
            // unbound, so the effect carries no response digest (AAC-05 §5.2).
            effect_status: if response_digest.is_some() {
                "confirmed"
            } else {
                "dispatched"
            }
            .to_string(),
            effect_type: "inference_completion".to_string(),
            effect_request_digest: request_digest.map(str::to_string),
            effect_response_digest: response_digest.map(str::to_string),
            effect_attestation: "host_served_observed".to_string(),
            disposition_decision: "accept".to_string(),
            disposition_approver: "policy".to_string(),
            disposition_human_disposed: false,
            disposition_verdict_class: "executed".to_string(),
            chain,
            store_nonce: crate::producer::capsule::fresh_store_nonce(),
        };

        let mut capsule = match &split {
            Some(split) => crate::producer::stage::seal_split_main_record(&input, split)?,
            None => seal(&input)?,
        };
        let capsule_id = capsule["capsule_id"]
            .as_str()
            .expect("seal() always sets capsule_id")
            .to_string();
        // Attach the inline producer-signature envelope BEFORE building the
        // detached `.cose` statement, so the ledgered capsule, the `.cose`
        // payload, and any pushed half are ONE shape (`signature`/`key_id`
        // present) -- a peer that receives just the pushed half verifies it in
        // isolation, exactly as a `capsule_emit.seal()` capsule does. Excluded
        // from the `capsule_id` preimage, so `capsule_id` is unchanged.
        crate::producer::capsule::attach_producer_envelope(&mut capsule, &self.keys.signing_key)
            .expect("seal() always sets a hex capsule_id");
        let payload = crate::producer::capsule::payload_bytes(&capsule);
        let statement = build_signed_statement(
            &SignedStatementInput {
                payload: &payload,
                issuer: &self.node_id,
                subject: &capsule_id,
                content_type: &CAPSULE_CONTENT_TYPE,
            },
            &self.keys.signing_key,
        );
        ledger.append(&capsule, &statement)?;
        sequence_counters.commit(&self.node_id, &requesting_party, sequence)?;

        Ok(EmittedCapsule {
            capsule_id,
            capsule,
        })
    }
}

/// What an index rebuild found: every record re-read and re-checked from disk.
pub struct IndexRebuild {
    pub records_checked: usize,
    pub head: Option<String>,
}

impl CapsuleState {
    /// Seal an owner cleanup ([`OwnerMaintenance`]) onto this node's chain
    /// through the same single-writer path every other local record uses.
    pub fn emit_owner_maintenance(
        &self,
        action: &OwnerMaintenance,
    ) -> anyhow::Result<EmittedCapsule> {
        let mut ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let capsule =
            seal_owner_maintenance_record(action, ledger.chain_head(), &self.keys.signing_key)?;
        let capsule_id = capsule["capsule_id"]
            .as_str()
            .expect("seal_owner_maintenance_record always sets capsule_id")
            .to_string();
        let payload = crate::producer::capsule::payload_bytes(&capsule);
        let statement = build_signed_statement(
            &SignedStatementInput {
                payload: &payload,
                issuer: &self.node_id,
                subject: &capsule_id,
                content_type: &CAPSULE_CONTENT_TYPE,
            },
            &self.keys.signing_key,
        );
        ledger.append(&capsule, &statement)?;
        Ok(EmittedCapsule {
            capsule_id,
            capsule,
        })
    }

    /// Every record's `capsule_id`, in chain order.
    pub fn capsule_ids_in_order(&self) -> Vec<String> {
        self.ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .capsule_ids_in_order()
    }

    pub fn ledger_dir(&self) -> &Path {
        &self.ledger_dir
    }

    /// Rebuild the in-memory index by re-opening the ledger from disk, which
    /// re-checks every record (id, chain link, signed statement). Swapped in
    /// only when the rebuilt chain ends at the same head this node already
    /// holds; a mismatch means something other than this process changed the
    /// file, and that is reported, never papered over.
    pub fn rebuild_index(&self) -> anyhow::Result<IndexRebuild> {
        let mut ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (rebuilt, report) = crate::producer::index::open_ledger(&self.ledger_dir)?;
        if rebuilt.chain_head() != ledger.chain_head() {
            anyhow::bail!(
                "the records on disk end at {:?} but this node holds {:?}; not replacing the index",
                rebuilt.chain_head(),
                ledger.chain_head()
            );
        }
        let head = rebuilt.chain_head().map(str::to_string);
        *ledger = rebuilt;
        Ok(IndexRebuild {
            records_checked: report.valid_entries,
            head,
        })
    }
}

/// The provenance a received counterparty half carries into its citing record
/// -- re-exported from `crate::producer` so `main.rs`/`record_push_bridge.rs`
/// name ONE type. See `crate::producer::capsule::ReceivedHalfProvenance`.
pub use crate::producer::capsule::ReceivedHalfProvenance;

impl CapsuleState {
    /// Seal, chain, and ledger the
    /// LOCAL CITING record for a received counterparty half, through the SAME
    /// single-writer path (`seal` -> `attach_producer_envelope` ->
    /// `Ledger::append`) every other local capsule uses -- so the received
    /// half produces a chained, checkpoint-covered record OF OURS without the
    /// foreign body ever entering `capsules.jsonl`. The record-push receiver
    /// has already verified the half and stored its bytes in the held-artifact
    /// store `received-capsules.jsonl`; this is the chained citation of it.
    /// "cite, never mutate."
    ///
    /// One writer: this locks the SAME ledger mutex `emit_for_exchange` /
    /// `emit_for_observed_host_exchange` lock, reads the current head, and
    /// appends -- so a citing record chains cleanly onto whatever this node
    /// last sealed, and no second process ever writes `capsules.jsonl`.
    ///
    /// DEDUP by foreign `capsule_id` (live-path sibling of the backfill's
    /// `already_cited`): a RE-pushed foreign half -- an authorized peer can
    /// re-push the same capsule any number of times -- must not seal a
    /// duplicate citing record every time (two fsyncs + a chain line per
    /// push: disk amplification). The ledger itself tracks which foreign ids
    /// its citing records already cite (`Ledger::cites_counterparty_half`,
    /// rebuilt on open, keyed on `citation_purpose == "counterparty_half"`
    /// alone); a duplicate returns `Ok(None)` --
    /// the half IS still received/held (the receiver stored it), there is just
    /// nothing new to cite.
    pub fn emit_citing_record(
        &self,
        prov: &ReceivedHalfProvenance,
    ) -> anyhow::Result<Option<EmittedCapsule>> {
        let mut ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if ledger.cites_counterparty_half(prov.foreign_capsule_id) {
            return Ok(None);
        }
        let capsule = crate::producer::capsule::seal_citing_record(
            prov,
            ledger.chain_head(),
            &self.keys.signing_key,
        )?;
        let capsule_id = capsule["capsule_id"]
            .as_str()
            .expect("seal_citing_record always sets capsule_id")
            .to_string();
        let payload = crate::producer::capsule::payload_bytes(&capsule);
        let statement = build_signed_statement(
            &SignedStatementInput {
                payload: &payload,
                issuer: &self.node_id,
                subject: &capsule_id,
                content_type: &CAPSULE_CONTENT_TYPE,
            },
            &self.keys.signing_key,
        );
        ledger.append(&capsule, &statement)?;
        Ok(Some(EmittedCapsule {
            capsule_id,
            capsule,
        }))
    }

    /// Seal, chain, and ledger the `counterparty_inclusion` citing record for
    /// a held half's inclusion evidence (a pushed bundle's proof + covering
    /// checkpoint, verified and stored by the receiver) -- the same single-writer
    /// path as [`Self::emit_citing_record`], and deduped the same way: a
    /// re-pushed bundle for a half already covered seals nothing (`Ok(None)`).
    pub fn emit_inclusion_citing_record(
        &self,
        citation: &InclusionCitation,
    ) -> anyhow::Result<Option<EmittedCapsule>> {
        let mut ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if ledger.cites_counterparty_inclusion(citation.half_capsule_id) {
            return Ok(None);
        }
        let capsule = crate::producer::capsule::seal_inclusion_citing_record(
            citation,
            ledger.chain_head(),
            &self.keys.signing_key,
        )?;
        let capsule_id = capsule["capsule_id"]
            .as_str()
            .expect("seal_inclusion_citing_record always sets capsule_id")
            .to_string();
        let payload = crate::producer::capsule::payload_bytes(&capsule);
        let statement = build_signed_statement(
            &SignedStatementInput {
                payload: &payload,
                issuer: &self.node_id,
                subject: &capsule_id,
                content_type: &CAPSULE_CONTENT_TYPE,
            },
            &self.keys.signing_key,
        );
        ledger.append(&capsule, &statement)?;
        Ok(Some(EmittedCapsule {
            capsule_id,
            capsule,
        }))
    }

    /// Seal the REFEREE's own record of a verdict it issued (see
    /// `crate::producer::capsule::seal_adjudication_issued_record`), on the
    /// same single-writer path as [`Self::emit_citing_record`]. A verdict
    /// already recorded seals nothing (`Ok(None)`).
    pub fn emit_adjudication_issued(
        &self,
        facts: &VerdictFacts,
        referee_capsule_id: Option<&str>,
        issued_at: &str,
    ) -> anyhow::Result<Option<EmittedCapsule>> {
        self.emit_adjudication_record(
            crate::producer::capsule::ADJUDICATION_ISSUED_BLOCK,
            facts,
            |head, key| {
                crate::producer::capsule::seal_adjudication_issued_record(
                    facts,
                    referee_capsule_id,
                    issued_at,
                    head,
                    key,
                )
            },
        )
    }

    /// Seal this node's own record of a verdict delivered to it (see
    /// `crate::producer::capsule::seal_adjudication_received_record`). A
    /// verdict already recorded seals nothing (`Ok(None)`).
    pub fn emit_adjudication_received(
        &self,
        facts: &VerdictFacts,
        held_half_capsule_id: &str,
        received_from: &str,
        received_at: &str,
    ) -> anyhow::Result<Option<EmittedCapsule>> {
        self.emit_adjudication_record(
            crate::producer::capsule::ADJUDICATION_RECEIVED_BLOCK,
            facts,
            |head, key| {
                crate::producer::capsule::seal_adjudication_received_record(
                    facts,
                    held_half_capsule_id,
                    received_from,
                    received_at,
                    head,
                    key,
                )
            },
        )
    }

    /// Seal the courier's own record of a delivery its receiver refused (see
    /// `crate::producer::capsule::seal_adjudication_ack_refused_record`). One
    /// per verdict and refusing peer; a repeat seals nothing (`Ok(None)`).
    pub fn emit_adjudication_ack_refused(
        &self,
        refused: &crate::producer::capsule::RefusedDelivery,
    ) -> anyhow::Result<Option<EmittedCapsule>> {
        self.emit_keyed_record(
            crate::producer::capsule::ADJUDICATION_ACK_REFUSED_BLOCK,
            &format!("{}/{}", refused.verdict_capsule_id, refused.refused_by),
            |head, key| {
                crate::producer::capsule::seal_adjudication_ack_refused_record(refused, head, key)
            },
        )
    }

    fn emit_adjudication_record(
        &self,
        block: &str,
        facts: &VerdictFacts,
        seal: impl FnOnce(
            Option<&str>,
            &ed25519_dalek::SigningKey,
        ) -> Result<Value, crate::producer::capsule::SealError>,
    ) -> anyhow::Result<Option<EmittedCapsule>> {
        self.emit_keyed_record(block, facts.verdict_capsule_id, seal)
    }

    fn emit_keyed_record(
        &self,
        block: &str,
        key: &str,
        seal: impl FnOnce(
            Option<&str>,
            &ed25519_dalek::SigningKey,
        ) -> Result<Value, crate::producer::capsule::SealError>,
    ) -> anyhow::Result<Option<EmittedCapsule>> {
        let mut ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if ledger.has_index_key(&crate::producer::index::adjudication_key(block, key)) {
            return Ok(None);
        }
        let capsule = seal(ledger.chain_head(), &self.keys.signing_key)?;
        let capsule_id = capsule["capsule_id"]
            .as_str()
            .expect("an adjudication record always sets capsule_id")
            .to_string();
        let payload = crate::producer::capsule::payload_bytes(&capsule);
        let statement = build_signed_statement(
            &SignedStatementInput {
                payload: &payload,
                issuer: &self.node_id,
                subject: &capsule_id,
                content_type: &CAPSULE_CONTENT_TYPE,
            },
            &self.keys.signing_key,
        );
        ledger.append(&capsule, &statement)?;
        Ok(Some(EmittedCapsule {
            capsule_id,
            capsule,
        }))
    }

    /// Seal a local routing choice (block or unblock) onto the same
    /// single-writer chain, committing to the peer with the caller's salt. The
    /// caller keeps that salt; it is the only way to say later which peer the
    /// record's commitment names.
    pub fn emit_local_routing_choice(
        &self,
        change: crate::producer::capsule::RoutingChoiceChange,
        peer_id: &str,
        until: Option<&str>,
        salt: &[u8; 32],
        rule: Option<&crate::producer::capsule::RoutingRuleCitation<'_>>,
    ) -> anyhow::Result<EmittedRoutingChoice> {
        let choice = crate::producer::capsule::LocalRoutingChoice {
            change,
            peer_id,
            salt,
            until,
            rule,
        };
        let mut ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let capsule = crate::producer::capsule::seal_local_routing_choice(
            &choice,
            ledger.chain_head(),
            &self.keys.signing_key,
        )?;
        let capsule_id = capsule["capsule_id"]
            .as_str()
            .expect("seal_local_routing_choice always sets capsule_id")
            .to_string();
        let payload = crate::producer::capsule::payload_bytes(&capsule);
        let statement = build_signed_statement(
            &SignedStatementInput {
                payload: &payload,
                issuer: &self.node_id,
                subject: &capsule_id,
                content_type: &CAPSULE_CONTENT_TYPE,
            },
            &self.keys.signing_key,
        );
        ledger.append(&capsule, &statement)?;
        Ok(EmittedRoutingChoice {
            capsule_id,
            peer_commitment: crate::producer::capsule::peer_commitment(peer_id, salt),
        })
    }
}

/// What [`CapsuleState::emit_local_routing_choice`] hands back to the host's
/// block store.
pub struct EmittedRoutingChoice {
    pub capsule_id: String,
    pub peer_commitment: String,
}

/// See `crate::producer::capsule::InclusionCitation`.
pub use crate::producer::capsule::InclusionCitation;
/// See `crate::producer::capsule::VerdictFacts`.
pub use crate::producer::capsule::VerdictFacts;

impl CapsuleState {
    /// Seal, chain, and ledger one SETTLEMENT record -- this payer node's
    /// sealed observation of one checked `payment.lifecycle.v1` event (see
    /// `crate::producer::capsule::seal_settlement_record` for what the record
    /// does and does not claim). Same single-writer path as
    /// [`CapsuleState::emit_citing_record`]: the one ledger mutex, the current
    /// head, the detached `.cose` statement, `Ledger::append`.
    ///
    /// DEDUP by `event_ref`: the host may deliver the same event more than
    /// once, and each delivery must not seal another record. The ledger tracks
    /// the `event_ref`s its settlement records already carry
    /// (`Ledger::has_settlement_event`, rebuilt on open); a repeat returns
    /// `Ok(None)`.
    pub fn emit_settlement_record(
        &self,
        ev: &crate::producer::capsule::SettlementObservation,
    ) -> anyhow::Result<Option<EmittedCapsule>> {
        let mut ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if ledger.has_index_key(&crate::producer::index::settlement_event_key(ev.event_ref)) {
            return Ok(None);
        }
        let capsule = crate::producer::capsule::seal_settlement_record(
            ev,
            ledger.chain_head(),
            &self.keys.signing_key,
        )?;
        let capsule_id = capsule["capsule_id"]
            .as_str()
            .expect("seal_settlement_record always sets capsule_id")
            .to_string();
        let payload = crate::producer::capsule::payload_bytes(&capsule);
        let statement = build_signed_statement(
            &SignedStatementInput {
                payload: &payload,
                issuer: &self.node_id,
                subject: &capsule_id,
                content_type: &CAPSULE_CONTENT_TYPE,
            },
            &self.keys.signing_key,
        );
        ledger.append(&capsule, &statement)?;
        Ok(Some(EmittedCapsule {
            capsule_id,
            capsule,
        }))
    }

    /// Seal one settlement-records leg (draft-mih-agent-settlement-records-00),
    /// chained onto this node's head, with its detached statement. A leg
    /// already on the ledger (same leg kind and payment, or same delivery) is
    /// not sealed again: `Ok(None)`.
    pub fn emit_settlement_leg(
        &self,
        action_id: String,
        member: Value,
        references: Option<Value>,
    ) -> anyhow::Result<Option<EmittedCapsule>> {
        let (leg, subject) =
            crate::producer::index::settlement_leg_subject(&member).ok_or_else(|| {
                anyhow::anyhow!("settlement leg has no payment hash or delivery subject")
            })?;
        let mut ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if ledger.has_index_key(&crate::producer::index::settlement_leg_key(&leg, &subject)) {
            return Ok(None);
        }
        let chain = ledger.chain_head().map(|parent| ChainLink {
            parent_capsule_id: parent.to_string(),
            relation: "follows".to_string(),
        });
        let capsule = crate::producer::capsule::seal_settlement_leg(
            action_id,
            member,
            references,
            chain,
            &self.keys.signing_key,
        )?;
        let capsule_id = capsule["capsule_id"]
            .as_str()
            .expect("seal_settlement_leg always sets capsule_id")
            .to_string();
        let payload = crate::producer::capsule::payload_bytes(&capsule);
        let statement = build_signed_statement(
            &SignedStatementInput {
                payload: &payload,
                issuer: &self.node_id,
                subject: &capsule_id,
                content_type: &CAPSULE_CONTENT_TYPE,
            },
            &self.keys.signing_key,
        );
        ledger.append(&capsule, &statement)?;
        Ok(Some(EmittedCapsule {
            capsule_id,
            capsule,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_content_type_profile_is_the_sealed_spec_version() {
        let profile = CAPSULE_CONTENT_TYPE
            .split_once("; profile=")
            .map(|(_, profile)| profile);
        assert_eq!(profile, Some(crate::producer::capsule::SPEC_VERSION));
        assert_eq!(profile, Some("draft-mih-scitt-agent-action-capsule-05"));
    }

    /// PARITY PIN: `output_sub_digests` over the REAL SETI@Home / web_search
    /// response computes a `tool_calls_digest` byte-for-byte identical to the
    /// Python reference `agent_action_capsule.json_digest(tool_calls)` — the
    /// value recorded requester-side in a live run. A non-reasoning model yields
    /// an absent reasoning digest (honest null), never fabricated.
    #[test]
    fn output_sub_digests_over_real_seti_response_matches_python_reference() {
        let body = br#"{"id":"chatcmpl-seti","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"","tool_calls":[{"function":{"arguments":"{\"query\": \"mesh-llm vs SETI@Home\"}","name":"web_search"},"id":"call_719a955fb46a41008dd847d412f00795","type":"function"}]},"finish_reason":"tool_calls"}]}"#;
        let (tool_calls_digest, reasoning_digest) = output_sub_digests(body);
        assert_eq!(
            tool_calls_digest.as_deref(),
            Some("f294be8a53bb9c29cd94472721f0857591f34b23fe010882de79b9fb210b1395"),
            "plugin tool_calls_digest must equal the Python reference json_digest(tool_calls)"
        );
        assert!(
            reasoning_digest.is_none(),
            "a non-reasoning model must yield an absent reasoning digest, not a fabricated one"
        );
    }

    /// A block then an unblock both land on the chain, in order (the unblock
    /// chains onto the block), each carrying the commitment the caller's salt
    /// gives.
    #[test]
    fn routing_choices_chain_in_order_with_the_callers_salt() {
        use crate::producer::capsule::{peer_commitment, RoutingChoiceChange};
        let dir = std::env::temp_dir().join(format!("cap-route-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let peer = "e5ba9d1001".repeat(6);
        let (block_salt, unblock_salt) = ([1u8; 32], [2u8; 32]);

        let block = state
            .emit_local_routing_choice(
                RoutingChoiceChange::Block,
                &peer,
                Some("2026-10-04T00:00:00Z"),
                &block_salt,
                None,
            )
            .expect("seal block");
        let unblock = state
            .emit_local_routing_choice(
                RoutingChoiceChange::Unblock,
                &peer,
                None,
                &unblock_salt,
                None,
            )
            .expect("seal unblock");

        let ledger =
            std::fs::read_to_string(dir.join("ledger").join("capsules.jsonl")).expect("ledger");
        let unblock_line: Value = ledger
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .find(|record| record["capsule_id"] == unblock.capsule_id.as_str())
            .expect("the unblock record is on the chain");
        assert_eq!(
            unblock_line["chain"]["parent_capsule_id"].as_str(),
            Some(block.capsule_id.as_str()),
            "the unblock chains onto the block"
        );
        assert_eq!(
            state.chain_head().as_deref(),
            Some(unblock.capsule_id.as_str())
        );
        assert_eq!(block.peer_commitment, peer_commitment(&peer, &block_salt));
        assert_eq!(
            unblock.peer_commitment,
            peer_commitment(&peer, &unblock_salt)
        );

        drop(state);
        let reopened = CapsuleState::open(&dir, "node-under-test").expect("reopen state");
        assert_eq!(
            reopened.chain_head().as_deref(),
            Some(unblock.capsule_id.as_str())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `emit_citing_record` seals a
    /// citing record onto the SAME single-writer chain, chained onto whatever
    /// this node last sealed (`chain.relation == "follows"` -- the citation is
    /// the `references[]` entry), citing the foreign
    /// half by CPB typed digest -- and it reopens clean (proving the ledger
    /// accepts the new record shape without any `Ledger::open` change).
    #[test]
    fn emit_citing_record_chains_onto_the_local_head_and_reopens_clean() {
        let dir = std::env::temp_dir().join(format!("cap-cite-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");

        // First seal one of THIS node's own exchanges, so the citing record has
        // a real head to chain onto (the single-writer invariant in action).
        let exchange = ExchangeRecord {
            model: "m",
            client_nonce: Some("n"),
            request_bytes: br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
            response_bytes:
                br#"{"id":"x","choices":[{"message":{"role":"assistant","content":"y"}}]}"#,
            latency_ms: 1.0,
            exchange_id: Some("e-1"),
            requesting_party: Some("party-1"),
            host_provenance: None,
        };
        let local = state.emit_for_exchange(&exchange).expect("seal local");

        let foreign_id = "d".repeat(64);
        let prov = ReceivedHalfProvenance {
            foreign_capsule_id: &foreign_id,
            received_from: "node-b",
            via: "push",
            received_at: "2026-09-25T00:00:00Z",
            signature_ok: true,
            digest_match: None,
            foreign_agent_input_digest: Some("a".repeat(64).leak()),
            foreign_agent_output_digest: None,
        };
        let citing = state
            .emit_citing_record(&prov)
            .expect("seal citing record")
            .expect("first citation of this half must seal a record");

        // Chained onto the local head with the bare "follows" relation (the
        // citation is the references[] entry, never a relation value), citing
        // the foreign half.
        assert_eq!(
            citing.capsule["chain"]["parent_capsule_id"].as_str(),
            Some(local.capsule_id.as_str())
        );
        assert_eq!(citing.capsule["chain"]["relation"], Value::from("follows"));
        assert_eq!(
            citing.capsule["references"][0]["digest"],
            Value::from(foreign_id.as_str())
        );
        assert_eq!(
            citing.capsule["references"][0]["citation_purpose"],
            Value::from("counterparty_half")
        );
        // The chain head advanced to the citing record.
        assert_eq!(
            state.chain_head().as_deref(),
            Some(citing.capsule_id.as_str())
        );

        // LIVE-PATH DEDUP: a re-pushed foreign half (same capsule_id) seals
        // NOTHING new -- Ok(None), head unchanged, no duplicate chain line.
        let dup = state
            .emit_citing_record(&prov)
            .expect("dedup path must not error");
        assert!(
            dup.is_none(),
            "a re-pushed half must not seal a duplicate citing record"
        );
        assert_eq!(
            state.chain_head().as_deref(),
            Some(citing.capsule_id.as_str())
        );

        // Cold reopen: a ledger with a local + a citing record recovers clean,
        // AND the dedup set is rebuilt from disk -- a re-push is still a
        // no-op after a restart.
        drop(state);
        let reopened = CapsuleState::open(&dir, "node-under-test").expect("reopen state");
        assert_eq!(
            reopened.chain_head().as_deref(),
            Some(citing.capsule_id.as_str())
        );
        let dup_after_restart = reopened
            .emit_citing_record(&prov)
            .expect("dedup path must not error after reopen");
        assert!(
            dup_after_restart.is_none(),
            "the dedup set must be rebuilt from the ledger on open"
        );
        assert_eq!(
            reopened.chain_head().as_deref(),
            Some(citing.capsule_id.as_str())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The inclusion citing record lands on the SAME chain after the half's
    /// own citing record, and a re-pushed bundle (live or after a restart)
    /// seals nothing new.
    #[test]
    fn emit_inclusion_citing_record_chains_after_the_half_citation_and_dedups() {
        let dir = std::env::temp_dir().join(format!("cap-incl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let half = "d".repeat(64);
        let prov = ReceivedHalfProvenance {
            foreign_capsule_id: &half,
            received_from: "node-b",
            via: "push",
            received_at: "2026-09-27T00:00:00Z",
            signature_ok: true,
            digest_match: None,
            foreign_agent_input_digest: None,
            foreign_agent_output_digest: None,
        };
        let half_citation = state.emit_citing_record(&prov).unwrap().unwrap();
        let citation = InclusionCitation {
            half_capsule_id: &half,
            received_from: "node-b",
            via: "push",
            received_at: "2026-09-27T00:00:00Z",
            leaf_index: 3,
            mmr_size: 7,
            checkpoint_digest: "1".repeat(64).leak(),
            inclusion_proof_digest: "2".repeat(64).leak(),
        };
        let inclusion = state
            .emit_inclusion_citing_record(&citation)
            .unwrap()
            .unwrap();
        assert_eq!(
            inclusion.capsule["chain"]["parent_capsule_id"].as_str(),
            Some(half_citation.capsule_id.as_str())
        );
        assert!(state
            .emit_inclusion_citing_record(&citation)
            .unwrap()
            .is_none());

        drop(state);
        let reopened = CapsuleState::open(&dir, "node-under-test").expect("reopen");
        assert!(reopened
            .emit_inclusion_citing_record(&citation)
            .unwrap()
            .is_none());
        assert_eq!(
            reopened.chain_head().as_deref(),
            Some(inclusion.capsule_id.as_str())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A response with no tool call yields an absent tool_calls digest — never a
    /// digest over `[]`.
    #[test]
    fn output_sub_digests_absent_when_no_tool_calls() {
        let body = br#"{"choices":[{"message":{"role":"assistant","content":"hi"}}]}"#;
        let (tcd, rd) = output_sub_digests(body);
        assert!(tcd.is_none());
        assert!(rd.is_none());
    }

    /// A request carrying a spread of sampling knobs (float, int, and the
    /// llama.cpp-style `top_k`/`min_p`/`repeat_penalty`) seals ALL of them, with
    /// floats stringified (matching the digest-path convention) and ints kept as
    /// JSON numbers. This is the direct fix for the hardcoded `temperature=0.0`.
    #[test]
    fn parse_generation_parameters_captures_all_present_sampling_knobs() {
        let body = br#"{
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "min_p": 0.05,
            "seed": 12345,
            "repeat_penalty": 1.1,
            "max_tokens": 512,
            "n": 1,
            "stop": ["\n\n"]
        }"#;
        let gp = parse_generation_parameters(body);
        // Floats stringified through the same convention as the digest path.
        assert_eq!(gp["temperature"], Value::String("0.7".into()));
        assert_eq!(gp["top_p"], Value::String("0.95".into()));
        assert_eq!(gp["min_p"], Value::String("0.05".into()));
        assert_eq!(gp["repeat_penalty"], Value::String("1.1".into()));
        // Integers stay JSON numbers.
        assert_eq!(gp["top_k"], Value::from(40));
        assert_eq!(gp["seed"], Value::from(12345));
        assert_eq!(gp["max_tokens"], Value::from(512));
        assert_eq!(gp["n"], Value::from(1));
        // Non-numeric values (stop list) carried verbatim.
        assert_eq!(gp["stop"], Value::Array(vec![Value::String("\n\n".into())]));
        // NOT the old fabricated default.
        assert_ne!(gp["temperature"], Value::String("0.0".into()));
    }

    /// Honest-by-absence: a request that sent ONLY `temperature` and `seed`
    /// seals exactly those two keys — no other sampling knob is defaulted into
    /// the capsule. Absent stays absent.
    #[test]
    fn parse_generation_parameters_absent_stays_absent() {
        let body = br#"{"model":"m","temperature":0.2,"seed":7}"#;
        let gp = parse_generation_parameters(body);
        assert_eq!(gp.len(), 2, "only the two present keys are sealed");
        assert!(gp.contains_key("temperature"));
        assert!(gp.contains_key("seed"));
        // None of the omitted knobs are present — not even as a zero/default.
        for absent in [
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "repeat_penalty",
            "stop",
        ] {
            assert!(
                !gp.contains_key(absent),
                "{absent} was not in the request and must NOT be sealed"
            );
        }
    }

    /// A JSON `null` value is treated as absent (not sealed as a fabricated
    /// value), matching the Python reference's `is not None` guard.
    #[test]
    fn parse_generation_parameters_treats_null_as_absent() {
        let body = br#"{"temperature":0.5,"top_p":null}"#;
        let gp = parse_generation_parameters(body);
        assert!(gp.contains_key("temperature"));
        assert!(!gp.contains_key("top_p"));
    }

    /// A non-object / non-JSON body claims NO generation parameters (empty map),
    /// never a fabricated default.
    #[test]
    fn parse_generation_parameters_empty_for_non_object_body() {
        assert!(parse_generation_parameters(b"not json").is_empty());
        assert!(parse_generation_parameters(b"[1,2,3]").is_empty());
    }

    /// END-TO-END (served path): a real admitted exchange whose request carried
    /// `temperature`, `top_k`, `min_p`, `repeat_penalty`, and `seed` seals a
    /// capsule whose `x-mesh-poc-v1.generation_parameters` holds exactly those,
    /// and NO hardcoded `temperature=0.0`.
    #[test]
    fn emit_for_exchange_seals_the_real_requested_generation_parameters() {
        let dir = std::env::temp_dir().join(format!("cap-gp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");

        let request_body = br#"{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":0.7,"top_k":40,"min_p":0.05,"repeat_penalty":1.1,"seed":99}"#;
        let response_body = br#"{"id":"x","choices":[{"message":{"role":"assistant","content":"hello"}}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}"#;
        let exchange = ExchangeRecord {
            model: "m",
            client_nonce: Some("nonce-1"),
            request_bytes: request_body,
            response_bytes: response_body,
            latency_ms: 12.0,
            exchange_id: Some("e-1"),
            requesting_party: Some("party-1"),
            host_provenance: None,
        };
        let emitted = state.emit_for_exchange(&exchange).expect("seal exchange");
        let gp = &emitted.capsule["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"]
            ["generation_parameters"];
        assert_eq!(gp["temperature"], Value::String("0.7".into()));
        assert_eq!(gp["min_p"], Value::String("0.05".into()));
        assert_eq!(gp["repeat_penalty"], Value::String("1.1".into()));
        assert_eq!(gp["top_k"], Value::from(40));
        assert_eq!(gp["seed"], Value::from(99));
        // Only what was requested — no other knob defaulted in.
        let obj = gp.as_object().expect("generation_parameters is an object");
        assert_eq!(obj.len(), 5);
        // The old fabricated default is gone.
        assert_ne!(gp["temperature"], Value::String("0.0".into()));

        // When AAC_CAPSULE_EXPORT_DIR is set, export the sealed capsule + its
        // detached COSE_Sign1 + the pubkey so an EXTERNAL verifier (the Python
        // agent_action_capsule / Go COSE reference) can confirm a capsule that
        // carries REAL requested generation parameters still verifies GREEN and
        // its capsule_id (which commits generation_parameters) is intact.
        if let Ok(export_dir) = std::env::var("AAC_CAPSULE_EXPORT_DIR") {
            let export = std::path::Path::new(&export_dir);
            std::fs::create_dir_all(export).expect("mk export dir");
            // Export the EXACT canonical payload bytes the COSE statement signs
            // (not a pretty reserialization), so an external verifier can
            // byte-compare the capsule.json against the COSE payload.
            std::fs::write(
                export.join("SEALED-with-generation-params.json"),
                crate::producer::capsule::payload_bytes(&emitted.capsule),
            )
            .expect("write capsule");
            let cose_src = dir
                .join("ledger")
                .join("signed-statements")
                .join(format!("{}.cose", emitted.capsule_id));
            std::fs::copy(&cose_src, export.join("SEALED-with-generation-params.cose"))
                .expect("copy cose");
            std::fs::copy(
                dir.join("keys").join("node-key.pub.pem"),
                export.join("node-key.pub.pem"),
            )
            .expect("copy pubkey");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The executable is read and hashed once per process, however many
    /// records are sealed, on either seal path, and every record carries the
    /// same measurement. (The read counter is process-wide, so this holds
    /// across every test in this binary, not only the seals below.)
    #[test]
    fn the_executable_is_read_once_per_process_across_many_seals() {
        let dir = std::env::temp_dir().join(format!("cap-self-hash-once-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let digest_of = |capsule: &Value| {
            capsule
                .pointer("/model_attestation/compute_attestation/x-mesh-poc-v1/evidence_refs/binary_attestation/digest")
                .and_then(Value::as_str)
                .expect("a binary measurement on the record")
                .to_string()
        };
        let mut digests = Vec::new();
        for n in 0..4 {
            let id = format!("e-{n}");
            let exchange = ExchangeRecord {
                model: "m",
                client_nonce: Some("nonce"),
                request_bytes: br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
                response_bytes:
                    br#"{"id":"x","choices":[{"message":{"role":"assistant","content":"hello"}}]}"#,
                latency_ms: 1.0,
                exchange_id: Some(id.as_str()),
                requesting_party: Some("party"),
                host_provenance: None,
            };
            digests.push(digest_of(
                &state.emit_for_exchange(&exchange).expect("seal").capsule,
            ));
            let observed = ObservedHostExchange {
                model: "m",
                exchange_id: Some(id.as_str()),
                request_digest: Some("a".repeat(64).leak()),
                response_digest: None,
                tool_calls_digest: None,
                reasoning_digest: None,
                usage: None,
                host_provenance: HostProvenance::default(),
                dispatch_path: DispatchPath::RawProxy,
                nonce: None,
                peer_capsule_id: None,
                peer_capsule_id_provenance: None,
                twin_bracket_id: None,
                response_text_digest: None,
            };
            digests.push(digest_of(
                &state
                    .emit_for_observed_host_exchange(&observed)
                    .expect("seal")
                    .capsule,
            ));
        }
        assert_eq!(digests.len(), 8);
        assert!(digests.iter().all(|d| d == &digests[0]), "{digests:?}");
        assert_eq!(
            crate::producer::runtime_attest::exe_reads(),
            1,
            "the executable was read more than once in this process"
        );
    }

    /// The host-served OBSERVE path holds no request bytes (only a forwarded
    /// digest), so it seals an EMPTY generation-parameter set — never the old
    /// fabricated `temperature=0.0`. Absent facts stay absent.
    #[test]
    fn observed_host_exchange_seals_no_fabricated_generation_parameters() {
        let dir = std::env::temp_dir().join(format!("cap-gpo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let observed = ObservedHostExchange {
            model: "m",
            exchange_id: Some("e"),
            request_digest: Some("a".repeat(64).leak()),
            response_digest: None,
            tool_calls_digest: None,
            reasoning_digest: None,
            usage: None,
            host_provenance: HostProvenance::default(),
            dispatch_path: DispatchPath::RawProxy,
            nonce: None,
            peer_capsule_id: None,
            peer_capsule_id_provenance: None,
            twin_bracket_id: None,
            response_text_digest: None,
        };
        let emitted = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal");
        let gp = &emitted.capsule["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"]
            ["generation_parameters"];
        let obj = gp.as_object().expect("generation_parameters is an object");
        assert!(
            obj.is_empty(),
            "observe path holds no request -> no sampling knobs, and NO fabricated temperature=0.0"
        );
        assert!(gp.get("temperature").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// END-TO-END: a host-served observed exchange that forwarded a real
    /// `request_digest` seals a `host_binding` group carrying that SAME value
    /// under mesh-llm's own construction — an independent second claim
    /// alongside `agent_input_digest`, not a re-derivation of it.
    #[test]
    fn observed_host_exchange_seals_host_binding_from_forwarded_request_digest() {
        let dir = std::env::temp_dir().join(format!("cap-hb-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let observed = ObservedHostExchange {
            model: "m",
            exchange_id: Some("e"),
            request_digest: Some(
                "a6329c5ebb66562f38a8136a8d8511b6aeed166e4c7d889b9133ac96fc49a9d5",
            ),
            response_digest: None,
            tool_calls_digest: None,
            reasoning_digest: None,
            usage: None,
            host_provenance: HostProvenance::default(),
            dispatch_path: DispatchPath::RawProxy,
            nonce: None,
            peer_capsule_id: None,
            peer_capsule_id_provenance: None,
            twin_bracket_id: None,
            response_text_digest: None,
        };
        let emitted = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal");
        let ca = &emitted.capsule["model_attestation"]["compute_attestation"];
        assert_eq!(
            ca["host_binding"]["digest"],
            "a6329c5ebb66562f38a8136a8d8511b6aeed166e4c7d889b9133ac96fc49a9d5"
        );
        assert_eq!(
            ca["host_binding"]["construction"],
            crate::producer::capsule::MESH_LLM_REQUEST_BODY_SHA256_V1
        );
        assert_eq!(
            ca["host_binding"]["purpose"],
            crate::producer::capsule::HOST_LOG_JOIN
        );
        // Both are the SAME value here (this rig forwards one digest for
        // both), but structurally they are two independent claims: the
        // capsule carries them under two different keys, and the sealed
        // capsule validates structurally without comparing the two.
        assert_eq!(ca["agent_input_digest"], ca["host_binding"]["digest"]);
        assert!(crate::producer::capsule::validate_host_binding(&emitted.capsule).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Absence rule on the OBSERVE path: when the host forwards NO
    /// `request_digest`, `host_binding` is OMITTED entirely — never `null`,
    /// never a digest over an empty/unknown input.
    #[test]
    fn observed_host_exchange_omits_host_binding_when_host_forwards_no_digest() {
        let dir = std::env::temp_dir().join(format!("cap-hbn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let observed = ObservedHostExchange {
            model: "m",
            exchange_id: Some("e"),
            request_digest: None,
            response_digest: None,
            tool_calls_digest: None,
            reasoning_digest: None,
            usage: None,
            host_provenance: HostProvenance::default(),
            dispatch_path: DispatchPath::RawProxy,
            nonce: None,
            peer_capsule_id: None,
            peer_capsule_id_provenance: None,
            twin_bracket_id: None,
            response_text_digest: None,
        };
        let emitted = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal");
        let ca = &emitted.capsule["model_attestation"]["compute_attestation"];
        assert!(
            ca.get("host_binding").is_none(),
            "no host-forwarded digest -> host_binding must be absent, not null"
        );
        assert!(crate::producer::capsule::validate_host_binding(&emitted.capsule).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The plugin-served (`emit_for_exchange`) path is not the host-served
    /// observe path — there is no separate mesh-llm-native digest to bind, so
    /// `host_binding` stays absent there too.
    #[test]
    fn plugin_served_exchange_carries_no_host_binding() {
        let dir = std::env::temp_dir().join(format!("cap-hbe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let exchange = ExchangeRecord {
            model: "m",
            client_nonce: Some("n"),
            request_bytes: br#"{"model":"m","messages":[]}"#,
            response_bytes: br#"{"model":"m","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#,
            latency_ms: 1.0,
            exchange_id: Some("e"),
            requesting_party: Some("client"),
            host_provenance: None,
        };
        let emitted = state.emit_for_exchange(&exchange).expect("seal");
        let ca = &emitted.capsule["model_attestation"]["compute_attestation"];
        assert!(ca.get("host_binding").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Evidence Layer -00 §12.1 on the live seal path: every sealed record
    /// carries a fresh store nonce (never the client's) and minute-granular
    /// committed times; padding to the bucket goes through the same ledger
    /// writer, stays off the chain, and the ledger reopens clean with the
    /// next seal chaining to the last REAL record.
    #[test]
    fn seals_carry_a_store_nonce_and_coarse_time_and_padding_stays_off_the_chain() {
        let dir = std::env::temp_dir().join(format!("cap-pad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let exchange = ExchangeRecord {
            model: "m",
            client_nonce: Some("client-nonce"),
            request_bytes: br#"{"model":"m","messages":[]}"#,
            response_bytes: br#"{"model":"m","choices":[]}"#,
            latency_ms: 1.0,
            exchange_id: Some("e"),
            requesting_party: Some("client"),
            host_provenance: None,
        };
        let first = state.emit_for_exchange(&exchange).expect("seal");
        let second = state.emit_for_exchange(&exchange).expect("seal");
        let nonce = |c: &Value| {
            c["model_attestation"]["compute_attestation"]["store_nonce"]
                .as_str()
                .expect("store_nonce present")
                .to_string()
        };
        assert_eq!(nonce(&first.capsule).len(), 64);
        assert_ne!(nonce(&first.capsule), nonce(&second.capsule));
        for c in [&first.capsule, &second.capsule] {
            let ts = c["timestamp"].as_str().unwrap();
            assert!(crate::producer::timestamp::is_minute_granular(ts), "{ts}");
            let measured = &c["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"]
                ["evidence_refs"]["binary_attestation"]["measured_at"];
            if let Some(m) = measured.as_str() {
                assert!(crate::producer::timestamp::is_minute_granular(m), "{m}");
            }
        }

        assert_eq!(state.pad_ledger_to_bucket(32).expect("pad"), 32);
        assert_eq!(
            state.pad_ledger_to_bucket(32).expect("pad again"),
            32,
            "no double pad"
        );
        assert_eq!(
            state.chain_head().as_deref(),
            Some(second.capsule_id.as_str())
        );
        let third = state
            .emit_for_exchange(&exchange)
            .expect("seal after padding");
        assert_eq!(
            third.capsule["chain"]["parent_capsule_id"],
            second.capsule_id.as_str(),
            "the chain skips padding"
        );
        drop(state);

        let (ledger, report) =
            crate::producer::index::open_ledger(&dir.join("ledger")).expect("reopen");
        assert_eq!(report.valid_entries, 33);
        assert_eq!(ledger.chain_head(), Some(third.capsule_id.as_str()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// END-TO-END: a host-served observed exchange that forwarded a real
    /// `tool_calls_digest` and `response_digest` seals a capsule whose
    /// `compute_attestation.tool_calls_digest` equals the forwarded (Python-
    /// reference) value, and whose `agent_output_digest` binds the forwarded
    /// REAL response digest — not the terminal-facts fallback.
    #[test]
    fn observed_host_exchange_seals_real_tool_calls_and_response_digest() {
        let dir = std::env::temp_dir().join(format!("cap-tcd-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");

        // The REAL served SETI@Home response body carrying the model's real
        // web_search tool call (the exact tool_calls from a live capture), plus real usage. The host
        // computes response_digest / tool_calls_digest over exactly this body at
        // its JSON-relay delivery point; we reproduce those here to bind the
        // exported capsule to real values throughout (no placeholder digests).
        let response_body = br#"{"id":"chatcmpl-seti","object":"chat.completion","created":1788060371,"model":"llama-3.2-3b-instruct","choices":[{"index":0,"message":{"role":"assistant","content":"","tool_calls":[{"function":{"arguments":"{\"query\": \"mesh-llm vs SETI@Home\"}","name":"web_search"},"id":"call_719a955fb46a41008dd847d412f00795","type":"function"}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":202,"completion_tokens":25,"total_tokens":227}}"#;
        // Real response-body digest (host's `response_digest`) and real
        // tool_calls_digest (host's `tool_calls_digest`) over that body.
        let response_digest = canonical_body_digest(response_body).expect("resp digest");
        let (tool_calls_digest_owned, reasoning_digest_owned) = output_sub_digests(response_body);
        let tool_calls_digest = tool_calls_digest_owned.expect("real tool_calls digest");
        // Independent sanity: the tool_calls_digest equals the Python-reference value.
        assert_eq!(
            tool_calls_digest,
            "f294be8a53bb9c29cd94472721f0857591f34b23fe010882de79b9fb210b1395"
        );
        assert!(reasoning_digest_owned.is_none());
        let observed = ObservedHostExchange {
            model: "llama-3.2-3b-instruct",
            exchange_id: Some("exch-seti"),
            request_digest: Some(
                "a6329c5ebb66562f38a8136a8d8511b6aeed166e4c7d889b9133ac96fc49a9d5",
            ),
            response_digest: Some(&response_digest),
            tool_calls_digest: Some(&tool_calls_digest),
            reasoning_digest: None,
            usage: Some(TokenUsage {
                prompt_tokens: 202,
                completion_tokens: 25,
                total_tokens: 227,
            }),
            host_provenance: HostProvenance {
                architecture: Some("llama".to_string()),
                model_identity_hash: Some("904548955b8a6478".to_string()),
                ..Default::default()
            },
            dispatch_path: DispatchPath::RawProxy,
            nonce: None,
            peer_capsule_id: None,
            peer_capsule_id_provenance: None,
            twin_bracket_id: None,
            response_text_digest: None,
        };
        let emitted = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal observed host exchange");
        let ca = &emitted.capsule["model_attestation"]["compute_attestation"];
        // The REAL tool_calls_digest is sealed into the capsule.
        assert_eq!(ca["tool_calls_digest"], tool_calls_digest);
        // agent_output_digest binds the forwarded REAL response digest.
        assert_eq!(ca["agent_output_digest"], response_digest);
        // Non-reasoning model -> reasoning_digest omitted entirely (honest null).
        assert!(ca.get("reasoning_digest").is_none());

        // When AAC_CAPSULE_EXPORT_DIR is set, export the sealed capsule + its
        // detached COSE_Sign1 statement + the signing pubkey to that dir, so an
        // EXTERNAL verifier (agent-action-capsule / the Rust verify_capsule bin)
        // can confirm the real-tool_calls capsule verifies GREEN out-of-process.
        if let Ok(export_dir) = std::env::var("AAC_CAPSULE_EXPORT_DIR") {
            let export = std::path::Path::new(&export_dir);
            std::fs::create_dir_all(export).expect("mk export dir");
            std::fs::write(
                export.join("SEALED-with-tool-calls.json"),
                serde_json::to_vec_pretty(&emitted.capsule).unwrap(),
            )
            .expect("write capsule");
            let cose_src = dir
                .join("ledger")
                .join("signed-statements")
                .join(format!("{}.cose", emitted.capsule_id));
            std::fs::copy(&cose_src, export.join("SEALED-with-tool-calls.cose"))
                .expect("copy cose");
            std::fs::copy(
                dir.join("keys").join("node-key.pub.pem"),
                export.join("node-key.pub.pem"),
            )
            .expect("copy pubkey");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// When the host forwarded NO response digest, `agent_output_digest` is
    /// OMITTED (EM review #2): a digest of the terminal facts (model + usage)
    /// sealed as a bare 64-hex reads as a body digest and gets compared as
    /// one. tool_calls_digest is likewise absent when none was forwarded.
    #[test]
    fn observed_host_exchange_falls_back_when_no_response_digest_forwarded() {
        let dir = std::env::temp_dir().join(format!("cap-fb-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let observed = ObservedHostExchange {
            model: "m",
            exchange_id: Some("e"),
            request_digest: Some("a".repeat(64).leak()),
            response_digest: None,
            tool_calls_digest: None,
            reasoning_digest: None,
            usage: None,
            host_provenance: HostProvenance {
                architecture: Some("llama".to_string()),
                ..Default::default()
            },
            dispatch_path: DispatchPath::RawProxy,
            nonce: None,
            peer_capsule_id: None,
            peer_capsule_id_provenance: None,
            twin_bracket_id: None,
            response_text_digest: None,
        };
        let emitted = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal");
        let ca = &emitted.capsule["model_attestation"]["compute_attestation"];
        assert!(ca.get("tool_calls_digest").is_none());
        assert!(
            ca.get("agent_output_digest").is_none(),
            "no body digest, no output digest"
        );
        // The request side is unaffected.
        assert_eq!(ca["agent_input_digest"], "a".repeat(64));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // =======================================================================
    // twin_bracket_id -- forwarded verbatim off the terminal envelope, never
    // minted here.
    // =======================================================================

    /// An observed host exchange whose terminal envelope carried a
    /// `twin_bracket_id` seals a capsule that carries the SAME id verbatim
    /// under `serving_provenance`. MUTANT: drop the forward (stop passing
    /// `twin_bracket_id` through in `emit_for_observed_host_exchange`) and
    /// this assertion goes red.
    #[test]
    fn observed_host_exchange_with_twin_bracket_id_seals_it_verbatim() {
        let dir = std::env::temp_dir().join(format!("cap-twin-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let observed = ObservedHostExchange {
            model: "m",
            exchange_id: Some("e"),
            request_digest: None,
            response_digest: None,
            tool_calls_digest: None,
            reasoning_digest: None,
            usage: None,
            host_provenance: HostProvenance::default(),
            dispatch_path: DispatchPath::RawProxy,
            nonce: None,
            peer_capsule_id: None,
            peer_capsule_id_provenance: None,
            twin_bracket_id: Some("twin-abc123"),
            response_text_digest: None,
        };
        let emitted = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal");
        let sp = &emitted.capsule["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"]
            ["serving_provenance"];
        assert_eq!(sp["twin_bracket_id"], "twin-abc123");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Absence rule: when the terminal envelope carried no `twin_bracket_id`
    /// (the overwhelming majority of exchanges), the sealed capsule omits
    /// the field entirely -- never `null` -- so the row renders as an
    /// ordinary row, never a half-bracket.
    #[test]
    fn observed_host_exchange_without_twin_bracket_id_omits_it() {
        let dir = std::env::temp_dir().join(format!("cap-twin-abs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let observed = ObservedHostExchange {
            model: "m",
            exchange_id: Some("e"),
            request_digest: None,
            response_digest: None,
            tool_calls_digest: None,
            reasoning_digest: None,
            usage: None,
            host_provenance: HostProvenance::default(),
            dispatch_path: DispatchPath::RawProxy,
            nonce: None,
            peer_capsule_id: None,
            peer_capsule_id_provenance: None,
            twin_bracket_id: None,
            response_text_digest: None,
        };
        let emitted = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal");
        let sp = &emitted.capsule["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"]
            ["serving_provenance"];
        assert!(
            sp.get("twin_bracket_id").is_none(),
            "an absent twin_bracket_id must be omitted, not null"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two rows of one real twin -- this node's own requester-side capsule
    /// and a peer's served-side capsule (simulated here as two separate
    /// `CapsuleState`s, mirroring the requester/served pairing test above) --
    /// both forward the SAME host-minted `twin_bracket_id` off their own
    /// terminal envelope, so both sealed capsules carry the identical value
    /// even though everything else about them (node, exchange id) differs.
    /// This is what lets a reader bracket the two rows together.
    #[test]
    fn two_observed_host_exchanges_sharing_a_twin_bracket_id_seal_the_identical_value() {
        let dir_a = std::env::temp_dir().join(format!("cap-twin-a-{}", std::process::id()));
        let dir_b = std::env::temp_dir().join(format!("cap-twin-b-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
        let state_a = CapsuleState::open(&dir_a, "node-a").expect("open state a");
        let state_b = CapsuleState::open(&dir_b, "node-b").expect("open state b");

        let observed_a = ObservedHostExchange {
            model: "m",
            exchange_id: Some("exch-a"),
            request_digest: None,
            response_digest: None,
            tool_calls_digest: None,
            reasoning_digest: None,
            usage: None,
            host_provenance: HostProvenance::default(),
            dispatch_path: DispatchPath::RawProxy,
            nonce: None,
            peer_capsule_id: None,
            peer_capsule_id_provenance: None,
            twin_bracket_id: Some("twin-shared"),
            response_text_digest: None,
        };
        let observed_b = ObservedHostExchange {
            model: "m",
            exchange_id: Some("exch-b"),
            request_digest: None,
            response_digest: None,
            tool_calls_digest: None,
            reasoning_digest: None,
            usage: None,
            host_provenance: HostProvenance::default(),
            dispatch_path: DispatchPath::RawProxy,
            nonce: None,
            peer_capsule_id: None,
            peer_capsule_id_provenance: None,
            twin_bracket_id: Some("twin-shared"),
            response_text_digest: None,
        };

        let emitted_a = state_a
            .emit_for_observed_host_exchange(&observed_a)
            .expect("seal a");
        let emitted_b = state_b
            .emit_for_observed_host_exchange(&observed_b)
            .expect("seal b");
        let sp_a = &emitted_a.capsule["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"]
            ["serving_provenance"];
        let sp_b = &emitted_b.capsule["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"]
            ["serving_provenance"];
        assert_eq!(sp_a["twin_bracket_id"], sp_b["twin_bracket_id"]);
        assert_eq!(sp_a["twin_bracket_id"], "twin-shared");
        // Different exchanges -- distinct capsule_ids, same bracket.
        assert_ne!(
            emitted_a.capsule["capsule_id"],
            emitted_b.capsule["capsule_id"]
        );

        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }

    /// Regression: two byte-different, semantically
    /// identical request bodies (key order + whitespace only) must digest to
    /// the SAME value. Before the fix, `emit_for_exchange` hashed the raw wire
    /// bytes (`hex_sha256`), so this pair produced two different digests --
    /// the exact defect this test pins closed. Reproduced against the
    /// pre-fix `hex_sha256(bytes)` path directly, for contrast.
    #[test]
    fn canonical_body_digest_is_stable_across_key_order_and_whitespace() {
        let a = br#"{"model":"m","temperature":0.7}"#;
        let b = br#"{"temperature": 0.7, "model": "m"}"#;

        let digest_a = canonical_body_digest(a).expect("digest a");
        let digest_b = canonical_body_digest(b).expect("digest b");
        assert_eq!(
            digest_a, digest_b,
            "canonical_body_digest must be format-invariant"
        );

        // contrast: the pre-fix raw-byte hash IS format-sensitive -- proves
        // this pair is a real positive control, not a vacuous equality.
        assert_ne!(
            hex_sha256(a),
            hex_sha256(b),
            "sanity: raw hex_sha256 over wire bytes must differ for this pair \
             (otherwise the pair doesn't exercise the bug this test guards)"
        );
    }

    /// Digest equality with the reference implementation: the expected digest
    /// is the Agent Action Capsule reference JSON-DIGEST
    /// (`agent_action_capsule.canonical.json_digest`) over the identical JSON
    /// value, its floats written as decimal strings the way
    /// `stringify_floats` writes them (`"temperature": "0.7"`,
    /// `"top_p": "1.0"`).
    ///
    /// `top_p: 1.0` exercises the whole-number-float edge case
    /// (`python_repr_f64` must emit "1.0", not Rust's default "1") that a
    /// naive float-to-string port would get wrong.
    #[test]
    fn canonical_body_digest_matches_the_reference_json_digest() {
        let body = br#"{"model": "hermes-2-pro-mistral-7b", "messages": [{"role": "user", "content": "hello"}], "temperature": 0.7, "top_p": 1.0, "max_tokens": 512}"#;
        let expected = "a6329c5ebb66562f38a8136a8d8511b6aeed166e4c7d889b9133ac96fc49a9d5";
        assert_eq!(canonical_body_digest(body).expect("digest"), expected);
    }

    /// Digest equality with the reference implementation for a request that
    /// carries tool-call ids: an assistant turn's `tool_calls[].id` and the
    /// `tool` message's `tool_call_id` answering it. The expected digests are
    /// `agent_action_capsule.canonical.json_digest` over the same values
    /// (floats written as decimal strings), for the body as sent and for the
    /// same body with only the tool-call id changed: the id is committed, not
    /// normalized away.
    #[test]
    fn canonical_body_digest_commits_to_tool_call_ids_like_the_reference() {
        let body = br#"{"model": "hermes-2-pro-mistral-7b", "messages": [{"role": "user", "content": "What's the weather in Paris?"}, {"role": "assistant", "content": null, "tool_calls": [{"id": "call_7f3a9c", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]}, {"role": "tool", "tool_call_id": "call_7f3a9c", "content": "{\"temp_c\":18.5}"}], "temperature": 0.2, "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}"#;
        assert_eq!(
            canonical_body_digest(body).expect("digest"),
            "c3f3013671f173dbe338890ff2c725c0cf32ec94942fee5f552ec5987eb389bc"
        );
        let renamed = String::from_utf8(body.to_vec())
            .unwrap()
            .replace("call_7f3a9c", "call_000000");
        assert_eq!(
            canonical_body_digest(renamed.as_bytes()).expect("digest"),
            "951164bcd7fe49eef92566623bd2cce86bc123e4adc65f93f0c6ad778a1a3dd3"
        );
    }

    /// `parse_usage` lifts REAL token counts from the response body's `usage`
    /// object — the only honest source of usage the plugin has.
    #[test]
    fn parse_usage_reads_real_token_counts_from_the_response_body() {
        let body = br#"{"id":"x","usage":{"prompt_tokens":128,"completion_tokens":64,"total_tokens":192}}"#;
        let usage = parse_usage(body).expect("usage present");
        assert_eq!(usage.prompt_tokens, 128);
        assert_eq!(usage.completion_tokens, 64);
        assert_eq!(usage.total_tokens, 192);
    }

    /// A body with no `usage` yields `None` — never a fabricated zero-usage
    /// object. Absence of a fact is recorded as absence, not invented as zero.
    #[test]
    fn parse_usage_is_none_when_the_body_has_no_usage() {
        let body = br#"{"id":"x","choices":[]}"#;
        assert!(parse_usage(body).is_none());
    }

    /// When the body omits `total_tokens`, it is DERIVED from the two real
    /// counts (a genuine sum), not left blank or faked.
    #[test]
    fn parse_usage_derives_total_from_the_two_real_counts_when_absent() {
        let body = br#"{"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;
        let usage = parse_usage(body).expect("usage present");
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn python_repr_f64_keeps_the_decimal_point_python_repr_does() {
        assert_eq!(python_repr_f64(1.0), "1.0");
        assert_eq!(python_repr_f64(0.0), "0.0");
        assert_eq!(python_repr_f64(0.7), "0.7");
    }

    /// [B3, extended by rung 3a] The runtime/binary-attestation rung: a real
    /// sealed exchange records the measured hash of the serving binary (the
    /// running test binary, via `current_exe`) in BOTH the
    /// `evidence_refs.binary_attestation` slot and the
    /// `compute_attestation.runtime` field, carrying an honesty label plus a
    /// node-key signature that verifies. This is the end-to-end proof the
    /// rung ships the hash + the label, not just the unit.
    ///
    /// The label itself is HOST-DEPENDENT and this test does not hardcode it:
    /// on macOS with a working kernel query the rung upgrades to
    /// `os_measured` (rung 3a — see `runtime_attest.rs`); elsewhere it is
    /// `self_measured`. What must hold regardless of platform: a real digest,
    /// a verifying signature, and an in-band trust-ceiling `context` naming
    /// whichever grade was actually produced.
    #[test]
    fn emit_for_exchange_records_binary_attestation_rung_honestly() {
        let dir = std::env::temp_dir().join(format!("cap-binary-rung-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");

        let request_body =
            br#"{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":0.7}"#;
        let response_body = br#"{"id":"x","choices":[{"message":{"role":"assistant","content":"hello"}}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}"#;
        let exchange = ExchangeRecord {
            model: "m",
            client_nonce: Some("nonce-1"),
            request_bytes: request_body,
            response_bytes: response_body,
            latency_ms: 12.0,
            exchange_id: Some("e-1"),
            requesting_party: Some("party-1"),
            host_provenance: None,
        };
        let emitted = state.emit_for_exchange(&exchange).expect("seal exchange");

        // The test runner IS a resolvable, readable binary, so the rung must have
        // measured it (this is not the graceful-degradation branch).
        let mesh_poc =
            &emitted.capsule["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"];
        let att = &mesh_poc["evidence_refs"]["binary_attestation"];

        // THE honesty label ships, in the evidence slot — self_measured or
        // os_measured, whichever this host's kernel query actually produced.
        let measurement_class = att["measurement_class"]
            .as_str()
            .expect("measurement_class present")
            .to_string();
        assert!(
            measurement_class == "self_measured" || measurement_class == "os_measured",
            "the rung must label itself one of the two grades it can honestly produce, got: {measurement_class}"
        );
        assert_eq!(att["type"], "binary_attestation");

        // It records a REAL 64-hex SHA-256 of the measured binary (not a
        // fabricated zero-hash), and the runtime field carries that same hash.
        let digest = att["digest"].as_str().expect("digest present");
        assert_eq!(digest.len(), 64, "records a real sha256 hex digest");
        assert!(
            digest.chars().all(|c| c.is_ascii_hexdigit()),
            "digest is hex"
        );
        assert_ne!(digest, "0".repeat(64), "not the placeholder zero-hash");

        // CHANGED (runtime/model extension draft): runtime moved from a flat
        // "<digest>:<class>:<name>" string to a {name, runtime_digest,
        // measurement_class} object -- same facts, structured fields.
        let runtime = &emitted.capsule["model_attestation"]["compute_attestation"]["runtime"];
        assert_eq!(
            runtime["runtime_digest"]
                .as_str()
                .expect("runtime_digest present"),
            digest,
            "runtime field carries the real measured binary hash, not the 0*64 placeholder"
        );
        assert_eq!(
            runtime["measurement_class"]
                .as_str()
                .expect("measurement_class present"),
            measurement_class,
            "runtime field carries whichever grade was actually produced"
        );

        // The trust ceiling is stated IN the sealed record (not only in docs),
        // whichever grade actually shipped.
        let context = att["context"].as_str().expect("context present");
        if measurement_class == "os_measured" {
            assert!(context.contains("os_measured"));
            assert!(context.contains("SIP"));
        } else {
            assert!(context.contains("self-measured"));
            assert!(context.contains("untampered before hashing"));
        }

        // The node-key signature over the hex digest verifies under the same key
        // the capsule is signed with (code-signing gesture, end to end).
        // Re-measure this binary with the SAME node key and confirm the sealed
        // attestation's signature verifies via the crate's own verify helper.
        let remeasured = crate::producer::runtime_attest::measure_self(
            &state.keys,
            "2026-08-30T00:00:00Z".to_string(),
        )
        .expect("test binary is measurable");
        assert_eq!(
            remeasured.binary_sha256, digest,
            "self-measurement is deterministic for the same binary"
        );
        assert_eq!(
            remeasured.signature,
            att["signature"].as_str().unwrap(),
            "the sealed signature is exactly the node key's signature over the binary hash"
        );
        assert!(
            remeasured.verify_signature(&state.keys.verifying_key()),
            "node key must verify the binary-hash signature it produced"
        );
        assert_eq!(
            att["signature"].as_str().unwrap().len(),
            128,
            "ed25519 hex sig"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every string value in a sealed record, at any depth.
    fn sealed_strings(value: &Value, out: &mut Vec<String>) {
        match value {
            Value::String(s) => out.push(s.clone()),
            Value::Array(items) => items.iter().for_each(|v| sealed_strings(v, out)),
            Value::Object(map) => map.values().for_each(|v| sealed_strings(v, out)),
            _ => {}
        }
    }

    /// Fails if any sealed string names `hostname` or looks like an absolute
    /// path (POSIX root, Windows drive, this binary's directory or `$HOME`).
    fn assert_no_host_identifying_strings(capsule: &Value, hostname: &str) {
        let exe = std::env::current_exe().expect("test binary path");
        let exe_dir = exe
            .parent()
            .expect("exe has a parent")
            .display()
            .to_string();
        let home = std::env::var("HOME").ok().filter(|h| h.len() > 1);
        let mut strings = Vec::new();
        sealed_strings(capsule, &mut strings);
        for s in &strings {
            assert!(
                !s.contains(hostname),
                "sealed record carries the hostname: {s}"
            );
            assert!(
                !s.starts_with('/'),
                "sealed record carries an absolute path: {s}"
            );
            let bytes = s.as_bytes();
            assert!(
                !(bytes.len() > 2 && bytes[0].is_ascii_alphabetic() && &bytes[1..3] == b":\\"),
                "sealed record carries a drive path: {s}"
            );
            assert!(
                !s.contains(&exe_dir),
                "sealed record carries the binary's directory: {s}"
            );
            if let Some(home) = &home {
                assert!(
                    !s.contains(home.as_str()),
                    "sealed record carries $HOME: {s}"
                );
            }
        }
    }

    /// AAC-05 §5.2 on the OBSERVE path: with no host digest of the real
    /// response body, the effect is `dispatched` and carries no
    /// `response_digest` (never `confirmed` over the terminal-facts digest),
    /// no request digest when none was forwarded (never the sentinel), and
    /// the record is `fyi` -- the plugin decided nothing.
    #[test]
    fn observe_path_without_host_digests_is_dispatched_fyi_with_no_placeholders() {
        let dir = std::env::temp_dir().join(format!("cap-obs-nodig-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let emitted = state
            .emit_for_observed_host_exchange(&observed_with(DispatchPath::RawProxy, None))
            .expect("seal observed");
        let c = &emitted.capsule;
        assert_eq!(c["action_type"], "fyi");
        assert_eq!(c["effect"]["status"], "dispatched");
        assert_eq!(c["effect"]["type"], "inference_completion");
        assert_eq!(c["effect"]["effect_attestation"], "host_served_observed");
        assert!(c["effect"].get("response_digest").is_none());
        assert!(c["effect"].get("request_digest").is_none());
        assert_eq!(c["assurance"]["effect_mode"], "dispatched_unconfirmed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// With the host's real request/response body digests, the observe path
    /// seals `confirmed` over exactly those digests -- still `fyi`.
    #[test]
    fn observe_path_with_host_digests_is_confirmed_over_the_real_bodies() {
        let dir = std::env::temp_dir().join(format!("cap-obs-dig-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let req = "1".repeat(64);
        let resp = "2".repeat(64);
        let mut observed = observed_with(DispatchPath::RawProxy, None);
        observed.request_digest = Some(&req);
        observed.response_digest = Some(&resp);
        let c = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal")
            .capsule;
        assert_eq!(c["action_type"], "fyi");
        assert_eq!(c["effect"]["status"], "confirmed");
        assert_eq!(c["effect"]["request_digest"], req);
        assert_eq!(c["effect"]["response_digest"], resp);
        assert_eq!(c["assurance"]["effect_mode"], "confirmed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The admitted path is where the plugin decided: `decide`, `confirmed`
    /// over the digests of the bodies it held; the nonce fallback names this
    /// plugin, not a sidecar that isn't there.
    #[test]
    fn admitted_path_is_decide_and_confirmed_with_plugin_generated_fallback() {
        let dir = std::env::temp_dir().join(format!("cap-adm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let mut exchange = sample_exchange("party-1");
        exchange.client_nonce = None;
        let c = state.emit_for_exchange(&exchange).expect("seal").capsule;
        assert_eq!(c["action_type"], "decide");
        assert_eq!(c["effect"]["status"], "confirmed");
        assert_eq!(
            c["effect"]["response_digest"],
            canonical_body_digest(exchange.response_bytes).unwrap()
        );
        let poc = &c["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"];
        assert_eq!(poc["client_nonce"], "plugin_generated");
        assert_eq!(poc["client_nonce_source"], "plugin_generated_fallback");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// EM review #1: host digests in a recoverable spelling (uppercase hex,
    /// a `sha256:` prefix) are normalised, not refused -- a refusal drops
    /// the record.
    #[test]
    fn observe_path_normalises_host_digests_instead_of_refusing() {
        let dir = std::env::temp_dir().join(format!("cap-norm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let req = format!("sha256:{}", "1".repeat(64));
        let resp = "A".repeat(64);
        let mut observed = observed_with(DispatchPath::RawProxy, None);
        observed.request_digest = Some(&req);
        observed.response_digest = Some(&resp);
        let c = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal")
            .capsule;
        assert_eq!(c["effect"]["status"], "confirmed");
        assert_eq!(c["effect"]["request_digest"], "1".repeat(64));
        assert_eq!(c["effect"]["response_digest"], "a".repeat(64));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An unusable host digest is treated as absent: the record still seals,
    /// `dispatched` with no response digest.
    #[test]
    fn observe_path_treats_an_unusable_host_digest_as_absent() {
        let dir = std::env::temp_dir().join(format!("cap-bad-dig-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let mut observed = observed_with(DispatchPath::RawProxy, None);
        observed.request_digest = Some("md5:abc");
        observed.response_digest = Some("not-a-digest");
        let c = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal")
            .capsule;
        assert_eq!(c["effect"]["status"], "dispatched");
        assert!(c["effect"].get("request_digest").is_none());
        assert!(c["effect"].get("response_digest").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// EM review #1: a seal that fails must not consume a `seq` (the next
    /// good record would chain `prev_seq` to a record never written, which
    /// reads as withholding) and must be counted, never dropped silently.
    #[test]
    fn a_refused_seal_consumes_no_seq_and_is_counted() {
        let dir = std::env::temp_dir().join(format!("cap-noseq-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        // Make the ledger write fail: the statements directory becomes a file.
        let statements = dir.join("ledger").join("signed-statements");
        std::fs::remove_dir_all(&statements).expect("statements dir exists");
        std::fs::write(&statements, b"not a directory").unwrap();
        let observed = observed_with(DispatchPath::RawProxy, None);
        assert!(state.emit_for_observed_host_exchange(&observed).is_err());
        assert_eq!(state.observed_not_sealed(), 1);

        std::fs::remove_file(&statements).unwrap();
        std::fs::create_dir_all(&statements).unwrap();
        let c = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal")
            .capsule;
        assert_eq!(seq_of(&c), (1, None), "the refused attempt consumed no seq");
        assert_eq!(state.observed_not_sealed(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    const OPERATOR_HOSTNAME: &str = "operators-machine.local";

    /// The sealed body is pushed to every counterparty at completion, so
    /// with no operator opt-in it must carry neither the host-reported
    /// hostname nor any absolute path -- on both seal paths. The binary
    /// attestation still names the measured file, by basename.
    #[test]
    fn sealed_records_carry_no_hostname_and_no_absolute_path() {
        let dir = std::env::temp_dir().join(format!("cap-nohost-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let host = HostProvenance {
            hostname: Some(OPERATOR_HOSTNAME.to_string()),
            ..Default::default()
        };

        let mut exchange = sample_exchange("party-1");
        exchange.host_provenance = Some(host.clone());
        let admitted = state.emit_for_exchange(&exchange).expect("seal admitted");

        let mut observed = observed_with(DispatchPath::TypedFrontend, Some("node-under-test"));
        observed.host_provenance = host;
        let observed = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal observed");

        for capsule in [&admitted.capsule, &observed.capsule] {
            let poc = &capsule["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"];
            assert!(poc["serving_provenance"].get("hostname").is_none());
            assert_no_host_identifying_strings(capsule, OPERATOR_HOSTNAME);
            let att = &poc["evidence_refs"]["binary_attestation"];
            let exe = std::env::current_exe().unwrap();
            assert_eq!(
                att["binary_path"].as_str().expect("binary measured"),
                exe.file_name().unwrap().to_string_lossy(),
                "the measured binary is named by basename only"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hostname_is_sealed_only_on_explicit_opt_in() {
        for raw in [
            None,
            Some(""),
            Some("0"),
            Some("off"),
            Some("yes"),
            Some("TRUE"),
        ] {
            assert!(!hostname_opt_in_for(raw), "{raw:?} must not opt in");
        }
        for raw in ["1", "true", "on"] {
            assert!(hostname_opt_in_for(Some(raw)), "{raw} opts in");
        }
        let host = HostProvenance {
            hostname: Some(OPERATOR_HOSTNAME.to_string()),
            ..Default::default()
        };
        assert_eq!(sealed_hostname(&host, false), None);
        assert_eq!(
            sealed_hostname(&host, true).as_deref(),
            Some(OPERATOR_HOSTNAME)
        );
    }

    fn seq_of(capsule: &Value) -> (u64, Option<u64>) {
        let sp = &capsule["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"]
            ["serving_provenance"];
        (
            sp["seq"].as_u64().expect("seq is a u64"),
            sp["prev_seq"].as_u64(),
        )
    }

    fn sample_exchange<'a>(requesting_party: &'a str) -> ExchangeRecord<'a> {
        ExchangeRecord {
            model: "m",
            client_nonce: Some("nonce-1"),
            request_bytes: br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
            response_bytes:
                br#"{"id":"x","choices":[{"message":{"role":"assistant","content":"hello"}}]}"#,
            latency_ms: 5.0,
            exchange_id: Some("e-1"),
            requesting_party: Some(requesting_party),
            host_provenance: None,
        }
    }

    /// Sequence per counterparty: two exchanges served for the SAME
    /// requesting party get a monotone `seq`/`prev_seq` for that pair
    /// (1 -> None, 2 -> Some(1)); a THIRD exchange for a DIFFERENT
    /// requesting party is an independent counter starting back at 1 -- the
    /// pair, not the node, is what's sequenced.
    #[test]
    fn emit_for_exchange_seals_monotone_seq_per_counterparty_pair() {
        let dir = std::env::temp_dir().join(format!("cap-seq-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");

        let first = state
            .emit_for_exchange(&sample_exchange("party-1"))
            .expect("seal 1");
        assert_eq!(seq_of(&first.capsule), (1, None));

        let second = state
            .emit_for_exchange(&sample_exchange("party-1"))
            .expect("seal 2");
        assert_eq!(seq_of(&second.capsule), (2, Some(1)));

        // A different counterparty is a DIFFERENT pair -- its own counter.
        let other_party = state
            .emit_for_exchange(&sample_exchange("party-2"))
            .expect("seal other party");
        assert_eq!(seq_of(&other_party.capsule), (1, None));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A host-served OBSERVED exchange carries no requester identity, so it
    /// is keyed on the honest "unknown" counterparty bucket -- never
    /// invented -- and still sequences monotonically within that bucket.
    #[test]
    fn observed_host_exchange_seals_monotone_seq_under_the_unknown_counterparty_bucket() {
        let dir = std::env::temp_dir().join(format!("cap-seq-unk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let observed = ObservedHostExchange {
            model: "m",
            exchange_id: Some("e"),
            request_digest: None,
            response_digest: None,
            tool_calls_digest: None,
            reasoning_digest: None,
            usage: None,
            host_provenance: HostProvenance::default(),
            dispatch_path: DispatchPath::RawProxy,
            nonce: None,
            peer_capsule_id: None,
            peer_capsule_id_provenance: None,
            twin_bracket_id: None,
            response_text_digest: None,
        };
        let first = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal 1");
        assert_eq!(seq_of(&first.capsule), (1, None));
        let second = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal 2");
        assert_eq!(seq_of(&second.capsule), (2, Some(1)));
        assert_eq!(
            first.capsule["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"]
                ["serving_provenance"]["requesting_party"],
            "unknown"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// evidencebook#69 finding 1: a host-served exchange a mesh peer asked
    /// for names that peer, the host's authenticated `requested_by_node_id`,
    /// as its requesting party, so the peer is the record's counterparty.
    /// Each requester is its own sequenced pair. A requester-side half never
    /// takes the field (this node asked, it was not asked), and a blank one
    /// stays the honest "unknown".
    #[test]
    fn observed_host_exchange_names_the_host_authenticated_requester() {
        let dir = tempfile::tempdir().unwrap();
        let state = CapsuleState::open(dir.path(), "node-under-test").expect("open state");
        let observed =
            |dispatch_path: DispatchPath, requested_by: Option<&str>| ObservedHostExchange {
                model: "m",
                exchange_id: Some("e"),
                request_digest: None,
                response_digest: None,
                tool_calls_digest: None,
                reasoning_digest: None,
                usage: None,
                host_provenance: HostProvenance {
                    requested_by_node_id: requested_by.map(str::to_string),
                    ..HostProvenance::default()
                },
                dispatch_path,
                nonce: None,
                peer_capsule_id: None,
                peer_capsule_id_provenance: None,
                twin_bracket_id: None,
                response_text_digest: None,
            };
        let seal = |dispatch_path, requested_by| {
            let emitted = state
                .emit_for_observed_host_exchange(&observed(dispatch_path, requested_by))
                .expect("seal");
            let party = poc_block(&emitted.capsule)["serving_provenance"]["requesting_party"]
                .as_str()
                .unwrap()
                .to_string();
            (party, seq_of(&emitted.capsule))
        };

        assert_eq!(
            seal(DispatchPath::RawProxy, Some("peer-a")),
            ("peer-a".to_string(), (1, None))
        );
        assert_eq!(
            seal(DispatchPath::RawProxy, Some(" peer-a ")),
            ("peer-a".to_string(), (2, Some(1)))
        );
        assert_eq!(
            seal(DispatchPath::RawProxy, Some("peer-b")),
            ("peer-b".to_string(), (1, None))
        );
        assert_eq!(seal(DispatchPath::RemoteMesh, Some("peer-a")).0, "unknown");
        assert_eq!(seal(DispatchPath::RawProxy, Some(" ")).0, "unknown");
        assert_eq!(seal(DispatchPath::RawProxy, None).0, "unknown");
    }

    /// RESTART SAFETY: the sequence counter is persisted beside the ledger
    /// (`sequence_counters.json`), not held only in memory -- reopening
    /// `CapsuleState` against the SAME `data_dir` (simulating a process
    /// restart) resumes the pair's counter at 3, never repeating 1.
    #[test]
    fn sequence_counter_survives_a_simulated_restart() {
        let dir = std::env::temp_dir().join(format!("cap-seq-restart-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        {
            let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
            state
                .emit_for_exchange(&sample_exchange("party-1"))
                .expect("seal 1");
            state
                .emit_for_exchange(&sample_exchange("party-1"))
                .expect("seal 2");
        }
        // Fresh CapsuleState over the same data_dir -- the restart.
        let restarted = CapsuleState::open(&dir, "node-under-test").expect("reopen state");
        let third = restarted
            .emit_for_exchange(&sample_exchange("party-1"))
            .expect("seal 3");
        assert_eq!(seq_of(&third.capsule), (3, Some(2)));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The 2026-09-06 role ruling: `x-mesh-poc-v1.role`/`observation_point`
    /// live TOP-LEVEL, siblings of `serving_provenance` -- NOT nested inside
    /// it (that block has its own unrelated `role`-shaped fields on the
    /// Python sidecar path; colliding with them was the whole reason this
    /// ruling called out "deliberately top-level siblings").
    fn poc_block(capsule: &Value) -> &Value {
        &capsule["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"]
    }

    fn observed_with(
        dispatch_path: DispatchPath,
        served_by_node_id: Option<&str>,
    ) -> ObservedHostExchange<'static> {
        ObservedHostExchange {
            model: "m",
            exchange_id: Some("e"),
            request_digest: None,
            response_digest: None,
            tool_calls_digest: None,
            reasoning_digest: None,
            usage: None,
            host_provenance: HostProvenance {
                served_by_node_id: served_by_node_id.map(str::to_string),
                architecture: Some("llama".to_string()),
                ..Default::default()
            },
            dispatch_path,
            nonce: None,
            peer_capsule_id: None,
            peer_capsule_id_provenance: None,
            twin_bracket_id: None,
            response_text_digest: None,
        }
    }

    fn split_case(name: &str) -> Value {
        let cases: Value =
            serde_json::from_str(include_str!("../vectors/split-stage/hop-cases.json")).unwrap();
        cases
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .unwrap()
            .clone()
    }

    /// A released split for `name`, with every stage record sealed under its
    /// own key and handed to the collector.
    fn released_split(name: &str) -> (crate::split_stage::SplitPlan, Vec<Value>) {
        use crate::producer::stage::{seal_stage_record, CoordinatorReceipt, StageBlock};
        use crate::split_stage::{plan_split, SplitCollector, StageEvent};
        let case = split_case(name);
        let receipt = CoordinatorReceipt::from_value(&case["receipt"]).unwrap();
        let collector = SplitCollector::<()>::default();
        collector
            .on_stage_zero(
                StageEvent {
                    block: StageBlock::from_value(&case["own"]).unwrap(),
                    exchange_id: Some("e".into()),
                    coordinator_term: Some(7),
                    topology: Some(
                        receipt
                            .topology
                            .iter()
                            .map(|t| t.assignment.clone().unwrap())
                            .collect(),
                    ),
                },
                0,
            )
            .unwrap();
        collector.hold(Some("e"), ());
        let records: Vec<Value> = case["carried"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let key = KeyPair::generate();
                seal_stage_record(
                    &StageBlock::from_value(&c["block"]).unwrap(),
                    None,
                    None,
                    &key.signing_key,
                )
                .unwrap()
            })
            .collect();
        for r in &records {
            let k = r["model_attestation"]["compute_attestation"]["x-mesh-stage-v1"]["stage_index"]
                .as_u64()
                .unwrap() as usize;
            let sender = receipt.topology[k]
                .assignment
                .as_ref()
                .unwrap()
                .node_id
                .clone();
            collector.on_stage_record(r, &sender, 1);
        }
        let ready = collector.take_due(2, 2_000).pop().unwrap();
        (plan_split(&ready, 2_000).unwrap(), records)
    }

    #[test]
    fn a_split_main_record_chains_after_its_stage_exchange_records_and_cites_them() {
        let dir = std::env::temp_dir().join(format!("cap-split-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let (plan, records) = released_split("relayed_all_agree");
        let main = state
            .emit_for_observed_split_exchange(&observed_with(DispatchPath::RawProxy, None), &plan)
            .expect("seal split");
        let lines: Vec<Value> = std::fs::read_to_string(dir.join("ledger").join("capsules.jsonl"))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(
            lines.len(),
            3,
            "two stage-exchange records, then the main record"
        );
        let (ex1, ex2, last) = (&lines[0], &lines[1], &lines[2]);
        assert_eq!(last["capsule_id"], Value::from(main.capsule_id.as_str()));
        assert_eq!(ex2["chain"]["parent_capsule_id"], ex1["capsule_id"]);
        assert_eq!(last["chain"]["parent_capsule_id"], ex2["capsule_id"]);
        assert_eq!(last["references"][0]["digest"], ex1["capsule_id"]);
        assert_eq!(last["references"][1]["digest"], ex2["capsule_id"]);
        assert_eq!(ex1["references"][0]["digest"], records[0]["capsule_id"]);
        assert_eq!(
            ex1["references"][0]["citation_purpose"],
            Value::from("counterparty_half")
        );
        // The ledger reopens clean over the new record kinds.
        drop(state);
        let (_, report) = crate::producer::index::open_ledger(&dir.join("ledger")).expect("reopen");
        assert_eq!(report.valid_entries, 3);
        // And the requester's check over exactly what was sealed agrees.
        let verdict =
            crate::producer::stage_verify::verify_split_records(&main.capsule, &plan.carried)
                .unwrap();
        assert!(verdict.handoffs_agree, "{verdict:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The coordinator's push body for a split: the main record, covered by
    /// a real checkpoint, with the stage records it cites.
    fn real_split_bundle(dir: &Path) -> Value {
        use crate::producer::anchor::AnchorClient;
        use crate::producer::checkpoint::{CheckpointCadenceConfig, CheckpointState};
        let state = CapsuleState::open(dir, "rust-node").expect("open state");
        let (plan, _) = released_split("relayed_all_agree");
        let main = state
            .emit_for_observed_split_exchange(&observed_with(DispatchPath::RawProxy, None), &plan)
            .expect("seal split");
        let (mut checkpoints, _) = CheckpointState::load(
            &dir.join("ledger"),
            "rust-node",
            CheckpointCadenceConfig::default(),
        )
        .expect("load checkpoint state");
        let coverage = checkpoints
            .checkpoint_covering(
                &main.capsule_id,
                state.signing_key(),
                &AnchorClient::new("http://127.0.0.1:1"),
            )
            .expect("cover the main record");
        crate::record_push_bridge::split_bundle_body(&main.capsule, &coverage, &plan.carried)
    }

    #[test]
    fn a_split_bundle_carries_exactly_the_stage_records_its_receipt_names() {
        let dir = std::env::temp_dir().join(format!("cap-split-bundle-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let body = real_split_bundle(&dir);
        let carried = body[crate::record_push_bridge::SPLIT_STAGE_RECORDS]
            .as_array()
            .unwrap();
        let receipt = &body["capsule"]["model_attestation"]["compute_attestation"]
            ["x-mesh-coordinator-receipt-v1"];
        let named: Vec<&Value> = receipt["stages"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|s| s.get("bundle_ref"))
            .map(|r| &r["digest"])
            .collect();
        let carried_ids: Vec<&Value> = carried.iter().map(|r| &r["capsule_id"]).collect();
        assert_eq!(named, carried_ids);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Writes `vectors/split-stage/rust-split-bundle.json`, which the
    /// Python reference's split test verifies. Run to regenerate:
    ///   cargo test writes_the_split_bundle_fixture -- --ignored
    #[test]
    #[ignore = "regenerates a committed fixture"]
    fn writes_the_split_bundle_fixture() {
        let dir = std::env::temp_dir().join(format!("split-bundle-fixture-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let body = real_split_bundle(&dir);
        let out = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("vectors/split-stage/rust-split-bundle.json");
        std::fs::write(&out, serde_json::to_string_pretty(&body).unwrap() + "\n").unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_split_is_refused_on_a_record_that_is_not_served_and_nothing_is_appended() {
        let dir = std::env::temp_dir().join(format!("cap-split-refuse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let (plan, _) = released_split("relayed_all_agree");
        let refused = state.emit_for_observed_split_exchange(
            &observed_with(DispatchPath::RemoteMesh, None),
            &plan,
        );
        assert!(refused.is_err());
        assert!(
            state.chain_head().is_none(),
            "no stage-exchange record without its main record"
        );
        assert_eq!(state.observed_not_sealed(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stage_record_is_sealed_through_the_single_writer_path() {
        let dir = std::env::temp_dir().join(format!("cap-stage-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let case = split_case("relayed_all_agree");
        let block =
            crate::producer::stage::StageBlock::from_value(&case["carried"][0]["block"]).unwrap();
        let emitted = state.emit_stage_record(&block).expect("seal stage record");
        assert_eq!(
            state.chain_head().as_deref(),
            Some(emitted.capsule_id.as_str())
        );
        assert_eq!(
            emitted.capsule["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"]["role"],
            Value::from("stage")
        );
        assert!(
            emitted.capsule.get("signature").is_some(),
            "enveloped like every pushed record"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// THE BUG THIS RULING CLOSES: a `RemoteMesh`-routed exchange (this node
    /// is the REQUESTER) must never be silently labeled `served` just
    /// because no `served_by_node_id` was reported (bare upstream, pre-#1668
    /// enrichment). `dispatch_path` alone is authoritative here.
    #[test]
    fn remote_mesh_with_no_enrichment_seals_role_requested_not_served() {
        let dir = std::env::temp_dir().join(format!("cap-role-rm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let observed = observed_with(DispatchPath::RemoteMesh, None);
        let emitted = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal");
        let poc = poc_block(&emitted.capsule);
        assert_eq!(poc["role"], "requested");
        assert_eq!(poc["observation_point"], "client_egress");
        let sp = &poc["serving_provenance"];
        assert_eq!(sp["dispatch_path"], "remote_mesh");
        // No enrichment reported who served it -- honest "unknown", never a
        // fabricated claim that THIS node (the requester) served itself.
        assert_eq!(sp["served_by_node_id"], "unknown");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `TypedFrontend`/`RawProxy` both mean this node served the exchange.
    /// With no fork enrichment AND nothing learned yet, `served_by_node_id`
    /// seals the honest "unknown" -- NEVER `PLUGIN_ID`/`node_id` (the
    /// 2026-09-06 domain fix: a plugin-type label is not a mesh node id, so
    /// it must never be substituted as if it were one).
    #[test]
    fn typed_frontend_and_raw_proxy_seal_role_served() {
        for dispatch_path in [DispatchPath::TypedFrontend, DispatchPath::RawProxy] {
            let dir = std::env::temp_dir().join(format!(
                "cap-role-served-{:?}-{}",
                dispatch_path,
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
            let observed = observed_with(dispatch_path, None);
            let emitted = state
                .emit_for_observed_host_exchange(&observed)
                .expect("seal");
            let poc = poc_block(&emitted.capsule);
            assert_eq!(poc["role"], "served");
            assert!(poc["observation_point"].is_null());
            assert_eq!(poc["serving_provenance"]["served_by_node_id"], "unknown");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// An unrecognized `dispatch_path` (a host newer than this mirror) is
    /// labeled `role: "unknown"` -- NEVER silently defaulted to `"served"`.
    /// This is the actual bug the ruling closes: a missing/unrecognized
    /// signal must never become a claim, regardless of what
    /// `served_by_node_id` says.
    #[test]
    fn unknown_dispatch_path_seals_role_unknown_never_served() {
        let dir = std::env::temp_dir().join(format!("cap-role-unk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        // Even a served_by_node_id that would otherwise agree with "served"
        // must not upgrade an unrecognized dispatch_path out of "unknown".
        let observed = observed_with(DispatchPath::Unknown, Some("node-under-test"));
        let emitted = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal");
        let poc = poc_block(&emitted.capsule);
        assert_eq!(poc["role"], "unknown");
        assert!(poc["observation_point"].is_null());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// CONSISTENCY CHECK, agreeing case: a `RemoteMesh` event whose
    /// fork-enrichment `served_by_node_id` names a DIFFERENT node agrees
    /// with the dispatch_path-derived "requested" -- role stays "requested",
    /// not "conflict". Also: SKIP-PATH -- this node has never learned a
    /// self id (no prior locally-served event), so the check has nothing to
    /// compare against regardless; the dispatch-derived role is sealed
    /// unchanged either way.
    #[test]
    fn remote_mesh_with_agreeing_served_by_node_id_stays_requested() {
        let dir = std::env::temp_dir().join(format!("cap-role-rm-agree-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let observed = observed_with(DispatchPath::RemoteMesh, Some("peer-node-3"));
        let emitted = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal");
        let poc = poc_block(&emitted.capsule);
        assert_eq!(poc["role"], "requested");
        assert_eq!(
            poc["serving_provenance"]["served_by_node_id"],
            "peer-node-3"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// SKIP-PATH, explicit: BEFORE any self id has been learned, a
    /// `RemoteMesh` event whose `served_by_node_id` happens to equal this
    /// node's OWN `node_id` (the old, buggy fallback value) must NOT be
    /// treated as a conflict -- `node_id` is a plugin-type label, not this
    /// node's mesh identity, and the check is skipped until a real self id
    /// is learned. This is the regression guard for the domain-confusion
    /// bug: comparing against `PLUGIN_ID`/`node_id` directly (as the old
    /// code did) would have wrongly sealed `role: "conflict"` here.
    #[test]
    fn remote_mesh_before_self_id_learned_never_conflicts_even_if_it_names_node_id() {
        let dir = std::env::temp_dir().join(format!("cap-role-rm-preskip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let observed = observed_with(DispatchPath::RemoteMesh, Some("node-under-test"));
        let emitted = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal");
        let poc = poc_block(&emitted.capsule);
        assert_eq!(
            poc["role"], "requested",
            "no self id learned yet -- the check must be skipped, never compared \
             against the plugin-type node_id label"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// LEARN: the FIRST locally-served (`TypedFrontend`/`RawProxy`) event
    /// that carries the fork's `served_by_node_id` enrichment teaches this
    /// node its own mesh identity -- THE regression guard for the domain
    /// bug: `served_by_node_id` here is a real mesh-node-id-shaped value
    /// (hash-like, never equal to `PLUGIN_ID`/`node_id`), and role must seal
    /// "served", never "conflict" (mutant: reintroducing a naive `self.node_id`
    /// comparison here would flip this red).
    #[test]
    fn typed_frontend_learns_self_node_id_from_first_observation_and_seals_served() {
        let dir = std::env::temp_dir().join(format!("cap-role-tf-learn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let real_mesh_node_id = "fa28d0dfe5f0b2c4a8f0fcb15838075e4e5f0b32d6dd5df029588e8992fad5ac";
        let observed = observed_with(DispatchPath::TypedFrontend, Some(real_mesh_node_id));
        let emitted = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal");
        let poc = poc_block(&emitted.capsule);
        assert_eq!(poc["role"], "served");
        assert_eq!(
            poc["serving_provenance"]["served_by_node_id"],
            real_mesh_node_id
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ROTATE: a SECOND locally-served event naming a DIFFERENT
    /// `served_by_node_id` than the one already learned replaces the
    /// cached self id -- a rotation, never a "conflict" (per the standing
    /// ruling: locally-served events are the source of truth for this
    /// node's own identity and can never conflict with themselves).
    #[test]
    fn typed_frontend_with_new_served_by_node_id_rotates_learned_self_not_conflict() {
        let dir = std::env::temp_dir().join(format!("cap-role-tf-rotate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");

        let first = state
            .emit_for_observed_host_exchange(&observed_with(
                DispatchPath::TypedFrontend,
                Some("node-id-a"),
            ))
            .expect("seal 1");
        assert_eq!(poc_block(&first.capsule)["role"], "served");

        let second = state
            .emit_for_observed_host_exchange(&observed_with(
                DispatchPath::TypedFrontend,
                Some("node-id-b"),
            ))
            .expect("seal 2");
        let poc = poc_block(&second.capsule);
        assert_eq!(poc["role"], "served", "a rotated self id is NOT a conflict");
        assert_eq!(poc["serving_provenance"]["served_by_node_id"], "node-id-b");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// CONSISTENCY CHECK, genuine conflict: ONLY reachable after a self id
    /// has been learned from a locally-served event. A subsequent
    /// `RemoteMesh` event (this node routed it out, "requested") whose
    /// fork-enrichment `served_by_node_id` names THIS node's own LEARNED id
    /// contradicts that -- seals `role: "conflict"`. Mutant guard: disabling
    /// the consistency check (always skip) would flip this red, since it
    /// would seal "requested" instead.
    #[test]
    fn remote_mesh_with_learned_self_node_id_seals_conflict() {
        let dir = std::env::temp_dir().join(format!("cap-role-rm-conflict-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");

        // First, learn this node's real mesh identity from a locally-served
        // event -- the same way `typed_frontend_learns_...` does.
        state
            .emit_for_observed_host_exchange(&observed_with(
                DispatchPath::TypedFrontend,
                Some("node-under-test-real-id"),
            ))
            .expect("seal learning event");

        // Now a RemoteMesh event claims served_by_node_id == the id we just
        // learned -- a real anomaly: we dispatched this exchange to a peer,
        // yet the enrichment says we ourselves served it.
        let observed = observed_with(DispatchPath::RemoteMesh, Some("node-under-test-real-id"));
        let emitted = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal");
        let poc = poc_block(&emitted.capsule);
        assert_eq!(poc["role"], "conflict");
        // Complementary vantage provenance is independent of the conflict --
        // this node still observed the exchange at its client-egress vantage.
        assert_eq!(poc["observation_point"], "client_egress");
        let sp = &poc["serving_provenance"];
        assert_eq!(sp["dispatch_path"], "remote_mesh");
        assert_eq!(sp["served_by_node_id"], "node-under-test-real-id");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The plugin's own directly-served `/v1` exchanges have no host
    /// envelope at all, so no `dispatch_path` signal exists on that path --
    /// but they are unambiguously "served" (this plugin admitted and served
    /// them itself), so `role` is sealed accordingly, never left `null`.
    #[test]
    fn plugin_served_exchange_seals_role_served_with_no_dispatch_path() {
        let dir = std::env::temp_dir().join(format!("cap-role-plugin-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let emitted = state
            .emit_for_exchange(&sample_exchange("party-1"))
            .expect("seal");
        let poc = poc_block(&emitted.capsule);
        assert_eq!(poc["role"], "served");
        assert!(poc["observation_point"].is_null());
        assert!(poc["serving_provenance"]["dispatch_path"].is_null());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// MUTANT GUARD (kills the restored-fallback mutant): the plugin's own
    /// directly-served `/v1` path (`emit_for_exchange`, no host envelope at
    /// all) seals the honest "unknown" `served_by_node_id` when nothing has
    /// been learned yet -- NEVER `PLUGIN_ID`/`node_id`. Reintroducing
    /// `unwrap_or_else(|| self.node_id.clone())` here would flip this red.
    #[test]
    fn plugin_served_exchange_seals_unknown_served_by_node_id_before_learning() {
        let dir = std::env::temp_dir().join(format!("cap-role-plugin-unk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
        let emitted = state
            .emit_for_exchange(&sample_exchange("party-1"))
            .expect("seal");
        let poc = poc_block(&emitted.capsule);
        assert_eq!(poc["serving_provenance"]["served_by_node_id"], "unknown");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Once a self id HAS been learned (from an earlier locally-served
    /// observed-host exchange on the SAME `CapsuleState`), the plugin's own
    /// directly-served path uses that real learned identity instead of
    /// "unknown" -- an honest fact this node genuinely knows, never a
    /// fabricated one.
    #[test]
    fn plugin_served_exchange_uses_learned_self_node_id_once_known() {
        let dir =
            std::env::temp_dir().join(format!("cap-role-plugin-learned-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");

        state
            .emit_for_observed_host_exchange(&observed_with(
                DispatchPath::TypedFrontend,
                Some("learned-real-id"),
            ))
            .expect("seal learning event");

        let emitted = state
            .emit_for_exchange(&sample_exchange("party-1"))
            .expect("seal");
        let poc = poc_block(&emitted.capsule);
        assert_eq!(
            poc["serving_provenance"]["served_by_node_id"],
            "learned-real-id"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// RESTART SAFETY: the learned self id is persisted beside the ledger
    /// (`learned_self_node_id.json`), not held only in memory -- reopening
    /// `CapsuleState` against the SAME `data_dir` (simulating a process
    /// restart) still has it learned, and a subsequent `RemoteMesh` event
    /// naming that same id still seals a genuine conflict.
    #[test]
    fn learned_self_node_id_survives_a_simulated_restart() {
        let dir = std::env::temp_dir().join(format!("cap-role-restart-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        {
            let state = CapsuleState::open(&dir, "node-under-test").expect("open state");
            state
                .emit_for_observed_host_exchange(&observed_with(
                    DispatchPath::TypedFrontend,
                    Some("persisted-real-id"),
                ))
                .expect("seal learning event");
        }
        // Fresh CapsuleState over the same data_dir -- the restart.
        let restarted = CapsuleState::open(&dir, "node-under-test").expect("reopen state");
        let emitted = restarted
            .emit_for_observed_host_exchange(&observed_with(
                DispatchPath::RemoteMesh,
                Some("persisted-real-id"),
            ))
            .expect("seal after restart");
        assert_eq!(poc_block(&emitted.capsule)["role"], "conflict");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Requester-side sealing on a proxied exchange: the host-forwarded
    /// terminal-event nonce becomes `client_nonce`/`client_nonce_source` on a
    /// `RemoteMesh` (requester-side) observed capsule -- THE JOIN KEY a
    /// proxied exchange's two halves share. Absent nonce keeps the
    /// pre-existing honest default, never a fabricated value.
    #[test]
    fn requester_side_capsule_carries_the_forwarded_nonce_as_join_key() {
        let dir = std::env::temp_dir().join(format!("cap-req-nonce-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "router-node").expect("open state");

        let mut observed = observed_with(DispatchPath::RemoteMesh, Some("peer-node"));
        observed.nonce = Some("shared-nonce-42");
        let emitted = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal");
        let poc = poc_block(&emitted.capsule);
        assert_eq!(poc["client_nonce"], "shared-nonce-42");
        assert_eq!(poc["client_nonce_source"], "host_forwarded_nonce");

        let no_nonce = observed_with(DispatchPath::RemoteMesh, Some("peer-node"));
        let emitted = state
            .emit_for_observed_host_exchange(&no_nonce)
            .expect("seal");
        let poc = poc_block(&emitted.capsule);
        assert_eq!(poc["client_nonce"], "host-served-no-nonce");
        assert_eq!(poc["client_nonce_source"], "host_served_observed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The two-sided ledger's peer fetch: the requester-side
    /// capsule carries the PEER's self-asserted capsule id -- the lookup key
    /// a ledger fetch will dereference to populate the two-sided
    /// ledger's "theirs" column. Forwarded verbatim, labeled as self-attested
    /// (`peer_asserted`), never promoted to a verified claim.
    #[test]
    fn remote_mesh_seals_the_peer_asserted_capsule_id_as_the_fetch_join_key() {
        let dir = std::env::temp_dir().join(format!("cap-peer-capid-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "router-node").expect("open state");

        let mut observed = observed_with(DispatchPath::RemoteMesh, Some("peer-node"));
        observed.peer_capsule_id = Some("peer-cap-987");
        observed.peer_capsule_id_provenance = Some("peer_asserted");
        let emitted = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal");
        let sp = &poc_block(&emitted.capsule)["serving_provenance"];
        assert_eq!(sp["peer_capsule_id"], "peer-cap-987");
        assert_eq!(sp["peer_capsule_id_provenance"], "peer_asserted");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Negative half of the same check (R4): when the RemoteMesh event
    /// carries no peer-asserted capsule id (the router observed none),
    /// `peer_capsule_id` stays `null` -- never a fabricated value standing
    /// in for an absent fact.
    #[test]
    fn remote_mesh_without_a_peer_capsule_id_seals_it_null_never_fabricated() {
        let dir =
            std::env::temp_dir().join(format!("cap-peer-capid-absent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "router-node").expect("open state");

        let observed = observed_with(DispatchPath::RemoteMesh, Some("peer-node"));
        let emitted = state
            .emit_for_observed_host_exchange(&observed)
            .expect("seal");
        let sp = &poc_block(&emitted.capsule)["serving_provenance"];
        assert!(sp["peer_capsule_id"].is_null());
        assert!(sp["peer_capsule_id_provenance"].is_null());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The plugin's own `/v1` handler path never receives a host envelope at
    /// all -- there is no peer half to name on it, so `emit_for_exchange`
    /// must always seal `peer_capsule_id: None`, never invent one. This is
    /// the sibling guarantee to `main.rs::seal_observed_host_exchange`'s
    /// `CapsuleIdProvenance::PeerAsserted` match (which stops a locally-
    /// served envelope's `SelfMinted` marker from being mislabeled as a
    /// peer's claim on the OBSERVE path) -- this test pins the SERVE path.
    #[test]
    fn plugin_served_exchange_never_seals_a_peer_capsule_id() {
        let dir =
            std::env::temp_dir().join(format!("cap-no-peer-on-served-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "solo-node").expect("open state");

        let emitted = state
            .emit_for_exchange(&ExchangeRecord {
                model: "m",
                client_nonce: None,
                request_bytes: br#"{"model":"m","messages":[]}"#,
                response_bytes: br#"{"id":"x","choices":[]}"#,
                latency_ms: 1.0,
                exchange_id: Some("e"),
                requesting_party: None,
                host_provenance: None,
            })
            .expect("seal");
        let sp = &poc_block(&emitted.capsule)["serving_provenance"];
        assert!(sp["peer_capsule_id"].is_null());
        assert!(sp["peer_capsule_id_provenance"].is_null());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// THE ACCEPTANCE TEST for requester-side sealing on a proxied exchange: a
    /// proxied exchange (peer serves it, router routes it) seals TWO
    /// independently offline-verifiable capsules -- `served` on the peer,
    /// `requested` on the router -- joined by the SHARED nonce, never by
    /// `exchange_id` (minted per-node, deliberately different on each side
    /// here to prove the join does not depend on it).
    #[test]
    fn proxied_exchange_seals_two_joinable_verified_capsules_by_nonce() {
        let peer_dir = std::env::temp_dir().join(format!("cap-join-peer-{}", std::process::id()));
        let router_dir =
            std::env::temp_dir().join(format!("cap-join-router-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&peer_dir);
        let _ = std::fs::remove_dir_all(&router_dir);

        let shared_nonce = "shared-nonce-join-key";

        // The PEER served the exchange directly through its own `/v1` handler
        // -- the pre-existing, unchanged "served" path.
        let peer_state = CapsuleState::open(&peer_dir, "peer-node").expect("open peer state");
        let peer_emitted = peer_state
            .emit_for_exchange(&ExchangeRecord {
                model: "m",
                client_nonce: Some(shared_nonce),
                request_bytes: br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
                response_bytes:
                    br#"{"id":"x","choices":[{"message":{"role":"assistant","content":"hello"}}]}"#,
                latency_ms: 5.0,
                exchange_id: Some("peer-minted-exchange-id"),
                requesting_party: Some("router-node"),
                host_provenance: None,
            })
            .expect("peer seals its served half");
        let peer_poc = poc_block(&peer_emitted.capsule);
        assert_eq!(peer_poc["role"], "served");
        assert_eq!(peer_poc["client_nonce"], shared_nonce);

        // The ROUTER observed its own `RemoteMesh` terminal event for the
        // SAME exchange -- a DIFFERENT, per-node `exchange_id`, but the SAME
        // client-forwarded nonce -- and seals its own requester-role half.
        let router_state =
            CapsuleState::open(&router_dir, "router-node").expect("open router state");
        let router_observed = ObservedHostExchange {
            model: "m",
            exchange_id: Some("router-minted-exchange-id"),
            request_digest: None,
            response_digest: None,
            tool_calls_digest: None,
            reasoning_digest: None,
            usage: None,
            host_provenance: HostProvenance {
                served_by_node_id: Some("peer-node".to_string()),
                ..Default::default()
            },
            dispatch_path: DispatchPath::RemoteMesh,
            nonce: Some(shared_nonce),
            peer_capsule_id: None,
            peer_capsule_id_provenance: None,
            twin_bracket_id: None,
            response_text_digest: None,
        };
        let router_emitted = router_state
            .emit_for_observed_host_exchange(&router_observed)
            .expect("router seals its requester half");
        let router_poc = poc_block(&router_emitted.capsule);
        assert_eq!(router_poc["role"], "requested");
        assert_eq!(router_poc["client_nonce"], shared_nonce);
        assert_eq!(
            router_poc["serving_provenance"]["served_by_node_id"],
            "peer-node"
        );

        // THE JOIN: both halves carry the identical nonce...
        assert_eq!(peer_poc["client_nonce"], router_poc["client_nonce"]);
        // ...while their exchange_ids deliberately differ (per-node minted) --
        // proving the join does not, and must not, depend on exchange_id.
        assert_ne!(
            peer_poc["serving_provenance"]["exchange_id"],
            router_poc["serving_provenance"]["exchange_id"]
        );

        // BOTH halves independently verify offline -- `verify().ok()`.
        let peer_vk = peer_state.keys.verifying_key();
        let (peer_ledger, _) = crate::producer::index::open_ledger(&peer_dir.join("ledger"))
            .expect("reopen peer ledger");
        let peer_entry = peer_ledger
            .lookup(&peer_emitted.capsule_id)
            .expect("lookup ok")
            .expect("peer capsule in ledger");
        let peer_report = crate::producer::verify::verify_offline(
            &peer_entry.capsule,
            &peer_entry.signed_statement,
            &peer_vk,
            None,
        );
        assert!(
            peer_report.ok(),
            "peer served-role capsule must verify offline: {:?}",
            peer_report.findings
        );

        let router_vk = router_state.keys.verifying_key();
        let (router_ledger, _) = crate::producer::index::open_ledger(&router_dir.join("ledger"))
            .expect("reopen router ledger");
        let router_entry = router_ledger
            .lookup(&router_emitted.capsule_id)
            .expect("lookup ok")
            .expect("router capsule in ledger");
        let router_report = crate::producer::verify::verify_offline(
            &router_entry.capsule,
            &router_entry.signed_statement,
            &router_vk,
            None,
        );
        assert!(
            router_report.ok(),
            "router requester-role capsule must verify offline: {:?}",
            router_report.findings
        );

        let _ = std::fs::remove_dir_all(&peer_dir);
        let _ = std::fs::remove_dir_all(&router_dir);
    }

    /// REGRESSION GUARD: a local-served (non-proxy) exchange still seals
    /// exactly ONE capsule -- the requester-side predicate must never also
    /// fire for a `TypedFrontend`/`RawProxy` dispatch path, and the plugin's
    /// own `/v1` handler (`emit_for_exchange`) is the only thing that seals
    /// it. This mirrors `is_sealable_requester_side`'s own mutual-exclusion
    /// unit test in `lifecycle_channel.rs`, at the level that matters here:
    /// the actual capsule count for one exchange.
    #[test]
    fn local_served_exchange_still_seals_exactly_one_capsule() {
        let dir = std::env::temp_dir().join(format!("cap-local-one-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = CapsuleState::open(&dir, "node-under-test").expect("open state");

        // The plugin's own handler seals the one capsule for a locally-served
        // exchange.
        state
            .emit_for_exchange(&sample_exchange("party-1"))
            .expect("seal the one served capsule");

        // The corresponding `TypedFrontend`/`RawProxy` terminal event observed
        // on the mesh channel must NOT also be sealable as the requester's
        // half (that predicate requires `RemoteMesh`) -- so a caller wiring
        // both checks together (as `main.rs` does) never double-seals.
        for dispatch_path in [DispatchPath::TypedFrontend, DispatchPath::RawProxy] {
            let wire = format!(
                r#"{{"dispatch_path":"{}","phase":"terminal","model":"m","status":200}}"#,
                crate::lifecycle_channel::dispatch_path_wire_value(&dispatch_path)
            );
            let envelope: crate::lifecycle_channel::OpenAiExchangeEnvelope =
                serde_json::from_str(&wire).expect("parse");
            assert!(
                !crate::lifecycle_channel::ObservedLifecycleEvents::is_sealable_requester_side(
                    &envelope
                ),
                "a locally-served dispatch path must never also seal a requester-role capsule"
            );
        }

        let (_ledger, report) =
            crate::producer::index::open_ledger(&dir.join("ledger")).expect("reopen ledger");
        assert_eq!(
            report.valid_entries, 1,
            "a local-served exchange must seal exactly one capsule, never two"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod settlement_tests {
    use super::*;
    use crate::settlement_channel::{host_shaped_event, parse_and_check, Phase};

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cap-settle-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn exchange(exchange_id: &str) -> ExchangeRecord<'_> {
        ExchangeRecord {
            model: "m",
            client_nonce: Some("n"),
            request_bytes: br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
            response_bytes:
                br#"{"id":"x","choices":[{"message":{"role":"assistant","content":"y"}}]}"#,
            latency_ms: 1.0,
            exchange_id: Some(exchange_id),
            requesting_party: Some("party-1"),
            host_provenance: None,
        }
    }

    fn checked(body: &Value) -> crate::settlement_channel::PaymentLifecycleEvent {
        parse_and_check(&serde_json::to_vec(body).unwrap()).expect("host-shaped event checks")
    }

    /// A checked event seals one record onto the local head with every value
    /// copied verbatim (the amount included), under
    /// `compute_attestation["x-mesh-settlement-v1"]`.
    #[test]
    fn settlement_record_seals_verbatim_and_chains_onto_the_head() {
        let dir = temp_dir("happy");
        let state = CapsuleState::open(&dir, "node-under-test").unwrap();
        let local = state.emit_for_exchange(&exchange("ex-paid-1")).unwrap();

        let hash = "bb".repeat(32);
        let body = host_shaped_event(
            "ex-paid-1",
            Phase::OutputSettlementObserved,
            Some(1),
            Some(&hash),
            123457,
        );
        let event = checked(&body);
        let sealed = state
            .emit_settlement_record(&event.observation())
            .unwrap()
            .expect("first observation seals");

        assert_eq!(
            sealed.capsule["chain"]["parent_capsule_id"].as_str(),
            Some(local.capsule_id.as_str())
        );
        assert_eq!(sealed.capsule["chain"]["relation"], json!("follows"));
        assert!(sealed.capsule.get("references").is_none());
        let obs =
            &sealed.capsule["model_attestation"]["compute_attestation"]["x-mesh-settlement-v1"];
        assert_eq!(
            obs,
            &json!({
                "v": 1,
                "observed_by": "payer",
                "channel": "payment.lifecycle.v1",
                "exchange_id": "ex-paid-1",
                "event_ref": body["event_ref"],
                "terms_digest": body["terms_digest"],
                "phase": "output_settlement_observed",
                "source": "wallet_reported",
                "settlement": "terminal",
                "segment": 1,
                "payment_hash": hash,
                "amount_msat": 123457,
            })
        );
        assert_eq!(
            state.chain_head().as_deref(),
            Some(sealed.capsule_id.as_str())
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// The same event twice seals one record; still one after a reopen (the
    /// dedup set is rebuilt from the ledger).
    #[test]
    fn settlement_record_dedups_by_event_ref_across_reopen() {
        let dir = temp_dir("dedup");
        let state = CapsuleState::open(&dir, "node-under-test").unwrap();
        let body = host_shaped_event("ex-paid-1", Phase::TermsAccepted, None, None, 5000);
        let event = checked(&body);
        let first = state
            .emit_settlement_record(&event.observation())
            .unwrap()
            .expect("first observation seals");
        assert!(state
            .emit_settlement_record(&event.observation())
            .unwrap()
            .is_none());
        assert_eq!(
            state.chain_head().as_deref(),
            Some(first.capsule_id.as_str())
        );

        drop(state);
        let reopened = CapsuleState::open(&dir, "node-under-test").unwrap();
        assert!(reopened
            .emit_settlement_record(&event.observation())
            .unwrap()
            .is_none());
        assert_eq!(
            reopened.chain_head().as_deref(),
            Some(first.capsule_id.as_str())
        );
        let lines = fs::read_to_string(dir.join("ledger").join("capsules.jsonl")).unwrap();
        assert_eq!(lines.lines().count(), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Writes a ledger fixture for host-side reader tests: two exchange
    /// records ("ex-paid-1", "ex-free-1") and the six lifecycle settlement
    /// records for "ex-paid-1". Output dir from `SETTLEMENT_FIXTURE_OUT`.
    #[test]
    #[ignore]
    fn write_settlement_ledger_fixture() {
        let out = PathBuf::from(
            std::env::var("SETTLEMENT_FIXTURE_OUT").expect("set SETTLEMENT_FIXTURE_OUT"),
        );
        let _ = fs::remove_dir_all(&out);
        let state = CapsuleState::open(&out, "fixture-node").unwrap();
        state.emit_for_exchange(&exchange("ex-paid-1")).unwrap();
        state.emit_for_exchange(&exchange("ex-free-1")).unwrap();
        let (input_hash, output_hash) = ("aa".repeat(32), "bb".repeat(32));
        for (phase, segment, hash, amount) in [
            (Phase::TermsAccepted, None, None, 5000),
            (Phase::InputInvoiceIssued, Some(0), Some(&input_hash), 1200),
            (
                Phase::InputSettlementObserved,
                Some(0),
                Some(&input_hash),
                1200,
            ),
            (
                Phase::OutputInvoiceIssued,
                Some(1),
                Some(&output_hash),
                2300,
            ),
            (
                Phase::OutputSettlementObserved,
                Some(1),
                Some(&output_hash),
                2300,
            ),
            (Phase::FinalAccounted, None, None, 3502),
        ] {
            let body = host_shaped_event(
                "ex-paid-1",
                phase,
                segment,
                hash.map(String::as_str),
                amount,
            );
            let event = checked(&body);
            let sealed = state
                .emit_settlement_record(&event.observation())
                .unwrap()
                .expect("fresh event seals");
            println!("{}", serde_json::to_string(&sealed.capsule).unwrap());
        }
    }
}
