//! Ask #5's carrier, minus ask #5's own build: mesh-llm's *existing*
//! `OpenMeshStreamRequest`/`connect_stream` plugin-mesh-stream mechanism
//! (`crates/mesh-llm-host-runtime/src/mesh/plugin_streams.rs`, verified on
//! `main` at `cae3620` and reconfirmed against a fresh `mesh-llm` checkout for
//! this task) already lets a plugin open a bidirectional QUIC stream to the
//! *same plugin* on a target peer, gated on both ends declaring the channel
//! in their manifest (`plugin_event_channel_declared`). This module declares
//! `evidence-request/1` and bridges bytes on both ends -- mesh never parses
//! either side. Zero upstream (`mesh-llm`) code.
//!
//! **Responder role** (`on_open_stream`): the host already gated this call on
//! the channel being declared before `connect_stream` ever reaches us. This
//! plugin also declares `ledger-fetch/1` and `record-push/1`; `main.rs`
//! dispatches an inbound `OpenStreamRequest` to whichever module owns it
//! before `handle_open_stream` below is called, so it only ever receives a
//! mesh-inbound evidence request. We bind a local listener, hand its
//! endpoint back, and once the host bridges the remote bytes to it, read the
//! whole request (bounded in bytes and in time) and answer it **in-process**
//! (`evidence_answer`, over the `evidence-request` crate): an
//! artifact or a signed refusal. At most [`MAX_IN_FLIGHT_ANSWERS`] requests
//! are answered at once; one more is declined with a signed
//! `policy_declined` before the ledger is read. Every answer is logged to
//! the "asked of you" log (`received_log`).
//!
//! An `adjudicate` request (a requester asking this node to referee two
//! twins) is not a -00 subject: it is plugin-internal, carried on the same
//! stream, and answered by the referee's door (`crate::referee::service`)
//! with a signed verdict or a signed refusal. A verdict's
//! `adjudication_issued` record is sealed on this node's chain before the
//! reply leaves; if that seal fails, nothing is sent.
//!
//! **Requester role**: the streamed-HTTP-binding path mesh-llm ships
//! (`handle_streamed_http_binding`) forwards a binding's *static* manifest
//! path/method into `OpenStreamRequest.metadata_json` before any request body
//! exists -- it cannot carry a caller-chosen `target_peer_id`, so it cannot
//! drive an outbound `OpenMeshStreamRequest` on this plugin's behalf. The
//! mechanism that CAN -- `mesh_llm_plugin`'s tool/operation router
//! (`ToolRouter`/`invoke_operation`, reachable locally over
//! `POST /api/plugins/capsules/tools/mesh_evidence_request`) -- hands
//! its handler the full JSON arguments AND a `&mut PluginContext` in the same
//! call, so a local client drives this tool;
//! the handler opens the mesh stream, writes the E14 request bytes, and
//! returns whatever the peer's own responder wrote back (Artifact or signed
//! Refusal, untouched) as the tool's JSON result. A peer that never declared
//! the channel drops the stream at the host with no reply; bounded by
//! `REQUESTER_IDLE_TIMEOUT` so that reads a clean failure, never a hang.
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use mesh_llm_plugin::proto::{OpenMeshStreamRequest, OpenStreamRequest, OpenStreamResponse};
use mesh_llm_plugin::{bind_side_stream, LocalStream, PluginContext, PluginError, PluginResult};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// The mesh channel this plugin declares on both ends -- gates both the
/// outbound `open_outbound_plugin_mesh_stream` call (checked against THIS
/// plugin's own manifest) and the inbound `handle_plugin_mesh_stream` call
/// (checked against the responder's manifest).
pub const EVIDENCE_REQUEST_CHANNEL: &str = "evidence-request/1";

/// The tool name a local client calls
/// (`POST /api/plugins/<plugin_id>/tools/mesh_evidence_request`).
pub const EVIDENCE_REQUEST_OPERATION: &str = "mesh_evidence_request";

/// How long the RESPONDER waits at each step of an already-open mesh stream:
/// the requester's connection, its request bytes, and writing the answer.
/// Bounds a silent peer; mesh-llm's own `idle_timeout_ms` on the
/// `OpenMeshStreamRequest` bounds the mesh stream separately, host-side.
/// Overridable so a test proving the bound is actually enforced does not
/// have to wait out the production default.
fn responder_http_timeout() -> Duration {
    env_millis("CAPSULES_EVIDENCE_HTTP_TIMEOUT_MS", 10_000)
}

/// How long the REQUESTER waits for a response once the mesh stream is open
/// -- covers exactly the "peer never declared the channel, host drops the
/// stream with no reply" case the acceptance check requires read as a clean
/// failure, never a hang.
fn requester_idle_timeout_ms() -> u64 {
    env_millis("CAPSULES_MESH_REQUEST_TIMEOUT_MS", 8_000).as_millis() as u64
}

/// Shared with `ledger_fetch_bridge`, the other plugin-mesh-stream carrier --
/// same env-var-with-default shape, no reason for two copies.
pub(crate) fn env_millis(var: &str, default_ms: u64) -> Duration {
    Duration::from_millis(
        crate::settings::var(var)
            .ok()
            .and_then(|raw| raw.parse().ok())
            .unwrap_or(default_ms),
    )
}

static STREAM_NONCE: AtomicU64 = AtomicU64::new(1);

/// Shared with `ledger_fetch_bridge` -- one nonce space for every
/// plugin-mesh-stream this process opens, so stream ids never collide
/// between the two carriers.
pub(crate) fn next_stream_id(prefix: &str) -> String {
    format!(
        "{prefix}-{}-{}",
        std::process::id(),
        STREAM_NONCE.fetch_add(1, Ordering::Relaxed)
    )
}

/// Cap on every peer-controlled side-stream read this plugin performs (both
/// bridges, both roles). The SDK's 16 MiB frame cap does NOT apply to raw
/// side-stream bytes and `expected_bytes` is never enforced, so an unbounded
/// `read_to_end` here was a remote OOM: a peer could stream gigabytes into
/// this process's memory. A capsule / E14 request / E15 answer is a few KB
/// -- 1 MiB is generous headroom, not a constraint anyone legitimate hits.
pub(crate) const MAX_SIDE_STREAM_BYTES: u64 = 1024 * 1024;

/// Bounded replacement for `read_to_end` on a side-stream: reads at most
/// [`MAX_SIDE_STREAM_BYTES`] and errors -- naming `what` -- when the peer
/// sends more, instead of buffering an attacker-chosen amount of memory.
pub(crate) async fn read_to_end_bounded<R>(reader: R, what: &str) -> std::io::Result<Vec<u8>>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut bytes = Vec::new();
    // Read ONE byte past the cap so "exactly at the cap" and "over the cap"
    // are distinguishable without trusting the peer to half-close honestly.
    let mut bounded = reader.take(MAX_SIDE_STREAM_BYTES + 1);
    bounded.read_to_end(&mut bytes).await?;
    if bytes.len() as u64 > MAX_SIDE_STREAM_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{what} exceeded the {MAX_SIDE_STREAM_BYTES}-byte side-stream cap -- refusing \
                 an unbounded peer-controlled read"
            ),
        ));
    }
    Ok(bytes)
}

// ---------------------------------------------------------------------
// Responder role: mesh-inbound `evidence-request/1` stream -> answered in-process
// ---------------------------------------------------------------------

/// Registered as this plugin's `on_open_stream` handler. Never fabricates an
/// Artifact/Refusal itself -- a transport failure (the peer went silent)
/// just drops the stream; only `evidence_answer` ever produces a signed
/// response, so nothing unsigned is ever returned in its place.
pub async fn handle_open_stream(
    request: OpenStreamRequest,
    _context: &mut PluginContext<'_>,
    capsules: std::sync::Arc<crate::capsule_emit::CapsuleState>,
    self_peer: crate::self_peer::SelfPeer,
) -> PluginResult<Option<OpenStreamResponse>> {
    let listener = bind_side_stream(crate::PLUGIN_ID, &request.stream_id)
        .await
        .map_err(|error| PluginError::internal(error.to_string()))?;
    let response = listener.open_stream_response(&request);
    // The id askers address this node by: what the host reports, the id a
    // peer opens the stream to. Never the operator's override, which names
    // this node to pushes, not to the mesh. Until the host reports it, no
    // signed request is addressed to this node and each is refused.
    let own_node = self_peer.learned();
    if own_node.is_none() {
        tracing::warn!(
            "this node's own mesh id is not known yet: signed evidence requests are refused \
             until the host reports it"
        );
    }

    tokio::spawn(async move {
        if let Err(error) = bridge_inbound_evidence_stream(listener, capsules, own_node).await {
            tracing::warn!(%error, "mesh evidence-request responder bridge failed");
        }
    });

    Ok(Some(response))
}

/// The most evidence requests answered at once; see the module doc.
pub const MAX_IN_FLIGHT_ANSWERS: usize = 8;

static IN_FLIGHT: std::sync::OnceLock<tokio::sync::Semaphore> = std::sync::OnceLock::new();

async fn bridge_inbound_evidence_stream(
    listener: mesh_llm_plugin::LocalListener,
    capsules: std::sync::Arc<crate::capsule_emit::CapsuleState>,
    own_node: Option<String>,
) -> anyhow::Result<()> {
    let bound = responder_http_timeout();
    let local = tokio::time::timeout(bound, listener.accept()).await??;
    let (mut read_half, mut write_half) = local.into_split();

    // The remote requester writes the whole request then half-closes
    // (mesh-llm's own streamed-http-binding convention: a full write
    // followed by `shutdown()`, never a length prefix) -- read to EOF, but
    // BOUNDED in bytes (see `read_to_end_bounded`) and in time: the
    // requester controls these bytes and must never control this process's
    // memory, or hold a stream open by going silent.
    let request_bytes = tokio::time::timeout(
        bound,
        read_to_end_bounded(&mut read_half, "mesh-inbound evidence request"),
    )
    .await??;

    let response_bytes = match answer_in_process(capsules, request_bytes, own_node).await {
        Some(bytes) => bytes,
        // Nothing this node can sign: send nothing, never an unsigned
        // stand-in. The requester records an absence.
        None => return Ok(()),
    };

    tokio::time::timeout(bound, async {
        write_half.write_all(&response_bytes).await?;
        write_half.shutdown().await
    })
    .await??;
    Ok(())
}

/// Answer in-process, on the blocking pool (the answer reads the ledger and
/// signs), and log it. `None` when there is nothing to send.
async fn answer_in_process(
    capsules: std::sync::Arc<crate::capsule_emit::CapsuleState>,
    request_bytes: Vec<u8>,
    own_node: Option<String>,
) -> Option<Vec<u8>> {
    let permit = IN_FLIGHT
        .get_or_init(|| tokio::sync::Semaphore::new(MAX_IN_FLIGHT_ANSWERS))
        .try_acquire();
    // A verdict this node issues lands on its own counts: the opt-in
    // stop-routing rule looks again (it does nothing while off).
    let adjudicate = serde_json::from_slice::<serde_json::Value>(&request_bytes)
        .is_ok_and(|request| crate::referee::service::is_adjudicate_request(&request));
    let for_rule = capsules.clone();
    let answered = tokio::task::spawn_blocking(move || {
        let now = crate::evidence_answer::now_utc();
        // Past the in-flight cap, an adjudicate request is declined like any
        // other (below), before any work on it.
        if permit.is_ok() {
            if let Some(verdict_or_refusal) =
                referee_answer(&capsules, &request_bytes, &now, own_node.as_deref())
            {
                drop(permit);
                return verdict_or_refusal;
            }
        }
        let announced = crate::peer_keys::announced::snapshot();
        let configured = crate::settings::var(crate::peer_keys::ENV_PEER_KEYS).ok();
        let responder = crate::evidence_answer::Responder {
            ledger_dir: capsules.ledger_dir(),
            signing_key: capsules.signing_key(),
            now: &now,
            history_segments: crate::share_policy::history_segments(),
            share_standard_history: crate::share_policy::shares_standard_history(),
            keys: crate::requester_auth::Keys {
                announced: &announced,
                configured: configured.as_deref(),
            },
            replay: crate::requester_auth::replay_cache(),
            node_id: own_node.as_deref(),
        };
        let log_dir = capsules
            .ledger_dir()
            .parent()
            .and_then(crate::evidence_routes::EvidenceSource::received_log_dir);
        let outcome = crate::evidence_answer::answer_and_log(
            &request_bytes,
            permit.is_err(),
            &responder,
            log_dir.as_deref(),
        );
        drop(permit);
        if let crate::evidence_answer::Outcome::Unanswerable(why) = &outcome {
            tracing::warn!(%why, "an evidence request was not answered");
        }
        outcome.wire_bytes()
    })
    .await;
    if adjudicate {
        drop(for_rule);
        crate::routing_rule::mark_due();
    }
    answered.ok().flatten()
}

/// Why an evidence request got no answer.
#[derive(Debug)]
pub(crate) enum SendError {
    /// The stream to the peer's plugin could not be opened (unreachable, or
    /// it runs no plugin that takes the channel).
    NotOpened(String),
    /// Opened, but nothing came back in time (the text says why, with a
    /// leading ": " when there is detail).
    NoAnswer(String),
    /// Something came back that is not JSON.
    Malformed(String),
}

/// Send `request` to `peer_id` over the evidence stream and read the answer:
/// `(the request bytes sent, the peer's JSON)`. Signed as this node
/// (`requester_auth::sign_outgoing`) once its key is announced: under the
/// referee's tag when it is an `adjudicate` request, which the referee's
/// door on the far end checks it against, and the Evidence Request's
/// otherwise.
pub(crate) async fn send_evidence_request(
    context: &mut PluginContext<'_>,
    peer_id: &str,
    request: &serde_json::Value,
) -> Result<(Vec<u8>, serde_json::Value), SendError> {
    // The host reads a hex id in either case; a node's own id is lowercase.
    // One form, for the stream and for the audience it is signed for.
    let peer_id = peer_id.trim();
    let peer_id = if peer_id.bytes().all(|b| b.is_ascii_hexdigit()) {
        peer_id.to_ascii_lowercase()
    } else {
        peer_id.to_string()
    };
    let open_request = OpenMeshStreamRequest {
        stream_id: next_stream_id("evidence-request"),
        target_peer_id: peer_id.clone(),
        plugin_id: String::new(), // host fills this in from the connection.
        channel: EVIDENCE_REQUEST_CHANNEL.to_string(),
        purpose: mesh_llm_plugin::proto::StreamPurpose::Generic as i32,
        mode: mesh_llm_plugin::proto::StreamMode::RawBytes as i32,
        bidirectional: true,
        content_type: Some("application/json".to_string()),
        correlation_id: Some(next_stream_id("evidence-correlation")),
        metadata_json: None,
        expected_bytes: None,
        idle_timeout_ms: Some(requester_idle_timeout_ms()),
    };
    let stream: LocalStream = context
        .connect_mesh_stream(open_request)
        .await
        .map_err(|error| SendError::NotOpened(error.to_string()))?;
    let (mut read_half, mut write_half) = stream.into_split();
    let mut request = request.clone();
    let channel = if crate::referee::service::is_adjudicate_request(&request) {
        crate::requester_auth::Channel::RefereeAdjudicate
    } else {
        crate::requester_auth::Channel::EvidenceRequest
    };
    crate::requester_auth::sign_outgoing(channel, &mut request, &peer_id);
    let request_bytes =
        serde_json::to_vec(&request).map_err(|error| SendError::Malformed(error.to_string()))?;
    let write_and_read = async {
        write_half.write_all(&request_bytes).await?;
        write_half.shutdown().await?;
        // Bounded: the peer's answer is peer-controlled bytes too.
        read_to_end_bounded(&mut read_half, "peer evidence-request response").await
    };
    let response_bytes = tokio::time::timeout(
        Duration::from_millis(requester_idle_timeout_ms()),
        write_and_read,
    )
    .await
    .map_err(|_| SendError::NoAnswer(String::new()))?
    .map_err(|error| SendError::NoAnswer(format!(": {error}")))?;
    let answer = serde_json::from_slice(&response_bytes)
        .map_err(|error| SendError::Malformed(error.to_string()))?;
    Ok((request_bytes, answer))
}

/// The referee's answer to an `adjudicate` request, as wire bytes; `None`
/// for any other request. `Some(None)` sends nothing: a verdict whose
/// `adjudication_issued` record did not seal is never sent.
fn referee_answer(
    capsules: &crate::capsule_emit::CapsuleState,
    request_bytes: &[u8],
    now: &str,
    own_node: Option<&str>,
) -> Option<Option<Vec<u8>>> {
    let parsed: serde_json::Value = serde_json::from_slice(request_bytes).ok()?;
    if !crate::referee::service::is_adjudicate_request(&parsed) {
        return None;
    }
    let peer_keys = crate::peer_keys::registry();
    let referee = crate::referee::service::Referee {
        ledger_dir: capsules.ledger_dir(),
        signing_key: capsules.signing_key(),
        peer_keys: peer_keys.as_deref(),
    };
    let announced = crate::peer_keys::announced::snapshot();
    let configured = crate::settings::var(crate::peer_keys::ENV_PEER_KEYS).ok();
    let asked = crate::referee::service::Asked {
        keys: crate::requester_auth::Keys {
            announced: &announced,
            configured: configured.as_deref(),
        },
        replay: crate::requester_auth::replay_cache(),
        own_node,
    };
    let reply = match crate::referee::service::handle_asked_adjudicate_request(
        &referee,
        request_bytes,
        now,
        &asked,
    ) {
        Ok(reply) => reply,
        Err(error) => {
            tracing::warn!(%error, "an adjudicate request was not answered: the verdict could not be held");
            return Some(None);
        }
    };
    if reply
        .get(crate::referee::service::ADJUDICATION_VERDICT_MARKER)
        .is_some()
    {
        if let Err(error) = seal_issued_verdict(capsules, &reply) {
            tracing::warn!(%error, "a verdict was issued but its record did not seal; nothing sent");
            return Some(None);
        }
    }
    Some(serde_json::to_vec(&reply).ok())
}

/// Seal this node's `adjudication_issued` record of the verdict in `reply`.
/// Keyed on the verdict id: a repeated request seals nothing more.
fn seal_issued_verdict(
    capsules: &crate::capsule_emit::CapsuleState,
    reply: &serde_json::Value,
) -> anyhow::Result<()> {
    use serde_json::Value;
    let text = |key: &str| reply.get(key).and_then(Value::as_str);
    let pair = |key: &str| -> Option<[&str; 2]> {
        match reply.get(key)?.as_array()?.as_slice() {
            [a, b] => Some([a.as_str()?, b.as_str()?]),
            _ => None,
        }
    };
    let (Some(verdict), Some(id), Some(referee), Some(halves), Some(nodes)) = (
        text("verdict"),
        text("verdict_capsule_id"),
        text("referee_node_id"),
        pair("halves"),
        pair("half_node_ids"),
    ) else {
        anyhow::bail!("the issued verdict's facts are incomplete");
    };
    let model_hash = reply
        .pointer("/verdict_capsule/model_attestation/compute_attestation/adjudication/model_hash")
        .and_then(Value::as_str);
    let facts = crate::capsule_emit::VerdictFacts {
        verdict,
        verdict_capsule_id: id,
        referee_node_id: referee,
        halves,
        half_node_ids: nodes,
        twin_bracket_id: text("twin_bracket_id"),
        model_hash,
    };
    capsules.emit_adjudication_issued(
        &facts,
        text("referee_capsule_id"),
        text("issued_at").unwrap_or_default(),
    )?;
    Ok(())
}

// ---------------------------------------------------------------------
// Requester role: local tool call -> outbound `evidence-request/1` stream
// ---------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MeshEvidenceRequestArgs {
    /// Hex-encoded mesh peer id to ask, e.g. `iroh::EndpointId::to_string()`.
    pub peer_id: String,
    /// The request map (draft-mih-agent-evidence-request-00), sent as given.
    /// When it names no `requester_id`, this node's own mesh id is added (it
    /// is this node's word). Once this node's key is announced, the request
    /// is also signed (`requester_auth`): its `audience` is set to `peer_id`,
    /// a `nonce` and a `deadline` are added unless it carries them, then
    /// `requester_auth` over the whole map. The responder answers what
    /// depends on who asks only to a signed request addressed to it.
    pub request: serde_json::Value,
    /// Check what comes back against the peer's announced key
    /// (`CAPSULES_PEER_KEYS`). **On unless the caller turns it off**
    /// with an explicit `"verify": false`. The result is
    /// `{"answer": <the peer's JSON>, "request_digest", "verification"}`
    /// (`evidence_answer::verify_response`, or `no_announced_key` when the
    /// registry names no key for the peer). With `"verify": false` it is the
    /// peer's JSON alone, for a caller that verifies it itself.
    #[serde(default = "verify_by_default")]
    pub verify: bool,
}

/// `verify` is on unless a caller names the opt-out.
fn verify_by_default() -> bool {
    true
}

/// Registered as the `mesh_evidence_request` tool/operation handler
/// (`ToolRouter::add_json`). Returns the peer's Artifact-or-Refusal JSON,
/// unchanged, beside its verification (or alone with `"verify": false`); any
/// failure (unroutable peer, channel undeclared on the peer, response never
/// arrives) surfaces as a tool error -- `PluginError` -- which the host's
/// `/tools/` HTTP route reports as a `502`, never a hang.
pub async fn handle_mesh_evidence_request(
    mut args: MeshEvidenceRequestArgs,
    context: &mut PluginContext<'_>,
    ledger_dir: std::path::PathBuf,
    self_id: Option<String>,
) -> PluginResult<serde_json::Value> {
    if args.peer_id.trim().is_empty() {
        return Err(PluginError::invalid_params("peer_id must not be empty"));
    }
    // An adjudicate request names the referee this node CHOSE for a pair:
    // recorded before it is sent, so a verdict about that pair is only ever
    // taken from that referee.
    if let Some(asked) = requested_adjudication(&args.peer_id, &args.request) {
        record_requested_adjudication(&ledger_dir, &asked).map_err(|e| {
            PluginError::internal(format!("could not record the adjudication request: {e}"))
        })?;
    } else {
        add_requester_id(&mut args.request, self_id.as_deref());
    }

    let (request_bytes, answer) =
        match send_evidence_request(context, &args.peer_id, &args.request).await {
            Ok(sent) => sent,
            Err(SendError::NotOpened(error)) => {
                return Err(PluginError::internal(format!(
                    "could not reach peer: {error}"
                )))
            }
            Err(SendError::NoAnswer(error)) => {
                return Err(PluginError::internal(format!(
                    "peer does not answer evidence requests{error}"
                )))
            }
            Err(SendError::Malformed(error)) => {
                return Err(PluginError::internal(format!(
                    "peer returned malformed response: {error}"
                )))
            }
        };
    if !args.verify {
        return Ok(answer);
    }
    let registry = crate::peer_keys::registry();
    let verification = match crate::peer_keys::announced_key_in(registry.as_deref(), &args.peer_id)
        .and_then(|key_id| verifying_key(&key_id))
    {
        Some(key) => crate::evidence_answer::verify_response(
            &request_bytes,
            &answer,
            &key,
            &crate::checkpoint_cadence::active_witness_keys(),
        ),
        None => serde_json::json!({"state": "no_announced_key"}),
    };
    Ok(serde_json::json!({
        "answer": answer,
        "request_digest": evidence_request::digest::request_digest(&request_bytes),
        "verification": verification,
    }))
}

/// Add this node's id as the request's `requester_id`, unless the request
/// already names one (or is not a map, or this node does not know its id).
fn add_requester_id(request: &mut serde_json::Value, self_id: Option<&str>) {
    if let (Some(map), Some(id)) = (request.as_object_mut(), self_id) {
        map.entry("requester_id")
            .or_insert_with(|| serde_json::json!(id));
    }
}

/// An announced `key_id` (raw Ed25519 public key, lowercase hex) as a key.
fn verifying_key(key_id: &str) -> Option<ed25519_dalek::VerifyingKey> {
    let bytes: [u8; 32] = hex::decode(key_id).ok()?.try_into().ok()?;
    ed25519_dalek::VerifyingKey::from_bytes(&bytes).ok()
}

/// The file beside the ledger that lists every adjudicate request this node
/// sent (`crate::verdict_counts::REQUESTED_FILENAME`).
pub const REQUESTED_ADJUDICATIONS_FILENAME: &str = "requested-adjudications.jsonl";

/// `{referee, halves, twin_bracket_id, selection_tier, asked_at}` for an
/// adjudicate request (the tier it was asked at: a verdict that seals another
/// is refused when delivered, `crate::referee::hold`),
/// or `None` for any other evidence request.
pub fn requested_adjudication(
    peer_id: &str,
    request: &serde_json::Value,
) -> Option<serde_json::Value> {
    if request.pointer("/subject/kind").and_then(|k| k.as_str()) != Some("adjudicate") {
        return None;
    }
    let halves: Vec<&str> = request
        .get("halves")?
        .as_array()?
        .iter()
        .map(|h| h.pointer("/capsule/capsule_id").and_then(|id| id.as_str()))
        .collect::<Option<_>>()?;
    if halves.len() != 2 {
        return None;
    }
    Some(serde_json::json!({
        "referee": peer_id,
        "halves": halves,
        "twin_bracket_id": request.get("twin_bracket_id").cloned().unwrap_or(serde_json::Value::Null),
        "selection_tier": request.get("selection_tier").cloned().unwrap_or(serde_json::Value::Null),
        "asked_at": crate::producer::timestamp::utc_now_minute(),
    }))
}

pub(crate) fn record_requested_adjudication(
    ledger_dir: &std::path::Path,
    asked: &serde_json::Value,
) -> std::io::Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(ledger_dir)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(ledger_dir.join(REQUESTED_ADJUDICATIONS_FILENAME))?;
    writeln!(file, "{asked}")?;
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The remote-OOM bound: bytes up to the cap pass through untouched; one
    /// byte over is a descriptive error, never an unbounded buffer.
    #[tokio::test]
    async fn read_to_end_bounded_accepts_at_cap_and_refuses_over_cap() {
        let at_cap = vec![7u8; MAX_SIDE_STREAM_BYTES as usize];
        let ok = read_to_end_bounded(at_cap.as_slice(), "test bytes")
            .await
            .expect("exactly-at-cap must be accepted");
        assert_eq!(ok.len() as u64, MAX_SIDE_STREAM_BYTES);

        let over_cap = vec![7u8; MAX_SIDE_STREAM_BYTES as usize + 1];
        let err = read_to_end_bounded(over_cap.as_slice(), "test bytes")
            .await
            .expect_err("over-cap must be refused");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("side-stream cap"), "{err}");
    }

    /// The stream's referee door is the one that checks who asks: a signed
    /// adjudicate request addressed to another node is refused
    /// `not_authorized`, and so is one addressed here but signed under the
    /// Evidence Request's tag; the same request unsigned, or signed under the
    /// referee's tag, gets past that check and is refused for something else
    /// (this node is named by no key here).
    #[test]
    fn the_stream_checks_an_adjudicate_requests_signature() {
        let dir = tempfile::tempdir().unwrap();
        let capsules = crate::capsule_emit::CapsuleState::open(dir.path(), "referee-door").unwrap();
        let request = serde_json::json!({
            "subject": {"kind": "adjudicate"},
            "selection_tier": 2,
            "halves": [{"capsule": {"capsule_id": "a"}}, {"capsule": {"capsule_id": "b"}}],
        });
        let now = crate::evidence_answer::now_utc();
        let at = chrono::DateTime::parse_from_rfc3339(&now).unwrap().into();
        let key = ed25519_dalek::SigningKey::from_bytes(&[9; 32]);
        let unsigned = serde_json::to_vec(&request).unwrap();
        let signed = |channel, audience| {
            let mut request = request.clone();
            crate::requester_auth::sign(channel, &mut request, &key, at, audience);
            serde_json::to_vec(&request).unwrap()
        };
        use crate::requester_auth::Channel;
        let relayed = signed(Channel::RefereeAdjudicate, "node-x");
        let mistagged = signed(Channel::EvidenceRequest, "node-r");
        let tagged = signed(Channel::RefereeAdjudicate, "node-r");
        let reason = |bytes: &[u8]| {
            let reply = referee_answer(&capsules, bytes, &now, Some("node-r"))
                .flatten()
                .expect("a signed reply");
            serde_json::from_slice::<serde_json::Value>(&reply).unwrap()["reason"].clone()
        };
        assert_eq!(
            reason(&relayed),
            crate::referee::service::REASON_NOT_AUTHORIZED
        );
        assert_eq!(
            reason(&mistagged),
            crate::referee::service::REASON_NOT_AUTHORIZED
        );
        assert_ne!(
            reason(&unsigned),
            crate::referee::service::REASON_NOT_AUTHORIZED
        );
        assert_ne!(
            reason(&tagged),
            crate::referee::service::REASON_NOT_AUTHORIZED
        );
    }

    #[test]
    fn an_adjudicate_request_is_recorded_with_its_referee_and_pair() {
        let dir = tempfile::tempdir().unwrap();
        let request = serde_json::json!({
            "subject": {"kind": "adjudicate"},
            "twin_bracket_id": "bracket-1",
            "selection_tier": 2,
            "halves": [{"capsule": {"capsule_id": "a"}}, {"capsule": {"capsule_id": "b"}}],
        });
        let asked =
            requested_adjudication("referee-node", &request).expect("an adjudicate request");
        record_requested_adjudication(dir.path(), &asked).unwrap();
        let line =
            std::fs::read_to_string(dir.path().join(REQUESTED_ADJUDICATIONS_FILENAME)).unwrap();
        let recorded: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(recorded["referee"], serde_json::json!("referee-node"));
        assert_eq!(recorded["halves"], serde_json::json!(["a", "b"]));
        assert_eq!(recorded["twin_bracket_id"], serde_json::json!("bracket-1"));
        assert_eq!(recorded["selection_tier"], serde_json::json!(2));
    }

    #[test]
    fn any_other_evidence_request_is_not_an_adjudication() {
        let request =
            serde_json::json!({"subject": {"kind": "correlation", "by": "nonce", "value": "n"}});
        assert!(requested_adjudication("peer", &request).is_none());
        let one_half = serde_json::json!({"subject": {"kind": "adjudicate"}, "halves": [{"capsule": {"capsule_id": "a"}}]});
        assert!(requested_adjudication("peer", &one_half).is_none());
    }
}
