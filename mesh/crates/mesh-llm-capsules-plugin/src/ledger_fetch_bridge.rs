//! `ledger-fetch/1`: the plugin-mesh-stream carrier for the E9/E10 two-sided
//! ledger's peer fetch -- on demand, off the request hot path, a
//! peer reads back ONE of THIS node's own sealed ledger entries by
//! `capsule_id` (the join key `lifecycle_channel::peer_capsule_id_for_seal`
//! already threads onto the requester's own sealed row). Modeled directly on
//! `mesh_evidence_bridge`'s responder/requester shape -- same
//! `OpenMeshStreamRequest`/`connect_mesh_stream` mechanism, zero new
//! mesh-llm-host-runtime protocol code -- but answers from
//! `CapsuleState::lookup`, in-process: this channel carries no Bundle
//! semantics and no checkpoint-coverage requirement, so a capsule sealed
//! seconds ago is fetchable immediately.
//!
//! **Plugin-internal, not a standardized wire.** This stream and its message
//! shape are this plugin's own behaviour between two copies of itself. No
//! public specification defines them, they are not a protocol other
//! implementations should target, and they may change between releases.
//! Nothing here stores and forwards a record or promises delivery.
//!
//! **Witness-level recompute only, never a second attestation.** The
//! responder returns the raw, *unsigned* `{capsule, signed_statement_b64,
//! node_pub_key_pem}` for a found entry -- nothing here signs, endorses, or
//! judges it. The requester (and, upstream, the browser's own
//! `recompute-identity.ts`) independently recomputes and verifies; this
//! bridge only carries bytes.
//!
//! **Sharing the plugin's one `on_open_stream` slot.** `DeclarativePluginBuilder`
//! keeps a single responder-side stream handler, and `OpenStreamRequest` (the
//! shape actually delivered to it) carries no channel name -- the mesh host
//! resolves channel gating before the plugin ever sees the request. `main.rs`
//! tells the two carriers apart by `OpenStreamRequest.metadata_json`: this
//! module's requester tags its open with `LEDGER_FETCH_METADATA_TAG`;
//! `mesh_evidence_bridge`'s requester always sends `None`. Anything other
//! than the exact tag falls through to the evidence-request path, so this is
//! purely additive -- no existing behavior changes.
//!
//! **Who asks** is proven as for an evidence request: the asker signs its
//! request (`requester_auth`, once its host took its key), addressed to the
//! node it fetches from, and the responder checks that signature, its
//! `audience`, its deadline and nonce, and the key's binding before it looks
//! anything up ([`answer`]). A record goes only to the node the signature
//! proves, never to the `requester_id` an asker states; unless the operator
//! serves records to any peer (`peers`), an unsigned request is refused
//! before the ledger is read.
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use mesh_llm_plugin::proto::{OpenMeshStreamRequest, OpenStreamRequest, OpenStreamResponse};
use mesh_llm_plugin::{
    bind_side_stream, LocalListener, LocalStream, PluginContext, PluginError, PluginResult,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::capsule_emit::CapsuleState;
use crate::mesh_evidence_bridge::{env_millis, next_stream_id};

/// The mesh channel this plugin declares on both ends, alongside
/// `mesh_evidence_bridge::EVIDENCE_REQUEST_CHANNEL`.
pub const LEDGER_FETCH_CHANNEL: &str = "ledger-fetch/1";

/// The tool name the requester side calls locally
/// (`POST /api/plugins/<plugin_id>/tools/mesh_ledger_fetch`).
pub const LEDGER_FETCH_OPERATION: &str = "mesh_ledger_fetch";

/// Tags an outbound `OpenMeshStreamRequest.metadata_json` as a ledger-fetch
/// open -- see module doc for why this, and not the channel name, is what
/// `main.rs`'s shared `on_open_stream` handler dispatches on.
const LEDGER_FETCH_METADATA_TAG: &str = "ledger-fetch/1";

/// True exactly when an inbound `OpenStreamRequest` is this module's; the
/// requester below is the only code in this plugin that ever sets this tag.
pub fn is_ledger_fetch_request(request: &OpenStreamRequest) -> bool {
    request.metadata_json.as_deref() == Some(LEDGER_FETCH_METADATA_TAG)
}

/// How long the RESPONDER waits for the mesh stream to actually deliver a
/// request (and separately, once it has one, to write the reply) before
/// giving up -- bounds a peer that opens the stream and then goes silent.
/// Overridable so a test proving the bound is enforced does not have to wait
/// out the production default.
fn responder_timeout() -> Duration {
    env_millis("CAPSULES_LEDGER_FETCH_HTTP_TIMEOUT_MS", 10_000)
}

/// How long the REQUESTER waits for a response once the mesh stream is open
/// -- mirrors `mesh_evidence_bridge::requester_idle_timeout_ms`'s contract:
/// a peer that never declared the channel has its stream dropped by the host
/// with no reply, and this bound turns that into a clean failure, not a hang.
fn requester_idle_timeout_ms() -> u64 {
    env_millis("CAPSULES_LEDGER_FETCH_TIMEOUT_MS", 8_000).as_millis() as u64
}

// ---------------------------------------------------------------------
// Responder role: mesh-inbound `ledger-fetch/1` stream -> local ledger lookup
// ---------------------------------------------------------------------

/// Registered (via `main.rs`'s shared dispatcher) as the ledger-fetch half
/// of this plugin's `on_open_stream` handler. Never fabricates a capsule --
/// a lookup miss or a malformed request both produce an honest, explicit
/// response variant (see [`LedgerFetchResponse`]); only a real
/// `CapsuleState::lookup` hit ever carries capsule bytes.
pub async fn handle_open_stream(
    request: OpenStreamRequest,
    _context: &mut PluginContext<'_>,
    capsules: Arc<CapsuleState>,
    self_peer: crate::self_peer::SelfPeer,
) -> PluginResult<Option<OpenStreamResponse>> {
    let listener = bind_side_stream(crate::PLUGIN_ID, &request.stream_id)
        .await
        .map_err(|error| PluginError::internal(error.to_string()))?;
    let response = listener.open_stream_response(&request);
    // The id askers address this node by, as for an evidence request
    // (`mesh_evidence_bridge::handle_open_stream`).
    let own_node = self_peer.learned();
    if own_node.is_none() {
        tracing::warn!(
            "this node's own mesh id is not known yet: signed ledger-fetch requests are refused \
             until the host reports it"
        );
    }

    tokio::spawn(async move {
        if let Err(error) = bridge_inbound_ledger_fetch_stream(listener, capsules, own_node).await {
            tracing::warn!(%error, "mesh ledger-fetch responder bridge failed");
        }
    });

    Ok(Some(response))
}

#[derive(Debug, Deserialize)]
struct LedgerFetchRequest {
    capsule_id: String,
}

/// Every shape the responder can answer with -- deliberately explicit rather
/// than a bare `Option`/error string, so a caller (and a test) can tell
/// "no such entry" from "your request was malformed" from "here are the
/// bytes" without guessing at string contents.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
enum LedgerFetchResponse {
    Found {
        capsule: Value,
        /// Raw COSE_Sign1 bytes, base64-standard-encoded for the JSON
        /// carrier -- the same bytes `Ledger::lookup` returns, untouched.
        signed_statement_b64: String,
        node_pub_key_pem: String,
    },
    NotFound {
        capsule_id: String,
    },
    /// The record exists or not; either way this node does not give it to
    /// this asker (see [`may_serve`]), or the asker proved nothing where it
    /// had to ([`answer`]).
    NotAuthorized {
        capsule_id: String,
    },
    /// The record belongs to an earlier history of this node, set aside when
    /// its owner started a new one: it exists, it is not served from here.
    Archived {
        capsule_id: String,
    },
    Error {
        message: String,
    },
}

async fn bridge_inbound_ledger_fetch_stream(
    listener: LocalListener,
    capsules: Arc<CapsuleState>,
    own_node: Option<String>,
) -> anyhow::Result<()> {
    let local = tokio::time::timeout(responder_timeout(), listener.accept()).await??;
    let (mut read_half, mut write_half) = local.into_split();

    // The requester writes the whole `{"capsule_id": ...}` request then
    // half-closes (same convention `mesh_evidence_bridge` uses) -- read to
    // EOF to get exactly the request bytes.
    let mut request_bytes = Vec::new();
    tokio::time::timeout(
        responder_timeout(),
        read_half.read_to_end(&mut request_bytes),
    )
    .await??;

    let now = crate::evidence_answer::now_utc();
    let announced = crate::peer_keys::announced::snapshot();
    let configured = crate::settings::var(crate::peer_keys::ENV_PEER_KEYS).ok();
    let door = Door {
        now: &now,
        keys: crate::requester_auth::Keys {
            announced: &announced,
            configured: configured.as_deref(),
        },
        replay: crate::requester_auth::replay_cache(),
        own_node: own_node.as_deref(),
    };
    let response = answer(
        &capsules,
        &request_bytes,
        crate::share_policy::history_segments(),
        &door,
    );

    let response_bytes = serde_json::to_vec(&response)?;
    tokio::time::timeout(responder_timeout(), async {
        write_half.write_all(&response_bytes).await?;
        write_half.shutdown().await
    })
    .await??;
    Ok(())
}

/// Whether `record` goes to the asker under the node's `history_segments`
/// switch. `off` serves nothing. A local block and an owner-maintenance
/// record never leave this node. Under `peers` any other record goes to any
/// node; otherwise a record goes only to the node it names as the other side
/// of its exchange, so a record that names no other side is not served.
/// `requester_id` is the node the request's signature proved ([`answer`]).
fn may_serve(record: &Value, requester_id: Option<&str>, history_segments: &str) -> bool {
    if history_segments == "off"
        || crate::evidence_panes::is_local_routing_choice(record)
        || record
            .pointer("/model_attestation/compute_attestation")
            .and_then(|c| c.get(crate::producer::capsule::OWNER_MAINTENANCE_BLOCK))
            .is_some()
    {
        return false;
    }
    if history_segments == "peers" {
        return true;
    }
    crate::evidence_panes::full_counterparty_node_id(record)
        .is_some_and(|other_side| requester_id == Some(other_side.as_str()))
}

/// What a request is checked against before the ledger is read.
struct Door<'a> {
    /// Now, RFC 3339 UTC.
    now: &'a str,
    keys: crate::requester_auth::Keys<'a>,
    replay: &'a crate::requester_auth::ReplayCache,
    /// This node's own mesh id, as the host reported it.
    own_node: Option<&'a str>,
}

#[cfg(test)]
thread_local! {
    static LOOKUPS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many times this thread has looked a record up: a test shows a
/// refusal came before the ledger was read.
#[cfg(test)]
fn lookups() -> u64 {
    LOOKUPS.with(std::cell::Cell::get)
}

/// The pure request-to-response step, split out of the async I/O wrapper
/// so it is directly unit-testable without a live mesh stream. In order:
/// parse; the switch `off` serves nothing; who asks
/// (`requester_auth::principal`: a signature that fails its deadline,
/// window, key, `audience`, nonce or stated id is refused, and an unsigned
/// request is refused unless records go to any peer); and only then the
/// lookup, served to the proven node alone.
fn answer(
    capsules: &CapsuleState,
    request_bytes: &[u8],
    history_segments: &str,
    door: &Door<'_>,
) -> LedgerFetchResponse {
    let malformed = |error: &dyn std::fmt::Display| LedgerFetchResponse::Error {
        message: format!("malformed ledger-fetch request: {error}"),
    };
    let parsed = match serde_json::from_slice::<LedgerFetchRequest>(request_bytes) {
        Ok(parsed) => parsed,
        Err(error) => return malformed(&error),
    };
    let Ok(Value::Object(map)) = serde_json::from_slice::<Value>(request_bytes) else {
        return malformed(&"not a JSON object");
    };
    if crate::requester_auth::form(&map) == crate::requester_auth::Form::Malformed {
        return malformed(&crate::requester_auth::MEMBER);
    }
    let not_authorized = LedgerFetchResponse::NotAuthorized {
        capsule_id: parsed.capsule_id.clone(),
    };
    if history_segments == "off" {
        return not_authorized;
    }
    let Some(now) = capsule_emit_evidence_request::time::parse_utc(door.now) else {
        return LedgerFetchResponse::Error {
            message: "this node's clock is not an RFC 3339 UTC time".into(),
        };
    };
    let principal = match crate::requester_auth::principal(
        crate::requester_auth::Channel::LedgerFetch,
        &map,
        now,
        door.keys,
        door.replay,
        door.own_node,
        history_segments != "peers",
    ) {
        Ok(principal) => principal,
        Err(crate::requester_auth::Refused::Malformed) => {
            return malformed(&crate::requester_auth::MEMBER)
        }
        Err(crate::requester_auth::Refused::NotAuthorized) => return not_authorized,
    };
    let requester_id = principal.as_ref().map(|p| p.node_id.as_str());

    #[cfg(test)]
    LOOKUPS.with(|n| n.set(n.get() + 1));
    match capsules.lookup(&parsed.capsule_id) {
        Ok(Some(entry)) if !may_serve(&entry.capsule, requester_id, history_segments) => {
            LedgerFetchResponse::NotAuthorized {
                capsule_id: parsed.capsule_id,
            }
        }
        Ok(Some(entry)) => LedgerFetchResponse::Found {
            capsule: entry.capsule,
            signed_statement_b64: base64::engine::general_purpose::STANDARD
                .encode(&entry.signed_statement),
            node_pub_key_pem: capsules.public_key_pem(),
        },
        Ok(None)
            if capsules.ledger_dir().parent().is_some_and(|data_dir| {
                crate::owner_maintenance::is_archived(data_dir, &parsed.capsule_id)
            }) =>
        {
            LedgerFetchResponse::Archived {
                capsule_id: parsed.capsule_id,
            }
        }
        Ok(None) => LedgerFetchResponse::NotFound {
            capsule_id: parsed.capsule_id,
        },
        Err(error) => LedgerFetchResponse::Error {
            message: format!("ledger lookup failed: {error}"),
        },
    }
}

// ---------------------------------------------------------------------
// Requester role: local tool call -> outbound `ledger-fetch/1` stream
// ---------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MeshLedgerFetchArgs {
    /// Hex-encoded mesh peer id whose ledger to fetch from, e.g.
    /// `iroh::EndpointId::to_string()`.
    pub peer_id: String,
    /// The peer-asserted `capsule_id` to fetch -- the join key piece 1
    /// (`lifecycle_channel::peer_capsule_id_for_seal`) threads onto this
    /// node's own sealed row for a `RemoteMesh` exchange.
    pub capsule_id: String,
}

/// Registered as the `mesh_ledger_fetch` tool/operation handler. Returns the
/// peer's raw `LedgerFetchResponse` JSON unchanged on success; any transport
/// failure (unroutable peer, channel undeclared on the peer, response never
/// arrives) surfaces as a tool error, which the host's `/tools/` HTTP route
/// reports as a `502`, never a hang.
pub async fn handle_mesh_ledger_fetch(
    args: MeshLedgerFetchArgs,
    context: &mut PluginContext<'_>,
    self_id: Option<String>,
) -> PluginResult<Value> {
    if args.peer_id.trim().is_empty() {
        return Err(PluginError::invalid_params("peer_id must not be empty"));
    }
    if args.capsule_id.trim().is_empty() {
        return Err(PluginError::invalid_params("capsule_id must not be empty"));
    }

    // One form of the id, for the stream and for the audience it is signed
    // for (`mesh_evidence_bridge::send_evidence_request`).
    let peer_id = args.peer_id.trim();
    let peer_id = if peer_id.bytes().all(|b| b.is_ascii_hexdigit()) {
        peer_id.to_ascii_lowercase()
    } else {
        peer_id.to_string()
    };
    let open_request = OpenMeshStreamRequest {
        stream_id: next_stream_id("ledger-fetch"),
        target_peer_id: peer_id.clone(),
        plugin_id: String::new(), // host fills this in from the connection.
        channel: LEDGER_FETCH_CHANNEL.to_string(),
        purpose: mesh_llm_plugin::proto::StreamPurpose::Generic as i32,
        mode: mesh_llm_plugin::proto::StreamMode::RawBytes as i32,
        bidirectional: true,
        content_type: Some("application/json".to_string()),
        correlation_id: Some(next_stream_id("ledger-fetch-correlation")),
        metadata_json: Some(LEDGER_FETCH_METADATA_TAG.to_string()),
        expected_bytes: None,
        idle_timeout_ms: Some(requester_idle_timeout_ms()),
    };

    let stream: LocalStream = context
        .connect_mesh_stream(open_request)
        .await
        .map_err(|error| PluginError::internal(format!("could not reach peer: {error}")))?;
    let (mut read_half, mut write_half) = stream.into_split();

    // Signed as this node once its key is announced; the responder serves a
    // record only to the node the signature proves.
    let mut request = serde_json::json!({ "capsule_id": args.capsule_id, "requester_id": self_id });
    crate::requester_auth::sign_outgoing(
        crate::requester_auth::Channel::LedgerFetch,
        &mut request,
        &peer_id,
    );
    let request_bytes = serde_json::to_vec(&request).expect("a map of strings always serializes");

    let write_and_read = async {
        write_half.write_all(&request_bytes).await?;
        write_half.shutdown().await?;
        let mut response_bytes = Vec::new();
        read_half.read_to_end(&mut response_bytes).await?;
        Ok::<Vec<u8>, std::io::Error>(response_bytes)
    };

    let response_bytes = tokio::time::timeout(
        Duration::from_millis(requester_idle_timeout_ms()),
        write_and_read,
    )
    .await
    .map_err(|_| PluginError::internal("peer does not answer ledger-fetch requests"))?
    .map_err(|error| {
        PluginError::internal(format!(
            "peer does not answer ledger-fetch requests: {error}"
        ))
    })?;

    serde_json::from_slice(&response_bytes).map_err(|error| {
        PluginError::internal(format!(
            "peer returned malformed ledger-fetch response: {error}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capsule_emit::ExchangeRecord;
    use crate::evidence_request_parity::{announced_keys, asker_key, RESPONDER_NODE};
    use crate::requester_auth::{Keys, ReplayCache};
    use ed25519_dalek::SigningKey;
    use serde_json::json;

    const NOW: &str = "2026-10-14T00:00:00Z";

    fn at(s: &str) -> chrono::DateTime<chrono::Utc> {
        chrono::DateTime::parse_from_rfc3339(s).unwrap().into()
    }

    /// `request` signed with `key` at [`NOW`], addressed to `audience`.
    fn signed_with(key: &SigningKey, audience: &str, request: Value) -> Vec<u8> {
        let mut request = request;
        crate::requester_auth::sign(
            crate::requester_auth::Channel::LedgerFetch,
            &mut request,
            key,
            at(NOW),
            audience,
        );
        serde_json::to_vec(&request).unwrap()
    }

    /// A fetch of `capsule_id` by `node`, stating its id, signed by it.
    fn signed_by(node: &str, capsule_id: &str) -> Vec<u8> {
        signed_with(
            &asker_key(node),
            RESPONDER_NODE,
            json!({"capsule_id": capsule_id, "requester_id": node}),
        )
    }

    /// `bytes` asked at `now` of a node switched to `tier`, with `replay`.
    fn answer_at(
        state: &CapsuleState,
        bytes: &[u8],
        tier: &str,
        now: &str,
        replay: &ReplayCache,
    ) -> LedgerFetchResponse {
        let announced = announced_keys();
        let door = Door {
            now,
            keys: Keys {
                announced: &announced,
                configured: None,
            },
            replay,
            own_node: Some(RESPONDER_NODE),
        };
        answer(state, bytes, tier, &door)
    }

    /// `bytes` asked at [`NOW`] with a fresh replay cache.
    fn answer_now(state: &CapsuleState, bytes: &[u8], tier: &str) -> LedgerFetchResponse {
        answer_at(state, bytes, tier, NOW, &ReplayCache::default())
    }

    /// `answer_now`, and whether it looked anything up.
    fn answer_counted(
        state: &CapsuleState,
        bytes: &[u8],
        tier: &str,
    ) -> (LedgerFetchResponse, bool) {
        let before = lookups();
        let response = answer_now(state, bytes, tier);
        (response, lookups() != before)
    }

    /// Same convention `capsule_emit`'s own tests use -- a process-id-suffixed
    /// dir under the OS temp root, wiped before open -- rather than adding a
    /// `tempfile` dependency for one test module.
    fn open_state(label: &str) -> CapsuleState {
        let dir = std::env::temp_dir().join(format!("ledger-fetch-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        CapsuleState::open(&dir, "ledger-fetch-test").expect("open capsule state")
    }

    /// The record's other side: a node whose key the host announced.
    const ASKER: &str = "node-a";

    fn seal_one(state: &CapsuleState) -> String {
        seal_for(state, Some(ASKER))
    }

    fn seal_for(state: &CapsuleState, requesting_party: Option<&str>) -> String {
        let emitted = state
            .emit_for_exchange(&ExchangeRecord {
                model: "test-model",
                client_nonce: None,
                request_bytes: b"{}",
                response_bytes: b"{}",
                latency_ms: 1.0,
                exchange_id: Some("exchange-1"),
                requesting_party,
                host_provenance: None,
            })
            .expect("seal one capsule");
        emitted.capsule_id
    }

    /// (positive) A real hit round-trips the exact capsule bytes and this
    /// node's real pubkey -- proves the responder answers from the ledger,
    /// not from thin air.
    #[test]
    fn found_entry_carries_the_real_capsule_and_pubkey() {
        let state = open_state("found");
        let capsule_id = seal_one(&state);

        let response = answer_now(&state, &signed_by(ASKER, &capsule_id), "counterparties");

        match response {
            LedgerFetchResponse::Found {
                capsule,
                signed_statement_b64,
                node_pub_key_pem,
            } => {
                assert_eq!(capsule["capsule_id"], capsule_id);
                assert!(!signed_statement_b64.is_empty());
                assert_eq!(node_pub_key_pem, state.public_key_pem());
            }
            other => panic!("expected Found, got {other:?}"),
        }
    }

    /// (negative, R4 other half) A capsule_id this node never sealed must
    /// come back `NotFound`, never a fabricated/empty `Found`.
    #[test]
    fn unknown_capsule_id_is_reported_not_found_never_fabricated() {
        let state = open_state("not-found");
        seal_one(&state); // ledger is non-empty, but the id below is not in it.

        let response = answer_now(&state, &signed_by(ASKER, "not-a-real-id"), "counterparties");

        assert_eq!(
            response,
            LedgerFetchResponse::NotFound {
                capsule_id: "not-a-real-id".to_string()
            }
        );
    }

    /// A fetch signed by `node`, or unsigned with no stated id.
    fn ask(
        state: &CapsuleState,
        capsule_id: &str,
        node: Option<&str>,
        tier: &str,
    ) -> LedgerFetchResponse {
        let request = match node {
            Some(node) => signed_by(node, capsule_id),
            None => serde_json::to_vec(&json!({ "capsule_id": capsule_id })).unwrap(),
        };
        answer_now(state, &request, tier)
    }

    fn is_found(response: &LedgerFetchResponse) -> bool {
        matches!(response, LedgerFetchResponse::Found { .. })
    }

    fn is_refused(response: &LedgerFetchResponse) -> bool {
        matches!(response, LedgerFetchResponse::NotAuthorized { .. })
    }

    /// Only the other side of the exchange gets the record: another node, or
    /// an asker that names no one, is declined. MUTANT: drop the
    /// counterparty check and any node reads any record.
    #[test]
    fn a_record_goes_only_to_the_node_it_names_as_the_other_side() {
        let state = open_state("counterparty");
        let id = seal_one(&state);
        {
            let tier = "counterparties";
            assert!(is_found(&ask(&state, &id, Some(ASKER), tier)));
            let other = ask(&state, &id, Some("node-y"), tier);
            assert_eq!(
                other,
                LedgerFetchResponse::NotAuthorized {
                    capsule_id: id.clone()
                }
            );
            assert!(is_refused(&ask(&state, &id, None, tier)));
        }
    }

    #[test]
    fn the_sharing_switch_decides_off_serves_nothing_and_peers_serves_any_node() {
        let state = open_state("switch");
        let id = seal_one(&state);
        assert!(is_refused(&ask(&state, &id, Some(ASKER), "off")));
        // Refused before the lookup: a record it never sealed gets the same.
        assert_eq!(
            ask(&state, "never-sealed", Some(ASKER), "off"),
            LedgerFetchResponse::NotAuthorized {
                capsule_id: "never-sealed".into()
            }
        );
        assert!(is_found(&ask(&state, &id, Some("node-y"), "peers")));
        assert!(is_found(&ask(&state, &id, None, "peers")));
    }

    /// A local block names no other side of an exchange: it never leaves this
    /// node, not even under `peers`, not even to the blocked node itself.
    #[test]
    fn a_local_block_record_is_never_served() {
        let state = open_state("block");
        let block = state
            .emit_local_routing_choice(
                crate::producer::capsule::RoutingChoiceChange::Block,
                ASKER,
                None,
                &[3u8; 32],
                None,
            )
            .expect("seal block");
        for tier in ["counterparties", "peers"] {
            let response = ask(&state, &block.capsule_id, Some(ASKER), tier);
            assert_eq!(
                response,
                LedgerFetchResponse::NotAuthorized {
                    capsule_id: block.capsule_id.clone()
                }
            );
        }
    }

    /// A record that names no other side (no requesting party) is never
    /// served under the default.
    #[test]
    fn a_record_naming_no_other_side_is_declined() {
        let state = open_state("no-side");
        let id = seal_for(&state, None);
        assert!(is_refused(&ask(&state, &id, Some(ASKER), "counterparties")));
    }

    /// (negative) Malformed request bytes must produce an explicit `Error`,
    /// never be mistaken for a `NotFound` (which would look like "the ledger
    /// doesn't have anything with this key" instead of "your request was
    /// unreadable").
    #[test]
    fn malformed_request_bytes_are_reported_as_error_not_not_found() {
        let state = open_state("malformed");

        let response = answer_now(&state, b"not json at all", "counterparties");

        match response {
            LedgerFetchResponse::Error { message } => {
                assert!(message.contains("malformed"), "message was: {message}");
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    /// R-13 on this channel (PRD T3): node-y fetches node-a's record
    /// claiming to be node-a, unsigned, with its own key, or with node-a's
    /// key id over its own signature. Each is `not_authorized` before
    /// anything is looked up. The same fetch signed by node-a is served.
    /// MUTANT: serve on the stated `requester_id` and node-y reads it.
    #[test]
    fn a_node_claiming_to_be_the_other_side_is_refused_before_the_ledger_is_read() {
        let state = open_state("claims-other-side");
        let id = seal_one(&state);
        let request = json!({"capsule_id": id, "requester_id": ASKER});
        let unsigned = serde_json::to_vec(&request).unwrap();
        let own_key_claims_a = signed_with(&asker_key("node-y"), RESPONDER_NODE, request);
        let mut a_key_y_sig: Value = serde_json::from_slice(&own_key_claims_a).unwrap();
        a_key_y_sig["requester_auth"]["key_id"] =
            json!(hex::encode(asker_key(ASKER).verifying_key().to_bytes()));
        let a_key_y_sig = serde_json::to_vec(&a_key_y_sig).unwrap();
        {
            let tier = "counterparties";
            for bytes in [&unsigned, &own_key_claims_a, &a_key_y_sig] {
                let (response, looked) = answer_counted(&state, bytes, tier);
                assert_eq!(
                    response,
                    LedgerFetchResponse::NotAuthorized {
                        capsule_id: id.clone()
                    },
                    "{tier}"
                );
                assert!(!looked, "the ledger was read ({tier})");
            }
            assert!(is_found(&answer_now(&state, &signed_by(ASKER, &id), tier)));
        }
    }

    /// A fetch node-a signed for another node, relayed here, is refused
    /// before anything is looked up: the signature covers its `audience`.
    #[test]
    fn a_fetch_addressed_to_another_node_is_refused_before_the_ledger_is_read() {
        let state = open_state("audience");
        let id = seal_one(&state);
        let relayed = signed_with(
            &asker_key(ASKER),
            "node-x",
            json!({"capsule_id": id, "requester_id": ASKER}),
        );
        for tier in ["counterparties", "peers"] {
            let (response, looked) = answer_counted(&state, &relayed, tier);
            assert_eq!(
                response,
                LedgerFetchResponse::NotAuthorized {
                    capsule_id: id.clone()
                }
            );
            assert!(!looked, "the ledger was read ({tier})");
        }
    }

    /// A fetch the record's other side signed for another channel (an
    /// Evidence Request, a referee's `adjudicate`) is refused before anything
    /// is looked up, though its body is a fetch addressed here: the door
    /// checks it under `ledger-fetch/1`'s tag (ruling u338). MUTANT: one
    /// tag shared by every channel and it is served.
    #[test]
    fn a_fetch_signed_for_another_channel_is_refused_before_the_ledger_is_read() {
        use crate::requester_auth::Channel;
        let state = open_state("channel");
        let id = seal_one(&state);
        for channel in [Channel::EvidenceRequest, Channel::RefereeAdjudicate] {
            let mut request = json!({"capsule_id": id, "requester_id": ASKER});
            crate::requester_auth::sign(
                channel,
                &mut request,
                &asker_key(ASKER),
                at(NOW),
                RESPONDER_NODE,
            );
            let bytes = serde_json::to_vec(&request).unwrap();
            for tier in ["counterparties", "peers"] {
                let (response, looked) = answer_counted(&state, &bytes, tier);
                assert_eq!(
                    response,
                    LedgerFetchResponse::NotAuthorized {
                        capsule_id: id.clone()
                    },
                    "{channel:?} ({tier})"
                );
                assert!(!looked, "the ledger was read: {channel:?} ({tier})");
            }
        }
        assert!(is_found(&ask(&state, &id, Some(ASKER), "counterparties")));
    }

    /// A signed fetch is answered once: the same bytes again before their
    /// deadline, or any time after it, are refused before the ledger is read.
    #[test]
    fn a_replayed_or_late_fetch_is_refused_before_the_ledger_is_read() {
        let state = open_state("replay");
        let id = seal_one(&state);
        let bytes = signed_by(ASKER, &id);
        let replay = ReplayCache::default();
        assert!(is_found(&answer_at(
            &state,
            &bytes,
            "counterparties",
            NOW,
            &replay
        )));
        let before = lookups();
        assert!(is_refused(&answer_at(
            &state,
            &bytes,
            "counterparties",
            NOW,
            &replay
        )));
        let late = "2026-10-14T00:02:01Z";
        let fresh = ReplayCache::default();
        assert!(is_refused(&answer_at(
            &state,
            &bytes,
            "counterparties",
            late,
            &fresh
        )));
        assert_eq!(lookups(), before, "the ledger was read");
    }

    /// Where records go to any peer no identity is needed, yet a signature
    /// that fails is still refused (never read as unsigned); a valid one under
    /// a key no host binds reads as unsigned, as an unsigned fetch does.
    #[test]
    fn under_peers_a_failed_signature_is_refused_and_an_unbound_key_reads_as_unsigned() {
        let state = open_state("peers-signed");
        let id = seal_one(&state);
        let mut tampered: Value = serde_json::from_slice(&signed_by(ASKER, &id)).unwrap();
        tampered["capsule_id"] = json!("another-id");
        let tampered = serde_json::to_vec(&tampered).unwrap();
        let (response, looked) = answer_counted(&state, &tampered, "peers");
        assert!(matches!(
            response,
            LedgerFetchResponse::NotAuthorized { .. }
        ));
        assert!(!looked, "the ledger was read");
        let unbound = signed_with(
            &SigningKey::from_bytes(&[9; 32]),
            RESPONDER_NODE,
            json!({"capsule_id": id}),
        );
        assert!(is_found(&answer_now(&state, &unbound, "peers")));
        let (response, looked) = answer_counted(&state, &unbound, "counterparties");
        assert!(is_refused(&response), "{response:?}");
        assert!(!looked, "the ledger was read");
    }

    /// A malformed `requester_auth` is an `Error`, never served.
    #[test]
    fn a_malformed_signature_member_is_an_error() {
        let state = open_state("malformed-auth");
        let id = seal_one(&state);
        let bytes = serde_json::to_vec(&json!({"capsule_id": id, "requester_auth": "x"})).unwrap();
        for tier in ["counterparties", "peers", "off"] {
            let (response, looked) = answer_counted(&state, &bytes, tier);
            assert!(
                matches!(response, LedgerFetchResponse::Error { .. }),
                "{response:?}"
            );
            assert!(!looked);
        }
    }

    /// (positive) `is_ledger_fetch_request` is what `main.rs`'s shared
    /// `on_open_stream` dispatcher relies on to route inbound streams to
    /// this module instead of `mesh_evidence_bridge` -- prove the tag it
    /// checks is exactly the tag the requester above actually sends.
    #[test]
    fn dispatch_tag_matches_the_requesters_own_metadata_json() {
        let tagged = OpenStreamRequest {
            stream_id: "s".to_string(),
            purpose: mesh_llm_plugin::proto::StreamPurpose::Generic as i32,
            mode: mesh_llm_plugin::proto::StreamMode::RawBytes as i32,
            bidirectional: true,
            content_type: None,
            correlation_id: None,
            metadata_json: Some(LEDGER_FETCH_METADATA_TAG.to_string()),
            expected_bytes: None,
            idle_timeout_ms: None,
        };
        assert!(is_ledger_fetch_request(&tagged));
    }

    /// (negative, R4 other half) The evidence-request carrier's own
    /// `metadata_json: None` open (and any other untagged open) must NOT be
    /// claimed by this module -- otherwise every existing evidence-request
    /// stream would silently misroute into the ledger-fetch responder.
    #[test]
    fn untagged_or_differently_tagged_requests_are_never_claimed() {
        let untagged = OpenStreamRequest {
            stream_id: "s".to_string(),
            purpose: mesh_llm_plugin::proto::StreamPurpose::Generic as i32,
            mode: mesh_llm_plugin::proto::StreamMode::RawBytes as i32,
            bidirectional: true,
            content_type: None,
            correlation_id: None,
            metadata_json: None,
            expected_bytes: None,
            idle_timeout_ms: None,
        };
        assert!(!is_ledger_fetch_request(&untagged));

        let mut other_tag = untagged;
        other_tag.metadata_json = Some("some-other-carrier/1".to_string());
        assert!(!is_ledger_fetch_request(&other_tag));
    }
}
