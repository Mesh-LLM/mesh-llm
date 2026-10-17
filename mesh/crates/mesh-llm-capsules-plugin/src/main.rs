mod adjudication_records;
// Library-shaped: the split-stage verifier, the frame fold and a few record
// kinds are exercised by their own tests and the split-stage fixtures, not
// yet by a live path of this binary.
mod capsule_emit;
mod checkpoint_cadence;
mod data_dir;
mod decision;
mod evidence_answer;
mod evidence_artifact;
mod evidence_log;
mod evidence_panes;
#[cfg(test)]
mod evidence_request_parity;
mod evidence_routes;
mod exchange_corroboration;
mod exchange_text;
mod ledger_fetch_bridge;
mod lifecycle_channel;
mod mesh_evidence_bridge;
mod owner_maintenance;
mod peer_blocks_seen;
mod peer_keys;
/// Not wired into `on_mesh_event` yet -- see the module doc for why
/// (`mesh-llm-plugin = "0.75"` predates the `checkpoint` field this needs to
/// read off `event.peer`). Exercised entirely by its own unit tests today;
/// kept as the tested, ready-to-wire reconciliation logic for when the
/// dependency bumps.
#[allow(dead_code)]
mod peer_root_ledger;
#[allow(dead_code)]
mod producer;
mod received_log;
mod record_push_bridge;
#[cfg(test)]
mod record_push_parity;
mod record_push_receive;
mod referee;
mod requester_auth;
mod routing_choice_bridge;
mod routing_rule;
mod self_peer;
mod served_summary;
mod settings;
mod settlement_channel;
mod settlement_legs;
mod share_policy;
mod split_stage;
mod strict_json;
#[cfg(test)]
mod two_node_e2e;
mod verdict_counts;
mod web_ui_manifest;
mod witness_status;

use crate::producer::capsule::TokenUsage;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use capsule_emit::{CapsuleState, HostProvenance, ObservedHostExchange};
use decision::Decision;
use ledger_fetch_bridge::{MeshLedgerFetchArgs, LEDGER_FETCH_CHANNEL, LEDGER_FETCH_OPERATION};
use lifecycle_channel::{
    peer_capsule_id_for_seal, HostServingProvenance, MirrorUsage, ObservedLifecycleEvents,
    OpenAiExchangeEnvelope, OPENAI_EXCHANGE_CHANNEL,
};
use mesh_evidence_bridge::{
    MeshEvidenceRequestArgs, EVIDENCE_REQUEST_CHANNEL, EVIDENCE_REQUEST_OPERATION,
};
use mesh_llm_plugin::{
    capability, events, inference, mcp, mesh_channel, plugin_server_info, DeclarativePluginBuilder,
    PluginMetadata, PluginRuntime,
};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;

const PLUGIN_ID: &str = "capsules";
/// The plugin's name before it was renamed `capsules`, and the log id every
/// node shared before log ids were node-unique. A log checkpointed under it
/// keeps it (see `owner_maintenance::resolve_log_id`).
const OLD_SHARED_LOG_ID: &str = "capsule-emit-mesh";
const PLUGIN_VERSION: &str = env!("CARGO_PKG_VERSION");
const ENDPOINT_ID: &str = "admission-policy-openai";
/// Declared only alongside the admission endpoint (see `plugin_builder`).
const ADMISSION_CAPABILITY: &str = "admission_policy.v1";

/// The set of model names this plugin advertises via `/v1/models` — i.e. the
/// set it will actually be routed requests for. mesh-llm's real model routing
/// (`mesh-llm-host-runtime::network::openai::ingress`) only falls through to a
/// plugin-hosted inference endpoint when no local/remote backend already
/// serves the named model, matched by exact string — there is no wildcard.
/// The plugin therefore enforces its blocklist by being the
/// registered provider for every blocked name it knows about in advance, not
/// by observing every exchange regardless of model (see docs/PROTOCOL-NOTE.md
/// for why that's a real architectural difference from the private spike).
fn blocked_models() -> Vec<String> {
    blocked_models_for(
        crate::settings::var("CAPSULES_BLOCKED_MODELS")
            .ok()
            .as_deref(),
    )
}

/// The blocked model names this node advertises: only those the operator (or
/// a test) lists in `CAPSULES_BLOCKED_MODELS`. Unset, empty, `none`
/// or `off` advertises none. A real node must never offer a model that always
/// answers 403: a client that takes the first listed model, or "Mesh
/// automatic" routing, would pick it.
fn blocked_models_for(raw: Option<&str>) -> Vec<String> {
    match raw.map(str::trim) {
        None | Some("none") | Some("off") => Vec::new(),
        Some(raw) => raw
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
    }
}

/// The counterparty peer id for a push-at-completion, when this node can
/// truthfully name one -- whoever sealed the other half of this exchange:
///
/// - A `RemoteMesh` terminal (this node routed the exchange TO a peer): the
///   node that served it, the host's `served_by_node_id`.
/// - A host-served terminal (this node SERVED a peer's request): the node that
///   asked, `requested_by_node_id` -- set by the host only for a request that
///   came over the mesh tunnel, from the tunnel's authenticated remote id.
///   A request that reached this node's local API has no requesting mesh
///   node, so nothing is pushed.
///
/// `None` rather than a guess whenever the host did not name one.
fn push_counterparty(envelope: &OpenAiExchangeEnvelope) -> Option<&str> {
    let provenance = envelope.serving_provenance.as_ref()?;
    let counterparty = if envelope.dispatch_path == lifecycle_channel::DispatchPath::RemoteMesh {
        provenance.served_by_node_id.as_deref()
    } else if ObservedLifecycleEvents::is_sealable_host_served(envelope) {
        provenance.requested_by_node_id.as_deref()
    } else {
        None
    };
    counterparty.filter(|id| !id.is_empty() && *id != "unknown")
}

#[derive(Clone)]
struct AppState {
    models: Arc<Vec<String>>,
    capsules: Arc<CapsuleState>,
    /// The host serving-provenance the plugin has observed on the
    /// `openai.exchange.v1` channel, so the seal path can read the host's real
    /// quantization/hardware/model-digest for the served model.
    lifecycle_events: Arc<ObservedLifecycleEvents>,
    /// This node's latest self-produced checkpoint head, if the checkpoint
    /// cadence task is enabled (`checkpoint_cadence::is_enabled`) --
    /// the checkpoint-head source's sending half reads this to feed
    /// `PeerAnnouncement.checkpoint`. On by default; `None` when the
    /// operator has opted out (`CAPSULES_CHECKPOINT_CADENCE=off`)
    /// or hasn't produced a checkpoint since startup.
    #[allow(dead_code)]
    checkpoint_head: Option<checkpoint_cadence::LatestHead>,
}

/// Adapt the lifecycle-channel's mirror of the host `serving_provenance` block
/// into the capsule-emit capture struct. A straight field copy — every field
/// stays `Option`, so a fact the host did not report remains `None` and lands
/// as an honest `unknown`/`null` in the capsule (never fabricated here).
fn host_provenance_from(observed: HostServingProvenance) -> HostProvenance {
    HostProvenance {
        served_by_node_id: observed.served_by_node_id,
        hostname: observed.hostname,
        quantization: observed.quantization,
        architecture: observed.architecture,
        context_length: observed.context_length,
        parameter_size: observed.parameter_size,
        layer_count: observed.layer_count,
        model_identity_hash: observed.model_identity_hash,
        weights_digest: observed.weights_digest,
        model_canonical_ref: observed.model_canonical_ref,
        model_revision: observed.model_revision,
        gpu: observed.gpu,
        vram_bytes: observed.vram_bytes,
        is_soc: observed.is_soc,
    }
}

/// Map the mirror's real token usage into the producer's `TokenUsage`.
/// A straight copy of real counts — never fabricated.
fn token_usage_from(usage: MirrorUsage) -> TokenUsage {
    TokenUsage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
    }
}

/// One `openai.exchange.v1` event, as the channel handler takes it.
///
/// The bodies an opted-in host handed over come off the event first, always:
/// nothing after this (the event window, the split collector, the log) ever
/// holds them. They are written only when `keeping_text` (the owner's opt-in,
/// `exchange_text`).
///
/// Seal-on-observe: a HOST-SERVED terminal exchange (a real loaded GGUF routed
/// host->native-runtime) never reaches this plugin's own HTTP handler, so
/// nothing else produces a capsule for it. When the observed terminal carries
/// real served-model identity (only a real GGUF does; the plugin's own stub
/// does not, so it is not double-sealed), seal a capsule from the observed
/// serving provenance, real usage and the host-forwarded request digest.
///
/// Seal-on-observe, REQUESTER side (on a proxied exchange): a `RemoteMesh`
/// terminal event means THIS node routed the exchange to a peer: it is the
/// requester's own half. Mutually exclusive with the host-served branch (see
/// `is_sealable_host_served`'s doc comment). The coordinator of a split holds
/// its terminal event until the stage records are in (see `split_stage`); the
/// background tick seals it.
///
/// Every event is recorded in `lifecycle_events`. Returns the envelope and
/// the capsule just sealed for it, for the caller to push (a push needs the
/// host context this function does not take).
async fn observe_exchange_event(
    body: &[u8],
    keeping_text: bool,
    capsules: &Arc<CapsuleState>,
    splits: &Splits,
    lifecycle_events: &ObservedLifecycleEvents,
) -> Option<(OpenAiExchangeEnvelope, Value)> {
    let mut envelope = match serde_json::from_slice::<OpenAiExchangeEnvelope>(body) {
        Ok(envelope) => envelope,
        Err(error) => {
            tracing::warn!(%error, "unparseable openai.exchange.v1 envelope");
            return None;
        }
    };
    if let Some(pending) = exchange_text::take(&mut envelope, keeping_text) {
        let ledger_dir = capsules.ledger_dir().to_path_buf();
        let _ = tokio::task::spawn_blocking(move || {
            if let Err(error) = pending.write(&ledger_dir) {
                tracing::warn!(%error, "could not keep this exchange's text");
            }
        })
        .await;
    }
    let envelope = if ObservedLifecycleEvents::is_sealable_host_served(&envelope) {
        let exchange_id = envelope.exchange_id.clone();
        splits.collector.hold(exchange_id.as_deref(), envelope)?
    } else {
        envelope
    };
    if !(ObservedLifecycleEvents::is_sealable_host_served(&envelope)
        || ObservedLifecycleEvents::is_sealable_requester_side(&envelope))
    {
        lifecycle_events.record(envelope);
        return None;
    }
    // Seal on `spawn_blocking`: the emit path holds the std ledger mutex
    // across two `sync_all()`s, which must not stall this tokio worker (the
    // envelope moves into the closure and back out).
    let capsules_for_seal = capsules.clone();
    match tokio::task::spawn_blocking(move || {
        let sealed = seal_observed_host_exchange(&capsules_for_seal, &envelope);
        (envelope, sealed)
    })
    .await
    {
        Ok((envelope, sealed)) => {
            let to_push = sealed.map(|capsule_json| (envelope.clone(), capsule_json));
            lifecycle_events.record(envelope);
            to_push
        }
        Err(join_error) => {
            // Best-effort observability: a panicked/cancelled seal task is
            // logged, never propagated (the envelope was consumed by the task,
            // so there is nothing left to record).
            tracing::warn!(%join_error, "observed-exchange seal task did not complete");
            None
        }
    }
}

/// Seal a capsule for a host-served terminal exchange this plugin only OBSERVED.
/// Best-effort observability: a producer error is logged, never propagated (an
/// observe-path failure must not disturb the host). The three real facts —
/// serving provenance, usage, request digest — come straight off the terminal
/// event; nothing is fabricated.
///
/// Returns the emitted capsule's own JSON on success, `None` on a producer
/// error -- Seam A1's push-at-completion call
/// site needs the just-sealed capsule's bytes to push; every other existing
/// caller of this function predates that need and only used the side effect,
/// so returning the value here is additive, not a behavior change for them.
fn seal_observed_host_exchange(
    capsules: &CapsuleState,
    envelope: &OpenAiExchangeEnvelope,
) -> Option<Value> {
    seal_observed(capsules, envelope, None)
}

/// [`seal_observed_host_exchange`] for the coordinator's main record of a
/// split: the same exchange record, with the split's stage-exchange records
/// sealed first and cited from it.
fn seal_observed_split_exchange(
    capsules: &CapsuleState,
    envelope: &OpenAiExchangeEnvelope,
    plan: &split_stage::SplitPlan,
) -> Option<Value> {
    seal_observed(capsules, envelope, Some(plan))
}

fn seal_observed(
    capsules: &CapsuleState,
    envelope: &OpenAiExchangeEnvelope,
    split: Option<&split_stage::SplitPlan>,
) -> Option<Value> {
    let host_provenance = envelope
        .serving_provenance
        .clone()
        .map(host_provenance_from)
        .unwrap_or_default();
    let usage = envelope.usage.map(token_usage_from);
    let observed = ObservedHostExchange {
        model: &envelope.model,
        exchange_id: envelope.exchange_id.as_deref(),
        request_digest: envelope.request_digest.as_deref(),
        // The host-forwarded digests over the REAL response body: bind
        // agent_output_digest to the real response, and seal the real
        // tool_calls_digest / reasoning_digest (absent when the model had none).
        response_digest: envelope.response_digest.as_deref(),
        tool_calls_digest: envelope.tool_calls_digest.as_deref(),
        reasoning_digest: envelope.reasoning_digest.as_deref(),
        usage,
        host_provenance,
        dispatch_path: envelope.dispatch_path.clone(),
        nonce: envelope.nonce.as_deref(),
        // See `lifecycle_channel::peer_capsule_id_for_seal`'s doc for why
        // this is never a raw `envelope.capsule_id.as_deref()` -- a
        // `SelfMinted` value must never be surfaced as a peer's claim.
        peer_capsule_id: peer_capsule_id_for_seal(envelope).map(|(id, _)| id),
        peer_capsule_id_provenance: peer_capsule_id_for_seal(envelope).map(|(_, prov)| prov),
        twin_bracket_id: envelope.twin_bracket_id.as_deref(),
        response_text_digest: envelope.response_text_digest.as_deref(),
    };
    let emitted = match split {
        Some(plan) => capsules.emit_for_observed_split_exchange(&observed, plan),
        None => capsules.emit_for_observed_host_exchange(&observed),
    };
    match emitted {
        Ok(emitted) => {
            tracing::info!(
                capsule_id = %emitted.capsule_id,
                model = %envelope.model,
                "SEALED AAC for host-served (observed) exchange"
            );
            Some(emitted.capsule)
        }
        Err(error) => {
            tracing::warn!(
                %error,
                observed_not_sealed = capsules.observed_not_sealed(),
                "failed to seal capsule for observed host-served exchange"
            );
            None
        }
    }
}

/// Seam A1's pure eligibility decision,
/// isolated from the network call so "policy off -> no push" is a real,
/// directly testable mutant, not just a structural early-return no test
/// exercises. `Some((peer_id, self_id))` only when EVERY condition holds:
/// pushing is not turned off, a counterparty is knowable, and this node has
/// a configured self peer id to self-declare. Any missing condition is an
/// honest no-push, never a guess.
fn push_eligibility<'a>(
    envelope: &'a OpenAiExchangeEnvelope,
    record_at_completion_off: bool,
    self_peer_id: Option<&'a str>,
) -> Option<(&'a str, &'a str)> {
    if record_at_completion_off {
        return None;
    }
    let peer_id = push_counterparty(envelope)?;
    let self_id = self_peer_id?;
    Some((peer_id, self_id))
}

/// The counterparty a push would have gone to, when the only thing stopping
/// it is an unknown own peer id -- so the caller can say so instead of
/// skipping silently.
fn skipped_for_missing_self_id<'a>(
    envelope: &'a OpenAiExchangeEnvelope,
    record_at_completion_off: bool,
    self_peer_id: Option<&str>,
) -> Option<&'a str> {
    if record_at_completion_off || self_peer_id.is_some() {
        return None;
    }
    push_counterparty(envelope)
}

/// Settlement-records legs ([`settlement_legs`]) from a checked lifecycle
/// event: seal what is due, push a payer's terms leg to the serving peer, and
/// retry terms legs whose earlier push failed. Best-effort, like the trail.
async fn settlement_legs_on_lifecycle(
    context: &mut mesh_llm_plugin::PluginContext<'_>,
    event: &settlement_channel::PaymentLifecycleEvent,
    capsules: &Arc<CapsuleState>,
    self_peer: &self_peer::SelfPeer,
) {
    let (capsules, event) = (capsules.clone(), event.clone());
    let pushes = tokio::task::spawn_blocking(move || {
        let legs = capsules.settlement_legs();
        let mut pushes = legs.retry();
        pushes.extend(legs.on_lifecycle(&capsules, &event));
        (capsules, pushes)
    })
    .await;
    if let Ok((capsules, pushes)) = pushes {
        push_settlement_legs(context, &capsules, pushes, self_peer).await;
    }
}

/// Settlement-records legs from an exchange event: a remote exchange names
/// the serving peer (where this node's terms legs go) and, when terminal,
/// the response digest (this node's delivered leg).
async fn settlement_legs_on_exchange(
    context: &mut mesh_llm_plugin::PluginContext<'_>,
    body: &[u8],
    capsules: &Arc<CapsuleState>,
    self_peer: &self_peer::SelfPeer,
) {
    let Ok(envelope) = serde_json::from_slice::<OpenAiExchangeEnvelope>(body) else {
        return;
    };
    if envelope.dispatch_path != lifecycle_channel::DispatchPath::RemoteMesh {
        return;
    }
    let Some(exchange_id) = envelope.exchange_id.clone() else {
        return;
    };
    let peer = push_counterparty(&envelope).map(str::to_string);
    let digest = envelope.response_digest.clone();
    let capsules = capsules.clone();
    let pushes = tokio::task::spawn_blocking(move || {
        let pushes = capsules.settlement_legs().on_exchange(
            &capsules,
            &exchange_id,
            peer.as_deref(),
            digest.as_deref(),
        );
        (capsules, pushes)
    })
    .await;
    if let Ok((capsules, pushes)) = pushes {
        push_settlement_legs(context, &capsules, pushes, self_peer).await;
    }
}

/// Push payer terms legs; one not sent (pushing off, own peer id unknown, or
/// a failed push) is queued again for the next settlement event.
async fn push_settlement_legs(
    context: &mut mesh_llm_plugin::PluginContext<'_>,
    capsules: &CapsuleState,
    pushes: Vec<settlement_legs::Push>,
    self_peer: &self_peer::SelfPeer,
) {
    if pushes.is_empty() {
        return;
    }
    let self_id = self_peer.current();
    let skip = if share_policy::record_at_completion_is_off() {
        Some("record_at_completion is off")
    } else if self_id.is_none() {
        Some("this node's own peer id is unknown")
    } else {
        None
    };
    if let Some(why) = skip {
        tracing::info!(
            count = pushes.len(),
            reason = why,
            "settlement terms legs not pushed; held for retry"
        );
        for push in &pushes {
            capsules.settlement_legs().requeue(push);
        }
        return;
    }
    let self_id = self_id.unwrap_or_default();
    for push in pushes {
        match record_push_bridge::push_capsule_to_peer(
            context,
            &push.peer,
            &self_id,
            &push.capsule,
            None,
            &[],
        )
        .await
        {
            Ok(()) => tracing::debug!(peer = %push.peer, "pushed settlement terms leg"),
            Err(error) => {
                tracing::warn!(%error, peer = %push.peer, "settlement terms leg push failed; held for retry");
                capsules.settlement_legs().requeue(&push);
            }
        }
    }
}

/// Seam A1 -- push `capsule_json` (this
/// node's own just-sealed capsule) to `envelope`'s counterparty, when
/// `push_eligibility` says to. Best-effort: logs success/failure, never
/// propagates -- a push failure must not disturb sealing or channel-message
/// processing, same discipline as `seal_observed_host_exchange`'s own
/// producer-error handling.
///
/// Push-a-bundle: with the checkpoint cadence on (`checkpoints` is `Some`),
/// the half goes out with a checkpoint covering it and its inclusion proof,
/// so the receiver reaches "in their log" in one step. If no coverage can be
/// had (cadence off, or a checkpoint-layer error) the bare record is pushed,
/// as before -- the half is never held back for want of a proof.
async fn push_at_completion_if_configured(
    context: &mut mesh_llm_plugin::PluginContext<'_>,
    envelope: &OpenAiExchangeEnvelope,
    capsule_json: Value,
    checkpoints: Option<&checkpoint_cadence::CheckpointHandle>,
    self_peer: &self_peer::SelfPeer,
    split_stage_records: &[Value],
) {
    // Learned from the host's mesh events (see `self_peer`), unless the
    // operator overrides it. `None` means the host has not reported it yet.
    let self_id = self_peer.current();
    let record_at_completion_off = share_policy::record_at_completion_is_off();
    if let Some(peer_id) =
        skipped_for_missing_self_id(envelope, record_at_completion_off, self_id.as_deref())
    {
        tracing::warn!(
            %peer_id,
            "record-push at completion skipped: this node's own peer id is unknown (the host has not reported it on a mesh event yet and CAPSULES_SELF_PEER_ID is unset)"
        );
        return;
    }
    let Some((peer_id, self_id)) =
        push_eligibility(envelope, record_at_completion_off, self_id.as_deref())
    else {
        return;
    };
    let coverage = match (
        checkpoints,
        capsule_json.get("capsule_id").and_then(Value::as_str),
    ) {
        (Some(handle), Some(capsule_id)) => match handle.coverage_for(capsule_id).await {
            Ok(coverage) => Some(coverage),
            Err(error) => {
                tracing::warn!(%error, %peer_id, "no checkpoint covering the pushed half -- pushing the bare record");
                None
            }
        },
        _ => None,
    };
    match record_push_bridge::push_capsule_to_peer(
        context,
        peer_id,
        self_id,
        &capsule_json,
        coverage.as_ref(),
        split_stage_records,
    )
    .await
    {
        Ok(()) => {
            tracing::info!(%peer_id, "pushed sealed capsule to counterparty at completion");
        }
        Err(error) => {
            tracing::warn!(%error, %peer_id, "record-push at completion failed");
        }
    }
}

/// The shared split state: the collector, and main records sealed on the
/// background tick whose push waits for the next handler with a host context
/// (a push needs one, and the tick has none).
#[derive(Clone)]
struct Splits {
    collector: Arc<split_stage::SplitCollector<OpenAiExchangeEnvelope>>,
    pushes: Arc<std::sync::Mutex<Vec<SplitPush>>>,
}

struct SplitPush {
    envelope: OpenAiExchangeEnvelope,
    main: Value,
    carried: Vec<Value>,
}

impl Splits {
    fn new() -> Self {
        Splits {
            collector: Arc::default(),
            pushes: Arc::default(),
        }
    }

    fn take_pushes(&self) -> Vec<SplitPush> {
        std::mem::take(
            &mut *self
                .pushes
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }
}

/// Seal every split the collector releases: its stage-exchange records and
/// main record, or -- when the plan cannot be built -- the ordinary exchange
/// record, so the requester's exchange is never left unsealed.
fn seal_released_splits(
    capsules: &CapsuleState,
    splits: &Splits,
    lifecycle_events: &ObservedLifecycleEvents,
) {
    let deadline_ms = split_stage::stage_deadline_ms();
    for ready in splits
        .collector
        .take_due(split_stage::now_ms(), deadline_ms)
    {
        let (sealed, carried) = match split_stage::plan_split(&ready, deadline_ms) {
            Ok(plan) => (
                seal_observed_split_exchange(capsules, &ready.envelope, &plan),
                plan.carried,
            ),
            Err(error) => {
                tracing::warn!(%error, "split plan refused -- sealing the ordinary exchange record");
                (
                    seal_observed_host_exchange(capsules, &ready.envelope),
                    Vec::new(),
                )
            }
        };
        if let Some(main) = sealed {
            splits
                .pushes
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(SplitPush {
                    envelope: ready.envelope.clone(),
                    main,
                    carried,
                });
        }
        lifecycle_events.record(ready.envelope);
    }
}

/// The plugin's background tasks, started once from `main`: the split tick,
/// and the age-out of kept exchange text, which runs at start and hourly
/// whether or not keeping is still on (`exchange_text`).
fn start_background_tasks(
    capsules: &Arc<CapsuleState>,
    splits: &Splits,
    lifecycle_events: &Arc<ObservedLifecycleEvents>,
) {
    spawn_split_tick(capsules.clone(), splits.clone(), lifecycle_events.clone());
    exchange_text::spawn_retention(capsules.ledger_dir().to_path_buf());
}

fn spawn_split_tick(
    capsules: Arc<CapsuleState>,
    splits: Splits,
    lifecycle_events: Arc<ObservedLifecycleEvents>,
) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_millis(200));
        loop {
            tick.tick().await;
            let (capsules, splits, lifecycle_events) =
                (capsules.clone(), splits.clone(), lifecycle_events.clone());
            if let Err(join_error) = tokio::task::spawn_blocking(move || {
                seal_released_splits(&capsules, &splits, &lifecycle_events);
            })
            .await
            {
                tracing::warn!(%join_error, "split seal task did not complete");
            }
        }
    });
}

/// Push the main records sealed since the last handler ran, with the stage
/// records they cite.
async fn drain_split_pushes(
    context: &mut mesh_llm_plugin::PluginContext<'_>,
    splits: &Splits,
    checkpoints: Option<&checkpoint_cadence::CheckpointHandle>,
    self_peer: &self_peer::SelfPeer,
) {
    for push in splits.take_pushes() {
        push_at_completion_if_configured(
            context,
            &push.envelope,
            push.main,
            checkpoints,
            self_peer,
            &push.carried,
        )
        .await;
    }
}

/// One `skippy.stage.v1` event. A stage seals its record and pushes it to
/// the coordinator; stage 0 opens the split in the collector.
async fn handle_stage_event(
    context: &mut mesh_llm_plugin::PluginContext<'_>,
    body: &[u8],
    capsules: &Arc<CapsuleState>,
    splits: &Splits,
    checkpoints: Option<&checkpoint_cadence::CheckpointHandle>,
    self_peer: &self_peer::SelfPeer,
) {
    let event = match split_stage::StageEvent::parse(body) {
        Ok(event) => event,
        Err(error) => {
            tracing::warn!(%error, "refused skippy.stage.v1 event");
            return;
        }
    };
    if event.block.side == crate::producer::stage::Side::Coordinator {
        if let Err(error) = splits.collector.on_stage_zero(event, split_stage::now_ms()) {
            tracing::warn!(%error, dropped = splits.collector.dropped(), "stage-0 event not taken");
        }
        return;
    }
    let block = event.block;
    let coordinator = block.coordinator_node_id.clone();
    let capsules = capsules.clone();
    let sealed = match tokio::task::spawn_blocking(move || capsules.emit_stage_record(&block)).await
    {
        Ok(Ok(emitted)) => emitted,
        Ok(Err(error)) => {
            tracing::warn!(%error, "stage record NOT sealed");
            return;
        }
        Err(join_error) => {
            tracing::warn!(%join_error, "stage record seal task did not complete");
            return;
        }
    };
    tracing::info!(capsule_id = %sealed.capsule_id, "SEALED stage record of a split");
    if share_policy::record_at_completion_is_off() {
        return;
    }
    let Some(self_id) = self_peer.current() else {
        tracing::warn!(%coordinator, "stage record not pushed: this node's own peer id is unknown");
        return;
    };
    let coverage = match checkpoints {
        Some(handle) => handle.coverage_for(&sealed.capsule_id).await.ok(),
        None => None,
    };
    if let Err(error) = record_push_bridge::push_capsule_to_peer(
        context,
        &coordinator,
        &self_id,
        &sealed.capsule,
        coverage.as_ref(),
        &[],
    )
    .await
    {
        tracing::warn!(%error, %coordinator, "stage record push to the coordinator failed");
    }
}

async fn list_models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": state.models
            .iter()
            .map(|id| json!({"id": id, "object": "model", "owned_by": PLUGIN_ID}))
            .collect::<Vec<_>>(),
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> (StatusCode, Json<Value>) {
    let started = Instant::now();
    match decision::decide(&body) {
        Decision::Allow => {
            let parsed_request = serde_json::from_slice::<Value>(&body).ok();
            let model = parsed_request
                .as_ref()
                .and_then(|v| v.get("model").and_then(|m| m.as_str()))
                .unwrap_or("unknown-model")
                .to_string();
            let client_nonce = parsed_request
                .as_ref()
                .and_then(|v| v.get("client_nonce").and_then(|n| n.as_str()))
                .map(str::to_string);

            // Stable per-exchange correlation id for the response and the
            // capsule's `serving_provenance.exchange_id`, so both views of the
            // same exchange share one id (mirrors the host's `x-request-id` /
            // `exchange_id` lineage — the plugin mints its own when serving
            // directly). `usage` is an OpenAI-shaped `usage` object so a real
            // token count would flow straight through `parse_usage`; the stub
            // reports zero work honestly (empty completion) rather than faking
            // counts it did not produce.
            let exchange_id = format!("chatcmpl-{}", agent_input_digest_short(&body));
            let mut response = json!({
                "id": exchange_id,
                "object": "chat.completion",
                "choices": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "admission_policy": {"decision": "allow"},
            });
            let response_bytes = serde_json::to_vec(&response).expect("response is valid JSON");
            let latency_ms = started.elapsed().as_secs_f64() * 1000.0;

            // The host's real quantization/hardware/model-digest for this model,
            // as most recently observed on the openai.exchange.v1 channel. `None`
            // when the host has published none yet -> capsule keeps honest
            // defaults; never fabricated.
            let host_provenance = state
                .lifecycle_events
                .latest_provenance_for_model(&model)
                .map(host_provenance_from);

            // Seal on `spawn_blocking`: `emit_for_exchange` holds the std
            // ledger mutex across two `sync_all()`s -- real disk waits that
            // must not stall a tokio worker thread (and, transitively, every
            // other task scheduled on it). The mutex stays std (the
            // sync-fn-holds-lock design is sound); only the CALL moves onto
            // the blocking pool.
            let capsules = state.capsules.clone();
            let model_for_seal = model.clone();
            let exchange_id_for_seal = exchange_id.clone();
            let body_for_seal = body.clone();
            let emit_result = tokio::task::spawn_blocking(move || {
                capsules.emit_for_exchange(&capsule_emit::ExchangeRecord {
                    model: &model_for_seal,
                    client_nonce: client_nonce.as_deref(),
                    request_bytes: &body_for_seal,
                    response_bytes: &response_bytes,
                    latency_ms,
                    exchange_id: Some(&exchange_id_for_seal),
                    // The requesting party beyond the client nonce is not
                    // carried on this direct-serve path — recorded as
                    // "unknown", not invented.
                    requesting_party: None,
                    host_provenance,
                })
            })
            .await;
            match emit_result {
                Ok(Ok(emitted)) => {
                    tracing::info!(capsule_id = %emitted.capsule_id, %model, "emitted AAC for admitted exchange");
                    response["admission_policy"]["capsule_id"] = json!(emitted.capsule_id);
                    response["admission_policy"]["capsule"] = emitted.capsule;
                }
                Ok(Err(error)) => {
                    // Capsule production is best-effort observability, not a
                    // gate: never let a producer bug turn an admitted
                    // exchange into a denied one.
                    tracing::warn!(%error, "failed to emit capsule for admitted exchange");
                }
                Err(join_error) => {
                    // Same best-effort discipline for a panicked/cancelled
                    // seal task: log, never deny the admitted exchange.
                    tracing::warn!(%join_error, "capsule seal task did not complete");
                }
            }
            (StatusCode::OK, Json(response))
        }
        Decision::Deny { reason } => (
            StatusCode::FORBIDDEN,
            Json(json!({
                "error": {
                    "message": reason,
                    "type": "admission_policy_denied",
                    "code": "blocked_model_prefix",
                }
            })),
        ),
    }
}

/// A short, deterministic per-exchange tag derived from the request body, used
/// to mint a stable correlation id shared by the response `id` and the
/// capsule's `serving_provenance.exchange_id`. Deterministic (not random) so
/// the same request yields the same id — a real correlation handle, not noise.
fn agent_input_digest_short(request_bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(request_bytes);
    hex::encode(&digest[..8])
}

async fn serve_admission_http(listener: TcpListener, state: AppState) {
    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);
    axum::serve(listener, app)
        .await
        .expect("the plugin HTTP server crashed");
}

/// Every operation this plugin serves outside its HTTP routes. Each one goes
/// in through `mcp_item`, which declares it in the manifest as well as routing
/// it: a handler added to the router alone is never published by the host
/// (its tool call is a 404).
fn with_evidence_operations(
    builder: DeclarativePluginBuilder,
    maintenance: Arc<owner_maintenance::Maintenance>,
    capsules: Arc<CapsuleState>,
    self_peer: self_peer::SelfPeer,
) -> DeclarativePluginBuilder {
    let capsules_for_delivery = capsules.clone();
    let mut builder = with_owner_maintenance(builder, maintenance);
    builder = builder.mcp_item(
        mcp::tool(EVIDENCE_REQUEST_OPERATION)
            .description(
                "Ask a mesh peer's capsules plugin for evidence (a \
                 draft-mih-agent-evidence-request-00 request map) over the plugin mesh stream. \
                 Returns the peer's own artifact or signed refusal unchanged, beside its \
                 verification against the peer's announced key (verify: false returns it alone).",
            )
            .input::<MeshEvidenceRequestArgs>()
            .handle({
                let ledger_dir = capsules_for_delivery.ledger_dir().to_path_buf();
                let self_peer = self_peer.clone();
                move |args, context| {
                    let self_id = self_peer.current();
                    Box::pin(mesh_evidence_bridge::handle_mesh_evidence_request(
                        args,
                        context,
                        ledger_dir.clone(),
                        self_id,
                    ))
                }
            }),
    );
    builder = builder.mcp_item(
        mcp::tool(exchange_corroboration::HISTORICAL_REFERENCE_OPERATION)
            .description(
                "Ask a peer X for its half of an earlier exchange, by the SHA-256 of the \
                 exchange's client nonce (or X's own record of it), then ask each named Y once \
                 for Y's half. Returns, per Y, whether the halves agree, disagree (which \
                 fields), are not joinable, one-sided, refused, absent or unverifiable, with \
                 its frame and independence stated separately. No figure across Ys.",
            )
            .input::<exchange_corroboration::RunArgs>()
            .handle({
                let ledger_dir = capsules_for_delivery.ledger_dir().to_path_buf();
                move |args, context| {
                    Box::pin(exchange_corroboration::handle_historical_reference(
                        args,
                        context,
                        ledger_dir.clone(),
                    ))
                }
            }),
    );
    builder = builder.mcp_item(
        mcp::tool(routing_choice_bridge::LOCAL_ROUTING_CHOICE_OPERATION)
            .description(
                "Seal a record that this node stopped (or resumed) routing to a peer. The record \
                 names the peer only by a salted commitment; the salt is returned to the caller.",
            )
            .input::<routing_choice_bridge::LocalRoutingChoiceArgs>()
            .handle(move |args, _context| {
                let capsules = capsules.clone();
                Box::pin(async move {
                    routing_choice_bridge::handle_local_routing_choice(&capsules, args)
                })
            }),
    );
    builder = builder.mcp_item(
        mcp::tool(adjudication_records::DELIVER_OPERATION)
            .description(
                "Deliver a referee's signed twin verdict to a node it concerns, as its courier, \
                 over the record-push stream. Returns the receiver's answer unchanged; a refusal \
                 is also sealed on this node's own chain (adjudication_ack_refused).",
            )
            .input::<adjudication_records::DeliverAdjudicationArgs>()
            .handle({
                let capsules = capsules_for_delivery.clone();
                let self_peer = self_peer.clone();
                move |args, context| {
                    let capsules = capsules.clone();
                    let self_id = self_peer.current();
                    Box::pin(adjudication_records::deliver(
                        args, context, capsules, self_id,
                    ))
                }
            }),
    );
    builder = builder.mcp_item(
        mcp::tool(referee::live::ASK_AGAIN_OPERATION)
            .description(
                "Ask again for an independent check of a twin pair this node asked (by its twin \
                 bracket id). A pair whose one referee call is used is not asked again; one that \
                 found no eligible referee looks for one again. Returns the pair's row.",
            )
            .input::<referee::live::AskAgainArgs>()
            .handle({
                let capsules = capsules_for_delivery.clone();
                let self_peer = self_peer.clone();
                move |args, context| {
                    let capsules = capsules.clone();
                    let self_id = self_peer.current();
                    Box::pin(referee::live::ask_again(args, context, capsules, self_id))
                }
            }),
    );
    builder = builder.mcp_item(
        mcp::tool(referee::supplied::SUPPLY_TWIN_TEXTS_OPERATION)
            .description(
                "Give this node the request and both answers of a twin pair its client marked (by \
                 twin bracket id), so a referee can be asked. They are kept only when the referee \
                 is on, the request and each answer are the bodies this node's own records sealed, \
                 and the answers differ. A referee that is asked receives the request and both \
                 answers. Returns whether they were kept, and the pair's row.",
            )
            .input::<referee::supplied::SupplyTwinTextsArgs>()
            .handle({
                let capsules = capsules_for_delivery.clone();
                let self_peer = self_peer.clone();
                move |args, context| {
                    let capsules = capsules.clone();
                    let self_id = self_peer.current();
                    Box::pin(referee::supplied::supply(args, context, capsules, self_id))
                }
            }),
    );
    builder = builder.mcp_item(
        mcp::tool(LEDGER_FETCH_OPERATION)
            .description(
                "Ask a mesh peer's capsules plugin for one of ITS sealed ledger entries by \
                 capsule_id -- the witness-level fetch half of the two-sided ledger. Returns the raw \
                 unsigned {capsule, signed_statement_b64, node_pub_key_pem} for independent recompute; \
                 never a second attestation.",
            )
            .input::<MeshLedgerFetchArgs>()
            .handle(
                {
                    let self_peer = self_peer.clone();
                    move |args, context| {
                        let self_id = self_peer.current();
                        Box::pin(ledger_fetch_bridge::handle_mesh_ledger_fetch(args, context, self_id))
                    }
                },
            ),
    );
    builder
}

/// The Evidence tab's `Your records` status and its three cleanups, each on
/// the blocking pool: they fsync, and a cleanup must not stall a tokio worker.
fn with_owner_maintenance(
    mut builder: DeclarativePluginBuilder,
    maintenance: Arc<owner_maintenance::Maintenance>,
) -> DeclarativePluginBuilder {
    use mesh_llm_plugin::PluginError;
    use owner_maintenance::{NoArgs, StartNewHistoryArgs};

    async fn blocking<T, F>(f: F) -> mesh_llm_plugin::PluginResult<T>
    where
        F: FnOnce() -> mesh_llm_plugin::PluginResult<T> + Send + 'static,
        T: Send + 'static,
    {
        tokio::task::spawn_blocking(f)
            .await
            .map_err(|e| PluginError::internal(format!("cleanup task did not complete: {e}")))?
    }

    let m = maintenance.clone();
    builder = builder.mcp_item(
        mcp::tool(owner_maintenance::STATUS_OPERATION)
            .description("This node's own records: where they are, how many, stored text count, and the sharing switches as applied.")
            .input::<NoArgs>()
            .handle(
                move |_args: NoArgs, _context| {
                    let m = m.clone();
                    Box::pin(blocking(move || Ok(m.status())))
                },
            ),
    );
    let m = maintenance.clone();
    builder = builder.mcp_item(
        mcp::tool(owner_maintenance::DELETE_STORED_TEXT_OPERATION)
            .description("Delete stored prompt and answer text. Records keep their digests and stay valid; a record of the deletion is sealed.")
            .input::<NoArgs>()
            .handle(
                move |_args: NoArgs, _context| {
                    let m = m.clone();
                    Box::pin(blocking(move || m.delete_stored_text()))
                },
            ),
    );
    let m = maintenance.clone();
    builder = builder.mcp_item(
        mcp::tool(owner_maintenance::REBUILD_INDEX_OPERATION)
            .description("Re-read and re-check every record from disk and rebuild the index. Nothing is lost; a record of the rebuild is sealed.")
            .input::<NoArgs>()
            .handle(
                move |_args: NoArgs, _context| {
                    let m = m.clone();
                    Box::pin(blocking(move || m.rebuild_index()))
                },
            ),
    );
    let m = maintenance;
    builder = builder.mcp_item(
        mcp::tool(owner_maintenance::START_NEW_HISTORY_OPERATION)
            .description("Seal a closing record and start a new history the next time this node starts. The current records are kept whole in an archive.")
            .input::<StartNewHistoryArgs>()
            .handle(
                move |args: StartNewHistoryArgs, _context| {
                    let m = m.clone();
                    Box::pin(blocking(move || m.start_new_history(args)))
                },
            ),
    );
    builder
}

/// What the plugin declares to the host. By default: evidence only -- its
/// mesh channels and events, the Evidence page, its config and HTTP routes
/// (added by the caller). With `admission` (the address of the
/// OpenAI-compatible admission endpoint, present only when
/// `CAPSULES_BLOCKED_MODELS` names models) it also registers that
/// endpoint as an inference provider and declares `admission_policy.v1`.
fn plugin_builder(admission: Option<&str>) -> DeclarativePluginBuilder {
    let description = match admission {
        None => "Seals a signed, hash-chained record of every exchange this node takes part in and serves the Evidence page. It serves no models; its admission-policy test endpoint is opt-in (CAPSULES_BLOCKED_MODELS).",
        Some(_) => "Seals a signed, hash-chained record of every exchange this node takes part in and serves the Evidence page. As configured (CAPSULES_BLOCKED_MODELS), it also serves an OpenAI-compatible admission-policy endpoint that denies the listed models.",
    };
    let builder = DeclarativePluginBuilder::new(PluginMetadata::new(
        PLUGIN_ID,
        PLUGIN_VERSION,
        plugin_server_info(
            PLUGIN_ID,
            PLUGIN_VERSION,
            "Capsule evidence",
            description,
            None::<String>,
        ),
    ))
    .config_item(share_policy::share_policy_config_schema(PLUGIN_ID))
    .web_ui_item(web_ui_manifest::evidence_web_ui())
    .mesh_item(mesh_channel(OPENAI_EXCHANGE_CHANNEL))
    .mesh_item(mesh_channel(EVIDENCE_REQUEST_CHANNEL))
    .mesh_item(mesh_channel(record_push_bridge::RECORD_PUSH_CHANNEL))
    .mesh_item(mesh_channel(LEDGER_FETCH_CHANNEL))
    .mesh_item(mesh_channel(settlement_channel::PAYMENT_LIFECYCLE_CHANNEL))
    // The host's routing choices (blocks and unblocks), so a blocked peer is
    // never chosen as a referee or asked to be blocked again.
    .mesh_item(mesh_channel(peer_blocks_seen::ROUTING_CHOICE_CHANNEL))
    // Any mesh event carries this node's own peer id; the host sends these
    // kinds as a snapshot right after the plugin loads (see `self_peer`).
    .event_item(events::local_accepting())
    .event_item(events::local_standby())
    .event_item(events::mesh_id_updated())
    .event_item(events::peer_up());
    match admission {
        Some(address) => builder
            .provide(capability(ADMISSION_CAPABILITY))
            .inference_item(inference::provider(ENDPOINT_ID, address)),
        None => builder,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Packaging time: print the `plugin-manifest.json` the installer reads
    // (config schema + web UI block) and exit, before any listener, ledger,
    // or key is touched.
    if std::env::args().nth(1).as_deref() == Some("--print-package-manifest") {
        println!(
            "{}",
            web_ui_manifest::package_manifest_json_for(
                PLUGIN_ID,
                PLUGIN_VERSION,
                share_policy::share_policy_config_schema(PLUGIN_ID),
            )?
        );
        return Ok(());
    }

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    // The OpenAI-compatible admission endpoint exists only when blocked
    // models are named (see `plugin_builder`); by default there is no
    // listener, no provider and no admission capability.
    let models = blocked_models();
    let admission = if models.is_empty() {
        None
    } else {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = format!("http://127.0.0.1:{}", listener.local_addr()?.port());
        Some((listener, address))
    };

    // Absolute, never the working directory's; see `data_dir`.
    let data_dir = data_dir::data_dir()?;
    // One process per data directory, held until exit: before the key, the
    // ledger or the checkpoint cadence opens anything in it.
    let _data_dir_lock = data_dir::lock(&data_dir)?;
    tracing::info!(data_dir = %data_dir.display(), "plugin data directory");
    // The log id is node-unique by default (the node's signing key id), and a
    // new history the owner asked for last run starts HERE, before the
    // ledger or the checkpoint cadence opens (see `owner_maintenance`).
    let node_key_id = producer::keys::load_or_create(&data_dir.join("keys"))?.key_id();
    let log_id = owner_maintenance::resolve_log_id(
        &data_dir,
        PLUGIN_ID,
        OLD_SHARED_LOG_ID,
        &node_key_id,
        settings::var(owner_maintenance::LOG_ID_SETTING)
            .ok()
            .as_deref(),
    )?;
    tracing::info!(log_id = %log_id, "checkpoint log id");
    let capsules = Arc::new(CapsuleState::open(&data_dir, PLUGIN_ID)?);
    let evidence_pub_key_pem = capsules.public_key_pem();
    let own_public_key = capsules.public_key();
    let own_signing_key = capsules.signing_key().clone();
    owner_maintenance::finish_pending_after_open(&data_dir, &capsules, &log_id)?;
    tracing::info!(
        chain_head = ?capsules.chain_head(),
        "record sealing ready"
    );
    // The opt-in stop-routing rule (off unless its N is set) also runs once
    // at start, for verdicts recorded while it was off.
    let lifecycle_events = Arc::new(ObservedLifecycleEvents::open(&data_dir)?);
    let self_peer = self_peer::SelfPeer::new(&data_dir);
    let self_peer_for_events = self_peer.clone();

    // Checkpoint cadence: on by default (see checkpoint_cadence.rs's module
    // doc). `checkpoint_shutdown_tx` held for the
    // process lifetime so dropping it doesn't fire the watch early.
    let (checkpoint_shutdown_tx, checkpoint_shutdown_rx) = tokio::sync::watch::channel(false);
    let checkpoints = if checkpoint_cadence::is_enabled() {
        let ledger_dir = data_dir.join("ledger");
        let handle = checkpoint_cadence::spawn(
            ledger_dir,
            log_id.clone(),
            capsules.signing_key().clone(),
            capsules.clone(),
            checkpoint_shutdown_rx,
        )?;
        tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            let _ = checkpoint_shutdown_tx.send(true);
        });
        Some(handle)
    } else {
        drop(checkpoint_shutdown_rx);
        None
    };
    let checkpoint_head = checkpoints
        .as_ref()
        .map(checkpoint_cadence::CheckpointHandle::latest_head);
    let checkpoints_for_handler = checkpoints.clone();

    let capsules_for_handler = capsules.clone();
    // Local routing choices seal onto the same single-writer chain.
    let capsules_for_routing_choice = capsules.clone();
    // the record-push responder's
    // citing-record seal writes to the SAME single-writer ledger -- one
    // `CapsuleState`, shared by Arc, never a second writer.
    let capsules_for_stream = capsules.clone();
    let self_peer_for_stream = self_peer.clone();
    let admission_address = match admission {
        Some((listener, address)) => {
            let app_state = AppState {
                models: Arc::new(models),
                capsules,
                lifecycle_events: lifecycle_events.clone(),
                checkpoint_head,
            };
            tokio::spawn(serve_admission_http(listener, app_state));
            Some(address)
        }
        None => None,
    };

    let lifecycle_events_for_handler = lifecycle_events.clone();
    let splits = Splits::new();
    start_background_tasks(&capsules_for_handler, &splits, &lifecycle_events);
    let splits_for_handler = splits.clone();
    let splits_for_stream = splits.clone();

    let maintenance = Arc::new(owner_maintenance::Maintenance::new(
        data_dir.clone(),
        capsules_for_handler.clone(),
        log_id.clone(),
    ));

    let plugin = plugin_builder(admission_address.as_deref());
    // The Evidence page's data: this plugin's own ledger, served at
    // `/api/plugins/<plugin>/http/...` (see `evidence_routes`).
    let plugin = evidence_routes::with_routes(
        plugin,
        evidence_routes::EvidenceSource {
            ledger_dir: data_dir.join("ledger"),
            node_pub_key_pem: Some(evidence_pub_key_pem),
            received_log_dir: evidence_routes::EvidenceSource::received_log_dir(&data_dir),
        },
    );
    let plugin = with_evidence_operations(
        plugin,
        maintenance,
        capsules_for_routing_choice,
        self_peer.clone(),
    )
    .customize(move |plugin| {
        plugin.on_channel_message(move |message, context| {
            let lifecycle_events = lifecycle_events_for_handler.clone();
            let capsules = capsules_for_handler.clone();
            let checkpoints = checkpoints_for_handler.clone();
            let self_peer = self_peer.clone();
            let splits = splits_for_handler.clone();
            Box::pin(async move {
                drain_split_pushes(context, &splits, checkpoints.as_ref(), &self_peer).await;
                if message.channel == split_stage::SKIPPY_STAGE_CHANNEL {
                    handle_stage_event(context, &message.body, &capsules, &splits, checkpoints.as_ref(), &self_peer)
                        .await;
                    return Ok(());
                }
                if message.channel == OPENAI_EXCHANGE_CHANNEL {
                    settlement_legs_on_exchange(context, &message.body, &capsules, &self_peer).await;
                    if let Some((envelope, capsule_json)) = observe_exchange_event(
                        &message.body,
                        exchange_text::enabled(),
                        &capsules,
                        &splits,
                        &lifecycle_events,
                    )
                    .await
                    {
                        // Seam A1: push this node's own just-sealed capsule
                        // to the counterparty at completion, when one is
                        // knowable and pushing is configured on. See
                        // `push_at_completion_if_configured`'s own doc for
                        // the honest-absence rules.
                        push_at_completion_if_configured(
                            context,
                            &envelope,
                            capsule_json,
                            checkpoints.as_ref(),
                            &self_peer,
                            &[],
                        )
                        .await;
                        // A twin of a pair a client marked: when both twins'
                        // halves are held here, the referee rules apply
                        // (`referee::request`). No bracket id, nothing runs.
                        if let (Some(bracket), Some(self_id)) =
                            (envelope.twin_bracket_id.as_deref(), self_peer.current())
                        {
                            referee::live::consider(context, &capsules, bracket, &self_id, false).await;
                        }
                    }
                } else if message.channel == peer_blocks_seen::ROUTING_CHOICE_CHANNEL {
                    // The host's own routing choices only; a frame relayed
                    // from a peer is refused like a payment event.
                    if settlement_channel::is_local_host_broadcast(
                        &message.source_peer_id,
                        &message.target_peer_id,
                    ) {
                        if let Err(error) =
                            peer_blocks_seen::record(capsules.ledger_dir(), &message.body)
                        {
                            tracing::warn!(%error, "a routing choice from the host was not kept");
                        }
                    } else {
                        tracing::warn!("refused routing.choice.v1 message not from the local host");
                    }
                } else if message.channel == settlement_channel::PAYMENT_LIFECYCLE_CHANNEL
                    && !settlement_channel::is_local_host_broadcast(
                        &message.source_peer_id,
                        &message.target_peer_id,
                    )
                {
                    // Payment lifecycle events are this node's own
                    // observations; a frame relayed from a peer is refused
                    // before it is parsed (see `is_local_host_broadcast`).
                    tracing::warn!("refused payment.lifecycle.v1 message not from the local host");
                } else if message.channel == settlement_channel::PAYMENT_LIFECYCLE_CHANNEL {
                    // One settlement record per checked payment lifecycle
                    // event, sealed on `spawn_blocking` like the exchange seal
                    // above (same ledger mutex across two fsyncs). Never
                    // pushed to peers. Every failure is logged, never
                    // propagated: a refused or unsealed event is missing
                    // evidence, not a failed exchange.
                    match settlement_channel::parse_and_check(&message.body) {
                        Ok(event) => {
                            let legs_event = event.clone();
                            let capsules_for_seal = capsules.clone();
                            match tokio::task::spawn_blocking(move || {
                                capsules_for_seal.emit_settlement_record(&event.observation())
                            })
                            .await
                            {
                                Ok(Ok(Some(sealed))) => {
                                    tracing::info!(capsule_id = %sealed.capsule_id, "sealed settlement record");
                                }
                                Ok(Ok(None)) => {
                                    tracing::debug!("payment lifecycle event already sealed");
                                }
                                Ok(Err(error)) => {
                                    tracing::warn!(%error, "settlement record seal failed");
                                }
                                Err(join_error) => {
                                    tracing::warn!(%join_error, "settlement seal task did not complete");
                                }
                            }
                            settlement_legs_on_lifecycle(context, &legs_event, &capsules, &self_peer).await;
                        }
                        Err(error) => {
                            tracing::warn!(%error, "refused payment.lifecycle.v1 event; not sealed");
                        }
                    }
                }
                // The stop-routing rule, when it is on and a verdict landed:
                // it asks the host through this handler's context.
                routing_rule::evaluate_if_due(context, &capsules).await;
                Ok(())
            })
        })
    })
    .customize(move |plugin| {
        // the record-push responder
        // seals a LOCAL citing record onto THIS node's own chain after the
        // receiver holds a received half -- so it needs the SAME single-writer
        // `CapsuleState` every other local seal uses. Cloned into the handler
        // closure (an `Arc`, so this is a refcount bump, one ledger).
        let capsules_for_stream = capsules_for_stream.clone();
        plugin.on_open_stream(move |request, context| {
            let capsules = capsules_for_stream.clone();
            let self_peer = self_peer_for_stream.clone();
            let collector = splits_for_stream.collector.clone();
            Box::pin(async move {
                // Seam A1: `OpenStreamRequest`
                // carries no channel name (see `record_push_bridge`'s module
                // doc), so the single `on_open_stream` slot dispatches on
                // `content_type`, the one field both carriers set to a
                // distinct, stable value for exactly this purpose.
                let answered = if request.content_type.as_deref() == Some(record_push_bridge::RECORD_PUSH_CONTENT_TYPE) {
                    record_push_bridge::handle_open_stream(request, context, capsules.clone(), collector).await
                } else if ledger_fetch_bridge::is_ledger_fetch_request(&request) {
                    // See `ledger_fetch_bridge`'s module doc for why
                    // `metadata_json` is its dispatch key.
                    ledger_fetch_bridge::handle_open_stream(request, context, capsules.clone(), self_peer.clone()).await
                } else {
                    mesh_evidence_bridge::handle_open_stream(request, context, capsules.clone(), self_peer).await
                };
                // A verdict delivered over a stream lands here: the rule
                // (when on) is evaluated with this handler's context.
                routing_rule::evaluate_if_due(context, &capsules).await;
                answered
            })
        })
    })
    .customize(move |plugin| {
        plugin.on_mesh_event(move |event, _context| {
            let self_peer = self_peer_for_events.clone();
            Box::pin(async move {
                if self_peer.learn(&event.local_peer_id) {
                    tracing::info!(peer_id = %event.local_peer_id, "own mesh peer id reported by the host");
                }
                Ok(())
            })
        })
    })
    .customize(move |plugin| {
        // This node's own key, announced through the host
        // (`peer_keys::own`), and peers' keys for this plugin, when the host
        // announces them (`peer_keys::announced`); CAPSULES_PEER_KEYS stays
        // the fallback. Once the host took this node's key, every evidence
        // request it sends is signed with it (`requester_auth`).
        plugin.on_initialized(move |context| {
            let announces = context.host_supports(peer_keys::announced::CAPABILITY);
            let own_signing_key = own_signing_key.clone();
            Box::pin(async move {
                peer_keys::announced::start(announces, PLUGIN_ID);
                announce_own_key(
                    announces,
                    own_public_key,
                    own_signing_key,
                    requester_auth::asker_key(),
                    |key| context.announce_plugin_key(key),
                )
                .await;
                Ok(())
            })
        })
    })
    .build();

    PluginRuntime::run(plugin).await
}

/// Ask the host to announce this node's key (`peer_keys::own::announce`),
/// and sign the requests this node sends with it only once the host took it.
/// Refused, or a host without the capability, leaves them unsigned: no peer
/// could bind the key, so a signature would prove nothing more than an
/// unsigned request does, and both get the answer any asker gets.
async fn announce_own_key<F, Fut, R, E>(
    announces: bool,
    public_key: [u8; 32],
    signing_key: ed25519_dalek::SigningKey,
    asker: &requester_auth::AskerKey,
    send: F,
) -> peer_keys::own::Announced
where
    F: FnOnce(Vec<u8>) -> Fut,
    Fut: std::future::Future<Output = Result<R, E>>,
    E: std::fmt::Display,
{
    let announced = peer_keys::own::announce(announces, public_key, send).await;
    if announced == peer_keys::own::Announced::Sent {
        asker.set_announced(signing_key);
    }
    announced
}

#[cfg(test)]
mod own_key_tests {
    use super::*;
    use peer_keys::own::Announced;
    use serde_json::json;

    /// What the plugin does at start with each answer the host can give:
    /// whether a request it then sends is signed.
    async fn signs_after(announces: bool, host: Result<(), &'static str>) -> (Announced, bool) {
        let key = ed25519_dalek::SigningKey::from_bytes(&[7; 32]);
        let public = key.verifying_key().to_bytes();
        let asker = requester_auth::AskerKey::default();
        let announced =
            announce_own_key(announces, public, key, &asker, |_| async move { host }).await;
        let mut request = json!({"subject": {"checkpoints": null}});
        asker.sign_outgoing(
            requester_auth::Channel::EvidenceRequest,
            &mut request,
            "node-r",
        );
        let signed =
            requester_auth::form(request.as_object().unwrap()) != requester_auth::Form::Unsigned;
        assert_eq!(signed, asker.is_announced());
        (announced, signed)
    }

    /// Only a key the host took signs: refused, or a host without
    /// `plugin_keys.v1`, sends every request unsigned.
    #[tokio::test]
    async fn requests_are_signed_only_once_the_host_took_the_key() {
        assert_eq!(signs_after(true, Ok(())).await, (Announced::Sent, true));
        assert_eq!(
            signs_after(false, Ok(())).await,
            (Announced::HostLacksCapability, false)
        );
        assert_eq!(
            signs_after(true, Err("busy")).await,
            (Announced::Refused("busy".into()), false)
        );
    }
}

#[cfg(test)]
mod manifest_tests {
    use super::*;
    use mesh_llm_plugin::Plugin;

    fn manifest_for(admission: Option<&str>) -> mesh_llm_plugin::proto::PluginManifest {
        plugin_builder(admission)
            .build()
            .manifest()
            .expect("manifest")
    }

    fn has_inference_endpoint(manifest: &mesh_llm_plugin::proto::PluginManifest) -> bool {
        manifest
            .endpoints
            .iter()
            .any(|e| e.kind == mesh_llm_plugin::proto::EndpointKind::Inference as i32)
    }

    #[test]
    fn the_default_manifest_has_no_inference_provider_and_no_admission_capability() {
        let manifest = manifest_for(None);
        assert!(
            !has_inference_endpoint(&manifest),
            "{:?}",
            manifest.endpoints
        );
        assert!(
            !manifest
                .capabilities
                .iter()
                .any(|c| c == ADMISSION_CAPABILITY),
            "{:?}",
            manifest.capabilities
        );
        // Still the evidence plugin: its channels and page are declared.
        assert!(manifest.web_ui.is_some());
        assert!(!manifest.mesh_channels.is_empty());
    }

    /// R-7 (PRD T7), this plugin's half: what it declares to its host names
    /// no evidence capability, and `evidence-request/1` is its one
    /// evidence-request item, a channel the host needs to deliver a request
    /// to it. That the host passes no plugin channel on to other nodes is
    /// mesh-llm's behaviour and is not tested here.
    #[test]
    fn the_manifest_declares_no_evidence_capability() {
        for manifest in [manifest_for(None), manifest_for(Some("http://127.0.0.1:9"))] {
            assert!(
                !manifest.capabilities.iter().any(|c| c.contains("evidence")),
                "{:?}",
                manifest.capabilities
            );
            let evidence_items: Vec<&str> = manifest
                .mesh_channels
                .iter()
                .map(|c| c.name.as_str())
                .filter(|name| name.contains("evidence"))
                .collect();
            assert_eq!(evidence_items, [EVIDENCE_REQUEST_CHANNEL]);
        }
    }

    #[test]
    fn a_configured_manifest_has_both() {
        let manifest = manifest_for(Some("http://127.0.0.1:9"));
        assert!(has_inference_endpoint(&manifest));
        assert!(manifest
            .capabilities
            .iter()
            .any(|c| c == ADMISSION_CAPABILITY));
    }

    #[test]
    fn unset_empty_none_and_off_all_give_the_default_manifest() {
        let default = manifest_for(None);
        for raw in [None, Some(""), Some(" , "), Some("none"), Some("off")] {
            let models = blocked_models_for(raw);
            assert!(models.is_empty(), "{raw:?}");
            // main() asks for the admission endpoint only for a non-empty list.
            let admission = (!models.is_empty()).then_some("http://127.0.0.1:9");
            assert_eq!(manifest_for(admission), default, "{raw:?}");
        }
    }
}

#[cfg(test)]
mod blocked_models_tests {
    use super::blocked_models_for;

    #[test]
    fn unset_advertises_no_blocked_model() {
        assert!(blocked_models_for(None).is_empty());
    }

    #[test]
    fn empty_advertises_no_blocked_model() {
        assert!(blocked_models_for(Some("")).is_empty());
        assert!(blocked_models_for(Some("  ")).is_empty());
        assert!(blocked_models_for(Some(" , ,")).is_empty());
    }

    #[test]
    fn none_or_off_advertises_no_blocked_model() {
        assert!(blocked_models_for(Some("none")).is_empty());
        assert!(blocked_models_for(Some("off")).is_empty());
    }

    #[test]
    fn an_explicit_list_advertises_exactly_its_names() {
        assert_eq!(
            blocked_models_for(Some("blocked-test-model")),
            vec!["blocked-test-model".to_string()]
        );
        assert_eq!(
            blocked_models_for(Some("a, b")),
            vec!["a".to_string(), "b".to_string()]
        );
    }
}

#[cfg(test)]
mod push_eligibility_tests {
    use super::*;
    use lifecycle_channel::{DispatchPath, HostServingProvenance, Phase};

    fn remote_mesh_envelope(served_by_node_id: Option<&str>) -> OpenAiExchangeEnvelope {
        OpenAiExchangeEnvelope {
            exchange_id: None,
            dispatch_path: DispatchPath::RemoteMesh,
            phase: Phase::Terminal,
            model: "some-model".to_string(),
            status: Some(200),
            capsule_id: None,
            capsule_id_provenance: None,
            nonce: None,
            serving_provenance: served_by_node_id.map(|id| HostServingProvenance {
                served_by_node_id: Some(id.to_string()),
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
                gpu: None,
                vram_bytes: None,
                is_soc: None,
                requested_by_node_id: None,
            }),
            usage: None,
            request_digest: None,
            response_digest: None,
            tool_calls_digest: None,
            reasoning_digest: None,
            twin_bracket_id: None,
            response_text_digest: None,
            exchange_bodies: None,
        }
    }

    fn host_served_envelope() -> OpenAiExchangeEnvelope {
        let mut envelope = remote_mesh_envelope(Some("irrelevant-self-id"));
        envelope.dispatch_path = DispatchPath::TypedFrontend;
        envelope
    }

    /// A real host-served terminal as the fork host publishes it for a request
    /// that arrived over the mesh tunnel: raw-proxy path, 2xx, this node's own
    /// loaded model (a `weights_digest`), and `requested_by_node_id` naming the
    /// node that asked.
    fn tunneled_host_served_envelope(requested_by: Option<&str>) -> OpenAiExchangeEnvelope {
        let mut envelope = remote_mesh_envelope(Some("self-node-a"));
        envelope.dispatch_path = DispatchPath::RawProxy;
        let provenance = envelope
            .serving_provenance
            .as_mut()
            .expect("fixture has provenance");
        provenance.weights_digest = Some("sha256:loaded-weights".to_string());
        provenance.requested_by_node_id = requested_by.map(str::to_string);
        envelope
    }

    // MUTANT: delete `push_eligibility`'s `if record_at_completion_off {
    // return None; }` early return and this goes red -- the required
    // "policy off -> no push" behavior.
    #[test]
    fn policy_off_never_pushes_even_with_a_known_counterparty_and_self_id() {
        let envelope = remote_mesh_envelope(Some("peer-node-a"));
        assert_eq!(push_eligibility(&envelope, true, Some("self-node-b")), None);
    }

    #[test]
    fn policy_on_pushes_when_counterparty_and_self_id_are_both_known() {
        let envelope = remote_mesh_envelope(Some("peer-node-a"));
        assert_eq!(
            push_eligibility(&envelope, false, Some("self-node-b")),
            Some(("peer-node-a", "self-node-b"))
        );
    }

    #[test]
    fn no_self_peer_id_configured_never_pushes() {
        let envelope = remote_mesh_envelope(Some("peer-node-a"));
        assert_eq!(push_eligibility(&envelope, false, None), None);
    }

    // MUTANT: make `skipped_for_missing_self_id` return None unconditionally
    // (the old silent skip) and the first assertion goes red.
    #[test]
    fn a_push_skipped_only_for_a_missing_self_id_is_reported() {
        let envelope = remote_mesh_envelope(Some("peer-node-a"));
        assert_eq!(
            skipped_for_missing_self_id(&envelope, false, None),
            Some("peer-node-a")
        );
        // Not reported when the push goes ahead, when policy is off, or when
        // there is no counterparty to push to anyway.
        assert_eq!(
            skipped_for_missing_self_id(&envelope, false, Some("self-node-b")),
            None
        );
        assert_eq!(skipped_for_missing_self_id(&envelope, true, None), None);
        let nobody = remote_mesh_envelope(None);
        assert_eq!(skipped_for_missing_self_id(&nobody, false, None), None);
    }

    // Option A: the provider pushes its
    // sealed half to the node that asked -- the other direction of the same
    // exchange. MUTANT: return `None` from `push_counterparty`'s host-served
    // arm and this goes red.
    #[test]
    fn host_served_terminal_pushes_to_the_node_that_asked() {
        let envelope = tunneled_host_served_envelope(Some("peer-node-b"));
        assert_eq!(
            push_eligibility(&envelope, false, Some("self-node-a")),
            Some(("peer-node-b", "self-node-a"))
        );
    }

    // Never to itself: the served-by id (this node) is not a counterparty.
    #[test]
    fn host_served_terminal_never_pushes_to_its_own_served_by_id() {
        let envelope = tunneled_host_served_envelope(Some("peer-node-b"));
        assert_ne!(push_counterparty(&envelope), Some("self-node-a"));
    }

    // A request on this node's local API has no requesting mesh node; the host
    // sends none, so nothing is pushed -- never fabricated.
    #[test]
    fn host_served_terminal_from_the_local_api_never_pushes() {
        let envelope = tunneled_host_served_envelope(None);
        assert_eq!(
            push_eligibility(&envelope, false, Some("self-node-a")),
            None
        );
    }

    #[test]
    fn host_served_terminal_with_unknown_requester_never_pushes() {
        let envelope = tunneled_host_served_envelope(Some("unknown"));
        assert_eq!(
            push_eligibility(&envelope, false, Some("self-node-a")),
            None
        );
    }

    // Policy off wins on the provider side too.
    #[test]
    fn host_served_terminal_with_policy_off_never_pushes() {
        let envelope = tunneled_host_served_envelope(Some("peer-node-b"));
        assert_eq!(push_eligibility(&envelope, true, Some("self-node-a")), None);
    }

    // Not a real served exchange (no loaded-model identity: the plugin's own
    // stub shape), so it is not a sealable host-served terminal -- a requester
    // id on it is never used.
    #[test]
    fn a_non_sealable_host_served_shape_never_pushes_even_with_a_requester() {
        let mut envelope = host_served_envelope();
        envelope
            .serving_provenance
            .as_mut()
            .unwrap()
            .requested_by_node_id = Some("peer-node-b".to_string());
        assert_eq!(
            push_eligibility(&envelope, false, Some("self-node-a")),
            None
        );
    }

    // The host's JSON field lands in the mirror (and is optional on the wire).
    #[test]
    fn requested_by_node_id_deserializes_from_the_host_envelope_and_defaults_absent() {
        let with: HostServingProvenance = serde_json::from_value(
            serde_json::json!({"served_by_node_id": "node-a", "requested_by_node_id": "node-b"}),
        )
        .expect("parse");
        assert_eq!(with.requested_by_node_id.as_deref(), Some("node-b"));
        let without: HostServingProvenance =
            serde_json::from_value(serde_json::json!({"served_by_node_id": "node-a"}))
                .expect("parse");
        assert_eq!(without.requested_by_node_id, None);
    }

    #[test]
    fn remote_mesh_with_no_served_by_node_id_has_no_knowable_counterparty() {
        let envelope = remote_mesh_envelope(None);
        assert_eq!(
            push_eligibility(&envelope, false, Some("self-node-b")),
            None
        );
    }

    #[test]
    fn remote_mesh_with_unknown_served_by_node_id_has_no_knowable_counterparty() {
        let envelope = remote_mesh_envelope(Some("unknown"));
        assert_eq!(
            push_eligibility(&envelope, false, Some("self-node-b")),
            None
        );
    }
}

#[cfg(test)]
mod published_operations_tests {
    use super::*;
    use mesh_llm_plugin::Plugin;

    /// The host publishes only the operations the manifest declares, so a
    /// handler that is routed but not declared is a 404 on the live node.
    #[test]
    fn every_evidence_operation_is_declared_in_the_manifest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let capsules =
            Arc::new(CapsuleState::open(dir.path(), "node-under-test").expect("open state"));
        let maintenance = Arc::new(owner_maintenance::Maintenance::new(
            dir.path().to_path_buf(),
            capsules.clone(),
            "node-under-test".into(),
        ));
        let builder = DeclarativePluginBuilder::new(PluginMetadata::new(
            PLUGIN_ID,
            PLUGIN_VERSION,
            plugin_server_info(PLUGIN_ID, PLUGIN_VERSION, "test", "test", None::<String>),
        ));
        let plugin = with_evidence_operations(
            builder,
            maintenance,
            capsules,
            self_peer::SelfPeer::new(dir.path()),
        )
        .build();
        let manifest = plugin.manifest().expect("manifest");
        let declared: Vec<&str> = manifest
            .operations
            .iter()
            .map(|op| op.name.as_str())
            .collect();
        for name in [
            EVIDENCE_REQUEST_OPERATION,
            exchange_corroboration::HISTORICAL_REFERENCE_OPERATION,
            routing_choice_bridge::LOCAL_ROUTING_CHOICE_OPERATION,
            LEDGER_FETCH_OPERATION,
            adjudication_records::DELIVER_OPERATION,
            owner_maintenance::STATUS_OPERATION,
            owner_maintenance::DELETE_STORED_TEXT_OPERATION,
            owner_maintenance::REBUILD_INDEX_OPERATION,
            owner_maintenance::START_NEW_HISTORY_OPERATION,
        ] {
            assert!(
                declared.contains(&name),
                "{name} is routed but not declared: {declared:?}"
            );
        }
    }
}

#[cfg(test)]
mod exchange_text_wiring_tests {
    use super::*;
    use std::path::Path;

    /// A terminal event from a host that hands bodies to plugins, as the
    /// channel delivers it (wire bytes).
    const EVENT_WITH_BODIES: &str = r#"{"dispatch_path":"raw_proxy","phase":"terminal","model":"m","status":200,
        "exchange_id":"0f8fad5b-d9cb-469f-a165-70867728950e",
        "exchange_bodies":{"request":{"messages":[{"role":"user","content":"SECRET-PROMPT"}]},
                           "response":{"choices":[{"message":{"content":"SECRET-ANSWER"}}]}}}"#;

    /// The channel handler's own path for this event
    /// (`observe_exchange_event`), with the given keep-text switch.
    async fn observe(data_dir: &Path, keeping: bool) -> ObservedLifecycleEvents {
        let capsules = Arc::new(CapsuleState::open(data_dir, "wiring-test").unwrap());
        let store = ObservedLifecycleEvents::open(data_dir).unwrap();
        let pushed = observe_exchange_event(
            EVENT_WITH_BODIES.as_bytes(),
            keeping,
            &capsules,
            &Splits::new(),
            &store,
        )
        .await;
        assert!(
            pushed.is_none(),
            "an unsealable event is recorded, never pushed"
        );
        store
    }

    fn by_exchange(data_dir: &Path) -> Vec<String> {
        std::fs::read_dir(data_dir.join("ledger/disclosures/by-exchange"))
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Host on, plugin off: nothing on disk, and the text is held nowhere
    /// in memory or in the events log either.
    #[tokio::test]
    async fn host_on_plugin_off_keeps_nothing_on_disk_or_in_memory() {
        let dir = tempfile::tempdir().unwrap();
        let store = observe(dir.path(), false).await;
        assert!(by_exchange(dir.path()).is_empty());
        assert!(store.snapshot().iter().all(|e| e.exchange_bodies.is_none()));
        let log =
            std::fs::read_to_string(dir.path().join("lifecycle-events.jsonl")).unwrap_or_default();
        assert!(!log.contains("SECRET"), "{log}");
    }

    /// The background tasks `main` starts include the kept-text age-out: a
    /// stale text goes without any write. MUTANT: drop `spawn_retention` from
    /// `start_background_tasks` and the stale text stays.
    #[tokio::test]
    async fn the_background_tasks_age_out_kept_text_without_a_write() {
        let dir = tempfile::tempdir().unwrap();
        let capsules = Arc::new(CapsuleState::open(dir.path(), "retention-test").unwrap());
        let by_exchange = capsules.ledger_dir().join("disclosures/by-exchange");
        std::fs::create_dir_all(&by_exchange).unwrap();
        let stale = by_exchange.join("0f8fad5b-d9cb-469f-a165-000000000000.json");
        std::fs::write(&stale, b"{}").unwrap();
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(400 * 24 * 60 * 60);
        std::fs::File::options()
            .write(true)
            .open(&stale)
            .unwrap()
            .set_modified(old)
            .unwrap();
        let store = Arc::new(ObservedLifecycleEvents::open(dir.path()).unwrap());

        start_background_tasks(&capsules, &Splits::new(), &store);

        for _ in 0..100 {
            if !stale.exists() {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        panic!("the stale kept text was not aged out by the background tasks");
    }

    /// Plugin on: one file on disk, and still no copy in memory once written.
    #[tokio::test]
    async fn plugin_on_writes_the_file_then_holds_no_copy() {
        let dir = tempfile::tempdir().unwrap();
        let store = observe(dir.path(), true).await;
        assert_eq!(
            by_exchange(dir.path()),
            vec!["0f8fad5b-d9cb-469f-a165-70867728950e.json".to_string()]
        );
        assert!(store.snapshot().iter().all(|e| e.exchange_bodies.is_none()));
        let log =
            std::fs::read_to_string(dir.path().join("lifecycle-events.jsonl")).unwrap_or_default();
        assert!(!log.contains("SECRET"), "{log}");
    }
}
