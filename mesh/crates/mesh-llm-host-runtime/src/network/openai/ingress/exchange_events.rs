//! Publishing of the raw-proxy route's `openai.exchange.v1` events: the
//! effective and terminal envelopes, the serving provenance, token usage,
//! digests and client nonce they carry, and the channel a route publishes to.
use crate::mesh;
use crate::network::openai::transport as proxy;
use crate::plugin::openai_exchange::{
    ClientNonceSource, ExchangeUsage, OpenAiExchangeChannel, OpenAiExchangeDispatchPath,
    OpenAiExchangeEnvelope, ServingProvenance,
};

use super::{IngressRouteContext, twin_bracket_id};

/// The status code an out-of-process plugin sees for path 2's terminal
/// event, best-effort from [`proxy::RouteDispatchOutcome`] — `None` when the
/// outcome carries no HTTP status at all (a dropped/failed connection).
pub(super) fn plugin_route_status(outcome: &proxy::RouteDispatchOutcome) -> Option<u16> {
    match *outcome {
        proxy::RouteDispatchOutcome::PolicyDenied => Some(403),
        proxy::RouteDispatchOutcome::RequiredHookFailed => Some(503),
        proxy::RouteDispatchOutcome::Responded(status) => Some(status),
        proxy::RouteDispatchOutcome::RespondedWithUsage { status_code, .. } => Some(status_code),
        proxy::RouteDispatchOutcome::RespondedWithDigests { status_code, .. } => Some(status_code),
        proxy::RouteDispatchOutcome::FailedWithStatus { status_code, .. } => Some(status_code),
        proxy::RouteDispatchOutcome::Failed(_) | proxy::RouteDispatchOutcome::Dropped(_) => None,
    }
}

/// Gather the maximum inference provenance the host *actually knows* for the
/// exchange just served — what ran, at what fidelity, on whose hardware — from
/// state the local node already holds: the served-model descriptor (quant,
/// architecture, context length, identity hash, weights digest, revision) and
/// this host's startup hardware survey (gpu, vram, soc, hostname). Every
/// field is a real value or omitted; nothing is invented. Returned as a plain
/// data struct so the wire event stays independent of the node internals.
/// `requested_by_node_id` is the mesh node that asked, when the request came
/// over the mesh HTTP tunnel (see [`IngressRouteContext::requested_by_node_id`]).
pub(super) async fn serving_provenance_for_model(
    node: &mesh::Node,
    model_name: &str,
    requested_by_node_id: Option<&str>,
) -> ServingProvenance {
    // The served-model descriptor for exactly this model, if the node has one.
    // We match on the served identity's `model_name`; a miss (peer-served or
    // not-yet-described) leaves every model field `None` rather than guessing.
    let descriptor = node
        .served_model_descriptors()
        .await
        .into_iter()
        .find(|d| d.identity.model_name == model_name);

    let (identity, metadata) = match descriptor {
        Some(d) => (Some(d.identity), d.metadata),
        None => (None, None),
    };

    ServingProvenance {
        served_by_node_id: node.id().to_string(),
        hostname: node.hostname.clone(),
        quantization: metadata.as_ref().and_then(|m| m.quant.clone()),
        architecture: metadata.as_ref().and_then(|m| m.architecture.clone()),
        context_length: metadata.as_ref().and_then(|m| m.native_context_length),
        parameter_size: metadata.as_ref().and_then(|m| m.parameter_size.clone()),
        layer_count: metadata.as_ref().and_then(|m| m.layer_count),
        model_identity_hash: identity.as_ref().and_then(|i| i.identity_hash.clone()),
        model_canonical_ref: identity.as_ref().and_then(|i| i.canonical_ref.clone()),
        model_revision: identity.as_ref().and_then(|i| i.revision.clone()),
        weights_digest: identity.as_ref().and_then(|i| i.weights_digest.clone()),
        gpu: node.gpu_name.clone(),
        // `advertised_memory.total_bytes` is 0 when no accelerator memory was
        // enumerated; surface it only when it is a real, non-zero figure.
        vram_bytes: (node.advertised_memory.total_bytes != 0)
            .then_some(node.advertised_memory.total_bytes),
        is_soc: node.is_soc,
        requested_by_node_id: requested_by_node_id.map(str::to_string),
    }
}

/// Extract the served backend's real token usage from a dispatch outcome, when
/// it carried one. Only `RespondedWithUsage` — the outcome the host-served
/// `route_model_request` returns after reading the backend's `usage` object —
/// yields counts; every other outcome (plugin stub, status-only, error, drop)
/// yields `None`, so the terminal envelope omits `usage` rather than reporting
/// fabricated zeros.
pub(super) fn exchange_usage_from_outcome(
    outcome: &proxy::RouteDispatchOutcome,
) -> Option<ExchangeUsage> {
    let proxy::RouteDispatchOutcome::RespondedWithUsage { usage, .. } = outcome else {
        return None;
    };
    // All three or none: a backend that reports two counts but not the
    // third has told us something is missing, and this mirrors the
    // invariant `mesh_llm_events::logging::events::TokenUsage::from_counts`
    // already documents elsewhere -- "missing, overflowing, or internally
    // inconsistent usage must not be estimated." Never a derived total (a
    // backend's real total can legitimately disagree with
    // prompt+completion, e.g. reasoning tokens folded into `total`), and
    // never a zero standing in for an absent count. `cached_prompt_tokens`
    // rides along when the backend reported it -- for billing
    // reconciliation, dropping it is usually the difference between a right
    // and a wrong number.
    let (Some(prompt_tokens), Some(completion_tokens), Some(total_tokens)) = (
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
    ) else {
        return None;
    };
    Some(ExchangeUsage {
        prompt_tokens,
        cached_prompt_tokens: usage.cached_prompt_tokens,
        completion_tokens,
        total_tokens,
    })
}

/// Whether a dispatch outcome actually means inference ran and a response
/// body was returned to the client — the only case `ServingProvenance`'s
/// contract ("what ran, at what fidelity, on whose hardware") can honestly
/// describe. A 503/`Failed`/`Dropped` outcome served nothing, so attaching
/// provenance there would be exactly the fabrication the envelope promises
/// never to do.
pub(super) fn outcome_was_served(outcome: &proxy::RouteDispatchOutcome) -> bool {
    matches!(
        outcome,
        proxy::RouteDispatchOutcome::Responded(200..=299)
            | proxy::RouteDispatchOutcome::RespondedWithDigests {
                status_code: 200..=299,
                ..
            }
            | proxy::RouteDispatchOutcome::RespondedWithUsage {
                status_code: 200..=299,
                ..
            }
    )
}

/// The serving provenance a routing node can state for an exchange a peer
/// delivered: only `served_by_node_id`, naming that peer, never this node.
/// `serving_provenance_for_model` above describes this node's own model and
/// hardware, so it is never used on this path.
///
/// The peer is the one `route_model_request` saw deliver the attempt
/// (`observed_served_by_hex`, from [`proxy::ServedByNodeIdSink`]), which
/// covers a request that named no `x-mesh-target`. An explicit `x-mesh-target`
/// is the fallback: routing narrows the candidates to exactly that peer.
/// `None` when the outcome was not served or neither is known. Every hardware
/// and model field stays absent, since this node never touched that hardware.
pub(super) fn serving_provenance_for_remote_mesh(
    target: Option<iroh::EndpointId>,
    observed_served_by_hex: Option<String>,
    outcome: &proxy::RouteDispatchOutcome,
) -> Option<ServingProvenance> {
    if !outcome_was_served(outcome) {
        return None;
    }
    let served_by_node_id =
        observed_served_by_hex.or_else(|| target.map(|id| hex::encode(id.as_bytes())))?;
    Some(ServingProvenance {
        served_by_node_id,
        hostname: None,
        quantization: None,
        architecture: None,
        context_length: None,
        parameter_size: None,
        layer_count: None,
        model_identity_hash: None,
        model_canonical_ref: None,
        model_revision: None,
        weights_digest: None,
        gpu: None,
        vram_bytes: None,
        is_soc: None,
        // This node routed the exchange; nobody asked it to serve.
        requested_by_node_id: None,
    })
}

/// The per-exchange facts [`publish_raw_proxy_terminal`] attaches beyond the
/// outcome itself.
#[derive(Clone, Copy, Default)]
pub(in crate::network::openai) struct RawProxyTerminalFacts<'a> {
    /// `true` only on the host-served branch (this node's own weights and
    /// hardware survey); see [`publish_raw_proxy_terminal`].
    pub(in crate::network::openai) served_locally: bool,
    /// Canonical digest of the real request body the host dispatched.
    pub(in crate::network::openai) request_digest: Option<&'a str>,
    /// The mesh node that asked, from the tunnel's authenticated remote id
    /// ([`IngressRouteContext::requested_by_node_id`]). Host-served only.
    pub(in crate::network::openai) requested_by_node_id: Option<&'a str>,
    /// The request's stabilized `x-capsule-client-nonce` and who contributed
    /// it ([`request_client_nonce`]), as the `RemoteMesh` terminal reports it.
    /// Host-served only.
    pub(in crate::network::openai) client_nonce: Option<(&'a str, ClientNonceSource)>,
}

/// Publish the raw-proxy path's terminal event for a served exchange, enriched
/// with the serving provenance the host resolved from the node (what ran / at
/// what fidelity / on whose hardware), the real token usage the dispatch
/// outcome carried, and the canonical digest of the real request body. Only
/// real values ride along; anything the node doesn't know for this model, or
/// the dispatch didn't carry, stays omitted (never fabricated). No
/// `X-Capsule-Id` marker exists on this path (it never runs through
/// `skippy-inference-api`'s `InferenceHookPolicy`, the only place a marker is minted),
/// so `capsule_id` is `None`. `nonce`/`nonce_source` come from the request's
/// own stabilized client nonce when the caller passes one (the host-served
/// branch does), so a node serving a routed exchange reports the same nonce as
/// the routing node's `RemoteMesh` terminal; otherwise they are `None`.
///
/// `served_locally` distinguishes the host-served path (this node's own
/// weights, via `route_model_request`) from the plugin-served path (a plugin
/// endpoint that may proxy anywhere, including a third-party service). The
/// whole `serving_provenance` block — not just the hardware fields — is
/// attached only when `served_locally` is true: a plugin endpoint can return
/// 200 for an exchange this node's GPU/VRAM/hostname never touched, and the
/// model-name-keyed descriptor lookup itself is a second, independent source
/// of staleness (the descriptor list and the routing target table update on
/// separate schedules, so a teardown window can hand a plugin-served exchange
/// this node's own served-model quant/architecture/identity_hash). Omitting
/// the whole block on the plugin path closes both windows at once rather than
/// leaving the model-identity half open.
pub(in crate::network::openai) async fn publish_raw_proxy_terminal(
    node: &mesh::Node,
    channel: &dyn OpenAiExchangeChannel,
    exchange_id: &str,
    model_name: &str,
    final_outcome: &proxy::RouteDispatchOutcome,
    facts: RawProxyTerminalFacts<'_>,
) {
    let RawProxyTerminalFacts {
        served_locally,
        request_digest,
        requested_by_node_id,
        client_nonce,
    } = facts;
    let mut envelope = OpenAiExchangeEnvelope::terminal(
        exchange_id.to_string(),
        OpenAiExchangeDispatchPath::RawProxy,
        model_name,
        plugin_route_status(final_outcome),
        None,
        client_nonce.map(|(_, source)| source),
    );
    envelope.nonce = client_nonce.map(|(nonce, _)| nonce.to_string());
    // Nothing was served on a 503 / `Failed` / `Dropped` outcome, so there is
    // no hardware or model identity to report — and no reason to pay the
    // `served_model_descriptors()` lock for a lookup whose result would be
    // thrown away. On the plugin-served path, skip it regardless of outcome:
    // see the `served_locally` doc above.
    if served_locally && outcome_was_served(final_outcome) {
        let provenance = serving_provenance_for_model(node, model_name, requested_by_node_id).await;
        envelope = envelope.with_serving_provenance(provenance);
    }
    // The host-served (real-weights) branch reaches this via
    // `route_model_request`, whose outcome carries the served backend's own
    // `usage` object. Attach it so a downstream plugin can seal the REAL token
    // counts of the exchange, not a zeroed stub. A plugin-served exchange has
    // no such usage on the outcome, so this stays absent for it — never zeroed.
    if let Some(usage) = exchange_usage_from_outcome(final_outcome) {
        envelope = envelope.with_usage(usage);
    }
    // The canonical digest of the REAL request body the host dispatched, so a
    // downstream capsule can bind its `agent_input_digest` to the real bytes.
    // Computed by the caller (which holds the parsed request); absent only when
    // the host held no JSON body to digest — never fabricated.
    if let Some(digest) = request_digest {
        envelope = envelope.with_request_digest(digest.to_string());
    }
    // The digests over the REAL served response (response body / tool_calls /
    // reasoning), computed at the JSON-relay delivery point — or, on a
    // streamed delivery, assembled from the chunks actually sent to the
    // client — and carried up on the dispatch outcome. A no-op for an
    // all-`None` bundle (nothing was captured or assembled), so nothing is
    // fabricated. Unlike `serving_provenance`, this is not gated on
    // `served_locally`/2xx: a digest of what the host actually returned is
    // defined on every outcome that had a body, plugin-served included;
    // absent exactly when there was no body to digest.
    let output_digests = exchange_output_digests_from_outcome(final_outcome);
    if output_digests.has_any() {
        envelope = envelope.with_output_digests(output_digests);
    }
    channel.publish(&envelope).await;
}

/// Lift the response / tool_calls / reasoning digests off a dispatch outcome.
/// Only `RespondedWithUsage` carries them (the outcome the host-served
/// `route_model_request` returns after the JSON-relay delivery point, or the
/// streaming assembler, computed them); every other outcome yields an
/// all-`None` bundle, so the terminal envelope simply omits those digests
/// rather than fabricating any.
pub(super) fn exchange_output_digests_from_outcome(
    outcome: &proxy::RouteDispatchOutcome,
) -> crate::plugin::openai_exchange::ExchangeOutputDigests {
    match outcome {
        proxy::RouteDispatchOutcome::RespondedWithUsage { output_digests, .. }
        | proxy::RouteDispatchOutcome::RespondedWithDigests { output_digests, .. } => {
            *output_digests
        }
        _ => Default::default(),
    }
}

/// Path 2's own "effective request" moment: the plugin/endpoint is resolved
/// and dispatch is about to happen. There is no typed `ChatCompletionRequest`
/// on this path (see the #1331 design note), so the envelope carries only
/// the model — the same narrow route fact path 1's `ChatExchangeRoute`
/// carries. Mints the exchange id here, at admission, so it can pair this
/// effective event with its terminal event even when concurrent raw-proxy
/// requests share the same model. Returns `None` — no id minted, no publish
/// — when nobody declares `openai.exchange.v1`: nothing downstream would
/// ever see it.
pub(super) async fn mint_and_publish_effective_raw_proxy(
    plugin_manager: &crate::plugin::PluginManager,
    model_name: &str,
) -> Option<String> {
    if !plugin_manager.has_subscriber().await {
        return None;
    }
    let exchange_id = uuid::Uuid::new_v4().to_string();
    plugin_manager
        .publish(&OpenAiExchangeEnvelope::effective(
            exchange_id.clone(),
            OpenAiExchangeDispatchPath::RawProxy,
            model_name,
        ))
        .await;
    Some(exchange_id)
}

/// Map a request nonce's origin marker (see
/// [`proxy::BufferedHttpRequest::capsule_nonce_headers`]) to the tri-state a
/// plugin uses to judge trust. An origin marker present means THIS frontend
/// minted the nonce (the client sent none, so a fallback was generated at
/// ingress); its absence means the value came from the client unchanged.
/// `None` exactly when there is no nonce to report at all.
pub(super) fn client_nonce_source(
    nonce: &Option<String>,
    nonce_origin: &Option<String>,
) -> Option<ClientNonceSource> {
    nonce.as_ref().map(|_| {
        if nonce_origin.is_some() {
            ClientNonceSource::SidecarGeneratedFallback
        } else {
            ClientNonceSource::ClientSupplied
        }
    })
}

/// The request's stabilized client nonce with its [`client_nonce_source`],
/// for the host-served terminal. On a node serving a tunnelled request this
/// is the routing node's nonce: ingress keeps a single valid inbound nonce and
/// strips the origin marker, so it reads `ClientSupplied` (the cross-node
/// asymmetry documented on `OpenAiExchangeEnvelope::nonce_source`).
pub(super) fn request_client_nonce(
    request: &proxy::BufferedHttpRequest,
) -> Option<(String, ClientNonceSource)> {
    let (nonce, nonce_origin) = request.capsule_nonce_headers();
    let source = client_nonce_source(&nonce, &nonce_origin)?;
    nonce.map(|nonce| (nonce, source))
}

/// What this node knows about an exchange a peer delivered while this node
/// only routed it.
pub(super) struct RemoteDeliveredFacts<'a> {
    pub(super) exchange_id: String,
    pub(super) model_name: &'a str,
    pub(super) nonce: Option<String>,
    pub(super) nonce_source: Option<ClientNonceSource>,
    pub(super) peer_capsule_id: Option<String>,
    /// An explicit `x-mesh-target`, when the client named the peer.
    pub(super) target: Option<iroh::EndpointId>,
    /// The peer `route_model_request` saw deliver the attempt.
    pub(super) observed_served_by_hex: Option<String>,
    pub(super) request_digest: Option<String>,
    /// The client's `x-mesh-twin-bracket` value, copied unread.
    pub(super) twin_bracket_id: Option<String>,
}

/// The terminal envelope for an exchange a peer delivered: `RemoteMesh`,
/// naming the peer that served it (never this node), with the digest of the
/// request body this node forwarded and the digests over the response bytes
/// it relayed. Both routes that can end at a peer publish it: the remote-mesh
/// branch, and the local-candidates branch when election picked a peer over
/// this node's own copy of the model.
pub(super) fn remote_delivered_terminal(
    facts: RemoteDeliveredFacts<'_>,
    outcome: &proxy::RouteDispatchOutcome,
) -> OpenAiExchangeEnvelope {
    let mut terminal = OpenAiExchangeEnvelope::terminal_remote_mesh(
        facts.exchange_id,
        facts.model_name,
        plugin_route_status(outcome),
        facts.nonce,
        facts.nonce_source,
        facts.peer_capsule_id,
    )
    .with_twin_bracket_id(facts.twin_bracket_id);
    if let Some(provenance) =
        serving_provenance_for_remote_mesh(facts.target, facts.observed_served_by_hex, outcome)
    {
        terminal = terminal.with_serving_provenance(provenance);
    }
    if let Some(digest) = facts.request_digest {
        terminal = terminal.with_request_digest(digest);
    }
    // Digests over the response bytes this node relayed, carried up on the
    // outcome by the relay; an all-`None` bundle (nothing captured) adds
    // nothing.
    let output_digests = exchange_output_digests_from_outcome(outcome);
    if output_digests.has_any() {
        terminal = terminal.with_output_digests(output_digests);
    }
    terminal
}

/// The facts for a local-candidates exchange that election delivered from a
/// peer: the request's nonce and twin bracket, as on the remote-mesh branch,
/// and the peer that delivered it.
pub(super) fn local_route_delivered_by_peer<'a>(
    request: &proxy::BufferedHttpRequest,
    exchange_id: String,
    model_name: &'a str,
    peer_hex: String,
    peer_capsule_id: Option<String>,
    request_digest: Option<String>,
) -> RemoteDeliveredFacts<'a> {
    let (nonce, nonce_origin) = request.capsule_nonce_headers();
    let nonce_source = client_nonce_source(&nonce, &nonce_origin);
    RemoteDeliveredFacts {
        exchange_id,
        model_name,
        nonce,
        nonce_source,
        peer_capsule_id,
        target: None,
        observed_served_by_hex: Some(peer_hex),
        request_digest,
        // Validated in `route_request`; copied unread.
        twin_bracket_id: twin_bracket_id(request).ok().flatten(),
    }
}

/// The terminal for the local-candidates route. Election there may deliver
/// from a peer that also serves the model rather than from this node: then
/// this node only routed the exchange, so it records it as the asking side
/// (`RemoteMesh`, the peer as the server). Recording it as host-served would
/// name this node as the server. Otherwise this node served it.
pub(super) async fn publish_local_route_terminal(
    node: &mesh::Node,
    channel: &dyn OpenAiExchangeChannel,
    exchange_id: &str,
    model_name: &str,
    outcome: &proxy::RouteDispatchOutcome,
    delivered_by_peer: Option<RemoteDeliveredFacts<'_>>,
    host_served: RawProxyTerminalFacts<'_>,
) {
    match delivered_by_peer {
        Some(facts) => {
            channel
                .publish(&remote_delivered_terminal(facts, outcome))
                .await;
        }
        None => {
            publish_raw_proxy_terminal(
                node,
                channel,
                exchange_id,
                model_name,
                outcome,
                host_served,
            )
            .await;
        }
    }
}

impl<'a> IngressRouteContext<'a> {
    /// Where this route publishes its `openai.exchange.v1` events:
    /// `plugin_manager` in production, or the double a test injected.
    pub(super) fn exchange_events_channel(&self) -> Option<&'a dyn OpenAiExchangeChannel> {
        #[cfg(test)]
        if let Some(channel) = self.exchange_channel {
            return Some(channel);
        }
        self.plugin_manager
            .map(|plugin_manager| plugin_manager as &dyn OpenAiExchangeChannel)
    }
}
