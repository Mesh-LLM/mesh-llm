//! Requests naming a model this node does not host-serve: a mesh peer that
//! serves it, a retryable 503 while it loads here, a plugin, or a 404.
use super::*;

/// Route a request whose model is not being served from local candidates
/// (either genuinely absent locally, or forced away from local by
/// `x-mesh-target`/`x-mesh-exclude` -- see `route_request`), trying remote
/// mesh peers, then local-unavailable/plugin fallbacks, then 404.
#[allow(clippy::too_many_arguments)]
pub(super) async fn route_missing_local_model(
    mut tcp_stream: ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    model_name: &str,
    target: Option<iroh::EndpointId>,
    excluded: &[iroh::EndpointId],
    required_tokens: Option<u32>,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    // An explicit self-target can never be found by `resolve_remote_mesh_route`
    // below -- it only searches OTHER peers' advertised hosts -- so it always
    // failed closed with a spurious 409 even when this node serves the model
    // itself via a plugin (ndizazzo P2a, PR #1671 round 2). The caller only
    // reaches this function for a self-target once local HOST-served
    // candidates already came up empty (see `route_request`), so what's left
    // to check here is plugin-served availability.
    if target == Some(ctx.node.id()) {
        return route_self_targeted_model(
            tcp_stream,
            request,
            ctx,
            model_name,
            excluded,
            route_observer,
        )
        .await;
    }

    // No candidate at all for this name (the model has no local target and the
    // election snapshot carries none) is also the shape a plugin-served name
    // takes when a peer advertises the same id for a workload class this
    // request does not want. The remote half below would then find that peer,
    // filter it out on class, and fail the request as having no eligible target
    // even though a local plugin endpoint serves this model for this endpoint.
    // The fallback keeps every routable peer first: it dispatches only when a
    // plugin endpoint matches the model and the request path and no mesh
    // candidate admits the workload. A caller that forced a peer traffic
    // header never reaches this point, so a forced remote stays forced, and an
    // `x-mesh-exclude` naming this node still fails closed below rather than
    // serving the request here.
    // `local_execution_only` is false here by construction: peer inference sets
    // it, and `route_request` answers a forced remote with 409 and a missing
    // candidate with `local_model_unavailable` before it calls this function,
    // so peer ingress never reaches this fallback.
    if target.is_none() && !excluded.contains(&ctx.node.id()) {
        match try_plugin_when_no_mesh_candidate_serves(
            false,
            tcp_stream,
            request,
            ctx,
            model_name,
            excluded,
            route_observer,
        )
        .await
        {
            Ok(outcome) => return outcome,
            Err(stream) => tcp_stream = stream,
        }
    }

    // Try remote mesh first.
    match resolve_remote_mesh_route(ctx, model_name, target, excluded).await {
        RemoteMeshRoute::TargetUnavailable {
            target_hex,
            blocked,
        } => {
            // Fail closed: never substitute another peer for an explicitly
            // named `x-mesh-target` that doesn't (or no longer) serve this
            // model -- that would silently defeat the live-twin check the
            // header exists for. When the operator's own block is the reason
            // the peer is absent, say so rather than implying it never served
            // the model.
            let message = unavailable_target_message(&target_hex, model_name, blocked);
            return response_outcome(
                409,
                proxy::send_error_observed(tcp_stream, 409, &message, route_observer).await,
            );
        }
        RemoteMeshRoute::Targets(mesh_targets) => {
            // This node is routing the exchange to a peer, not serving it --
            // publish the same effective/terminal pair try_route_plugin_model
            // already does for its own dispatch below, with `RemoteMesh` in
            // place of `RawProxy`, so a plugin on the ROUTING node can observe
            // this exchange too (previously it observed nothing at all for a
            // routed exchange). The peer's `X-Capsule-Id` response header IS
            // read back here -- see `PeerCapsuleIdSink` below.
            let exchange_id = uuid::Uuid::new_v4().to_string();
            // The client-contributed capsule nonce, already stabilized (and,
            // if the client sent none, minted) by `finalize_forwarded_request`
            // at ingress -- read back off the already-buffered request rather
            // than minted here: a second fallback minted on THIS node would
            // not match whatever was already stamped into the request this
            // node forwards to the peer byte-for-byte, breaking "same nonce
            // both sides."
            let (forwarded_nonce, nonce_origin) = request.capsule_nonce_headers();
            let nonce_source = client_nonce_source(&forwarded_nonce, &nonce_origin);
            // Validated in `route_request`; copied onto both events unread.
            let twin_bracket = twin_bracket_id(request).ok().flatten();
            let channel = ctx.exchange_events_channel();
            if let Some(ch) = channel {
                ch.publish(
                    &OpenAiExchangeEnvelope::effective_remote_mesh(
                        exchange_id.clone(),
                        model_name,
                        forwarded_nonce.clone(),
                        nonce_source,
                    )
                    .with_twin_bracket_id(twin_bracket.clone()),
                )
                .await;
            }
            // Only echoed when the client asked for a specific peer via
            // `x-mesh-target` -- absent headers must produce today's
            // response byte-for-byte, with no `x-mesh-served-by` added.
            let served_by_hex = target.map(|id| hex::encode(id.as_bytes()));
            // Where `route_model_request` records the peer's `X-Capsule-Id`
            // response header, when it reads one back. See
            // `PeerCapsuleIdSink` -- this is the peer's own UNVERIFIED
            // assertion of its capsule, never elevated to verified here
            // (that happens in whatever later pulls the capsule via a later
            // out-of-band fetch and checks its digest).
            let peer_capsule_id_sink = proxy::PeerCapsuleIdSink::new();
            // Where `route_model_request` records which peer delivered, so the
            // terminal names the server even when the client named no
            // `x-mesh-target`. See `serving_provenance_for_remote_mesh`.
            let served_by_node_id_sink = proxy::ServedByNodeIdSink::new();
            let outcome = proxy::route_model_request(
                ctx.node.clone(),
                tcp_stream,
                &mesh_targets,
                model_name,
                request,
                proxy::RouteModelRequestContext {
                    local_execution_only: false,
                    exchange_id: Some(&exchange_id),
                    required_tokens,
                    affinity: ctx.affinity,
                    route_observer,
                    served_by_header: served_by_hex.as_deref(),
                    peer_capsule_id: Some(&peer_capsule_id_sink),
                    served_by_node_id: Some(&served_by_node_id_sink),
                },
            )
            .await;
            if let Some(ch) = channel {
                // The canonical digest of the request body this node forwarded
                // to the peer, built the same way as on the host-served branch.
                // `ensure_body_json` is idempotent and only paid when there is
                // a channel to publish to.
                request.ensure_body_json();
                let request_digest = request
                    .body_json
                    .as_ref()
                    .and_then(|body| request_body_digest(body, request.body_bytes.as_deref()));
                let terminal = remote_delivered_terminal(
                    RemoteDeliveredFacts {
                        exchange_id,
                        model_name,
                        nonce: forwarded_nonce,
                        nonce_source,
                        peer_capsule_id: peer_capsule_id_sink.take(),
                        target,
                        observed_served_by_hex: served_by_node_id_sink.take(),
                        request_digest,
                        twin_bracket_id: twin_bracket,
                    },
                    &outcome,
                );
                ch.publish(&terminal).await;
            }
            return outcome;
        }
        RemoteMeshRoute::NoRemoteHost => {}
    }

    route_without_a_mesh_host(
        tcp_stream,
        request,
        ctx,
        model_name,
        excluded,
        route_observer,
    )
    .await
}

/// A model no mesh peer serves: a retryable 503 while it loads or drains
/// here, the exclusion 409, a plugin, or a 404.
async fn route_without_a_mesh_host(
    tcp_stream: ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    model_name: &str,
    excluded: &[iroh::EndpointId],
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    // Return 503 while the model loads or drains here so clients can retry;
    // return 404 only when the model truly doesn't exist anywhere.
    if unavailable_locally(ctx, model_name, excluded).await {
        record_pre_routing_rejection(ctx, Some(model_name), 503).await;
        return response_outcome(
            503,
            proxy::send_503_observed(
                tcp_stream,
                &format!("model '{model_name}' is unavailable locally (loading or draining)"),
                route_observer,
            )
            .await,
        );
    }

    // `x-mesh-exclude` naming this node must block LOCAL PLUGIN fallback too
    // -- otherwise the client's explicit "not this node" is honored for
    // host-served candidates (via `mesh_headers_force_remote`, which is why
    // execution reached this function at all) but silently ignored the
    // moment the model turns out to be plugin-served instead, and the
    // excluded node serves the request anyway (ndizazzo P2b, PR #1671
    // round 2). Fail closed with 409, matching `x-mesh-target`'s contract,
    // rather than the generic "not found anywhere" 404.
    if excluded.contains(&ctx.node.id()) {
        return response_outcome(
            409,
            proxy::send_error_observed(
                tcp_stream,
                409,
                &format!(
                    "this node ({}) is excluded via x-mesh-exclude and no other candidate serves model '{model_name}'",
                    hex::encode(ctx.node.id().as_bytes())
                ),
                route_observer,
            )
            .await,
        );
    }

    // Try plugin dispatch (admission-checked inside).
    if ctx.plugin_manager.is_some() {
        return try_route_plugin_model(ctx, tcp_stream, request, model_name, route_observer).await;
    }

    // Model not found anywhere — return 404. This replaces the old fallback behavior that would
    // route to an arbitrary target, which was confusing when no host served this model.
    record_pre_routing_rejection(ctx, Some(model_name), 404).await;
    response_outcome(
        404,
        proxy::send_error_observed(
            tcp_stream,
            404,
            &format!("model '{model_name}' not found (no local or remote host serving this model)"),
            route_observer,
        )
        .await,
    )
}

fn unavailable_target_message(target_hex: &str, model_name: &str, blocked: bool) -> String {
    if blocked {
        format!(
            "x-mesh-target '{target_hex}' is blocked by this operator -- refusing to fall back to another peer"
        )
    } else {
        format!(
            "x-mesh-target '{target_hex}' does not serve model '{model_name}' -- refusing to fall back to another peer"
        )
    }
}

/// Resolve an explicit `x-mesh-target` naming THIS node against local
/// availability -- host-served or plugin-served -- instead of asking
/// `resolve_remote_mesh_route` to find "self" among other peers, which it
/// never will (ndizazzo P2a, PR #1671 round 2). The caller guarantees this
/// node does not currently host-serve `model_name` (see
/// `route_missing_local_model`), so only plugin-served availability remains
/// to check. Answers a retryable 503 while this node is still loading the
/// model. Fails closed with 409 -- the same contract `x-mesh-target` uses
/// for a peer that turns out not to serve the model -- when this node
/// doesn't serve it either, or when `x-mesh-exclude` names this node too (a
/// self-target contradicted by a self-exclude).
pub(super) async fn route_self_targeted_model(
    tcp_stream: ClientStream,
    request: &proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    model_name: &str,
    excluded: &[iroh::EndpointId],
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    let self_id = ctx.node.id();
    if let Some(plugin_manager) = ctx.plugin_manager {
        match plugin_manager
            .inference_endpoint_for_request(model_name, &request.client_path)
            .await
        {
            Err(error) => {
                // Transient resolution failure — degrade gracefully with 503,
                // not 409. A 409 would mis-state the reason: the peer did not
                // refuse to serve the model; we failed to ask it. The caller
                // (client) should retry; the peer is not at fault.
                tracing::warn!(
                    %error,
                    "route_self_targeted_model: failed to resolve plugin endpoint for '{model_name}'"
                );
                record_pre_routing_rejection(ctx, Some(model_name), 503).await;
                return response_outcome(
                    503,
                    proxy::send_503_observed(
                        tcp_stream,
                        &format!(
                            "plugin endpoint for model '{model_name}' unavailable (resolution error)"
                        ),
                        route_observer,
                    )
                    .await,
                );
            }
            Ok(Some(_endpoint)) => {
                // Self is the target and serves the model via plugin.
                if !excluded.contains(&self_id) {
                    return try_route_plugin_model(
                        ctx,
                        tcp_stream,
                        request,
                        model_name,
                        route_observer,
                    )
                    .await;
                }
            }
            Ok(None) => {}
        }
    }
    // This node may still be loading the model: a first load has its serving
    // assignment but no target yet, a reload has all-None candidates. Answer
    // the same retryable 503 as an untargeted request. A self-exclude keeps
    // the 409 below, since retrying would never help that client.
    if !excluded.contains(&self_id)
        && (assigned_here(ctx, model_name).await
            || has_local_unavailable_candidates(ctx.targets, model_name))
    {
        return response_outcome(
            503,
            proxy::send_503_observed(
                tcp_stream,
                &format!("model '{model_name}' is unavailable locally (loading or draining)"),
                route_observer,
            )
            .await,
        );
    }
    response_outcome(
        409,
        proxy::send_error_observed(
            tcp_stream,
            409,
            &format!(
                "x-mesh-target '{}' does not serve model '{model_name}' -- refusing to fall back to another peer",
                hex::encode(self_id.as_bytes())
            ),
            route_observer,
        )
        .await,
    )
}

/// Whether the model is known locally but unavailable, which earns an
/// untargeted request a retryable 503: loading, draining or failed with
/// all-None candidates, or a first load, which has no target yet, only its
/// serving assignment. A client that excluded this node gets the exclusion 409
/// instead, since retrying here would never help it, and a plugin serving the
/// same name keeps the plugin route it had before.
async fn unavailable_locally(
    ctx: &IngressRouteContext<'_>,
    model_name: &str,
    excluded: &[iroh::EndpointId],
) -> bool {
    if has_local_unavailable_candidates(ctx.targets, model_name) {
        return true;
    }
    !excluded.contains(&ctx.node.id())
        && assigned_here(ctx, model_name).await
        && !plugin_serves_model(ctx, model_name).await
}

/// Whether this node's serving assignment names `model_name`. A first load has
/// it before any target exists.
async fn assigned_here(ctx: &IngressRouteContext<'_>, model_name: &str) -> bool {
    ctx.node
        .serving_models()
        .await
        .iter()
        .any(|served| served == model_name)
}

/// Whether a plugin inference endpoint claims `model_name`; a lookup error
/// counts as claimed, as in `local_model_unavailable`.
async fn plugin_serves_model(ctx: &IngressRouteContext<'_>, model_name: &str) -> bool {
    match ctx.plugin_manager {
        Some(manager) => !matches!(
            manager.inference_endpoint_for_model(model_name).await,
            Ok(None)
        ),
        None => false,
    }
}

/// Check whether the model is known locally but currently unavailable — all local candidates are None.
pub(super) fn has_local_unavailable_candidates(
    targets: &election::ModelTargets,
    model_name: &str,
) -> bool {
    let cands = targets.candidates(model_name);
    !cands.is_empty()
        && cands
            .iter()
            .all(|t| matches!(t, election::InferenceTarget::None))
}
