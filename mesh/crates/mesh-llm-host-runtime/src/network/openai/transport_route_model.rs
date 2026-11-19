#[path = "transport_route_exhaustion.rs"]
mod exhaustion;

use super::super::exchange_admission::SelectedRouteAdmission;
use super::*;
use crate::network::openai::routing_rank::{RankedCandidates, rank_targets_by_context};
use crate::network::reservations::RoutingReservation;
use crate::network::router::PlacementWeight;

pub(crate) struct RouteModelRequestContext<'a> {
    /// Peer ingress may execute local targets only, never discover a fallback peer.
    pub(crate) local_execution_only: bool,
    pub(crate) automatic_retry: Option<&'a super::super::automatic_retry::AutomaticRetryHandoff>,
    pub(crate) required_tokens: Option<u32>,
    pub(crate) affinity: &'a AffinityRouter,
    pub(crate) route_observer: OpenAiRouteObserver<'a>,
    /// Hex-encoded `EndpointId` to echo back as `x-mesh-served-by` once the
    /// response is delivered. Set only when the caller (an `x-mesh-target`
    /// forced single-candidate dispatch) already knows the exact peer that
    /// will serve the request; `None` everywhere else, including ordinary
    /// multi-candidate remote-mesh routing.
    pub(crate) served_by_header: Option<&'a str>,
    /// Where to record a peer's `X-Capsule-Id` response header. Set only by
    /// the `RemoteMesh` dispatch path (`ingress::route_missing_local_model`),
    /// which reads it back out after this call to attach it to its own
    /// terminal plugin event; `None` for every other caller.
    pub(crate) peer_capsule_id: Option<&'a PeerCapsuleIdSink>,
    /// Where to record which peer delivered a served attempt. Set by the
    /// ingress paths that publish a terminal exchange event, which read it
    /// back after this call to name the serving node even when the client
    /// sent no `x-mesh-target`; `None` for every other caller.
    pub(crate) served_by_node_id: Option<&'a ServedByNodeIdSink>,
}

/// A delivered 2xx outcome: the routing-side test for "this attempt was
/// served", which gates the `ServedByNodeIdSink` write. Mirrors
/// `ingress::exchange_events::outcome_was_served`, kept here where the
/// per-attempt target is in scope.
fn route_outcome_was_served(outcome: &RouteDispatchOutcome) -> bool {
    matches!(
        outcome,
        RouteDispatchOutcome::Responded(200..=299)
            | RouteDispatchOutcome::RespondedWithDigests {
                status_code: 200..=299,
                ..
            }
            | RouteDispatchOutcome::RespondedWithUsage {
                status_code: 200..=299,
                ..
            }
    )
}

/// Name the peer that delivered this attempt, so the terminal event can carry
/// `served_by_node_id` even when the client sent no `x-mesh-target`. Only for
/// a served outcome from a `Remote` target: never a value for a local serve or
/// a failure.
fn record_served_by_node_id(
    sink: Option<&ServedByNodeIdSink>,
    outcome: &RouteDispatchOutcome,
    target: &election::InferenceTarget,
) {
    if let (Some(sink), true) = (sink, route_outcome_was_served(outcome))
        && let election::InferenceTarget::Remote(endpoint_id) = target
    {
        sink.set(hex::encode(endpoint_id.as_bytes()));
    }
}

pub async fn route_model_request(
    node: mesh::Node,
    tcp_stream: ClientStream,
    targets: &election::ModelTargets,
    model: &str,
    request: &BufferedHttpRequest,
    context: RouteModelRequestContext<'_>,
) -> RouteDispatchOutcome {
    let args = RouteModelRequestArgs {
        node,
        tcp_stream,
        targets,
        model,
        request,
        local_execution_only: context.local_execution_only,
        automatic_retry: context.automatic_retry,
        required_tokens: context.required_tokens,
        affinity: context.affinity,
        route_observer: context.route_observer,
        served_by_header: context.served_by_header,
        peer_capsule_id: context.peer_capsule_id,
        served_by_node_id: context.served_by_node_id,
    };
    route_model_request_inner(args).await
}

struct RouteModelRequestArgs<'a> {
    automatic_retry: Option<&'a super::super::automatic_retry::AutomaticRetryHandoff>,
    local_execution_only: bool,
    node: mesh::Node,
    tcp_stream: ClientStream,
    targets: &'a election::ModelTargets,
    model: &'a str,
    request: &'a BufferedHttpRequest,
    required_tokens: Option<u32>,
    affinity: &'a AffinityRouter,
    route_observer: OpenAiRouteObserver<'a>,
    served_by_header: Option<&'a str>,
    peer_capsule_id: Option<&'a PeerCapsuleIdSink>,
    served_by_node_id: Option<&'a ServedByNodeIdSink>,
}

struct RouteModelState {
    route_started: Instant,
    attempts: usize,
    timeout_attempts: usize,
    refreshed: bool,
    /// Prompt bytes of the request body, to learn the model's bytes per token
    /// from the reply's `usage.prompt_tokens`.
    prompt_bytes: Option<usize>,
}

enum RouteModelDisposition {
    Continue,
    Return(RouteDispatchOutcome),
}

fn no_context_eligible_target_reason(model: &str, required_tokens: Option<u32>) -> String {
    match required_tokens {
        Some(tokens) => format!(
            "no context-compatible target for model '{model}' can fit approximately {tokens} tokens"
        ),
        None => format!("no eligible target for model '{model}'"),
    }
}

async fn cache_target_for_request(
    node: &mesh::Node,
    affinity: &AffinityRouter,
    model: &str,
    affinity_body: Option<&serde_json::Value>,
    candidates: &[election::InferenceTarget],
) -> Option<election::InferenceTarget> {
    let prefix_hash = crate::network::affinity::cache_prefix_hash(affinity_body)?;
    let request = crate::network::affinity::request_cost_input(affinity_body, affinity, model);
    node.cache_target_with_lease(affinity, model, prefix_hash, candidates, request)
        .await
}

pub(super) async fn execution_candidates(
    node: &mesh::Node,
    model: &str,
    request: &BufferedHttpRequest,
    targets: &election::ModelTargets,
    local_execution_only: bool,
) -> Vec<election::InferenceTarget> {
    if local_execution_only {
        let local = targets
            .candidates(model)
            .into_iter()
            .filter(|target| matches!(target, election::InferenceTarget::Local(_)))
            .collect::<Vec<_>>();
        super::super::workload_routing::eligible_targets(node, model, &request.client_path, &local)
            .await
    } else if request.automatic_route
        || request
            .model_name
            .as_deref()
            .is_none_or(super::super::automatic::is_directive)
    {
        // Automatic routing chose the model; the host is drawn over every
        // placement, this node's and its peers', so capacity decides the
        // split instead of whichever node the request happened to reach.
        // The flag, not the name: ingress already rewrote the directive into
        // the resolved model's name before this point.
        automatic_execution_candidates(node, model, request, targets).await
    } else {
        super::super::workload_routing::ingress_candidates(
            node,
            model,
            &request.client_path,
            targets,
        )
        .await
    }
}

/// The automatic host draw, bounded by the client's routing headers. Ingress
/// already applied `x-mesh-target` / `x-mesh-exclude` to `targets`; widening
/// to every peer must not undo that. A target fixes the host, so nothing is
/// added; an excluded peer is never re-added. A malformed header was rejected
/// at ingress, so it cannot reach here, and widens nothing if it does. The
/// widened draw keeps only placements that can see the request's image or
/// audio input.
async fn automatic_execution_candidates(
    node: &mesh::Node,
    model: &str,
    request: &BufferedHttpRequest,
    targets: &election::ModelTargets,
) -> Vec<election::InferenceTarget> {
    let path = &request.client_path;
    let Ok((None, excluded)) = super::super::ingress::parse_mesh_routing_headers(request) else {
        return super::super::workload_routing::eligible_targets(
            node,
            model,
            path,
            &targets.candidates(model),
        )
        .await;
    };
    let mut candidates =
        super::super::workload_routing::mesh_candidates(node, model, path, targets).await;
    candidates.retain(|target| {
        !matches!(target, election::InferenceTarget::Remote(peer) if excluded.contains(peer))
    });
    super::super::workload_routing::media_capable_targets(
        node,
        model,
        path,
        request.body_json.as_ref(),
        candidates,
    )
    .await
}

fn clear_peer_capsule(sink: Option<&PeerCapsuleIdSink>) {
    if let Some(sink) = sink {
        sink.clear();
    }
}

struct PreparedModelCandidates {
    ranked: RankedCandidates<election::InferenceTarget>,
    ordered_candidates: Vec<election::InferenceTarget>,
}

struct ModelRouteRejection {
    status: u16,
    reason: String,
}

impl ModelRouteRejection {
    async fn respond(
        self,
        stream: ClientStream,
        observer: OpenAiRouteObserver<'_>,
    ) -> RouteDispatchOutcome {
        let result = match self.status {
            503 => send_503_observed(stream, &self.reason, observer).await,
            status => send_error_observed(stream, status, &self.reason, observer).await,
        };
        response_outcome(self.status, result)
    }
}

/// Prepare the context/health eligible candidates before affinity and
/// reservation selection. A rejection carries the existing HTTP error boundary.
#[derive(Clone, Copy)]
struct ModelCandidatePolicy {
    local_execution_only: bool,
    strict_health: bool,
}

async fn prepare_model_candidates(
    node: &mesh::Node,
    targets: &election::ModelTargets,
    model: &str,
    request: &BufferedHttpRequest,
    required_tokens: Option<u32>,
    affinity: &AffinityRouter,
    policy: ModelCandidatePolicy,
) -> Result<PreparedModelCandidates, ModelRouteRejection> {
    let candidates =
        execution_candidates(node, model, request, targets, policy.local_execution_only).await;
    let ranked = rank_targets_by_context(node, model, required_tokens, &candidates).await;
    let ordered_candidates =
        route_health_candidates(affinity, model, &ranked.ordered, policy.strict_health);
    if ordered_candidates.is_empty() {
        record_route_model_unavailable(node, model, 0);
        return Err(ModelRouteRejection {
            status: 503,
            reason: no_context_eligible_target_reason(model, required_tokens),
        });
    }
    Ok(PreparedModelCandidates {
        ranked,
        ordered_candidates,
    })
}

async fn route_model_request_inner(args: RouteModelRequestArgs<'_>) -> RouteDispatchOutcome {
    let RouteModelRequestArgs {
        node,
        tcp_stream,
        targets,
        model,
        request,
        local_execution_only,
        automatic_retry,
        required_tokens,
        affinity,
        route_observer,
        served_by_header,
        peer_capsule_id,
        served_by_node_id,
    } = args;
    let route_started = Instant::now();
    let mut tcp_stream = tcp_stream;
    let strict_health = automatic_retry.is_some_and(|handoff| handoff.requires_ready_host());
    let PreparedModelCandidates {
        ranked,
        ordered_candidates,
    } = match prepare_model_candidates(
        &node,
        targets,
        model,
        request,
        required_tokens,
        affinity,
        ModelCandidatePolicy {
            local_execution_only,
            strict_health,
        },
    )
    .await
    {
        Ok(prepared) => prepared,
        Err(rejection) => {
            if let Some(handoff) = automatic_retry {
                handoff.retain(tcp_stream);
                return RouteDispatchOutcome::Failed("automatic_model_exhausted");
            }
            return rejection.respond(tcp_stream, route_observer).await;
        }
    };
    route_observer.route_selected(Some(model));

    let affinity_body = super::super::workload_routing::affinity_body(request);
    let cache_target =
        cache_target_for_request(&node, affinity, model, affinity_body, &ordered_candidates).await;
    let deprioritized = node.deprioritized_peer_ids().await;
    let Some(ReservedModelRoute {
        selection,
        ordered,
        mut reservation,
    }) = select_and_reserve_model_route_with_health(
        targets,
        &ranked,
        model,
        affinity_body,
        affinity,
        cache_target,
        strict_health,
        &deprioritized,
    )
    else {
        if let Some(handoff) = automatic_retry {
            handoff.retain(tcp_stream);
            return RouteDispatchOutcome::Failed("automatic_model_exhausted");
        }
        return send_route_model_none_target(&node, tcp_stream, model, route_observer).await;
    };
    let total_targets = ordered.len();
    let mut state = RouteModelState {
        route_started,
        attempts: 0,
        timeout_attempts: 0,
        refreshed: false,
        prompt_bytes: crate::network::affinity::prompt_bytes(affinity_body),
    };
    // `request.raw` was already stabilized at ingress (finalize_forwarded_request
    // resolves the capsule client nonce once, before target selection), so every
    // attempt here — including a timeout retry to a different target — forwards
    // the identical nonce instead of letting each target's frontend mint its own.
    let forwarding_raw = request.raw.as_slice();
    for (idx, target) in ordered.into_iter().enumerate() {
        // A prior attempt in this same loop may have probed a response,
        // recorded ITS peer's `X-Capsule-Id` into the sink, and then still
        // been classified retryable (e.g. context overflow) -- drain that
        // stale value before moving on, or a retry that never sees (or
        // never sets) the header would silently inherit the PREVIOUS
        // peer's capsule_id and misattribute it to this attempt.
        clear_peer_capsule(peer_capsule_id);
        reservation.transfer_to(&target);
        let attempt_started = Instant::now();
        let mut retry_policy = ResponseRetryPolicy::next_target_available(
            idx + 1 < total_targets || automatic_retry.is_some(),
        )
        .with_caller_completion_cap(request.completion_tokens);
        // An answer the last host produced is terminal even for an automatic
        // directive: the model spill covers uncommitted exhaustion (busy,
        // unavailable, 402, context overflow), not a delivered response the
        // quality probe dislikes. A `finish_reason: length` reply is the
        // client's own `max_tokens` at work and recurs on every model, so
        // rejecting it here turned a served answer into "all eligible
        // automatic models exhausted" (CI smoke, max_tokens 4, 2026-11-08).
        retry_policy.response_quality = idx + 1 < total_targets;
        match super::super::exchange_admission::admit_selected_candidate(
            &node,
            &mut tcp_stream,
            request,
            Some(model),
            "mesh",
            super::super::exchange_admission::SelectedRouteTarget::Mesh(&target),
            idx + 1,
            idx + 1 < total_targets,
        )
        .await
        {
            SelectedRouteAdmission::Proceed => {}
            // A plugin said no to this peer, not to the request: try the next.
            SelectedRouteAdmission::NextCandidate => continue,
            SelectedRouteAdmission::Rejected(outcome) => return outcome,
        }
        // Only a contacted candidate is an attempt: a skipped one must not
        // turn an all-timeouts route into a 503 or inflate attempt metrics.
        state.attempts += 1;
        let attempt_result = route_attempt_for_target(
            &node,
            &mut tcp_stream,
            &target,
            Some(model),
            forwarding_raw,
            retry_policy,
            RouteAttemptLoggingContext {
                request_id: request.request_id,
                retry_policy,
                response_adapter: request.response_adapter,
                route_observer,
                served_by: served_by_header,
                peer_capsule_id,
            },
        )
        .await;
        let queue_wait = attempt_started.duration_since(route_started);
        let attempt_time = attempt_started.elapsed();
        record_route_model_attempt(
            &node,
            model,
            &target,
            queue_wait,
            attempt_time,
            &attempt_result,
        );
        affinity.record_target_outcome(
            Some(model),
            &target,
            target_health_outcome_for_attempt(&attempt_result),
        );
        tracing::info!(
            model = model,
            target = ?target,
            attempt = state.attempts,
            total_targets = total_targets,
            outcome = route_attempt_result_label(&attempt_result),
            attempt_ms = attempt_started.elapsed().as_millis(),
            total_route_ms = route_started.elapsed().as_millis(),
            "openai route_model_request attempt"
        );
        match handle_route_model_attempt_result(
            &node,
            model,
            &target,
            &selection,
            affinity,
            attempt_result,
            &mut state,
        ) {
            RouteModelDisposition::Continue => continue,
            RouteModelDisposition::Return(result) => {
                record_served_by_node_id(served_by_node_id, &result, &target);
                return finalize_route_model_result(
                    &node,
                    model,
                    request,
                    route_started,
                    state.attempts,
                    result,
                    &target,
                );
            }
        }
    }

    if state.timeout_attempts == 0
        && let Some(handoff) = automatic_retry
    {
        handoff.retain(tcp_stream);
        return RouteDispatchOutcome::Failed("automatic_model_exhausted");
    }
    exhaustion::finish_exhausted_route_model_request(
        &node,
        tcp_stream,
        model,
        total_targets,
        &state,
        route_observer,
    )
    .await
}

struct ReservedModelRoute {
    selection: TargetSelection,
    ordered: Vec<election::InferenceTarget>,
    reservation: RoutingReservation,
}

fn route_health_candidates(
    affinity: &AffinityRouter,
    model: &str,
    candidates: &[election::InferenceTarget],
    strict: bool,
) -> Vec<election::InferenceTarget> {
    if strict {
        affinity.route_strict_eligible_candidates(model, candidates)
    } else {
        affinity.route_eligible_candidates(model, candidates)
    }
}

#[cfg(test)]
fn select_and_reserve_model_route(
    targets: &election::ModelTargets,
    ranked: &RankedCandidates<election::InferenceTarget>,
    model: &str,
    parsed_body: Option<&serde_json::Value>,
    affinity: &AffinityRouter,
    cache_target: Option<election::InferenceTarget>,
) -> Option<ReservedModelRoute> {
    select_and_reserve_model_route_with_health(
        targets,
        ranked,
        model,
        parsed_body,
        affinity,
        cache_target,
        false,
        &Default::default(),
    )
}

#[allow(clippy::too_many_arguments)]
fn select_and_reserve_model_route_with_health(
    targets: &election::ModelTargets,
    ranked: &RankedCandidates<election::InferenceTarget>,
    model: &str,
    parsed_body: Option<&serde_json::Value>,
    affinity: &AffinityRouter,
    cache_target: Option<election::InferenceTarget>,
    strict_health: bool,
    deprioritized: &std::collections::HashSet<iroh::EndpointId>,
) -> Option<ReservedModelRoute> {
    // Cache lookup can await while another request cools a target. Refresh
    // health once here, then use exactly this snapshot for selection,
    // reservation spreading, and retries. A second filter inside selection
    // would let the reservation undo a health decision it never saw.
    let mut ordered = route_health_candidates(affinity, model, &ranked.ordered, strict_health);
    let mut selection = crate::network::affinity::select_model_target_from_eligible_candidates(
        targets,
        &ordered,
        parsed_body,
        affinity,
        cache_target,
    );
    if matches!(selection.target, election::InferenceTarget::None) {
        return None;
    }
    // No affinity: replace the round-robin default with the two-choice draw
    // (capacity × locality weighted, fewer local in-flight wins) so a faster
    // or nearer replica carries proportionally more of the new sessions. The
    // draw reserves its pick itself; the reservation must not re-spread it by
    // raw in-flight count.
    let drawn = if selection.affinity_applied {
        None
    } else {
        affinity.reserve_two_choice_target(
            model,
            &ordered,
            |target| {
                matches!(target, election::InferenceTarget::Remote(peer) if deprioritized.contains(peer))
            },
            |target| {
                ranked
                    .placements
                    .get(target)
                    .copied()
                    .unwrap_or_else(PlacementWeight::unmeasured)
            },
        )
    };
    let (target, reservation) = match drawn {
        Some(drawn) => drawn,
        None => {
            // Health policy can remove or reorder candidates, so the original
            // equivalent run can shrink or disappear. Never let its old length
            // admit a lower-ranked fallback.
            let spread_limit = ordered
                .iter()
                .take_while(|candidate| {
                    ranked.ordered[..ranked.equivalent_prefix].contains(candidate)
                })
                .count();
            affinity.reserve_route(
                model,
                &ordered,
                spread_limit,
                &selection.target,
                selection.affinity_applied,
            )?
        }
    };
    selection.target = target;
    move_target_first(&mut ordered, &selection.target);
    Some(ReservedModelRoute {
        selection,
        ordered,
        reservation,
    })
}

fn record_route_model_unavailable(node: &mesh::Node, model: &str, attempts: usize) {
    node.record_routed_request(
        Some(model),
        attempts,
        crate::network::metrics::RequestOutcome::Unavailable,
    );
}

async fn send_route_model_none_target(
    node: &mesh::Node,
    tcp_stream: ClientStream,
    model: &str,
    route_observer: OpenAiRouteObserver<'_>,
) -> RouteDispatchOutcome {
    record_route_model_unavailable(node, model, 0);
    let result = send_503_observed(
        tcp_stream,
        &format!("target for model '{model}' resolved to None (election in progress or host down)"),
        route_observer,
    )
    .await;
    response_outcome(503, result)
}

fn handle_route_model_attempt_result(
    node: &mesh::Node,
    model: &str,
    target: &election::InferenceTarget,
    selection: &TargetSelection,
    affinity: &AffinityRouter,
    attempt_result: RouteAttemptResult,
    state: &mut RouteModelState,
) -> RouteModelDisposition {
    match attempt_result {
        RouteAttemptResult::Delivered {
            status_code,
            usage,
            cache_cost,
            output_digests,
        } => handle_delivered_route_model_attempt(
            DeliveredRouteModelContext {
                node,
                model,
                target,
                selection,
                affinity,
                state,
            },
            status_code,
            usage,
            cache_cost,
            output_digests,
        ),
        RouteAttemptResult::RetryableContextOverflow => {
            handle_retryable_route_model_context(target)
        }
        RouteAttemptResult::RetryableShedding { .. } => RouteModelDisposition::Continue,
        RouteAttemptResult::RetryableGateUnsatisfied => {
            tracing::info!("Target {target:?} answered 402 (gate unsatisfied), trying next");
            RouteModelDisposition::Continue
        }
        RouteAttemptResult::RetryableResponseQuality(failure) => {
            handle_retryable_route_model_response_quality(target, failure)
        }
        RouteAttemptResult::RetryableTimeout => {
            handle_retryable_route_model_timeout(node, target, state)
        }
        RouteAttemptResult::RetryableUnavailable
        | RouteAttemptResult::RetryableUpstreamUnavailable => {
            handle_retryable_route_model_unavailable(node, target, state)
        }
        RouteAttemptResult::CommittedStreamFailure { status_code, cause } => {
            // Committed, so terminal here; but the inference failed, so the
            // request counts as failed, not as the success its 200 head says.
            record_route_model_unavailable(node, model, state.attempts);
            RouteModelDisposition::Return(RouteDispatchOutcome::FailedWithStatus {
                status_code,
                reason: cause.reason(),
            })
        }
        RouteAttemptResult::ClientDisconnected => {
            tracing::info!(
                model = model,
                attempts = state.attempts,
                route_ms = state.route_started.elapsed().as_millis(),
                "openai route_model_request downstream disconnected"
            );
            RouteModelDisposition::Return(RouteDispatchOutcome::Dropped("client_disconnected"))
        }
    }
}

struct DeliveredRouteModelContext<'a> {
    node: &'a mesh::Node,
    model: &'a str,
    target: &'a election::InferenceTarget,
    selection: &'a TargetSelection,
    affinity: &'a AffinityRouter,
    state: &'a RouteModelState,
}

fn handle_delivered_route_model_attempt(
    context: DeliveredRouteModelContext<'_>,
    status_code: u16,
    usage: Option<TokenUsage>,
    cache_cost: Option<CacheCostObservation>,
    output_digests: crate::plugin::openai_exchange::ExchangeOutputDigests,
) -> RouteModelDisposition {
    update_local_cache_evidence(&context, status_code, usage.as_ref(), cache_cost);
    if (200..300).contains(&status_code)
        && let Some((bytes, tokens)) = context
            .state
            .prompt_bytes
            .zip(usage.as_ref().and_then(|usage| usage.prompt_tokens))
    {
        context
            .affinity
            .observe_prompt_size(context.model, bytes, tokens);
    }
    context.node.record_routed_request(
        Some(context.model),
        context.state.attempts,
        request_outcome_for_status(status_code, request_service_for_target(context.target)),
    );
    tracing::info!(
        model = context.model,
        attempts = context.state.attempts,
        status_code = status_code,
        route_ms = context.state.route_started.elapsed().as_millis(),
        "openai route_model_request delivered"
    );
    RouteModelDisposition::Return(delivered_outcome(status_code, usage, output_digests))
}

fn update_local_cache_evidence(
    context: &DeliveredRouteModelContext<'_>,
    status_code: u16,
    usage: Option<&TokenUsage>,
    cache_cost: Option<CacheCostObservation>,
) {
    if !(200..400).contains(&status_code) {
        return;
    }
    let is_local = matches!(context.target, election::InferenceTarget::Local(_));
    if is_local {
        context.node.observe_local_prefill_cost(
            context.model,
            cache_cost.and_then(|cost| cost.prefill_micros_per_token),
        );
    }

    let Some(prefix_hash) = context.selection.prefix_hash else {
        return;
    };
    let Some(cached_tokens) = usage.and_then(|usage| usage.cached_prompt_tokens) else {
        return;
    };
    if cached_tokens > 0 {
        if is_local {
            let suffix = usage
                .and_then(|usage| usage.prompt_tokens)
                .unwrap_or(cached_tokens)
                .saturating_sub(cached_tokens);
            context.node.record_local_cache_hit_with_cost(
                context.model,
                prefix_hash,
                u32::try_from(cached_tokens).unwrap_or(u32::MAX),
                u32::try_from(suffix).unwrap_or(u32::MAX),
                crate::network::affinity::LocalCacheCost {
                    queue_delay_micros: cache_cost.map_or(0, |cost| cost.queue_delay_micros),
                    restore_micros: cache_cost.map_or(0, |cost| cost.restore_micros),
                    // The model-level observation was recorded above so each
                    // response contributes exactly once to its EWMA.
                    prefill_micros_per_token: None,
                },
            );
        }
        return;
    }

    let inventory_invalidated = is_local
        && context
            .node
            .invalidate_local_cache_evidence(context.model, prefix_hash);
    let lease_invalidated = context
        .affinity
        .forget_cache_lease(context.model, prefix_hash);
    if inventory_invalidated || lease_invalidated {
        tracing::debug!(
            model = context.model,
            prefix_hash,
            inventory_invalidated,
            lease_invalidated,
            "invalidated cache affinity after authoritative miss"
        );
    }
}

fn handle_retryable_route_model_context(
    target: &election::InferenceTarget,
) -> RouteModelDisposition {
    tracing::warn!(
        "Target {target:?} rejected request with context overflow-style 400, trying next"
    );
    RouteModelDisposition::Continue
}

fn handle_retryable_route_model_response_quality(
    target: &election::InferenceTarget,
    failure: ResponseQualityFailure,
) -> RouteModelDisposition {
    tracing::warn!(
        reason = failure.label(),
        "Target {target:?} returned low-quality success response, trying next"
    );
    RouteModelDisposition::Continue
}

fn handle_retryable_route_model_timeout(
    node: &mesh::Node,
    target: &election::InferenceTarget,
    state: &mut RouteModelState,
) -> RouteModelDisposition {
    state.timeout_attempts += 1;
    spawn_mesh_refresh_once(node, &mut state.refreshed);
    tracing::warn!("Target {target:?} timed out, trying next");
    RouteModelDisposition::Continue
}

fn handle_retryable_route_model_unavailable(
    node: &mesh::Node,
    target: &election::InferenceTarget,
    state: &mut RouteModelState,
) -> RouteModelDisposition {
    spawn_mesh_refresh_once(node, &mut state.refreshed);
    tracing::warn!("Target {target:?} unavailable, trying next");
    RouteModelDisposition::Continue
}

pub(crate) fn finalize_route_model_result(
    node: &mesh::Node,
    model: &str,
    _request: &BufferedHttpRequest,
    _route_started: Instant,
    _attempts: usize,
    result: RouteDispatchOutcome,
    target: &election::InferenceTarget,
) -> RouteDispatchOutcome {
    if let RouteDispatchOutcome::RespondedWithUsage {
        status_code, usage, ..
    } = result
    {
        node.record_prompt_shape(
            Some(model),
            usage.prompt_tokens,
            usage.completion_tokens,
            request_outcome_for_status(status_code, request_service_for_target(target)),
        );
    }
    result
}

fn record_route_model_attempt(
    node: &mesh::Node,
    model: &str,
    target: &election::InferenceTarget,
    queue_wait: Duration,
    attempt_time: Duration,
    attempt_result: &RouteAttemptResult,
) {
    node.record_inference_attempt_report(
        Some(model),
        target,
        attempt_report_for_result(attempt_result, queue_wait, attempt_time, false),
    );
}

#[cfg(test)]
#[path = "transport_route_model_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "transport_route_admission_tests.rs"]
mod admission_tests;
