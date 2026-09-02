use super::*;

pub(crate) struct RouteModelRequestContext<'a> {
    pub(crate) required_tokens: Option<u32>,
    pub(crate) affinity: &'a AffinityRouter,
    pub(crate) route_observer: OpenAiRouteObserver<'a>,
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
        required_tokens: context.required_tokens,
        affinity: context.affinity,
        route_observer: context.route_observer,
    };
    route_model_request_inner(args).await
}

struct RouteModelRequestArgs<'a> {
    node: mesh::Node,
    tcp_stream: ClientStream,
    targets: &'a election::ModelTargets,
    model: &'a str,
    request: &'a BufferedHttpRequest,
    required_tokens: Option<u32>,
    affinity: &'a AffinityRouter,
    route_observer: OpenAiRouteObserver<'a>,
}

struct RouteModelState {
    route_started: Instant,
    attempts: usize,
    refreshed: bool,
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
    prefix_hash: Option<u64>,
    candidates: &[election::InferenceTarget],
) -> Option<election::InferenceTarget> {
    let prefix_hash = prefix_hash?;
    if let Some(target) = affinity.lookup_cache_lease(model, prefix_hash, candidates) {
        return Some(target);
    }

    let selected = node
        .select_cache_target(model, prefix_hash, candidates)
        .await;
    if let Some(target) = selected.as_ref() {
        affinity.remember_cache_lease(model, prefix_hash, target);
    }
    selected
}

async fn route_model_request_inner(args: RouteModelRequestArgs<'_>) -> RouteDispatchOutcome {
    let RouteModelRequestArgs {
        node,
        tcp_stream,
        targets,
        model,
        request,
        required_tokens,
        affinity,
        route_observer,
    } = args;
    let route_started = Instant::now();
    let mut tcp_stream = tcp_stream;
    let ordered_candidates =
        order_targets_by_context(&node, model, required_tokens, &targets.candidates(model)).await;
    let ordered_candidates = affinity.route_eligible_candidates(model, &ordered_candidates);
    if ordered_candidates.is_empty() {
        record_route_model_unavailable(&node, model, 0);
        let reason = no_context_eligible_target_reason(model, required_tokens);
        return response_outcome(
            503,
            send_503_observed(tcp_stream, &reason, route_observer).await,
        );
    }
    route_observer.route_selected(Some(model));

    let prefix_hash = crate::network::affinity::cache_prefix_hash(request.body_json.as_ref());
    let cache_target =
        cache_target_for_request(&node, affinity, model, prefix_hash, &ordered_candidates).await;
    let selection = crate::network::affinity::select_model_target_from_candidates(
        targets,
        &ordered_candidates,
        model,
        request.body_json.as_ref(),
        affinity,
        cache_target,
    );
    if matches!(selection.target, election::InferenceTarget::None) {
        return send_route_model_none_target(&node, tcp_stream, model, route_observer).await;
    }
    let mut ordered = ordered_candidates;
    move_target_first(&mut ordered, &selection.target);
    let total_targets = ordered.len();
    let mut state = RouteModelState {
        route_started,
        attempts: 0,
        refreshed: false,
    };
    for (idx, target) in ordered.into_iter().enumerate() {
        state.attempts += 1;
        let attempt_started = Instant::now();
        let retry_policy = ResponseRetryPolicy::next_target_available(idx + 1 < total_targets);
        let attempt_result = route_attempt_for_target(
            &node,
            &mut tcp_stream,
            &target,
            &request.raw,
            retry_policy,
            RouteAttemptLoggingContext {
                request_id: request.request_id,
                retry_policy,
                response_adapter: request.response_adapter,
                route_observer,
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
            attempt_result,
            &mut state,
        ) {
            RouteModelDisposition::Continue => continue,
            RouteModelDisposition::Return(result) => {
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

    finish_exhausted_route_model_request(
        &node,
        tcp_stream,
        model,
        total_targets,
        &state,
        route_observer,
    )
    .await
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

async fn finish_exhausted_route_model_request(
    node: &mesh::Node,
    tcp_stream: ClientStream,
    model: &str,
    total_targets: usize,
    state: &RouteModelState,
    route_observer: OpenAiRouteObserver<'_>,
) -> RouteDispatchOutcome {
    let result = send_503_observed(
        tcp_stream,
        &format!("all {} target(s) for model '{model}' failed", total_targets),
        route_observer,
    )
    .await;
    record_route_model_unavailable(node, model, state.attempts);
    tracing::warn!(
        model = model,
        attempts = state.attempts,
        route_ms = state.route_started.elapsed().as_millis(),
        "openai route_model_request exhausted targets"
    );
    response_outcome(503, result)
}

fn handle_route_model_attempt_result(
    node: &mesh::Node,
    model: &str,
    target: &election::InferenceTarget,
    selection: &TargetSelection,
    attempt_result: RouteAttemptResult,
    state: &mut RouteModelState,
) -> RouteModelDisposition {
    match attempt_result {
        RouteAttemptResult::Delivered { status_code, usage } => {
            handle_delivered_route_model_attempt(
                DeliveredRouteModelContext {
                    node,
                    model,
                    target,
                    selection,
                    state,
                },
                status_code,
                usage,
            )
        }
        RouteAttemptResult::RetryableContextOverflow => {
            handle_retryable_route_model_context(target)
        }
        RouteAttemptResult::RetryableResponseQuality(failure) => {
            handle_retryable_route_model_response_quality(target, failure)
        }
        RouteAttemptResult::RetryableTimeout => {
            handle_retryable_route_model_timeout(node, target, state)
        }
        RouteAttemptResult::RetryableUnavailable => {
            handle_retryable_route_model_unavailable(node, target, state)
        }
        RouteAttemptResult::CommittedStreamFailure { status_code } => {
            RouteModelDisposition::Return(RouteDispatchOutcome::FailedWithStatus {
                status_code,
                reason: "upstream_stream_incomplete",
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
    state: &'a RouteModelState,
}

fn handle_delivered_route_model_attempt(
    context: DeliveredRouteModelContext<'_>,
    status_code: u16,
    usage: Option<TokenUsage>,
) -> RouteModelDisposition {
    if (200..400).contains(&status_code)
        && let (Some(prefix_hash), election::InferenceTarget::Local(_), Some(usage)) = (
            context.selection.prefix_hash,
            context.target,
            usage.as_ref(),
        )
        && let Some(cached_tokens) = usage.cached_prompt_tokens.filter(|count| *count > 0)
    {
        let suffix = usage
            .prompt_tokens
            .unwrap_or(cached_tokens)
            .saturating_sub(cached_tokens);
        context.node.record_local_cache_hit(
            context.model,
            prefix_hash,
            u32::try_from(cached_tokens).unwrap_or(u32::MAX),
            u32::try_from(suffix).unwrap_or(u32::MAX),
            0,
        );
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
    RouteModelDisposition::Return(
        usage.map_or(RouteDispatchOutcome::Responded(status_code), |usage| {
            RouteDispatchOutcome::RespondedWithUsage { status_code, usage }
        }),
    )
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
    if let RouteDispatchOutcome::RespondedWithUsage { status_code, usage } = result {
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
    if matches!(attempt_result, RouteAttemptResult::ClientDisconnected) {
        return;
    }
    node.record_inference_attempt(
        Some(model),
        target,
        queue_wait,
        attempt_time,
        attempt_outcome_for_result(attempt_result),
        completion_tokens_for_result(attempt_result),
    );
}
