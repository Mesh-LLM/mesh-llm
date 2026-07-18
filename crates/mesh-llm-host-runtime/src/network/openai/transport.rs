//! HTTP proxy plumbing — request parsing, model routing, response helpers.
//!
//! Used by the API proxy (port 9337), bootstrap proxy, and passive mode.
//! All inference traffic flows through these functions.

use crate::inference::election;
use crate::mesh;
use crate::network::affinity::{
    AffinityRouter, PreparedTargets, TargetSelection, prepare_remote_targets_for_request,
};
use crate::network::openai::auto_route;
use crate::network::openai::response_quality::ResponseQualityFailure;
use crate::network::router;
use std::time::{Duration, Instant};
use tokio::net::TcpStream;

pub use super::request_normalize::{ResponseAdapter, release_request_objects};
pub use super::request_parse::{
    BufferedHttpRequest, inject_mesh_hooks_flag, is_drop_request, is_models_list_request,
    read_http_request, read_http_request_with_plugin_manager, rewrite_model_field,
    rewrite_public_model_alias,
};
pub(crate) use super::response::{
    PipelineProxyResult, append_safe_header, pipeline_proxy_local, send_400, send_503, send_error,
    send_json_ok, send_json_ok_with_headers, send_json_with_status_and_headers,
    send_models_list_with_descriptors,
};
pub(crate) use super::routing_rank::{
    capabilities_for_model, descriptor_metadata_for_model, request_budget_tokens_from_parts,
};

use super::response::{
    ResponseRetryPolicy, RouteAttemptResult, attempt_outcome_for_result,
    completion_tokens_for_result, request_outcome_for_status, request_service_for_target,
    route_attempt_result_label, route_http_endpoint_attempt, route_local_attempt,
    route_remote_attempt, target_health_outcome_for_attempt,
};
use super::routing_rank::{
    cached_auto_model_satisfies_media_requirements, move_target_first,
    order_remote_hosts_by_context, order_targets_by_context,
};

const REMOTE_UNCOMMITTED_RETRIES: usize = 1;

enum AutoModelResolution {
    Model(Option<String>),
    UnsupportedMedia,
}

enum MeshTargetResolution {
    Hosts(Vec<iroh::EndpointId>),
    ModelUnavailable(String),
    NoHostsAvailable,
}

struct MeshRequestPlan {
    effective_model: Option<String>,
    auto_session_key: Option<u64>,
    prepared: PreparedTargets,
    target_hosts: Vec<iroh::EndpointId>,
}

enum MeshRequestFailure {
    UnsupportedMedia,
    ModelUnavailable(String),
    NoHostsAvailable,
}

struct MeshAttemptState {
    route_started: Instant,
    attempts: usize,
    last_retryable: bool,
    refreshed: bool,
}

enum MeshAttemptDisposition {
    Continue,
    Return,
}

fn should_learn_affinity(status_code: u16) -> bool {
    (200..400).contains(&status_code)
}

fn capture_path_for_request(request: &BufferedHttpRequest) -> &str {
    &request.client_path
}

// ── Model-aware tunnel routing ──

/// The common request-handling path used by idle proxy, passive proxy, and bootstrap proxy.
///
/// Peeks at the HTTP request, handles `/v1/models`, resolves the target host
/// by model name (or falls back to any host), and tunnels the request via QUIC.
///
/// Set `track_demand` to record requests for demand-based rebalancing.
pub async fn handle_mesh_request(
    node: mesh::Node,
    tcp_stream: TcpStream,
    track_demand: bool,
    affinity: AffinityRouter,
) {
    let mut tcp_stream = tcp_stream;
    let source_addr = tcp_stream.peer_addr().ok();
    let plugin_manager = node.plugin_manager().await;
    let mut request =
        match read_http_request_with_plugin_manager(&mut tcp_stream, plugin_manager.as_ref()).await
        {
            Ok(v) => v,
            Err(err) => {
                let _ = send_400(tcp_stream, &err.to_string()).await;
                return;
            }
        };
    if node.swarm_capture_enabled() {
        node.capture_http_request(crate::mesh::HttpCaptureEvent {
            event: "openai_ingress_http_request",
            source_addr,
            method: &request.method,
            path: capture_path_for_request(&request),
            body_len_bytes: request.body_len_bytes,
            model_name: request.model_name.as_deref(),
            completion_tokens: request.completion_tokens,
            stream: request.stream,
        });
    }

    // Handle /v1/models
    if is_models_list_request(&request.method, &request.path) {
        let served = node.models_being_served().await;
        let descriptors = node.all_served_model_descriptors().await;
        let runtimes = node.all_model_runtime_descriptors().await;
        let _ =
            send_models_list_with_descriptors(tcp_stream, &served, &descriptors, &runtimes).await;
        return;
    }

    // MoA routing directive: `model: "mesh"` triggers mixture-of-agents
    // fan-out. Orchestration happens here, regardless of whether this node
    // is serving models locally — the worker pool is built from gossip.
    // On a pure --client node every backend is remote (QUIC tunnels to
    // peers serving each model); on a host node the locally-served model
    // is wired directly to its skippy port via the targets table.
    //
    // try_handle_moa self-gates on the model name and returns the stream
    // back unchanged if this isn't a MoA request, so we can call it
    // unconditionally here.
    let moa_model_name = request.model_name.clone();
    let moa_required_tokens =
        request_budget_tokens_from_parts(request.body_len_bytes, request.completion_tokens);
    let tcp_stream = match crate::network::openai::moa_gateway::try_handle_moa(
        &node,
        tcp_stream,
        &mut request,
        moa_model_name.as_deref(),
        None, // passive path has no local targets table
        moa_required_tokens,
    )
    .await
    {
        Some(stream) => stream,
        None => {
            // MoA handled the request and consumed the stream.
            release_request_objects(&node, &request.request_object_request_ids).await;
            return;
        }
    };

    let plan = match build_mesh_request_plan(&node, &mut request, track_demand, &affinity).await {
        Ok(plan) => plan,
        Err(failure) => {
            handle_mesh_request_failure(&node, tcp_stream, &request, failure).await;
            return;
        }
    };
    if let Some(tcp_stream) =
        route_mesh_request_attempts(&node, tcp_stream, &request, &plan, &affinity).await
    {
        finish_exhausted_mesh_request(
            &node,
            tcp_stream,
            plan.effective_model.as_deref(),
            plan.target_hosts.len(),
            &affinity,
        )
        .await;
    }
    release_request_objects(&node, &request.request_object_request_ids).await;
}

async fn build_mesh_request_plan(
    node: &mesh::Node,
    request: &mut BufferedHttpRequest,
    track_demand: bool,
    affinity: &AffinityRouter,
) -> std::result::Result<MeshRequestPlan, MeshRequestFailure> {
    let served = node.models_being_served().await;
    let descriptors = node.all_served_model_descriptors().await;
    rewrite_public_model_alias(request, &served, &descriptors);

    let is_auto_request =
        request.model_name.is_none() || request.model_name.as_deref() == Some("auto");
    let auto_session_key = auto_session_key_for_request(request, is_auto_request);
    let required_tokens =
        request_budget_tokens_from_parts(request.body_len_bytes, request.completion_tokens);
    let effective_model = match resolve_auto_model_request(AutoModelRequestArgs {
        node,
        request,
        served: &served,
        descriptors: &descriptors,
        is_auto_request,
        auto_session_key,
        required_tokens,
        affinity,
    })
    .await
    {
        AutoModelResolution::Model(model) => model.or(request.model_name.clone()),
        AutoModelResolution::UnsupportedMedia => return Err(MeshRequestFailure::UnsupportedMedia),
    };
    rewrite_effective_model(request, effective_model.as_deref());
    if is_auto_request {
        inject_mesh_hooks_flag(&mut request.raw, true);
    }
    if track_demand && let Some(name) = effective_model.as_deref() {
        node.record_request(name);
    }

    let resolved_hosts = match resolve_mesh_target_hosts(node, effective_model.as_deref()).await {
        MeshTargetResolution::Hosts(hosts) => hosts,
        MeshTargetResolution::ModelUnavailable(model) => {
            return Err(MeshRequestFailure::ModelUnavailable(model));
        }
        MeshTargetResolution::NoHostsAvailable => return Err(MeshRequestFailure::NoHostsAvailable),
    };

    let prepared = prepare_mesh_targets(
        request,
        effective_model.as_deref(),
        &resolved_hosts,
        affinity,
    );
    let target_hosts = order_mesh_target_hosts(
        node,
        effective_model.as_deref(),
        required_tokens,
        &prepared,
        affinity,
    )
    .await;
    Ok(MeshRequestPlan {
        effective_model,
        auto_session_key,
        prepared,
        target_hosts,
    })
}

fn rewrite_effective_model(request: &mut BufferedHttpRequest, effective_model: Option<&str>) {
    if let Some(name) = effective_model
        && request.model_name.as_deref() != Some(name)
    {
        rewrite_model_field(request, name);
    }
}

fn prepare_mesh_targets(
    request: &mut BufferedHttpRequest,
    effective_model: Option<&str>,
    target_hosts: &[iroh::EndpointId],
    affinity: &AffinityRouter,
) -> PreparedTargets {
    if effective_model.is_some() && target_hosts.len() > 1 {
        request.ensure_body_json();
    }
    let body_json = request.body_json.as_ref();
    effective_model
        .map(|name| prepare_remote_targets_for_request(name, target_hosts, body_json, affinity))
        .unwrap_or(PreparedTargets {
            ordered: target_hosts
                .iter()
                .copied()
                .map(election::InferenceTarget::Remote)
                .collect(),
            learn_prefix_hash: None,
            cached_target: None,
        })
}

async fn order_mesh_target_hosts(
    node: &mesh::Node,
    effective_model: Option<&str>,
    required_tokens: Option<u32>,
    prepared: &PreparedTargets,
    affinity: &AffinityRouter,
) -> Vec<iroh::EndpointId> {
    let target_hosts: Vec<iroh::EndpointId> = prepared
        .ordered
        .iter()
        .filter_map(|target| match target {
            election::InferenceTarget::Remote(host_id) => Some(*host_id),
            _ => None,
        })
        .collect();
    let Some(name) = effective_model else {
        return target_hosts;
    };
    let mut ordered =
        order_remote_hosts_by_context(node, name, required_tokens, &target_hosts).await;
    if let (Some(prefix_hash), Some(election::InferenceTarget::Remote(cached_host))) =
        (prepared.learn_prefix_hash, prepared.cached_target.as_ref())
    {
        let cached_context = node.peer_model_context_length(*cached_host, name).await;
        if matches!((required_tokens, cached_context), (Some(required), Some(context)) if context < required)
        {
            affinity.forget_target(
                name,
                prefix_hash,
                &election::InferenceTarget::Remote(*cached_host),
            );
        } else {
            move_target_first(&mut ordered, cached_host);
        }
    }
    ordered
}

async fn handle_mesh_request_failure(
    node: &mesh::Node,
    tcp_stream: TcpStream,
    request: &BufferedHttpRequest,
    failure: MeshRequestFailure,
) {
    let mut tcp_stream = Some(tcp_stream);
    match failure {
        MeshRequestFailure::UnsupportedMedia => {
            let _ = send_error(
                tcp_stream.take().unwrap(),
                422,
                "no served model can satisfy the requested media inputs",
            )
            .await;
        }
        MeshRequestFailure::ModelUnavailable(model) => {
            node.record_routed_request(
                Some(&model),
                0,
                crate::network::metrics::RequestOutcome::Unavailable,
            );
            tracing::warn!(
                "API proxy: model {:?} not available, no hosts serving it",
                model
            );
            let _ = send_error(
                tcp_stream.take().unwrap(),
                429,
                &format!("model {:?} not currently available — retry later", model),
            )
            .await;
        }
        MeshRequestFailure::NoHostsAvailable => {
            node.record_routed_request(
                None,
                0,
                crate::network::metrics::RequestOutcome::Unavailable,
            );
            let _ = send_503(
                tcp_stream.take().unwrap(),
                "no peers serving any model (mesh empty or gossip stale)",
            )
            .await;
        }
    }
    release_request_objects(node, &request.request_object_request_ids).await;
}

async fn route_mesh_request_attempts(
    node: &mesh::Node,
    mut tcp_stream: TcpStream,
    request: &BufferedHttpRequest,
    plan: &MeshRequestPlan,
    affinity: &AffinityRouter,
) -> Option<TcpStream> {
    let effective_model = plan.effective_model.as_deref();
    let auto_session_key = plan.auto_session_key;
    let prepared = &plan.prepared;
    let target_hosts = &plan.target_hosts;
    let total_targets = target_hosts.len();
    let mut state = MeshAttemptState {
        route_started: Instant::now(),
        attempts: 0,
        last_retryable: false,
        refreshed: false,
    };
    for (idx, target_host) in target_hosts.iter().enumerate() {
        state.attempts += 1;
        let attempt_started = Instant::now();
        let attempt_result = route_remote_attempt_with_retry(
            node,
            &mut tcp_stream,
            *target_host,
            &request.raw,
            ResponseRetryPolicy::next_target_available(idx + 1 < total_targets),
            request.response_adapter,
        )
        .await;
        let attempt_target = election::InferenceTarget::Remote(*target_host);
        record_mesh_request_attempt(
            node,
            effective_model,
            &attempt_target,
            attempt_started.duration_since(state.route_started),
            attempt_started.elapsed(),
            &attempt_result,
        );
        affinity.record_target_outcome(
            effective_model,
            &attempt_target,
            target_health_outcome_for_attempt(&attempt_result),
        );
        let mut context = MeshAttemptResultContext {
            node,
            effective_model,
            auto_session_key,
            prepared,
            attempt_target: &attempt_target,
            target_host: *target_host,
            state: &mut state,
            affinity,
        };
        match handle_mesh_attempt_result(&mut context, attempt_result) {
            MeshAttemptDisposition::Continue => continue,
            MeshAttemptDisposition::Return => return None,
        }
    }
    if state.last_retryable {
        tracing::warn!("All hosts failed for model {:?}", effective_model);
        if let Some(key) = auto_session_key {
            tracing::debug!(
                "auto: all hosts failed for cached model, forgetting session {key:016x}"
            );
            affinity.forget_auto_model(key);
        }
    }
    node.record_routed_request(
        effective_model,
        state.attempts,
        crate::network::metrics::RequestOutcome::Unavailable,
    );
    Some(tcp_stream)
}

fn record_mesh_request_attempt(
    node: &mesh::Node,
    effective_model: Option<&str>,
    attempt_target: &election::InferenceTarget,
    queue_wait: Duration,
    attempt_time: Duration,
    attempt_result: &RouteAttemptResult,
) {
    if matches!(attempt_result, RouteAttemptResult::ClientDisconnected) {
        return;
    }
    node.record_inference_attempt(
        effective_model,
        attempt_target,
        queue_wait,
        attempt_time,
        attempt_outcome_for_result(attempt_result),
        completion_tokens_for_result(attempt_result),
    );
}

struct MeshAttemptResultContext<'a> {
    node: &'a mesh::Node,
    effective_model: Option<&'a str>,
    auto_session_key: Option<u64>,
    prepared: &'a PreparedTargets,
    attempt_target: &'a election::InferenceTarget,
    target_host: iroh::EndpointId,
    state: &'a mut MeshAttemptState,
    affinity: &'a AffinityRouter,
}

fn handle_mesh_attempt_result(
    context: &mut MeshAttemptResultContext<'_>,
    attempt_result: RouteAttemptResult,
) -> MeshAttemptDisposition {
    match attempt_result {
        RouteAttemptResult::Delivered { status_code, .. } => {
            handle_delivered_mesh_attempt(context, status_code)
        }
        RouteAttemptResult::RetryableContextOverflow => handle_retryable_context_overflow(context),
        RouteAttemptResult::RetryableResponseQuality(failure) => {
            handle_retryable_mesh_response_quality(context, failure)
        }
        RouteAttemptResult::RetryableTimeout => handle_retryable_mesh_timeout(context),
        RouteAttemptResult::RetryableUnavailable => handle_retryable_mesh_unavailable(context),
        RouteAttemptResult::ClientDisconnected => {
            tracing::info!(
                "Downstream client disconnected while routing to host {}",
                context.target_host.fmt_short()
            );
            MeshAttemptDisposition::Return
        }
    }
}

fn handle_delivered_mesh_attempt(
    context: &MeshAttemptResultContext<'_>,
    status_code: u16,
) -> MeshAttemptDisposition {
    if should_learn_affinity(status_code) {
        if let (Some(name), Some(prefix_hash)) =
            (context.effective_model, context.prepared.learn_prefix_hash)
        {
            context
                .affinity
                .learn_target(name, prefix_hash, context.attempt_target);
        }
    } else if let Some(key) = context
        .auto_session_key
        .filter(|_| (500..600).contains(&status_code))
    {
        tracing::debug!(
            "auto: upstream returned {status_code}, forgetting cached model for session {key:016x}"
        );
        context.affinity.forget_auto_model(key);
    }
    context.node.record_routed_request(
        context.effective_model,
        context.state.attempts,
        request_outcome_for_status(status_code, crate::network::metrics::RequestService::Remote),
    );
    MeshAttemptDisposition::Return
}

fn handle_retryable_context_overflow(
    context: &mut MeshAttemptResultContext<'_>,
) -> MeshAttemptDisposition {
    forget_mesh_cached_target(
        context.effective_model,
        context.prepared,
        context.attempt_target,
        context.affinity,
    );
    tracing::warn!(
        "Host {} rejected request with context overflow-style 400, trying next",
        context.target_host.fmt_short()
    );
    context.state.last_retryable = true;
    MeshAttemptDisposition::Continue
}

fn handle_retryable_mesh_response_quality(
    context: &mut MeshAttemptResultContext<'_>,
    failure: ResponseQualityFailure,
) -> MeshAttemptDisposition {
    forget_mesh_cached_target(
        context.effective_model,
        context.prepared,
        context.attempt_target,
        context.affinity,
    );
    tracing::warn!(
        reason = failure.label(),
        "Host {} returned low-quality success response, trying next",
        context.target_host.fmt_short()
    );
    context.state.last_retryable = true;
    MeshAttemptDisposition::Continue
}

fn handle_retryable_mesh_timeout(
    context: &mut MeshAttemptResultContext<'_>,
) -> MeshAttemptDisposition {
    tracing::warn!(
        "Host {} timed out, trying next",
        context.target_host.fmt_short()
    );
    context.state.last_retryable = true;
    spawn_mesh_refresh_once(context.node, &mut context.state.refreshed);
    MeshAttemptDisposition::Continue
}

fn handle_retryable_mesh_unavailable(
    context: &mut MeshAttemptResultContext<'_>,
) -> MeshAttemptDisposition {
    forget_mesh_cached_target(
        context.effective_model,
        context.prepared,
        context.attempt_target,
        context.affinity,
    );
    tracing::warn!(
        "Failed to tunnel to host {}, trying next",
        context.target_host.fmt_short()
    );
    context.state.last_retryable = true;
    spawn_mesh_refresh_once(context.node, &mut context.state.refreshed);
    MeshAttemptDisposition::Continue
}

fn forget_mesh_cached_target(
    effective_model: Option<&str>,
    prepared: &PreparedTargets,
    failed_target: &election::InferenceTarget,
    affinity: &AffinityRouter,
) {
    if let (Some(name), Some(prefix_hash), Some(cached_target)) = (
        effective_model,
        prepared.learn_prefix_hash,
        prepared.cached_target.as_ref(),
    ) && cached_target == failed_target
    {
        affinity.forget_target(name, prefix_hash, failed_target);
    }
}

fn spawn_mesh_refresh_once(node: &mesh::Node, refreshed: &mut bool) {
    if *refreshed {
        return;
    }
    let refresh_node = node.clone();
    tokio::spawn(async move {
        refresh_node.gossip_one_peer().await;
    });
    *refreshed = true;
}

async fn finish_exhausted_mesh_request(
    node: &mesh::Node,
    tcp_stream: TcpStream,
    effective_model: Option<&str>,
    total_targets: usize,
    affinity: &AffinityRouter,
) {
    let reason = format!(
        "all {} tunnel(s) to hosts for {:?} failed (mesh request)",
        total_targets, effective_model,
    );
    let _ = affinity;
    let _ = node;
    let _ = send_503(tcp_stream, &reason).await;
}

fn auto_session_key_for_request(
    request: &mut BufferedHttpRequest,
    is_auto_request: bool,
) -> Option<u64> {
    if !is_auto_request {
        return None;
    }
    request.ensure_body_json();
    request
        .body_json
        .as_ref()
        .and_then(|body| crate::network::affinity::auto_model_session_key(Some(body)))
}

struct AutoModelRequestArgs<'a> {
    node: &'a mesh::Node,
    request: &'a mut BufferedHttpRequest,
    served: &'a [String],
    descriptors: &'a [mesh::ServedModelDescriptor],
    is_auto_request: bool,
    auto_session_key: Option<u64>,
    required_tokens: Option<u32>,
    affinity: &'a AffinityRouter,
}

async fn resolve_auto_model_request(args: AutoModelRequestArgs<'_>) -> AutoModelResolution {
    let AutoModelRequestArgs {
        node,
        request,
        served,
        descriptors,
        is_auto_request,
        auto_session_key,
        required_tokens,
        affinity,
    } = args;
    if !is_auto_request {
        return AutoModelResolution::Model(None);
    }
    request.ensure_body_json();
    let Some(body_json) = request.body_json.as_ref() else {
        return AutoModelResolution::Model(None);
    };
    let media = router::media_requirements(body_json);
    // Build candidates with observed throughput so pick_model_classified
    // can weight by locally-measured tok/s where samples exist.
    let routing_metrics = node.routing_metrics();
    let with_caps: Vec<router::RoutingCandidate<'_>> = served
        .iter()
        .map(|name| {
            let caps = capabilities_for_model(name, descriptors);
            let (tps_hint, throughput_samples) = routing_metrics
                .tps_for_model(name)
                .map(|(tps, samples)| (Some(tps), samples))
                .unwrap_or((None, 0));
            router::RoutingCandidate {
                name: name.as_str(),
                caps,
                parameter_count_b: descriptor_metadata_for_model(name, descriptors)
                    .and_then(|metadata| metadata.parameter_count_b),
                tps_hint,
                throughput_samples,
            }
        })
        .collect();
    let available = router::filter_media_compatible_candidates(&with_caps, &media);
    let ready_models = if let Some(available) = available.as_ref() {
        auto_route::ready_remote_models(node, required_tokens, available, affinity).await
    } else {
        Vec::new()
    };
    if let Some(model) = lookup_cached_auto_model(
        node,
        descriptors,
        affinity,
        auto_session_key,
        &media,
        &ready_models,
    )
    .await
    {
        return AutoModelResolution::Model(Some(model));
    }

    let Some(available) = available else {
        return AutoModelResolution::UnsupportedMedia;
    };
    let available = auto_route::pool_for_ready_models(&available, &ready_models);
    let cl = router::classify(body_json);
    let picked = router::pick_model_classified(&cl, &available).map(str::to_string);
    if let Some(name) = picked.as_deref() {
        tracing::info!(
            "router: {:?}/{:?} tools={} media={} → {name}",
            cl.category,
            cl.complexity,
            cl.needs_tools,
            cl.has_media_inputs
        );
        if let Some(key) = auto_session_key {
            affinity.remember_auto_model(key, name);
        }
    }
    AutoModelResolution::Model(picked)
}

async fn lookup_cached_auto_model(
    node: &mesh::Node,
    descriptors: &[mesh::ServedModelDescriptor],
    affinity: &AffinityRouter,
    auto_session_key: Option<u64>,
    media: &router::MediaRequirements,
    ready_models: &[&str],
) -> Option<String> {
    let key = auto_session_key?;
    let model = affinity.lookup_auto_model(key)?;
    if let Some(reason) =
        cached_auto_model_reclassify_reason(node, &model, media, descriptors, ready_models).await
    {
        tracing::debug!("auto: cached model {model} {reason}, reclassifying");
        affinity.forget_auto_model(key);
        return None;
    }
    tracing::debug!("auto: reusing cached model {model} for session {key:016x}");
    Some(model)
}

async fn cached_auto_model_reclassify_reason(
    node: &mesh::Node,
    model: &str,
    media: &router::MediaRequirements,
    descriptors: &[mesh::ServedModelDescriptor],
    ready_models: &[&str],
) -> Option<&'static str> {
    if cached_auto_model_missing(node, model).await {
        return Some("no longer served");
    }
    if cached_auto_model_needs_reclassify(model, media, descriptors) {
        return Some("cannot satisfy media requirements");
    }
    if !ready_models.is_empty() && !ready_models.contains(&model) {
        return Some("has no eligible target for this request");
    }
    None
}

async fn cached_auto_model_missing(node: &mesh::Node, model: &str) -> bool {
    node.hosts_for_model(model).await.is_empty()
}

fn cached_auto_model_needs_reclassify(
    model: &str,
    media: &router::MediaRequirements,
    descriptors: &[mesh::ServedModelDescriptor],
) -> bool {
    !cached_auto_model_satisfies_media_requirements(model, media, descriptors)
}

async fn resolve_mesh_target_hosts(
    node: &mesh::Node,
    effective_model: Option<&str>,
) -> MeshTargetResolution {
    let target_hosts = if let Some(name) = effective_model {
        node.hosts_for_model(name).await
    } else {
        Vec::new()
    };
    if !target_hosts.is_empty() {
        return MeshTargetResolution::Hosts(target_hosts);
    }
    if let Some(model) = effective_model {
        return MeshTargetResolution::ModelUnavailable(model.to_string());
    }
    match node.any_host().await {
        Some(peer) => MeshTargetResolution::Hosts(vec![peer.id]),
        None => MeshTargetResolution::NoHostsAvailable,
    }
}

async fn route_attempt_for_target(
    node: &mesh::Node,
    tcp_stream: &mut TcpStream,
    target: &election::InferenceTarget,
    prefetched: &[u8],
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
) -> RouteAttemptResult {
    match target {
        election::InferenceTarget::Local(port) => {
            route_local_attempt(
                node,
                tcp_stream,
                *port,
                prefetched,
                retry_policy,
                response_adapter,
            )
            .await
        }
        election::InferenceTarget::Remote(host_id) => {
            route_remote_attempt_with_retry(
                node,
                tcp_stream,
                *host_id,
                prefetched,
                retry_policy,
                response_adapter,
            )
            .await
        }
        election::InferenceTarget::None => RouteAttemptResult::RetryableUnavailable,
    }
}

async fn route_remote_attempt_with_retry(
    node: &mesh::Node,
    tcp_stream: &mut TcpStream,
    host_id: iroh::EndpointId,
    prefetched: &[u8],
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
) -> RouteAttemptResult {
    let mut result = route_remote_attempt(
        node,
        tcp_stream,
        host_id,
        prefetched,
        retry_policy,
        response_adapter,
    )
    .await;
    for retry in 1..=REMOTE_UNCOMMITTED_RETRIES {
        if !should_retry_uncommitted_remote_attempt(result) {
            return result;
        }
        tracing::warn!(
            host = %host_id.fmt_short(),
            retry,
            outcome = route_attempt_result_label(&result),
            "API proxy: retrying remote target on fresh tunnel before committing response"
        );
        result = route_remote_attempt(
            node,
            tcp_stream,
            host_id,
            prefetched,
            retry_policy,
            response_adapter,
        )
        .await;
    }
    result
}

fn should_retry_uncommitted_remote_attempt(result: RouteAttemptResult) -> bool {
    matches!(
        result,
        RouteAttemptResult::RetryableTimeout | RouteAttemptResult::RetryableUnavailable
    )
}

pub async fn route_model_request(
    node: mesh::Node,
    tcp_stream: TcpStream,
    targets: &election::ModelTargets,
    model: &str,
    request: &BufferedHttpRequest,
    required_tokens: Option<u32>,
    affinity: &AffinityRouter,
) -> bool {
    let args = RouteModelRequestArgs {
        node,
        tcp_stream,
        targets,
        model,
        request,
        required_tokens,
        affinity,
    };
    route_model_request_inner(args).await
}

struct RouteModelRequestArgs<'a> {
    node: mesh::Node,
    tcp_stream: TcpStream,
    targets: &'a election::ModelTargets,
    model: &'a str,
    request: &'a BufferedHttpRequest,
    required_tokens: Option<u32>,
    affinity: &'a AffinityRouter,
}

struct RouteModelState {
    route_started: Instant,
    attempts: usize,
    refreshed: bool,
}

enum RouteModelDisposition {
    Continue,
    Return(bool),
}

fn no_context_eligible_target_reason(model: &str, required_tokens: Option<u32>) -> String {
    match required_tokens {
        Some(tokens) => format!(
            "no context-compatible target for model '{model}' can fit approximately {tokens} tokens"
        ),
        None => format!("no eligible target for model '{model}'"),
    }
}

async fn route_model_request_inner(args: RouteModelRequestArgs<'_>) -> bool {
    let RouteModelRequestArgs {
        node,
        tcp_stream,
        targets,
        model,
        request,
        required_tokens,
        affinity,
    } = args;
    let route_started = Instant::now();
    let mut tcp_stream = tcp_stream;
    let ordered_candidates =
        order_targets_by_context(&node, model, required_tokens, &targets.candidates(model)).await;
    let ordered_candidates = affinity.route_eligible_candidates(model, &ordered_candidates);
    if ordered_candidates.is_empty() {
        record_route_model_unavailable(&node, model, 0);
        let reason = no_context_eligible_target_reason(model, required_tokens);
        let _ = send_503(tcp_stream, &reason).await;
        return true;
    }

    let selection = crate::network::affinity::select_model_target_from_candidates(
        targets,
        &ordered_candidates,
        model,
        request.body_json.as_ref(),
        affinity,
    );
    if matches!(selection.target, election::InferenceTarget::None) {
        return send_route_model_none_target(&node, tcp_stream, model).await;
    }
    forget_route_model_context_mismatch(&node, model, required_tokens, &selection, affinity).await;

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
            request.response_adapter,
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
            affinity,
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

    finish_exhausted_route_model_request(&node, tcp_stream, model, total_targets, &state).await;
    true
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
    tcp_stream: TcpStream,
    model: &str,
) -> bool {
    record_route_model_unavailable(node, model, 0);
    let _ = send_503(
        tcp_stream,
        &format!("target for model '{model}' resolved to None (election in progress or host down)"),
    )
    .await;
    true
}

async fn finish_exhausted_route_model_request(
    node: &mesh::Node,
    tcp_stream: TcpStream,
    model: &str,
    total_targets: usize,
    state: &RouteModelState,
) {
    let _ = send_503(
        tcp_stream,
        &format!("all {} target(s) for model '{model}' failed", total_targets),
    )
    .await;
    record_route_model_unavailable(node, model, state.attempts);
    tracing::warn!(
        model = model,
        attempts = state.attempts,
        route_ms = state.route_started.elapsed().as_millis(),
        "openai route_model_request exhausted targets"
    );
}

async fn forget_route_model_context_mismatch(
    node: &mesh::Node,
    model: &str,
    required_tokens: Option<u32>,
    selection: &TargetSelection,
    affinity: &AffinityRouter,
) {
    let (Some(prefix_hash), Some(cached_target)) = (
        selection.learn_prefix_hash,
        selection.cached_target.as_ref(),
    ) else {
        return;
    };
    let cached_context = match cached_target {
        election::InferenceTarget::Local(_) => node.local_model_context_length(model).await,
        election::InferenceTarget::Remote(peer_id) => {
            node.peer_model_context_length(*peer_id, model).await
        }
        election::InferenceTarget::None => None,
    };
    if matches!((required_tokens, cached_context), (Some(required), Some(context)) if context < required)
    {
        affinity.forget_target(model, prefix_hash, cached_target);
    }
}

fn handle_route_model_attempt_result(
    node: &mesh::Node,
    model: &str,
    target: &election::InferenceTarget,
    selection: &TargetSelection,
    attempt_result: RouteAttemptResult,
    state: &mut RouteModelState,
    affinity: &AffinityRouter,
) -> RouteModelDisposition {
    match attempt_result {
        RouteAttemptResult::Delivered { status_code, .. } => handle_delivered_route_model_attempt(
            node,
            model,
            target,
            selection,
            status_code,
            state,
            affinity,
        ),
        RouteAttemptResult::RetryableContextOverflow => {
            handle_retryable_route_model_context(model, target, selection, affinity)
        }
        RouteAttemptResult::RetryableResponseQuality(failure) => {
            handle_retryable_route_model_response_quality(
                model, target, selection, affinity, failure,
            )
        }
        RouteAttemptResult::RetryableTimeout => {
            handle_retryable_route_model_timeout(node, model, target, selection, state, affinity)
        }
        RouteAttemptResult::RetryableUnavailable => handle_retryable_route_model_unavailable(
            node, model, target, selection, state, affinity,
        ),
        RouteAttemptResult::ClientDisconnected => {
            tracing::info!(
                model = model,
                attempts = state.attempts,
                route_ms = state.route_started.elapsed().as_millis(),
                "openai route_model_request downstream disconnected"
            );
            RouteModelDisposition::Return(true)
        }
    }
}

fn handle_delivered_route_model_attempt(
    node: &mesh::Node,
    model: &str,
    target: &election::InferenceTarget,
    selection: &TargetSelection,
    status_code: u16,
    state: &RouteModelState,
    affinity: &AffinityRouter,
) -> RouteModelDisposition {
    if should_learn_affinity(status_code)
        && let Some(prefix_hash) = selection.learn_prefix_hash
    {
        affinity.learn_target(model, prefix_hash, target);
    }
    node.record_routed_request(
        Some(model),
        state.attempts,
        request_outcome_for_status(status_code, request_service_for_target(target)),
    );
    tracing::info!(
        model = model,
        attempts = state.attempts,
        status_code = status_code,
        route_ms = state.route_started.elapsed().as_millis(),
        "openai route_model_request delivered"
    );
    RouteModelDisposition::Return(true)
}

fn handle_retryable_route_model_context(
    model: &str,
    target: &election::InferenceTarget,
    selection: &TargetSelection,
    affinity: &AffinityRouter,
) -> RouteModelDisposition {
    forget_selected_route_model_target(model, target, selection, affinity);
    tracing::warn!(
        "Target {target:?} rejected request with context overflow-style 400, trying next"
    );
    RouteModelDisposition::Continue
}

fn handle_retryable_route_model_response_quality(
    model: &str,
    target: &election::InferenceTarget,
    selection: &TargetSelection,
    affinity: &AffinityRouter,
    failure: ResponseQualityFailure,
) -> RouteModelDisposition {
    forget_selected_route_model_target(model, target, selection, affinity);
    tracing::warn!(
        reason = failure.label(),
        "Target {target:?} returned low-quality success response, trying next"
    );
    RouteModelDisposition::Continue
}

fn handle_retryable_route_model_timeout(
    node: &mesh::Node,
    model: &str,
    target: &election::InferenceTarget,
    selection: &TargetSelection,
    state: &mut RouteModelState,
    affinity: &AffinityRouter,
) -> RouteModelDisposition {
    forget_selected_route_model_target(model, target, selection, affinity);
    spawn_mesh_refresh_once(node, &mut state.refreshed);
    tracing::warn!("Target {target:?} timed out, trying next");
    RouteModelDisposition::Continue
}

fn handle_retryable_route_model_unavailable(
    node: &mesh::Node,
    model: &str,
    target: &election::InferenceTarget,
    selection: &TargetSelection,
    state: &mut RouteModelState,
    affinity: &AffinityRouter,
) -> RouteModelDisposition {
    forget_selected_route_model_target(model, target, selection, affinity);
    spawn_mesh_refresh_once(node, &mut state.refreshed);
    tracing::warn!("Target {target:?} unavailable, trying next");
    RouteModelDisposition::Continue
}

fn forget_selected_route_model_target(
    model: &str,
    target: &election::InferenceTarget,
    selection: &TargetSelection,
    affinity: &AffinityRouter,
) {
    if let (Some(prefix_hash), Some(cached_target)) = (
        selection.learn_prefix_hash,
        selection.cached_target.as_ref(),
    ) && cached_target == target
    {
        affinity.forget_target(model, prefix_hash, target);
    }
}

fn finalize_route_model_result(
    _node: &mesh::Node,
    _model: &str,
    _request: &BufferedHttpRequest,
    _route_started: Instant,
    _attempts: usize,
    result: bool,
    _target: &election::InferenceTarget,
) -> bool {
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

/// Route a request to a known inference target (local OpenAI surface or remote host).
///
/// Used by the API proxy after election has determined the target.
pub async fn route_to_target(
    node: mesh::Node,
    tcp_stream: TcpStream,
    model: Option<&str>,
    target: election::InferenceTarget,
    prefetched: &[u8],
    response_adapter: ResponseAdapter,
) -> bool {
    let route_started = Instant::now();
    let mut tcp_stream = tcp_stream;
    tracing::info!("API proxy: routing to target {target:?}");
    let result = route_attempt_for_target(
        &node,
        &mut tcp_stream,
        &target,
        prefetched,
        ResponseRetryPolicy::next_target_available(false),
        response_adapter,
    )
    .await;
    node.record_inference_attempt(
        model,
        &target,
        Duration::ZERO,
        route_started.elapsed(),
        attempt_outcome_for_result(&result),
        completion_tokens_for_result(&result),
    );
    tracing::info!(
        target = ?target,
        outcome = route_attempt_result_label(&result),
        route_ms = route_started.elapsed().as_millis(),
        "openai route_to_target result"
    );
    match result {
        RouteAttemptResult::Delivered {
            status_code,
            completion_tokens: _,
        } => {
            let service = request_service_for_target(&target);
            node.record_routed_request(model, 1, request_outcome_for_status(status_code, service));
            true
        }
        RouteAttemptResult::RetryableTimeout
        | RouteAttemptResult::RetryableContextOverflow
        | RouteAttemptResult::RetryableResponseQuality(_)
        | RouteAttemptResult::RetryableUnavailable => {
            node.record_routed_request(
                model,
                1,
                crate::network::metrics::RequestOutcome::Unavailable,
            );
            let _ = send_503(
                tcp_stream,
                &format!("single target {target:?} unavailable (route_to_target)"),
            )
            .await;
            false
        }
        RouteAttemptResult::ClientDisconnected => true,
    }
}

pub async fn route_http_endpoint_request(
    node: &mesh::Node,
    model: Option<&str>,
    tcp_stream: &mut TcpStream,
    base_url: &str,
    prefetched: &[u8],
    request_path: &str,
    response_adapter: ResponseAdapter,
) -> bool {
    let started = Instant::now();
    let result = route_http_endpoint_attempt(
        tcp_stream,
        base_url,
        prefetched,
        request_path,
        ResponseRetryPolicy::next_target_available(false),
        response_adapter,
    )
    .await;
    node.record_endpoint_attempt(
        model,
        base_url,
        Duration::ZERO,
        started.elapsed(),
        attempt_outcome_for_result(&result),
        completion_tokens_for_result(&result),
    );
    tracing::info!(
        endpoint = base_url,
        path = request_path,
        outcome = route_attempt_result_label(&result),
        route_ms = started.elapsed().as_millis(),
        "openai route_http_endpoint_request result"
    );
    match result {
        RouteAttemptResult::Delivered {
            status_code,
            completion_tokens: _,
        } => {
            node.record_routed_request(
                model,
                1,
                request_outcome_for_status(
                    status_code,
                    crate::network::metrics::RequestService::Endpoint,
                ),
            );
            true
        }
        RouteAttemptResult::RetryableTimeout
        | RouteAttemptResult::RetryableContextOverflow
        | RouteAttemptResult::RetryableResponseQuality(_)
        | RouteAttemptResult::RetryableUnavailable => {
            node.record_routed_request(
                model,
                1,
                crate::network::metrics::RequestOutcome::Unavailable,
            );
            false
        }
        RouteAttemptResult::ClientDisconnected => true,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::target_health::TargetHealthOutcome;
    use anyhow::Result;
    use std::collections::HashMap;

    fn test_peer_serving_model(peer_id: iroh::EndpointId, model: &str) -> mesh::PeerInfo {
        mesh::PeerInfo {
            id: peer_id,
            addr: iroh::EndpointAddr {
                id: peer_id,
                addrs: Default::default(),
            },
            mesh_id: None,
            mesh_policy_hash: None,
            genesis_policy: None,
            role: mesh::NodeRole::Host { http_port: 9337 },
            first_joined_mesh_ts: None,
            models: vec![model.to_string()],
            vram_bytes: 16 * 1024 * 1024 * 1024,
            rtt_ms: None,
            model_source: None,
            admitted: true,
            serving_models: vec![model.to_string()],
            hosted_models: vec![model.to_string()],
            hosted_models_known: true,
            available_models: vec![],
            requested_models: vec![],
            explicit_model_interests: vec![],
            last_seen: std::time::Instant::now(),
            last_mentioned: std::time::Instant::now(),
            version: None,
            gpu_name: None,
            hostname: None,
            is_soc: None,
            gpu_vram: None,
            gpu_reserved_bytes: None,
            gpu_mem_bandwidth_gbps: None,
            gpu_compute_tflops_fp32: None,
            gpu_compute_tflops_fp16: None,
            available_model_metadata: vec![],
            experts_summary: None,
            available_model_sizes: HashMap::new(),
            served_model_descriptors: vec![local_gguf_descriptor(model)],
            served_model_runtime: vec![],
            owner_attestation: None,
            release_attestation_summary: crate::ReleaseAttestationSummary::default(),
            artifact_transfer_supported: false,
            stage_protocol_generation_supported: false,
            stage_status_list_supported: false,
            advertised_model_throughput: vec![],
            display_rtt: None,
            selected_path: None,
            propagated_latency: None,
            owner_summary: crate::crypto::OwnershipSummary::default(),
        }
    }

    async fn test_node_with_remote_models(models: &[(&str, iroh::EndpointId)]) -> mesh::Node {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
            .await
            .expect("test node should start");
        for (model, peer_id) in models {
            node.insert_test_peer(test_peer_serving_model(*peer_id, model))
                .await;
        }
        node
    }
    fn text_auto_request() -> BufferedHttpRequest {
        let body = serde_json::json!({
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let body_bytes = serde_json::to_vec(&body).expect("request body should serialize");
        BufferedHttpRequest {
            raw: Vec::new(),
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            client_path: "/v1/chat/completions".to_string(),
            body_json: Some(body),
            body_json_attempted: true,
            body_bytes: Some(body_bytes),
            body_len_bytes: 0,
            completion_tokens: None,
            model_name: Some("auto".to_string()),
            stream: None,
            request_object_request_ids: Vec::new(),
            response_adapter: ResponseAdapter::None,
        }
    }
    fn local_gguf_descriptor(model_name: &str) -> mesh::ServedModelDescriptor {
        mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: model_name.to_string(),
                source_kind: mesh::ModelSourceKind::LocalGguf,
                local_file_name: Some(format!("{model_name}.gguf")),
                ..Default::default()
            },
            ..Default::default()
        }
    }
    #[test]
    fn test_remote_retry_policy_only_retries_uncommitted_failures() {
        assert!(should_retry_uncommitted_remote_attempt(
            RouteAttemptResult::RetryableUnavailable
        ));
        assert!(should_retry_uncommitted_remote_attempt(
            RouteAttemptResult::RetryableTimeout
        ));
        assert!(!should_retry_uncommitted_remote_attempt(
            RouteAttemptResult::RetryableContextOverflow
        ));
        assert!(!should_retry_uncommitted_remote_attempt(
            RouteAttemptResult::RetryableResponseQuality(
                ResponseQualityFailure::EmptyAssistantOutput
            )
        ));
        assert!(!should_retry_uncommitted_remote_attempt(
            RouteAttemptResult::ClientDisconnected
        ));
        assert!(!should_retry_uncommitted_remote_attempt(
            RouteAttemptResult::Delivered {
                status_code: 200,
                completion_tokens: None,
            }
        ));
    }
    #[tokio::test]
    async fn cached_auto_model_stays_sticky_when_no_ready_remote_model_exists() -> Result<()> {
        let cached_model = "cached-cooling-model-31B";
        let alternate_model = "alternate-cooling-model-31B";
        let cached_peer = iroh::EndpointId::from(iroh::SecretKey::generate().public());
        let alternate_peer = iroh::EndpointId::from(iroh::SecretKey::generate().public());
        let node = test_node_with_remote_models(&[
            (cached_model, cached_peer),
            (alternate_model, alternate_peer),
        ])
        .await;
        let affinity = AffinityRouter::new();
        let key = 0xA11CE;
        affinity.remember_auto_model(key, cached_model);
        affinity.record_target_outcome(
            Some(cached_model),
            &election::InferenceTarget::Remote(cached_peer),
            TargetHealthOutcome::Unavailable,
        );
        affinity.record_target_outcome(
            Some(alternate_model),
            &election::InferenceTarget::Remote(alternate_peer),
            TargetHealthOutcome::Unavailable,
        );
        let descriptors = vec![
            local_gguf_descriptor(cached_model),
            local_gguf_descriptor(alternate_model),
        ];
        let media = router::MediaRequirements::default();
        let caps = crate::models::ModelCapabilities::default();
        let available = vec![
            router::RoutingCandidate::unscored(cached_model, caps),
            router::RoutingCandidate::unscored(alternate_model, caps),
        ];
        let ready_models =
            auto_route::ready_remote_models(&node, None, &available, &affinity).await;
        assert!(ready_models.is_empty());

        let cached = lookup_cached_auto_model(
            &node,
            &descriptors,
            &affinity,
            Some(key),
            &media,
            &ready_models,
        )
        .await;

        assert_eq!(cached.as_deref(), Some(cached_model));
        assert_eq!(
            affinity.lookup_auto_model(key).as_deref(),
            Some(cached_model)
        );
        Ok(())
    }

    #[tokio::test]
    async fn auto_model_cache_switches_when_ready_alternate_exists() -> Result<()> {
        let cached_model = "cached-cooling-model-31B";
        let alternate_model = "ready-alternate-model-31B";
        let cached_peer = iroh::EndpointId::from(iroh::SecretKey::generate().public());
        let alternate_peer = iroh::EndpointId::from(iroh::SecretKey::generate().public());
        let node = test_node_with_remote_models(&[
            (cached_model, cached_peer),
            (alternate_model, alternate_peer),
        ])
        .await;
        let affinity = AffinityRouter::new();
        let key = 0xB0B;
        affinity.remember_auto_model(key, cached_model);
        affinity.record_target_outcome(
            Some(cached_model),
            &election::InferenceTarget::Remote(cached_peer),
            TargetHealthOutcome::Unavailable,
        );
        let served = vec![cached_model.to_string(), alternate_model.to_string()];
        let descriptors = vec![
            local_gguf_descriptor(cached_model),
            local_gguf_descriptor(alternate_model),
        ];
        let mut request = text_auto_request();

        let resolved = resolve_auto_model_request(AutoModelRequestArgs {
            node: &node,
            request: &mut request,
            served: &served,
            descriptors: &descriptors,
            is_auto_request: true,
            auto_session_key: Some(key),
            required_tokens: None,
            affinity: &affinity,
        })
        .await;

        assert!(matches!(
            resolved,
            AutoModelResolution::Model(Some(model)) if model == alternate_model
        ));
        assert_eq!(
            affinity.lookup_auto_model(key).as_deref(),
            Some(alternate_model)
        );
        Ok(())
    }
    #[test]
    fn test_capture_path_for_request_uses_client_path() {
        let request = BufferedHttpRequest {
            raw: Vec::new(),
            method: "POST".to_string(),
            path: "/v1/chat/completions?foo=1".to_string(),
            client_path: "/v1/responses?foo=1".to_string(),
            body_json: None,
            body_json_attempted: false,
            body_bytes: None,
            body_len_bytes: 0,
            completion_tokens: None,
            stream: None,
            model_name: Some("qwen".to_string()),
            request_object_request_ids: Vec::new(),
            response_adapter: ResponseAdapter::OpenAiResponsesStream,
        };

        assert_eq!(capture_path_for_request(&request), "/v1/responses?foo=1");
    }
}
