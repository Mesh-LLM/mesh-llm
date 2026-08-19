use crate::inference::{election, pipeline};
use crate::logging::{OpenAiLifecycleAttachment, OpenAiRouteObserver};
use crate::mesh;
use crate::network::affinity;
use crate::network::openai::auto_route;
use crate::network::openai::automatic;
use crate::network::openai::transport as proxy;
use crate::network::router;
use mesh_llm_events::audit::{audit_events, emit_audit};
use mesh_llm_events::{OutputEvent, emit_event};
use mesh_mixture_of_agents as moa;

enum AutoRouteResolution {
    Continue {
        effective_model: Option<String>,
        classification: Option<router::Classification>,
    },
    MediaUnsupported,
}

struct IngressRouteContext<'a> {
    node: &'a mesh::Node,
    targets: &'a election::ModelTargets,
    affinity: &'a affinity::AffinityRouter,
    plugin_manager: Option<&'a crate::plugin::PluginManager>,
}

struct ProxyConnectionContext<'a> {
    route: IngressRouteContext<'a>,
}

struct AutoRouteDecision {
    effective_model: Option<String>,
    classification: Option<router::Classification>,
    required_tokens: Option<u32>,
}

fn terminal_outcome_for_dispatch(
    outcome: proxy::RouteDispatchOutcome,
) -> crate::logging::TerminalOutcome {
    outcome.terminal_outcome()
}

fn model_access_succeeded(outcome: proxy::RouteDispatchOutcome) -> bool {
    matches!(
        outcome,
        proxy::RouteDispatchOutcome::Responded(200..=299)
            | proxy::RouteDispatchOutcome::RespondedWithUsage {
                status_code: 200..=299,
                ..
            }
    )
}

fn response_outcome(status_code: u16, result: std::io::Result<()>) -> proxy::RouteDispatchOutcome {
    match result {
        Ok(()) => proxy::RouteDispatchOutcome::Responded(status_code),
        Err(_) => proxy::RouteDispatchOutcome::Dropped("response_write_failed"),
    }
}

/// Check activity policy admission and reject with 503 if paused.
async fn check_activity_admission(
    tcp_stream: tokio::net::TcpStream,
    guard: &crate::runtime::ActivityPolicyGuard,
    ingress_type: crate::runtime::IngressType,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<tokio::net::TcpStream, proxy::RouteDispatchOutcome> {
    match guard.check_admission(ingress_type) {
        crate::runtime::AdmissionResult::Allowed => Ok(tcp_stream),
        crate::runtime::AdmissionResult::Paused { reason, .. } => {
            tracing::debug!(reason, "Ingress rejected by activity policy");
            Err(response_outcome(
                503,
                proxy::send_503_observed(
                    tcp_stream,
                    &format!("inference paused: {reason}"),
                    route_observer,
                )
                .await,
            ))
        }
    }
}

/// Parse a model identifier that may include a profile suffix.
///
/// Returns `(model_ref, profile)` where:
/// - `model_ref` is the base model identifier (without `#profile`)
/// - `profile` is `Some(profile_name)` if `#profile` was present, `None` otherwise
///
/// Examples:
/// - `"Qwen/Qwen3-8B:Q4_K_M"` → `("Qwen/Qwen3-8B:Q4_K_M", None)`
/// - `"Qwen/Qwen3-8B:Q4_K_M#low-ctx"` → `("Qwen/Qwen3-8B:Q4_K_M", Some("low-ctx"))`
/// - `"model#"` → `("model", None)` (empty profile treated as None)
pub(super) fn parse_model_with_profile(model: &str) -> (&str, &str) {
    if let Some(hash_pos) = model.rfind('#') {
        let model_ref = &model[..hash_pos];
        let profile = &model[hash_pos + 1..];
        if profile.is_empty() {
            (model_ref, "")
        } else {
            (model_ref, profile)
        }
    } else {
        (model, "")
    }
}

async fn bind_api_proxy_listener(
    port: u16,
    existing_listener: Option<tokio::net::TcpListener>,
    listen_all: bool,
) -> Option<tokio::net::TcpListener> {
    match existing_listener {
        Some(listener) => Some(listener),
        None => {
            let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
            match tokio::net::TcpListener::bind(format!("{addr}:{port}")).await {
                Ok(listener) => Some(listener),
                Err(error) => {
                    tracing::error!("Failed to bind API proxy to port {port}: {error}");
                    None
                }
            }
        }
    }
}

async fn handle_models_list_request(
    tcp_stream: tokio::net::TcpStream,
    node: &mesh::Node,
    targets: &election::ModelTargets,
    plugin_manager: Option<&crate::plugin::PluginManager>,
) -> proxy::RouteDispatchOutcome {
    let mut models = callable_models(targets);
    models.extend(node.models_being_served().await);
    if let Some(plugin_manager) = plugin_manager
        && let Ok(mut external_models) = plugin_manager.inference_models().await
    {
        models.append(&mut external_models);
    }
    models.sort();
    models.dedup();
    let descriptors = node.all_served_model_descriptors().await;
    let runtimes = node.all_model_runtime_descriptors().await;
    response_outcome(
        200,
        proxy::send_models_list_with_descriptors(tcp_stream, &models, &descriptors, &runtimes)
            .await,
    )
}

async fn collect_available_models_for_auto_route(
    node: &mesh::Node,
    targets: &election::ModelTargets,
    plugin_manager: Option<&crate::plugin::PluginManager>,
) -> Vec<String> {
    let mut available_models = callable_models(targets);
    for name in node.models_being_served().await {
        if !available_models.iter().any(|existing| existing == &name) {
            available_models.push(name);
        }
    }
    if let Some(plugin_manager) = plugin_manager
        && let Ok(external_models) = plugin_manager.inference_models().await
    {
        for name in external_models {
            if !available_models.iter().any(|existing| existing == &name) {
                available_models.push(name);
            }
        }
    }
    available_models
}

async fn resolve_auto_routed_model(
    node: &mesh::Node,
    request: &mut proxy::BufferedHttpRequest,
    targets: &election::ModelTargets,
    plugin_manager: Option<&crate::plugin::PluginManager>,
    descriptors: &[crate::mesh::ServedModelDescriptor],
    required_tokens: Option<u32>,
    affinity: &affinity::AffinityRouter,
) -> AutoRouteResolution {
    // An explicitly named model routes to itself. The automatic directive (in
    // either spelling) resolves here, so `mesh` reaches the same
    // media-capability filter and readiness/affinity selection as `auto` —
    // without it, a `mesh` request with an image skipped the filter entirely
    // and had its image silently dropped downstream.
    if let Some(model) = request.model_name.as_deref()
        && !automatic::is_directive(model)
    {
        return AutoRouteResolution::Continue {
            effective_model: request.model_name.clone(),
            classification: None,
        };
    }

    request.ensure_body_json();
    let Some(body_json) = request.body_json.as_ref() else {
        return AutoRouteResolution::Continue {
            effective_model: None,
            classification: None,
        };
    };

    automatic::warn_if_deprecated_alias(request.model_name.as_deref());

    let mode = automatic::serving_mode(automatic::AutomaticRequest {
        model: request.model_name.as_deref(),
        // The forwarded path: `/v1/responses` has already been normalised onto
        // chat completions by this point, so it stays committee-eligible.
        path: &request.path,
        body: body_json,
    });
    match mode {
        // Committee mode keeps the directive as the effective model so the MoA
        // gateway picks the request up.
        automatic::ServingMode::Committee => {
            return AutoRouteResolution::Continue {
                effective_model: Some(automatic::DIRECTIVE.to_string()),
                classification: None,
            };
        }
        // Single-model mode falls through to the capability-, readiness- and
        // affinity-aware selection below.
        automatic::ServingMode::SingleModel(reason) => tracing::debug!(
            reason = reason.as_str(),
            "automatic routing: serving from a single model"
        ),
    }

    let classification = router::classify(body_json);
    let media = router::media_requirements(body_json);
    let available_models =
        collect_available_models_for_auto_route(node, targets, plugin_manager).await;
    let metrics = node.routing_metrics();
    let available: Vec<router::RoutingCandidate<'_>> = available_models
        .iter()
        .map(|name| {
            let caps = proxy::capabilities_for_model(name, descriptors);
            let (tps_hint, throughput_samples) = metrics
                .tps_for_model(name)
                .map(|(t, s)| (Some(t), s))
                .unwrap_or((None, 0));
            router::RoutingCandidate {
                name: name.as_str(),
                caps,
                parameter_count_b: proxy::descriptor_metadata_for_model(name, descriptors)
                    .and_then(|metadata| metadata.parameter_count_b),
                tps_hint,
                throughput_samples,
            }
        })
        .collect();
    let Some(available) = router::filter_media_compatible_candidates(&available, &media) else {
        proxy::release_request_objects(node, &request.request_object_request_ids).await;
        return AutoRouteResolution::MediaUnsupported;
    };
    let available =
        auto_route_pool_for_ready_models(node, targets, required_tokens, &available, affinity)
            .await;

    let effective_model = router::pick_model_classified(&classification, &available).map(|name| {
        tracing::info!(
            "router: {:?}/{:?} tools={} → {name}",
            classification.category,
            classification.complexity,
            classification.needs_tools
        );
        name.to_string()
    });

    AutoRouteResolution::Continue {
        effective_model,
        classification: Some(classification),
    }
}

async fn auto_route_pool_for_ready_models<'a>(
    node: &mesh::Node,
    targets: &election::ModelTargets,
    required_tokens: Option<u32>,
    available: &[router::RoutingCandidate<'a>],
    affinity: &affinity::AffinityRouter,
) -> Vec<router::RoutingCandidate<'a>> {
    let mut ready_models = Vec::new();
    for candidate in available {
        if auto_route_model_has_ready_ingress_target(
            node,
            targets,
            candidate.name,
            required_tokens,
            affinity,
        )
        .await
        {
            ready_models.push(candidate.name);
        }
    }
    auto_route::pool_for_ready_models(available, &ready_models)
}

async fn auto_route_model_has_ready_ingress_target(
    node: &mesh::Node,
    targets: &election::ModelTargets,
    model: &str,
    required_tokens: Option<u32>,
    affinity: &affinity::AffinityRouter,
) -> bool {
    let local_candidates = targets.candidates(model);
    if contains_routable_candidate(&local_candidates) {
        return auto_route::model_has_eligible_target(
            node,
            model,
            required_tokens,
            &local_candidates,
            affinity,
        )
        .await;
    }

    let remote_candidates = node
        .hosts_for_model(model)
        .await
        .into_iter()
        .map(election::InferenceTarget::Remote)
        .collect::<Vec<_>>();
    if !remote_candidates.is_empty() {
        return auto_route::model_has_eligible_target(
            node,
            model,
            required_tokens,
            &remote_candidates,
            affinity,
        )
        .await;
    }

    // No routable local target and no peer advertises this model. Fail closed:
    // such a model is a phantom (stale gossip, or a peer that unloaded it), and
    // letting it stay in the pool means `auto` can pick a model that then 404s
    // on a model the user never named. Explicit requests still 404 honestly.
    //
    // The one exception is a freshly started serve node: its own model is
    // loaded and in `serving_models` before the target table and gossip catch
    // up, so keep it eligible rather than excluding this node's own model.
    auto_route::model_is_locally_served(node, model).await
}

/// Commit an automatic routing decision to the request that will be forwarded.
///
/// The backend reads `model` out of `request.raw` and rejects anything that is
/// not its own advertised identity (`ensure_requested_model`,
/// `skippy-server/src/frontend/generation/parsing.rs:21-32`), so a selected
/// model that is not written back 404s. This gate previously matched only
/// `None` and `auto`, which left a single-model `mesh` request forwarding
/// `"model":"mesh"` while routing to a concrete target.
///
/// The tokenize guard is load-bearing, not a precaution. Tokenize forces
/// `model_name` to `expected_identity.model_id` (`request_parse.rs:319-330`),
/// and that string could itself be `mesh` or `auto` — which the widened
/// predicate below would match and then rewrite, diverging the wire identity
/// from the authoritative expected identity. `transport.rs` carries the same
/// guard for the same reason.
fn maybe_enable_auto_route_hooks(
    request: &mut proxy::BufferedHttpRequest,
    effective_model: Option<&str>,
) {
    if request.is_tokenize_request() {
        return;
    }
    let is_automatic = request
        .model_name
        .as_deref()
        .is_none_or(automatic::is_directive);
    if !is_automatic {
        return;
    }
    proxy::inject_mesh_hooks_flag(&mut request.raw, true);
    // The directive itself is not a model. Committee mode keeps it in the body
    // for the MoA gateway; only a resolved concrete model gets written back.
    if let Some(model) = effective_model.filter(|name| !automatic::is_directive(name)) {
        proxy::rewrite_model_field(request, model);
    }
}

async fn try_pipeline_proxy(
    node: &mesh::Node,
    tcp_stream: &mut tokio::net::TcpStream,
    request: &mut proxy::BufferedHttpRequest,
    targets: &election::ModelTargets,
    strong_name: &str,
) -> Option<proxy::RouteDispatchOutcome> {
    let (planner_name, planner_port, strong_port) = pipeline_local_ports(targets, strong_name)?;

    request.ensure_body_json();
    let Some(body_json) = request.body_json.clone() else {
        warn_pipeline_fallback(strong_name);
        return None;
    };

    tracing::info!("pipeline: {planner_name} (plan) → {strong_name} (execute)");
    let result = proxy::pipeline_proxy_local(
        tcp_stream,
        &request.path,
        body_json,
        planner_port,
        &planner_name,
        strong_port,
        node,
    )
    .await;
    match result {
        proxy::PipelineProxyResult::Responded(status) => {
            Some(proxy::RouteDispatchOutcome::Responded(status))
        }
        proxy::PipelineProxyResult::RespondedWithUsage { status_code, usage } => {
            Some(proxy::RouteDispatchOutcome::RespondedWithUsage { status_code, usage })
        }
        proxy::PipelineProxyResult::Dropped => Some(proxy::RouteDispatchOutcome::Dropped(
            "pipeline_response_write_failed",
        )),
        proxy::PipelineProxyResult::FallbackToDirect => {
            warn_pipeline_fallback(strong_name);
            None
        }
    }
}

fn pipeline_local_ports(
    targets: &election::ModelTargets,
    strong_name: &str,
) -> Option<(String, u16, u16)> {
    let (planner_name, planner_port) = targets
        .targets
        .iter()
        .find(|(name, target_vec)| {
            *name != strong_name
                && target_vec
                    .iter()
                    .any(|target| matches!(target, election::InferenceTarget::Local(_)))
        })
        .and_then(|(name, target_vec)| {
            target_vec.iter().find_map(|target| match target {
                election::InferenceTarget::Local(port) => Some((name.clone(), *port)),
                _ => None,
            })
        })?;
    let strong_port = targets.targets.get(strong_name).and_then(|target_vec| {
        target_vec.iter().find_map(|target| match target {
            election::InferenceTarget::Local(port) => Some(*port),
            _ => None,
        })
    })?;
    Some((planner_name, planner_port, strong_port))
}

fn warn_pipeline_fallback(strong_name: &str) {
    tracing::warn!("pipeline: falling back to direct proxy for {strong_name}");
}

async fn route_missing_local_model(
    tcp_stream: tokio::net::TcpStream,
    request: &proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    model_name: &str,
    required_tokens: Option<u32>,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    // Try remote mesh first.
    if let Some(mesh_targets) = remote_mesh_targets(ctx, model_name).await {
        return proxy::route_model_request(
            ctx.node.clone(),
            tcp_stream,
            &mesh_targets,
            model_name,
            request,
            proxy::RouteModelRequestContext {
                required_tokens,
                affinity: ctx.affinity,
                route_observer,
            },
        )
        .await;
    }

    // Check if the model is known locally but unavailable
    // (e.g., loading/draining/failed with all-None candidates). Return 503 for these cases so
    // clients can retry; return 404 only when the model truly doesn't exist anywhere.
    if has_local_unavailable_candidates(ctx.targets, model_name) {
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

    // Try plugin dispatch (admission-checked inside).
    if ctx.plugin_manager.is_some() {
        return try_route_plugin_model(ctx, tcp_stream, request, model_name, route_observer).await;
    }

    // Model not found anywhere — return 404. This replaces the old fallback behavior that would
    // route to an arbitrary target, which was confusing when no host served this model.
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

/// Check whether the model is known locally but currently unavailable — all local candidates are None.
fn has_local_unavailable_candidates(targets: &election::ModelTargets, model_name: &str) -> bool {
    let cands = targets.candidates(model_name);
    !cands.is_empty()
        && cands
            .iter()
            .all(|t| matches!(t, election::InferenceTarget::None))
}

async fn remote_mesh_targets(
    ctx: &IngressRouteContext<'_>,
    model_name: &str,
) -> Option<election::ModelTargets> {
    let remote_hosts = ctx.node.hosts_for_model(model_name).await;
    if remote_hosts.is_empty() {
        return None;
    }
    let mut mesh_targets = ctx.targets.clone();
    mesh_targets.targets.insert(
        model_name.to_string(),
        remote_hosts
            .into_iter()
            .map(election::InferenceTarget::Remote)
            .collect(),
    );
    Some(mesh_targets)
}

async fn try_route_plugin_model(
    ctx: &IngressRouteContext<'_>,
    mut tcp_stream: tokio::net::TcpStream,
    request: &proxy::BufferedHttpRequest,
    model_name: &str,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    // Admission check for plugin dispatch ingress.
    match check_activity_admission(
        tcp_stream,
        &ctx.node.activity_policy_guard,
        crate::runtime::IngressType::PluginDispatch,
        route_observer,
    )
    .await
    {
        Ok(stream) => tcp_stream = stream,
        Err(outcome) => {
            route_observer.route_selected_with_metadata(
                Some(model_name),
                Some("plugin"),
                Some("admission"),
            );
            return outcome;
        }
    }

    let plugin_manager = ctx
        .plugin_manager
        .expect("plugin route called without plugin manager");
    match plugin_manager
        .inference_endpoint_for_model(model_name)
        .await
    {
        Ok(Some(endpoint)) => {
            let outcome = proxy::route_http_endpoint_request(
                ctx.node,
                Some(model_name),
                proxy::RouteSelectionMetadata {
                    provider: Some(&endpoint.plugin_name),
                    engine: Some(&endpoint.endpoint_id),
                },
                &mut tcp_stream,
                &endpoint.address,
                request,
                route_observer,
            )
            .await;
            if !outcome.response_written()
                && !matches!(outcome, proxy::RouteDispatchOutcome::Dropped(_))
            {
                response_outcome(
                    503,
                    proxy::send_503_observed(
                        tcp_stream,
                        &format!("plugin endpoint for model '{model_name}' failed"),
                        route_observer,
                    )
                    .await,
                )
            } else {
                outcome
            }
        }
        Ok(None) => {
            route_observer.route_selected_with_metadata(
                Some(model_name),
                Some("plugin"),
                Some("inference_endpoint"),
            );
            // Plugin manager exists but doesn't serve this model — send 404.
            response_outcome(
                404,
                proxy::send_error_observed(
                    tcp_stream,
                    404,
                    &format!(
                        "model '{model_name}' not found (no local or remote host serving this model)"
                    ),
                    route_observer,
                )
                .await,
            )
        }
        Err(error) => {
            tracing::warn!(
                "API proxy: failed to resolve external endpoint for model '{}': {}",
                model_name,
                error
            );
            route_observer.route_selected_with_metadata(
                Some(model_name),
                Some("plugin"),
                Some("inference_endpoint"),
            );
            // Plugin resolution failure — degrade gracefully with 503. The daemon stays alive; only the affected capability is degraded.
            response_outcome(
                503,
                proxy::send_503_observed(
                    tcp_stream,
                    &format!("plugin endpoint for model '{model_name}' unavailable"),
                    route_observer,
                )
                .await,
            )
        }
    }
}

async fn route_request(
    tcp_stream: tokio::net::TcpStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    effective_model: Option<&str>,
    required_tokens: Option<u32>,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    if let Some(model_name) = effective_model {
        // Model explicitly requested. Check local candidates first.
        if !has_available_candidates(ctx.targets, model_name) {
            return route_missing_local_model(
                tcp_stream,
                request,
                ctx,
                model_name,
                required_tokens,
                route_observer,
            )
            .await;
        }

        // Local candidates available — route normally.
        if !request.is_tokenize_request() && ctx.targets.candidates(model_name).len() > 1 {
            request.ensure_body_json();
        }
        proxy::route_model_request(
            ctx.node.clone(),
            tcp_stream,
            ctx.targets,
            model_name,
            request,
            proxy::RouteModelRequestContext {
                required_tokens,
                affinity: ctx.affinity,
                route_observer,
            },
        )
        .await
    } else {
        // No model specified — generic fallback routing to first available target.

        proxy::route_to_target(
            ctx.node.clone(),
            tcp_stream,
            None,
            first_available_target(ctx.targets),
            &request.raw,
            proxy::RouteTargetContext {
                request_id: request.request_id,
                response_adapter: request.response_adapter,
                route_observer,
            },
        )
        .await
    }
}

async fn prepare_auto_route_decision(
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    descriptors: &[crate::mesh::ServedModelDescriptor],
) -> Result<AutoRouteDecision, ()> {
    let required_tokens = proxy::request_context_budget(request);
    match resolve_auto_routed_model(
        ctx.node,
        request,
        ctx.targets,
        ctx.plugin_manager,
        descriptors,
        required_tokens,
        ctx.affinity,
    )
    .await
    {
        AutoRouteResolution::Continue {
            effective_model,
            classification,
        } => {
            maybe_enable_auto_route_hooks(request, effective_model.as_deref());
            if let Some(name) = effective_model.as_ref() {
                ctx.node.record_request(name);
            }
            Ok(AutoRouteDecision {
                effective_model,
                classification,
                required_tokens,
            })
        }
        AutoRouteResolution::MediaUnsupported => Err(()),
    }
}

async fn send_media_unsupported(
    tcp_stream: tokio::net::TcpStream,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    response_outcome(
        422,
        proxy::send_error_observed(
            tcp_stream,
            422,
            "no served model can satisfy the requested media inputs",
            route_observer,
        )
        .await,
    )
}

fn callable_models_with_local_served(
    targets: &election::ModelTargets,
    local_models: Vec<String>,
) -> Vec<String> {
    let mut callable = callable_models(targets);
    for name in local_models {
        if !callable.iter().any(|existing| existing == &name) {
            callable.push(name);
        }
    }
    callable.sort();
    callable
}

async fn maybe_handle_control_request(
    tcp_stream: tokio::net::TcpStream,
    request: &proxy::BufferedHttpRequest,
    ctx: &ProxyConnectionContext<'_>,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<proxy::RouteDispatchOutcome, tokio::net::TcpStream> {
    if proxy::is_legacy_lifecycle_path(&request.path) {
        return Ok(proxy::reject_legacy_lifecycle_request(tcp_stream, route_observer).await);
    }

    if proxy::is_models_list_request(&request.method, &request.path) {
        let outcome = handle_models_list_request(
            tcp_stream,
            ctx.route.node,
            ctx.route.targets,
            ctx.route.plugin_manager,
        )
        .await;
        return Ok(outcome);
    }

    Err(tcp_stream)
}

fn pipeline_route_model<'a>(
    request: &proxy::BufferedHttpRequest,
    decision: &AutoRouteDecision,
    routing_model: Option<&'a str>,
) -> Option<&'a str> {
    let use_pipeline = decision
        .classification
        .as_ref()
        .map(pipeline::should_pipeline)
        .unwrap_or(false)
        && request.response_adapter == proxy::ResponseAdapter::None;
    use_pipeline.then_some(routing_model).flatten()
}

async fn try_pipeline_route(
    tcp_stream: &mut tokio::net::TcpStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    decision: &AutoRouteDecision,
    routing_model: Option<&str>,
) -> Option<proxy::RouteDispatchOutcome> {
    let strong_name = pipeline_route_model(request, decision, routing_model)?;
    try_pipeline_proxy(ctx.node, tcp_stream, request, ctx.targets, strong_name).await
}

enum MoaInterceptResult {
    /// MoA handled the request; the response has been written and the stream
    /// is consumed.
    Handled(proxy::RouteDispatchOutcome),
    /// Not an MoA request — caller should continue with normal routing,
    /// reusing the returned stream.
    NotMoa(tokio::net::TcpStream),
    /// MoA could not form a committee but degraded `model=mesh` to a real
    /// single model (already rewritten on the request). Caller routes it
    /// normally, but must use this model rather than the stale
    /// `decision.effective_model` (still "mesh").
    Degraded {
        stream: tokio::net::TcpStream,
        model: Option<String>,
    },
}

/// Dispatch to the MoA gateway when `model == "mesh"`. Self-gates on the
/// effective model so the call site is unconditional.
async fn try_handle_moa_intercept(
    tcp_stream: tokio::net::TcpStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &ProxyConnectionContext<'_>,
    decision: &AutoRouteDecision,
    route_observer: OpenAiRouteObserver<'_>,
) -> MoaInterceptResult {
    if decision.effective_model.as_deref() != Some(moa::VIRTUAL_MODEL_NAME) {
        return MoaInterceptResult::NotMoa(tcp_stream);
    }
    // `try_handle_moa` self-gates on the model name and consumes the
    // stream when it accepts. The outer gate above guarantees the gate
    // matches, so the inner call always returns `None` here — the stream
    // is gone, either with the MoA response, a 503, or a 400. Discard
    // the return value explicitly. The previous shape kept an
    // `if let Some(_) = … { tracing::error!(...) }` branch that could
    // never fire and made the control flow confusing to read.
    let result = crate::network::openai::moa_gateway::try_handle_moa(
        ctx.route.node,
        tcp_stream,
        request,
        decision.effective_model.as_deref(),
        Some(ctx.route.targets),
        decision.required_tokens,
        route_observer,
    )
    .await;
    match result {
        crate::network::openai::moa_gateway::MoaDispatchResult::Passthrough(stream) => {
            // The gateway hands the stream back in two cases: the request was
            // never MoA-shaped, or MoA degraded `model=mesh` to a real single
            // model by rewriting the request in place. The outer gate above
            // guarantees we got here with `effective_model == "mesh"`, so this
            // is the degrade case: routing must use the rewritten model, not
            // the stale decision.
            MoaInterceptResult::Degraded {
                stream,
                model: request.model_name.clone(),
            }
        }
        crate::network::openai::moa_gateway::MoaDispatchResult::Responded(status) => {
            proxy::release_request_objects(ctx.route.node, &request.request_object_request_ids)
                .await;
            MoaInterceptResult::Handled(proxy::RouteDispatchOutcome::Responded(status))
        }
        crate::network::openai::moa_gateway::MoaDispatchResult::RespondedWithUsage {
            status_code,
            usage,
        } => {
            proxy::release_request_objects(ctx.route.node, &request.request_object_request_ids)
                .await;
            MoaInterceptResult::Handled(proxy::RouteDispatchOutcome::RespondedWithUsage {
                status_code,
                usage,
            })
        }
        crate::network::openai::moa_gateway::MoaDispatchResult::FailedWithStatus {
            status_code,
            reason,
        } => {
            proxy::release_request_objects(ctx.route.node, &request.request_object_request_ids)
                .await;
            MoaInterceptResult::Handled(proxy::RouteDispatchOutcome::FailedWithStatus {
                status_code,
                reason,
            })
        }
        crate::network::openai::moa_gateway::MoaDispatchResult::Dropped(reason) => {
            proxy::release_request_objects(ctx.route.node, &request.request_object_request_ids)
                .await;
            MoaInterceptResult::Handled(proxy::RouteDispatchOutcome::Dropped(reason))
        }
    }
}

async fn handle_buffered_api_request(
    tcp_stream: tokio::net::TcpStream,
    mut request: proxy::BufferedHttpRequest,
    ctx: ProxyConnectionContext<'_>,
) {
    // Claim the parent at host OpenAI ingress. All downstream dispatch sees
    // only a metadata observer; this scope remains the sole terminal owner.
    let request_metadata =
        crate::logging::RequestSummaryMetadata::from_openai_ingress_path(&request.client_path)
            .with_source(Some("direct_http"))
            .with_method(Some(&request.method));
    let mut lifecycle = crate::logging_runtime_state()
        .map(|state| state.openai_ingress_attachment(request.request_id, request_metadata))
        .unwrap_or_else(OpenAiLifecycleAttachment::unowned);
    if lifecycle.owns_parent() {
        request.mark_raw_lifecycle_owned();
        if let Some(body) = request.body_bytes.as_deref() {
            lifecycle.capture_request_body(body, request.artifact_request_media_kind());
        }
    }

    let tcp_stream =
        match maybe_handle_control_request(tcp_stream, &request, &ctx, lifecycle.route_observer())
            .await
        {
            Ok(outcome) => {
                lifecycle.terminal(terminal_outcome_for_dispatch(outcome));
                return;
            }
            Err(tcp_stream) => tcp_stream,
        };

    let local_models = ctx.route.node.models_being_served().await;
    let callable = callable_models_with_local_served(ctx.route.targets, local_models);
    let descriptors = ctx.route.node.all_served_model_descriptors().await;
    proxy::rewrite_public_model_alias(&mut request, &callable, &descriptors);

    // Admission applies to inference work after control-path rejection.
    let tcp_stream = match check_activity_admission(
        tcp_stream,
        &ctx.route.node.activity_policy_guard,
        crate::runtime::IngressType::LocalOpenAi,
        lifecycle.route_observer(),
    )
    .await
    {
        Ok(stream) => stream,
        Err(outcome) => {
            lifecycle.terminal(terminal_outcome_for_dispatch(outcome));
            return;
        }
    };

    let decision = match prepare_auto_route_decision(&mut request, &ctx.route, &descriptors).await {
        Ok(decision) => decision,
        Err(()) => {
            let outcome = send_media_unsupported(tcp_stream, lifecycle.route_observer()).await;
            lifecycle.terminal(terminal_outcome_for_dispatch(outcome));
            return;
        }
    };

    let mut routing_model = decision.effective_model.clone();
    let tcp_stream = match try_handle_moa_intercept(
        tcp_stream,
        &mut request,
        &ctx,
        &decision,
        lifecycle.route_observer(),
    )
    .await
    {
        MoaInterceptResult::Handled(outcome) => {
            proxy::record_moa_stream_lifecycle(
                lifecycle.route_observer(),
                request.response_adapter,
                outcome,
            );
            lifecycle.terminal(terminal_outcome_for_dispatch(outcome));
            return;
        }
        MoaInterceptResult::NotMoa(stream) => stream,
        MoaInterceptResult::Degraded { stream, model } => {
            routing_model = model;
            stream
        }
    };

    let mut tcp_stream = tcp_stream;
    if let Some(outcome) = try_pipeline_route(
        &mut tcp_stream,
        &mut request,
        &ctx.route,
        &decision,
        routing_model.as_deref(),
    )
    .await
    {
        proxy::release_request_objects(ctx.route.node, &request.request_object_request_ids).await;
        lifecycle.terminal(terminal_outcome_for_dispatch(outcome));
        return;
    }

    let outcome = {
        let route_observer = lifecycle.route_observer();
        route_request(
            tcp_stream,
            &mut request,
            &ctx.route,
            routing_model.as_deref(),
            decision.required_tokens,
            route_observer,
        )
        .await
    };
    if let Some(model) = routing_model.as_deref()
        && !request.is_tokenize_request()
    {
        let mut event =
            audit_events::model_access(None, model, "route", model_access_succeeded(outcome));
        if let Some(cid) = request.correlation_id.as_deref() {
            event = event.with_metadata("request_id", serde_json::Value::String(cid.to_string()));
        }
        let _ = emit_audit(event);
    }
    proxy::release_request_objects(ctx.route.node, &request.request_object_request_ids).await;
    lifecycle.terminal(terminal_outcome_for_dispatch(outcome));
}

async fn handle_api_proxy_connection(
    node: mesh::Node,
    mut tcp_stream: tokio::net::TcpStream,
    targets: election::ModelTargets,
    affinity: affinity::AffinityRouter,
) {
    let plugin_manager = node.plugin_manager().await;
    match proxy::read_http_request_with_plugin_manager_with_context(
        &mut tcp_stream,
        plugin_manager.as_ref(),
    )
    .await
    {
        Ok(request) => {
            let route = IngressRouteContext {
                node: &node,
                targets: &targets,
                affinity: &affinity,
                plugin_manager: plugin_manager.as_ref(),
            };
            handle_buffered_api_request(tcp_stream, request, ProxyConnectionContext { route })
                .await;
        }
        Err(error) => {
            let _ = super::parse_failure::send_read_failure(tcp_stream, &error).await;
        }
    }
}

/// Model-aware API proxy. Parses the "model" field from POST request bodies
/// and routes to the correct host. Falls back to the first available target
/// if model is not specified or not found.
pub(crate) async fn api_proxy(
    node: mesh::Node,
    port: u16,
    target_rx: tokio::sync::watch::Receiver<election::ModelTargets>,
    existing_listener: Option<tokio::net::TcpListener>,
    listen_all: bool,
    affinity: affinity::AffinityRouter,
) {
    let Some(listener) = bind_api_proxy_listener(port, existing_listener, listen_all).await else {
        return;
    };

    loop {
        let (tcp_stream, _addr) = match listener.accept().await {
            Ok(r) => r,
            Err(_) => break,
        };
        let _ = tcp_stream.set_nodelay(true);

        let targets = target_rx.borrow().clone();
        let node = node.clone();
        let affinity = affinity.clone();
        tokio::spawn(async move {
            handle_api_proxy_connection(node, tcp_stream, targets, affinity).await;
        });
    }
}

/// Bootstrap proxy: runs during GPU startup, tunnels all requests to mesh hosts.
/// Returns the TcpListener when signaled to stop (so api_proxy can take it over).
pub(crate) async fn bootstrap_proxy(
    node: mesh::Node,
    port: u16,
    mut stop_rx: tokio::sync::mpsc::Receiver<tokio::sync::oneshot::Sender<tokio::net::TcpListener>>,
    listen_all: bool,
    affinity: affinity::AffinityRouter,
) {
    let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
    let listener = match tokio::net::TcpListener::bind(format!("{addr}:{port}")).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Bootstrap proxy: failed to bind to port {port}: {e}");
            return;
        }
    };
    let _ = emit_event(OutputEvent::Info {
        message: format!("API ready (bootstrap): http://localhost:{port}"),
        context: Some("bootstrap_proxy".to_string()),
    });
    let _ = emit_event(OutputEvent::Info {
        message: "Requests tunneled to mesh while GPU loads...".to_string(),
        context: Some("bootstrap_proxy".to_string()),
    });

    loop {
        tokio::select! {
            accept = listener.accept() => {
                let (tcp_stream, _addr) = match accept {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                let _ = tcp_stream.set_nodelay(true);
                let node = node.clone();
                let affinity = affinity.clone();
                tokio::spawn(Box::pin(proxy::handle_mesh_request(node, tcp_stream, true, affinity)));
            }
            resp_tx = stop_rx.recv() => {
                if let Some(tx) = resp_tx {
                    let _ = emit_event(OutputEvent::Info {
                        message: "Bootstrap proxy handing off to full API proxy".to_string(),
                        context: Some("bootstrap_proxy".to_string()),
                    });
                    let _ = tx.send(listener);
                }
                return;
            }
        }
    }
}

fn first_available_target(targets: &election::ModelTargets) -> election::InferenceTarget {
    for hosts in targets.targets.values() {
        for target in hosts {
            if !matches!(target, election::InferenceTarget::None) {
                return target.clone();
            }
        }
    }
    election::InferenceTarget::None
}

fn has_available_candidates(targets: &election::ModelTargets, model: &str) -> bool {
    contains_routable_candidate(&targets.candidates(model))
}

fn contains_routable_candidate(candidates: &[election::InferenceTarget]) -> bool {
    candidates
        .iter()
        .any(|target| !matches!(target, election::InferenceTarget::None))
}

pub(crate) fn callable_models(targets: &election::ModelTargets) -> Vec<String> {
    let mut models: Vec<String> = targets
        .targets
        .iter()
        .filter(|(_, hosts)| {
            hosts
                .iter()
                .any(|target| !matches!(target, election::InferenceTarget::None))
        })
        .map(|(name, _)| name.clone())
        .collect();
    models.sort();
    models
}

#[cfg(test)]
#[path = "ingress_tests/durable_artifacts.rs"]
mod durable_artifacts;

#[cfg(test)]
#[path = "ingress_tests/automatic_routing.rs"]
mod automatic_routing;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::logging::{
        LoggingService, OpenAiLifecycleAttachment, RawMeshLifecycleOwners, RawMeshRequestLifecycle,
        TerminalOutcome,
    };
    use mesh_llm_events::logging::{events::LifecycleEvent, identifiers::RequestId};

    use super::*;

    fn large_tokenize_request(model: &str) -> proxy::BufferedHttpRequest {
        proxy::BufferedHttpRequest {
            raw: b"unchanged tokenizer wire".to_vec(),
            method: "POST".to_owned(),
            path: "/v1/tokenize".to_owned(),
            client_path: "/v1/tokenize".to_owned(),
            request_id: RequestId::default(),
            body_json: None,
            body_json_attempted: false,
            body_bytes: None,
            body_len_bytes: 140_000,
            completion_tokens: None,
            stream: None,
            model_name: Some(model.to_owned()),
            request_object_request_ids: Vec::new(),
            response_adapter: proxy::ResponseAdapter::None,
            correlation_id: None,
        }
    }

    fn recorded_lifecycle_events(service: &LoggingService) -> Vec<LifecycleEvent> {
        service
            .bus_ref()
            .replay_window()
            .records
            .into_iter()
            .filter_map(|record| {
                let envelope =
                    serde_json::from_str::<serde_json::Value>(&record.entry.payload).ok()?;
                let payload = envelope.get("payload")?.as_str()?;
                serde_json::from_str(payload).ok()
            })
            .collect()
    }

    fn plugin_lifecycle() -> (Arc<LoggingService>, OpenAiLifecycleAttachment) {
        let service = Arc::new(LoggingService::new_disabled(Default::default()));
        let parent = RawMeshRequestLifecycle::register(
            Arc::clone(&service),
            Arc::new(RawMeshLifecycleOwners::default()),
            RequestId::new(),
        )
        .expect("plugin test should claim one parent");
        (service, OpenAiLifecycleAttachment::new(Some(parent)))
    }

    fn record_plugin_attempt(
        observer: OpenAiRouteObserver<'_>,
        model: &str,
        provider: &str,
        engine: &str,
        result: super::super::response::RouteAttemptResult,
    ) -> super::super::response::RouteAttemptResult {
        observer.route_selected_with_metadata(Some(model), Some(provider), Some(engine));
        let attempt_id = observer.start_attempt();
        match result {
            super::super::response::RouteAttemptResult::Delivered { status_code, .. } => {
                observer.complete_attempt(attempt_id, status_code);
            }
            _ => observer.fail_attempt(
                attempt_id,
                super::super::response::route_attempt_result_label(&result),
            ),
        }
        result
    }

    fn assert_payload_free(events: &[LifecycleEvent]) {
        let serialized = serde_json::to_string(events).expect("events should serialize");
        for forbidden in [
            "body",
            "headers",
            "prompt",
            "authorization",
            "secret",
            "completion",
        ] {
            assert!(!serialized.to_ascii_lowercase().contains(forbidden));
        }
    }

    /// A model nobody serves must not stay in the auto pool. Before this,
    /// the readiness check fell through to `true`, so `auto` could select a
    /// phantom (stale gossip, or a peer that unloaded) and 404 the caller on
    /// a model they never named.
    #[tokio::test]
    async fn phantom_model_is_not_auto_route_eligible() {
        let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
            .await
            .expect("test node");
        let targets = election::ModelTargets::default();
        let affinity = affinity::AffinityRouter::new();

        let eligible = auto_route_model_has_ready_ingress_target(
            &node,
            &targets,
            "phantom/model:Q4_K_M",
            None,
            &affinity,
        )
        .await;

        assert!(
            !eligible,
            "a model with no local target and no remote host must not be auto-route eligible"
        );
    }

    /// A freshly started serve node records its model before the target table
    /// and gossip catch up. Failing closed must not exclude this node's own
    /// model during that window.
    #[tokio::test]
    async fn freshly_served_local_model_is_auto_route_eligible() {
        let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
            .await
            .expect("test node");
        node.set_hosted_models(vec!["local/fresh-model:Q4_K_M".to_string()])
            .await;
        let targets = election::ModelTargets::default();
        let affinity = affinity::AffinityRouter::new();

        let eligible = auto_route_model_has_ready_ingress_target(
            &node,
            &targets,
            "local/fresh-model:Q4_K_M",
            None,
            &affinity,
        )
        .await;

        assert!(
            eligible,
            "a locally served model must stay eligible before targets populate"
        );
    }

    #[test]
    fn parse_model_with_profile_with_named_profile() {
        let (model_ref, profile) = parse_model_with_profile("Qwen3-8B#low-ctx");
        assert_eq!(model_ref, "Qwen3-8B");
        assert_eq!(profile, "low-ctx");
    }

    #[test]
    fn parse_model_with_profile_without_profile() {
        let (model_ref, profile) = parse_model_with_profile("Qwen3-8B");
        assert_eq!(model_ref, "Qwen3-8B");
        assert_eq!(profile, "");
    }

    #[test]
    fn parse_model_with_profile_empty_profile_after_hash() {
        let (model_ref, profile) = parse_model_with_profile("Qwen3-8B#");
        assert_eq!(model_ref, "Qwen3-8B");
        assert_eq!(profile, "");
    }

    #[test]
    fn parse_model_with_profile_huggingface_ref_with_quant() {
        let (model_ref, profile) = parse_model_with_profile("org/repo:Q4_K_M#profile");
        assert_eq!(model_ref, "org/repo:Q4_K_M");
        assert_eq!(profile, "profile");
    }

    #[test]
    fn parse_model_with_profile_multiple_hashes_uses_last() {
        let (model_ref, profile) = parse_model_with_profile("model#with#hash#profile");
        assert_eq!(model_ref, "model#with#hash");
        assert_eq!(profile, "profile");
    }

    /// Regression: the MoA intercept must surface a single-model degradation.
    ///
    /// This helper-level test verifies that the gateway rewrites the request
    /// and returns the resolved model to its caller. The separate pipeline
    /// regression below verifies that the caller actually consumes that model.
    #[tokio::test]
    async fn moa_single_model_degrade_rewrites_routing_model() {
        let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
            .await
            .expect("test node");
        node.set_hosted_models(vec!["local/only-model:Q4_K_M".to_string()])
            .await;
        let mut targets = election::ModelTargets::default();
        targets.targets.insert(
            "local/only-model:Q4_K_M".to_string(),
            vec![election::InferenceTarget::Local(1)],
        );
        let affinity = affinity::AffinityRouter::new();

        // The helper returns the connected stream; this test only inspects the
        // degradation result and intentionally does not dispatch it.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        let client = tokio::net::TcpStream::connect(addr);
        let server = async { listener.accept().await.map(|(stream, _)| stream) };
        let (_client_side, server_side) = tokio::join!(client, server);
        let tcp_stream = server_side.expect("accept");

        let body = br#"{"model":"mesh","messages":[{"role":"user","content":"hi"}]}"#;
        let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .into_bytes()
        .into_iter()
        .chain(body.iter().copied())
        .collect::<Vec<u8>>();
        let mut request = proxy::BufferedHttpRequest {
            raw,
            method: "POST".to_owned(),
            path: "/v1/chat/completions".to_owned(),
            client_path: "/v1/chat/completions".to_owned(),
            request_id: RequestId::default(),
            body_json: None,
            body_json_attempted: false,
            body_bytes: None,
            body_len_bytes: body.len(),
            completion_tokens: None,
            stream: None,
            model_name: Some("mesh".to_owned()),
            request_object_request_ids: Vec::new(),
            response_adapter: proxy::ResponseAdapter::OpenAiChatCompletionsJson,
            correlation_id: None,
        };
        let decision = AutoRouteDecision {
            effective_model: Some("mesh".to_owned()),
            classification: None,
            required_tokens: None,
        };
        let ctx = ProxyConnectionContext {
            route: IngressRouteContext {
                node: &node,
                targets: &targets,
                affinity: &affinity,
                plugin_manager: None,
            },
        };
        let lifecycle = OpenAiLifecycleAttachment::unowned();

        let result = try_handle_moa_intercept(
            tcp_stream,
            &mut request,
            &ctx,
            &decision,
            lifecycle.route_observer(),
        )
        .await;

        match result {
            MoaInterceptResult::Degraded { model, .. } => {
                assert_eq!(
                    model.as_deref(),
                    Some("local/only-model:Q4_K_M"),
                    "degrade must carry the rewritten real model for routing"
                );
                assert_eq!(
                    request.model_name.as_deref(),
                    Some("local/only-model:Q4_K_M"),
                    "request must be rewritten in place"
                );
            }
            MoaInterceptResult::NotMoa(_) => {
                panic!(
                    "single-model model=mesh fell through as NotMoa — routing would use the stale \
                     'mesh' model and 404 (#1175 regression)"
                );
            }
            MoaInterceptResult::Handled(outcome) => {
                panic!("expected degrade passthrough, got handled outcome: {outcome:?}");
            }
        }
    }

    #[test]
    fn moa_degraded_model_is_consumed_by_pipeline_dispatch() {
        use crate::network::router::{Category, Classification, Complexity};

        let request = proxy::BufferedHttpRequest {
            raw: Vec::new(),
            method: "POST".to_owned(),
            path: "/v1/chat/completions".to_owned(),
            client_path: "/v1/chat/completions".to_owned(),
            request_id: RequestId::default(),
            body_json: None,
            body_json_attempted: false,
            body_bytes: None,
            body_len_bytes: 0,
            completion_tokens: None,
            stream: None,
            model_name: Some("local/only-model:Q4_K_M".to_owned()),
            request_object_request_ids: Vec::new(),
            response_adapter: proxy::ResponseAdapter::None,
            correlation_id: None,
        };
        let decision = AutoRouteDecision {
            effective_model: Some("mesh".to_owned()),
            classification: Some(Classification {
                category: Category::Code,
                complexity: Complexity::Deep,
                needs_tools: true,
                has_media_inputs: false,
            }),
            required_tokens: None,
        };

        assert_eq!(
            pipeline_route_model(&request, &decision, request.model_name.as_deref(),),
            Some("local/only-model:Q4_K_M"),
            "pipeline dispatch must consume the post-degradation model, not stale 'mesh'"
        );
    }

    // --- Routing behavior tests for model-independent daemon support ---

    #[test]
    fn has_local_unavailable_returns_false_for_empty_targets() {
        let targets = election::ModelTargets::default();
        assert!(!has_local_unavailable_candidates(&targets, "nonexistent"));
    }

    #[test]
    fn has_local_unavailable_returns_true_when_all_none() {
        let mut targets = election::ModelTargets::default();
        targets.targets.insert(
            "loading-model".to_string(),
            vec![
                election::InferenceTarget::None,
                election::InferenceTarget::None,
            ],
        );
        assert!(has_local_unavailable_candidates(&targets, "loading-model"));
    }

    #[test]
    fn has_local_unavailable_returns_false_when_any_available() {
        let mut targets = election::ModelTargets::default();
        targets.targets.insert(
            "partial-model".to_string(),
            vec![
                election::InferenceTarget::None,
                election::InferenceTarget::Local(9337),
            ],
        );
        assert!(!has_local_unavailable_candidates(&targets, "partial-model"));
    }

    #[test]
    fn callable_models_excludes_all_none_targets() {
        let mut targets = election::ModelTargets::default();
        // Available model - included in callable list
        targets.targets.insert(
            "available".to_string(),
            vec![election::InferenceTarget::Local(9337)],
        );
        // Unavailable model (loading/draining) - excluded from callable list
        targets.targets.insert(
            "unavailable".to_string(),
            vec![
                election::InferenceTarget::None,
                election::InferenceTarget::None,
            ],
        );

        let models = callable_models(&targets);
        assert!(models.contains(&"available".to_string()));
        assert!(!models.contains(&"unavailable".to_string()));
    }

    #[test]
    fn callable_models_returns_empty_when_no_targets() {
        let targets = election::ModelTargets::default();
        let models = callable_models(&targets);
        assert!(models.is_empty());
    }

    // --- Daemon state derivation tests for plugin-only and remote-only daemons ---

    #[test]
    fn daemon_ready_proxying_when_only_plugins_available() {
        use crate::api::status::{DaemonState, derive_daemon_state};

        assert_eq!(
            derive_daemon_state(
                false, // shutdown_requested
                false, // has_terminal_failure
                false, // priority_degraded
                false, // local_serving - no local models
                true,  // proxying - plugin endpoints available for routing
                true,  // listeners_ready
            ),
            DaemonState::ReadyProxying,
        );
    }

    #[test]
    fn daemon_ready_proxying_when_only_remote_mesh_available() {
        use crate::api::status::{DaemonState, derive_daemon_state};

        assert_eq!(
            derive_daemon_state(
                false, // shutdown_requested
                false, // has_terminal_failure
                false, // priority_degraded
                false, // local_serving - no local models
                true,  // proxying - remote mesh targets available for routing
                true,  // listeners_ready
            ),
            DaemonState::ReadyProxying,
        );
    }

    #[test]
    fn daemon_degraded_on_terminal_failure_not_killed() {
        use crate::api::status::{DaemonState, derive_daemon_state};

        assert_eq!(
            derive_daemon_state(
                false, // shutdown_requested - NOT stopping
                true,  // has_terminal_failure - model failed
                false, // priority_degraded
                false, // local_serving
                true,  // proxying still works for other capabilities
                true,  // listeners_ready
            ),
            DaemonState::Degraded,
        );
    }

    #[test]
    fn daemon_stopping_only_when_shutdown_requested() {
        use crate::api::status::{DaemonState, derive_daemon_state};

        assert_eq!(
            derive_daemon_state(
                true,  // shutdown_requested - explicitly stopping
                false, // has_terminal_failure
                false, // priority_degraded
                true,  // local_serving (irrelevant when stopping)
                true,  // proxying (irrelevant when stopping)
                true,  // listeners_ready
            ),
            DaemonState::Stopping,
        );
    }

    #[test]
    fn daemon_ready_idle_when_no_models_but_listeners_up() {
        use crate::api::status::{DaemonState, derive_daemon_state};

        assert_eq!(
            derive_daemon_state(
                false, // shutdown_requested
                false, // has_terminal_failure
                false, // priority_degraded
                false, // local_serving - no models loaded yet (on-demand mode)
                false, // proxying - not yet routing to mesh or plugins
                true,  // listeners_ready - HTTP listeners are up and accepting connections
            ),
            DaemonState::ReadyIdle,
        );
    }

    #[tokio::test]
    async fn api_proxy_tokenizer_route_ignores_generation_context_budget() {
        let model = "acme/code-model:Q4_K_M";
        let mut request = large_tokenize_request(model);
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
            .await
            .expect("test node should start");
        node.set_model_runtime_context_length(model, Some(32_768))
            .await;
        let target = election::InferenceTarget::Local(19_337);
        let mut targets = election::ModelTargets::default();
        targets
            .targets
            .insert(model.to_owned(), vec![target.clone()]);
        let affinity = affinity::AffinityRouter::new();
        let ctx = IngressRouteContext {
            node: &node,
            targets: &targets,
            affinity: &affinity,
            plugin_manager: None,
        };
        let raw_before_decision = request.raw.clone();

        let generation_budget = proxy::request_budget_tokens_from_parts(
            request.body_len_bytes,
            request.completion_tokens,
        );
        assert!(generation_budget.is_some_and(|tokens| tokens > 32_768));
        assert!(
            crate::network::openai::routing_rank::order_targets_by_context(
                &node,
                model,
                generation_budget,
                std::slice::from_ref(&target),
            )
            .await
            .is_empty(),
            "a generation budget would incorrectly reject the tokenizer target"
        );

        let decision = prepare_auto_route_decision(&mut request, &ctx, &[])
            .await
            .expect("tokenizer route should not enter media auto-routing");
        assert_eq!(decision.effective_model.as_deref(), Some(model));
        assert_eq!(decision.required_tokens, None);
        assert_eq!(request.raw, raw_before_decision);
        assert!(request.body_json.is_none());
        assert!(!request.body_json_attempted);
        assert_eq!(proxy::request_context_budget(&request), None);
        assert_eq!(
            crate::network::openai::routing_rank::order_targets_by_context(
                &node,
                model,
                proxy::request_context_budget(&request),
                std::slice::from_ref(&target),
            )
            .await,
            vec![target]
        );
    }

    #[test]
    fn plugin_route_success_records_one_attempt_and_one_terminal_outcome() {
        let (service, mut attachment) = plugin_lifecycle();
        let observer = attachment.route_observer();
        let result = record_plugin_attempt(
            observer,
            "plugin-model",
            "acme/plugin",
            "endpoint-prod",
            super::super::response::RouteAttemptResult::Delivered {
                status_code: 200,
                usage: None,
            },
        );
        assert!(matches!(
            result,
            super::super::response::RouteAttemptResult::Delivered {
                status_code: 200,
                ..
            }
        ));

        attachment.terminal(terminal_outcome_for_dispatch(
            proxy::RouteDispatchOutcome::Responded(200),
        ));
        attachment.terminal(TerminalOutcome::Failed("late_plugin_failure".into()));

        let events = recorded_lifecycle_events(&service);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, LifecycleEvent::RouteSelected { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, LifecycleEvent::AttemptStarted { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, LifecycleEvent::AttemptCompleted { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, LifecycleEvent::Completed { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, LifecycleEvent::Failed { .. }))
                .count(),
            0
        );
        match events
            .iter()
            .find(|event| matches!(event, LifecycleEvent::RouteSelected { .. }))
        {
            Some(LifecycleEvent::RouteSelected {
                model,
                provider,
                engine,
            }) => {
                assert_eq!(model.as_deref(), Some("plugin-model"));
                assert_eq!(provider.as_deref(), Some("acme/plugin"));
                assert_eq!(engine.as_deref(), Some("endpoint-prod"));
            }
            other => panic!("expected one plugin route selection, got {other:?}"),
        }
        assert_payload_free(&events);
    }

    #[test]
    fn plugin_route_failure_records_failed_attempt_and_terminal_outcome() {
        let (service, mut attachment) = plugin_lifecycle();
        let observer = attachment.route_observer();
        let result = record_plugin_attempt(
            observer,
            "plugin-model",
            "plugin.example",
            "sk_test",
            super::super::response::RouteAttemptResult::RetryableUnavailable,
        );
        assert_eq!(
            result,
            super::super::response::RouteAttemptResult::RetryableUnavailable
        );

        attachment.terminal(terminal_outcome_for_dispatch(
            proxy::RouteDispatchOutcome::Failed("plugin_endpoint_failed"),
        ));
        attachment.terminal(TerminalOutcome::Completed);

        let events = recorded_lifecycle_events(&service);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, LifecycleEvent::AttemptStarted { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, LifecycleEvent::AttemptFailed { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, LifecycleEvent::Failed { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, LifecycleEvent::Completed { .. }))
                .count(),
            0
        );
        match events
            .iter()
            .find(|event| matches!(event, LifecycleEvent::AttemptFailed { .. }))
        {
            Some(LifecycleEvent::AttemptFailed { error, .. }) => {
                assert_eq!(error.as_deref(), Some("retryable_unavailable"));
            }
            other => panic!("expected one plugin attempt failure, got {other:?}"),
        }
        assert_payload_free(&events);
    }

    #[test]
    fn plugin_route_without_endpoint_records_decision_without_attempt_or_payload() {
        let (service, mut attachment) = plugin_lifecycle();
        let observer = attachment.route_observer();
        observer.route_selected_with_metadata(
            Some("plugin-model"),
            Some("plugin"),
            Some("inference_endpoint"),
        );
        attachment.terminal(terminal_outcome_for_dispatch(
            proxy::RouteDispatchOutcome::Responded(404),
        ));
        attachment.terminal(TerminalOutcome::Failed("late_plugin_failure".into()));

        let events = recorded_lifecycle_events(&service);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, LifecycleEvent::RouteSelected { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, LifecycleEvent::AttemptStarted { .. }))
                .count(),
            0
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, LifecycleEvent::Rejected { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, LifecycleEvent::Failed { .. }))
                .count(),
            0
        );
        assert_payload_free(&events);
    }

    #[test]
    fn load_and_unload_error_statuses_never_complete_lifecycle() {
        for status in [400, 404, 409, 500, 503] {
            assert!(!matches!(
                terminal_outcome_for_dispatch(proxy::RouteDispatchOutcome::Responded(status)),
                TerminalOutcome::Completed
            ));
        }
    }

    #[test]
    fn unknown_and_unavailable_models_map_to_rejected_and_failed() {
        assert!(matches!(
            terminal_outcome_for_dispatch(proxy::RouteDispatchOutcome::Responded(404)),
            TerminalOutcome::RejectedWithStatus {
                status_code: 404,
                ..
            }
        ));
        assert!(matches!(
            terminal_outcome_for_dispatch(proxy::RouteDispatchOutcome::Responded(503)),
            TerminalOutcome::FailedWithStatus {
                status_code: 503,
                ..
            }
        ));
    }

    #[test]
    fn invalid_and_failed_moa_responses_map_from_http_status() {
        assert!(matches!(
            terminal_outcome_for_dispatch(proxy::RouteDispatchOutcome::Responded(400)),
            TerminalOutcome::RejectedWithStatus {
                status_code: 400,
                ..
            }
        ));
        assert!(matches!(
            terminal_outcome_for_dispatch(proxy::RouteDispatchOutcome::Responded(502)),
            TerminalOutcome::FailedWithStatus {
                status_code: 502,
                ..
            }
        ));
    }

    #[test]
    fn usage_never_turns_moa_or_pipeline_error_statuses_into_success() {
        let usage = mesh_llm_events::logging::events::TokenUsage {
            prompt_tokens: Some(8),
            completion_tokens: Some(5),
            total_tokens: Some(13),
        };
        assert!(matches!(
            terminal_outcome_for_dispatch(proxy::RouteDispatchOutcome::RespondedWithUsage {
                status_code: 400,
                usage,
            }),
            TerminalOutcome::RejectedWithStatus {
                status_code: 400,
                ..
            }
        ));
        assert!(matches!(
            terminal_outcome_for_dispatch(proxy::RouteDispatchOutcome::RespondedWithUsage {
                status_code: 502,
                usage,
            }),
            TerminalOutcome::FailedWithStatus {
                status_code: 502,
                ..
            }
        ));
    }

    #[test]
    fn streamed_moa_chat_and_responses_record_compatible_usage_lifecycle() {
        let usage = mesh_llm_events::logging::events::TokenUsage {
            prompt_tokens: Some(8),
            completion_tokens: Some(5),
            total_tokens: Some(13),
        };
        for adapter in [
            proxy::ResponseAdapter::OpenAiChatCompletionsStream,
            proxy::ResponseAdapter::OpenAiResponsesStream,
        ] {
            let (service, mut attachment) = plugin_lifecycle();
            let outcome = proxy::RouteDispatchOutcome::RespondedWithUsage {
                status_code: 200,
                usage,
            };
            proxy::record_moa_stream_lifecycle(attachment.route_observer(), adapter, outcome);
            attachment.terminal(terminal_outcome_for_dispatch(outcome));

            let events = recorded_lifecycle_events(&service);
            assert!(events.iter().any(|event| matches!(
                event,
                LifecycleEvent::StreamStarted { model }
                    if model.as_deref() == Some(moa::VIRTUAL_MODEL_NAME)
            )));
            assert!(events.iter().any(|event| matches!(
                event,
                LifecycleEvent::StreamCompleted {
                    tokens: Some(5),
                    usage: Some(recorded),
                } if *recorded == usage
            )));
            assert!(events.iter().any(|event| matches!(
                event,
                LifecycleEvent::Completed {
                    status_code: Some(200),
                    usage: Some(recorded),
                    ..
                } if *recorded == usage
            )));
        }
    }

    #[test]
    fn pipeline_server_error_is_failed_not_completed() {
        assert!(matches!(
            terminal_outcome_for_dispatch(proxy::RouteDispatchOutcome::Responded(500)),
            TerminalOutcome::FailedWithStatus {
                status_code: 500,
                ..
            }
        ));
    }

    #[test]
    fn disconnect_is_dropped_and_cannot_audit_model_access_as_success() {
        let outcome = proxy::RouteDispatchOutcome::Dropped("client_disconnected");
        assert!(matches!(
            terminal_outcome_for_dispatch(outcome),
            TerminalOutcome::Dropped(_)
        ));
        assert!(!model_access_succeeded(outcome));
        assert!(!model_access_succeeded(
            proxy::RouteDispatchOutcome::Responded(502)
        ));
        assert!(model_access_succeeded(
            proxy::RouteDispatchOutcome::Responded(200)
        ));
    }
}
