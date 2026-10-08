use crate::logging::OpenAiRouteObserver;
use crate::mesh;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::transport as proxy;
use crate::plugin::{BridgeFuture, PluginManager, PluginRpcBridge, RpcResult, proto};
use mesh_llm_events::logging::events::TokenUsage;
use mesh_llm_plugin::{HostInferenceRequest, HostInferenceResponse};
use mesh_llm_plugin::{VirtualModelCandidate, VirtualModelInvocation};
use std::sync::Arc;

mod exchange_admission;
mod progress;
mod stream_adapters;

pub(crate) fn advertisable_routes(
    routes: Vec<crate::plugin::VirtualModelRoute>,
    concrete_models: &[String],
) -> Vec<crate::plugin::VirtualModelRoute> {
    let has_candidates = !concrete_models.is_empty();
    routes
        .into_iter()
        .filter(|route| !route.requires_candidates || has_candidates)
        .collect()
}

#[derive(Clone)]
pub(crate) struct InferenceRpcBridge {
    api_port: u16,
    plugins: PluginManager,
    client: reqwest::Client,
}

impl InferenceRpcBridge {
    pub(crate) fn new(api_port: u16, plugins: PluginManager) -> Self {
        Self {
            api_port,
            plugins,
            client: reqwest::Client::new(),
        }
    }

    async fn infer(&self, params_json: &str) -> Result<RpcResult, proto::ErrorResponse> {
        let mut request: HostInferenceRequest = serde_json::from_str(params_json)
            .map_err(|error| invalid_params(format!("invalid inference request: {error}")))?;
        let model_id = request.model_id.trim().to_string();
        if model_id.is_empty() {
            return Err(invalid_params("model_id is required"));
        }
        if super::automatic::is_directive(&model_id)
            || self
                .plugins
                .virtual_model_for_model(&model_id)
                .await
                .map_err(internal)?
                .is_some()
        {
            return Err(invalid_params(format!(
                "nested inference cannot target virtual model '{}'",
                model_id
            )));
        }
        let object = request
            .request
            .as_object_mut()
            .ok_or_else(|| invalid_params("request must be a JSON object"))?;
        object.insert("model".into(), model_id.into());
        object.insert("stream".into(), false.into());
        let timeout = std::time::Duration::from_millis(
            request.timeout_ms.unwrap_or(60_000).clamp(1, 300_000),
        );
        let mut request_builder = self
            .client
            .post(format!(
                "http://127.0.0.1:{}/v1/chat/completions",
                self.api_port
            ))
            .timeout(timeout)
            .json(&request.request);
        if let Some(target_node_id) = request.target_node_id {
            request_builder =
                request_builder.header(super::request_parse::MESH_TARGET_HEADER, target_node_id);
        }
        let response = request_builder.send().await.map_err(internal)?;
        let status_code = response.status().as_u16();
        let served_by = response
            .headers()
            .get("x-mesh-served-by")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let body = response.json().await.map_err(internal)?;
        let result = HostInferenceResponse {
            status_code,
            body,
            served_by,
        };
        Ok(RpcResult {
            result_json: serde_json::to_string(&result).map_err(internal)?,
        })
    }
}

impl PluginRpcBridge for InferenceRpcBridge {
    fn handle_request(
        &self,
        _plugin_name: String,
        method: String,
        params_json: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
        let this = self.clone();
        Box::pin(async move {
            match method.as_str() {
                "inference/chat_completions" => this.infer(&params_json).await,
                _ => Err(proto::ErrorResponse {
                    code: -32601,
                    message: format!("Unsupported host RPC method '{method}'"),
                    data_json: String::new(),
                }),
            }
        })
    }

    fn handle_notification(
        &self,
        _plugin_name: String,
        _method: String,
        _params_json: String,
    ) -> BridgeFuture<()> {
        Box::pin(async {})
    }
}

fn invalid_params(message: impl Into<String>) -> proto::ErrorResponse {
    proto::ErrorResponse {
        code: -32602,
        message: message.into(),
        data_json: String::new(),
    }
}

fn internal(error: impl std::fmt::Display) -> proto::ErrorResponse {
    proto::ErrorResponse {
        code: -32603,
        message: error.to_string(),
        data_json: String::new(),
    }
}

pub(crate) async fn install_inference_bridge(plugins: &PluginManager, api_port: u16) {
    plugins
        .set_rpc_bridge(Some(Arc::new(InferenceRpcBridge::new(
            api_port,
            plugins.clone(),
        ))))
        .await;
}

pub(crate) enum VirtualModelDispatchResult {
    NotVirtual(ClientStream),
    Responded(proxy::RouteDispatchOutcome),
}

// `RouteDispatchOutcome` is deliberately `Copy`; its usage-plus-output-digests variant
// (three optional 32-byte digests inline) exceeds clippy's 128-byte `Err` threshold.
#[allow(clippy::result_large_err)]
pub(crate) async fn route_virtual_model_or_passthrough(
    node: &mesh::Node,
    tcp_stream: ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<ClientStream, proxy::RouteDispatchOutcome> {
    if request.is_tokenize_request() {
        return Ok(tcp_stream);
    }
    let Some(mut model_id) = request.model_name.clone() else {
        return Ok(tcp_stream);
    };
    if super::automatic::is_directive(&model_id) {
        super::automatic::warn_if_deprecated_alias(Some(&model_id));
        request.ensure_body_json();
        let Some(body) = request.body_json.as_ref() else {
            return Ok(tcp_stream);
        };
        if matches!(
            super::automatic::serving_mode(super::automatic::AutomaticRequest {
                model: Some(&model_id),
                path: &request.path,
                body,
            }),
            super::automatic::ServingMode::SingleModel(_)
        ) {
            return Ok(tcp_stream);
        }
        model_id = super::automatic::DIRECTIVE.to_string();
    }
    let Some(plugin_manager) = node.plugin_manager().await else {
        return Ok(tcp_stream);
    };
    if plugin_manager
        .virtual_model_for_model(&model_id)
        .await
        .ok()
        .flatten()
        .is_none()
    {
        return Ok(tcp_stream);
    }
    if super::ingress::mesh_routing_headers_requested(request) {
        let write = proxy::send_error_observed(
            tcp_stream,
            409,
            "x-mesh-target/x-mesh-exclude are not supported for virtual models",
            route_observer,
        )
        .await;
        return Err(if write.is_ok() {
            proxy::RouteDispatchOutcome::Responded(409)
        } else {
            proxy::RouteDispatchOutcome::Dropped("virtual_model_response_write_failed")
        });
    }
    request.ensure_body_json();
    let Some(body) = request.body_json.clone() else {
        let write = proxy::send_400_observed(
            tcp_stream,
            "virtual models require a JSON body",
            route_observer,
        )
        .await;
        return Err(if write.is_ok() {
            proxy::RouteDispatchOutcome::Responded(400)
        } else {
            proxy::RouteDispatchOutcome::Dropped("virtual_model_response_write_failed")
        });
    };
    let mut candidates = node.models_being_served().await;
    candidates.extend(node.serving_models().await);
    if let Ok(inference_models) = plugin_manager.inference_models().await {
        candidates.extend(inference_models);
    }
    candidates.extend(
        node.all_served_model_descriptors()
            .await
            .into_iter()
            .map(|descriptor| descriptor.identity.model_name),
    );
    match try_handle_virtual_model(
        &plugin_manager,
        node,
        tcp_stream,
        &request.path,
        &request.client_path,
        &model_id,
        body,
        candidates,
        proxy::request_context_budget(request),
        request.response_adapter,
        route_observer,
        request.exchange_observation_id.as_deref(),
        request.request_id.as_uuid().to_string(),
    )
    .await
    {
        VirtualModelDispatchResult::NotVirtual(stream) => Ok(stream),
        VirtualModelDispatchResult::Responded(outcome) => Err(outcome),
    }
}

fn validate_virtual_request(
    forwarded_path: &str,
    model_id: &str,
    route: &crate::plugin::VirtualModelRoute,
    request_body: &serde_json::Value,
    candidate_models: &[String],
) -> Option<(u16, String)> {
    if !super::automatic::is_chat_shaped_path(forwarded_path) {
        return Some((
            422,
            format!("virtual model '{model_id}' supports chat-shaped requests only"),
        ));
    }
    if candidate_models
        .iter()
        .any(|candidate| candidate == model_id)
    {
        return Some((
            409,
            format!("virtual model '{model_id}' collides with a concrete model"),
        ));
    }
    let requests_tools = request_body.get("tools").is_some_and(|tools| {
        !tools.is_null() && tools.as_array().is_none_or(|items| !items.is_empty())
    });
    if requests_tools && !route.supports_tools {
        return Some((
            422,
            format!("virtual model '{model_id}' does not support tools"),
        ));
    }
    let requests_stream = request_body
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if requests_stream && !route.supports_streaming {
        return Some((
            422,
            format!("virtual model '{model_id}' does not support streaming"),
        ));
    }
    None
}

#[allow(clippy::cognitive_complexity, clippy::too_many_arguments)]
pub(crate) async fn try_handle_virtual_model(
    plugins: &PluginManager,
    node: &mesh::Node,
    tcp_stream: ClientStream,
    forwarded_path: &str,
    client_path: &str,
    model_id: &str,
    request_body: serde_json::Value,
    candidate_models: Vec<String>,
    required_tokens: Option<u32>,
    response_adapter: proxy::ResponseAdapter,
    route_observer: OpenAiRouteObserver<'_>,
    observation_id: Option<&str>,
    exchange_id: String,
) -> VirtualModelDispatchResult {
    let route = match plugins.virtual_model_for_model(model_id).await {
        Ok(Some(route)) => route,
        Ok(None) => return VirtualModelDispatchResult::NotVirtual(tcp_stream),
        Err(error) => {
            let result = proxy::send_error_observed(
                tcp_stream,
                503,
                &format!("virtual model registry error: {error}"),
                route_observer,
            )
            .await;
            return VirtualModelDispatchResult::Responded(response_outcome(503, result));
        }
    };
    if let Some((status, message)) = validate_virtual_request(
        forwarded_path,
        model_id,
        &route,
        &request_body,
        &candidate_models,
    ) {
        let result = proxy::send_error_observed(tcp_stream, status, &message, route_observer).await;
        return VirtualModelDispatchResult::Responded(response_outcome(status, result));
    }
    let virtual_ids = plugins
        .virtual_models()
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|route| route.model_id)
        .collect::<std::collections::BTreeSet<_>>();
    let candidates =
        virtual_model_candidates(node, candidate_models, &virtual_ids, required_tokens).await;
    let (input_json, requests_stream) =
        match exchange_admission::prepare_invocation(request_body, candidates, response_adapter) {
            Ok(value) => value,
            Err(error) => {
                let result =
                    proxy::send_error_observed(tcp_stream, 500, &error, route_observer).await;
                return VirtualModelDispatchResult::Responded(response_outcome(500, result));
            }
        };
    let tcp_stream = match exchange_admission::admit_invocation(
        node,
        tcp_stream,
        super::response::prepared_dispatch::PreparedSubdispatch {
            observation_id,
            exchange_id,
            bytes: input_json.as_bytes(),
            client_path,
            encoding: "plugin_invocation_json",
            model: model_id,
            provider: "virtual_model",
            target: &route.plugin_name,
            attempt: 1,
        },
        route_observer,
    )
    .await
    {
        Ok(stream) => stream,
        Err(outcome) => return VirtualModelDispatchResult::Responded(*outcome),
    };
    let invocation = plugins.invoke_virtual_model(
        &route,
        &input_json,
        Some(std::time::Duration::from_secs(300)),
    );
    // A virtual model answers once, so a streaming caller would otherwise wait
    // out the whole turn with nothing on the wire. When the route declares
    // progress lines the host commits the head and drips them while the turn
    // runs; see `progress` for what that commit costs.
    let plan = progress::ProgressPlan::for_route(&route, response_adapter, requests_stream);
    let (tcp_stream, response, continuation) =
        match progress::drive(plan, tcp_stream, invocation).await {
            progress::Driven::Buffered { stream, response } => {
                if !(200..=599).contains(&response.status_code) {
                    let result = proxy::send_error_observed(
                        stream,
                        502,
                        &format!(
                            "virtual model '{}' returned invalid HTTP status {}",
                            route.model_id, response.status_code
                        ),
                        route_observer,
                    )
                    .await;
                    return VirtualModelDispatchResult::Responded(response_outcome(502, result));
                }
                (stream, response, None)
            }
            progress::Driven::Committed {
                stream,
                response,
                continuation,
            } => (stream, response, Some(continuation)),
            progress::Driven::WorkFailed { stream, error } => {
                let result = proxy::send_error_observed(
                    stream,
                    502,
                    &format!("virtual model '{}' failed: {error}", route.model_id),
                    route_observer,
                )
                .await;
                return VirtualModelDispatchResult::Responded(response_outcome(502, result));
            }
            progress::Driven::FailedAfterCommit { reason } => {
                // The head is committed, so the turn's failure was already
                // delivered in-band and only the outcome is left to record.
                return VirtualModelDispatchResult::Responded(
                    proxy::RouteDispatchOutcome::FailedWithStatus {
                        status_code: 200,
                        reason,
                    },
                );
            }
            progress::Driven::ClientGone => {
                return VirtualModelDispatchResult::Responded(
                    proxy::RouteDispatchOutcome::Dropped("virtual_model_progress_write_failed"),
                );
            }
        };
    let status = response.status_code;
    let headers: Vec<(&str, String)> = if continuation.is_some() {
        // The committed path sent its own head first, so the plugin's
        // result-derived headers can no longer reach the caller.
        Vec::new()
    } else {
        response
            .headers
            .iter()
            .take(32)
            .filter(|(name, value)| virtual_response_header_allowed(name, value))
            .map(|(name, value)| (name.as_str(), value.clone()))
            .collect()
    };
    let write = if requests_stream && (200..300).contains(&status) {
        let header_already_sent = continuation.is_some();
        match response_adapter {
            proxy::ResponseAdapter::OpenAiResponsesStream => {
                stream_adapters::send_responses_sse(
                    tcp_stream,
                    &response.body,
                    &headers,
                    continuation,
                )
                .await
            }
            proxy::ResponseAdapter::AnthropicMessagesStream => {
                stream_adapters::send_anthropic_messages_sse(
                    tcp_stream,
                    &response.body,
                    &headers,
                    header_already_sent,
                )
                .await
            }
            _ => {
                stream_adapters::send_chat_sse(
                    tcp_stream,
                    &response.body,
                    &headers,
                    header_already_sent,
                )
                .await
            }
        }
    } else {
        let body = translated_json_body(&response.body, response_adapter, status);
        proxy::send_json_with_status_and_headers_observed(
            tcp_stream,
            status,
            &body,
            &headers,
            route_observer,
        )
        .await
    };
    if write.is_err() {
        return VirtualModelDispatchResult::Responded(proxy::RouteDispatchOutcome::Dropped(
            "virtual_model_response_write_failed",
        ));
    }
    let usage = parse_usage(&response.body);
    let outcome = if !(200..300).contains(&status) {
        proxy::RouteDispatchOutcome::FailedWithStatus {
            status_code: status,
            reason: "virtual_model_failed",
        }
    } else if let Some(usage) = usage {
        proxy::RouteDispatchOutcome::RespondedWithUsage {
            status_code: status,
            usage,
            output_digests: Default::default(),
        }
    } else {
        proxy::RouteDispatchOutcome::Responded(status)
    };
    VirtualModelDispatchResult::Responded(outcome)
}

/// The concrete models a virtual model may route to, with the capability and
/// sizing facts a plugin needs to choose between them.
async fn virtual_model_candidates(
    node: &mesh::Node,
    candidate_models: Vec<String>,
    virtual_ids: &std::collections::BTreeSet<String>,
    required_tokens: Option<u32>,
) -> Vec<VirtualModelCandidate> {
    let descriptors = node.all_served_model_descriptors().await;
    let local_models = node.hosted_models().await;
    let deprioritized_peers = node
        .peers()
        .await
        .into_iter()
        .filter(|peer| {
            peer.inference_admission_state
                == Some(crate::proto::node::InferenceAdmissionState::AcceptingDeprioritized)
        })
        .map(|peer| hex::encode(peer.id.as_bytes()))
        .collect::<std::collections::BTreeSet<_>>();
    let mut model_ids = candidate_models
        .into_iter()
        .filter(|candidate| !virtual_ids.contains(candidate))
        .collect::<Vec<_>>();
    model_ids.sort();
    model_ids.dedup();

    let mut candidates = Vec::new();
    for model_id in model_ids {
        let descriptor = descriptors
            .iter()
            .find(|descriptor| descriptor.identity.model_name == model_id);
        let candidate_from = |target_node_id: Option<String>, context_length: Option<u32>| {
            let deprioritized = target_node_id
                .as_ref()
                .is_some_and(|target| deprioritized_peers.contains(target));
            VirtualModelCandidate {
                model_id: model_id.clone(),
                target_node_id,
                parameter_count_b: descriptor
                    .and_then(|descriptor| descriptor.metadata.as_ref())
                    .and_then(|metadata| metadata.parameter_count_b),
                context_length,
                deprioritized,
                supports_tools: descriptor.is_some_and(|descriptor| {
                    descriptor.capabilities.tool_use != crate::models::CapabilityLevel::None
                }),
                supports_vision: descriptor
                    .is_some_and(|descriptor| descriptor.capabilities.supports_vision_runtime()),
                supports_audio: descriptor
                    .is_some_and(|descriptor| descriptor.capabilities.supports_audio_runtime()),
            }
        };
        let has_local_instance = local_models.iter().any(|local| local == &model_id);
        if has_local_instance {
            let context_length = node.local_model_context_length(&model_id).await;
            if context_can_satisfy(required_tokens, context_length) {
                candidates.push(candidate_from(
                    Some(hex::encode(node.id().as_bytes())),
                    context_length,
                ));
            }
        }

        let discovered_hosts = node.hosts_for_model(&model_id).await;
        let has_discovered_remote_host = !discovered_hosts.is_empty();
        // MoA workers call peers without the payment protocol, so a peer that
        // charges for this model answers 402 and, because 402 is not a
        // retryable replica error, takes the whole worker down even when a free
        // replica exists. Keep the committee on free replicas until
        // committee-level payment is designed (#2059) — the candidate path's
        // half of the retired MoA self-fill exclusion (#2097).
        let mut remote_hosts = crate::network::openai::payment_routing::exclude_paid_hosts(
            node,
            &model_id,
            discovered_hosts,
        )
        .await;
        // Explicitly pinned nested calls fail closed on a blocked peer. Do not
        // offer it as a virtual-model candidate in the first place, or it can
        // become the direct route's non-retryable first choice.
        node.peer_blocks
            .retain_unblocked(&mut remote_hosts, crate::network::peer_blocks::now_ms());
        for host in remote_hosts {
            let context_length = node.peer_model_context_length(host, &model_id).await;
            if context_can_satisfy(required_tokens, context_length) {
                candidates.push(candidate_from(
                    Some(hex::encode(host.as_bytes())),
                    context_length,
                ));
            }
        }

        // Plugin-backed inference models have no mesh endpoint. Keep one
        // untargeted candidate when discovery found no host instance at all —
        // never when instances exist but every one was filtered out, or the
        // plugin's nested call would re-enter the ingress filter that just
        // excluded them.
        if !has_local_instance && !has_discovered_remote_host {
            candidates.push(candidate_from(None, None));
        }
    }
    candidates.sort_by(|left, right| {
        left.model_id
            .cmp(&right.model_id)
            .then_with(|| left.target_node_id.cmp(&right.target_node_id))
    });
    candidates
}

fn context_can_satisfy(required_tokens: Option<u32>, context_length: Option<u32>) -> bool {
    !matches!(
        (required_tokens, context_length),
        (Some(required), Some(context)) if context < required
    )
}

fn virtual_response_header_allowed(name: &str, value: &str) -> bool {
    if value.len() > 8 * 1024 || !proxy::is_valid_header_name(name) {
        return false;
    }
    !matches!(
        name.to_ascii_lowercase().as_str(),
        "connection"
            | "content-length"
            | "content-type"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

fn response_outcome(status: u16, result: std::io::Result<()>) -> proxy::RouteDispatchOutcome {
    if result.is_ok() {
        proxy::RouteDispatchOutcome::Responded(status)
    } else {
        proxy::RouteDispatchOutcome::Dropped("virtual_model_response_write_failed")
    }
}

fn parse_usage(body: &serde_json::Value) -> Option<TokenUsage> {
    super::response::parse_token_usage_from_json_body(body.to_string().as_bytes())
}

/// Adapt a buffered body to the caller's non-streaming protocol.
///
/// A failed turn that carries no top-level `error` object is still translated
/// for the Anthropic adapter: the Messages envelope has no way to express a
/// chat completion whose `finish_reason` is `error`.
fn translated_json_body(
    body: &serde_json::Value,
    response_adapter: proxy::ResponseAdapter,
    status: u16,
) -> serde_json::Value {
    match response_adapter {
        proxy::ResponseAdapter::OpenAiResponsesJson if (200..300).contains(&status) => {
            stream_adapters::chat_completion_to_responses_json(body)
        }
        proxy::ResponseAdapter::AnthropicMessagesJson => {
            stream_adapters::chat_completion_to_messages_json(body, !(200..300).contains(&status))
        }
        _ => body.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route(model_id: &str, requires_candidates: bool) -> crate::plugin::VirtualModelRoute {
        crate::plugin::VirtualModelRoute {
            plugin_name: "test-plugin".into(),
            model_id: model_id.into(),
            handler: "handle".into(),
            input_modalities: vec!["text".into()],
            output_modalities: vec!["text".into()],
            supports_tools: false,
            supports_streaming: false,
            requires_candidates,
            progress_lines: Vec::new(),
        }
    }

    #[test]
    fn candidate_dependent_routes_are_advertised_only_when_candidates_exist() {
        let routes = vec![route("standalone", false), route("dependent", true)];

        assert_eq!(
            advertisable_routes(routes.clone(), &[]),
            vec![route("standalone", false)]
        );
        assert_eq!(
            advertisable_routes(routes.clone(), &["concrete".into()]),
            routes
        );
    }

    #[test]
    fn virtual_response_headers_cannot_override_http_framing() {
        assert!(virtual_response_header_allowed(
            "x-plugin-route",
            "worker-a"
        ));
        assert!(!virtual_response_header_allowed("content-length", "0"));
        assert!(!virtual_response_header_allowed(
            "Transfer-Encoding",
            "chunked"
        ));
        assert!(!virtual_response_header_allowed("bad\r\nname", "value"));
        assert!(!virtual_response_header_allowed(
            "x-too-large",
            &"x".repeat(8 * 1024 + 1)
        ));
    }

    #[test]
    fn context_filter_rejects_known_short_windows_but_keeps_unknowns() {
        assert!(!context_can_satisfy(Some(16_384), Some(4_096)));
        assert!(context_can_satisfy(Some(16_384), Some(32_768)));
        assert!(context_can_satisfy(Some(16_384), None));
        assert!(context_can_satisfy(None, Some(4_096)));
    }

    /// The candidate path's half of #2097: a replica that charges for the
    /// model must not be offered to the committee, and must not be swapped for
    /// an untargeted candidate that re-enters the ingress that filtered it.
    #[cfg(feature = "payments")]
    #[tokio::test]
    async fn paid_replicas_are_not_offered_as_virtual_model_candidates() {
        let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Client)
            .await
            .expect("test node must start");
        let free = replica_node().await;
        let paid = replica_node().await;
        advertise_replica(&node, &free, "paid-model", None).await;
        advertise_replica(&node, &paid, "paid-model", Some(replica_price())).await;
        assert_eq!(
            node.hosts_for_model("paid-model").await.len(),
            2,
            "both replicas must be discovered before the payment filter runs"
        );

        let candidates = virtual_model_candidates(
            &node,
            vec!["paid-model".to_string()],
            &Default::default(),
            None,
        )
        .await;

        let targets = candidates
            .iter()
            .map(|candidate| candidate.target_node_id.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            targets,
            vec![Some(hex::encode(free.id().as_bytes()))],
            "only the free replica may be offered as a candidate"
        );
        free.endpoint.close().await;
        paid.endpoint.close().await;
        node.endpoint.close().await;
    }

    #[cfg(feature = "payments")]
    #[tokio::test]
    async fn blocked_replica_is_not_offered_as_a_direct_fallback_candidate() {
        let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Client)
            .await
            .expect("test node must start");
        let blocked = replica_node().await;
        let healthy = replica_node().await;
        advertise_replica(&node, &blocked, "small-model", None).await;
        advertise_replica(&node, &healthy, "small-model", None).await;
        node.peer_blocks
            .block(
                &blocked.id(),
                crate::network::peer_blocks::BlockLength::UntilUndone,
                crate::network::peer_blocks::Requester::Operator,
                None,
                crate::network::peer_blocks::now_ms(),
            )
            .expect("block peer");
        let candidates =
            virtual_model_candidates(&node, vec!["small-model".into()], &Default::default(), None)
                .await;
        assert_eq!(candidates.len(), 1);
        assert_eq!(
            candidates[0].target_node_id,
            Some(hex::encode(healthy.id().as_bytes()))
        );
        blocked.endpoint.close().await;
        healthy.endpoint.close().await;
        node.endpoint.close().await;
    }

    /// Every replica paid: the model drops out of the candidate set entirely.
    /// An untargeted candidate here would route the nested call straight back
    /// to the paid peer the filter exists to avoid.
    #[cfg(feature = "payments")]
    #[tokio::test]
    async fn a_model_served_only_by_a_paid_replica_yields_no_candidate() {
        let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Client)
            .await
            .expect("test node must start");
        let paid = replica_node().await;
        advertise_replica(&node, &paid, "paid-only-model", Some(replica_price())).await;
        assert_eq!(node.hosts_for_model("paid-only-model").await.len(), 1);

        let candidates = virtual_model_candidates(
            &node,
            vec!["paid-only-model".to_string()],
            &Default::default(),
            None,
        )
        .await;

        assert!(
            candidates.is_empty(),
            "a paid-only model must yield no candidate: {candidates:?}"
        );
        paid.endpoint.close().await;
        node.endpoint.close().await;
    }

    #[cfg(feature = "payments")]
    async fn replica_node() -> crate::mesh::Node {
        crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Client)
            .await
            .expect("replica node must start")
    }

    #[cfg(feature = "payments")]
    fn replica_price() -> mesh_llm_payments_types::pricing::Pricing {
        mesh_llm_payments_types::pricing::Pricing::exact(1, 1)
    }

    /// Advertise `replica` to `node` as an HTTP host serving `model`, priced
    /// when `price` is given.
    #[cfg(feature = "payments")]
    async fn advertise_replica(
        node: &crate::mesh::Node,
        replica: &crate::mesh::Node,
        model: &str,
        price: Option<mesh_llm_payments_types::pricing::Pricing>,
    ) {
        let mut announcement =
            replica.build_local_announcement(replica.snapshot_local_announcement_data().await);
        announcement.role = crate::mesh::NodeRole::Host { http_port: 9337 };
        announcement.serving_models = vec![model.to_string()];
        announcement.hosted_models = Some(vec![model.to_string()]);
        if let Some(price) = price {
            announcement
                .lightning_offers
                .insert(model.to_string(), price);
        }
        // `insert_test_peer` marks liveness observed, which
        // `hosts_for_model` requires; the conversion from an announcement
        // leaves the peer unadmitted, so admit it the way gossip would.
        let mut peer = crate::mesh::PeerInfo::from_announcement(
            replica.id(),
            replica.endpoint.addr(),
            &announcement,
            crate::crypto::OwnershipSummary::default(),
        );
        peer.admitted = true;
        node.insert_test_peer(peer).await;
    }
}
