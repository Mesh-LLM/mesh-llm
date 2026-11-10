//! Accepted exchange orchestration for idle, passive and bootstrap HTTP callers.
use super::*;
use crate::network::openai::client_stream::ResponseSafetyBuffer;

fn attach_request_logging(
    request: &mut BufferedHttpRequest,
    source_addr: Option<std::net::SocketAddr>,
) -> OpenAiLifecycleAttachment {
    let caller_addr = source_addr.map(|addr| addr.to_string());
    let metadata =
        crate::logging::RequestSummaryMetadata::from_openai_ingress_path(&request.client_path)
            .with_source(Some("mesh_forwarded"))
            .with_method(Some(&request.method))
            .with_caller_identity(
                None,
                caller_addr.as_deref(),
                caller_addr.as_ref().map(|_| CallerPathType::LocalHttp),
            );
    let mut lifecycle = crate::logging_runtime_state()
        .map(|state| state.openai_ingress_attachment(request.request_id, metadata))
        .unwrap_or_else(OpenAiLifecycleAttachment::unowned);
    if lifecycle.owns_parent() {
        request.mark_raw_lifecycle_owned();
        if let Some(body) = request.body_bytes.as_deref() {
            lifecycle.defer_request_body(body, request.artifact_request_media_kind());
        }
    }
    lifecycle
}

/// The common request-handling path used by idle proxy, passive proxy, and bootstrap proxy.
///
/// Peeks at the HTTP request, handles `/v1/models`, resolves the target host
/// by model name (or falls back to any host), and tunnels the request via QUIC.
///
/// Set `track_demand` to record requests for demand-based rebalancing.
pub async fn handle_mesh_request(
    node: mesh::Node,
    tcp_stream: ClientStream,
    track_demand: bool,
    affinity: AffinityRouter,
) {
    let mut tcp_stream = tcp_stream;
    let source_addr = tcp_stream.peer_addr().ok();
    let plugin_manager = node.plugin_manager().await;
    let mut request = match read_http_request_with_plugin_manager_with_context(
        &mut tcp_stream,
        plugin_manager.as_ref(),
    )
    .await
    {
        Ok(v) => v,
        Err(err) => {
            let _ =
                crate::network::openai::parse_failure::send_read_failure(tcp_stream, &err).await;
            return;
        }
    };
    // The parsed host ingress owns the parent. Downstream route code receives
    // only the attachment's metadata observer and cannot terminalize it.
    let mut lifecycle = attach_request_logging(&mut request, source_addr);
    let (tcp_stream, response_safety) = ResponseSafetyBuffer::wrap(tcp_stream);
    lifecycle.with_response_capture_guard(response_safety.capture_guard());
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

    let Some(tcp_stream) =
        handle_mesh_control_request(&node, tcp_stream, &request, &mut lifecycle).await
    else {
        return;
    };

    // Manifest-declared virtual models run through their owning plugin.
    // Tokenization remains a direct capability RPC.
    let tcp_stream =
        match crate::network::openai::virtual_model::route_virtual_model_or_passthrough(
            &node,
            tcp_stream,
            &mut request,
            lifecycle.route_observer(),
        )
        .await
        {
            Ok(stream) => stream,
            Err(outcome) => {
                // A virtual-model plugin handled the request and consumed the
                // stream. A streamed virtual model owes the same stream lifecycle
                // the active ingress records, so its events are not invisible to a
                // passive (mesh transport) caller.
                record_virtual_model_stream_lifecycle(
                    lifecycle.route_observer(),
                    request.model_name.as_deref().unwrap_or("virtual-model"),
                    request.response_adapter,
                    outcome,
                );
                let outcome = response_safety.finish(&node, outcome).await;
                lifecycle.terminal(outcome.terminal_outcome());
                release_request_objects(&node, &request.request_object_request_ids).await;
                return;
            }
        };

    let plan = match build_mesh_request_plan(&node, &mut request, track_demand, &affinity).await {
        Ok(plan) => plan,
        Err(failure) => {
            let outcome = terminal_outcome_for_mesh_request_failure(&failure);
            handle_mesh_request_failure(
                &node,
                tcp_stream,
                &request,
                failure,
                lifecycle.route_observer(),
            )
            .await;
            lifecycle.terminal(outcome);
            return;
        }
    };
    let outcome = {
        let route_observer = lifecycle.route_observer();
        route_observer.route_selected(plan.effective_model.as_deref());
        match route_mesh_request_attempts(
            &node,
            tcp_stream,
            &request,
            &plan,
            &affinity,
            route_observer,
        )
        .await
        {
            MeshRouteResult::Exhausted(tcp_stream) => {
                finish_exhausted_mesh_request(
                    &node,
                    tcp_stream,
                    plan.effective_model.as_deref(),
                    plan.target_hosts.len(),
                    &affinity,
                    route_observer,
                )
                .await;
                crate::logging::TerminalOutcome::Failed("mesh_route_exhausted".into())
            }
            MeshRouteResult::Finished(result) => terminal_outcome_for_mesh_route_result(result),
            MeshRouteResult::AdmissionDenied(outcome) => outcome.terminal_outcome(),
        }
    };
    // Preserve the passive transport's existing usage and terminal reasons
    // unless safety replaces the response with a denial or unavailable error.
    let passthrough = RouteDispatchOutcome::Responded(200);
    let screened = response_safety.finish(&node, passthrough).await;
    lifecycle.terminal(if screened == passthrough {
        outcome
    } else {
        screened.terminal_outcome()
    });
    release_request_objects(&node, &request.request_object_request_ids).await;
}
