//! Shared accepted-exchange orchestration for local and tunneled ingress.
use super::*;

/// Drive one buffered OpenAI-shaped request through every ingress stage in
/// order: control-plane/admission gates, auto-route resolution, mesh routing
/// header enforcement, virtual-model dispatch, pipeline, and finally `route_request`'s ordinary
/// dispatch. Each stage either hands the stream to the next one or writes a
/// terminal response and returns -- this function owns the single terminal
/// lifecycle event for the request no matter which stage ends it.
#[allow(clippy::cognitive_complexity)]
pub(super) async fn handle_buffered_api_request(
    tcp_stream: ClientStream,
    mut request: proxy::BufferedHttpRequest,
    ctx: ProxyConnectionContext<'_>,
    source_addr: Option<std::net::SocketAddr>,
    ingress_type: crate::runtime::IngressType,
) {
    // Claim the parent at host OpenAI ingress. All downstream dispatch sees
    // only a metadata observer; this scope remains the sole terminal owner.
    let caller_addr = source_addr.map(|addr| addr.to_string());
    let request_metadata =
        crate::logging::RequestSummaryMetadata::from_openai_ingress_path(&request.client_path)
            .with_source(Some(
                if ingress_type == crate::runtime::IngressType::RemoteQuicHttp {
                    "remote_quic_http"
                } else {
                    "direct_http"
                },
            ))
            .with_method(Some(&request.method))
            .with_caller_identity(
                None,
                caller_addr.as_deref(),
                caller_addr.as_ref().map(|_| CallerPathType::LocalHttp),
            );
    let mut lifecycle = crate::logging_runtime_state()
        .map(|state| state.openai_ingress_attachment(request.request_id, request_metadata))
        .unwrap_or_else(OpenAiLifecycleAttachment::unowned);
    if lifecycle.owns_parent() {
        request.mark_raw_lifecycle_owned();
        if let Some(body) = request.body_bytes.as_deref() {
            lifecycle.capture_request_body(body, request.artifact_request_media_kind());
        }
    }
    let (tcp_stream, mut exchange) =
        match begin_exchange(tcp_stream, &mut request, &ctx, &mut lifecycle).await {
            Ok(admitted) => admitted,
            Err(()) => return,
        };
    if exchange.is_some() {
        request.mark_raw_lifecycle_owned();
    }

    let tcp_stream =
        match maybe_handle_control_request(tcp_stream, &request, &ctx, lifecycle.route_observer())
            .await
        {
            Ok(outcome) => {
                finish_exchange(&mut exchange, outcome).await;
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
    let tcp_stream =
        match admit_buffered_api_request(tcp_stream, &ctx, ingress_type, &lifecycle).await {
            Ok(stream) => stream,
            Err(outcome) => {
                finish_exchange(&mut exchange, outcome).await;
                lifecycle.terminal(terminal_outcome_for_dispatch(outcome));
                return;
            }
        };

    let decision = match prepare_auto_route_decision(&mut request, &ctx.route, &descriptors).await {
        Ok(decision) => decision,
        Err(rejection) => {
            let outcome = send_auto_route_rejection(
                tcp_stream,
                rejection,
                ctx.route.node,
                &request.request_object_request_ids,
                &request.client_path,
                lifecycle.route_observer(),
            )
            .await;
            finish_exchange(&mut exchange, outcome).await;
            lifecycle.terminal(terminal_outcome_for_dispatch(outcome));
            return;
        }
    };

    let routing_model = decision.effective_model.clone();
    let tcp_stream = match enforce_mesh_routing_headers_before_dispatch(
        tcp_stream,
        &request,
        &decision,
        routing_model.as_deref(),
        lifecycle.route_observer(),
    )
    .await
    {
        Ok(stream) => stream,
        Err(outcome) => {
            finish_exchange(&mut exchange, outcome).await;
            lifecycle.terminal(terminal_outcome_for_dispatch(outcome));
            return;
        }
    };

    let tcp_stream = match try_handle_virtual_model_intercept(
        tcp_stream,
        &mut request,
        &ctx,
        &decision,
        lifecycle.route_observer(),
    )
    .await
    {
        VirtualModelInterceptResult::Handled(outcome) => {
            proxy::record_virtual_model_stream_lifecycle(
                lifecycle.route_observer(),
                routing_model.as_deref().unwrap_or("virtual-model"),
                request.response_adapter,
                outcome,
            );
            finish_exchange(&mut exchange, outcome).await;
            lifecycle.terminal(terminal_outcome_for_dispatch(outcome));
            return;
        }
        VirtualModelInterceptResult::NotVirtual(stream) => stream,
    };

    let mut tcp_stream = tcp_stream;
    if let Some(outcome) = try_pipeline_route(
        &mut tcp_stream,
        &mut request,
        &ctx.route,
        &decision,
        routing_model.as_deref(),
        lifecycle.route_observer(),
    )
    .await
    {
        proxy::release_request_objects(ctx.route.node, &request.request_object_request_ids).await;
        finish_exchange(&mut exchange, outcome).await;
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
    finish_exchange(&mut exchange, outcome).await;
    lifecycle.terminal(terminal_outcome_for_dispatch(outcome));
}

async fn begin_exchange(
    stream: ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &ProxyConnectionContext<'_>,
    lifecycle: &mut OpenAiLifecycleAttachment,
) -> Result<(ClientStream, Option<crate::plugin::ExchangeSession>), ()> {
    let endpoint = match request.client_path.split('?').next() {
        Some("/v1/chat/completions") => "chat_completions",
        Some("/v1/completions") => "completions",
        Some("/v1/responses") => "responses",
        _ => return Ok((stream, None)),
    };
    let Some(manager) = ctx.route.plugin_manager else {
        return Ok((stream, None));
    };
    if !manager.has_exchange_hooks().await {
        return Ok((stream, None));
    }
    let mut headers = std::collections::BTreeMap::new();
    for line in request.raw.split(|b| *b == b'\n').skip(1) {
        if line == b"\r" || line.is_empty() {
            break;
        }
        if let Ok(line) = std::str::from_utf8(line)
            && let Some((name, value)) = line.split_once(':')
            && mesh_llm_config::safe_exchange_header(name)
        {
            headers.insert(name.to_ascii_lowercase(), value.trim().to_owned());
        }
    }
    let mut event = crate::plugin::request_event(
        request.request_id.as_uuid().to_string(),
        endpoint,
        &request.method,
        &request.client_path,
        request.body_bytes.as_deref().unwrap_or_default(),
        headers,
        ctx.route.requested_by_node_id.is_some(),
    );
    event["node_endpoint_id"] = serde_json::json!(ctx.route.node.id().to_string());
    event["emission_boundary"] = serde_json::json!("socket_write_accept");
    event["dispatch_path"] = serde_json::json!(if ctx.route.requested_by_node_id.is_some() {
        "remote_mesh"
    } else {
        "raw_proxy"
    });
    let (mut session, result) = crate::plugin::ExchangeSession::begin(manager, event).await;
    request.exchange_observation_id = Some(session.observation_id().to_owned());
    let stream = stream
        .with_wire_bytes_observer(session.observer())
        .with_response_metadata(result.headers.clone());
    if let Some(status) = result.error_status() {
        let _ = proxy::send_error_observed(
            stream,
            status,
            "OpenAI plugin admission rejected the exchange",
            lifecycle.route_observer(),
        )
        .await;
        proxy::release_request_objects(ctx.route.node, &request.request_object_request_ids).await;
        session
            .finish(if result.denied {
                "policy_denied"
            } else {
                "internal_hook_failure"
            })
            .await;
        lifecycle.terminal(terminal_outcome_for_dispatch(if result.denied {
            proxy::RouteDispatchOutcome::PolicyDenied
        } else {
            proxy::RouteDispatchOutcome::RequiredHookFailed
        }));
        return Err(());
    }
    Ok((stream, Some(session)))
}

async fn finish_exchange(
    session: &mut Option<crate::plugin::ExchangeSession>,
    outcome: proxy::RouteDispatchOutcome,
) {
    let Some(session) = session else { return };
    if let proxy::RouteDispatchOutcome::RespondedWithUsage { usage, .. } = outcome {
        session.record_usage(serde_json::json!(usage));
    }
    session.finish(exchange_outcome(outcome)).await;
}

fn exchange_outcome(outcome: proxy::RouteDispatchOutcome) -> &'static str {
    match outcome {
        proxy::RouteDispatchOutcome::PolicyDenied => "policy_denied",
        proxy::RouteDispatchOutcome::RequiredHookFailed => "internal_hook_failure",
        proxy::RouteDispatchOutcome::Responded(200..=299)
        | proxy::RouteDispatchOutcome::RespondedWithUsage {
            status_code: 200..=299,
            ..
        }
        | proxy::RouteDispatchOutcome::RespondedWithDigests {
            status_code: 200..=299,
            ..
        } => "completed",
        proxy::RouteDispatchOutcome::Responded(400..=499)
        | proxy::RouteDispatchOutcome::RespondedWithUsage {
            status_code: 400..=499,
            ..
        }
        | proxy::RouteDispatchOutcome::RespondedWithDigests {
            status_code: 400..=499,
            ..
        } => "request_invalid",
        proxy::RouteDispatchOutcome::Responded(504)
        | proxy::RouteDispatchOutcome::RespondedWithUsage {
            status_code: 504, ..
        }
        | proxy::RouteDispatchOutcome::RespondedWithDigests {
            status_code: 504, ..
        }
        | proxy::RouteDispatchOutcome::FailedWithStatus {
            reason: "timeout", ..
        } => "timed_out",
        proxy::RouteDispatchOutcome::Dropped(_) => "client_cancelled",
        proxy::RouteDispatchOutcome::Failed("timeout") => "timed_out",
        proxy::RouteDispatchOutcome::Failed(_)
        | proxy::RouteDispatchOutcome::FailedWithStatus { .. } => "transport_error",
        _ => "backend_error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn enriched_error_responses_preserve_validation_and_timeout_outcomes() {
        for (status, expected) in [
            (400, "request_invalid"),
            (422, "request_invalid"),
            (500, "backend_error"),
            (504, "timed_out"),
        ] {
            for outcome in [
                proxy::RouteDispatchOutcome::Responded(status),
                proxy::RouteDispatchOutcome::RespondedWithDigests {
                    status_code: status,
                    output_digests: Default::default(),
                },
                proxy::RouteDispatchOutcome::RespondedWithUsage {
                    status_code: status,
                    usage: Default::default(),
                    output_digests: Default::default(),
                },
            ] {
                assert_eq!(exchange_outcome(outcome), expected);
            }
        }
        assert_eq!(
            exchange_outcome(proxy::RouteDispatchOutcome::FailedWithStatus {
                status_code: 504,
                reason: "timeout"
            }),
            "timed_out"
        );
    }
}
