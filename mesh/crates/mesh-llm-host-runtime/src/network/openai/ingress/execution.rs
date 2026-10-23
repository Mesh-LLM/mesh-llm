//! Peer inference ingress is execution, not another gateway routing decision.
use super::*;

pub(super) async fn execute_peer_request(
    stream: ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    let mut targets = ctx.targets.clone();
    for candidates in targets.targets.values_mut() {
        candidates.retain(|candidate| !matches!(candidate, election::InferenceTarget::Remote(_)));
    }
    let descriptors = ctx.node.served_model_descriptors().await;
    let executable_names = callable_models(&targets);
    proxy::rewrite_public_model_alias(request, &executable_names, &descriptors);
    let Some(model) = request
        .model_name
        .clone()
        .filter(|name| !automatic::is_directive(name))
    else {
        return response_outcome(
            400,
            proxy::send_400_observed(
                stream,
                "peer inference requires a concrete local model",
                observer,
            )
            .await,
        );
    };
    let local_ctx = IngressRouteContext {
        targets: &targets,
        node: ctx.node,
        affinity: ctx.affinity,
        plugin_manager: ctx.plugin_manager,
        requested_by_node_id: ctx.requested_by_node_id.clone(),
        #[cfg(test)]
        exchange_channel: ctx.exchange_channel,
    };
    let required_tokens = proxy::request_context_budget(request);
    ctx.node.record_request(&model);
    route_request(
        true,
        stream,
        request,
        &local_ctx,
        Some(&model),
        required_tokens,
        observer,
    )
    .await
}

pub(super) async fn local_model_unavailable(
    stream: ClientStream,
    request: &proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    model: &str,
    observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    if !has_local_unavailable_candidates(ctx.targets, model)
        && let Some(manager) = ctx.plugin_manager
        && !matches!(
            manager
                .inference_endpoint_for_request(model, &request.client_path)
                .await,
            Ok(None)
        )
    {
        return try_route_plugin_model(ctx, stream, request, model, observer).await;
    }
    response_outcome(
        503,
        proxy::send_503_observed(
            stream,
            &format!("model '{model}' is unavailable on this execution host"),
            observer,
        )
        .await,
    )
}
