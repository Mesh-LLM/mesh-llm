//! Client-owned TEE policy and strict single-host request dispatch.

use super::*;
use crate::mesh::tee_attestation::TdxPolicy;
use std::path::Path;
use std::time::Duration;

pub(super) enum TeeDispatch {
    Continue(ClientStream),
    Handled(proxy::RouteDispatchOutcome),
}

pub(super) async fn route_tee_request(
    tcp_stream: ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &ProxyConnectionContext<'_>,
    ingress_type: crate::runtime::IngressType,
    route_observer: OpenAiRouteObserver<'_>,
) -> TeeDispatch {
    let header_required = match request.tee_required() {
        Ok(value) => value,
        Err(message) => return reject(tcp_stream, 400, &message, route_observer).await,
    };
    let local_only = match request.tee_local_only() {
        Ok(value) => value,
        Err(message) => return reject(tcp_stream, 400, &message, route_observer).await,
    };
    let remote_ingress = ingress_type == crate::runtime::IngressType::RemoteQuicHttp;
    if remote_ingress && header_required {
        return reject(
            tcp_stream,
            400,
            "TEE proof must be checked by the client before forwarding",
            route_observer,
        )
        .await;
    }
    let required =
        header_required || (!remote_ingress && std::env::var_os("MESH_TEE_REQUIRE_ALL").is_some());
    if required && local_only {
        return reject(
            tcp_stream,
            400,
            "TEE request cannot also claim to be an internal local-only hop",
            route_observer,
        )
        .await;
    }
    if local_only {
        return TeeDispatch::Handled(
            serve_locally_only(tcp_stream, request, &ctx.route, route_observer).await,
        );
    }
    if !required {
        return TeeDispatch::Continue(tcp_stream);
    }
    TeeDispatch::Handled(route_verified_peer(tcp_stream, request, &ctx.route, route_observer).await)
}

async fn reject(
    tcp_stream: ClientStream,
    status: u16,
    message: &str,
    route_observer: OpenAiRouteObserver<'_>,
) -> TeeDispatch {
    TeeDispatch::Handled(response_outcome(
        status,
        proxy::send_error_observed(tcp_stream, status, message, route_observer).await,
    ))
}

fn tee_model(request: &proxy::BufferedHttpRequest) -> Result<&str, &'static str> {
    if request.method != "POST"
        || request.client_path.split('?').next() != Some("/v1/chat/completions")
    {
        return Err("TEE-only routing currently supports /v1/chat/completions");
    }
    match request.model_name.as_deref() {
        Some(model) if model != "auto" && model != "mesh" => Ok(model),
        _ => Err("TEE-only routing requires one exact model, not auto or mesh fan-out"),
    }
}

async fn serve_locally_only(
    tcp_stream: ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    let model = match tee_model(request) {
        Ok(model) => model.to_owned(),
        Err(message) => return handled_error(tcp_stream, 400, message, route_observer).await,
    };
    match parse_mesh_routing_headers(request) {
        Ok((None, excluded)) if excluded.is_empty() => {}
        _ => {
            return handled_error(
                tcp_stream,
                400,
                "TEE local-only hop cannot carry mesh routing headers",
                route_observer,
            )
            .await;
        }
    }
    let local: Vec<_> = ctx
        .targets
        .candidates(&model)
        .into_iter()
        .filter(|target| matches!(target, election::InferenceTarget::Local(_)))
        .collect();
    if local.is_empty() {
        return handled_error(
            tcp_stream,
            409,
            "attested endpoint cannot serve the model locally",
            route_observer,
        )
        .await;
    }
    let mut targets = ctx.targets.clone();
    targets.targets.insert(model.clone(), local);
    let local_context = IngressRouteContext {
        node: ctx.node,
        targets: &targets,
        affinity: ctx.affinity,
        plugin_manager: None,
        // Preserve the tunnel-authenticated requester provenance across this
        // same-node re-dispatch so the terminal still records who sent it.
        requested_by_node_id: ctx.requested_by_node_id.clone(),
        #[cfg(test)]
        exchange_channel: None,
    };
    route_request(
        tcp_stream,
        request,
        &local_context,
        Some(&model),
        None,
        route_observer,
    )
    .await
}

async fn route_verified_peer(
    tcp_stream: ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    let model = match tee_model(request) {
        Ok(model) => model.to_owned(),
        Err(message) => return handled_error(tcp_stream, 400, message, route_observer).await,
    };
    let (target, excluded) = match parse_mesh_routing_headers(request) {
        Ok(headers) => headers,
        Err(message) => return handled_error(tcp_stream, 400, &message, route_observer).await,
    };
    let Some(path) = std::env::var_os("MESH_TEE_TDX_POLICY") else {
        return handled_error(
            tcp_stream,
            503,
            "client TDX policy is not configured",
            route_observer,
        )
        .await;
    };
    let policy = match TdxPolicy::load(Path::new(&path)) {
        Ok(policy) => policy,
        Err(error) => {
            tracing::warn!(%error, "invalid client TDX policy");
            return handled_error(
                tcp_stream,
                503,
                "client TDX policy is invalid",
                route_observer,
            )
            .await;
        }
    };
    let candidates: Vec<_> = ctx
        .node
        .hosts_for_model(&model)
        .await
        .into_iter()
        .filter(|peer| {
            target.is_none_or(|requested| requested == *peer) && !excluded.contains(peer)
        })
        .collect();
    let selected = tokio::time::timeout(Duration::from_secs(30), async {
        for peer in candidates {
            match ctx.node.verify_tdx_peer(peer, &policy).await {
                Ok(()) => return Some(peer),
                Err(error) => tracing::debug!(%error, %peer, "TEE candidate rejected"),
            }
        }
        None
    })
    .await
    .ok()
    .flatten();
    let Some(peer) = selected else {
        return handled_error(
            tcp_stream,
            503,
            "no live peer passed client TDX attestation for this model",
            route_observer,
        )
        .await;
    };
    if let Err(message) = request.require_tee_local_serving() {
        return handled_error(tcp_stream, 400, &message, route_observer).await;
    }
    route_missing_local_model(
        tcp_stream,
        request,
        ctx,
        &model,
        Some(peer),
        &excluded,
        None,
        route_observer,
    )
    .await
}

async fn handled_error(
    tcp_stream: ClientStream,
    status: u16,
    message: &str,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    response_outcome(
        status,
        proxy::send_error_observed(tcp_stream, status, message, route_observer).await,
    )
}
