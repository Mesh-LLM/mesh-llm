//! Hold native residency through the complete response, including SSE forwarding.
use super::*;

/// Hold a managed request lease through dispatch while preserving membership peer isolation.
pub(super) async fn route_request(
    local_only: bool,
    stream: ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    model: Option<&str>,
    required_tokens: Option<u32>,
    observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    let lifecycle = ctx
        .node
        .native_model_lifecycle
        .get()
        .filter(|lifecycle| model.is_some_and(|name| lifecycle.manages(name)));
    let (Some(lifecycle), Some(model_name)) = (lifecycle, model) else {
        return route_prepared_request(
            local_only,
            stream,
            request,
            ctx,
            model,
            required_tokens,
            observer,
        )
        .await;
    };
    let (target, excluded) = match parse_mesh_routing_headers(request)
        .and_then(|routing| twin_bracket_id(request).map(|_| routing))
    {
        Ok(routing) => routing,
        Err(message) => {
            return response_outcome(
                400,
                proxy::send_400_observed(stream, &message, observer).await,
            );
        }
    };
    if mesh_headers_force_remote(ctx.node.id(), target, &excluded) {
        return route_prepared_request(
            local_only,
            stream,
            request,
            ctx,
            model,
            required_tokens,
            observer,
        )
        .await;
    }
    let _lease = match lifecycle
        .acquire(
            model_name,
            local_only
                || target.is_some()
                || !excluded.is_empty()
                || !ctx
                    .targets
                    .candidates(model_name)
                    .iter()
                    .any(|target| matches!(target, election::InferenceTarget::Remote(_))),
        )
        .await
    {
        Ok(lease) => lease,
        Err(error) => {
            tracing::warn!(model = model_name, %error, "native model not ready");
            return response_outcome(
                503,
                proxy::send_503_observed(stream, "native model could not become ready", observer)
                    .await,
            );
        }
    };
    let execution_model = lifecycle.execution_model(model_name);
    let mut targets = refresh_local_targets(ctx.targets, &lifecycle.targets(), &execution_model);
    if local_only {
        for candidates in targets.targets.values_mut() {
            candidates.retain(|target| !matches!(target, election::InferenceTarget::Remote(_)));
        }
    }
    let route = IngressRouteContext {
        targets: &targets,
        requested_by_node_id: ctx.requested_by_node_id.clone(),
        ..*ctx
    };
    if execution_model != model_name {
        proxy::rewrite_model_field(request, &execution_model);
    }
    route_prepared_request(
        local_only,
        stream,
        request,
        &route,
        Some(&execution_model),
        required_tokens,
        observer,
    )
    .await
}

/// Refresh local residency without importing peers from another mesh membership.
fn refresh_local_targets(
    existing: &election::ModelTargets,
    fresh: &election::ModelTargets,
    model: &str,
) -> election::ModelTargets {
    // Memberships share local residency, but their peer routing tables remain
    // separate. Refresh local targets only; never import primary-mesh peers.
    let mut targets = existing.clone();
    let candidates = targets.targets.entry(model.to_owned()).or_default();
    candidates.retain(|target| !matches!(target, election::InferenceTarget::Local(_)));
    candidates.extend(
        fresh
            .candidates(model)
            .into_iter()
            .filter(|target| matches!(target, election::InferenceTarget::Local(_))),
    );
    targets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refreshing_residency_keeps_membership_peers_separate() {
        use election::{InferenceTarget, ModelTargets};
        let member_peer = iroh::SecretKey::from_bytes(&[1; 32]).public();
        let primary_peer = iroh::SecretKey::from_bytes(&[2; 32]).public();
        let mut membership = ModelTargets::default();
        membership.targets.insert(
            "test".into(),
            vec![
                InferenceTarget::Local(1111),
                InferenceTarget::Remote(member_peer),
            ],
        );
        let mut primary = ModelTargets::default();
        primary.targets.insert(
            "test".into(),
            vec![
                InferenceTarget::Local(2222),
                InferenceTarget::Remote(primary_peer),
            ],
        );
        let refreshed = refresh_local_targets(&membership, &primary, "test");
        assert_eq!(
            refreshed.candidates("test"),
            vec![
                InferenceTarget::Remote(member_peer),
                InferenceTarget::Local(2222)
            ]
        );
        let unloaded = refresh_local_targets(&refreshed, &ModelTargets::default(), "test");
        assert_eq!(
            unloaded.candidates("test"),
            vec![InferenceTarget::Remote(member_peer)]
        );
    }
}
