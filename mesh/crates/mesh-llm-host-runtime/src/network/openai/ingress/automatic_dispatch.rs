//! Bounded single-model retries for an explicit automatic directive.
use super::*;
use crate::network::openai::automatic_retry::AutomaticRetryHandoff;

/// At most three re-picks after the initially selected model.
const MAX_MODEL_REPICKS: usize = 3;

/// Preserve ordinary residency, admission, peer routing, and terminal accounting
/// on each concrete dispatch. Only an uncommitted, non-timeout exhaustion may
/// return the stream; a delivered response or any timeout is terminal.
pub(super) async fn dispatch(
    mut stream: ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    model: &mut Option<String>,
    required_tokens: Option<u32>,
    observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    let mut exhausted = Vec::new();
    for repicks in 0..=MAX_MODEL_REPICKS {
        let handoff = if repicks == 0 {
            AutomaticRetryHandoff::default()
        } else {
            AutomaticRetryHandoff::for_repick()
        };
        let route = IngressRouteContext {
            automatic_retry: Some(&handoff),
            requested_by_node_id: ctx.requested_by_node_id.clone(),
            ..*ctx
        };
        let outcome = route_request(
            false,
            stream,
            request,
            &route,
            model.as_deref(),
            required_tokens,
            observer,
        )
        .await;
        let Some(retained) = handoff.take() else {
            return outcome;
        };
        stream = retained;
        if let Some(name) = model.as_deref() {
            exhausted.push(name.to_owned());
        }
        let next = if repicks < MAX_MODEL_REPICKS {
            pick_remaining_model(request, ctx, required_tokens, &exhausted).await
        } else {
            None
        };
        let Some(next) = next else {
            let reason = format!(
                "all eligible automatic models exhausted after {} model(s)",
                exhausted.len()
            );
            return response_outcome(
                503,
                proxy::send_503_observed(stream, &reason, observer).await,
            );
        };
        tracing::info!(previous = ?model, next = %next, repick = repicks + 1, "automatic route spilling to another model");
        proxy::rewrite_model_field(request, &next);
        *model = Some(next);
    }
    unreachable!("the bounded retry loop returns on its final iteration")
}

async fn pick_remaining_model(
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    required_tokens: Option<u32>,
    exhausted: &[String],
) -> Option<String> {
    let mut names =
        collect_available_models_for_auto_route(ctx.node, ctx.targets, ctx.plugin_manager).await;
    names.retain(|name| !exhausted.contains(name) && !automatic::is_directive(name));
    let descriptors = ctx.node.all_served_model_descriptors().await;
    request.ensure_body_json();
    let media = workload_routing::request_media(&request.client_path, request.body_json.as_ref());
    let candidates =
        workload_routing::routing_candidates(ctx.node, &names, &request.client_path, &descriptors)
            .await;
    let candidates = router::filter_media_compatible_candidates(&candidates, &media)?;
    // Do not resurrect an all-cooling tier while spilling. Initial selection
    // retains its existing availability policy; re-picks require a ready host.
    let (candidates, ready) = auto_route::readiness::ready_models(
        ctx.node,
        Some(ctx.targets),
        request,
        &candidates,
        required_tokens,
        ctx.affinity,
    )
    .await;
    let candidates = candidates
        .into_iter()
        .filter(|candidate| ready.contains(&candidate.name))
        .collect::<Vec<_>>();
    let classification = request.body_json.as_ref().map_or(
        router::Classification {
            needs_tools: false,
            has_media_inputs: media.has_media,
        },
        router::classify,
    );
    router::pick_model_classified(&classification, &candidates).map(str::to_owned)
}
