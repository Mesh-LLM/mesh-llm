use super::IngressRouteContext;
use crate::network::metrics::{RequestOutcome, RequestService};

/// Count requests rejected before routing, attributing only known model names.
pub(super) async fn record_pre_routing_rejection(
    ctx: &IngressRouteContext<'_>,
    model: Option<&str>,
    status: u16,
) {
    let outcome = match status {
        409 => RequestOutcome::Rejected(RequestService::Local),
        _ => RequestOutcome::Unavailable,
    };
    let known = if let Some(name) = model {
        ctx.targets.targets.contains_key(name)
            || ctx
                .node
                .all_served_model_descriptors()
                .await
                .iter()
                .any(|descriptor| {
                    crate::mesh::model_identity::descriptor_matches_routable_name(descriptor, name)
                })
    } else {
        false
    };
    ctx.node
        .record_routed_request(model.filter(|_| known), 0, outcome);
}
