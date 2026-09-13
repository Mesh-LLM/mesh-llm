//! Every MoA role sends chat completions, including tool actors and reducers.
//! Encoder-decoder support for standalone generation does not certify that
//! conversational/tool contract. Exclude it, as well as stateless workloads,
//! from every committee role until a separate role capability is defined.
//! Legacy peers without workload metadata retain their historical chat role.

use crate::inference::election::InferenceTarget;
use crate::mesh::{self, ModelWorkloadClass, ServedModelDescriptor};

#[cfg(test)]
mod tests;

/// Accept causal chat or legacy metadata, not standalone encoder-decoder support.
pub(super) fn descriptor_supports_committee(descriptor: &ServedModelDescriptor) -> bool {
    matches!(
        descriptor
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.workload_class),
        None | Some(ModelWorkloadClass::CausalGeneration)
    )
}

/// Resolve model aliases before deciding whether any descriptor admits chat roles.
pub(super) fn model_supports_committee(model: &str, descriptors: &[ServedModelDescriptor]) -> bool {
    let mut matching = descriptors
        .iter()
        .filter(|descriptor| {
            crate::mesh::model_identity::descriptor_matches_routable_name(descriptor, model)
        })
        .peekable();
    matching.peek().is_none() || matching.any(descriptor_supports_committee)
}

/// Admission is target-local: another peer's descriptor cannot authorize this
/// endpoint. Apply before context ranking, reservations, and standby selection.
pub(super) async fn eligible_targets(
    node: &mesh::Node,
    model: &str,
    candidates: &[InferenceTarget],
) -> Vec<InferenceTarget> {
    let local = node.served_model_descriptors().await;
    let state = node.state.lock().await;
    candidates
        .iter()
        .filter(|target| {
            let descriptors = match target {
                InferenceTarget::Local(_) => local.as_slice(),
                InferenceTarget::Remote(id) => {
                    let Some(peer) = state.peers.get(id) else {
                        // A committee reserves work on known members. Unlike
                        // direct legacy routing, a vanished peer only shrinks
                        // the pool; it must not receive a committee role.
                        return false;
                    };
                    peer.served_model_descriptors.as_slice()
                }
                InferenceTarget::None => return false,
            };
            model_supports_committee(model, descriptors)
        })
        .cloned()
        .collect()
}

/// Apply committee admission to each known peer without borrowing another's class.
pub(super) async fn eligible_remote_hosts(
    node: &mesh::Node,
    model: &str,
    hosts: &[iroh::EndpointId],
) -> Vec<iroh::EndpointId> {
    let targets = hosts
        .iter()
        .copied()
        .map(InferenceTarget::Remote)
        .collect::<Vec<_>>();
    eligible_targets(node, model, &targets)
        .await
        .into_iter()
        .filter_map(|target| {
            if let InferenceTarget::Remote(id) = target {
                Some(id)
            } else {
                None
            }
        })
        .collect()
}
