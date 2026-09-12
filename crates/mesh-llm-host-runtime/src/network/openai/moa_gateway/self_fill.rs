//! Same-model committees: distinct physical clones, reserved for the turn.
use super::pool::canonical_base_name;
use super::workers::{LocalModelBackend, RemoteModelBackend, ReservedModelBackend};
use crate::inference::election::{InferenceTarget, ModelTargets};
use crate::mesh;
use crate::network::affinity::AffinityRouter;
use crate::network::openai::routing_rank::rank_aliased_targets_by_context;
use crate::network::reservations::RoutingReservation;
use mesh_mixture_of_agents as moa;
use std::sync::Arc;

/// Measured self-MoA width; fleet capacity must not increase fan-out cost.
const SELF_FILL_TARGET_WORKERS: usize = 2;

#[cfg(test)]
async fn select_clones(
    node: &mesh::Node,
    name: &str,
    required_tokens: Option<u32>,
    candidates: Vec<InferenceTarget>,
    affinity: Option<&AffinityRouter>,
) -> Vec<(InferenceTarget, Option<RoutingReservation>)> {
    select_aliased_clones(
        node,
        &canonical_base_name(name),
        required_tokens,
        candidates
            .into_iter()
            .map(|target| (target, name.to_string()))
            .collect(),
        affinity,
    )
    .await
    .into_iter()
    .map(|(target, _, reservation)| (target, reservation))
    .collect()
}

async fn select_aliased_clones(
    node: &mesh::Node,
    reservation_key: &str,
    required_tokens: Option<u32>,
    candidates: Vec<(InferenceTarget, String)>,
    affinity: Option<&AffinityRouter>,
) -> Vec<(InferenceTarget, String, Option<RoutingReservation>)> {
    use crate::proto::node::InferenceAdmissionState;

    let deprioritized: std::collections::HashSet<_> = node
        .peers()
        .await
        .into_iter()
        .filter(|peer| {
            peer.inference_admission_state == Some(InferenceAdmissionState::AcceptingDeprioritized)
        })
        .map(|peer| peer.id)
        .collect();
    // Preserve admission priority before context/throughput ranking. Local and
    // legacy peers stay healthy; hosts_for_model already excludes paused peers.
    let (mut healthy, mut spillover): (Vec<_>, Vec<_>) = candidates.into_iter().partition(
        |(target, _)| !matches!(target, InferenceTarget::Remote(id) if deprioritized.contains(id)),
    );
    let mut selected = Vec::with_capacity(SELF_FILL_TARGET_WORKERS);
    while selected.len() < SELF_FILL_TARGET_WORKERS {
        // Exhaust context-eligible healthy endpoints before considering spillover,
        // even when every healthy clone already has reservations from other turns.
        let mut ranked = rank_aliased_targets_by_context(node, required_tokens, &healthy).await;
        if ranked.ordered.is_empty() {
            ranked = rank_aliased_targets_by_context(node, required_tokens, &spillover).await;
        }
        let Some(preferred) = ranked.ordered.first() else {
            break;
        };
        let physical = ranked
            .ordered
            .iter()
            .map(|(target, _)| target.clone())
            .collect::<Vec<_>>();
        let preferred_target = &preferred.0;
        let (target, reservation) = affinity
            .and_then(|router| {
                router.reserve_route(
                    reservation_key,
                    &physical,
                    ranked.equivalent_prefix,
                    preferred_target,
                    false,
                )
            })
            .map(|(target, guard)| (target, Some(guard)))
            .unwrap_or_else(|| (preferred_target.clone(), None));
        let alias = ranked
            .ordered
            .iter()
            .find(|(candidate, _)| candidate == &target)
            .map(|(_, alias)| alias.clone())
            .expect("reserved aliased target came from ranked candidates");
        // Selection+reservation is atomic per slot; removing the endpoint
        // prevents duplicate workers even when other turns interleave slots.
        healthy.retain(|(candidate, _)| candidate != &target);
        spillover.retain(|(candidate, _)| candidate != &target);
        selected.push((target, alias, reservation));
    }
    selected
}

pub(super) async fn self_fill_from_extra_instances(
    node: &mesh::Node,
    targets: Option<&ModelTargets>,
    required_tokens: Option<u32>,
    http: &reqwest::Client,
    backends: &mut Vec<Arc<dyn moa::ModelBackend>>,
    models: &mut Vec<moa::ModelEntry>,
    affinity: Option<&AffinityRouter>,
) {
    let Some(existing) = models.first().cloned() else {
        return;
    };
    let base = canonical_base_name(&existing.name);
    let mut aliases = node.models_being_served().await;
    if let Some(targets) = targets {
        aliases.extend(targets.targets.keys().cloned());
    }
    aliases.retain(|alias| canonical_base_name(alias) == base);
    aliases.push(existing.name.clone());
    aliases.sort_by(|a, b| {
        (b == &existing.name)
            .cmp(&(a == &existing.name))
            .then_with(|| a.len().cmp(&b.len()))
            .then_with(|| a.cmp(b))
    });
    aliases.dedup();
    let mut candidates = Vec::new();
    for alias in &aliases {
        if let Some(local) = targets
            .and_then(|targets| targets.targets.get(alias))
            .and_then(|targets| {
                targets
                    .iter()
                    .find(|t| matches!(t, InferenceTarget::Local(_)))
            })
        {
            candidates.push((local.clone(), alias.clone()));
        }
        candidates.extend(
            node.hosts_for_model(alias)
                .await
                .into_iter()
                .map(|peer_id| (InferenceTarget::Remote(peer_id), alias.clone())),
        );
    }
    let mut physical = Vec::new();
    candidates.retain(|(target, _)| {
        if physical.contains(target) {
            false
        } else {
            physical.push(target.clone());
            true
        }
    });
    if candidates.len() < 2 {
        return;
    }
    let selected = select_aliased_clones(node, &base, required_tokens, candidates, affinity).await;
    if selected.len() < 2 {
        return; // Context filtering must not fabricate a second worker.
    }
    let mut filled_backends = Vec::with_capacity(selected.len());
    let mut filled_models = Vec::with_capacity(selected.len());
    for (backend_index, (target, name, reservation)) in selected.into_iter().enumerate() {
        let inner: Arc<dyn moa::ModelBackend> = match target {
            InferenceTarget::Local(port) => Arc::new(LocalModelBackend {
                port,
                http: http.clone(),
            }),
            // No failover onto a sibling slot: every worker is a distinct sample.
            InferenceTarget::Remote(peer_id) => Arc::new(RemoteModelBackend {
                node: node.clone(),
                peer_ids: vec![peer_id],
            }),
            InferenceTarget::None => unreachable!("self-fill only collects physical endpoints"),
        };
        filled_backends.push(match reservation {
            Some(reservation) => Arc::new(ReservedModelBackend {
                inner,
                _reservation: reservation,
            }) as Arc<dyn moa::ModelBackend>,
            None => inner,
        });
        filled_models.push(moa::ModelEntry {
            name,
            backend_index,
            ..existing.clone()
        });
    }
    *backends = filled_backends;
    *models = filled_models;
}

#[cfg(test)]
mod tests;
