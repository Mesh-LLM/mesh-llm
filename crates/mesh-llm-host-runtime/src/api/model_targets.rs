//! Ranked model-target aggregation for the management API.
//!
//! This module keeps raw mesh signals separate from the derived ranking and
//! wanted hints that API handlers expose to operators.

use super::{
    LocalModelInterest, MeshApi,
    model_target_capacity::{
        ModelTargetCapacityInput, ModelTargetSizeLookup, evaluate_model_target_capacity,
    },
    status::ModelTargetPayload,
};
use crate::mesh;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
struct ModelTargetAccumulator {
    model_ref: String,
    display_name: String,
    profile: String,
    model_name: Option<String>,
    explicit_interest_count: usize,
    request_count: u64,
    last_active_secs_ago: Option<u64>,
    serving_node_count: usize,
    requested: bool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ModelTargetKey {
    model_ref: String,
    profile: String,
}

impl ModelTargetKey {
    fn new(model_ref: impl Into<String>, profile: &str) -> Self {
        Self {
            model_ref: model_ref.into(),
            profile: profile.to_string(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ModelTargetLookup {
    pub(crate) targets: Vec<ModelTargetPayload>,
    pub(crate) by_model_name: HashMap<String, ModelTargetPayload>,
    pub(crate) by_model_ref: HashMap<String, ModelTargetPayload>,
    pub(crate) wanted_model_refs: Vec<String>,
}

#[derive(Debug, Default)]
struct CatalogTargetIndex {
    canonical_ref_by_model_name: HashMap<String, String>,
    model_name_by_ref: HashMap<String, String>,
    display_name_by_ref: HashMap<String, String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WantedReason {
    ExplicitInterest,
    ActiveDemand,
    Requested,
}

impl WantedReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitInterest => "explicit_interest",
            Self::ActiveDemand => "active_demand",
            Self::Requested => "requested",
        }
    }
}

impl MeshApi {
    pub(crate) async fn model_targets(&self) -> Vec<ModelTargetPayload> {
        self.model_target_lookup().await.targets
    }

    pub(crate) async fn wanted_model_refs(&self) -> Vec<String> {
        self.model_target_lookup().await.wanted_model_refs
    }

    pub(crate) async fn model_target_lookup(&self) -> ModelTargetLookup {
        let (node, local_interests) = {
            let inner = self.inner.lock().await;
            (
                inner.node.clone(),
                inner
                    .model_interests
                    .values()
                    .cloned()
                    .collect::<Vec<LocalModelInterest>>(),
            )
        };

        let local_role = node.role().await;
        let local_vram_bytes = node.vram_bytes();
        let peers = node.peers().await;
        let catalog = node.mesh_catalog_entries().await;
        let active_demand = node.active_demand().await;
        let requested_models = node.requested_models().await;
        let node_explicit_model_interests = node.explicit_model_interests().await;
        let my_hosted_models = node.hosted_models().await;

        build_model_target_lookup(ModelTargetSource {
            local_interests,
            node_explicit_model_interests,
            peers,
            catalog,
            active_demand,
            requested_models,
            my_hosted_models,
            local_role,
            local_vram_bytes,
            now: current_unix_secs(),
        })
    }
}

struct ModelTargetSource {
    local_interests: Vec<LocalModelInterest>,
    node_explicit_model_interests: Vec<String>,
    peers: Vec<mesh::PeerInfo>,
    catalog: Vec<mesh::MeshCatalogEntry>,
    active_demand: HashMap<String, mesh::ModelDemand>,
    requested_models: Vec<String>,
    my_hosted_models: Vec<String>,
    local_role: mesh::NodeRole,
    local_vram_bytes: u64,
    now: u64,
}

fn build_model_target_lookup(source: ModelTargetSource) -> ModelTargetLookup {
    let index = build_catalog_target_index(&source.catalog);
    let serving_count_by_ref =
        collect_serving_counts(&source.my_hosted_models, &source.peers, &index);
    let mut targets = HashMap::<ModelTargetKey, ModelTargetAccumulator>::new();

    apply_explicit_interest_signals(
        &mut targets,
        source.local_interests,
        source.node_explicit_model_interests,
        &source.peers,
        &index,
    );
    apply_active_demand_signals(&mut targets, source.active_demand, source.now, &index);
    apply_requested_model_signals(&mut targets, source.requested_models, &index);
    apply_serving_signals(&mut targets, serving_count_by_ref);

    let mut targets = targets.into_values().collect::<Vec<_>>();
    sort_model_targets(&mut targets);
    let size_lookup = ModelTargetSizeLookup::load();
    let payloads = build_target_payloads(
        targets,
        &source.local_role,
        source.local_vram_bytes,
        &source.peers,
        &size_lookup,
    );
    build_target_lookup(payloads)
}

fn build_catalog_target_index(catalog: &[mesh::MeshCatalogEntry]) -> CatalogTargetIndex {
    let mut index = CatalogTargetIndex::default();
    for entry in catalog {
        let model_ref = model_ref_for_catalog_entry(entry);
        let display_name = loaded_catalog_display_name(&entry.model_name);
        index
            .canonical_ref_by_model_name
            .insert(entry.model_name.clone(), model_ref.clone());
        index
            .model_name_by_ref
            .insert(model_ref.clone(), entry.model_name.clone());
        index
            .model_name_by_ref
            .insert(entry.model_name.clone(), entry.model_name.clone());
        index
            .display_name_by_ref
            .insert(model_ref.clone(), display_name.clone());
        index
            .display_name_by_ref
            .insert(entry.model_name.clone(), display_name);
    }
    index
}

fn collect_serving_counts(
    my_hosted_models: &[String],
    peers: &[mesh::PeerInfo],
    index: &CatalogTargetIndex,
) -> HashMap<String, usize> {
    let mut serving_count_by_ref = HashMap::new();
    for model_name in my_hosted_models {
        record_serving_model(model_name, index, &mut serving_count_by_ref);
    }
    for peer in peers {
        for model_name in peer.http_routable_models() {
            record_serving_model(&model_name, index, &mut serving_count_by_ref);
        }
    }
    serving_count_by_ref
}

fn record_serving_model(
    model_name: &str,
    index: &CatalogTargetIndex,
    serving_count_by_ref: &mut HashMap<String, usize>,
) {
    let (model_name, profile) = split_model_ref_and_profile(model_name);
    let model_ref = index
        .canonical_ref_by_model_name
        .get(model_name)
        .cloned()
        .unwrap_or_else(|| model_name.to_string());
    let tracks_canonical_alias = model_ref != model_name;
    *serving_count_by_ref
        .entry(model_identity_ref(&model_ref, profile))
        .or_insert(0usize) += 1;
    if tracks_canonical_alias {
        *serving_count_by_ref
            .entry(model_identity_ref(model_name, profile))
            .or_insert(0usize) += 1;
    }
}

fn apply_explicit_interest_signals(
    targets: &mut HashMap<ModelTargetKey, ModelTargetAccumulator>,
    local_interests: Vec<LocalModelInterest>,
    node_explicit_model_interests: Vec<String>,
    peers: &[mesh::PeerInfo],
    index: &CatalogTargetIndex,
) {
    let mut local_explicit_refs = HashSet::new();
    for interest in local_interests {
        let (model_ref, profile) = split_model_ref_and_profile(&interest.model_ref);
        local_explicit_refs.insert(ModelTargetKey::new(model_ref, profile));
        increment_explicit_interest(targets, model_ref.to_string(), profile, index);
    }
    for model_ref in node_explicit_model_interests {
        let (model_ref, profile) = split_model_ref_and_profile(&model_ref);
        if local_explicit_refs.insert(ModelTargetKey::new(model_ref, profile)) {
            increment_explicit_interest(targets, model_ref.to_string(), profile, index);
        }
    }

    for peer in peers {
        let mut peer_interests = HashSet::new();
        for model_ref in &peer.explicit_model_interests {
            let (model_ref, profile) = split_model_ref_and_profile(model_ref);
            if peer_interests.insert(ModelTargetKey::new(model_ref, profile)) {
                increment_explicit_interest(targets, model_ref.to_string(), profile, index);
            }
        }
    }
}

fn increment_explicit_interest(
    targets: &mut HashMap<ModelTargetKey, ModelTargetAccumulator>,
    model_ref: String,
    profile: &str,
    index: &CatalogTargetIndex,
) {
    let model_name = model_name_for_model_ref(&model_ref, index);
    let display_name = display_name_for_model_ref(&model_ref, index);
    ensure_model_target(targets, model_ref, model_name, display_name, profile)
        .explicit_interest_count += 1;
}

fn apply_active_demand_signals(
    targets: &mut HashMap<ModelTargetKey, ModelTargetAccumulator>,
    active_demand: HashMap<String, mesh::ModelDemand>,
    now: u64,
    index: &CatalogTargetIndex,
) {
    for (model_name, demand) in active_demand {
        let (model_ref, profile) = split_model_ref_and_profile(&model_name);
        let model_ref = preferred_target_ref_for_model_name(model_ref, profile, index, targets);
        let model_name =
            model_name_for_model_ref(&model_ref, index).or_else(|| Some(model_name.clone()));
        let display_name = display_name_for_model_ref(&model_ref, index);
        let target = ensure_model_target(targets, model_ref, model_name, display_name, profile);
        target.request_count = target.request_count.max(demand.request_count);
        target.last_active_secs_ago = Some(now.saturating_sub(demand.last_active));
    }
}

fn apply_requested_model_signals(
    targets: &mut HashMap<ModelTargetKey, ModelTargetAccumulator>,
    requested_models: Vec<String>,
    index: &CatalogTargetIndex,
) {
    for requested_model in requested_models {
        let (requested_model, profile) = split_model_ref_and_profile(&requested_model);
        let model_ref =
            preferred_target_ref_for_model_name(requested_model, profile, index, targets);
        let model_name = model_name_for_model_ref(&model_ref, index)
            .or_else(|| Some(requested_model.to_string()));
        let display_name = display_name_for_model_ref(&model_ref, index);
        ensure_model_target(targets, model_ref, model_name, display_name, profile).requested = true;
    }
}

fn apply_serving_signals(
    targets: &mut HashMap<ModelTargetKey, ModelTargetAccumulator>,
    serving_count_by_ref: HashMap<String, usize>,
) {
    for target in targets.values_mut() {
        target.serving_node_count = serving_count_by_ref
            .get(&model_identity_ref(&target.model_ref, &target.profile))
            .copied()
            .unwrap_or_default();
    }
}

fn build_target_payloads(
    targets: Vec<ModelTargetAccumulator>,
    local_role: &mesh::NodeRole,
    local_vram_bytes: u64,
    peers: &[mesh::PeerInfo],
    size_lookup: &ModelTargetSizeLookup,
) -> Vec<ModelTargetPayload> {
    targets
        .into_iter()
        .enumerate()
        .map(|(index, target)| {
            let wanted_reason = wanted_reason(&target);
            let capacity_advice = evaluate_model_target_capacity(ModelTargetCapacityInput {
                model_ref: &target.model_ref,
                model_name: target.model_name.as_deref(),
                serving_node_count: target.serving_node_count,
                local_role,
                local_vram_bytes,
                peers,
                size_lookup,
            });
            ModelTargetPayload {
                rank: index + 1,
                model_ref: target.model_ref,
                display_name: target.display_name,
                profile: target.profile,
                model_name: target.model_name,
                explicit_interest_count: target.explicit_interest_count,
                request_count: target.request_count,
                last_active_secs_ago: target.last_active_secs_ago,
                serving_node_count: target.serving_node_count,
                requested: target.requested,
                wanted: wanted_reason.is_some(),
                wanted_reason: wanted_reason.map(WantedReason::as_str),
                capacity_advice,
            }
        })
        .collect()
}

fn build_target_lookup(mut payloads: Vec<ModelTargetPayload>) -> ModelTargetLookup {
    let wanted_model_refs = payloads
        .iter()
        .filter(|target| target.wanted)
        .map(target_identity_ref)
        .collect::<Vec<_>>();
    let mut by_model_name = HashMap::new();
    let mut by_model_ref = HashMap::new();
    for payload in &payloads {
        by_model_ref.insert(target_identity_ref(payload), payload.clone());
        if let Some(model_name) = &payload.model_name {
            by_model_name.insert(
                model_identity_ref(model_name, &payload.profile),
                payload.clone(),
            );
        }
    }
    payloads.shrink_to_fit();

    ModelTargetLookup {
        targets: payloads,
        by_model_name,
        by_model_ref,
        wanted_model_refs,
    }
}

fn target_identity_ref(target: &ModelTargetPayload) -> String {
    model_identity_ref(&target.model_ref, &target.profile)
}

fn model_identity_ref(model_ref: &str, profile: &str) -> String {
    if profile.is_empty() {
        model_ref.to_string()
    } else {
        format!("{model_ref}#{profile}")
    }
}

fn sort_model_targets(targets: &mut [ModelTargetAccumulator]) {
    targets.sort_by(compare_model_targets);
}

fn compare_model_targets(
    left: &ModelTargetAccumulator,
    right: &ModelTargetAccumulator,
) -> Ordering {
    right
        .explicit_interest_count
        .cmp(&left.explicit_interest_count)
        .then_with(|| right.request_count.cmp(&left.request_count))
        .then_with(|| requested_only_priority(right).cmp(&requested_only_priority(left)))
        .then_with(|| {
            left.last_active_secs_ago
                .unwrap_or(u64::MAX)
                .cmp(&right.last_active_secs_ago.unwrap_or(u64::MAX))
        })
        .then_with(|| left.display_name.cmp(&right.display_name))
        .then_with(|| left.model_ref.cmp(&right.model_ref))
        .then_with(|| left.profile.cmp(&right.profile))
}

fn requested_only_priority(target: &ModelTargetAccumulator) -> bool {
    target.serving_node_count == 0
        && target.requested
        && target.explicit_interest_count == 0
        && target.request_count == 0
}

fn wanted_reason(target: &ModelTargetAccumulator) -> Option<WantedReason> {
    if target.serving_node_count > 0 {
        return None;
    }
    if target.explicit_interest_count > 0 {
        return Some(WantedReason::ExplicitInterest);
    }
    if target.request_count > 0 {
        return Some(WantedReason::ActiveDemand);
    }
    if target.requested {
        return Some(WantedReason::Requested);
    }
    None
}

fn model_ref_for_catalog_entry(entry: &mesh::MeshCatalogEntry) -> String {
    entry
        .descriptor
        .as_ref()
        .and_then(|descriptor| descriptor.identity.canonical_ref.clone())
        .unwrap_or_else(|| entry.model_name.clone())
}

fn loaded_catalog_display_name(model_name: &str) -> String {
    crate::models::remote_catalog::find_loaded_model_exact(model_name)
        .map(|model| model.name)
        .unwrap_or_else(|| model_name.to_string())
}

fn display_name_for_model_ref(model_ref: &str, index: &CatalogTargetIndex) -> String {
    index
        .display_name_by_ref
        .get(model_ref)
        .cloned()
        .unwrap_or_else(|| crate::models::installed_model_display_name(model_ref))
}

fn model_name_for_model_ref(model_ref: &str, index: &CatalogTargetIndex) -> Option<String> {
    index.model_name_by_ref.get(model_ref).cloned()
}

fn ensure_model_target<'a>(
    targets: &'a mut HashMap<ModelTargetKey, ModelTargetAccumulator>,
    model_ref: String,
    model_name: Option<String>,
    display_name: String,
    profile: &str,
) -> &'a mut ModelTargetAccumulator {
    let target = targets
        .entry(ModelTargetKey::new(model_ref.clone(), profile))
        .or_insert_with(|| ModelTargetAccumulator {
            model_ref,
            display_name,
            profile: profile.to_string(),
            model_name,
            explicit_interest_count: 0,
            request_count: 0,
            last_active_secs_ago: None,
            serving_node_count: 0,
            requested: false,
        });
    if target.profile.is_empty() {
        target.profile = profile.to_string();
    }
    target
}

fn split_model_ref_and_profile(model_ref: &str) -> (&str, &str) {
    if let Some(hash_pos) = model_ref.rfind('#') {
        let model_name_with_profile = model_ref;
        let model_ref = &model_name_with_profile[..hash_pos];
        let profile = &model_name_with_profile[hash_pos + 1..];
        if profile.is_empty() {
            (model_ref, "")
        } else {
            (model_ref, profile)
        }
    } else {
        (model_ref, "")
    }
}

fn preferred_target_ref_for_model_name(
    model_name: &str,
    profile: &str,
    index: &CatalogTargetIndex,
    targets: &HashMap<ModelTargetKey, ModelTargetAccumulator>,
) -> String {
    if targets.contains_key(&ModelTargetKey::new(model_name, profile)) {
        return model_name.to_string();
    }

    let canonical_ref = index
        .canonical_ref_by_model_name
        .get(model_name)
        .cloned()
        .unwrap_or_else(|| model_name.to_string());
    if targets.contains_key(&ModelTargetKey::new(&canonical_ref, profile)) {
        return canonical_ref;
    }

    canonical_ref
}

fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(model_ref: &str) -> ModelTargetAccumulator {
        ModelTargetAccumulator {
            model_ref: model_ref.to_string(),
            display_name: model_ref.to_string(),
            profile: String::new(),
            model_name: Some(model_ref.to_string()),
            explicit_interest_count: 0,
            request_count: 0,
            last_active_secs_ago: None,
            serving_node_count: 0,
            requested: false,
        }
    }

    #[test]
    fn requested_signal_does_not_double_count_existing_demand() {
        let mut demand_only = target("a-demand-only");
        demand_only.request_count = 7;

        let mut requested_with_same_demand = target("z-requested-with-demand");
        requested_with_same_demand.request_count = 7;
        requested_with_same_demand.requested = true;

        let mut targets = vec![requested_with_same_demand, demand_only];
        sort_model_targets(&mut targets);

        assert_eq!(targets[0].model_ref, "a-demand-only");
        assert_eq!(targets[1].model_ref, "z-requested-with-demand");
    }

    #[test]
    fn requested_model_profile_is_preserved() {
        let mut targets = HashMap::new();
        let index = CatalogTargetIndex::default();

        apply_requested_model_signals(&mut targets, vec!["model#low-ctx".to_string()], &index);

        assert_eq!(
            targets
                .get(&ModelTargetKey::new("model", "low-ctx"))
                .expect("missing model target")
                .profile,
            "low-ctx"
        );
    }

    #[test]
    fn requested_model_profiles_are_distinct() {
        let mut targets = HashMap::new();
        let index = CatalogTargetIndex::default();

        apply_requested_model_signals(
            &mut targets,
            vec!["model#fast".to_string(), "model#quality".to_string()],
            &index,
        );

        let mut payloads = build_target_payloads(
            targets.into_values().collect(),
            &mesh::NodeRole::Worker,
            0,
            &[],
            &ModelTargetSizeLookup::default(),
        );
        payloads.sort_by(|left, right| left.profile.cmp(&right.profile));
        let lookup = build_target_lookup(payloads.clone());

        assert_eq!(payloads.len(), 2);
        assert_eq!(payloads[0].model_ref, "model");
        assert_eq!(payloads[0].profile, "fast");
        assert_eq!(payloads[1].model_ref, "model");
        assert_eq!(payloads[1].profile, "quality");
        assert_eq!(
            lookup.wanted_model_refs,
            vec!["model#fast".to_string(), "model#quality".to_string()]
        );
        assert!(lookup.by_model_ref.contains_key("model#fast"));
        assert!(lookup.by_model_ref.contains_key("model#quality"));
    }

    #[test]
    fn explicit_interest_dedupes_per_profile() {
        let mut targets = HashMap::new();
        let index = CatalogTargetIndex::default();

        apply_explicit_interest_signals(
            &mut targets,
            vec![
                LocalModelInterest {
                    model_ref: "model#fast".to_string(),
                    submission_source: None,
                    created_at_unix: 1,
                    updated_at_unix: 1,
                },
                LocalModelInterest {
                    model_ref: "model#quality".to_string(),
                    submission_source: None,
                    created_at_unix: 1,
                    updated_at_unix: 1,
                },
            ],
            vec![
                "model#fast".to_string(),
                "model#quality".to_string(),
                "model#quality".to_string(),
            ],
            &[],
            &index,
        );

        let mut targets = targets.into_values().collect::<Vec<_>>();
        targets.sort_by(|left, right| left.profile.cmp(&right.profile));

        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0].profile, "fast");
        assert_eq!(targets[0].explicit_interest_count, 1);
        assert_eq!(targets[1].profile, "quality");
        assert_eq!(targets[1].explicit_interest_count, 1);
    }

    #[test]
    fn requested_only_signal_ranks_above_inert_targets() {
        let inert = target("a-inert");
        let mut requested = target("z-requested");
        requested.requested = true;

        let mut targets = vec![inert, requested];
        sort_model_targets(&mut targets);

        assert_eq!(targets[0].model_ref, "z-requested");
        assert_eq!(wanted_reason(&targets[0]), Some(WantedReason::Requested));
        assert_eq!(wanted_reason(&targets[1]), None);
    }

    #[test]
    fn served_targets_are_not_wanted_even_with_interest() {
        let mut interested = target("interested-served");
        interested.explicit_interest_count = 3;
        interested.serving_node_count = 1;

        assert_eq!(wanted_reason(&interested), None);
    }

    #[test]
    fn profiled_served_targets_are_not_wanted() {
        let mut targets = HashMap::new();
        let index = CatalogTargetIndex::default();

        apply_requested_model_signals(&mut targets, vec!["model#fast".to_string()], &index);
        apply_serving_signals(
            &mut targets,
            collect_serving_counts(&["model#fast".to_string()], &[], &index),
        );

        let target = targets
            .get(&ModelTargetKey::new("model", "fast"))
            .expect("profiled target should be present");
        assert_eq!(target.serving_node_count, 1);
        assert_eq!(wanted_reason(target), None);
    }
}
