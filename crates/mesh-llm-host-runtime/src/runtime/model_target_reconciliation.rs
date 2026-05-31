use crate::api::status::ModelTargetCapacityAdviceState;
use crate::mesh::NodeRole;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ModelTargetReconciliationPolicy {
    pub(crate) enabled: bool,
    pub(crate) max_loads_per_tick: usize,
    pub(crate) failure_cooldown_secs: u64,
    pub(crate) manual_unload_cooldown_secs: u64,
    pub(crate) demand_upgrades_enabled: bool,
    pub(crate) demand_upgrade_min_request_count: u64,
    pub(crate) demand_upgrade_max_age_secs: u64,
}

impl Default for ModelTargetReconciliationPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            max_loads_per_tick: 1,
            failure_cooldown_secs: 5 * 60,
            manual_unload_cooldown_secs: 5 * 60,
            demand_upgrades_enabled: false,
            demand_upgrade_min_request_count:
                mesh_llm_config::DEFAULT_MODEL_TARGET_DEMAND_UPGRADE_MIN_REQUESTS,
            demand_upgrade_max_age_secs:
                mesh_llm_config::DEFAULT_MODEL_TARGET_DEMAND_UPGRADE_MAX_AGE_SECS,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ModelTargetReconciliationState {
    in_flight_model_refs: BTreeSet<String>,
    failed_until_secs: BTreeMap<String, u64>,
    manual_unload_until_secs: BTreeMap<String, u64>,
}

impl ModelTargetReconciliationState {
    pub(crate) fn mark_load_started(&mut self, model_ref: &str) {
        self.in_flight_model_refs.insert(model_ref.to_string());
    }

    pub(crate) fn record_load_success(&mut self, model_ref: &str) {
        self.in_flight_model_refs.remove(model_ref);
        self.failed_until_secs.remove(model_ref);
    }

    pub(crate) fn record_load_failure(
        &mut self,
        model_ref: &str,
        now_secs: u64,
        policy: &ModelTargetReconciliationPolicy,
    ) {
        self.in_flight_model_refs.remove(model_ref);
        if policy.failure_cooldown_secs > 0 {
            self.failed_until_secs.insert(
                model_ref.to_string(),
                now_secs.saturating_add(policy.failure_cooldown_secs),
            );
        }
    }

    pub(crate) fn record_manual_unload(
        &mut self,
        model_ref: &str,
        now_secs: u64,
        policy: &ModelTargetReconciliationPolicy,
    ) {
        self.in_flight_model_refs.remove(model_ref);
        if policy.manual_unload_cooldown_secs > 0 {
            self.manual_unload_until_secs.insert(
                model_ref.to_string(),
                now_secs.saturating_add(policy.manual_unload_cooldown_secs),
            );
        }
    }

    pub(crate) fn prune_expired(&mut self, now_secs: u64) {
        self.failed_until_secs.retain(|_, until| *until > now_secs);
        self.manual_unload_until_secs
            .retain(|_, until| *until > now_secs);
    }

    fn suppressed(&self, model_ref: &str, model_name: Option<&str>, now_secs: u64) -> bool {
        self.in_flight_model_refs
            .iter()
            .any(|in_flight| model_identity_matches(in_flight, model_ref))
            || self.cooldown_active(&self.failed_until_secs, model_ref, model_name, now_secs)
            || self.cooldown_active(
                &self.manual_unload_until_secs,
                model_ref,
                model_name,
                now_secs,
            )
    }

    fn cooldown_active(
        &self,
        cooldowns: &BTreeMap<String, u64>,
        model_ref: &str,
        model_name: Option<&str>,
        now_secs: u64,
    ) -> bool {
        cooldowns.iter().any(|(key, until)| {
            *until > now_secs
                && (model_identity_matches(key, model_ref)
                    || model_name.is_some_and(|name| model_identity_matches(key, name)))
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ModelTargetReconciliationInput<'a> {
    pub(crate) now_secs: u64,
    pub(crate) local_role: NodeRole,
    pub(crate) local_interest_model_refs: &'a BTreeSet<String>,
    pub(crate) loaded_model_refs: &'a BTreeSet<String>,
    pub(crate) targets: &'a [ModelTargetReconciliationCandidate],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ModelTargetReconciliationCandidate {
    pub(crate) rank: usize,
    pub(crate) model_ref: String,
    pub(crate) model_name: Option<String>,
    pub(crate) wanted: bool,
    pub(crate) wanted_reason: Option<&'static str>,
    pub(crate) request_count: u64,
    pub(crate) last_active_secs_ago: Option<u64>,
    pub(crate) serving_node_count: usize,
    pub(crate) capacity_state: ModelTargetReconciliationCapacityState,
    pub(crate) local_path: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ModelTargetReconciliationCapacityState {
    AlreadyServing,
    SingleNodeFit,
    SplitCandidate,
    InsufficientCapacity,
    UnknownModelSize,
    UnknownCapacity,
    NoEligibleHosts,
}

impl From<ModelTargetCapacityAdviceState> for ModelTargetReconciliationCapacityState {
    fn from(value: ModelTargetCapacityAdviceState) -> Self {
        match value {
            ModelTargetCapacityAdviceState::AlreadyServing => Self::AlreadyServing,
            ModelTargetCapacityAdviceState::SingleNodeFit => Self::SingleNodeFit,
            ModelTargetCapacityAdviceState::SplitCandidate => Self::SplitCandidate,
            ModelTargetCapacityAdviceState::InsufficientCapacity => Self::InsufficientCapacity,
            ModelTargetCapacityAdviceState::UnknownModelSize => Self::UnknownModelSize,
            ModelTargetCapacityAdviceState::UnknownCapacity => Self::UnknownCapacity,
            ModelTargetCapacityAdviceState::NoEligibleHosts => Self::NoEligibleHosts,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ModelTargetReconciliationAction {
    pub(crate) model_ref: String,
    pub(crate) model_name: Option<String>,
    pub(crate) load_spec: PathBuf,
    pub(crate) replace_model_ref: Option<String>,
}

pub(crate) fn plan_model_target_reconciliation(
    policy: &ModelTargetReconciliationPolicy,
    state: &mut ModelTargetReconciliationState,
    input: ModelTargetReconciliationInput<'_>,
) -> Vec<ModelTargetReconciliationAction> {
    state.prune_expired(input.now_secs);
    if !policy.enabled
        || policy.max_loads_per_tick == 0
        || matches!(input.local_role, NodeRole::Client)
    {
        return Vec::new();
    }

    let mut actions = Vec::new();
    for target in input.targets {
        if actions.len() >= policy.max_loads_per_tick {
            break;
        }
        let Some(load_spec) = target.local_path.clone() else {
            continue;
        };
        let replace_model_ref =
            replacement_target(policy, input.loaded_model_refs, input.targets, target);
        let has_local_interest = input.local_interest_model_refs.contains(&target.model_ref);
        if !target.wanted
            || target.serving_node_count > 0
            || target.capacity_state != ModelTargetReconciliationCapacityState::SingleNodeFit
            || (!has_local_interest && replace_model_ref.is_none())
            || loaded_target(input.loaded_model_refs, target)
            || state.suppressed(
                &target.model_ref,
                target.model_name.as_deref(),
                input.now_secs,
            )
        {
            continue;
        }

        actions.push(ModelTargetReconciliationAction {
            model_ref: target.model_ref.clone(),
            model_name: target.model_name.clone(),
            load_spec,
            replace_model_ref,
        });
    }
    actions
}

fn replacement_target(
    policy: &ModelTargetReconciliationPolicy,
    loaded_model_refs: &BTreeSet<String>,
    targets: &[ModelTargetReconciliationCandidate],
    target: &ModelTargetReconciliationCandidate,
) -> Option<String> {
    if !demand_upgrade_candidate(policy, loaded_model_refs, target) {
        return None;
    }
    loaded_model_refs
        .iter()
        .find(|loaded| replacement_improves_target_mix(loaded, targets, target))
        .cloned()
}

fn demand_upgrade_candidate(
    policy: &ModelTargetReconciliationPolicy,
    loaded_model_refs: &BTreeSet<String>,
    target: &ModelTargetReconciliationCandidate,
) -> bool {
    policy.demand_upgrades_enabled
        && !loaded_model_refs.is_empty()
        && target.wanted_reason == Some("active_demand")
        && target.request_count >= policy.demand_upgrade_min_request_count
        && target
            .last_active_secs_ago
            .is_some_and(|age| age <= policy.demand_upgrade_max_age_secs)
}

fn replacement_improves_target_mix(
    loaded_model_ref: &str,
    targets: &[ModelTargetReconciliationCandidate],
    target: &ModelTargetReconciliationCandidate,
) -> bool {
    let Some(loaded) = targets
        .iter()
        .find(|candidate| model_target_matches_loaded(candidate, loaded_model_ref))
    else {
        return true;
    };
    if loaded.request_count >= target.request_count {
        return false;
    }
    target.rank < loaded.rank || loaded.request_count == 0
}

fn loaded_target(
    loaded_model_refs: &BTreeSet<String>,
    target: &ModelTargetReconciliationCandidate,
) -> bool {
    loaded_model_refs.iter().any(|loaded| {
        model_identity_matches(loaded, &target.model_ref)
            || target
                .model_name
                .as_deref()
                .is_some_and(|name| model_identity_matches(loaded, name))
    })
}

fn model_target_matches_loaded(
    target: &ModelTargetReconciliationCandidate,
    loaded_model_ref: &str,
) -> bool {
    model_identity_matches(loaded_model_ref, &target.model_ref)
        || target
            .model_name
            .as_deref()
            .is_some_and(|name| model_identity_matches(loaded_model_ref, name))
}

fn model_identity_matches(left: &str, right: &str) -> bool {
    if left == right {
        return true;
    }
    let (Ok(left), Ok(right)) = (
        model_ref::ModelRef::parse(left),
        model_ref::ModelRef::parse(right),
    ) else {
        return false;
    };
    left.repo == right.repo
        && left.selector == right.selector
        && revisions_match_for_reconciliation(left.revision.as_deref(), right.revision.as_deref())
}

fn revisions_match_for_reconciliation(left: Option<&str>, right: Option<&str>) -> bool {
    left == right || matches!((left, right), (None, Some("main")) | (Some("main"), None))
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_764_000_000;

    fn enabled_policy() -> ModelTargetReconciliationPolicy {
        ModelTargetReconciliationPolicy {
            enabled: true,
            ..ModelTargetReconciliationPolicy::default()
        }
    }

    fn demand_upgrade_policy() -> ModelTargetReconciliationPolicy {
        ModelTargetReconciliationPolicy {
            demand_upgrades_enabled: true,
            demand_upgrade_min_request_count: 2,
            demand_upgrade_max_age_secs: 60 * 60,
            ..enabled_policy()
        }
    }

    fn target(model_ref: &str) -> ModelTargetReconciliationCandidate {
        ModelTargetReconciliationCandidate {
            rank: 1,
            model_ref: model_ref.to_string(),
            model_name: Some("Qwen3-8B-Q4_K_M".to_string()),
            wanted: true,
            wanted_reason: Some("explicit_interest"),
            request_count: 0,
            last_active_secs_ago: None,
            serving_node_count: 0,
            capacity_state: ModelTargetReconciliationCapacityState::SingleNodeFit,
            local_path: Some(PathBuf::from("/models/qwen.gguf")),
        }
    }

    fn input<'a>(
        local_interests: &'a BTreeSet<String>,
        loaded: &'a BTreeSet<String>,
        targets: &'a [ModelTargetReconciliationCandidate],
    ) -> ModelTargetReconciliationInput<'a> {
        ModelTargetReconciliationInput {
            now_secs: NOW,
            local_role: NodeRole::Host { http_port: 9337 },
            local_interest_model_refs: local_interests,
            loaded_model_refs: loaded,
            targets,
        }
    }

    #[test]
    fn planner_is_disabled_by_default() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let loaded = BTreeSet::new();
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &ModelTargetReconciliationPolicy::default(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn plans_single_local_load_for_wanted_single_node_fit_interest() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let loaded = BTreeSet::new();
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &enabled_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert_eq!(
            actions,
            vec![ModelTargetReconciliationAction {
                model_ref: "org/model@main:file.gguf".to_string(),
                model_name: Some("Qwen3-8B-Q4_K_M".to_string()),
                load_spec: PathBuf::from("/models/qwen.gguf"),
                replace_model_ref: None,
            }]
        );
    }

    #[test]
    fn demand_upgrade_replaces_lower_demand_loaded_model() {
        let mut wanted_large = target("org/large@main:file.gguf");
        wanted_large.rank = 1;
        wanted_large.model_name = Some("Large".to_string());
        wanted_large.wanted_reason = Some("active_demand");
        wanted_large.request_count = 8;
        wanted_large.last_active_secs_ago = Some(30);
        wanted_large.local_path = Some(PathBuf::from("/models/large.gguf"));
        let mut loaded_small = target("org/small@main:file.gguf");
        loaded_small.rank = 2;
        loaded_small.model_name = Some("Small".to_string());
        loaded_small.wanted = false;
        loaded_small.request_count = 1;
        loaded_small.serving_node_count = 1;
        loaded_small.capacity_state = ModelTargetReconciliationCapacityState::AlreadyServing;
        loaded_small.local_path = None;
        let targets = vec![wanted_large, loaded_small];
        let local_interests = BTreeSet::new();
        let loaded = BTreeSet::from(["Small".to_string()]);
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &demand_upgrade_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert_eq!(
            actions,
            vec![ModelTargetReconciliationAction {
                model_ref: "org/large@main:file.gguf".to_string(),
                model_name: Some("Large".to_string()),
                load_spec: PathBuf::from("/models/large.gguf"),
                replace_model_ref: Some("Small".to_string()),
            }]
        );
    }

    #[test]
    fn demand_upgrade_requires_explicit_policy_opt_in() {
        let mut wanted_large = target("org/large@main:file.gguf");
        wanted_large.model_name = Some("Large".to_string());
        wanted_large.wanted_reason = Some("active_demand");
        wanted_large.request_count = 8;
        wanted_large.last_active_secs_ago = Some(30);
        let loaded = BTreeSet::from(["Small".to_string()]);
        let targets = vec![wanted_large];
        let local_interests = BTreeSet::new();
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &enabled_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn stale_demand_does_not_replace_loaded_model() {
        let mut wanted_large = target("org/large@main:file.gguf");
        wanted_large.model_name = Some("Large".to_string());
        wanted_large.wanted_reason = Some("active_demand");
        wanted_large.request_count = 8;
        wanted_large.last_active_secs_ago = Some(2 * 60 * 60);
        let loaded = BTreeSet::from(["Small".to_string()]);
        let targets = vec![wanted_large];
        let local_interests = BTreeSet::new();
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &demand_upgrade_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn requested_only_target_does_not_replace_loaded_model_without_request_demand() {
        let mut requested_only = target("org/requested@main:file.gguf");
        requested_only.request_count = 0;
        let targets = vec![requested_only];
        let local_interests = BTreeSet::new();
        let loaded = BTreeSet::from(["Small".to_string()]);
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &demand_upgrade_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn demand_upgrade_preserves_loaded_model_with_equal_or_higher_demand() {
        let mut wanted_large = target("org/large@main:file.gguf");
        wanted_large.rank = 2;
        wanted_large.model_name = Some("Large".to_string());
        wanted_large.wanted_reason = Some("active_demand");
        wanted_large.request_count = 3;
        wanted_large.last_active_secs_ago = Some(30);
        let mut loaded_hot = target("org/hot@main:file.gguf");
        loaded_hot.rank = 1;
        loaded_hot.model_name = Some("Hot".to_string());
        loaded_hot.wanted = false;
        loaded_hot.request_count = 3;
        loaded_hot.serving_node_count = 1;
        loaded_hot.capacity_state = ModelTargetReconciliationCapacityState::AlreadyServing;
        loaded_hot.local_path = None;
        let targets = vec![loaded_hot, wanted_large];
        let local_interests = BTreeSet::new();
        let loaded = BTreeSet::from(["Hot".to_string()]);
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &demand_upgrade_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn skips_peer_only_or_requested_targets_without_local_interest() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::new();
        let loaded = BTreeSet::new();
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &enabled_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn skips_non_single_node_or_already_available_targets() {
        let mut split = target("org/split@main:file.gguf");
        split.capacity_state = ModelTargetReconciliationCapacityState::SplitCandidate;
        let mut served = target("org/served@main:file.gguf");
        served.serving_node_count = 1;
        let mut missing_path = target("org/missing@main:file.gguf");
        missing_path.local_path = None;
        let targets = vec![split, served, missing_path];
        let local_interests = BTreeSet::from([
            "org/split@main:file.gguf".to_string(),
            "org/served@main:file.gguf".to_string(),
            "org/missing@main:file.gguf".to_string(),
        ]);
        let loaded = BTreeSet::new();
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &enabled_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn cooldowns_and_in_flight_entries_suppress_until_expired() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let loaded = BTreeSet::new();
        let policy = enabled_policy();
        let mut state = ModelTargetReconciliationState::default();
        state.record_load_failure("org/model@main:file.gguf", NOW, &policy);

        let actions = plan_model_target_reconciliation(
            &policy,
            &mut state,
            input(&local_interests, &loaded, &targets),
        );
        assert!(actions.is_empty());

        let actions = plan_model_target_reconciliation(
            &policy,
            &mut state,
            ModelTargetReconciliationInput {
                now_secs: NOW + policy.failure_cooldown_secs + 1,
                ..input(&local_interests, &loaded, &targets)
            },
        );
        assert_eq!(actions.len(), 1);
    }

    #[test]
    fn loaded_model_name_suppresses_duplicate_action() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let loaded = BTreeSet::from(["Qwen3-8B-Q4_K_M".to_string()]);
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &enabled_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn client_role_never_reconciles_local_loads() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let loaded = BTreeSet::new();
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &enabled_policy(),
            &mut state,
            ModelTargetReconciliationInput {
                local_role: NodeRole::Client,
                ..input(&local_interests, &loaded, &targets)
            },
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn max_loads_per_tick_caps_eligible_targets() {
        let mut first = target("org/first@main:file.gguf");
        first.model_name = Some("First".to_string());
        let mut second = target("org/second@main:file.gguf");
        second.model_name = Some("Second".to_string());
        let targets = vec![first, second];
        let local_interests = BTreeSet::from([
            "org/first@main:file.gguf".to_string(),
            "org/second@main:file.gguf".to_string(),
        ]);
        let loaded = BTreeSet::new();
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &enabled_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].model_ref, "org/first@main:file.gguf");
    }

    #[test]
    fn loaded_model_ref_suppresses_duplicate_action() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let loaded = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &enabled_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn loaded_hf_selector_without_revision_suppresses_main_revision_target() {
        let mut target = target("unsloth/Qwen3-8B-GGUF@main:Q4_K_M");
        target.model_name = None;
        let targets = vec![target];
        let local_interests = BTreeSet::from(["unsloth/Qwen3-8B-GGUF@main:Q4_K_M".to_string()]);
        let loaded = BTreeSet::from(["unsloth/Qwen3-8B-GGUF:Q4_K_M".to_string()]);
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &enabled_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn loaded_hf_selector_without_revision_does_not_suppress_non_main_revision_target() {
        let mut target = target("unsloth/Qwen3-8B-GGUF@feature:Q4_K_M");
        target.model_name = None;
        let targets = vec![target];
        let local_interests = BTreeSet::from(["unsloth/Qwen3-8B-GGUF@feature:Q4_K_M".to_string()]);
        let loaded = BTreeSet::from(["unsloth/Qwen3-8B-GGUF:Q4_K_M".to_string()]);
        let mut state = ModelTargetReconciliationState::default();

        let actions = plan_model_target_reconciliation(
            &enabled_policy(),
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert_eq!(actions.len(), 1);
    }

    #[test]
    fn in_flight_load_suppresses_until_completion() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let loaded = BTreeSet::new();
        let policy = enabled_policy();
        let mut state = ModelTargetReconciliationState::default();
        state.mark_load_started("org/model@main:file.gguf");

        let actions = plan_model_target_reconciliation(
            &policy,
            &mut state,
            input(&local_interests, &loaded, &targets),
        );
        assert!(actions.is_empty());

        state.record_load_success("org/model@main:file.gguf");
        let actions = plan_model_target_reconciliation(
            &policy,
            &mut state,
            input(&local_interests, &loaded, &targets),
        );
        assert_eq!(actions.len(), 1);
    }

    #[test]
    fn manual_unload_cooldown_suppresses_main_revision_target_by_loaded_alias() {
        let mut target = target("unsloth/Qwen3-8B-GGUF@main:Q4_K_M");
        target.model_name = None;
        let targets = vec![target];
        let local_interests = BTreeSet::from(["unsloth/Qwen3-8B-GGUF@main:Q4_K_M".to_string()]);
        let loaded = BTreeSet::new();
        let policy = enabled_policy();
        let mut state = ModelTargetReconciliationState::default();
        state.record_manual_unload("unsloth/Qwen3-8B-GGUF:Q4_K_M", NOW, &policy);

        let actions = plan_model_target_reconciliation(
            &policy,
            &mut state,
            input(&local_interests, &loaded, &targets),
        );

        assert!(actions.is_empty());
    }

    #[test]
    fn manual_unload_cooldown_suppresses_by_model_ref_or_name() {
        let targets = vec![target("org/model@main:file.gguf")];
        let local_interests = BTreeSet::from(["org/model@main:file.gguf".to_string()]);
        let loaded = BTreeSet::new();
        let policy = enabled_policy();
        let mut state = ModelTargetReconciliationState::default();
        state.record_manual_unload("Qwen3-8B-Q4_K_M", NOW, &policy);

        let actions = plan_model_target_reconciliation(
            &policy,
            &mut state,
            input(&local_interests, &loaded, &targets),
        );
        assert!(actions.is_empty());

        let actions = plan_model_target_reconciliation(
            &policy,
            &mut state,
            ModelTargetReconciliationInput {
                now_secs: NOW + policy.manual_unload_cooldown_secs + 1,
                ..input(&local_interests, &loaded, &targets)
            },
        );
        assert_eq!(actions.len(), 1);
    }
}
