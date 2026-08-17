use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::{Arc, Mutex},
};

use crate::api;

use super::intent::{DesiredModelState, DesiredRuntimeIntent, IntentPersistence, IntentSource};
use super::model_identity_matches;

type LoadCompletionTx = tokio::sync::oneshot::Sender<anyhow::Result<api::RuntimeLoadResponse>>;
type UnloadCompletionTx = tokio::sync::oneshot::Sender<anyhow::Result<api::RuntimeUnloadResponse>>;

#[derive(Default)]
struct PendingLoadCompletions {
    completions: VecDeque<LoadCompletionTx>,
}

#[derive(Default)]
struct PendingUnloadCompletions {
    completions: VecDeque<UnloadCompletionTx>,
}

#[derive(Default)]
pub(crate) struct ModelTargetReconciliationState {
    /// Models currently being loaded (not yet serving).
    pub(crate) in_flight_models: BTreeSet<(String, String)>,
    /// Models that failed to load; key includes cooldown expiry.
    pub(crate) failed_models: BTreeMap<(String, String), u64>,
    /// Models manually unloaded; protected by cooldown from auto-reload.
    pub(crate) manual_unload_models: BTreeMap<(String, String), u64>,
    pending_load_completions: BTreeMap<(String, String), PendingLoadCompletions>,
    pending_unload_completions: BTreeMap<String, PendingUnloadCompletions>,
    intent_history: VecDeque<DesiredRuntimeIntent>,
    retired_intents: BTreeSet<String>,
    next_intent_sequence: u64,
    shared_history: Option<Arc<Mutex<Vec<DesiredRuntimeIntent>>>>,
}

impl ModelTargetReconciliationState {
    pub(crate) fn with_shared_history(
        shared_history: Arc<Mutex<Vec<DesiredRuntimeIntent>>>,
    ) -> Self {
        Self {
            shared_history: Some(shared_history),
            ..Self::default()
        }
    }

    // ─── Desired state management ──────────────────────────────────────────

    pub(crate) fn add_desired(
        &mut self,
        spec: &str,
        profile: &str,
        source: IntentSource,
    ) -> String {
        self.add_desired_with_id(spec, profile, source, None)
    }

    pub(crate) fn add_desired_with_id(
        &mut self,
        spec: &str,
        profile: &str,
        source: IntentSource,
        intent_id: Option<String>,
    ) -> String {
        self.record_desired_intent_with_id(
            intent_id,
            spec,
            profile,
            None,
            DesiredModelState::Present,
            source,
            current_time_secs(),
        )
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "convenience wrapper is exercised by pure precedence tests"
        )
    )]
    pub(crate) fn suppress_desired(
        &mut self,
        spec: &str,
        profile: &str,
        instance_target: Option<String>,
        source: IntentSource,
        draining: bool,
    ) -> String {
        self.suppress_desired_with_id(spec, profile, instance_target, source, draining, None)
    }

    pub(crate) fn suppress_desired_with_id(
        &mut self,
        spec: &str,
        profile: &str,
        instance_target: Option<String>,
        source: IntentSource,
        draining: bool,
        intent_id: Option<String>,
    ) -> String {
        self.record_desired_intent_with_id(
            intent_id,
            spec,
            profile,
            instance_target,
            if draining {
                DesiredModelState::Draining
            } else {
                DesiredModelState::Absent
            },
            source,
            current_time_secs(),
        )
    }

    pub(crate) fn is_desired(&self, spec: &str, profile: &str) -> bool {
        self.effective_intent(spec, profile)
            .is_some_and(|intent| intent.desired_state == DesiredModelState::Present)
    }

    pub(crate) fn desired_profile(&self, model_ref: &str) -> Option<&str> {
        let mut profiles = self
            .intent_history
            .iter()
            .filter(|intent| {
                model_identity_matches(&intent.canonical_model_ref, model_ref)
                    && self.is_effective_intent(
                        &intent.intent_id,
                        &intent.canonical_model_ref,
                        &intent.profile,
                    )
                    && intent.desired_state == DesiredModelState::Present
            })
            .map(|intent| intent.profile.as_str());
        let profile = profiles.next()?;
        profiles
            .all(|candidate| candidate == profile)
            .then_some(profile)
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "fake-clock tests use explicit timestamps through this constructor"
        )
    )]
    pub(crate) fn record_desired_intent(
        &mut self,
        model_ref: &str,
        profile: &str,
        instance_target: Option<String>,
        desired_state: DesiredModelState,
        source: IntentSource,
        now_secs: u64,
    ) -> String {
        self.record_desired_intent_with_id(
            None,
            model_ref,
            profile,
            instance_target,
            desired_state,
            source,
            now_secs,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "intent construction keeps all persisted contract fields explicit at the single history insertion boundary"
    )]
    fn record_desired_intent_with_id(
        &mut self,
        requested_intent_id: Option<String>,
        model_ref: &str,
        profile: &str,
        instance_target: Option<String>,
        desired_state: DesiredModelState,
        source: IntentSource,
        now_secs: u64,
    ) -> String {
        self.next_intent_sequence = self.next_intent_sequence.saturating_add(1);
        let intent_id = requested_intent_id
            .unwrap_or_else(|| format!("runtime-intent-{}", self.next_intent_sequence));
        let persistence = match source {
            IntentSource::StartupConfig => IntentPersistence::Process,
            IntentSource::MeshDemand => IntentPersistence::Ephemeral,
            _ => IntentPersistence::Session,
        };
        self.intent_history.push_back(DesiredRuntimeIntent {
            intent_id: intent_id.clone(),
            canonical_model_ref: model_ref.to_string(),
            profile: profile.to_string(),
            instance_target,
            desired_state,
            source,
            persistence,
            created_at_secs: now_secs,
            updated_at_secs: now_secs,
            last_error: None,
        });
        while self.intent_history.len() > 256 {
            let removable = self
                .intent_history
                .iter()
                .position(|intent| {
                    self.retired_intents.contains(&intent.intent_id)
                        || !intent.source.is_maintained()
                })
                .unwrap_or(0);
            if let Some(removed) = self.intent_history.remove(removable) {
                self.retired_intents.remove(&removed.intent_id);
            }
        }
        self.publish_history();
        intent_id
    }

    pub(crate) fn effective_intent(
        &self,
        model_ref: &str,
        profile: &str,
    ) -> Option<&DesiredRuntimeIntent> {
        self.intent_history
            .iter()
            .filter(|intent| {
                model_identity_matches(&intent.canonical_model_ref, model_ref)
                    && intent.profile == profile
                    && !self.retired_intents.contains(&intent.intent_id)
            })
            .max_by_key(|intent| {
                (
                    intent.source.precedence(),
                    intent.updated_at_secs,
                    intent.created_at_secs,
                )
            })
    }

    pub(crate) fn is_effective_intent(
        &self,
        intent_id: &str,
        model_ref: &str,
        profile: &str,
    ) -> bool {
        self.effective_intent(model_ref, profile)
            .is_some_and(|intent| intent.intent_id == intent_id)
    }

    pub(crate) fn retarget_intent_model_ref(&mut self, intent_id: &str, model_ref: &str) {
        if let Some(intent) = self
            .intent_history
            .iter_mut()
            .find(|intent| intent.intent_id == intent_id)
        {
            if intent.canonical_model_ref == model_ref {
                return;
            }
            intent.canonical_model_ref = model_ref.to_string();
            self.publish_history();
        }
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "history accessor is used by bounded-state contract tests"
        )
    )]
    pub(crate) fn intent_history(&self) -> &VecDeque<DesiredRuntimeIntent> {
        &self.intent_history
    }

    pub(crate) fn retire_one_shot_present(&mut self, intent_id: &str) {
        if self.intent_history.iter().any(|intent| {
            intent.intent_id == intent_id
                && intent.desired_state == DesiredModelState::Present
                && !intent.source.is_maintained()
        }) {
            self.retired_intents.insert(intent_id.to_string());
        }
    }

    pub(crate) fn transition_drain_to_absent(&mut self, intent_id: &str, now_secs: u64) {
        if let Some(intent) = self
            .intent_history
            .iter_mut()
            .find(|intent| intent.intent_id == intent_id)
            && intent.desired_state == DesiredModelState::Draining
        {
            intent.desired_state = DesiredModelState::Absent;
            intent.updated_at_secs = now_secs;
            self.publish_history();
        }
    }

    pub(crate) fn set_intent_error(&mut self, intent_id: &str, error: impl Into<String>) {
        if let Some(intent) = self
            .intent_history
            .iter_mut()
            .find(|intent| intent.intent_id == intent_id)
        {
            intent.set_last_error(error);
            self.publish_history();
        }
    }

    pub(crate) fn set_effective_intent_error(
        &mut self,
        model_ref: &str,
        profile: &str,
        error: impl Into<String>,
    ) {
        let Some(intent_id) = self
            .effective_intent(model_ref, profile)
            .map(|intent| intent.intent_id.clone())
        else {
            return;
        };
        self.set_intent_error(&intent_id, error);
    }

    fn publish_history(&self) {
        let Some(shared) = &self.shared_history else {
            return;
        };
        *shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            self.intent_history.iter().cloned().collect();
    }

    // ─── Load tracking ─────────────────────────────────────────────────────

    pub(crate) fn mark_load_started(&mut self, model_ref: &str, profile: &str) {
        self.in_flight_models
            .insert((model_ref.to_string(), profile.to_string()));
    }

    pub(crate) fn record_load_success(&mut self, model_ref: &str, profile: &str) {
        self.in_flight_models
            .remove(&(model_ref.to_string(), profile.to_string()));
        self.failed_models
            .remove(&(model_ref.to_string(), profile.to_string()));
    }

    pub(crate) fn record_load_failure(
        &mut self,
        model_ref: &str,
        profile: &str,
        now_secs: u64,
        policy: &super::planner::ModelTargetReconciliationPolicy,
    ) {
        self.in_flight_models
            .remove(&(model_ref.to_string(), profile.to_string()));
        if policy.failure_cooldown_secs > 0 {
            self.failed_models.insert(
                (model_ref.to_string(), profile.to_string()),
                now_secs.saturating_add(policy.failure_cooldown_secs),
            );
        }
    }

    pub(crate) fn record_manual_unload(
        &mut self,
        model_ref: &str,
        profile: &str,
        now_secs: u64,
        policy: &super::planner::ModelTargetReconciliationPolicy,
    ) {
        self.in_flight_models
            .remove(&(model_ref.to_string(), profile.to_string()));
        if policy.manual_unload_cooldown_secs > 0 {
            self.manual_unload_models.insert(
                (model_ref.to_string(), profile.to_string()),
                now_secs.saturating_add(policy.manual_unload_cooldown_secs),
            );
        }
    }

    // ─── Cooldown & suppression ────────────────────────────────────────────

    pub(crate) fn prune_expired(&mut self, now_secs: u64) {
        self.failed_models.retain(|_, until| *until > now_secs);
        self.manual_unload_models
            .retain(|_, until| *until > now_secs);
    }

    pub(crate) fn suppressed(
        &self,
        model_ref: &str,
        profile: &str,
        model_name: Option<&str>,
        now_secs: u64,
    ) -> bool {
        let compound_key = (model_ref.to_string(), profile.to_string());
        self.in_flight_models.contains(&compound_key)
            || self.cooldown_active(
                &self.failed_models,
                model_ref,
                profile,
                model_name,
                now_secs,
            )
            || self.cooldown_active(
                &self.manual_unload_models,
                model_ref,
                profile,
                model_name,
                now_secs,
            )
    }

    pub(crate) fn stack_load_completion(
        &mut self,
        model_ref: &str,
        profile: &str,
        tx: LoadCompletionTx,
    ) {
        let key = self
            .matching_load_key(model_ref, profile)
            .unwrap_or_else(|| (model_ref.to_string(), profile.to_string()));
        self.pending_load_completions
            .entry(key)
            .or_default()
            .completions
            .push_back(tx);
    }

    pub(crate) fn stack_unload_completion(&mut self, unload_key: &str, tx: UnloadCompletionTx) {
        self.pending_unload_completions
            .entry(unload_key.to_string())
            .or_default()
            .completions
            .push_back(tx);
    }

    pub(crate) fn is_load_pending(&self, model_ref: &str, profile: &str) -> bool {
        self.matching_load_key(model_ref, profile).is_some()
    }

    pub(crate) fn is_unload_pending(&self, unload_key: &str) -> bool {
        self.pending_unload_completions.contains_key(unload_key)
    }

    pub(crate) fn notify_load_success(
        &mut self,
        model_ref: &str,
        profile: &str,
        response: api::RuntimeLoadResponse,
    ) {
        if let Some(key) = self.matching_pending_load_key(model_ref, profile)
            && let Some(mut pending) = self.pending_load_completions.remove(&key)
        {
            while let Some(tx) = pending.completions.pop_front() {
                let _ = tx.send(Ok(response.clone()));
            }
        }
    }

    pub(crate) fn notify_load_failure(
        &mut self,
        model_ref: &str,
        profile: &str,
        error: &anyhow::Error,
    ) {
        if let Some(key) = self.matching_pending_load_key(model_ref, profile)
            && let Some(mut pending) = self.pending_load_completions.remove(&key)
        {
            let error = error.to_string();
            while let Some(tx) = pending.completions.pop_front() {
                let _ = tx.send(Err(anyhow::anyhow!(error.clone())));
            }
        }
    }

    pub(crate) fn notify_unload_success(
        &mut self,
        unload_key: &str,
        response: api::RuntimeUnloadResponse,
    ) {
        if let Some(mut pending) = self.pending_unload_completions.remove(unload_key) {
            while let Some(tx) = pending.completions.pop_front() {
                let _ = tx.send(Ok(response.clone()));
            }
        }
    }

    pub(crate) fn notify_unload_failure(&mut self, unload_key: &str, error: &anyhow::Error) {
        if let Some(mut pending) = self.pending_unload_completions.remove(unload_key) {
            let error = error.to_string();
            while let Some(tx) = pending.completions.pop_front() {
                let _ = tx.send(Err(anyhow::anyhow!(error.clone())));
            }
        }
    }

    fn cooldown_active(
        &self,
        cooldowns: &BTreeMap<(String, String), u64>,
        model_ref: &str,
        profile: &str,
        model_name: Option<&str>,
        now_secs: u64,
    ) -> bool {
        cooldowns.iter().any(|(key, until)| {
            *until > now_secs
                && key.1 == profile
                && (model_identity_matches(&key.0, model_ref)
                    || model_name.is_some_and(|name| model_identity_matches(&key.0, name)))
        })
    }

    fn matching_load_key(&self, model_ref: &str, profile: &str) -> Option<(String, String)> {
        self.in_flight_models
            .iter()
            .chain(self.pending_load_completions.keys())
            .find(|(pending_ref, pending_profile)| {
                pending_profile == profile && model_identity_matches(pending_ref, model_ref)
            })
            .cloned()
    }

    fn matching_pending_load_key(
        &self,
        model_ref: &str,
        profile: &str,
    ) -> Option<(String, String)> {
        self.pending_load_completions
            .keys()
            .find(|(pending_ref, pending_profile)| {
                pending_profile == profile && model_identity_matches(pending_ref, model_ref)
            })
            .cloned()
    }
}

pub(crate) fn current_time_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod desired_intent_tests {
    use super::*;

    const MODEL: &str = "org/model@main:model.gguf";

    #[test]
    fn intent_reconciliation_converges_duplicate_present_sources() {
        let mut state = ModelTargetReconciliationState::default();
        state.record_desired_intent(
            MODEL,
            "",
            None,
            DesiredModelState::Present,
            IntentSource::StartupConfig,
            10,
        );
        let owner = state.record_desired_intent(
            MODEL,
            "",
            None,
            DesiredModelState::Present,
            IntentSource::OwnerEnsure,
            11,
        );
        state.record_desired_intent(
            MODEL,
            "",
            None,
            DesiredModelState::Present,
            IntentSource::MeshDemand,
            12,
        );

        let effective = state.effective_intent(MODEL, "").unwrap();
        assert_eq!(effective.intent_id, owner);
        assert_eq!(effective.desired_state, DesiredModelState::Present);
        assert!(effective.source.is_maintained());
    }

    #[test]
    fn desired_profile_requires_one_effective_profile() {
        let mut state = ModelTargetReconciliationState::default();
        state.record_desired_intent(
            MODEL,
            "low-ctx",
            None,
            DesiredModelState::Present,
            IntentSource::StartupConfig,
            10,
        );
        assert_eq!(state.desired_profile(MODEL), Some("low-ctx"));

        state.record_desired_intent(
            MODEL,
            "high-ctx",
            None,
            DesiredModelState::Present,
            IntentSource::StartupConfig,
            11,
        );
        assert_eq!(state.desired_profile(MODEL), None);
    }

    #[test]
    fn intent_reconciliation_suppresses_advisory_after_explicit_unload() {
        let mut state = ModelTargetReconciliationState::default();
        state.record_desired_intent(
            MODEL,
            "",
            None,
            DesiredModelState::Absent,
            IntentSource::ApiUnload,
            10,
        );
        state.record_desired_intent(
            MODEL,
            "",
            None,
            DesiredModelState::Present,
            IntentSource::MeshDemand,
            99,
        );

        assert_eq!(
            state.effective_intent(MODEL, "").unwrap().desired_state,
            DesiredModelState::Absent
        );
        assert!(!state.is_desired(MODEL, ""));
    }

    #[test]
    fn effective_state_excludes_suppressed_advisory_intent() {
        let mut state = ModelTargetReconciliationState::default();
        state.suppress_desired(MODEL, "", None, IntentSource::ApiUnload, false);
        let advisory = state.add_desired(MODEL, "", IntentSource::MeshDemand);

        assert!(!state.is_effective_intent(&advisory, MODEL, ""));
        assert!(!state.is_desired(MODEL, ""));
    }

    #[test]
    fn retiring_one_shot_restores_lower_maintained_intent() {
        let mut state = ModelTargetReconciliationState::default();
        let startup = state.add_desired(MODEL, "", IntentSource::StartupConfig);
        let one_shot = state.add_desired(MODEL, "", IntentSource::OwnerLoad);
        assert!(state.is_effective_intent(&one_shot, MODEL, ""));

        state.retire_one_shot_present(&one_shot);

        assert!(state.is_effective_intent(&startup, MODEL, ""));
        assert!(state.is_desired(MODEL, ""));
    }

    #[test]
    fn retargeting_startup_intent_preserves_desired_state_under_canonical_ref() {
        let mut state = ModelTargetReconciliationState::default();
        let original = "/models/smollm.gguf";
        let canonical = "SmolLM2-135M-Instruct-Q8_0";
        let startup = state.add_desired(original, "", IntentSource::StartupConfig);

        state.retarget_intent_model_ref(&startup, canonical);

        assert!(!state.is_desired(original, ""));
        assert!(state.is_desired(canonical, ""));
        assert!(state.is_effective_intent(&startup, canonical, ""));
    }

    #[test]
    fn effective_intent_matches_implicit_main_revision() {
        let mut state = ModelTargetReconciliationState::default();
        let intent = state.add_desired("org/model:model.gguf", "", IntentSource::StartupConfig);

        assert!(state.is_effective_intent(&intent, "org/model@main:model.gguf", ""));
    }

    #[test]
    fn load_failure_cooldown_is_profile_scoped() {
        let mut state = ModelTargetReconciliationState::default();
        let policy = super::super::planner::ModelTargetReconciliationPolicy::default();
        state.record_load_failure(MODEL, "low-ctx", 10, &policy);

        assert!(state.suppressed(MODEL, "low-ctx", None, 11));
        assert!(!state.suppressed(MODEL, "high-ctx", None, 11));
    }

    #[test]
    fn intent_precedence_and_equal_source_last_write_matrix() {
        let sources = [
            IntentSource::MeshDemand,
            IntentSource::StartupConfig,
            IntentSource::OwnerEnsure,
            IntentSource::ApiLoad,
        ];
        for (low_index, low) in sources.iter().enumerate() {
            for (high_index, high) in sources.iter().enumerate() {
                let mut state = ModelTargetReconciliationState::default();
                state.record_desired_intent(MODEL, "", None, DesiredModelState::Present, *low, 100);
                state.record_desired_intent(MODEL, "", None, DesiredModelState::Absent, *high, 101);
                let expected = if high_index >= low_index {
                    DesiredModelState::Absent
                } else {
                    DesiredModelState::Present
                };
                assert_eq!(
                    state.effective_intent(MODEL, "").unwrap().desired_state,
                    expected,
                    "low={low:?} high={high:?}"
                );
            }
        }
    }

    #[test]
    fn one_shot_retirement_drain_transition_and_bounds() {
        let mut state = ModelTargetReconciliationState::default();
        let load = state.record_desired_intent(
            MODEL,
            "",
            None,
            DesiredModelState::Present,
            IntentSource::ApiLoad,
            1,
        );
        state.retire_one_shot_present(&load);
        assert!(state.effective_intent(MODEL, "").is_none());

        let drain = state.record_desired_intent(
            MODEL,
            "",
            None,
            DesiredModelState::Draining,
            IntentSource::OwnerDrain,
            2,
        );
        state.transition_drain_to_absent(&drain, 3);
        assert_eq!(
            state.effective_intent(MODEL, "").unwrap().desired_state,
            DesiredModelState::Absent
        );

        state.set_intent_error(&drain, "é".repeat(600));
        assert!(
            state
                .intent_history()
                .back()
                .unwrap()
                .last_error
                .as_ref()
                .unwrap()
                .len()
                <= 512
        );
        let maintained_model = "org/maintained@main:model.gguf";
        let maintained = state.record_desired_intent(
            maintained_model,
            "",
            None,
            DesiredModelState::Present,
            IntentSource::StartupConfig,
            4,
        );
        for index in 0..300 {
            state.record_desired_intent(
                &format!("org/model-{index}"),
                "",
                None,
                DesiredModelState::Present,
                IntentSource::MeshDemand,
                index,
            );
        }
        assert_eq!(state.intent_history().len(), 256);
        assert_eq!(
            state
                .effective_intent(maintained_model, "")
                .unwrap()
                .intent_id,
            maintained,
            "ephemeral demand must not evict an active maintained intent"
        );
    }
}
