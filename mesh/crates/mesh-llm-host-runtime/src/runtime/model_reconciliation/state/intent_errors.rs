use super::{DesiredModelState, ModelTargetReconciliationState, model_identity_matches};

impl ModelTargetReconciliationState {
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

    pub(crate) fn clear_intent_error(&mut self, intent_id: &str) {
        if let Some(intent) = self
            .intent_history
            .iter_mut()
            .find(|intent| intent.intent_id == intent_id)
        {
            intent.last_error = None;
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

    pub(crate) fn clear_resolved_unload_errors(
        &mut self,
        model_ref: &str,
        profile: &str,
        instance_id: &str,
        model_unloaded: bool,
    ) {
        for intent in &mut self.intent_history {
            let resolved_target =
                model_unloaded || intent.instance_target.as_deref() == Some(instance_id);
            if intent.desired_state != DesiredModelState::Present
                && model_identity_matches(&intent.canonical_model_ref, model_ref)
                && intent.profile == profile
                && resolved_target
            {
                intent.last_error = None;
            }
        }
        self.publish_history();
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Arc, IntentSource, Mutex};
    use super::*;

    const MODEL: &str = "org/model@main:model.gguf";

    #[test]
    fn native_status_only_reports_current_unresolved_intent_errors() {
        let collector = crate::runtime_data::RuntimeDataCollector::new();
        let history = Arc::new(Mutex::new(Vec::new()));
        let mut state =
            ModelTargetReconciliationState::with_shared_history(history.clone(), collector.clone());
        let startup = state.add_desired(MODEL, "", IntentSource::StartupConfig);
        state.set_intent_error(&startup, "startup failed");
        assert_eq!(
            collector
                .runtime_status_snapshot()
                .model_intent_error
                .as_deref(),
            Some("startup failed")
        );
        let owner = state.add_desired(MODEL, "", IntentSource::OwnerEnsure);
        assert_eq!(collector.runtime_status_snapshot().model_intent_error, None);
        assert_eq!(
            history.lock().unwrap()[0].last_error.as_deref(),
            Some("startup failed")
        );
        state.set_intent_error(&owner, "retry failed");
        assert_eq!(
            collector
                .runtime_status_snapshot()
                .model_intent_error
                .as_deref(),
            Some("retry failed")
        );
        state.record_load_success(MODEL, "");
        assert_eq!(collector.runtime_status_snapshot().model_intent_error, None);
        assert!(
            history
                .lock()
                .unwrap()
                .iter()
                .all(|intent| intent.last_error.is_none())
        );
        let one_shot = state.add_desired(MODEL, "", IntentSource::ApiLoad);
        state.set_intent_error(&one_shot, "one-shot failed");
        assert_eq!(
            collector
                .runtime_status_snapshot()
                .model_intent_error
                .as_deref(),
            Some("one-shot failed")
        );
        state.retire_one_shot_present(&one_shot);
        assert_eq!(collector.runtime_status_snapshot().model_intent_error, None);
        let unload = state.suppress_desired(MODEL, "", None, IntentSource::ApiUnload, false);
        state.set_intent_error(&unload, "unload failed");
        state.record_load_success(MODEL, "");
        assert_eq!(
            collector
                .runtime_status_snapshot()
                .model_intent_error
                .as_deref(),
            Some("unload failed")
        );
    }

    #[test]
    fn unload_recovery_clears_absent_and_draining_intent_errors() {
        for draining in [false, true] {
            let collector = crate::runtime_data::RuntimeDataCollector::new();
            let history = Arc::new(Mutex::new(Vec::new()));
            let mut state = ModelTargetReconciliationState::with_shared_history(
                history.clone(),
                collector.clone(),
            );
            let intent =
                state.suppress_desired(MODEL, "", None, IntentSource::OwnerDrain, draining);
            state.set_intent_error(&intent, "unload failed");
            assert_eq!(
                collector
                    .runtime_status_snapshot()
                    .model_intent_error
                    .as_deref(),
                Some("unload failed")
            );
            state.clear_intent_error(&intent);
            assert_eq!(collector.runtime_status_snapshot().model_intent_error, None);
            assert!(
                history
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|intent| intent.last_error.is_none())
            );
        }
    }

    #[test]
    fn lower_precedence_unload_success_resolves_the_older_effective_failure() {
        let collector = crate::runtime_data::RuntimeDataCollector::new();
        let history = Arc::new(Mutex::new(Vec::new()));
        let mut state =
            ModelTargetReconciliationState::with_shared_history(history.clone(), collector.clone());
        let older =
            state.suppress_desired(MODEL, "", Some("a".into()), IntentSource::ApiUnload, false);
        state.set_intent_error(&older, "first unload failed");
        state.suppress_desired(
            MODEL,
            "",
            Some("a".into()),
            IntentSource::OwnerUnload,
            false,
        );
        assert_eq!(
            collector
                .runtime_status_snapshot()
                .model_intent_error
                .as_deref(),
            Some("first unload failed")
        );
        state.clear_resolved_unload_errors(MODEL, "", "a", false);
        assert_eq!(collector.runtime_status_snapshot().model_intent_error, None);
        assert!(
            history
                .lock()
                .unwrap()
                .iter()
                .all(|intent| intent.last_error.is_none())
        );
    }

    #[test]
    fn unload_error_recovery_respects_instance_profile_and_desired_state() {
        let mut state = ModelTargetReconciliationState::default();
        let cases = [
            ("", Some("a"), DesiredModelState::Absent, true),
            ("", Some("a"), DesiredModelState::Draining, true),
            ("", Some("b"), DesiredModelState::Absent, false),
            ("other", Some("a"), DesiredModelState::Absent, false),
            ("", Some("a"), DesiredModelState::Present, false),
            ("", None, DesiredModelState::Absent, false),
        ];
        let intents: Vec<_> = cases
            .into_iter()
            .map(|(profile, target, desired, cleared)| {
                let source = if desired == DesiredModelState::Present {
                    IntentSource::ApiLoad
                } else {
                    IntentSource::ApiUnload
                };
                let id = state.record_desired_intent(
                    MODEL,
                    profile,
                    target.map(str::to_owned),
                    desired,
                    source,
                    1,
                );
                state.set_intent_error(&id, "unresolved");
                (id, cleared)
            })
            .collect();
        state.clear_resolved_unload_errors("org/model:model.gguf", "", "a", false);
        for (id, cleared) in &intents {
            let intent = state
                .intent_history
                .iter()
                .find(|intent| &intent.intent_id == id)
                .unwrap();
            assert_eq!(intent.last_error.is_none(), *cleared, "{id}");
        }
        state.clear_resolved_unload_errors(MODEL, "", "a", true);
        let model_wide = state
            .intent_history
            .iter()
            .find(|intent| intent.instance_target.is_none())
            .unwrap();
        assert!(model_wide.last_error.is_none());
        let other_instance = state
            .intent_history
            .iter()
            .find(|intent| intent.instance_target.as_deref() == Some("b"))
            .unwrap();
        assert!(other_instance.last_error.is_none());
        let other_profile = state
            .intent_history
            .iter()
            .find(|intent| intent.profile == "other")
            .unwrap();
        assert!(other_profile.last_error.is_some());
        let load = state
            .intent_history
            .iter()
            .find(|intent| intent.desired_state == DesiredModelState::Present)
            .unwrap();
        assert!(load.last_error.is_some());
    }
}
