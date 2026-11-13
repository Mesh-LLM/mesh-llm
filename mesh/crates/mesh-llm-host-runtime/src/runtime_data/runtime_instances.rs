//! Mirror active instance identities at the synchronous observation boundary.

use super::collector::{
    build_llama_runtime_snapshot, select_runtime_llama_projection,
    should_replace_model_runtime_projection,
};
use super::snapshots::RuntimeStatusSnapshot;
use super::{RuntimeDataCollector, RuntimeDataDirty, RuntimeLlamaSlotsSnapshot};

fn remove_instance_observation(status: &mut RuntimeStatusSnapshot, instance: &str) -> bool {
    let Some(removed) = status.llama_runtime_by_instance.remove(instance) else {
        return false;
    };
    if let Some(model) = removed.slots.model.as_ref()
        && status
            .llama_runtime_by_model
            .get(model)
            .is_some_and(|runtime| runtime.slots.instance_id.as_deref() == Some(instance))
    {
        status.llama_runtime_by_model.remove(model);
        if let Some(replacement) = status
            .llama_runtime_by_instance
            .values()
            .find(|runtime| runtime.slots.model.as_deref() == Some(model))
        {
            status
                .llama_runtime_by_model
                .insert(model.clone(), replacement.clone());
        }
    }
    if status.llama_runtime.slots.instance_id.as_deref() == Some(instance) {
        status.llama_runtime =
            build_llama_runtime_snapshot(&status.llama_runtime.metrics, Default::default());
    }
    status.llama_runtime = select_runtime_llama_projection(status);
    true
}

impl RuntimeDataCollector {
    pub(crate) fn replace_active_llama_slots_snapshot(
        &self,
        snapshot: RuntimeLlamaSlotsSnapshot,
    ) -> bool {
        self.update_runtime_status(RuntimeDataDirty::RUNTIME, |runtime_status| {
            if let Some(instance) = &snapshot.instance_id
                && runtime_status
                    .active_runtime_instances
                    .get(instance)
                    .is_none_or(|model| Some(model.as_str()) != snapshot.model.as_deref())
            {
                return false;
            }
            let next_runtime =
                build_llama_runtime_snapshot(&runtime_status.llama_runtime.metrics, snapshot);
            let mut changed = false;

            if let Some(instance_id) = next_runtime.slots.instance_id.clone()
                && runtime_status.llama_runtime_by_instance.get(&instance_id) != Some(&next_runtime)
            {
                runtime_status
                    .llama_runtime_by_instance
                    .insert(instance_id, next_runtime.clone());
                changed = true;
            }

            if let Some(model) = next_runtime.slots.model.clone() {
                if should_replace_model_runtime_projection(
                    runtime_status.llama_runtime_by_model.get(&model),
                    &next_runtime,
                ) {
                    runtime_status
                        .llama_runtime_by_model
                        .insert(model, next_runtime.clone());
                    changed = true;
                }

                let selected = select_runtime_llama_projection(runtime_status);
                if runtime_status.llama_runtime != selected {
                    runtime_status.llama_runtime = selected;
                    changed = true;
                }
            } else if runtime_status.llama_runtime != next_runtime {
                runtime_status.llama_runtime = next_runtime;
                changed = true;
            }

            changed
        })
    }

    pub(crate) fn register_runtime_instance(&self, model: &str, instance: &str) {
        self.update_runtime_status(RuntimeDataDirty::RUNTIME, |status| {
            let removed = remove_instance_observation(status, instance);
            let changed = status
                .active_runtime_instances
                .insert(instance.into(), model.into())
                .as_deref()
                != Some(model);
            removed || changed
        });
    }

    pub(crate) fn unregister_runtime_instance(&self, instance: &str) {
        self.update_runtime_status(RuntimeDataDirty::RUNTIME, |status| {
            let removed = status.active_runtime_instances.remove(instance).is_some();
            remove_instance_observation(status, instance) || removed
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_data::{RuntimeLlamaEndpointStatus, RuntimeLlamaSlotSnapshot};

    #[test]
    fn retired_instance_cannot_repopulate_status_or_leave_slot_items() {
        let collector = RuntimeDataCollector::new();
        collector.register_runtime_instance("model", "one");
        let observation = RuntimeLlamaSlotsSnapshot {
            model: Some("model".into()),
            instance_id: Some("one".into()),
            status: RuntimeLlamaEndpointStatus::Ready,
            slots: vec![RuntimeLlamaSlotSnapshot::default()],
            ..Default::default()
        };
        assert!(collector.replace_active_llama_slots_snapshot(observation.clone()));
        assert_eq!(collector.runtime_llama_snapshot().items.slots_total, 1);
        collector.unregister_runtime_instance("one");
        assert!(collector.runtime_llama_snapshots_by_instance().is_empty());
        assert_eq!(collector.runtime_llama_snapshot().items.slots_total, 0);
        assert!(!collector.replace_active_llama_slots_snapshot(observation));
        assert!(collector.runtime_llama_snapshots_by_instance().is_empty());
        assert_eq!(collector.runtime_llama_snapshot().items.slots_total, 0);
    }
}
