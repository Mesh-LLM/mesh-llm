//! Mirror active instance identities at the synchronous observation boundary.

use super::{RuntimeDataCollector, RuntimeDataDirty};

impl RuntimeDataCollector {
    pub(crate) fn register_runtime_instance(&self, model: &str, instance: &str) {
        self.update_runtime_status(RuntimeDataDirty::RUNTIME, |status| {
            // Anonymous observations cannot attest the newly registered instance.
            // Clear the model projection too, so a later update cannot restore it.
            let mut invalidated = false;
            for runtime in std::iter::once(&mut status.llama_runtime)
                .chain(status.llama_runtime_by_model.values_mut())
            {
                if runtime.slots.instance_id.is_none()
                    && runtime.slots.model.as_deref() == Some(model)
                {
                    runtime.slots = Default::default();
                    invalidated = true;
                }
            }
            let removed = status.llama_runtime_by_instance.remove(instance).is_some();
            let changed = status
                .active_runtime_instances
                .insert(instance.into(), model.into())
                .as_deref()
                != Some(model);
            invalidated || removed || changed
        });
    }

    pub(crate) fn unregister_runtime_instance(&self, instance: &str) {
        self.update_runtime_status(RuntimeDataDirty::RUNTIME, |status| {
            status.active_runtime_instances.remove(instance).is_some()
        });
    }
}
