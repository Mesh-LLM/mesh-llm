//! Mirror active instance identities at the synchronous observation boundary.

use super::{RuntimeDataCollector, RuntimeDataDirty};

impl RuntimeDataCollector {
    pub(crate) fn register_runtime_instance(&self, model: &str, instance: &str) {
        self.update_runtime_status(RuntimeDataDirty::RUNTIME, |status| {
            let removed = status.llama_runtime_by_instance.remove(instance).is_some();
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
            status.active_runtime_instances.remove(instance).is_some()
        });
    }
}
