use crate::proto;

use super::super::PluginConfigSettingBuilder;

impl PluginConfigSettingBuilder {
    pub fn control_behavior(
        mut self,
        control_behavior: proto::PluginConfigControlBehavior,
    ) -> Self {
        self.inner.control_behavior = Some(control_behavior);
        self
    }

    pub fn control_numeric(mut self, numeric: proto::PluginConfigNumericControl) -> Self {
        self.control_behavior_mut().numeric = Some(numeric);
        self
    }

    pub fn control_numeric_min(mut self, min: f64) -> Self {
        self.control_numeric_mut().min = Some(min);
        self
    }

    pub fn control_numeric_max(mut self, max: f64) -> Self {
        self.control_numeric_mut().max = Some(max);
        self
    }

    pub fn control_numeric_step(mut self, step: f64) -> Self {
        self.control_numeric_mut().step = Some(step);
        self
    }

    pub fn control_numeric_soft_min(mut self, soft_min: f64) -> Self {
        self.control_numeric_mut().soft_min = Some(soft_min);
        self
    }

    pub fn control_numeric_soft_max(mut self, soft_max: f64) -> Self {
        self.control_numeric_mut().soft_max = Some(soft_max);
        self
    }

    pub fn control_numeric_unit(mut self, unit: impl Into<String>) -> Self {
        self.control_numeric_mut().unit = Some(unit.into());
        self
    }

    pub fn control_text_format(mut self, text_format: proto::PluginConfigTextFormat) -> Self {
        self.control_behavior_mut().text_format = Some(text_format as i32);
        self
    }

    pub fn control_options_source(
        mut self,
        options_source: proto::PluginConfigOptionsSource,
    ) -> Self {
        self.control_behavior_mut().options_source = Some(options_source as i32);
        self
    }

    pub fn control_options_static(self) -> Self {
        self.control_options_source(proto::PluginConfigOptionsSource::Static)
    }

    pub fn control_options_runtime_gpus(self) -> Self {
        self.control_options_source(proto::PluginConfigOptionsSource::RuntimeGpus)
    }

    pub fn control_options_runtime_native_backends(self) -> Self {
        self.control_options_source(proto::PluginConfigOptionsSource::RuntimeNativeBackends)
    }

    pub fn control_options_runtime_local_models(self) -> Self {
        self.control_options_source(proto::PluginConfigOptionsSource::RuntimeLocalModels)
    }

    pub fn control_options_runtime_installed_plugins(self) -> Self {
        self.control_options_source(proto::PluginConfigOptionsSource::RuntimeInstalledPlugins)
    }

    pub fn control_options_runtime_mesh_peers(self) -> Self {
        self.control_options_source(proto::PluginConfigOptionsSource::RuntimeMeshPeers)
    }

    pub fn control_availability(
        mut self,
        enabled: bool,
        source: proto::PluginConfigControlAvailabilitySource,
    ) -> Self {
        let availability = self.control_availability_mut();
        availability.enabled = enabled;
        availability.source = source as i32;
        self
    }

    pub fn control_availability_reason(mut self, reason: impl Into<String>) -> Self {
        self.control_availability_mut().reason = Some(reason.into());
        self
    }

    pub fn control_availability_note(mut self, note: impl Into<String>) -> Self {
        self.control_availability_mut().note = Some(note.into());
        self
    }

    pub fn control_enable_when(mut self, condition: proto::PluginConfigControlCondition) -> Self {
        self.control_behavior_mut().enable_when.push(condition);
        self
    }

    pub fn control_disable_when(mut self, disable: proto::PluginConfigConditionalDisable) -> Self {
        self.control_behavior_mut().disable_when.push(disable);
        self
    }

    pub fn control_conflict(mut self, conflict: proto::PluginConfigConflictRule) -> Self {
        self.control_behavior_mut().conflicts.push(conflict);
        self
    }

    pub fn control_write_policy(mut self, policy: proto::PluginConfigDisabledWritePolicy) -> Self {
        self.control_behavior_mut().write_policy = Some(policy as i32);
        self
    }

    fn control_behavior_mut(&mut self) -> &mut proto::PluginConfigControlBehavior {
        self.inner
            .control_behavior
            .get_or_insert_with(proto::PluginConfigControlBehavior::default)
    }

    fn control_numeric_mut(&mut self) -> &mut proto::PluginConfigNumericControl {
        self.control_behavior_mut()
            .numeric
            .get_or_insert_with(proto::PluginConfigNumericControl::default)
    }

    fn control_availability_mut(&mut self) -> &mut proto::PluginConfigControlAvailability {
        self.control_behavior_mut().availability.get_or_insert(
            proto::PluginConfigControlAvailability {
                enabled: true,
                reason: None,
                note: None,
                source: proto::PluginConfigControlAvailabilitySource::Static as i32,
            },
        )
    }
}
