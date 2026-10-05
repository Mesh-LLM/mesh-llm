use super::{
    ConfigConditionOperator, ConfigConditionValue, ConfigConditionalDisable, ConfigConflictRule,
    ConfigControlAvailability, ConfigControlAvailabilitySource, ConfigControlBehavior,
    ConfigControlCondition, ConfigDisabledWritePolicy, ConfigNumericControl, ConfigOptionsSource,
    ConfigPath, ConfigTextFormat,
};
use mesh_llm_plugin_manager::{
    InstalledPluginConditionOperator, InstalledPluginConditionValue,
    InstalledPluginConditionalDisable, InstalledPluginConflictRule,
    InstalledPluginControlAvailability, InstalledPluginControlAvailabilitySource,
    InstalledPluginControlBehavior, InstalledPluginControlCondition,
    InstalledPluginDisabledWritePolicy, InstalledPluginOptionsSource, InstalledPluginTextFormat,
};

pub(super) fn plugin_control_behavior_from_installed(
    behavior: &InstalledPluginControlBehavior,
) -> ConfigControlBehavior {
    ConfigControlBehavior {
        numeric: behavior
            .numeric
            .as_ref()
            .map(|numeric| ConfigNumericControl {
                min: numeric.min,
                max: numeric.max,
                step: numeric.step,
                soft_min: numeric.soft_min,
                soft_max: numeric.soft_max,
                unit: numeric.unit.clone(),
            }),
        text_format: behavior.text_format.map(plugin_text_format_from_installed),
        options_source: behavior
            .options_source
            .map(plugin_options_source_from_installed),
        availability: behavior
            .availability
            .as_ref()
            .map(plugin_availability_from_installed),
        enable_when: behavior
            .enable_when
            .iter()
            .map(plugin_condition_from_installed)
            .collect(),
        disable_when: behavior
            .disable_when
            .iter()
            .map(plugin_disable_from_installed)
            .collect(),
        conflicts: behavior
            .conflicts
            .iter()
            .map(plugin_conflict_from_installed)
            .collect(),
        write_policy: behavior
            .write_policy
            .map(plugin_write_policy_from_installed),
    }
}

fn plugin_text_format_from_installed(format: InstalledPluginTextFormat) -> ConfigTextFormat {
    match format {
        InstalledPluginTextFormat::Plain => ConfigTextFormat::Plain,
        InstalledPluginTextFormat::Path => ConfigTextFormat::Path,
        InstalledPluginTextFormat::Url => ConfigTextFormat::Url,
        InstalledPluginTextFormat::SocketAddr => ConfigTextFormat::SocketAddr,
        InstalledPluginTextFormat::Semver => ConfigTextFormat::Semver,
        InstalledPluginTextFormat::Ed25519Key => ConfigTextFormat::Ed25519Key,
        InstalledPluginTextFormat::CsvPositiveInts => ConfigTextFormat::CsvPositiveInts,
    }
}

fn plugin_options_source_from_installed(
    source: InstalledPluginOptionsSource,
) -> ConfigOptionsSource {
    match source {
        InstalledPluginOptionsSource::Static => ConfigOptionsSource::Static,
        InstalledPluginOptionsSource::RuntimeGpus => ConfigOptionsSource::RuntimeGpus,
        InstalledPluginOptionsSource::RuntimeNativeBackends => {
            ConfigOptionsSource::RuntimeNativeBackends
        }
        InstalledPluginOptionsSource::RuntimeLocalModels => ConfigOptionsSource::RuntimeLocalModels,
        InstalledPluginOptionsSource::RuntimeInstalledPlugins => {
            ConfigOptionsSource::RuntimeInstalledPlugins
        }
        InstalledPluginOptionsSource::RuntimeMeshPeers => ConfigOptionsSource::RuntimeMeshPeers,
    }
}

fn plugin_availability_from_installed(
    availability: &InstalledPluginControlAvailability,
) -> ConfigControlAvailability {
    ConfigControlAvailability {
        enabled: availability.enabled,
        reason: availability.reason.clone(),
        note: availability.note.clone(),
        source: match availability.source {
            InstalledPluginControlAvailabilitySource::Static => {
                ConfigControlAvailabilitySource::Static
            }
            InstalledPluginControlAvailabilitySource::Runtime => {
                ConfigControlAvailabilitySource::Runtime
            }
            InstalledPluginControlAvailabilitySource::Dependency => {
                ConfigControlAvailabilitySource::Dependency
            }
            InstalledPluginControlAvailabilitySource::Conflict => {
                ConfigControlAvailabilitySource::Conflict
            }
        },
    }
}

fn plugin_condition_from_installed(
    condition: &InstalledPluginControlCondition,
) -> ConfigControlCondition {
    ConfigControlCondition {
        path: ConfigPath::field(condition.key.clone()),
        operator: match condition.operator {
            InstalledPluginConditionOperator::Equals => ConfigConditionOperator::Equals,
            InstalledPluginConditionOperator::NotEquals => ConfigConditionOperator::NotEquals,
            InstalledPluginConditionOperator::In => ConfigConditionOperator::In,
            InstalledPluginConditionOperator::NotIn => ConfigConditionOperator::NotIn,
            InstalledPluginConditionOperator::Present => ConfigConditionOperator::Present,
            InstalledPluginConditionOperator::Absent => ConfigConditionOperator::Absent,
            InstalledPluginConditionOperator::Truthy => ConfigConditionOperator::Truthy,
            InstalledPluginConditionOperator::Falsy => ConfigConditionOperator::Falsy,
            InstalledPluginConditionOperator::Range => ConfigConditionOperator::Range,
        },
        values: condition
            .values
            .iter()
            .map(|value| match value {
                InstalledPluginConditionValue::Bool(value) => ConfigConditionValue::Bool(*value),
                InstalledPluginConditionValue::Integer(value) => {
                    ConfigConditionValue::Integer(*value)
                }
                InstalledPluginConditionValue::Float(value) => ConfigConditionValue::Float(*value),
                InstalledPluginConditionValue::String(value) => {
                    ConfigConditionValue::String(value.clone())
                }
            })
            .collect(),
    }
}

fn plugin_disable_from_installed(
    disable: &InstalledPluginConditionalDisable,
) -> ConfigConditionalDisable {
    ConfigConditionalDisable {
        condition: plugin_condition_from_installed(&disable.condition),
        reason: disable.reason.clone(),
        note: disable.note.clone(),
        write_policy: plugin_write_policy_from_installed(disable.write_policy),
    }
}

fn plugin_conflict_from_installed(conflict: &InstalledPluginConflictRule) -> ConfigConflictRule {
    ConfigConflictRule {
        group: conflict.group.clone(),
        condition: plugin_condition_from_installed(&conflict.condition),
        reason: conflict.reason.clone(),
        preferred_path: conflict.preferred_key.as_ref().map(ConfigPath::field),
    }
}

fn plugin_write_policy_from_installed(
    policy: InstalledPluginDisabledWritePolicy,
) -> ConfigDisabledWritePolicy {
    match policy {
        InstalledPluginDisabledWritePolicy::PreserveExisting => {
            ConfigDisabledWritePolicy::PreserveExisting
        }
        InstalledPluginDisabledWritePolicy::OmitWhenDisabled => {
            ConfigDisabledWritePolicy::OmitWhenDisabled
        }
        InstalledPluginDisabledWritePolicy::RejectWhenDisabled => {
            ConfigDisabledWritePolicy::RejectWhenDisabled
        }
    }
}
