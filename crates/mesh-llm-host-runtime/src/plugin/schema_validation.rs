use mesh_llm_config::{
    PluginConditionOperator, PluginConditionValue, PluginConditionalDisable, PluginConfigSchema,
    PluginConflictRule, PluginControlAvailability, PluginControlAvailabilitySource,
    PluginControlBehavior, PluginControlCondition, PluginDisabledWritePolicy, PluginNumericControl,
    PluginObjectPropertySchema, PluginOptionsSource, PluginSchemaAvailability,
    PluginSettingConstraint, PluginSettingSchema, PluginTextFormat, PluginValueKind,
    PluginValueSchema,
};
use mesh_llm_plugin_manager::{
    InstalledPluginConditionOperator, InstalledPluginConditionValue,
    InstalledPluginConditionalDisable, InstalledPluginConfigSchema, InstalledPluginConflictRule,
    InstalledPluginConstraint, InstalledPluginControlAvailability,
    InstalledPluginControlAvailabilitySource, InstalledPluginControlBehavior,
    InstalledPluginControlCondition, InstalledPluginDisabledWritePolicy, InstalledPluginMetadata,
    InstalledPluginObjectProperty, InstalledPluginOptionsSource, InstalledPluginTextFormat,
    InstalledPluginValueKind, InstalledPluginValueSchema, PluginStore, default_store_root,
};
use std::path::Path;

pub(crate) fn strict_plugin_schema_availability(plugin_name: &str) -> PluginSchemaAvailability {
    let Ok(root) = default_store_root() else {
        return PluginSchemaAvailability::NotInstalled;
    };
    plugin_schema_availability_from_store_root(&root, plugin_name)
}

pub(crate) fn plugin_schema_availability_from_store_root(
    root: &Path,
    plugin_name: &str,
) -> PluginSchemaAvailability {
    let store = PluginStore::new(root);
    let Ok(metadata) = store.load_optional(plugin_name) else {
        return PluginSchemaAvailability::NotInstalled;
    };
    let Some(metadata) = metadata else {
        return PluginSchemaAvailability::NotInstalled;
    };
    plugin_schema_from_metadata(&metadata)
}

fn plugin_schema_from_metadata(metadata: &InstalledPluginMetadata) -> PluginSchemaAvailability {
    let Some(schema) = metadata
        .manifest
        .as_ref()
        .and_then(|manifest| manifest.config_schema.as_ref())
    else {
        return PluginSchemaAvailability::MissingSchema;
    };

    if schema.schema_version != mesh_llm_config::SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION {
        return PluginSchemaAvailability::UnsupportedVersion {
            version: schema.schema_version,
        };
    }

    PluginSchemaAvailability::Available(plugin_schema_from_installed(schema))
}

fn plugin_schema_from_installed(schema: &InstalledPluginConfigSchema) -> PluginConfigSchema {
    PluginConfigSchema {
        plugin_name: schema.plugin_name.clone(),
        schema_version: schema.schema_version,
        allow_unvalidated_config: schema.allow_unvalidated_config,
        settings: schema
            .settings
            .iter()
            .map(|setting| PluginSettingSchema {
                key: setting.key.clone(),
                value_schema: plugin_value_schema_from_installed(&setting.value_schema),
                required: setting.required,
                default_json: setting.default_json.clone(),
                constraints: setting
                    .constraints
                    .iter()
                    .map(plugin_constraint_from_installed)
                    .collect(),
                description: setting.description.clone(),
                control_behavior: setting
                    .control_behavior
                    .as_ref()
                    .map(plugin_control_behavior_from_installed),
            })
            .collect(),
    }
}

fn plugin_control_behavior_from_installed(
    behavior: &InstalledPluginControlBehavior,
) -> PluginControlBehavior {
    PluginControlBehavior {
        numeric: behavior
            .numeric
            .as_ref()
            .map(|numeric| PluginNumericControl {
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

fn plugin_value_schema_from_installed(schema: &InstalledPluginValueSchema) -> PluginValueSchema {
    PluginValueSchema {
        kind: match schema.kind {
            InstalledPluginValueKind::Boolean => PluginValueKind::Boolean,
            InstalledPluginValueKind::Integer => PluginValueKind::Integer,
            InstalledPluginValueKind::Float => PluginValueKind::Float,
            InstalledPluginValueKind::String => PluginValueKind::String,
            InstalledPluginValueKind::Path => PluginValueKind::Path,
            InstalledPluginValueKind::Url => PluginValueKind::Url,
            InstalledPluginValueKind::Enum => PluginValueKind::Enum,
            InstalledPluginValueKind::Array => PluginValueKind::Array,
            InstalledPluginValueKind::Object => PluginValueKind::Object,
        },
        enum_values: schema.enum_values.clone(),
        items: schema
            .items
            .as_deref()
            .map(plugin_value_schema_from_installed)
            .map(Box::new),
        object_properties: schema
            .object_properties
            .iter()
            .map(plugin_object_property_from_installed)
            .collect(),
        allow_additional_properties: schema.allow_additional_properties,
    }
}

fn plugin_object_property_from_installed(
    property: &InstalledPluginObjectProperty,
) -> PluginObjectPropertySchema {
    PluginObjectPropertySchema {
        key: property.key.clone(),
        value_schema: plugin_value_schema_from_installed(&property.value_schema),
        required: property.required,
        description: property.description.clone(),
    }
}

fn plugin_text_format_from_installed(format: InstalledPluginTextFormat) -> PluginTextFormat {
    match format {
        InstalledPluginTextFormat::Plain => PluginTextFormat::Plain,
        InstalledPluginTextFormat::Path => PluginTextFormat::Path,
        InstalledPluginTextFormat::Url => PluginTextFormat::Url,
        InstalledPluginTextFormat::SocketAddr => PluginTextFormat::SocketAddr,
        InstalledPluginTextFormat::Semver => PluginTextFormat::Semver,
        InstalledPluginTextFormat::Ed25519Key => PluginTextFormat::Ed25519Key,
        InstalledPluginTextFormat::CsvPositiveInts => PluginTextFormat::CsvPositiveInts,
    }
}

fn plugin_options_source_from_installed(
    source: InstalledPluginOptionsSource,
) -> PluginOptionsSource {
    match source {
        InstalledPluginOptionsSource::Static => PluginOptionsSource::Static,
        InstalledPluginOptionsSource::RuntimeGpus => PluginOptionsSource::RuntimeGpus,
        InstalledPluginOptionsSource::RuntimeNativeBackends => {
            PluginOptionsSource::RuntimeNativeBackends
        }
        InstalledPluginOptionsSource::RuntimeLocalModels => PluginOptionsSource::RuntimeLocalModels,
        InstalledPluginOptionsSource::RuntimeInstalledPlugins => {
            PluginOptionsSource::RuntimeInstalledPlugins
        }
        InstalledPluginOptionsSource::RuntimeMeshPeers => PluginOptionsSource::RuntimeMeshPeers,
    }
}

fn plugin_availability_from_installed(
    availability: &InstalledPluginControlAvailability,
) -> PluginControlAvailability {
    PluginControlAvailability {
        enabled: availability.enabled,
        reason: availability.reason.clone(),
        note: availability.note.clone(),
        source: match availability.source {
            InstalledPluginControlAvailabilitySource::Static => {
                PluginControlAvailabilitySource::Static
            }
            InstalledPluginControlAvailabilitySource::Runtime => {
                PluginControlAvailabilitySource::Runtime
            }
            InstalledPluginControlAvailabilitySource::Dependency => {
                PluginControlAvailabilitySource::Dependency
            }
            InstalledPluginControlAvailabilitySource::Conflict => {
                PluginControlAvailabilitySource::Conflict
            }
        },
    }
}

fn plugin_condition_from_installed(
    condition: &InstalledPluginControlCondition,
) -> PluginControlCondition {
    PluginControlCondition {
        key: condition.key.clone(),
        operator: match condition.operator {
            InstalledPluginConditionOperator::Equals => PluginConditionOperator::Equals,
            InstalledPluginConditionOperator::NotEquals => PluginConditionOperator::NotEquals,
            InstalledPluginConditionOperator::In => PluginConditionOperator::In,
            InstalledPluginConditionOperator::NotIn => PluginConditionOperator::NotIn,
            InstalledPluginConditionOperator::Present => PluginConditionOperator::Present,
            InstalledPluginConditionOperator::Absent => PluginConditionOperator::Absent,
            InstalledPluginConditionOperator::Truthy => PluginConditionOperator::Truthy,
            InstalledPluginConditionOperator::Falsy => PluginConditionOperator::Falsy,
            InstalledPluginConditionOperator::Range => PluginConditionOperator::Range,
        },
        values: condition
            .values
            .iter()
            .map(|value| match value {
                InstalledPluginConditionValue::Bool(value) => PluginConditionValue::Bool(*value),
                InstalledPluginConditionValue::Integer(value) => {
                    PluginConditionValue::Integer(*value)
                }
                InstalledPluginConditionValue::Float(value) => PluginConditionValue::Float(*value),
                InstalledPluginConditionValue::String(value) => {
                    PluginConditionValue::String(value.clone())
                }
            })
            .collect(),
    }
}

fn plugin_disable_from_installed(
    disable: &InstalledPluginConditionalDisable,
) -> PluginConditionalDisable {
    PluginConditionalDisable {
        condition: plugin_condition_from_installed(&disable.condition),
        reason: disable.reason.clone(),
        note: disable.note.clone(),
        write_policy: plugin_write_policy_from_installed(disable.write_policy),
    }
}

fn plugin_conflict_from_installed(conflict: &InstalledPluginConflictRule) -> PluginConflictRule {
    PluginConflictRule {
        group: conflict.group.clone(),
        condition: plugin_condition_from_installed(&conflict.condition),
        reason: conflict.reason.clone(),
        preferred_key: conflict.preferred_key.clone(),
    }
}

fn plugin_write_policy_from_installed(
    policy: InstalledPluginDisabledWritePolicy,
) -> PluginDisabledWritePolicy {
    match policy {
        InstalledPluginDisabledWritePolicy::PreserveExisting => {
            PluginDisabledWritePolicy::PreserveExisting
        }
        InstalledPluginDisabledWritePolicy::OmitWhenDisabled => {
            PluginDisabledWritePolicy::OmitWhenDisabled
        }
        InstalledPluginDisabledWritePolicy::RejectWhenDisabled => {
            PluginDisabledWritePolicy::RejectWhenDisabled
        }
    }
}

fn plugin_constraint_from_installed(
    constraint: &InstalledPluginConstraint,
) -> PluginSettingConstraint {
    match constraint {
        InstalledPluginConstraint::NonEmpty => PluginSettingConstraint::NonEmpty,
        InstalledPluginConstraint::Positive => PluginSettingConstraint::Positive,
        InstalledPluginConstraint::Range { min, max } => PluginSettingConstraint::Range {
            min: min.clone(),
            max: max.clone(),
        },
        InstalledPluginConstraint::AllowedValues { values } => {
            PluginSettingConstraint::AllowedValues {
                values: values.clone(),
            }
        }
        InstalledPluginConstraint::Requires { key } => {
            PluginSettingConstraint::Requires { key: key.clone() }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_plugin_manager::{
        InstalledPluginApplyMode, InstalledPluginManifestMetadata, InstalledPluginNumericControl,
        InstalledPluginRestartScope, InstalledPluginSettingSchema, InstalledPluginTextFormat,
        InstalledPluginVisibility,
    };
    use std::path::PathBuf;

    #[test]
    fn installed_schema_conversion_preserves_path_url_and_control_metadata() {
        let metadata = InstalledPluginMetadata {
            name: "blackboard".to_string(),
            source_repository: "https://github.com/mesh-llm/blackboard".to_string(),
            installed_version: "v1.0.0".to_string(),
            target_triple: "aarch64-apple-darwin".to_string(),
            downloaded_asset_name: "blackboard.tar.gz".to_string(),
            install_path: PathBuf::from("/tmp/blackboard"),
            enabled: true,
            manifest: Some(InstalledPluginManifestMetadata {
                config_schema: Some(InstalledPluginConfigSchema {
                    plugin_name: "blackboard".to_string(),
                    schema_version: mesh_llm_config::SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION,
                    allow_unvalidated_config: true,
                    settings: vec![
                        InstalledPluginSettingSchema {
                            key: "projector_path".to_string(),
                            value_schema: InstalledPluginValueSchema {
                                kind: InstalledPluginValueKind::Path,
                                enum_values: Vec::new(),
                                items: None,
                                object_properties: Vec::new(),
                                allow_additional_properties: false,
                            },
                            required: false,
                            default_json: None,
                            constraints: Vec::new(),
                            apply_mode: InstalledPluginApplyMode::DynamicValidationOnly,
                            restart_scope: InstalledPluginRestartScope::PluginProcess,
                            visibility: InstalledPluginVisibility::User,
                            description: None,
                            presentation: None,
                            control_behavior: Some(InstalledPluginControlBehavior {
                                numeric: Some(InstalledPluginNumericControl {
                                    min: Some(1.0),
                                    max: Some(2.0),
                                    step: Some(1.0),
                                    soft_min: None,
                                    soft_max: None,
                                    unit: Some("files".to_string()),
                                }),
                                text_format: Some(InstalledPluginTextFormat::Path),
                                options_source: Some(
                                    InstalledPluginOptionsSource::RuntimeInstalledPlugins,
                                ),
                                availability: Some(InstalledPluginControlAvailability {
                                    enabled: false,
                                    reason: Some("Waiting for discovery".to_string()),
                                    note: None,
                                    source: InstalledPluginControlAvailabilitySource::Runtime,
                                }),
                                enable_when: vec![InstalledPluginControlCondition {
                                    key: "mode".to_string(),
                                    operator: InstalledPluginConditionOperator::Present,
                                    values: Vec::new(),
                                }],
                                disable_when: Vec::new(),
                                conflicts: Vec::new(),
                                write_policy: Some(
                                    InstalledPluginDisabledWritePolicy::PreserveExisting,
                                ),
                            }),
                        },
                        InstalledPluginSettingSchema {
                            key: "endpoint_url".to_string(),
                            value_schema: InstalledPluginValueSchema {
                                kind: InstalledPluginValueKind::Url,
                                enum_values: Vec::new(),
                                items: None,
                                object_properties: Vec::new(),
                                allow_additional_properties: false,
                            },
                            required: false,
                            default_json: None,
                            constraints: Vec::new(),
                            apply_mode: InstalledPluginApplyMode::DynamicValidationOnly,
                            restart_scope: InstalledPluginRestartScope::PluginProcess,
                            visibility: InstalledPluginVisibility::User,
                            description: None,
                            presentation: None,
                            control_behavior: None,
                        },
                    ],
                }),
            }),
            last_protocol_version: None,
            last_status: None,
            last_error: None,
        };

        let PluginSchemaAvailability::Available(schema) = plugin_schema_from_metadata(&metadata)
        else {
            panic!("schema should be available");
        };

        assert!(schema.allow_unvalidated_config);
        assert_eq!(schema.settings[0].value_schema.kind, PluginValueKind::Path);
        assert_eq!(schema.settings[1].value_schema.kind, PluginValueKind::Url);
        let control_behavior = schema.settings[0]
            .control_behavior
            .as_ref()
            .expect("control behavior should be preserved");
        assert_eq!(control_behavior.text_format, Some(PluginTextFormat::Path));
        assert_eq!(
            control_behavior.options_source,
            Some(PluginOptionsSource::RuntimeInstalledPlugins)
        );
        assert_eq!(
            control_behavior
                .availability
                .as_ref()
                .map(|availability| availability.enabled),
            Some(false)
        );
        assert_eq!(control_behavior.enable_when.len(), 1);
        assert_eq!(
            control_behavior.write_policy,
            Some(PluginDisabledWritePolicy::PreserveExisting)
        );
    }
}
