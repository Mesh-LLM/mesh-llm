use mesh_llm_config::{
    PluginConfigSchema, PluginObjectPropertySchema, PluginSchemaAvailability,
    PluginSettingConstraint, PluginSettingSchema, PluginValueKind, PluginValueSchema,
};
use mesh_llm_plugin_manager::{
    InstalledPluginConfigSchema, InstalledPluginConstraint, InstalledPluginMetadata,
    InstalledPluginObjectProperty, InstalledPluginValueKind, InstalledPluginValueSchema,
    PluginStore, default_store_root,
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
            })
            .collect(),
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
