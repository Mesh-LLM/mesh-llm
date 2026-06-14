use anyhow::Context;
use mesh_llm_config::{
    ConfigAliasPolicy, ConfigApplyMode, ConfigConstraint, ConfigControlSurface, ConfigPath,
    ConfigRestartScope, ConfigSchema, ConfigSettingOwner, ConfigSettingSchema, ConfigSupportState,
    ConfigValueSchema, ConfigVisibility, built_in_config_schema,
};
use mesh_llm_plugin_manager::{
    InstalledPluginApplyMode, InstalledPluginConfigSchema, InstalledPluginConstraint,
    InstalledPluginMetadata, InstalledPluginRestartScope, InstalledPluginValueKind,
    InstalledPluginValueSchema, InstalledPluginVisibility, PluginStore, default_store_root,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregatedConfigSchema {
    settings_by_path: BTreeMap<ConfigPath, AggregatedConfigSchemaEntry>,
}

impl AggregatedConfigSchema {
    pub fn get(&self, path: &ConfigPath) -> Option<&AggregatedConfigSchemaEntry> {
        self.settings_by_path.get(path)
    }

    pub fn settings_by_path(&self) -> &BTreeMap<ConfigPath, AggregatedConfigSchemaEntry> {
        &self.settings_by_path
    }

    pub fn iter(&self) -> impl Iterator<Item = (&ConfigPath, &AggregatedConfigSchemaEntry)> {
        self.settings_by_path.iter()
    }

    pub fn export_reference(&self) -> ConfigSchemaReference {
        ConfigSchemaReference {
            settings: self
                .settings_by_path
                .values()
                .map(ConfigSchemaReferenceEntry::from)
                .collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ConfigSchemaReference {
    pub settings: Vec<ConfigSchemaReferenceEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ConfigSchemaReferenceEntry {
    pub canonical_path: String,
    pub owner: ConfigSettingOwner,
    pub source: ConfigSchemaReferenceSource,
    pub support: ConfigSupportState,
    pub control_surfaces: Vec<ConfigControlSurface>,
    pub apply_mode: ConfigApplyMode,
    pub restart_scope: ConfigRestartScope,
    pub visibility: ConfigVisibility,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl From<&AggregatedConfigSchemaEntry> for ConfigSchemaReferenceEntry {
    fn from(value: &AggregatedConfigSchemaEntry) -> Self {
        Self {
            canonical_path: value.setting.path.render(),
            owner: value.setting.owner,
            source: ConfigSchemaReferenceSource::from(&value.source),
            support: value.setting.support,
            control_surfaces: value.setting.control_surfaces.clone(),
            apply_mode: value.setting.apply_mode,
            restart_scope: value.setting.restart_scope,
            visibility: value.setting.visibility,
            description: value.setting.description.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConfigSchemaReferenceSource {
    BuiltIn,
    Engine {
        engine_id: String,
    },
    Plugin {
        plugin_name: String,
        allow_unvalidated_config: bool,
    },
}

impl From<&AggregatedConfigSchemaSource> for ConfigSchemaReferenceSource {
    fn from(value: &AggregatedConfigSchemaSource) -> Self {
        match value {
            AggregatedConfigSchemaSource::BuiltIn => Self::BuiltIn,
            AggregatedConfigSchemaSource::Engine { engine_id } => Self::Engine {
                engine_id: engine_id.clone(),
            },
            AggregatedConfigSchemaSource::Plugin {
                plugin_name,
                allow_unvalidated_config,
            } => Self::Plugin {
                plugin_name: plugin_name.clone(),
                allow_unvalidated_config: *allow_unvalidated_config,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregatedConfigSchemaEntry {
    pub setting: ConfigSettingSchema,
    pub source: AggregatedConfigSchemaSource,
    pub unknown_policy: AggregatedConfigUnknownPolicy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AggregatedConfigSchemaSource {
    BuiltIn,
    Engine {
        engine_id: String,
    },
    Plugin {
        plugin_name: String,
        allow_unvalidated_config: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregatedConfigUnknownPolicy {
    Reject,
    PreserveWithDiagnostics,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineConfigSchemaDescriptor {
    pub engine_id: String,
    pub schema: ConfigSchema,
}

#[derive(Debug)]
pub enum AggregatedConfigSchemaError {
    DuplicatePath {
        path: ConfigPath,
        existing_source: AggregatedConfigSchemaSource,
        incoming_source: AggregatedConfigSchemaSource,
    },
    PluginStore(anyhow::Error),
}

impl fmt::Display for AggregatedConfigSchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicatePath {
                path,
                existing_source,
                incoming_source,
            } => write!(
                f,
                "duplicate aggregated config schema path '{}' from {:?}; already registered by {:?}",
                path.render(),
                incoming_source,
                existing_source
            ),
            Self::PluginStore(error) => write!(
                f,
                "failed to load installed plugin schema metadata: {error}"
            ),
        }
    }
}

impl std::error::Error for AggregatedConfigSchemaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DuplicatePath { .. } => None,
            Self::PluginStore(error) => Some(error.as_ref()),
        }
    }
}

pub fn aggregate_runtime_config_schema(
    engine_schemas: impl IntoIterator<Item = EngineConfigSchemaDescriptor>,
) -> Result<AggregatedConfigSchema, AggregatedConfigSchemaError> {
    let root = default_store_root().map_err(AggregatedConfigSchemaError::PluginStore)?;
    let installed_plugins = PluginStore::new(root)
        .list()
        .context("list installed plugins")
        .map_err(AggregatedConfigSchemaError::PluginStore)?;
    aggregate_config_schema_sources(engine_schemas, installed_plugins)
}

pub fn export_runtime_config_schema_reference(
    engine_schemas: impl IntoIterator<Item = EngineConfigSchemaDescriptor>,
) -> Result<ConfigSchemaReference, AggregatedConfigSchemaError> {
    aggregate_runtime_config_schema(engine_schemas).map(|schema| schema.export_reference())
}

pub fn aggregate_config_schema_sources(
    engine_schemas: impl IntoIterator<Item = EngineConfigSchemaDescriptor>,
    installed_plugins: impl IntoIterator<Item = InstalledPluginMetadata>,
) -> Result<AggregatedConfigSchema, AggregatedConfigSchemaError> {
    let mut settings_by_path = BTreeMap::new();

    for setting in built_in_config_schema().settings {
        register_setting(
            &mut settings_by_path,
            setting,
            AggregatedConfigSchemaSource::BuiltIn,
            AggregatedConfigUnknownPolicy::Reject,
        )?;
    }

    for engine in engine_schemas {
        let source = AggregatedConfigSchemaSource::Engine {
            engine_id: engine.engine_id,
        };
        for setting in engine.schema.settings {
            register_setting(
                &mut settings_by_path,
                setting,
                source.clone(),
                AggregatedConfigUnknownPolicy::PreserveWithDiagnostics,
            )?;
        }
    }

    for plugin in installed_plugins {
        let Some(schema) = plugin
            .manifest
            .as_ref()
            .and_then(|manifest| manifest.config_schema.as_ref())
        else {
            continue;
        };

        let source = AggregatedConfigSchemaSource::Plugin {
            plugin_name: schema.plugin_name.clone(),
            allow_unvalidated_config: schema.allow_unvalidated_config,
        };

        for setting in plugin_settings_from_installed_schema(schema) {
            register_setting(
                &mut settings_by_path,
                setting,
                source.clone(),
                AggregatedConfigUnknownPolicy::Reject,
            )?;
        }
    }

    Ok(AggregatedConfigSchema { settings_by_path })
}

fn register_setting(
    settings_by_path: &mut BTreeMap<ConfigPath, AggregatedConfigSchemaEntry>,
    mut setting: ConfigSettingSchema,
    source: AggregatedConfigSchemaSource,
    unknown_policy: AggregatedConfigUnknownPolicy,
) -> Result<(), AggregatedConfigSchemaError> {
    setting.path = setting.path.normalize_builtin_layout();

    if let Some(existing) = settings_by_path.get(&setting.path) {
        return Err(AggregatedConfigSchemaError::DuplicatePath {
            path: setting.path.clone(),
            existing_source: existing.source.clone(),
            incoming_source: source,
        });
    }

    settings_by_path.insert(
        setting.path.clone(),
        AggregatedConfigSchemaEntry {
            setting,
            source,
            unknown_policy,
        },
    );
    Ok(())
}

fn plugin_settings_from_installed_schema(
    schema: &InstalledPluginConfigSchema,
) -> Vec<ConfigSettingSchema> {
    schema
        .settings
        .iter()
        .map(|setting| ConfigSettingSchema {
            path: ConfigPath::from_fields([
                "plugin",
                schema.plugin_name.as_str(),
                "settings",
                setting.key.as_str(),
            ]),
            alias_policy: ConfigAliasPolicy::default(),
            owner: ConfigSettingOwner::Plugin,
            value_schema: plugin_value_schema_from_installed(&setting.value_schema),
            support: ConfigSupportState::Supported,
            control_surfaces: vec![
                ConfigControlSurface::ConfigFile,
                ConfigControlSurface::OwnerControl,
                ConfigControlSurface::PluginManifest,
            ],
            apply_mode: plugin_apply_mode_from_installed(setting.apply_mode),
            restart_scope: plugin_restart_scope_from_installed(setting.restart_scope),
            visibility: plugin_visibility_from_installed(setting.visibility),
            constraints: setting
                .constraints
                .iter()
                .map(plugin_constraint_from_installed)
                .collect(),
            description: setting.description.clone(),
        })
        .collect()
}

fn plugin_value_schema_from_installed(schema: &InstalledPluginValueSchema) -> ConfigValueSchema {
    match schema.kind {
        InstalledPluginValueKind::Boolean => ConfigValueSchema::Boolean,
        InstalledPluginValueKind::Integer => ConfigValueSchema::Integer,
        InstalledPluginValueKind::Float => ConfigValueSchema::Float,
        InstalledPluginValueKind::String => ConfigValueSchema::String,
        InstalledPluginValueKind::Path => ConfigValueSchema::String,
        InstalledPluginValueKind::Url => ConfigValueSchema::String,
        InstalledPluginValueKind::Enum => ConfigValueSchema::Enum {
            values: schema.enum_values.clone(),
        },
        InstalledPluginValueKind::Array => ConfigValueSchema::Array {
            items: Box::new(
                schema
                    .items
                    .as_deref()
                    .map(plugin_value_schema_from_installed)
                    .unwrap_or(ConfigValueSchema::String),
            ),
        },
        InstalledPluginValueKind::Object => ConfigValueSchema::Object,
    }
}

fn plugin_constraint_from_installed(constraint: &InstalledPluginConstraint) -> ConfigConstraint {
    match constraint {
        InstalledPluginConstraint::NonEmpty => ConfigConstraint::NonEmpty,
        InstalledPluginConstraint::Positive => ConfigConstraint::Positive,
        InstalledPluginConstraint::Range { min, max } => ConfigConstraint::Range {
            min: min.clone(),
            max: max.clone(),
        },
        InstalledPluginConstraint::AllowedValues { values } => ConfigConstraint::AllowedValues {
            values: values.clone(),
        },
        InstalledPluginConstraint::Requires { key } => ConfigConstraint::Requires {
            path: ConfigPath::field(key.clone()),
        },
    }
}

fn plugin_apply_mode_from_installed(mode: InstalledPluginApplyMode) -> ConfigApplyMode {
    match mode {
        InstalledPluginApplyMode::StaticOnLoad => ConfigApplyMode::StaticOnLoad,
        InstalledPluginApplyMode::DynamicValidationOnly => ConfigApplyMode::DynamicValidationOnly,
        InstalledPluginApplyMode::DynamicApply => ConfigApplyMode::DynamicApply,
    }
}

fn plugin_restart_scope_from_installed(scope: InstalledPluginRestartScope) -> ConfigRestartScope {
    match scope {
        InstalledPluginRestartScope::None => ConfigRestartScope::None,
        InstalledPluginRestartScope::ModelReload => ConfigRestartScope::ModelReload,
        InstalledPluginRestartScope::ProcessRestart
        | InstalledPluginRestartScope::PluginProcess => ConfigRestartScope::ProcessRestart,
        InstalledPluginRestartScope::MeshRestart => ConfigRestartScope::MeshRestart,
    }
}

fn plugin_visibility_from_installed(visibility: InstalledPluginVisibility) -> ConfigVisibility {
    match visibility {
        InstalledPluginVisibility::User => ConfigVisibility::User,
        InstalledPluginVisibility::Advanced => ConfigVisibility::Advanced,
        InstalledPluginVisibility::Hidden => ConfigVisibility::Hidden,
        InstalledPluginVisibility::Internal => ConfigVisibility::Internal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_config::{ConfigSchemaBuilder, ConfigSettingSchemaBuilder};
    use mesh_llm_plugin_manager::{
        InstalledPluginConfigSchema, InstalledPluginManifestMetadata,
        InstalledPluginObjectProperty, InstalledPluginSettingSchema,
    };
    use serde::Serialize;
    use std::path::PathBuf;

    const CONFIG_SCHEMA_REFERENCE_FIXTURE: &str =
        include_str!("../tests/fixtures/config_schema_reference.json");
    const CONFIG_SCHEMA_DEFAULTS_UI_FIXTURE: &str =
        include_str!("../tests/fixtures/config_schema_defaults_ui_reference.json");

    #[derive(Serialize)]
    struct DefaultsUiSchemaReference {
        settings: Vec<DefaultsUiSchemaReferenceEntry>,
    }

    #[derive(Serialize)]
    struct DefaultsUiSchemaReferenceEntry {
        canonical_path: String,
        support: ConfigSupportState,
        source: ConfigSchemaReferenceSource,
    }

    #[test]
    fn aggregated_config_schema_sources() {
        let mut engine_schema = ConfigSchemaBuilder::new();
        let mut engine_setting = ConfigSettingSchemaBuilder::new(
            ConfigPath::from_fields(["defaults", "engine", "vllm", "temperature"]),
            ConfigValueSchema::Float,
        );
        engine_setting.owner(ConfigSettingOwner::Engine);
        engine_schema.setting(engine_setting.build());

        let aggregated = aggregate_config_schema_sources(
            [EngineConfigSchemaDescriptor {
                engine_id: "vllm".into(),
                schema: engine_schema.build(),
            }],
            [installed_plugin_metadata(
                "blackboard",
                vec![InstalledPluginSettingSchema {
                    key: "retention_days".into(),
                    value_schema: InstalledPluginValueSchema {
                        kind: InstalledPluginValueKind::Integer,
                        enum_values: Vec::new(),
                        items: None,
                        object_properties: vec![InstalledPluginObjectProperty {
                            key: "unused".into(),
                            value_schema: InstalledPluginValueSchema {
                                kind: InstalledPluginValueKind::String,
                                enum_values: Vec::new(),
                                items: None,
                                object_properties: Vec::new(),
                                allow_additional_properties: false,
                            },
                            required: false,
                            description: None,
                        }],
                        allow_additional_properties: false,
                    },
                    required: true,
                    default_json: Some("14".into()),
                    constraints: vec![InstalledPluginConstraint::Range {
                        min: Some("1".into()),
                        max: Some("365".into()),
                    }],
                    apply_mode: InstalledPluginApplyMode::DynamicApply,
                    restart_scope: InstalledPluginRestartScope::PluginProcess,
                    visibility: InstalledPluginVisibility::Advanced,
                    description: Some("Retention period in days".into()),
                }],
            )],
        )
        .expect("schema aggregation should succeed");

        let built_in = aggregated
            .get(&ConfigPath::field("version"))
            .expect("built-in version setting should be present");
        assert_eq!(built_in.source, AggregatedConfigSchemaSource::BuiltIn);
        assert_eq!(
            built_in.unknown_policy,
            AggregatedConfigUnknownPolicy::Reject
        );

        let engine = aggregated
            .get(&ConfigPath::from_fields([
                "defaults",
                "engine",
                "vllm",
                "temperature",
            ]))
            .expect("engine setting should be present");
        assert_eq!(
            engine.source,
            AggregatedConfigSchemaSource::Engine {
                engine_id: "vllm".into(),
            }
        );
        assert_eq!(
            engine.unknown_policy,
            AggregatedConfigUnknownPolicy::PreserveWithDiagnostics
        );

        let plugin = aggregated
            .get(&ConfigPath::from_fields([
                "plugin",
                "blackboard",
                "settings",
                "retention_days",
            ]))
            .expect("plugin setting should be present under canonical plugin path");
        assert_eq!(
            plugin.source,
            AggregatedConfigSchemaSource::Plugin {
                plugin_name: "blackboard".into(),
                allow_unvalidated_config: false,
            }
        );
        assert_eq!(plugin.unknown_policy, AggregatedConfigUnknownPolicy::Reject);
        assert_eq!(plugin.setting.owner, ConfigSettingOwner::Plugin);
        assert_eq!(
            plugin.setting.path.render(),
            "plugin.blackboard.settings.retention_days"
        );
    }

    #[test]
    fn aggregated_schema_duplicate_paths() {
        let mut duplicate_schema = ConfigSchemaBuilder::new();
        let mut duplicate_setting = ConfigSettingSchemaBuilder::new(
            ConfigPath::field("version"),
            ConfigValueSchema::Integer,
        );
        duplicate_setting.owner(ConfigSettingOwner::Engine);
        duplicate_schema.setting(duplicate_setting.build());

        let error = aggregate_config_schema_sources(
            [EngineConfigSchemaDescriptor {
                engine_id: "vllm".into(),
                schema: duplicate_schema.build(),
            }],
            Vec::<InstalledPluginMetadata>::new(),
        )
        .expect_err("duplicate canonical paths should fail deterministically");

        match error {
            AggregatedConfigSchemaError::DuplicatePath {
                path,
                existing_source,
                incoming_source,
            } => {
                assert_eq!(path.render(), "version");
                assert_eq!(existing_source, AggregatedConfigSchemaSource::BuiltIn);
                assert_eq!(
                    incoming_source,
                    AggregatedConfigSchemaSource::Engine {
                        engine_id: "vllm".into(),
                    }
                );
            }
            other => panic!("unexpected aggregation error: {other}"),
        }
    }

    #[test]
    fn schema_export_snapshot() {
        let mut engine_schema = ConfigSchemaBuilder::new();
        let mut engine_setting = ConfigSettingSchemaBuilder::new(
            ConfigPath::from_fields(["defaults", "engine", "vllm", "temperature"]),
            ConfigValueSchema::Float,
        );
        engine_setting.owner(ConfigSettingOwner::Engine);
        engine_setting.control_surface(ConfigControlSurface::Api);
        engine_setting.control_surface(ConfigControlSurface::OwnerControl);
        engine_setting.apply_mode(ConfigApplyMode::DynamicApply);
        engine_setting.restart_scope(ConfigRestartScope::None);
        engine_setting.visibility(ConfigVisibility::Advanced);
        engine_setting.description("Engine temperature override.");
        engine_schema.setting(engine_setting.build());

        let exported = aggregate_config_schema_sources(
            [EngineConfigSchemaDescriptor {
                engine_id: "vllm".into(),
                schema: engine_schema.build(),
            }],
            [installed_plugin_metadata(
                "blackboard",
                vec![InstalledPluginSettingSchema {
                    key: "retention_days".into(),
                    value_schema: InstalledPluginValueSchema {
                        kind: InstalledPluginValueKind::Integer,
                        enum_values: Vec::new(),
                        items: None,
                        object_properties: Vec::new(),
                        allow_additional_properties: false,
                    },
                    required: true,
                    default_json: Some("14".into()),
                    constraints: vec![InstalledPluginConstraint::Range {
                        min: Some("1".into()),
                        max: Some("365".into()),
                    }],
                    apply_mode: InstalledPluginApplyMode::DynamicApply,
                    restart_scope: InstalledPluginRestartScope::PluginProcess,
                    visibility: InstalledPluginVisibility::Advanced,
                    description: Some("Retention period in days".into()),
                }],
            )],
        )
        .expect("schema aggregation should succeed")
        .export_reference();

        let filtered = ConfigSchemaReference {
            settings: exported
                .settings
                .into_iter()
                .filter(|entry| {
                    matches!(
                        entry.canonical_path.as_str(),
                        "version"
                            | "telemetry.prompt_shape_metrics"
                            | "defaults.request_defaults.dry"
                            | "defaults.engine.vllm.temperature"
                            | "plugin.blackboard.settings.retention_days"
                    )
                })
                .collect(),
        };
        let actual = serde_json::to_string_pretty(&filtered)
            .expect("schema reference export should serialize to json");
        let expected = CONFIG_SCHEMA_REFERENCE_FIXTURE.trim();

        assert_eq!(actual, expected, "schema export snapshot drifted\n{actual}");
    }

    #[test]
    fn schema_export_omits_plugins_without_install_time_schema() {
        let exported = aggregate_config_schema_sources(
            Vec::<EngineConfigSchemaDescriptor>::new(),
            [InstalledPluginMetadata {
                name: "blackboard".into(),
                source_repository: "mesh-llm/blackboard".into(),
                installed_version: "0.1.0".into(),
                target_triple: "aarch64-apple-darwin".into(),
                downloaded_asset_name: "blackboard.tar.gz".into(),
                install_path: PathBuf::from("/tmp/blackboard"),
                enabled: true,
                manifest: Some(InstalledPluginManifestMetadata {
                    config_schema: None,
                }),
                last_protocol_version: None,
                last_status: None,
                last_error: None,
            }],
        )
        .expect("schema aggregation should succeed")
        .export_reference();

        assert!(exported.settings.iter().all(|entry| {
            !entry
                .canonical_path
                .starts_with("plugin.blackboard.settings.")
        }));
    }

    #[test]
    fn defaults_ui_schema_export_snapshot() {
        let exported = aggregate_config_schema_sources(
            Vec::<EngineConfigSchemaDescriptor>::new(),
            Vec::<InstalledPluginMetadata>::new(),
        )
        .expect("schema aggregation should succeed")
        .export_reference();

        let filtered = DefaultsUiSchemaReference {
            settings: exported
                .settings
                .into_iter()
                .filter(|entry| {
                    entry.source == ConfigSchemaReferenceSource::BuiltIn
                        && entry.support == ConfigSupportState::Supported
                        && entry.canonical_path.starts_with("defaults.")
                })
                .map(|entry| DefaultsUiSchemaReferenceEntry {
                    canonical_path: entry.canonical_path,
                    support: entry.support,
                    source: entry.source,
                })
                .collect(),
        };
        let actual = serde_json::to_string_pretty(&filtered)
            .expect("defaults ui schema reference should serialize to json");
        let expected = CONFIG_SCHEMA_DEFAULTS_UI_FIXTURE.trim();

        assert_eq!(
            actual, expected,
            "defaults UI schema export snapshot drifted\n{actual}"
        );
    }

    fn installed_plugin_metadata(
        plugin_name: &str,
        settings: Vec<InstalledPluginSettingSchema>,
    ) -> InstalledPluginMetadata {
        InstalledPluginMetadata {
            name: plugin_name.into(),
            source_repository: format!("mesh-llm/{plugin_name}"),
            installed_version: "0.1.0".into(),
            target_triple: "aarch64-apple-darwin".into(),
            downloaded_asset_name: format!("{plugin_name}.tar.gz"),
            install_path: PathBuf::from(format!("/tmp/{plugin_name}")),
            enabled: true,
            manifest: Some(InstalledPluginManifestMetadata {
                config_schema: Some(InstalledPluginConfigSchema {
                    plugin_name: plugin_name.into(),
                    schema_version: mesh_llm_plugin_manager::SUPPORTED_PLUGIN_SCHEMA_VERSION,
                    allow_unvalidated_config: false,
                    settings,
                }),
            }),
            last_protocol_version: None,
            last_status: None,
            last_error: None,
        }
    }
}
