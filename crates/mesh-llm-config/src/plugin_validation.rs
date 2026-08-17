pub mod control_behavior;

use self::control_behavior::PluginControlBehavior;

use crate::PluginConfigEntry;
use crate::diagnostic::DiagnosticResult;
use crate::model::ConfigPath;
use crate::validate::{
    ConfigDiagnostic, ConfigDiagnosticCode, ConfigDiagnosticSchemaSource, ConfigDiagnosticSeverity,
    ConfigDiagnosticSource,
};
use crate::validation_support::validation_diagnostic;
use std::collections::{BTreeMap, BTreeSet};
use toml::Value;

pub const SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
pub enum PluginSchemaAvailability {
    Available(PluginConfigSchema),
    NotInstalled,
    MissingSchema,
    UnsupportedVersion { version: u32 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct PluginConfigSchema {
    pub plugin_name: String,
    pub schema_version: u32,
    pub allow_unvalidated_config: bool,
    pub settings: Vec<PluginSettingSchema>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PluginSettingSchema {
    pub key: String,
    pub value_schema: PluginValueSchema,
    pub required: bool,
    pub default_json: Option<String>,
    pub constraints: Vec<PluginSettingConstraint>,
    pub description: Option<String>,
    pub control_behavior: Option<PluginControlBehavior>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginValueSchema {
    pub kind: PluginValueKind,
    pub enum_values: Vec<String>,
    pub items: Option<Box<PluginValueSchema>>,
    pub object_properties: Vec<PluginObjectPropertySchema>,
    pub allow_additional_properties: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginObjectPropertySchema {
    pub key: String,
    pub value_schema: PluginValueSchema,
    pub required: bool,
    pub description: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PluginValueKind {
    Boolean,
    Integer,
    Float,
    String,
    Path,
    Url,
    Enum,
    Array,
    Object,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PluginSettingConstraint {
    NonEmpty,
    Positive,
    Range {
        min: Option<String>,
        max: Option<String>,
    },
    AllowedValues {
        values: Vec<String>,
    },
    Requires {
        key: String,
    },
}

pub(crate) fn validate_plugin_entries(entries: &[PluginConfigEntry]) -> DiagnosticResult {
    for (index, entry) in entries.iter().enumerate() {
        validate_plugin_startup(entry, index)?;
    }
    Ok(())
}

pub(crate) fn validate_plugin_entries_strict<F>(
    entries: &[PluginConfigEntry],
    raw_toml: Option<&str>,
    mut schema_for_plugin: F,
) -> Vec<ConfigDiagnostic>
where
    F: FnMut(&str) -> PluginSchemaAvailability,
{
    let mut diagnostics = plugin_misplaced_key_diagnostics(raw_toml);

    for entry in entries {
        let has_custom_settings = !entry.settings.is_empty();
        let settings_path = plugin_settings_path(&entry.name);
        match schema_for_plugin(&entry.name) {
            PluginSchemaAvailability::Available(schema) => {
                if schema.schema_version != SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION {
                    diagnostics.push(plugin_diagnostic(
                        ConfigDiagnosticCode::UnsupportedSchemaVersion,
                        ConfigDiagnosticSeverity::Error,
                        settings_path,
                        format!(
                            "plugin '{}' declares unsupported config schema_version {}; expected {}",
                            entry.name, schema.schema_version, SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION
                        ),
                    ));
                    continue;
                }
                if schema.allow_unvalidated_config {
                    if has_custom_settings {
                        diagnostics.push(plugin_diagnostic(
                            ConfigDiagnosticCode::LegacyUnvalidatedConfig,
                            ConfigDiagnosticSeverity::Warning,
                            settings_path,
                            format!(
                                "plugin '{}' allows legacy unvalidated config; unknown custom settings are accepted, but declared settings are still schema-validated",
                                entry.name
                            ),
                        ));
                    }
                    diagnostics.extend(validate_plugin_settings_against_schema(
                        entry, &schema, true,
                    ));
                    continue;
                }
                diagnostics.extend(validate_plugin_settings_against_schema(
                    entry, &schema, false,
                ));
            }
            PluginSchemaAvailability::NotInstalled => {
                if has_custom_settings {
                    diagnostics.push(plugin_diagnostic(
                        ConfigDiagnosticCode::SchemaUnavailable,
                        ConfigDiagnosticSeverity::Error,
                        settings_path,
                        format!(
                            "plugin '{}' is not installed, so custom settings cannot be validated in strict mode",
                            entry.name
                        ),
                    ));
                }
            }
            PluginSchemaAvailability::MissingSchema => {
                if has_custom_settings {
                    diagnostics.push(plugin_diagnostic(
                        ConfigDiagnosticCode::SchemaUnavailable,
                        ConfigDiagnosticSeverity::Error,
                        settings_path,
                        format!(
                            "plugin '{}' does not expose install-time config schema metadata, so custom settings cannot be validated in strict mode",
                            entry.name
                        ),
                    ));
                }
            }
            PluginSchemaAvailability::UnsupportedVersion { version } => {
                diagnostics.push(plugin_diagnostic(
                    ConfigDiagnosticCode::UnsupportedSchemaVersion,
                    ConfigDiagnosticSeverity::Error,
                    settings_path,
                    format!(
                        "plugin '{}' declares unsupported config schema_version {}; expected {}",
                        entry.name, version, SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION
                    ),
                ));
            }
        }
    }

    diagnostics
}

fn validate_plugin_startup(entry: &PluginConfigEntry, index: usize) -> DiagnosticResult {
    if matches!(entry.startup.connect_timeout_secs, Some(0)) {
        return Err(validation_diagnostic(
            &format!("plugin[{index}].startup.connect_timeout_secs"),
            format!("plugin[{index}].startup.connect_timeout_secs must be at least 1 when set"),
        ));
    }
    if matches!(entry.startup.init_timeout_secs, Some(0)) {
        return Err(validation_diagnostic(
            &format!("plugin[{index}].startup.init_timeout_secs"),
            format!("plugin[{index}].startup.init_timeout_secs must be at least 1 when set"),
        ));
    }
    Ok(())
}

fn validate_plugin_settings_against_schema(
    entry: &PluginConfigEntry,
    schema: &PluginConfigSchema,
    allow_unknown_settings: bool,
) -> Vec<ConfigDiagnostic> {
    let mut diagnostics = Vec::new();
    let schema_by_key = schema
        .settings
        .iter()
        .map(|setting| (setting.key.as_str(), setting))
        .collect::<BTreeMap<_, _>>();

    for key in entry.settings.keys() {
        if !allow_unknown_settings && !schema_by_key.contains_key(key.as_str()) {
            diagnostics.push(plugin_diagnostic(
                ConfigDiagnosticCode::UnknownField,
                ConfigDiagnosticSeverity::Error,
                plugin_setting_path(&entry.name, [key.as_str()]),
                format!(
                    "plugin '{}' does not declare custom setting '{}' in [[plugin]].settings",
                    entry.name, key
                ),
            ));
        }
    }

    for setting in &schema.settings {
        let Some(value) = entry.settings.get(&setting.key) else {
            if setting.required {
                diagnostics.push(plugin_diagnostic(
                    ConfigDiagnosticCode::MissingRequiredValue,
                    ConfigDiagnosticSeverity::Error,
                    plugin_setting_path(&entry.name, [setting.key.as_str()]),
                    format!(
                        "plugin '{}' requires [[plugin]].settings.{} to be set",
                        entry.name, setting.key
                    ),
                ));
            }
            continue;
        };

        validate_plugin_value(
            &entry.name,
            &[setting.key.as_str()],
            value,
            &setting.value_schema,
            &setting.constraints,
            &entry.settings,
            &mut diagnostics,
        );
    }

    diagnostics
}

fn validate_plugin_value(
    plugin_name: &str,
    path_segments: &[&str],
    value: &Value,
    schema: &PluginValueSchema,
    constraints: &[PluginSettingConstraint],
    root_settings: &BTreeMap<String, Value>,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if let Err(message) = validate_plugin_value_kind(value, schema) {
        diagnostics.push(plugin_diagnostic(
            ConfigDiagnosticCode::InvalidValue,
            ConfigDiagnosticSeverity::Error,
            plugin_setting_path(plugin_name, path_segments.iter().copied()),
            message,
        ));
        return;
    }

    for constraint in constraints {
        if let Err(message) = validate_plugin_constraint(value, constraint, root_settings) {
            diagnostics.push(plugin_diagnostic(
                ConfigDiagnosticCode::InvalidValue,
                ConfigDiagnosticSeverity::Error,
                plugin_setting_path(plugin_name, path_segments.iter().copied()),
                message,
            ));
        }
    }

    match (&schema.kind, value) {
        (PluginValueKind::Array, Value::Array(items)) => {
            if let Some(item_schema) = schema.items.as_deref() {
                for (index, item) in items.iter().enumerate() {
                    let index_segment = index.to_string();
                    let mut nested = path_segments.to_vec();
                    nested.push(index_segment.as_str());
                    validate_plugin_value(
                        plugin_name,
                        &nested,
                        item,
                        item_schema,
                        &[],
                        root_settings,
                        diagnostics,
                    );
                }
            }
        }
        (PluginValueKind::Object, Value::Table(table)) => {
            let object_schema = schema
                .object_properties
                .iter()
                .map(|property| (property.key.as_str(), property))
                .collect::<BTreeMap<_, _>>();

            for key in table.keys() {
                if !schema.allow_additional_properties && !object_schema.contains_key(key.as_str())
                {
                    diagnostics.push(plugin_diagnostic(
                        ConfigDiagnosticCode::UnknownField,
                        ConfigDiagnosticSeverity::Error,
                        plugin_setting_path(
                            plugin_name,
                            path_segments.iter().copied().chain([key.as_str()]),
                        ),
                        format!(
                            "plugin '{}' does not allow object property '{}' here",
                            plugin_name, key
                        ),
                    ));
                }
            }

            for property in &schema.object_properties {
                let Some(property_value) = table.get(&property.key) else {
                    if property.required {
                        diagnostics.push(plugin_diagnostic(
                            ConfigDiagnosticCode::MissingRequiredValue,
                            ConfigDiagnosticSeverity::Error,
                            plugin_setting_path(
                                plugin_name,
                                path_segments.iter().copied().chain([property.key.as_str()]),
                            ),
                            format!(
                                "plugin '{}' requires object property '{}' here",
                                plugin_name, property.key
                            ),
                        ));
                    }
                    continue;
                };

                let mut nested = path_segments.to_vec();
                nested.push(property.key.as_str());
                validate_plugin_value(
                    plugin_name,
                    &nested,
                    property_value,
                    &property.value_schema,
                    &[],
                    root_settings,
                    diagnostics,
                );
            }
        }
        _ => {}
    }
}

fn validate_plugin_value_kind(value: &Value, schema: &PluginValueSchema) -> Result<(), String> {
    match schema.kind {
        PluginValueKind::Boolean if value.is_bool() => Ok(()),
        PluginValueKind::Integer if value.as_integer().is_some() => Ok(()),
        PluginValueKind::Float if numeric_value(value).is_some() => Ok(()),
        PluginValueKind::String | PluginValueKind::Path if value.as_str().is_some() => Ok(()),
        PluginValueKind::Url => {
            let Some(raw) = value.as_str() else {
                return Err("expected URL string".into());
            };
            if raw.contains("://") {
                Ok(())
            } else {
                Err(format!("expected valid URL, got {raw:?}"))
            }
        }
        PluginValueKind::Enum => {
            let Some(raw) = value.as_str() else {
                return Err("expected enum string".into());
            };
            if schema.enum_values.iter().any(|candidate| candidate == raw) {
                Ok(())
            } else {
                Err(format!(
                    "expected one of: {}",
                    schema.enum_values.join(", ")
                ))
            }
        }
        PluginValueKind::Array if value.as_array().is_some() => Ok(()),
        PluginValueKind::Object if value.as_table().is_some() => Ok(()),
        PluginValueKind::Boolean => Err("expected boolean".into()),
        PluginValueKind::Integer => Err("expected integer".into()),
        PluginValueKind::Float => Err("expected number".into()),
        PluginValueKind::String => Err("expected string".into()),
        PluginValueKind::Path => Err("expected path string".into()),
        PluginValueKind::Array => Err("expected array".into()),
        PluginValueKind::Object => Err("expected object/table".into()),
    }
}

fn validate_plugin_constraint(
    value: &Value,
    constraint: &PluginSettingConstraint,
    root_settings: &BTreeMap<String, Value>,
) -> Result<(), String> {
    match constraint {
        PluginSettingConstraint::NonEmpty => {
            let valid = match value {
                Value::String(inner) => !inner.trim().is_empty(),
                Value::Array(inner) => !inner.is_empty(),
                Value::Table(inner) => !inner.is_empty(),
                _ => true,
            };
            if valid {
                Ok(())
            } else {
                Err("must not be empty".into())
            }
        }
        PluginSettingConstraint::Positive => {
            let Some(number) = numeric_value(value) else {
                return Err("must be numeric to apply positive constraint".into());
            };
            if number > 0.0 {
                Ok(())
            } else {
                Err("must be greater than 0".into())
            }
        }
        PluginSettingConstraint::Range { min, max } => {
            let Some(number) = numeric_value(value) else {
                return Err("must be numeric to apply range constraint".into());
            };
            if let Some(min) = parse_optional_constraint_number("min", min.as_deref())?
                && number < min
            {
                return Err(format!("must be at least {}", render_number(min)));
            }
            if let Some(max) = parse_optional_constraint_number("max", max.as_deref())?
                && number > max
            {
                return Err(format!("must be at most {}", render_number(max)));
            }
            Ok(())
        }
        PluginSettingConstraint::AllowedValues { values } => {
            let Some(raw) = value.as_str() else {
                return Err("must be string-like to apply allowed-values constraint".into());
            };
            if values.iter().any(|candidate| candidate == raw) {
                Ok(())
            } else {
                Err(format!("expected one of: {}", values.join(", ")))
            }
        }
        PluginSettingConstraint::Requires { key } => {
            if root_settings.contains_key(key) {
                Ok(())
            } else {
                Err(format!("requires [[plugin]].settings.{key} to also be set"))
            }
        }
    }
}

fn plugin_misplaced_key_diagnostics(raw_toml: Option<&str>) -> Vec<ConfigDiagnostic> {
    let Some(raw_toml) = raw_toml else {
        return Vec::new();
    };
    let Ok(parsed) = toml::from_str::<Value>(raw_toml) else {
        return Vec::new();
    };
    let Some(plugin_entries) = parsed.get("plugin").and_then(Value::as_array) else {
        return Vec::new();
    };

    let allowed_top_level = BTreeSet::from([
        "name",
        "enabled",
        "web_ui_enabled",
        "command",
        "args",
        "url",
        "startup",
        "settings",
    ]);
    let allowed_startup = BTreeSet::from([
        "connect_timeout_secs",
        "init_timeout_secs",
        "optional",
        "lazy_start",
    ]);

    let mut diagnostics = Vec::new();
    for (index, item) in plugin_entries.iter().enumerate() {
        let Some(table) = item.as_table() else {
            continue;
        };
        let plugin_name = table
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("<plugin>")
            .to_string();

        for key in table.keys() {
            if allowed_top_level.contains(key.as_str()) {
                continue;
            }
            diagnostics.push(
                plugin_diagnostic(
                    ConfigDiagnosticCode::MisplacedField,
                    ConfigDiagnosticSeverity::Error,
                    plugin_setting_path(&plugin_name, [key.as_str()]),
                    format!(
                        "plugin[{index}].{key} is a custom plugin setting in a host-owned location; move it under [[plugin]].settings.{key}"
                    ),
                )
                .at_path(ConfigPath::parse_rendered(&format!("plugin[{index}].{key}")).unwrap_or_default()),
            );
        }

        if let Some(startup) = table.get("startup").and_then(Value::as_table) {
            for key in startup.keys() {
                if allowed_startup.contains(key.as_str()) {
                    continue;
                }
                diagnostics.push(
                    plugin_diagnostic(
                        ConfigDiagnosticCode::MisplacedField,
                        ConfigDiagnosticSeverity::Error,
                        plugin_setting_path(&plugin_name, [key.as_str()]),
                        format!(
                            "plugin[{index}].startup.{key} is not a host-owned startup key; plugin custom settings must live under [[plugin]].settings.{key}"
                        ),
                    )
                    .at_path(
                        ConfigPath::parse_rendered(&format!("plugin[{index}].startup.{key}"))
                            .unwrap_or_default(),
                    ),
                );
            }
        }
    }

    diagnostics
}

fn plugin_diagnostic(
    code: ConfigDiagnosticCode,
    severity: ConfigDiagnosticSeverity,
    path: ConfigPath,
    message: impl Into<String>,
) -> ConfigDiagnostic {
    ConfigDiagnostic::new(code, severity, ConfigDiagnosticSource::Plugin, message)
        .with_schema_source(ConfigDiagnosticSchemaSource::Plugin)
        .at_path(path.clone())
        .with_canonical_path(path)
}

fn plugin_settings_path(plugin_name: &str) -> ConfigPath {
    ConfigPath::from_fields(["plugin", plugin_name, "settings"])
}

fn plugin_setting_path<'a>(
    plugin_name: &str,
    segments: impl IntoIterator<Item = &'a str>,
) -> ConfigPath {
    let mut path = plugin_settings_path(plugin_name);
    for segment in segments {
        path.push_field(segment);
    }
    path
}

fn numeric_value(value: &Value) -> Option<f64> {
    value
        .as_float()
        .or_else(|| value.as_integer().map(|integer| integer as f64))
}

fn parse_optional_constraint_number(
    bound_name: &str,
    raw: Option<&str>,
) -> Result<Option<f64>, String> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    raw.parse::<f64>()
        .map(Some)
        .map_err(|_| format!("range constraint {bound_name} bound must be numeric, got {raw:?}"))
}

fn render_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PluginWebUiPreference;

    fn schema() -> PluginConfigSchema {
        PluginConfigSchema {
            plugin_name: "blackboard".into(),
            schema_version: SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION,
            allow_unvalidated_config: false,
            settings: vec![
                PluginSettingSchema {
                    key: "retention_days".into(),
                    value_schema: PluginValueSchema {
                        kind: PluginValueKind::Integer,
                        enum_values: Vec::new(),
                        items: None,
                        object_properties: Vec::new(),
                        allow_additional_properties: false,
                    },
                    required: true,
                    default_json: Some("14".into()),
                    constraints: vec![PluginSettingConstraint::Range {
                        min: Some("1".into()),
                        max: Some("365".into()),
                    }],
                    description: None,
                    control_behavior: None,
                },
                PluginSettingSchema {
                    key: "mode".into(),
                    value_schema: PluginValueSchema {
                        kind: PluginValueKind::Enum,
                        enum_values: vec!["strict".into(), "relaxed".into()],
                        items: None,
                        object_properties: Vec::new(),
                        allow_additional_properties: false,
                    },
                    required: false,
                    default_json: Some("\"strict\"".into()),
                    constraints: Vec::new(),
                    description: None,
                    control_behavior: None,
                },
            ],
        }
    }

    #[test]
    fn strict_plugin_validation_reports_misplaced_and_unknown_keys() {
        let config: crate::MeshConfig = toml::from_str(
            r#"
[[plugin]]
name = "blackboard"
retention_days = 14

[plugin.settings]
mode = "strict"
unknown = true
"#,
        )
        .unwrap();

        let diagnostics = validate_plugin_entries_strict(
            &config.plugins,
            Some(
                r#"
[[plugin]]
name = "blackboard"
retention_days = 14

[plugin.settings]
mode = "strict"
unknown = true
"#,
            ),
            |_| PluginSchemaAvailability::Available(schema()),
        );

        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == ConfigDiagnosticCode::MisplacedField)
        );
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == ConfigDiagnosticCode::UnknownField)
        );
    }

    #[test]
    fn plugin_web_ui_enabled_is_host_owned_and_not_a_custom_setting() {
        let raw = r#"
[[plugin]]
name = "blackboard"
web_ui_enabled = false

[plugin.settings]
retention_days = 14
"#;
        let config: crate::MeshConfig = toml::from_str(raw).unwrap();

        let diagnostics = validate_plugin_entries_strict(&config.plugins, Some(raw), |_| {
            PluginSchemaAvailability::Available(schema())
        });

        assert!(
            !diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == ConfigDiagnosticCode::MisplacedField)
        );
    }

    #[test]
    fn plugin_web_ui_preference_resolves_declared_absent_and_explicit_choices() {
        let absent = crate::PluginConfigEntry {
            name: "blackboard".to_string(),
            enabled: Some(false),
            web_ui_enabled: None,
            command: None,
            args: Vec::new(),
            url: None,
            settings: BTreeMap::new(),
            startup: Default::default(),
        };
        let disabled = crate::PluginConfigEntry {
            web_ui_enabled: Some(false),
            ..absent.clone()
        };
        let enabled = crate::PluginConfigEntry {
            web_ui_enabled: Some(true),
            ..absent.clone()
        };

        assert_eq!(
            absent.web_ui_preference(true),
            PluginWebUiPreference::Enabled
        );
        assert_eq!(
            disabled.web_ui_preference(true),
            PluginWebUiPreference::Disabled
        );
        assert_eq!(
            enabled.web_ui_preference(true),
            PluginWebUiPreference::Enabled
        );
        assert_eq!(
            disabled.web_ui_preference(false),
            PluginWebUiPreference::None
        );
    }

    #[test]
    fn strict_plugin_validation_rejects_required_settings_when_settings_table_is_absent() {
        let raw = r#"
[[plugin]]
name = "blackboard"
"#;
        let config: crate::MeshConfig = toml::from_str(raw).unwrap();

        let diagnostics = validate_plugin_entries_strict(&config.plugins, Some(raw), |_| {
            PluginSchemaAvailability::Available(schema())
        });

        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code == ConfigDiagnosticCode::MissingRequiredValue
                && diagnostic
                    .canonical_path
                    .as_ref()
                    .map(ConfigPath::render)
                    .as_deref()
                    == Some("plugin.blackboard.settings.retention_days")
        }));
    }

    #[test]
    fn strict_plugin_validation_rejects_malformed_range_bound() {
        let raw = r#"
[[plugin]]
name = "blackboard"

[plugin.settings]
retention_days = 14
"#;
        let config: crate::MeshConfig = toml::from_str(raw).unwrap();
        let mut malformed_schema = schema();
        malformed_schema.settings[0].constraints = vec![PluginSettingConstraint::Range {
            min: Some("low".into()),
            max: Some("365".into()),
        }];

        let diagnostics = validate_plugin_entries_strict(&config.plugins, Some(raw), |_| {
            PluginSchemaAvailability::Available(malformed_schema.clone())
        });

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].code, ConfigDiagnosticCode::InvalidValue);
        assert_eq!(diagnostics[0].severity, ConfigDiagnosticSeverity::Error);
        assert_eq!(
            diagnostics[0]
                .canonical_path
                .as_ref()
                .map(ConfigPath::render),
            Some("plugin.blackboard.settings.retention_days".to_string())
        );
        assert!(
            diagnostics[0]
                .message
                .contains("range constraint min bound must be numeric")
        );
    }

    #[test]
    fn strict_plugin_validation_rejects_missing_install_time_schema_metadata() {
        let raw = r#"
[[plugin]]
name = "blackboard"

[plugin.settings]
retention_days = 14
"#;
        let config: crate::MeshConfig = toml::from_str(raw).unwrap();

        let diagnostics = validate_plugin_entries_strict(&config.plugins, Some(raw), |_| {
            PluginSchemaAvailability::MissingSchema
        });

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].code, ConfigDiagnosticCode::SchemaUnavailable);
        assert_eq!(diagnostics[0].severity, ConfigDiagnosticSeverity::Error);
        assert_eq!(
            diagnostics[0]
                .canonical_path
                .as_ref()
                .map(ConfigPath::render),
            Some("plugin.blackboard.settings".to_string())
        );
    }

    #[test]
    fn strict_plugin_validation_rejects_uninstalled_plugins_with_custom_settings() {
        let raw = r#"
[[plugin]]
name = "blackboard"

[plugin.settings]
retention_days = 14
"#;
        let config: crate::MeshConfig = toml::from_str(raw).unwrap();

        let diagnostics = validate_plugin_entries_strict(&config.plugins, Some(raw), |_| {
            PluginSchemaAvailability::NotInstalled
        });

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].code, ConfigDiagnosticCode::SchemaUnavailable);
        assert!(
            diagnostics[0]
                .message
                .contains("custom settings cannot be validated in strict mode")
        );
    }

    #[test]
    fn strict_plugin_validation_only_allows_unbounded_settings_via_legacy_escape_hatch() {
        let raw = r#"
[[plugin]]
name = "blackboard"

[plugin.settings]
retention_days = 14
unknown = true
"#;
        let config: crate::MeshConfig = toml::from_str(raw).unwrap();
        let mut legacy_schema = schema();
        legacy_schema.allow_unvalidated_config = true;

        let diagnostics = validate_plugin_entries_strict(&config.plugins, Some(raw), |_| {
            PluginSchemaAvailability::Available(legacy_schema.clone())
        });

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].code,
            ConfigDiagnosticCode::LegacyUnvalidatedConfig
        );
        assert_eq!(diagnostics[0].severity, ConfigDiagnosticSeverity::Warning);
        assert!(
            diagnostics[0]
                .message
                .contains("allows legacy unvalidated config")
        );
    }

    #[test]
    fn strict_plugin_validation_legacy_escape_hatch_still_validates_known_settings() {
        let raw = r#"
[[plugin]]
name = "blackboard"

[plugin.settings]
retention_days = 0
mode = "mystery"
unknown = true
"#;
        let config: crate::MeshConfig = toml::from_str(raw).unwrap();
        let mut legacy_schema = schema();
        legacy_schema.allow_unvalidated_config = true;

        let diagnostics = validate_plugin_entries_strict(&config.plugins, Some(raw), |_| {
            PluginSchemaAvailability::Available(legacy_schema.clone())
        });

        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code == ConfigDiagnosticCode::LegacyUnvalidatedConfig
                && diagnostic.severity == ConfigDiagnosticSeverity::Warning
        }));
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code == ConfigDiagnosticCode::InvalidValue
                && diagnostic
                    .canonical_path
                    .as_ref()
                    .map(ConfigPath::render)
                    .as_deref()
                    == Some("plugin.blackboard.settings.retention_days")
        }));
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code == ConfigDiagnosticCode::InvalidValue
                && diagnostic
                    .canonical_path
                    .as_ref()
                    .map(ConfigPath::render)
                    .as_deref()
                    == Some("plugin.blackboard.settings.mode")
        }));
        assert!(!diagnostics.iter().any(|diagnostic| {
            diagnostic.code == ConfigDiagnosticCode::UnknownField
                && diagnostic
                    .canonical_path
                    .as_ref()
                    .map(ConfigPath::render)
                    .as_deref()
                    == Some("plugin.blackboard.settings.unknown")
        }));
    }

    #[test]
    fn strict_plugin_validation_rejects_unsupported_schema_version_boundaries() {
        let raw = r#"
[[plugin]]
name = "blackboard"

[plugin.settings]
retention_days = 14
"#;
        let config: crate::MeshConfig = toml::from_str(raw).unwrap();

        let mut mismatched_schema = schema();
        mismatched_schema.schema_version = SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION + 1;
        let available_diagnostics =
            validate_plugin_entries_strict(&config.plugins, Some(raw), |_| {
                PluginSchemaAvailability::Available(mismatched_schema.clone())
            });
        let unavailable_diagnostics =
            validate_plugin_entries_strict(&config.plugins, Some(raw), |_| {
                PluginSchemaAvailability::UnsupportedVersion {
                    version: SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION + 2,
                }
            });

        assert_eq!(
            available_diagnostics[0].code,
            ConfigDiagnosticCode::UnsupportedSchemaVersion
        );
        assert!(
            available_diagnostics[0]
                .message
                .contains("unsupported config schema_version")
        );
        assert_eq!(
            unavailable_diagnostics[0].code,
            ConfigDiagnosticCode::UnsupportedSchemaVersion
        );
        assert!(
            unavailable_diagnostics[0]
                .message
                .contains(&format!("{}", SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION + 2))
        );
    }
}
