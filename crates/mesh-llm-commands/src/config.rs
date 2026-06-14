use anyhow::{Context, Result, bail};
use mesh_llm_cli::Cli;
use mesh_llm_config::{
    ConfigDiagnostic, ConfigDiagnosticCode, ConfigDiagnosticSchemaSource, ConfigDiagnosticSeverity,
    ConfigDiagnosticSource, ConfigPath, MeshConfig, PluginConfigSchema, PluginObjectPropertySchema,
    PluginSchemaAvailability, PluginSettingConstraint, PluginSettingSchema, PluginValueKind,
    PluginValueSchema, SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION, config_path,
    validate_config_diagnostics_with_plugin_schemas,
};
use mesh_llm_plugin_manager::{
    InstalledPluginConfigSchema, InstalledPluginConstraint, InstalledPluginMetadata,
    InstalledPluginObjectProperty, InstalledPluginValueKind, InstalledPluginValueSchema,
    PluginStore, default_store_root,
};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
struct ConfigFileValidation {
    path: PathBuf,
    diagnostics: Vec<ConfigDiagnostic>,
}

pub fn run_config_validate(
    cli: &Cli,
    config_path_override: Option<&Path>,
    json: bool,
) -> Result<()> {
    let selected_path = config_path_override.or(cli.config.as_deref());
    let resolved_path = config_path(selected_path).ok();

    match validate_config_file(selected_path) {
        Ok(validation) => handle_validation_result(validation.path, validation.diagnostics, json),
        Err(err) => {
            print_validation_load_error(resolved_path.as_deref(), &err, json)?;
            Err(err).context("config validation failed")
        }
    }
}

fn validate_config_file(override_path: Option<&Path>) -> Result<ConfigFileValidation> {
    let path = config_path(override_path)?;
    if !path.exists() {
        bail!(
            "Failed to read config file {}: file does not exist",
            path.display()
        );
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read config {}", path.display()))?;
    let config: MeshConfig =
        toml::from_str(&raw).with_context(|| format!("Invalid config {}", path.display()))?;
    let diagnostics =
        validate_config_diagnostics_with_plugin_schemas(&config, Some(&raw), plugin_schema);
    Ok(ConfigFileValidation { path, diagnostics })
}

fn plugin_schema(plugin_name: &str) -> PluginSchemaAvailability {
    let Ok(root) = default_store_root() else {
        return PluginSchemaAvailability::NotInstalled;
    };
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

    if schema.schema_version != SUPPORTED_PLUGIN_CONFIG_SCHEMA_VERSION {
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

fn handle_validation_result(
    path: PathBuf,
    diagnostics: Vec<ConfigDiagnostic>,
    json: bool,
) -> Result<()> {
    let report = ConfigValidateReport::from_diagnostics(path, diagnostics);
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_human_report(&report);
    }

    if report.ok {
        Ok(())
    } else {
        bail!("config validation failed")
    }
}

fn print_validation_load_error(path: Option<&Path>, err: &anyhow::Error, json: bool) -> Result<()> {
    let report = ConfigValidateReport::from_error(path.map(Path::to_path_buf), err.to_string());
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    let path = report.path.as_deref().unwrap_or("<unresolved>");
    println!("Config invalid: {path}");
    println!("  error: {err}");
    Ok(())
}

fn print_human_report(report: &ConfigValidateReport) {
    let path = report.path.as_deref().unwrap_or("<unresolved>");
    if report.ok {
        println!("Config valid: {path}");
    } else {
        println!("Config invalid: {path}");
    }

    for diagnostic in &report.diagnostics {
        print_human_diagnostic(diagnostic);
    }
}

fn print_human_diagnostic(diagnostic: &ConfigDiagnosticPayload) {
    let path = diagnostic
        .path
        .as_deref()
        .map(|path| format!(" at {path}"))
        .unwrap_or_default();
    println!(
        "  {} {:?}{}: {}",
        severity_label(diagnostic.severity),
        diagnostic.code,
        path,
        diagnostic.message
    );
    if let Some(help) = diagnostic.help.as_deref() {
        println!("    help: {help}");
    }
}

const fn severity_label(severity: ConfigDiagnosticSeverity) -> &'static str {
    match severity {
        ConfigDiagnosticSeverity::Error => "error",
        ConfigDiagnosticSeverity::Warning => "warning",
        ConfigDiagnosticSeverity::Info => "info",
    }
}

#[derive(Clone, Debug, Serialize)]
struct ConfigValidateReport {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    diagnostics: Vec<ConfigDiagnosticPayload>,
}

impl ConfigValidateReport {
    fn from_diagnostics(path: PathBuf, diagnostics: Vec<ConfigDiagnostic>) -> Self {
        let ok = !diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error);
        Self {
            ok,
            path: Some(path.display().to_string()),
            error: None,
            diagnostics: diagnostics
                .iter()
                .map(ConfigDiagnosticPayload::from)
                .collect(),
        }
    }

    fn from_error(path: Option<PathBuf>, error: String) -> Self {
        Self {
            ok: false,
            path: path.map(|path| path.display().to_string()),
            error: Some(error),
            diagnostics: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct ConfigDiagnosticPayload {
    code: ConfigDiagnosticCode,
    severity: ConfigDiagnosticSeverity,
    source: ConfigDiagnosticSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    schema_source: Option<ConfigDiagnosticSchemaSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    canonical_path: Option<String>,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    help: Option<String>,
}

impl From<&ConfigDiagnostic> for ConfigDiagnosticPayload {
    fn from(diagnostic: &ConfigDiagnostic) -> Self {
        Self {
            code: diagnostic.code,
            severity: diagnostic.severity,
            source: diagnostic.source,
            schema_source: diagnostic.schema_source,
            path: diagnostic.path.as_ref().map(ConfigPath::render),
            canonical_path: diagnostic.canonical_path.as_ref().map(ConfigPath::render),
            message: diagnostic.message.clone(),
            help: diagnostic.help.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validate_report_keeps_warning_only_diagnostics_successful() {
        let diagnostic = ConfigDiagnostic::warning(
            ConfigDiagnosticCode::LegacyUnvalidatedConfig,
            ConfigDiagnosticSource::Plugin,
            "plugin accepts unvalidated settings",
        )
        .at_path(plugin_settings_path("flash-moe"));

        let report =
            ConfigValidateReport::from_diagnostics(PathBuf::from("config.toml"), vec![diagnostic]);

        assert!(report.ok);
        assert_eq!(
            report.diagnostics[0].path.as_deref(),
            Some("plugin[\"flash-moe\"].settings")
        );
    }

    #[test]
    fn config_validate_report_marks_error_diagnostics_invalid() {
        let diagnostic = ConfigDiagnostic::error(
            ConfigDiagnosticCode::MissingRequiredValue,
            ConfigDiagnosticSource::Schema,
            "required plugin setting is missing",
        )
        .at_path(plugin_settings_path("flash-moe"));

        let report =
            ConfigValidateReport::from_diagnostics(PathBuf::from("config.toml"), vec![diagnostic]);

        assert!(!report.ok);
    }

    #[test]
    fn config_validate_error_report_serializes_stable_json_shape() {
        let report = ConfigValidateReport::from_error(
            Some(PathBuf::from("/tmp/config.toml")),
            "failed to parse config TOML".to_string(),
        );
        let json = serde_json::to_value(report).unwrap();

        assert_eq!(json["ok"], false);
        assert_eq!(json["path"], "/tmp/config.toml");
        assert_eq!(json["error"], "failed to parse config TOML");
        assert_eq!(json["diagnostics"].as_array().unwrap().len(), 0);
    }

    fn plugin_settings_path(plugin_name: &str) -> ConfigPath {
        let mut path = ConfigPath::field("plugin");
        path.push_key(plugin_name).push_field("settings");
        path
    }
}
