pub(crate) use crate::diagnostic::DiagnosticResult;
pub use crate::diagnostic::{
    ConfigDiagnostic, ConfigDiagnosticCode, ConfigDiagnosticSchemaSource, ConfigDiagnosticSeverity,
    ConfigDiagnosticSource, alias_diagnostic, invalid_value_diagnostic,
    legacy_validation_error_text, rejected_field_diagnostic, unsupported_field_diagnostic,
};
use crate::model_validation::{
    collect_legacy_draft_model_path_warnings, validate_duplicate_model_entries,
    validate_model_defaults, validate_model_entry,
};
use crate::plugin_validation::{
    PluginSchemaAvailability, validate_plugin_entries, validate_plugin_entries_strict,
};
pub(crate) use crate::validation_support::validation_diagnostic;
pub use crate::validation_support::{
    built_in_support_diagnostic, canonical_builtin_diagnostic_path,
};
use crate::validation_support::{
    parse_node_version, validate_optional_http_url, validate_release_signer_key_shape,
    version_precedence_cmp,
};
use crate::*;
use anyhow::Result;

pub fn validate_config_diagnostics(config: &MeshConfig) -> Vec<ConfigDiagnostic> {
    let mut diagnostics = Vec::new();

    if let Some(version) = config.version
        && version != 1
    {
        diagnostics.push(validation_diagnostic(
            "version",
            format!("unsupported config version {version}; expected version = 1"),
        ));
    }
    if let Some(bind) = config.owner_control.bind
        && bind.port() == 0
        && !bind.ip().is_loopback()
    {
        diagnostics.push(validation_diagnostic(
            "owner_control.bind",
            "owner_control.bind must use a concrete port when binding a non-loopback address",
        ));
    }
    if let Some(advertise_addr) = config.owner_control.advertise_addr {
        match config.owner_control.bind {
            Some(bind) if bind.port() == 0 => {
                diagnostics.push(validation_diagnostic(
                    "owner_control.bind",
                    "owner_control.bind must use a concrete port when owner_control.advertise_addr is set",
                ));
            }
            Some(bind) if bind.port() != advertise_addr.port() => {
                diagnostics.push(validation_diagnostic(
                    "owner_control.advertise_addr",
                    "owner_control.advertise_addr must use the same port as owner_control.bind",
                ));
            }
            Some(_) => {}
            None => {
                diagnostics.push(validation_diagnostic(
                    "owner_control.advertise_addr",
                    "owner_control.advertise_addr requires owner_control.bind so the advertised port is actually listening",
                ));
            }
        }
        if advertise_addr.port() == 0 {
            diagnostics.push(validation_diagnostic(
                "owner_control.advertise_addr",
                "owner_control.advertise_addr must use a concrete port",
            ));
        }
        if advertise_addr.ip().is_unspecified() {
            diagnostics.push(validation_diagnostic(
                "owner_control.advertise_addr",
                "owner_control.advertise_addr must not use an unspecified IP address",
            ));
        }
    }
    if let Some(parallel) = config.gpu.parallel
        && parallel < 1
    {
        diagnostics.push(validation_diagnostic(
            "gpu.parallel",
            format!("gpu.parallel must be at least 1, got {parallel}"),
        ));
    }
    if let Err(diagnostic) = validate_mesh_requirements_config(&config.mesh_requirements) {
        diagnostics.push(diagnostic);
    }
    if let Err(diagnostic) = validate_telemetry_config(&config.telemetry) {
        diagnostics.push(diagnostic);
    }
    if let Err(diagnostic) = validate_runtime_config(&config.runtime) {
        diagnostics.push(diagnostic);
    }
    if let Err(diagnostic) = validate_plugin_entries(&config.plugins) {
        diagnostics.push(diagnostic);
    }
    let defaults_hardware = config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.hardware.as_ref());
    if let Some(defaults) = &config.defaults
        && let Err(diagnostic) =
            validate_model_defaults(defaults, "defaults", config.gpu.assignment)
    {
        diagnostics.push(diagnostic);
    }
    for (index, model) in config.models.iter().enumerate() {
        if model.model.trim().is_empty() {
            diagnostics.push(validation_diagnostic(
                &format!("models[{index}].model"),
                format!("models[{index}].model must not be empty"),
            ));
        }
        if let Err(diagnostic) = validate_model_entry(
            model,
            &format!("models[{index}]"),
            config.gpu.assignment,
            defaults_hardware,
        ) {
            diagnostics.push(diagnostic);
        }
    }

    collect_legacy_draft_model_path_warnings(config, &mut diagnostics);

    validate_duplicate_model_entries(&config.models, &mut diagnostics);

    diagnostics
}

fn validate_runtime_config(config: &RuntimeConfig) -> DiagnosticResult {
    let mesh_version = config.native_runtime.mesh_version.as_deref();
    let skippy_abi = config.native_runtime.skippy_abi.as_deref();
    let selection = config.native_runtime.selection.as_deref();
    if mesh_version.is_none() && (skippy_abi.is_some() || selection.is_some()) {
        return Err(validation_diagnostic(
            "runtime.native_runtime",
            "runtime.native_runtime override must set mesh_version when skippy_abi or selection is set",
        ));
    }
    if matches!(mesh_version, Some(value) if value.trim().is_empty()) {
        return Err(validation_diagnostic(
            "runtime.native_runtime.mesh_version",
            "runtime.native_runtime.mesh_version must not be empty",
        ));
    }
    if matches!(skippy_abi, Some(value) if value.trim().is_empty()) {
        return Err(validation_diagnostic(
            "runtime.native_runtime.skippy_abi",
            "runtime.native_runtime.skippy_abi must not be empty",
        ));
    }
    if matches!(selection, Some(value) if value.trim().is_empty()) {
        return Err(validation_diagnostic(
            "runtime.native_runtime.selection",
            "runtime.native_runtime.selection must not be empty",
        ));
    }
    Ok(())
}

pub fn validate_config_diagnostics_with_plugin_schemas<F>(
    config: &MeshConfig,
    raw_toml: Option<&str>,
    schema_for_plugin: F,
) -> Vec<ConfigDiagnostic>
where
    F: FnMut(&str) -> PluginSchemaAvailability,
{
    let mut diagnostics = validate_config_diagnostics(config);
    diagnostics.extend(validate_plugin_entries_strict(
        &config.plugins,
        raw_toml,
        schema_for_plugin,
    ));
    diagnostics
}

pub fn validate_config(config: &MeshConfig) -> Result<()> {
    let diagnostics = validate_config_diagnostics(config);
    let has_errors = diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error);
    if has_errors {
        Err(anyhow::anyhow!(legacy_validation_error_text(&diagnostics)))
    } else {
        Ok(())
    }
}

pub fn validate_config_with_plugin_schemas<F>(
    config: &MeshConfig,
    raw_toml: Option<&str>,
    schema_for_plugin: F,
) -> Result<()>
where
    F: FnMut(&str) -> PluginSchemaAvailability,
{
    let diagnostics =
        validate_config_diagnostics_with_plugin_schemas(config, raw_toml, schema_for_plugin);
    let has_errors = diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error);
    if has_errors {
        Err(anyhow::anyhow!(legacy_validation_error_text(&diagnostics)))
    } else {
        Ok(())
    }
}

fn validate_mesh_requirements_config(config: &MeshRequirementsConfig) -> DiagnosticResult {
    let min_node_version = config
        .min_node_version
        .as_deref()
        .map(|value| parse_node_version(value, "mesh_requirements.min_node_version"))
        .transpose()?;
    let max_node_version = config
        .max_node_version
        .as_deref()
        .map(|value| parse_node_version(value, "mesh_requirements.max_node_version"))
        .transpose()?;
    if let (Some(min), Some(max)) = (&min_node_version, &max_node_version)
        && version_precedence_cmp(min, max).is_gt()
    {
        return Err(validation_diagnostic(
            "mesh_requirements.min_node_version",
            "mesh_requirements.min_node_version must be less than or equal to mesh_requirements.max_node_version",
        ));
    }

    if let (Some(min), Some(max)) = (config.min_protocol_version, config.max_protocol_version)
        && min > max
    {
        return Err(validation_diagnostic(
            "mesh_requirements.min_protocol_version",
            "mesh_requirements.min_protocol_version must be less than or equal to mesh_requirements.max_protocol_version",
        ));
    }

    for signer_key in &config.release_signer_keys {
        validate_release_signer_key_shape(signer_key, "mesh_requirements.release_signer_keys")?;
    }
    if config.require_release_attestation && config.release_signer_keys.is_empty() {
        return Err(validation_diagnostic(
            "mesh_requirements.require_release_attestation",
            "mesh_requirements.require_release_attestation is true but mesh_requirements.release_signer_keys is empty; certified-build admission is not remote runtime attestation, so trust must be anchored in at least one release signer key",
        ));
    }

    Ok(())
}

fn validate_telemetry_config(config: &TelemetryConfig) -> DiagnosticResult {
    if let Some(service_name) = &config.service_name {
        let trimmed = service_name.trim();
        if !trimmed.is_empty() {
            // Validate service name: alphanumeric, dash, underscore only
            if !trimmed
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                return Err(validation_diagnostic(
                    "telemetry.service_name",
                    "telemetry.service_name must contain only alphanumeric characters, dashes, and underscores",
                ));
            }
        }
    }
    validate_optional_http_url(config.endpoint.as_deref(), "telemetry.endpoint")?;
    validate_optional_http_url(
        config.metrics.endpoint.as_deref(),
        "telemetry.metrics.endpoint",
    )?;
    for key in config.headers.keys() {
        if key.trim().is_empty() {
            return Err(validation_diagnostic(
                "telemetry.headers",
                "telemetry.headers keys must not be empty",
            ));
        }
    }
    if let Some(export_interval_secs) = config.export_interval_secs
        && export_interval_secs < 1
    {
        return Err(validation_diagnostic(
            "telemetry.export_interval_secs",
            "telemetry.export_interval_secs must be at least 1",
        ));
    }
    if let Some(queue_size) = config.queue_size
        && queue_size < 1
    {
        return Err(validation_diagnostic(
            "telemetry.queue_size",
            "telemetry.queue_size must be at least 1",
        ));
    }
    if config.prompt_shape_metrics {
        return Err(validation_diagnostic(
            "telemetry.prompt_shape_metrics",
            "telemetry.prompt_shape_metrics is not supported yet and must remain false",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod schema_tests {
    use super::*;

    #[test]
    fn owner_control_advertise_addr_requires_matching_bind_port() {
        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
advertise_addr = "127.0.0.1:17001"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        assert!(
            legacy_validation_error_text(&diagnostics).contains(
                "owner_control.advertise_addr requires owner_control.bind so the advertised port is actually listening"
            )
        );

        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
bind = "127.0.0.1:17002"
advertise_addr = "127.0.0.1:17001"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        assert!(
            legacy_validation_error_text(&diagnostics).contains(
                "owner_control.advertise_addr must use the same port as owner_control.bind"
            )
        );

        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
bind = "127.0.0.1:0"
advertise_addr = "127.0.0.1:17001"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        assert!(legacy_validation_error_text(&diagnostics).contains(
            "owner_control.bind must use a concrete port when owner_control.advertise_addr is set"
        ));

        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
bind = "127.0.0.1:17001"
advertise_addr = "127.0.0.1:17001"
"#,
        )
        .expect("config should parse before validation");

        validate_config(&config).expect("matching bind and advertise ports should validate");
    }

    #[test]
    fn structured_diagnostics_report_canonical_path_for_alias_backed_invalid_input() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-4B-Q4_K_M"
gpu_id = "metal:0"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        let diagnostic = diagnostics
            .iter()
            .find(|diagnostic| {
                diagnostic.canonical_path.as_ref().map(ConfigPath::render)
                    == Some("models.<model-ref>.hardware.device".to_string())
            })
            .expect("legacy gpu_id path should yield a canonical device diagnostic");

        assert_eq!(diagnostic.code, ConfigDiagnosticCode::InvalidValue);
        assert_eq!(diagnostic.severity, ConfigDiagnosticSeverity::Error);
        assert_eq!(
            diagnostic.schema_source,
            Some(ConfigDiagnosticSchemaSource::BuiltIn)
        );
        assert_eq!(
            diagnostic.path.as_ref().map(ConfigPath::render),
            Some("models[0].hardware.device".to_string())
        );
        assert_eq!(
            diagnostic.canonical_path.as_ref().map(ConfigPath::render),
            Some("models.<model-ref>.hardware.device".to_string())
        );
        assert_eq!(
            diagnostic.message,
            "models[0].hardware.device must not be set when gpu.assignment = \"auto\""
        );
    }

    #[test]
    fn legacy_validation_errors_derive_compatible_string_messages() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[[plugin]]
name = "metrics"
command = "mesh-llm-plugin-metrics"

[plugin.startup]
connect_timeout_secs = 0
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            legacy_validation_error_text(&diagnostics),
            "plugin[0].startup.connect_timeout_secs must be at least 1 when set"
        );

        let err =
            validate_config(&config).expect_err("legacy validation surface should still fail");
        assert_eq!(
            err.to_string(),
            "plugin[0].startup.connect_timeout_secs must be at least 1 when set"
        );
    }
}
