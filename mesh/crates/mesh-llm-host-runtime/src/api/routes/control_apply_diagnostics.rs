use serde::Serialize;

#[derive(Debug, Serialize)]
pub(super) struct LocalControlApplyDiagnosticPayload {
    code: String,
    severity: String,
    source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    schema_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    canonical_path: Option<String>,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    help: Option<String>,
}

pub(super) fn local_control_apply_diagnostic_payload(
    diagnostic: &mesh_client::proto::node::ConfigDiagnostic,
) -> LocalControlApplyDiagnosticPayload {
    let diagnostic = crate::protocol::convert::proto_config_diagnostic_to_local(diagnostic);
    LocalControlApplyDiagnosticPayload {
        code: control_diagnostic_code_label(diagnostic.code),
        severity: control_diagnostic_severity_label(diagnostic.severity),
        source: control_diagnostic_source_label(diagnostic.source),
        schema_source: diagnostic
            .schema_source
            .map(control_diagnostic_schema_source_label),
        path: diagnostic.path.map(|path| path.render()),
        canonical_path: diagnostic.canonical_path.map(|path| path.render()),
        message: diagnostic.message,
        help: diagnostic.help,
    }
}

pub(super) fn local_control_apply_diagnostic_payload_from_local(
    diagnostic: &mesh_llm_config::ConfigDiagnostic,
) -> LocalControlApplyDiagnosticPayload {
    LocalControlApplyDiagnosticPayload {
        code: control_diagnostic_code_label(diagnostic.code),
        severity: control_diagnostic_severity_label(diagnostic.severity),
        source: control_diagnostic_source_label(diagnostic.source),
        schema_source: diagnostic
            .schema_source
            .map(control_diagnostic_schema_source_label),
        path: diagnostic.path.clone().map(|path| path.render()),
        canonical_path: diagnostic.canonical_path.clone().map(|path| path.render()),
        message: diagnostic.message.clone(),
        help: diagnostic.help.clone(),
    }
}

fn control_diagnostic_code_label(value: mesh_llm_config::ConfigDiagnosticCode) -> String {
    match value {
        mesh_llm_config::ConfigDiagnosticCode::InvalidValue => "invalid_value",
        mesh_llm_config::ConfigDiagnosticCode::MissingRequiredValue => "missing_required_value",
        mesh_llm_config::ConfigDiagnosticCode::UnsupportedField => "unsupported_field",
        mesh_llm_config::ConfigDiagnosticCode::RejectedField => "rejected_field",
        mesh_llm_config::ConfigDiagnosticCode::AliasApplied => "alias_applied",
        mesh_llm_config::ConfigDiagnosticCode::MisplacedField => "misplaced_field",
        mesh_llm_config::ConfigDiagnosticCode::UnknownField => "unknown_field",
        mesh_llm_config::ConfigDiagnosticCode::SchemaUnavailable => "schema_unavailable",
        mesh_llm_config::ConfigDiagnosticCode::LegacyUnvalidatedConfig => {
            "legacy_unvalidated_config"
        }
        mesh_llm_config::ConfigDiagnosticCode::UnsupportedSchemaVersion => {
            "unsupported_schema_version"
        }
    }
    .to_string()
}

fn control_diagnostic_severity_label(value: mesh_llm_config::ConfigDiagnosticSeverity) -> String {
    match value {
        mesh_llm_config::ConfigDiagnosticSeverity::Error => "error",
        mesh_llm_config::ConfigDiagnosticSeverity::Warning => "warning",
        mesh_llm_config::ConfigDiagnosticSeverity::Info => "info",
    }
    .to_string()
}

fn control_diagnostic_source_label(value: mesh_llm_config::ConfigDiagnosticSource) -> String {
    match value {
        mesh_llm_config::ConfigDiagnosticSource::Validation => "validation",
        mesh_llm_config::ConfigDiagnosticSource::Schema => "schema",
        mesh_llm_config::ConfigDiagnosticSource::Plugin => "plugin",
        mesh_llm_config::ConfigDiagnosticSource::Compatibility => "compatibility",
    }
    .to_string()
}

fn control_diagnostic_schema_source_label(
    value: mesh_llm_config::ConfigDiagnosticSchemaSource,
) -> String {
    match value {
        mesh_llm_config::ConfigDiagnosticSchemaSource::BuiltIn => "built_in",
        mesh_llm_config::ConfigDiagnosticSchemaSource::Engine => "engine",
        mesh_llm_config::ConfigDiagnosticSchemaSource::Plugin => "plugin",
    }
    .to_string()
}
