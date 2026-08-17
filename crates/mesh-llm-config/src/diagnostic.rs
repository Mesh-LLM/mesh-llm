use crate::ConfigPath;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigDiagnosticSeverity {
    #[default]
    Error,
    Warning,
    Info,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigDiagnosticSource {
    #[default]
    Validation,
    Schema,
    Plugin,
    Compatibility,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigDiagnosticSchemaSource {
    BuiltIn,
    Engine,
    Plugin,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigDiagnosticCode {
    InvalidValue,
    MissingRequiredValue,
    UnsupportedField,
    RejectedField,
    AliasApplied,
    MisplacedField,
    UnknownField,
    SchemaUnavailable,
    LegacyUnvalidatedConfig,
    UnsupportedSchemaVersion,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ConfigDiagnostic {
    pub code: ConfigDiagnosticCode,
    pub severity: ConfigDiagnosticSeverity,
    pub source: ConfigDiagnosticSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_source: Option<ConfigDiagnosticSchemaSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<ConfigPath>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical_path: Option<ConfigPath>,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub help: Option<String>,
}

impl ConfigDiagnostic {
    pub fn new(
        code: ConfigDiagnosticCode,
        severity: ConfigDiagnosticSeverity,
        source: ConfigDiagnosticSource,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            severity,
            source,
            schema_source: None,
            path: None,
            canonical_path: None,
            message: message.into(),
            help: None,
        }
    }

    pub fn error(
        code: ConfigDiagnosticCode,
        source: ConfigDiagnosticSource,
        message: impl Into<String>,
    ) -> Self {
        Self::new(code, ConfigDiagnosticSeverity::Error, source, message)
    }

    pub fn warning(
        code: ConfigDiagnosticCode,
        source: ConfigDiagnosticSource,
        message: impl Into<String>,
    ) -> Self {
        Self::new(code, ConfigDiagnosticSeverity::Warning, source, message)
    }

    pub fn at_path(mut self, path: ConfigPath) -> Self {
        self.path = Some(path);
        self
    }

    pub fn with_schema_source(mut self, schema_source: ConfigDiagnosticSchemaSource) -> Self {
        self.schema_source = Some(schema_source);
        self
    }

    pub fn with_canonical_path(mut self, canonical_path: ConfigPath) -> Self {
        self.canonical_path = Some(canonical_path);
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    pub fn legacy_message(&self) -> &str {
        &self.message
    }
}

pub fn invalid_value_diagnostic(path: ConfigPath, message: impl Into<String>) -> ConfigDiagnostic {
    ConfigDiagnostic::error(
        ConfigDiagnosticCode::InvalidValue,
        ConfigDiagnosticSource::Validation,
        message,
    )
    .with_schema_source(ConfigDiagnosticSchemaSource::BuiltIn)
    .at_path(path)
}

pub fn unsupported_field_diagnostic(
    path: ConfigPath,
    message: impl Into<String>,
) -> ConfigDiagnostic {
    ConfigDiagnostic::error(
        ConfigDiagnosticCode::UnsupportedField,
        ConfigDiagnosticSource::Schema,
        message,
    )
    .with_schema_source(ConfigDiagnosticSchemaSource::BuiltIn)
    .at_path(path.clone())
    .with_canonical_path(path)
}

pub fn rejected_field_diagnostic(path: ConfigPath, message: impl Into<String>) -> ConfigDiagnostic {
    ConfigDiagnostic::error(
        ConfigDiagnosticCode::RejectedField,
        ConfigDiagnosticSource::Schema,
        message,
    )
    .with_schema_source(ConfigDiagnosticSchemaSource::BuiltIn)
    .at_path(path.clone())
    .with_canonical_path(path)
}

pub fn alias_diagnostic(
    used_path: ConfigPath,
    canonical_path: ConfigPath,
    message: impl Into<String>,
) -> ConfigDiagnostic {
    ConfigDiagnostic::warning(
        ConfigDiagnosticCode::AliasApplied,
        ConfigDiagnosticSource::Compatibility,
        message,
    )
    .with_schema_source(ConfigDiagnosticSchemaSource::BuiltIn)
    .at_path(used_path)
    .with_canonical_path(canonical_path)
}

pub(crate) type DiagnosticResult = std::result::Result<(), ConfigDiagnostic>;

pub fn legacy_validation_error_text(diagnostics: &[ConfigDiagnostic]) -> String {
    diagnostics
        .iter()
        .map(ConfigDiagnostic::legacy_message)
        .collect::<Vec<_>>()
        .join("\n")
}
