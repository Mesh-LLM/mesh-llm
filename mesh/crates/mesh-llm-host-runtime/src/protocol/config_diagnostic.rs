use mesh_llm_config::{
    ConfigDiagnostic, ConfigDiagnosticCode, ConfigDiagnosticSchemaSource, ConfigDiagnosticSeverity,
    ConfigDiagnosticSource, ConfigPath,
};

pub(crate) fn config_diagnostic_to_proto(
    diagnostic: &ConfigDiagnostic,
) -> crate::proto::node::ConfigDiagnostic {
    crate::proto::node::ConfigDiagnostic {
        code: match diagnostic.code {
            ConfigDiagnosticCode::InvalidValue => {
                crate::proto::node::ConfigDiagnosticCode::InvalidValue as i32
            }
            ConfigDiagnosticCode::MissingRequiredValue => {
                crate::proto::node::ConfigDiagnosticCode::MissingRequiredValue as i32
            }
            ConfigDiagnosticCode::UnsupportedField => {
                crate::proto::node::ConfigDiagnosticCode::UnsupportedField as i32
            }
            ConfigDiagnosticCode::RejectedField => {
                crate::proto::node::ConfigDiagnosticCode::RejectedField as i32
            }
            ConfigDiagnosticCode::AliasApplied => {
                crate::proto::node::ConfigDiagnosticCode::AliasApplied as i32
            }
            ConfigDiagnosticCode::MisplacedField => {
                crate::proto::node::ConfigDiagnosticCode::MisplacedField as i32
            }
            ConfigDiagnosticCode::UnknownField => {
                crate::proto::node::ConfigDiagnosticCode::UnknownField as i32
            }
            ConfigDiagnosticCode::SchemaUnavailable => {
                crate::proto::node::ConfigDiagnosticCode::SchemaUnavailable as i32
            }
            ConfigDiagnosticCode::LegacyUnvalidatedConfig => {
                crate::proto::node::ConfigDiagnosticCode::LegacyUnvalidatedConfig as i32
            }
            ConfigDiagnosticCode::UnsupportedSchemaVersion => {
                crate::proto::node::ConfigDiagnosticCode::UnsupportedSchemaVersion as i32
            }
        },
        severity: match diagnostic.severity {
            ConfigDiagnosticSeverity::Error => {
                crate::proto::node::ConfigDiagnosticSeverity::Error as i32
            }
            ConfigDiagnosticSeverity::Warning => {
                crate::proto::node::ConfigDiagnosticSeverity::Warning as i32
            }
            ConfigDiagnosticSeverity::Info => {
                crate::proto::node::ConfigDiagnosticSeverity::Info as i32
            }
        },
        source: match diagnostic.source {
            ConfigDiagnosticSource::Validation => {
                crate::proto::node::ConfigDiagnosticSource::Validation as i32
            }
            ConfigDiagnosticSource::Schema => {
                crate::proto::node::ConfigDiagnosticSource::Schema as i32
            }
            ConfigDiagnosticSource::Plugin => {
                crate::proto::node::ConfigDiagnosticSource::Plugin as i32
            }
            ConfigDiagnosticSource::Compatibility => {
                crate::proto::node::ConfigDiagnosticSource::Compatibility as i32
            }
        },
        schema_source: diagnostic
            .schema_source
            .map(|schema_source| match schema_source {
                ConfigDiagnosticSchemaSource::BuiltIn => {
                    crate::proto::node::ConfigDiagnosticSchemaSource::BuiltIn as i32
                }
                ConfigDiagnosticSchemaSource::Engine => {
                    crate::proto::node::ConfigDiagnosticSchemaSource::Engine as i32
                }
                ConfigDiagnosticSchemaSource::Plugin => {
                    crate::proto::node::ConfigDiagnosticSchemaSource::Plugin as i32
                }
            }),
        path: diagnostic.path.as_ref().map(ConfigPath::render),
        canonical_path: diagnostic.canonical_path.as_ref().map(ConfigPath::render),
        message: diagnostic.message.clone(),
        help: diagnostic.help.clone(),
    }
}

pub(crate) fn proto_config_diagnostic_to_local(
    diagnostic: &crate::proto::node::ConfigDiagnostic,
) -> ConfigDiagnostic {
    let mut local = ConfigDiagnostic::new(
        match crate::proto::node::ConfigDiagnosticCode::try_from(diagnostic.code)
            .unwrap_or(crate::proto::node::ConfigDiagnosticCode::InvalidValue)
        {
            crate::proto::node::ConfigDiagnosticCode::MissingRequiredValue => {
                ConfigDiagnosticCode::MissingRequiredValue
            }
            crate::proto::node::ConfigDiagnosticCode::UnsupportedField => {
                ConfigDiagnosticCode::UnsupportedField
            }
            crate::proto::node::ConfigDiagnosticCode::RejectedField => {
                ConfigDiagnosticCode::RejectedField
            }
            crate::proto::node::ConfigDiagnosticCode::AliasApplied => {
                ConfigDiagnosticCode::AliasApplied
            }
            crate::proto::node::ConfigDiagnosticCode::MisplacedField => {
                ConfigDiagnosticCode::MisplacedField
            }
            crate::proto::node::ConfigDiagnosticCode::UnknownField => {
                ConfigDiagnosticCode::UnknownField
            }
            crate::proto::node::ConfigDiagnosticCode::SchemaUnavailable => {
                ConfigDiagnosticCode::SchemaUnavailable
            }
            crate::proto::node::ConfigDiagnosticCode::LegacyUnvalidatedConfig => {
                ConfigDiagnosticCode::LegacyUnvalidatedConfig
            }
            crate::proto::node::ConfigDiagnosticCode::UnsupportedSchemaVersion => {
                ConfigDiagnosticCode::UnsupportedSchemaVersion
            }
            crate::proto::node::ConfigDiagnosticCode::InvalidValue
            | crate::proto::node::ConfigDiagnosticCode::Unspecified => {
                ConfigDiagnosticCode::InvalidValue
            }
        },
        match crate::proto::node::ConfigDiagnosticSeverity::try_from(diagnostic.severity)
            .unwrap_or(crate::proto::node::ConfigDiagnosticSeverity::Error)
        {
            crate::proto::node::ConfigDiagnosticSeverity::Warning => {
                ConfigDiagnosticSeverity::Warning
            }
            crate::proto::node::ConfigDiagnosticSeverity::Info => ConfigDiagnosticSeverity::Info,
            crate::proto::node::ConfigDiagnosticSeverity::Error
            | crate::proto::node::ConfigDiagnosticSeverity::Unspecified => {
                ConfigDiagnosticSeverity::Error
            }
        },
        match crate::proto::node::ConfigDiagnosticSource::try_from(diagnostic.source)
            .unwrap_or(crate::proto::node::ConfigDiagnosticSource::Validation)
        {
            crate::proto::node::ConfigDiagnosticSource::Schema => ConfigDiagnosticSource::Schema,
            crate::proto::node::ConfigDiagnosticSource::Plugin => ConfigDiagnosticSource::Plugin,
            crate::proto::node::ConfigDiagnosticSource::Compatibility => {
                ConfigDiagnosticSource::Compatibility
            }
            crate::proto::node::ConfigDiagnosticSource::Validation
            | crate::proto::node::ConfigDiagnosticSource::Unspecified => {
                ConfigDiagnosticSource::Validation
            }
        },
        diagnostic.message.clone(),
    );
    local.schema_source = diagnostic.schema_source.and_then(|schema_source| {
        match crate::proto::node::ConfigDiagnosticSchemaSource::try_from(schema_source)
            .unwrap_or(crate::proto::node::ConfigDiagnosticSchemaSource::Unspecified)
        {
            crate::proto::node::ConfigDiagnosticSchemaSource::BuiltIn => {
                Some(ConfigDiagnosticSchemaSource::BuiltIn)
            }
            crate::proto::node::ConfigDiagnosticSchemaSource::Engine => {
                Some(ConfigDiagnosticSchemaSource::Engine)
            }
            crate::proto::node::ConfigDiagnosticSchemaSource::Plugin => {
                Some(ConfigDiagnosticSchemaSource::Plugin)
            }
            crate::proto::node::ConfigDiagnosticSchemaSource::Unspecified => None,
        }
    });
    local.path = diagnostic
        .path
        .as_deref()
        .and_then(|path| ConfigPath::parse_rendered(path).ok());
    local.canonical_path = diagnostic
        .canonical_path
        .as_deref()
        .and_then(|path| ConfigPath::parse_rendered(path).ok());
    local.help = diagnostic.help.clone();
    local
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_diagnostic_proto_roundtrip_preserves_structured_fields() {
        let canonical_path = "models.<model-ref>.hardware.device";
        let parsed_canonical_path =
            ConfigPath::parse_rendered(canonical_path).expect("canonical path should parse");
        assert_eq!(parsed_canonical_path.render(), canonical_path);

        let diagnostic = ConfigDiagnostic::warning(
            ConfigDiagnosticCode::AliasApplied,
            ConfigDiagnosticSource::Compatibility,
            "legacy alias accepted",
        )
        .with_schema_source(ConfigDiagnosticSchemaSource::BuiltIn)
        .at_path(ConfigPath::parse_rendered("models[0].gpu_id").expect("valid path"))
        .with_canonical_path(parsed_canonical_path)
        .with_help("use models.<model-ref>.hardware.device instead");

        let proto = config_diagnostic_to_proto(&diagnostic);
        assert_eq!(proto.canonical_path.as_deref(), Some(canonical_path));

        let roundtripped = proto_config_diagnostic_to_local(&proto);

        assert_eq!(roundtripped, diagnostic);
        assert_eq!(
            roundtripped.canonical_path.as_ref().map(ConfigPath::render),
            Some(canonical_path.to_string())
        );
    }
}
