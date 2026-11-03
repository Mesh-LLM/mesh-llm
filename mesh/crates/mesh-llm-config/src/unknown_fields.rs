use crate::model::RawModelConfigEntry;
use crate::{
    ConfigDiagnostic, ConfigDiagnosticCode, ConfigDiagnosticSource, ConfigPath, MeshConfig,
};

/// Unknown top-level keys and tables. serde flattens every top-level key the
/// config struct does not name into `MeshConfig::extra`, so this needs no
/// parallel key inventory to drift out of sync.
pub(crate) fn unknown_top_level_field_diagnostics(config: &MeshConfig) -> Vec<ConfigDiagnostic> {
    config
        .extra
        .keys()
        .map(|key| {
            ConfigDiagnostic::warning(
                ConfigDiagnosticCode::UnknownField,
                ConfigDiagnosticSource::Validation,
                format!("unknown top-level key '{key}' is ignored"),
            )
            .with_help("check for a typo, or remove the key")
            .at_path(ConfigPath::field(key.clone()))
        })
        .collect()
}

/// Unknown fields inside `[[models]]` rows. The row struct silently drops
/// unknown keys while parsing, so this re-parses the raw TOML with that same
/// struct, whose flattened `extra` captures them: the probe and the parser
/// share one field definition, so there is no key inventory to drift.
pub(crate) fn unknown_model_row_field_diagnostics(raw_toml: Option<&str>) -> Vec<ConfigDiagnostic> {
    let Some(raw_toml) = raw_toml else {
        return Vec::new();
    };
    let Ok(value) = toml::from_str::<toml::Value>(raw_toml) else {
        return Vec::new();
    };
    let Some(models) = value.get("models").and_then(toml::Value::as_array) else {
        return Vec::new();
    };
    let mut diagnostics = Vec::new();
    for (index, row) in models.iter().enumerate() {
        let Ok(row) = row.clone().try_into::<RawModelConfigEntry>() else {
            continue;
        };
        for key in row.extra.keys() {
            let mut path = ConfigPath::field("models");
            path.push_index(index).push_field(key);
            diagnostics.push(
                ConfigDiagnostic::warning(
                    ConfigDiagnosticCode::UnknownField,
                    ConfigDiagnosticSource::Validation,
                    format!("unknown field '{key}' in model row {index} is ignored"),
                )
                .with_help("check for a typo, or remove the key")
                .at_path(path),
            );
        }
    }
    diagnostics
}

#[cfg(test)]
mod tests {
    use crate::{
        ConfigDiagnosticCode, ConfigDiagnosticSeverity,
        validate_config_diagnostics_with_plugin_schemas,
    };

    fn unknown_field_keys(raw: &str) -> Vec<String> {
        let config: crate::MeshConfig = toml::from_str(raw).expect("config parses");
        validate_config_diagnostics_with_plugin_schemas(&config, Some(raw), |_| {
            crate::PluginSchemaAvailability::NotInstalled
        })
        .into_iter()
        .filter(|diagnostic| diagnostic.code == ConfigDiagnosticCode::UnknownField)
        .map(|diagnostic| diagnostic.message)
        .collect()
    }

    #[test]
    fn warns_about_an_unknown_top_level_table() {
        let messages = unknown_field_keys(
            r#"
            [not_a_table]
            anything = 1
            "#,
        );

        assert_eq!(messages.len(), 1, "got {messages:?}");
        assert!(messages[0].contains("not_a_table"), "got {messages:?}");
    }

    #[test]
    fn warns_about_a_misspelt_model_row_key() {
        let messages = unknown_field_keys(
            r#"
            [[models]]
            model = "Org/Name:Q4_K_M"
            ctx_sizee = 4096
            "#,
        );

        assert_eq!(messages.len(), 1, "got {messages:?}");
        assert!(messages[0].contains("ctx_sizee"), "got {messages:?}");
    }

    #[test]
    fn accepts_a_config_without_unknown_fields() {
        let messages = unknown_field_keys(
            r#"
            [[models]]
            model = "Org/Name:Q4_K_M"
            ctx_size = 4096
            "#,
        );

        assert!(messages.is_empty(), "got {messages:?}");
    }

    #[test]
    fn unknown_fields_are_warnings_not_errors() {
        let raw = r#"
        [not_a_table]
        anything = 1
        "#;
        let config: crate::MeshConfig = toml::from_str(raw).expect("config parses");
        let diagnostics =
            validate_config_diagnostics_with_plugin_schemas(&config, Some(raw), |_| {
                crate::PluginSchemaAvailability::NotInstalled
            });
        let unknown = diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code == ConfigDiagnosticCode::UnknownField)
            .expect("unknown field is reported");

        assert_eq!(unknown.severity, ConfigDiagnosticSeverity::Warning);
    }
}
