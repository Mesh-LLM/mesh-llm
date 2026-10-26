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
        let diagnostics = validate_config_diagnostics_with_plugin_schemas(&config, Some(raw), |_| {
            crate::PluginSchemaAvailability::NotInstalled
        });
        let unknown = diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code == ConfigDiagnosticCode::UnknownField)
            .expect("unknown field is reported");

        assert_eq!(unknown.severity, ConfigDiagnosticSeverity::Warning);
    }
}
