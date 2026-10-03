use crate::diagnostic::{
    ConfigDiagnostic, ConfigDiagnosticCode, ConfigDiagnosticSchemaSource, ConfigDiagnosticSource,
};
use crate::model::{ConfigPath, MeshConfig, RequestDefaultsConfig};

/// The window llama.cpp's repetition and DRY samplers use when none is set.
const DEFAULT_PENALTY_LAST_N: i64 = 64;

/// Warns about penalty windows of -1. They once meant "the whole context",
/// which llama.cpp no longer supports, so the runtime now treats -1 as the
/// default window instead.
pub(crate) fn collect_legacy_penalty_window_warnings(
    config: &MeshConfig,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let defaults = config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.request_defaults.as_ref());
    if let Some(request_defaults) = defaults {
        let mut path = ConfigPath::field("defaults");
        path.push_field("request_defaults");
        push_request_defaults_warnings(request_defaults, &path, diagnostics);
    }
    for (index, model) in config.models.iter().enumerate() {
        if let Some(request_defaults) = model.request_defaults.as_ref() {
            let mut path = ConfigPath::field("models");
            path.push_index(index).push_field("request_defaults");
            push_request_defaults_warnings(request_defaults, &path, diagnostics);
        }
    }
}

fn push_request_defaults_warnings(
    request_defaults: &RequestDefaultsConfig,
    base_path: &ConfigPath,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if request_defaults.repeat_last_n == Some(-1) {
        let mut path = base_path.clone();
        path.push_field("repeat_last_n");
        diagnostics.push(legacy_window_warning(path));
    }
    let dry_window = request_defaults
        .dry
        .as_ref()
        .and_then(|dry| dry.penalty_last_n);
    if dry_window == Some(-1) {
        let mut path = base_path.clone();
        path.push_field("dry").push_field("penalty_last_n");
        diagnostics.push(legacy_window_warning(path));
    }
}

fn legacy_window_warning(path: ConfigPath) -> ConfigDiagnostic {
    let rendered = path.render();
    ConfigDiagnostic::warning(
        ConfigDiagnosticCode::AliasApplied,
        ConfigDiagnosticSource::Compatibility,
        format!(
            "{rendered} = -1 no longer means the whole context; it uses the default window of \
             {DEFAULT_PENALTY_LAST_N} tokens. Set an explicit token count instead"
        ),
    )
    .with_schema_source(ConfigDiagnosticSchemaSource::BuiltIn)
    .at_path(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn warnings_for(toml_source: &str) -> Vec<String> {
        let config: MeshConfig = toml::from_str(toml_source).expect("config should parse");
        let mut diagnostics = Vec::new();
        collect_legacy_penalty_window_warnings(&config, &mut diagnostics);
        diagnostics
            .iter()
            .map(|diagnostic| {
                assert_eq!(diagnostic.code, ConfigDiagnosticCode::AliasApplied);
                diagnostic.path.as_ref().expect("warning path").render()
            })
            .collect()
    }

    #[test]
    fn whole_context_windows_warn_at_each_configured_path() {
        let paths = warnings_for(
            r#"
version = 1

[defaults.request_defaults]
repeat_last_n = -1

[defaults.request_defaults.dry]
penalty_last_n = -1

[[models]]
model = "Qwen3-8B-Q4_K_M"

[[models]]
model = "Qwen3-4B-Q4_K_M"

[models.request_defaults]
repeat_last_n = -1
"#,
        );

        assert_eq!(
            paths,
            [
                "defaults.request_defaults.repeat_last_n",
                "defaults.request_defaults.dry.penalty_last_n",
                "models[1].request_defaults.repeat_last_n",
            ]
        );
    }

    #[test]
    fn explicit_windows_do_not_warn() {
        let paths = warnings_for(
            r#"
version = 1

[defaults.request_defaults]
repeat_last_n = 0

[defaults.request_defaults.dry]
penalty_last_n = 256
"#,
        );

        assert!(paths.is_empty(), "unexpected warnings: {paths:?}");
    }
}
