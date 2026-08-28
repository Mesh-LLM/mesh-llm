//! Derives [`ConfigDiagnostic`]s from [`WIRING_MANIFEST`] so
//! `mesh-llm config validate` agrees with real runtime support instead of
//! drifting from it.
//!
//! For every manifest entry whose [`WiringBehavior`] is not `None`, this
//! module checks whether the field is actually set in `[defaults.*]` or in
//! any `[[models]]` entry, and if so emits a diagnostic matching the
//! entry's behavior:
//!
//! - [`WiringBehavior::Rejected`] -> `RejectedField`, error.
//! - [`WiringBehavior::BailsDownstream`] -> `InvalidValue`, error (the field
//!   passes static validation today but fails at model load or request
//!   resolution; this moves that surprise earlier, to `config validate`).
//! - [`WiringBehavior::SilentNoOp`] -> `UnsupportedField`, warning (the
//!   field is accepted and resolved, but no runtime consumer reads it).

use crate::diagnostic::{ConfigDiagnostic, ConfigDiagnosticCode, ConfigDiagnosticSeverity};
use crate::model::{ConfigPath, MeshConfig, ModelConfigEntry, PluginConfigEntry};
use crate::wiring_status::{WIRING_MANIFEST, WiringBehavior};

/// Resolves `dotted` (e.g. `model_fit.kv_offload`) inside `value` (a TOML
/// table) to its leaf value, treating an empty array or empty string the
/// same as an absent key. `Vec<String>` fields without `Option<..>` always
/// serialize, even when the operator never set them (e.g.
/// `lora_adapters = []`), so a bare presence check on those fields would
/// produce a false-positive diagnostic for every model.
fn toml_path_leaf<'a>(value: &'a toml::Value, dotted: &str) -> Option<&'a toml::Value> {
    let mut current = value;
    for segment in dotted.split('.') {
        current = current.as_table().and_then(|table| table.get(segment))?;
    }
    match current {
        toml::Value::Array(items) if items.is_empty() => None,
        toml::Value::String(text) if text.is_empty() => None,
        _ => Some(current),
    }
}

/// A handful of manifest paths only fail downstream for specific values
/// (for example `model_fit.kv_unified` is discarded silently when
/// `false`/`auto`, but fails at model load when `true`), even though the
/// manifest classifies the whole path as one status. Any path not listed
/// here is flagged purely on presence, matching the manifest's stated
/// behavior.
fn manifest_diagnostic_applies(path: &str, leaf: &toml::Value) -> bool {
    match path {
        "model_fit.kv_unified" | "model_fit.context_shift" => {
            matches!(leaf, toml::Value::Boolean(true) | toml::Value::String(_))
                && !matches!(leaf, toml::Value::String(mode) if mode.eq_ignore_ascii_case("auto"))
                && !matches!(leaf, toml::Value::Boolean(false))
        }
        "model_fit.cache_ram_mib" | "model_fit.cache_idle_slots" | "model_fit.keep_tokens" => {
            leaf.as_integer().is_some_and(|amount| amount > 0)
        }
        "speculative.draft_acceptance_threshold" => {
            leaf.as_float().is_some_and(|threshold| threshold > 0.0)
                || leaf.as_integer().is_some_and(|threshold| threshold > 0)
        }
        _ => true,
    }
}

fn diagnostic_for_behavior(
    behavior: WiringBehavior,
    scope_path: String,
    manifest_path: &str,
    owner: &str,
    reason: &str,
) -> Option<ConfigDiagnostic> {
    let path = ConfigPath::parse_rendered(&scope_path).ok()?;
    let canonical_path = canonical_scope_path(&scope_path, manifest_path);
    match behavior {
        WiringBehavior::None => None,
        WiringBehavior::Rejected => Some(
            ConfigDiagnostic::error(
                ConfigDiagnosticCode::RejectedField,
                crate::diagnostic::ConfigDiagnosticSource::Schema,
                format!("{manifest_path} is not supported by mesh-llm config: {reason}"),
            )
            .at_path(path)
            .with_canonical_path(canonical_path)
            .with_help(format!(
                "remove {manifest_path} from config.toml; {reason}"
            )),
        ),
        WiringBehavior::BailsDownstream => Some(
            ConfigDiagnostic::error(
                ConfigDiagnosticCode::InvalidValue,
                crate::diagnostic::ConfigDiagnosticSource::Validation,
                format!(
                    "{manifest_path} is set, but this value fails downstream today: {reason}"
                ),
            )
            .at_path(path)
            .with_canonical_path(canonical_path)
            .with_help(format!(
                "remove {manifest_path} until {owner} wires this field, or it will fail at model load/resolution"
            )),
        ),
        WiringBehavior::SilentNoOp => Some(
            ConfigDiagnostic::warning(
                ConfigDiagnosticCode::UnsupportedField,
                crate::diagnostic::ConfigDiagnosticSource::Schema,
                format!(
                    "{manifest_path} is set, but it is a no-op today: {reason}"
                ),
            )
            .at_path(path)
            .with_canonical_path(canonical_path)
            .with_help(format!(
                "{manifest_path} does not change runtime behavior yet; {owner} wires it")),
        ),
    }
}

/// Normalize a triggering scope path (`defaults.X`, `models[N].X`, or
/// `plugin[N].X`) into the canonical placeholder form
/// (`defaults.X`, `models.<model-ref>.X`, `plugin.<plugin-name>.X`) used
/// across the rest of the built-in schema.
fn canonical_scope_path(scope_path: &str, manifest_path: &str) -> ConfigPath {
    if scope_path.starts_with("defaults.") {
        return ConfigPath::parse_rendered(&format!("defaults.{manifest_path}"))
            .unwrap_or_else(|_| ConfigPath::field(manifest_path));
    }
    if scope_path.starts_with("models[") {
        return ConfigPath::parse_rendered(&format!(
            "models.{}.{manifest_path}",
            crate::model::CANONICAL_MODEL_REF_SEGMENT
        ))
        .unwrap_or_else(|_| ConfigPath::field(manifest_path));
    }
    if scope_path.starts_with("plugin[") {
        return ConfigPath::parse_rendered(&format!(
            "plugin.{}.{manifest_path}",
            crate::model::CANONICAL_PLUGIN_NAME_SEGMENT
        ))
        .unwrap_or_else(|_| ConfigPath::field(manifest_path));
    }
    ConfigPath::field(manifest_path)
}

fn severity_of(behavior: WiringBehavior) -> Option<ConfigDiagnosticSeverity> {
    match behavior {
        WiringBehavior::None => None,
        WiringBehavior::Rejected | WiringBehavior::BailsDownstream => {
            Some(ConfigDiagnosticSeverity::Error)
        }
        WiringBehavior::SilentNoOp => Some(ConfigDiagnosticSeverity::Warning),
    }
}

fn push_plugin_diagnostics(plugins: &[PluginConfigEntry], diagnostics: &mut Vec<ConfigDiagnostic>) {
    let Some(url_entry) = WIRING_MANIFEST
        .iter()
        .find(|entry| entry.path == "plugin.<name>.url")
    else {
        return;
    };
    let Some(optional_entry) = WIRING_MANIFEST
        .iter()
        .find(|entry| entry.path == "plugin.<name>.startup.optional")
    else {
        return;
    };

    for (index, plugin) in plugins.iter().enumerate() {
        if plugin.url.is_some()
            && let Some(diagnostic) = diagnostic_for_behavior(
                url_entry.behavior,
                format!("plugin[{index}].url"),
                url_entry.path,
                url_entry.owner,
                url_entry.reason,
            )
        {
            diagnostics.push(diagnostic);
        }
        if plugin.startup.optional
            && let Some(diagnostic) = diagnostic_for_behavior(
                optional_entry.behavior,
                format!("plugin[{index}].startup.optional"),
                optional_entry.path,
                optional_entry.owner,
                optional_entry.reason,
            )
        {
            diagnostics.push(diagnostic);
        }
    }
}

fn push_scope_diagnostics(
    value: &toml::Value,
    scope_prefix: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    for entry in WIRING_MANIFEST {
        let Some(severity) = severity_of(entry.behavior) else {
            continue;
        };
        if entry.path.starts_with("plugin.") {
            continue;
        }
        let Some(leaf) = toml_path_leaf(value, entry.path) else {
            continue;
        };
        if !manifest_diagnostic_applies(entry.path, leaf) {
            continue;
        }
        let scope_path = format!("{scope_prefix}.{path}", path = entry.path);
        if let Some(mut diagnostic) = diagnostic_for_behavior(
            entry.behavior,
            scope_path,
            entry.path,
            entry.owner,
            entry.reason,
        ) {
            debug_assert_eq!(diagnostic.severity, severity);
            diagnostic.severity = severity;
            diagnostics.push(diagnostic);
        }
    }
}

/// Derive manifest-driven diagnostics for `config.defaults` and every
/// `config.models[]` entry, in addition to whatever `[[plugin]]` entries
/// set `url` or `startup.optional`.
pub fn wiring_manifest_diagnostics(config: &MeshConfig) -> Vec<ConfigDiagnostic> {
    let mut diagnostics = Vec::new();

    if let Some(defaults) = &config.defaults
        && let Ok(value) = toml::Value::try_from(defaults)
    {
        push_scope_diagnostics(&value, "defaults", &mut diagnostics);
    }

    for (index, model) in config.models.iter().enumerate() {
        if let Ok(value) = model_entry_toml_value(model) {
            push_scope_diagnostics(&value, &format!("models[{index}]"), &mut diagnostics);
        }
    }

    push_plugin_diagnostics(&config.plugins, &mut diagnostics);

    diagnostics
}

fn model_entry_toml_value(model: &ModelConfigEntry) -> Result<toml::Value, toml::ser::Error> {
    toml::Value::try_from(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeshConfig;

    #[test]
    fn rejected_request_defaults_field_produces_rejected_field_error() {
        let config: MeshConfig = toml::from_str(
            r#"
[defaults.request_defaults]
logprobs = { enabled = true }
"#,
        )
        .expect("config should parse");

        let diagnostics = wiring_manifest_diagnostics(&config);
        let diagnostic = diagnostics
            .iter()
            .find(|d| {
                d.path.as_ref().map(ConfigPath::render)
                    == Some("defaults.request_defaults.logprobs".to_string())
            })
            .expect("logprobs should produce a diagnostic");
        assert_eq!(diagnostic.code, ConfigDiagnosticCode::RejectedField);
        assert_eq!(diagnostic.severity, ConfigDiagnosticSeverity::Error);
    }

    #[test]
    fn bails_downstream_field_produces_error_before_model_load() {
        let config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "test-model"

[models.model_fit]
keep_tokens = 32
"#,
        )
        .expect("config should parse");

        let diagnostics = wiring_manifest_diagnostics(&config);
        assert!(diagnostics.iter().any(|d| {
            d.path.as_ref().map(ConfigPath::render)
                == Some("models[0].model_fit.keep_tokens".to_string())
                && d.severity == ConfigDiagnosticSeverity::Error
        }));
    }

    #[test]
    fn silent_no_op_field_produces_warning() {
        let config: MeshConfig = toml::from_str(
            r#"
[defaults.hardware]
warmup = true
"#,
        )
        .expect("config should parse");

        let diagnostics = wiring_manifest_diagnostics(&config);
        assert!(diagnostics.iter().any(|d| {
            d.path.as_ref().map(ConfigPath::render) == Some("defaults.hardware.warmup".to_string())
                && d.severity == ConfigDiagnosticSeverity::Warning
        }));
    }

    #[test]
    fn wired_fields_produce_no_manifest_diagnostics() {
        let config: MeshConfig = toml::from_str(
            r#"
[defaults.model_fit]
ctx_size = 4096
"#,
        )
        .expect("config should parse");

        let diagnostics = wiring_manifest_diagnostics(&config);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn unset_fields_produce_no_diagnostics() {
        let config = MeshConfig::default();
        assert!(wiring_manifest_diagnostics(&config).is_empty());
    }
}
