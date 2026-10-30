//! Configuration shared by host-managed inference providers.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// When an inference provider should construct its model sessions.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelLoadPolicy {
    #[default]
    Startup,
    OnDemand,
}

/// Model residency policy. Disk assets remain cached after idle unloading.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ModelLifecycle {
    pub load: ModelLoadPolicy,
    pub idle_timeout_secs: u64,
    pub prefetch: bool,
}

impl Default for ModelLifecycle {
    fn default() -> Self {
        Self {
            load: ModelLoadPolicy::Startup,
            idle_timeout_secs: 0,
            prefetch: true,
        }
    }
}

/// Explicit source overrides. A local directory never triggers a download.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ModelSource {
    pub local_dir: Option<PathBuf>,
    pub repo: Option<String>,
    pub revision: Option<String>,
}

/// A model selection handed to its configured inference plugin.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ManagedModelConfig {
    pub model: String,
    pub provider: String,
    #[serde(default = "default_variant")]
    pub variant: String,
    #[serde(default)]
    pub options: BTreeMap<String, toml::Value>,
    #[serde(default)]
    pub request_defaults: BTreeMap<String, toml::Value>,
    #[serde(default)]
    pub lifecycle: ModelLifecycle,
    #[serde(default)]
    pub source: ModelSource,
}

pub fn default_variant() -> String {
    "fp32".into()
}

/// Validate provider references using the caller's local installed inventory.
pub fn provider_diagnostics(
    config: &crate::MeshConfig,
    mut installed: impl FnMut(&str) -> bool,
) -> Vec<crate::ConfigDiagnostic> {
    config
        .models
        .iter()
        .filter_map(|model| {
            let provider = model.provider.as_deref()?;
            let entry = config.plugins.iter().find(|entry| entry.name == provider);
            let available = !matches!(provider, "blobstore" | "payments" | "wallet-lexe")
                && entry.is_none_or(|entry| entry.enabled != Some(false))
                && (entry.is_some_and(|entry| {
                    entry
                        .command
                        .as_deref()
                        .is_some_and(|command| !command.trim().is_empty())
                }) || installed(provider));
            (!available).then(|| {
                crate::ConfigDiagnostic::error(
                    crate::ConfigDiagnosticCode::InvalidValue,
                    crate::ConfigDiagnosticSource::Validation,
                    format!(
                        "model '{}' selects unavailable provider '{}'",
                        model.model, provider
                    ),
                )
            })
        })
        .collect()
}

impl crate::ModelConfigEntry {
    /// Return this entry's plugin selection, without inheriting LLM defaults.
    pub fn managed_model_config(&self) -> Option<ManagedModelConfig> {
        Some(ManagedModelConfig {
            model: self.model.clone(),
            provider: self.provider.clone()?,
            variant: self.variant.clone().unwrap_or_else(default_variant),
            options: self.options.clone(),
            request_defaults: self.provider_request_defaults.clone(),
            lifecycle: self.lifecycle.clone(),
            source: self.source.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"
[[plugin]]
name = "onnx"
command = "/opt/onnx"
[[models]]
model = "owner/model@pin"
provider = "onnx"
variant = "int8"
[models.options]
threads = 4
math = true
[models.request_defaults]
dpi = 200
mode = "auto"
[models.lifecycle]
load = "on_demand"
idle_timeout_secs = 300
prefetch = false
[models.source]
local_dir = "models/local"
"#;

    #[test]
    fn provider_options_round_trip_without_llm_default_coercion() {
        let config = crate::parse_config_toml(CONFIG).unwrap();
        let model = config.models[0].managed_model_config().unwrap();
        assert_eq!(model.lifecycle.load, ModelLoadPolicy::OnDemand);
        assert!(!model.lifecycle.prefetch);
        assert_eq!(model.options["math"].as_bool(), Some(true));
        assert_eq!(model.request_defaults["mode"].as_str(), Some("auto"));
        assert!(config.models[0].request_defaults.is_none());
        let saved = crate::config_to_toml(&config).unwrap();
        let round_trip = crate::parse_config_toml(&saved).unwrap();
        assert_eq!(round_trip.models[0].managed_model_config(), Some(model));
    }

    #[test]
    fn provider_does_not_accept_ignored_builtin_controls() {
        let config = CONFIG.replace("variant = \"int8\"", "variant = \"int8\"\nctx_size = 4096");
        assert!(crate::parse_config_toml(&config).is_err());
        let config = CONFIG.replace(
            "local_dir = \"models/local\"",
            "local_dir = \"models/local\"\nrepo = \"other/repo\"",
        );
        assert!(crate::parse_config_toml(&config).is_err());
    }

    #[test]
    fn lifecycle_typos_fail_structural_parsing() {
        assert!(
            crate::parse_config_toml(&CONFIG.replace("idle_timeout_secs", "idle_timout_secs"))
                .is_err()
        );
    }
}

#[cfg(test)]
mod provider_validation_tests {
    use super::*;

    #[test]
    fn provider_inventory_respects_commands_disabled_entries_and_builtins() {
        for (name, command, enabled, installed, valid) in [
            ("onnx", Some("/opt/onnx"), true, false, true),
            ("onnx", None, true, true, true),
            ("onnx", None, true, false, false),
            ("onnx", Some("/opt/onnx"), false, true, false),
            ("blobstore", None, true, true, false),
            ("payments", None, true, true, false),
        ] {
            let config = crate::MeshConfig {
                models: vec![crate::ModelConfigEntry {
                    model: "owner/model@pin".into(),
                    provider: Some(name.into()),
                    ..Default::default()
                }],
                plugins: vec![
                    toml::from_str(&format!(
                        "name = '{name}'\nenabled = {enabled}\n{}",
                        command
                            .map(|command| format!("command = '{command}'"))
                            .unwrap_or_default()
                    ))
                    .unwrap(),
                ],
                ..Default::default()
            };
            assert_eq!(
                provider_diagnostics(&config, |_| installed).is_empty(),
                valid,
                "{name} {command:?} {enabled}"
            );
        }
    }
}
