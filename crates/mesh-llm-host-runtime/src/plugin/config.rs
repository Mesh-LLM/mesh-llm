use super::{
    BLOBSTORE_PLUGIN_ID, FLASH_MOE_PLUGIN_ID, OPENAI_ENDPOINT_PLUGIN_ID, PluginSummary,
    TELEMETRY_PLUGIN_ID,
};
use anyhow::{Context, Result, bail};
#[allow(unused_imports)]
pub use mesh_llm_config::{
    AdvancedConfig, AdvancedServerConfig, BoolOrAuto, BoolOrString, ConfigEditor, ConfigStore,
    FlashAttentionType, GpuAssignment, GpuConfig, HardwareConfig, IntegerOrString,
    LocalServingNodeConfig, MeshConfig, ModelConfigDefaults, ModelConfigEditor, ModelConfigEntry,
    ModelDefaultsEditor, ModelFitConfig, ModelRuntimeKind, MultimodalConfig, OwnerControlConfig,
    PluginConfigEditor, PluginConfigEntry, PrefixCacheConfig, ReasoningBudget, ReasoningEnabled,
    RequestDefaultsConfig, ReservedObjectConfig, SkippyConfig, SpeculativeConfig,
    StringOrStringList, TelemetryConfig, TelemetryMetricsConfig, TensorSplitConfig,
    ThroughputConfig, config_path, config_to_toml, load_config, parse_config_toml, validate_config,
};
use mesh_llm_plugin::MeshVisibility;
use std::collections::BTreeMap;

const FLASH_MOE_INSTALL_HINT: &str = "Install Flash-MoE separately and set \
                                     `command` to its infer binary, or set \
                                     `url` to an already-running Flash-MoE /v1 endpoint.";

#[derive(Clone, Debug)]
pub struct ResolvedPlugins {
    pub externals: Vec<ExternalPluginSpec>,
    pub inactive: Vec<PluginSummary>,
}

#[derive(Clone, Debug)]
pub struct ExternalPluginSpec {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    /// Backend URL for inference endpoint plugins.
    pub url: Option<String>,
    /// Extra environment passed only to the plugin process.
    pub env: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug)]
pub struct PluginHostMode {
    pub mesh_visibility: MeshVisibility,
}

pub(crate) fn telemetry_plugin_enabled(config: &MeshConfig) -> bool {
    config
        .plugins
        .iter()
        .find(|entry| entry.name == TELEMETRY_PLUGIN_ID)
        .map(|entry| entry.enabled.unwrap_or(true))
        .unwrap_or(true)
}

pub fn resolve_plugins(config: &MeshConfig, _host_mode: PluginHostMode) -> Result<ResolvedPlugins> {
    let mut externals = Vec::new();
    let inactive = Vec::new();
    let mut names = BTreeMap::<String, ()>::new();
    let mut blobstore_enabled = true;
    let mut openai_endpoint_enabled = false;
    let mut openai_endpoint_url: Option<String> = None;
    let mut flash_moe_entry: Option<&PluginConfigEntry> = None;
    let mut telemetry_enabled = true;
    for entry in &config.plugins {
        if names.insert(entry.name.clone(), ()).is_some() {
            bail!("Duplicate plugin entry '{}'", entry.name);
        }
        let enabled = entry.enabled.unwrap_or(true);
        if entry.name == BLOBSTORE_PLUGIN_ID {
            if entry.command.is_some() || !entry.args.is_empty() || entry.url.is_some() {
                bail!(
                    "Plugin '{}' is served by mesh-llm itself; only `enabled` may be set",
                    BLOBSTORE_PLUGIN_ID
                );
            }
            blobstore_enabled = enabled;
            continue;
        }
        if entry.name == OPENAI_ENDPOINT_PLUGIN_ID {
            if entry.command.is_some() || !entry.args.is_empty() {
                bail!(
                    "Plugin '{}' is served by mesh-llm itself; only `enabled` and `url` may be set",
                    OPENAI_ENDPOINT_PLUGIN_ID
                );
            }
            openai_endpoint_enabled = enabled;
            if let Some(ref url) = entry.url {
                openai_endpoint_url = Some(url.clone());
            }
            continue;
        }
        if entry.name == FLASH_MOE_PLUGIN_ID {
            if !enabled {
                continue;
            }
            flash_moe_entry = Some(entry);
            continue;
        }
        if entry.name == TELEMETRY_PLUGIN_ID {
            if entry.command.is_some() || !entry.args.is_empty() || entry.url.is_some() {
                bail!(
                    "Plugin '{}' is served by mesh-llm itself; only `enabled` may be set",
                    TELEMETRY_PLUGIN_ID
                );
            }
            telemetry_enabled = enabled;
            continue;
        }
        if !enabled {
            continue;
        }
        let command = entry
            .command
            .clone()
            .with_context(|| format!("Plugin '{}' is enabled but missing command", entry.name))?;
        externals.push(ExternalPluginSpec {
            name: entry.name.clone(),
            command,
            args: entry.args.clone(),
            url: None,
            env: BTreeMap::new(),
        });
    }

    if telemetry_enabled {
        externals.insert(0, telemetry_plugin_spec()?);
    }
    if openai_endpoint_enabled {
        let mut spec = openai_endpoint_plugin_spec()?;
        spec.url = openai_endpoint_url;
        externals.push(spec);
    }
    if let Some(entry) = flash_moe_entry {
        externals.push(flash_moe_plugin_spec(entry)?);
    }
    if blobstore_enabled {
        externals.push(blobstore_plugin_spec()?);
    }

    Ok(ResolvedPlugins {
        externals,
        inactive,
    })
}

pub fn blobstore_plugin_spec() -> Result<ExternalPluginSpec> {
    let command = std::env::current_exe()
        .context("Cannot determine mesh-llm executable path")?
        .display()
        .to_string();
    Ok(ExternalPluginSpec {
        name: BLOBSTORE_PLUGIN_ID.to_string(),
        command,
        args: vec![
            "--log-format".into(),
            "json".into(),
            "--plugin".into(),
            BLOBSTORE_PLUGIN_ID.into(),
        ],
        url: None,
        env: BTreeMap::new(),
    })
}

pub fn openai_endpoint_plugin_spec() -> Result<ExternalPluginSpec> {
    let command = std::env::current_exe()
        .context("Cannot determine mesh-llm executable path")?
        .display()
        .to_string();
    Ok(ExternalPluginSpec {
        name: OPENAI_ENDPOINT_PLUGIN_ID.to_string(),
        command,
        args: vec![
            "--log-format".into(),
            "json".into(),
            "--plugin".into(),
            OPENAI_ENDPOINT_PLUGIN_ID.into(),
        ],
        url: None,
        env: BTreeMap::new(),
    })
}

pub fn flash_moe_plugin_spec(entry: &PluginConfigEntry) -> Result<ExternalPluginSpec> {
    let backend_command = entry
        .command
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let endpoint_url = entry
        .url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());

    if backend_command.is_some() && endpoint_url.is_some() {
        bail!(
            "Plugin '{}' accepts either `command` for a managed flash-moe process or `url` for an already-running endpoint, not both",
            FLASH_MOE_PLUGIN_ID
        );
    }
    if backend_command.is_none() && endpoint_url.is_none() {
        bail!(
            "Plugin '{}' requires `command` or `url`. {}",
            FLASH_MOE_PLUGIN_ID,
            FLASH_MOE_INSTALL_HINT
        );
    }
    if backend_command.is_none() && !entry.args.is_empty() {
        bail!("Plugin '{}' args require `command`", FLASH_MOE_PLUGIN_ID);
    }
    if entry
        .args
        .iter()
        .any(|arg| arg == "--serve" || arg.starts_with("--serve="))
    {
        bail!(
            "Plugin '{}' owns the flash-moe `--serve` port; remove `--serve` from args",
            FLASH_MOE_PLUGIN_ID
        );
    }

    let command = std::env::current_exe()
        .context("Cannot determine mesh-llm executable path")?
        .display()
        .to_string();
    let mut env = BTreeMap::new();
    if let Some(backend_command) = backend_command {
        env.insert(
            "MESH_LLM_FLASH_MOE_COMMAND".to_string(),
            backend_command.to_string(),
        );
        env.insert(
            "MESH_LLM_FLASH_MOE_ARGS_JSON".to_string(),
            serde_json::to_string(&entry.args)?,
        );
    }
    if let Some(url) = endpoint_url {
        env.insert("MESH_LLM_FLASH_MOE_URL".to_string(), url.to_string());
    }

    Ok(ExternalPluginSpec {
        name: FLASH_MOE_PLUGIN_ID.to_string(),
        command,
        args: vec![
            "--log-format".into(),
            "json".into(),
            "--plugin".into(),
            FLASH_MOE_PLUGIN_ID.into(),
        ],
        url: None,
        env,
    })
}

pub fn telemetry_plugin_spec() -> Result<ExternalPluginSpec> {
    let command = std::env::current_exe()
        .context("Cannot determine mesh-llm executable path")?
        .display()
        .to_string();
    Ok(ExternalPluginSpec {
        name: TELEMETRY_PLUGIN_ID.to_string(),
        command,
        args: vec![
            "--log-format".into(),
            "json".into(),
            "--plugin".into(),
            TELEMETRY_PLUGIN_ID.into(),
        ],
        url: None,
        env: BTreeMap::new(),
    })
}

pub fn bundled_cli_plugin_spec(_name: &str) -> Result<Option<ExternalPluginSpec>> {
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    const FULL_SURFACE_VALID_FIXTURE: &str =
        include_str!("../../tests/fixtures/skippy_full_surface_valid.toml");
    const FULL_SURFACE_INVALID_FIXTURE: &str =
        include_str!("../../tests/fixtures/skippy_full_surface_invalid.toml");

    fn documented_matrix_key_paths() -> BTreeSet<String> {
        let matrix = include_str!("../../../../docs/skippy/CONFIGURATION.md");
        matrix
            .lines()
            .filter(|line| line.starts_with('|'))
            .filter_map(|line| {
                let columns: Vec<_> = line.split('|').map(str::trim).collect();
                columns.get(3).copied()
            })
            .filter(|cell| cell.contains('`'))
            .flat_map(|cell| {
                cell.split("<br>")
                    .filter_map(|part| {
                        let trimmed = part.trim();
                        trimmed
                            .strip_prefix('`')
                            .and_then(|value| value.strip_suffix('`'))
                    })
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn test_model(name: &str) -> ModelConfigEntry {
        ModelConfigEntry {
            model: name.into(),
            mmproj: None,
            ctx_size: None,
            gpu_id: None,
            parallel: None,
            cache_type_k: None,
            cache_type_v: None,
            batch: None,
            ubatch: None,
            flash_attention: None,
            model_fit: None,
            hardware: None,
            throughput: None,
            skippy: None,
            speculative: None,
            request_defaults: None,
            multimodal: None,
            advanced: None,
            gpu_id_from_legacy_shim: false,
        }
    }

    #[test]
    fn parse_unified_config_keeps_plugins_and_models() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[owner_control]
bind = "127.0.0.1:7447"
advertise_addr = "203.0.113.10:18443"

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
ctx_size = 8192

[[models]]
model = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/model.gguf"
mmproj = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/mmproj.gguf"

[[plugin]]
name = "demo"
command = "/tmp/demo"
"#,
        )
        .unwrap();

        assert_eq!(config.version, Some(1));
        assert_eq!(
            config.owner_control.bind,
            Some("127.0.0.1:7447".parse().unwrap())
        );
        assert_eq!(
            config.owner_control.advertise_addr,
            Some("203.0.113.10:18443".parse().unwrap())
        );
        assert_eq!(config.gpu.assignment, GpuAssignment::Auto);
        assert_eq!(config.models.len(), 2);
        assert_eq!(config.models[0].model, "Qwen3-8B-Q4_K_M");
        assert_eq!(config.models[0].ctx_size, Some(8192));
        assert_eq!(config.models[0].gpu_id, None);
        assert_eq!(config.models[0].cache_type_k, None);
        assert_eq!(config.models[0].cache_type_v, None);
        assert_eq!(config.models[0].batch, None);
        assert_eq!(config.models[0].ubatch, None);
        assert_eq!(config.models[0].flash_attention, None);
        assert_eq!(
            config.models[1].mmproj.as_deref(),
            Some("bartowski/Qwen2.5-VL-7B-Instruct-GGUF/mmproj.gguf")
        );
        assert_eq!(config.models[1].gpu_id, None);
        assert_eq!(config.plugins.len(), 1);
        assert_eq!(config.plugins[0].name, "demo");
    }

    #[test]
    fn telemetry_config_deserializes_standard_metrics_settings() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[telemetry]
enabled = true
service_name = "mesh-llm"
endpoint = "https://otel.example.com"
headers = { "authorization" = "Bearer TOKEN" }
export_interval_secs = 15
queue_size = 2048
prompt_shape_metrics = false

[telemetry.metrics]
endpoint = "https://otel.example.com/v1/metrics"

[[plugin]]
name = "telemetry"
enabled = true
"#,
        )
        .unwrap();

        assert_eq!(config.telemetry.enabled, Some(true));
        assert_eq!(config.telemetry.service_name.as_deref(), Some("mesh-llm"));
        assert_eq!(
            config.telemetry.endpoint.as_deref(),
            Some("https://otel.example.com")
        );
        assert_eq!(
            config.telemetry.metrics.endpoint.as_deref(),
            Some("https://otel.example.com/v1/metrics")
        );
        assert_eq!(
            config
                .telemetry
                .headers
                .get("authorization")
                .map(String::as_str),
            Some("Bearer TOKEN")
        );
        assert_eq!(config.telemetry.export_interval_secs, Some(15));
        assert_eq!(config.telemetry.queue_size, Some(2048));
        assert!(!config.telemetry.prompt_shape_metrics);
    }

    #[test]
    fn telemetry_config_rejects_zero_queue_size() {
        let config: MeshConfig = toml::from_str(
            r#"
[telemetry]
queue_size = 0
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("telemetry.queue_size must be at least 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn owner_control_config_rejects_ephemeral_non_loopback_bind() {
        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
bind = "0.0.0.0:0"
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains(
            "owner_control.bind must use a concrete port when binding a non-loopback address"
        ));
    }

    #[test]
    fn owner_control_config_rejects_unspecified_advertise_addr() {
        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
advertise_addr = "0.0.0.0:18443"
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("owner_control.advertise_addr must not use an unspecified IP address")
        );
    }

    #[test]
    fn owner_control_config_rejects_ephemeral_advertise_addr() {
        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
advertise_addr = "127.0.0.1:0"
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("owner_control.advertise_addr must use a concrete port")
        );
    }

    #[test]
    fn telemetry_config_rejects_prompt_shape_metrics_until_reviewed() {
        let config: MeshConfig = toml::from_str(
            r#"
[telemetry]
prompt_shape_metrics = true
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("telemetry.prompt_shape_metrics is not supported yet"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn flash_moe_config_requires_external_command_or_endpoint_with_install_hint() {
        let entry = PluginConfigEntry {
            name: FLASH_MOE_PLUGIN_ID.to_string(),
            enabled: Some(true),
            command: None,
            args: Vec::new(),
            url: None,
        };

        let err = flash_moe_plugin_spec(&entry)
            .expect_err("flash-moe requires a managed command or attached endpoint");
        let message = err.to_string();

        assert!(message.contains("Install Flash-MoE separately"));
        assert!(message.contains("command"));
        assert!(message.contains("url"));
    }

    #[test]
    fn pinned_gpu_config_accepted_pinned_config() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen3-8B-Q4_K_M"
gpu_id = "pci:0000:65:00.0"
ctx_size = 8192
"#,
        )
        .unwrap();

        validate_config(&config).unwrap();
        assert_eq!(config.models[0].gpu_id.as_deref(), Some("pci:0000:65:00.0"));
    }

    #[test]
    fn pinned_gpu_config_missing_gpu_id_rejected() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Pinned,
                parallel: None,
            },
            models: vec![test_model("Qwen3-8B-Q4_K_M")],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains(
            "models[0].hardware.device must be set to a non-empty value when gpu.assignment = \"pinned\""
        ));
    }

    #[test]
    fn pinned_gpu_config_accepts_defaults_hardware_device_for_models() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "pinned"

[defaults.hardware]
device = "CUDA0"

[[models]]
model = "Qwen3-8B-Q4_K_M"
"#,
        )
        .unwrap();

        validate_config(&config).unwrap();
        assert!(config.models[0].hardware.is_none());
    }

    #[test]
    fn pinned_gpu_config_allows_defaults_hardware_without_device_when_models_pin_devices() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "pinned"

[defaults.hardware]
gpu_layers = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
device = "CUDA1"
"#,
        )
        .unwrap();

        validate_config(&config).unwrap();
        assert_eq!(config.models[0].gpu_id.as_deref(), Some("CUDA1"));
    }

    #[test]
    fn pinned_gpu_config_empty_gpu_id_rejected() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Pinned,
                parallel: None,
            },
            models: vec![ModelConfigEntry {
                gpu_id: Some("  \t  ".into()),
                gpu_id_from_legacy_shim: true,
                ..test_model("Qwen3-8B-Q4_K_M")
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("models[0].hardware.device must not be empty when set")
        );
    }

    #[test]
    fn hardware_gpu_layers_rejects_i32_overflow() {
        let config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
gpu_layers = 2147483648
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert_eq!(
            err.to_string(),
            "models[0].hardware.gpu_layers must be at most 2147483647"
        );
    }

    #[test]
    fn pinned_gpu_config_auto_assignment_rejects_gpu_id() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
                parallel: None,
            },
            models: vec![ModelConfigEntry {
                gpu_id: Some("pci:0000:65:00.0".into()),
                gpu_id_from_legacy_shim: true,
                ..test_model("Qwen3-8B-Q4_K_M")
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string().contains(
                "models[0].hardware.device must not be set when gpu.assignment = \"auto\""
            )
        );
    }

    #[test]
    fn pinned_gpu_config_preserves_accepted_gpu_id_string_exactly() {
        let raw = r#"
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen3-8B-Q4_K_M"
gpu_id = " pci:0000:65:00.0 "
"#;

        let config: MeshConfig = toml::from_str(raw).unwrap();
        validate_config(&config).unwrap();

        assert_eq!(
            config.models[0].gpu_id.as_deref(),
            Some(" pci:0000:65:00.0 ")
        );
    }

    // ── gpu.parallel validation ──

    #[test]
    fn gpu_parallel_field_deserializes_from_toml() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "auto"
parallel = 8

[[models]]
model = "Qwen3-8B-Q4_K_M"
"#,
        )
        .unwrap();

        assert_eq!(config.gpu.parallel, Some(8));
    }

    #[test]
    fn gpu_parallel_defaults_to_none_when_omitted() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
"#,
        )
        .unwrap();

        assert_eq!(config.gpu.parallel, None);
    }

    #[test]
    fn gpu_parallel_zero_rejected() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
                parallel: Some(0),
            },
            models: vec![test_model("Qwen3-8B-Q4_K_M")],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("gpu.parallel must be at least 1, got 0"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn gpu_parallel_one_accepted() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
                parallel: Some(1),
            },
            models: vec![test_model("Qwen3-8B-Q4_K_M")],
            ..MeshConfig::default()
        };

        validate_config(&config).unwrap();
    }

    #[test]
    fn gpu_parallel_none_accepted() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
                parallel: None,
            },
            models: vec![test_model("Qwen3-8B-Q4_K_M")],
            ..MeshConfig::default()
        };

        validate_config(&config).unwrap();
    }

    #[test]
    fn gpu_parallel_large_value_accepted() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
                parallel: Some(64),
            },
            models: vec![test_model("Qwen3-8B-Q4_K_M")],
            ..MeshConfig::default()
        };

        validate_config(&config).unwrap();
    }

    #[test]
    fn gpu_parallel_unwrap_or_default_is_4() {
        fn parsed_parallel(value: Option<usize>) -> usize {
            value.unwrap_or(4)
        }

        assert_eq!(parsed_parallel(None), 4);
        assert_eq!(parsed_parallel(Some(1)), 1);
        assert_eq!(parsed_parallel(Some(8)), 8);
        assert_eq!(parsed_parallel(Some(64)), 64);
    }

    #[test]
    fn per_model_parallel_valid_value_accepted() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                parallel: Some(8),
                ..test_model("Qwen3-8B-Q4_K_M")
            }],
            ..MeshConfig::default()
        };
        validate_config(&config).unwrap();
    }

    #[test]
    fn per_model_parallel_zero_rejected() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                parallel: Some(0),
                ..test_model("Qwen3-8B-Q4_K_M")
            }],
            ..MeshConfig::default()
        };
        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("models[0].throughput.parallel must be at least 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn per_model_parallel_none_accepted() {
        let config = MeshConfig {
            models: vec![test_model("Qwen3-8B-Q4_K_M")],
            ..MeshConfig::default()
        };
        validate_config(&config).unwrap();
    }

    #[test]
    fn model_runtime_overrides_deserialize_from_toml() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
cache_type_k = "q8_0"
cache_type_v = "q4_0"
batch = 2048
ubatch = 512
flash_attention = "enabled"
"#,
        )
        .unwrap();

        assert_eq!(config.models[0].cache_type_k.as_deref(), Some("q8_0"));
        assert_eq!(config.models[0].cache_type_v.as_deref(), Some("q4_0"));
        assert_eq!(config.models[0].batch, Some(2048));
        assert_eq!(config.models[0].ubatch, Some(512));
        assert_eq!(
            config.models[0].flash_attention,
            Some(FlashAttentionType::Enabled)
        );
    }

    #[test]
    fn model_cache_type_k_empty_rejected() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                cache_type_k: Some("   ".into()),
                ..test_model("Qwen3-8B-Q4_K_M")
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("models[0].model_fit.cache_type_k must not be empty when set")
        );
    }

    #[test]
    fn model_cache_type_v_empty_rejected() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                cache_type_v: Some("   ".into()),
                ..test_model("Qwen3-8B-Q4_K_M")
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("models[0].model_fit.cache_type_v must not be empty when set")
        );
    }

    #[test]
    fn model_batch_zero_rejected() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                batch: Some(0),
                ..test_model("Qwen3-8B-Q4_K_M")
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("models[0].model_fit.batch must be at least 1 when set")
        );
    }

    #[test]
    fn model_ubatch_zero_rejected() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                ubatch: Some(0),
                ..test_model("Qwen3-8B-Q4_K_M")
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("models[0].model_fit.ubatch must be at least 1 when set")
        );
    }

    #[test]
    fn defaults_nested_sections_preserve_existing_behavior_when_omitted() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
ctx_size = 8192
parallel = 4
"#,
        )
        .unwrap();

        validate_config(&config).unwrap();
        assert!(config.defaults.is_none());
        assert_eq!(config.models[0].ctx_size, Some(8192));
        assert_eq!(config.models[0].parallel, Some(4));
        assert_eq!(
            config.models[0].model_fit.as_ref().and_then(|v| v.ctx_size),
            Some(8192)
        );
        assert_eq!(
            config.models[0]
                .throughput
                .as_ref()
                .and_then(|v| v.parallel),
            Some(4)
        );
    }

    #[test]
    fn nested_defaults_parse_representative_sections() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[defaults.model_fit]
ctx_size = 4096
kv_cache_policy = "balanced"

[defaults.hardware]
model_runtime = "cuda"

[defaults.throughput]
parallel = 2

[defaults.skippy]
activation_wire_dtype = "f16"

[defaults.speculative]
mode = "ngram"

[defaults.request_defaults]
temperature = 0.2

[defaults.multimodal]
image_max_tokens = 4096

[defaults.advanced.server]
alias = "qwen-local"

[[models]]
model = "Qwen3-8B-Q4_K_M"
"#,
        )
        .unwrap();

        validate_config(&config).unwrap();
        let defaults = config.defaults.expect("defaults should parse");
        assert_eq!(defaults.model_fit.and_then(|v| v.ctx_size), Some(4096));
        assert_eq!(
            defaults.hardware.and_then(|v| v.model_runtime),
            Some(ModelRuntimeKind::Cuda)
        );
        assert_eq!(defaults.throughput.and_then(|v| v.parallel), Some(2));
        assert_eq!(
            defaults.skippy.and_then(|v| v.activation_wire_dtype),
            Some("f16".into())
        );
        assert_eq!(
            defaults.speculative.and_then(|v| v.mode),
            Some("ngram".into())
        );
    }

    #[test]
    fn canonical_plan_example_auto_sentinels_parse_and_validate() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "auto"

[defaults.model_fit]
ctx_size = 8192
batch = 512
ubatch = 128
kv_cache_policy = "auto"
cache_type_k = "auto"
cache_type_v = "auto"
kv_offload = "auto"
kv_unified = "auto"
cache_ram_mib = 0
cache_idle_slots = 0
prompt_cache = "auto"
context_shift = "auto"

[defaults.hardware]
model_runtime = "auto"
device = "auto"
gpu_layers = "auto"
tensor_split = []
split_mode = "auto"
main_gpu = 0
placement = "auto"
safety_margin_gb = 2.0
mmap = "auto"
mlock = false
direct_io = false
warmup = "auto"

[defaults.throughput]
parallel = 1
continuous_batching = "auto"
threads = 0
threads_batch = 0
tuning_profile = "balanced"
numa = "auto"
cpu_affinity = []

[defaults.skippy]
activation_wire_dtype = "auto"
prefill_chunking = "auto"
prefill_chunk_size = 0
binary_stage_transport = "auto"

[defaults.speculative]
mode = "auto"
draft_selection_policy = "auto"
pairing_fault = "warn_disable"
draft_max_tokens = 16
draft_min_tokens = 0
draft_acceptance_threshold = 0.0

[defaults.request_defaults]
temperature = 0.8
top_p = 0.95
top_k = 40
min_p = 0.0
repeat_penalty = 1.0
repeat_last_n = 64
reasoning_format = "auto"
reasoning_budget = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
ctx_size = 8192

[models.model_fit]
ctx_size = 16384
cache_type_k = "q8_0"
cache_type_v = "q8_0"

[models.hardware]
gpu_layers = 99
device = "cuda:0"
"#,
        )
        .unwrap();

        validate_config(&config).unwrap();
        let defaults = config.defaults.as_ref().expect("defaults should parse");
        assert!(matches!(
            defaults.model_fit.as_ref().and_then(|v| v.kv_unified.as_ref()),
            Some(BoolOrAuto::String(value)) if value == "auto"
        ));
        assert!(matches!(
            defaults.hardware.as_ref().and_then(|v| v.gpu_layers.as_ref()),
            Some(IntegerOrString::String(value)) if value == "auto"
        ));
        assert!(matches!(
            defaults.hardware.as_ref().and_then(|v| v.tensor_split.as_ref()),
            Some(TensorSplitConfig::Ratios(values)) if values.is_empty()
        ));
        assert!(matches!(
            defaults.request_defaults.as_ref().and_then(|v| v.reasoning_budget.as_ref()),
            Some(ReasoningBudget::String(value)) if value == "auto"
        ));
        assert_eq!(config.models[0].ctx_size, Some(16384));
        assert_eq!(config.models[0].gpu_id.as_deref(), Some("cuda:0"));
    }

    #[test]
    fn legacy_flat_fields_normalize_into_nested_sections() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[[models]]
model = "Qwen3-8B-Q4_K_M"
ctx_size = 8192
gpu_id = "pci:0000:65:00.0"
parallel = 6
cache_type_k = "q8_0"
cache_type_v = "q4_0"
batch = 1024
ubatch = 256
flash_attention = "enabled"
mmproj = "projector.gguf"
"#,
        )
        .unwrap();

        let model = &config.models[0];
        assert_eq!(
            model.model_fit.as_ref().and_then(|v| v.ctx_size),
            Some(8192)
        );
        assert_eq!(
            model.hardware.as_ref().and_then(|v| v.device.as_deref()),
            Some("pci:0000:65:00.0")
        );
        assert_eq!(model.throughput.as_ref().and_then(|v| v.parallel), Some(6));
        assert_eq!(
            model
                .model_fit
                .as_ref()
                .and_then(|v| v.cache_type_k.as_deref()),
            Some("q8_0")
        );
        assert_eq!(model.model_fit.as_ref().and_then(|v| v.batch), Some(1024));
        assert_eq!(
            model.multimodal.as_ref().and_then(|v| v.mmproj.as_deref()),
            Some("projector.gguf")
        );
    }

    #[test]
    fn nested_values_override_legacy_shims() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen3-8B-Q4_K_M"
ctx_size = 4096
gpu_id = "legacy-gpu"
parallel = 2
batch = 256
mmproj = "legacy.gguf"

[models.model_fit]
ctx_size = 8192
batch = 1024

[models.hardware]
device = "nested-gpu"

[models.throughput]
parallel = 8

[models.multimodal]
mmproj = "nested.gguf"
"#,
        )
        .unwrap();

        validate_config(&config).unwrap();
        let model = &config.models[0];
        assert_eq!(model.ctx_size, Some(8192));
        assert_eq!(model.batch, Some(1024));
        assert_eq!(model.gpu_id.as_deref(), Some("nested-gpu"));
        assert_eq!(model.parallel, Some(8));
        assert_eq!(model.mmproj.as_deref(), Some("nested.gguf"));
    }

    #[test]
    fn invalid_model_fit_batch_path_is_stable() {
        let config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.model_fit]
batch = 0
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert_eq!(
            err.to_string(),
            "models[0].model_fit.batch must be at least 1 when set"
        );
    }

    #[test]
    fn invalid_split_mode_path_is_stable() {
        let config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
split_mode = "diagonal"
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert_eq!(
            err.to_string(),
            "models[0].hardware.split_mode must be one of: auto, none, layer, row"
        );
    }

    #[test]
    fn invalid_reasoning_format_path_is_stable() {
        let config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.request_defaults]
reasoning_format = "mystery"
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert_eq!(
            err.to_string(),
            "models[0].request_defaults.reasoning_format must be one of: auto, none, deepseek, deepseek-legacy, hidden"
        );
    }

    #[test]
    fn deepseek_legacy_reasoning_format_is_accepted() {
        let config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.request_defaults]
reasoning_format = "deepseek-legacy"
"#,
        )
        .unwrap();

        validate_config(&config).expect("deepseek-legacy should remain accepted");
    }

    #[test]
    fn invalid_speculative_draft_requires_policy_path_is_stable() {
        let config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.speculative]
mode = "draft"
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert_eq!(
            err.to_string(),
            "models[0].speculative.draft_selection_policy must be set when models[0].speculative.mode = \"draft\" and no explicit draft model source is configured"
        );
    }

    #[test]
    fn invalid_mmproj_conflict_is_rejected() {
        let config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
mmproj = "hardware.gguf"

[models.multimodal]
mmproj = "multimodal.gguf"
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert_eq!(
            err.to_string(),
            "models[0].multimodal.mmproj must match models[0].hardware.mmproj when both are set"
        );
    }

    #[test]
    fn integrated_full_surface_fixture_parses_validates_and_tracks_docs() {
        let config: MeshConfig = toml::from_str(FULL_SURFACE_VALID_FIXTURE).unwrap();

        validate_config(&config).unwrap();
        assert_eq!(config.models.len(), 2);
        assert_eq!(
            config.owner_control.bind,
            Some("127.0.0.1:7447".parse().unwrap())
        );
        assert_eq!(
            config.owner_control.advertise_addr,
            Some("203.0.113.10:18443".parse().unwrap())
        );

        let defaults = config.defaults.as_ref().expect("defaults should parse");
        assert_eq!(
            defaults.model_fit.as_ref().and_then(|fit| fit.ctx_size),
            Some(8192)
        );
        assert_eq!(
            defaults
                .request_defaults
                .as_ref()
                .and_then(|request_defaults| request_defaults.temperature),
            Some(0.2)
        );

        let explicit = &config.models[0];
        assert_eq!(explicit.model, "Qwen/Qwen3-0.6B:Q4_K_M");
        assert_eq!(
            explicit.model_fit.as_ref().and_then(|fit| fit.ctx_size),
            Some(16384)
        );
        assert_eq!(
            explicit
                .hardware
                .as_ref()
                .and_then(|hardware| hardware.stage_layer_start),
            Some(12)
        );
        assert_eq!(
            explicit
                .skippy
                .as_ref()
                .and_then(|skippy| skippy.prefill_chunk_schedule.as_deref()),
            Some("128,256,384")
        );

        let omitted = &config.models[1];
        assert_eq!(omitted.model, "ggml-org/gemma-3-270m-it-GGUF:Q8_0");
        assert!(
            omitted.model_fit.is_none(),
            "omitted per-model model_fit should stay absent"
        );
        assert!(
            omitted.request_defaults.is_none(),
            "omitted per-model request defaults should stay absent"
        );

        let matrix = include_str!("../../../../docs/skippy/CONFIGURATION.md");
        let matrix_keys = documented_matrix_key_paths();
        assert!(
            matrix_keys.len() >= 100,
            "expected a substantial canonical key-path set, found {}",
            matrix_keys.len()
        );
        for key in [
            "model_fit.ctx_size",
            "model_fit.prefix_cache.max_entries",
            "hardware.stage_layer_start",
            "hardware.stage_layer_end",
            "skippy.prefill_chunk_schedule",
            "speculative.draft_gpu_layers",
            "request_defaults.reasoning_budget",
            "multimodal.mmproj",
            "advanced.server.alias",
        ] {
            assert!(matrix.contains(key), "missing matrix doc entry {key}");
        }

        let docs_readme = include_str!("../../../../docs/README.md");
        let usage = include_str!("../../../../docs/USAGE.md");
        let cli = include_str!("../../../../docs/CLI.md");
        assert!(docs_readme.contains("[skippy/CONFIGURATION.md](skippy/CONFIGURATION.md)"));
        assert!(usage.contains("request payload values still win"));
        assert!(cli.contains("Request defaults only fill absent or null request fields"));
        assert!(cli.contains("Staged-only controls stay staged-only."));
    }

    #[test]
    fn integrated_invalid_fixture_reports_batch_then_pinned_device_paths() {
        let invalid: MeshConfig = toml::from_str(FULL_SURFACE_INVALID_FIXTURE).unwrap();
        let batch_error = validate_config(&invalid).unwrap_err().to_string();
        assert_eq!(
            batch_error,
            "models[0].model_fit.batch must be at least 1 when set"
        );

        let repaired_batch = FULL_SURFACE_INVALID_FIXTURE.replace("batch = 0", "batch = 64");
        let repaired_batch =
            repaired_batch.replace("[defaults.hardware]\ndevice = \"CUDA0\"\n\n", "");
        let repaired: MeshConfig = toml::from_str(&repaired_batch).unwrap();
        let pinned_error = validate_config(&repaired).unwrap_err().to_string();
        assert_eq!(
            pinned_error,
            "models[0].hardware.device must be set to a non-empty value when gpu.assignment = \"pinned\""
        );
    }
}
