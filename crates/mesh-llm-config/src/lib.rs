mod authoring;
mod model;
mod store;
mod validate;

pub use authoring::{
    ConfigEditor, LocalServingNodeConfig, ModelConfigEditor, ModelDefaultsEditor,
    PluginConfigEditor,
};
pub use model::*;
pub use store::{ConfigStore, config_path, config_to_toml, load_config, parse_config_toml};
pub use validate::validate_config;

#[cfg(test)]
mod tests {
    use super::{
        ConfigStore, GpuAssignment, LocalServingNodeConfig, MeshConfig, ModelRuntimeKind,
        parse_config_toml,
    };
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn config_store_loads_missing_file_as_default() {
        let temp_dir = TempDir::new().unwrap();
        let store = ConfigStore::open(temp_dir.path().join("config.toml"));

        let config = store.load().unwrap();

        assert!(config.models.is_empty());
    }

    #[test]
    fn config_store_add_model_preserves_existing_fields() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("config.toml");
        fs::write(
            &path,
            r#"
version = 1

[defaults.model_fit]
ctx_size = 8192

[[models]]
model = "Qwen3-4B-Q4_K_M"
ctx_size = 4096
"#,
        )
        .unwrap();
        let store = ConfigStore::open(&path);

        let models = store.add_model_ref("  org/model-GGUF:Q5_K_M  ").unwrap();

        assert_eq!(
            models,
            vec![
                "Qwen3-4B-Q4_K_M".to_string(),
                "org/model-GGUF:Q5_K_M".to_string()
            ]
        );
        let raw = fs::read_to_string(&path).unwrap();
        assert!(raw.contains("[defaults.model_fit]"));
        assert!(raw.contains("ctx_size = 4096"));
        assert_eq!(raw.matches("org/model-GGUF:Q5_K_M").count(), 1);
    }

    #[test]
    fn config_store_save_validates_before_writing() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("config.toml");
        let store = ConfigStore::open(&path);
        let config = MeshConfig {
            version: Some(2),
            ..MeshConfig::default()
        };

        let err = store.save(&config).unwrap_err().to_string();

        assert!(err.contains("unsupported config version"));
        assert!(!path.exists());
    }

    #[test]
    fn config_store_update_writes_local_serving_node_without_callers_writing_toml() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("config.toml");
        let store = ConfigStore::open(&path);

        let config = store
            .update(|config| {
                config.configure_local_serving_node(LocalServingNodeConfig {
                    model: "Qwen/Qwen3-8B-GGUF:Q4_K_M".into(),
                    runtime: Some(ModelRuntimeKind::Metal),
                    device: Some("metal:0".into()),
                    context_size: Some(8192),
                    parallel: Some(2),
                    owner_control_bind: Some("127.0.0.1:0".parse().unwrap()),
                    gpu_assignment: Some(GpuAssignment::Auto),
                    ..LocalServingNodeConfig::default()
                })?;
                config
                    .upsert_model("Qwen/Qwen3-8B-GGUF:Q4_K_M")?
                    .max_tokens(1024)
                    .temperature(0.2);
                Ok(())
            })
            .unwrap();

        assert_eq!(config.models.len(), 1);
        assert_eq!(
            config.models[0]
                .hardware
                .as_ref()
                .and_then(|hardware| hardware.model_runtime),
            Some(ModelRuntimeKind::Metal)
        );
        let raw = fs::read_to_string(path).unwrap();
        assert!(raw.contains("model_runtime = \"metal\""));
        assert!(raw.contains("ctx_size = 8192"));
        assert!(raw.contains("temperature = 0.2"));
    }

    #[test]
    fn config_editor_updates_plugins_without_callers_writing_toml() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("config.toml");
        let store = ConfigStore::open(&path);

        let config = store
            .update(|config| {
                config.enable_builtin_plugin("telemetry")?;
                config
                    .upsert_plugin("endpoint-plugin")?
                    .enabled(true)
                    .url("http://localhost:8000/v1");
                config.upsert_external_plugin("custom-tool", "mesh-tool", ["--serve"])?;
                Ok(())
            })
            .unwrap();

        assert_eq!(config.plugins.len(), 3);
        assert_eq!(
            config
                .plugins
                .iter()
                .find(|plugin| plugin.name == "endpoint-plugin")
                .and_then(|plugin| plugin.url.as_deref()),
            Some("http://localhost:8000/v1")
        );
        assert!(fs::read_to_string(path).unwrap().contains("[[plugin]]"));
    }

    #[test]
    fn parse_config_toml_rejects_unknown_runtime_kind() {
        let err = parse_config_toml(
            r#"
version = 1

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
model_runtime = "bogus"
"#,
        )
        .unwrap_err();

        assert!(format!("{err:#}").contains("unknown variant"));
    }

    #[test]
    fn parse_config_toml_accepts_mixed_case_runtime_kind() {
        let config = parse_config_toml(
            r#"
version = 1

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
model_runtime = "Metal"
"#,
        )
        .unwrap();

        assert_eq!(
            config.models[0]
                .hardware
                .as_ref()
                .and_then(|hardware| hardware.model_runtime),
            Some(ModelRuntimeKind::Metal)
        );
    }

    #[test]
    fn runtime_model_target_reconciliation_deserializes_from_toml() {
        let config = parse_config_toml(
            r#"
version = 1

[runtime]
reconcile_model_targets = true
reconcile_model_target_demand_upgrades = true
model_target_demand_upgrade_min_requests = 4
model_target_demand_upgrade_max_age_secs = 900
"#,
        )
        .unwrap();

        assert!(config.runtime.reconcile_model_targets);
        assert!(config.runtime.reconcile_model_target_demand_upgrades);
        assert_eq!(config.runtime.model_target_demand_upgrade_min_requests, 4);
        assert_eq!(config.runtime.model_target_demand_upgrade_max_age_secs, 900);
    }

    #[test]
    fn nested_hardware_device_does_not_serialize_as_legacy_gpu_id() {
        let config = parse_config_toml(
            r#"
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
device = "cuda:0"
"#,
        )
        .unwrap();

        let toml = super::config_to_toml(&config).unwrap();

        assert!(toml.contains("device = \"cuda:0\""));
        assert!(!toml.contains("gpu_id"));
        parse_config_toml(&toml).unwrap();
    }

    #[test]
    fn explicit_legacy_gpu_id_still_serializes_for_legacy_round_trip() {
        let config = parse_config_toml(
            r#"
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen3-8B-Q4_K_M"
gpu_id = "pci:0000:65:00.0"
"#,
        )
        .unwrap();

        let toml = super::config_to_toml(&config).unwrap();

        assert!(toml.contains("gpu_id = \"pci:0000:65:00.0\""));
        parse_config_toml(&toml).unwrap();
    }
}
