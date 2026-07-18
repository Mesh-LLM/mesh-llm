use super::*;
use crate::plugin::{
    GpuAssignment, GpuConfig, HardwareConfig, ModelConfigEntry, PluginConfigEntry,
};

#[test]
fn config_sync_full_config_roundtrip() {
    let config = crate::plugin::MeshConfig {
        version: Some(1),
        gpu: GpuConfig {
            assignment: GpuAssignment::Pinned,
            parallel: None,
        },
        mesh_requirements: Default::default(),
        owner_control: Default::default(),
        telemetry: Default::default(),
        defaults: None,
        runtime: Default::default(),
        models: vec![ModelConfigEntry {
            model: "Qwen3-8B.gguf".to_string(),
            mmproj: Some("mm.gguf".to_string()),
            ctx_size: Some(8192),
            gpu_id: None,
            parallel: None,
            cache_type_k: None,
            cache_type_v: None,
            batch: None,
            ubatch: None,
            flash_attention: None,
            hardware: Some(HardwareConfig {
                device: Some("pci:0000:65:00.0".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        plugins: vec![PluginConfigEntry {
            name: "demo".to_string(),
            enabled: Some(true),
            web_ui_enabled: Some(false),
            command: Some("mesh-llm".to_string()),
            args: vec!["--plugin".to_string()],
            url: None,
            settings: Default::default(),
            startup: Default::default(),
        }],
        extra: Default::default(),
    };
    let snapshot = mesh_config_to_proto(&config);
    let restored = proto_config_to_mesh(&snapshot);
    assert_eq!(restored.version, config.version);
    assert_eq!(restored.models.len(), 1);
    assert_eq!(restored.models[0].model, "Qwen3-8B.gguf");
    assert_eq!(restored.models[0].mmproj.as_deref(), Some("mm.gguf"));
    assert_eq!(restored.models[0].ctx_size, Some(8192));
    assert_eq!(
        restored.models[0].gpu_id.as_deref(),
        Some("pci:0000:65:00.0")
    );
    assert_eq!(
        restored.models[0]
            .hardware
            .as_ref()
            .and_then(|hardware| hardware.device.as_deref()),
        Some("pci:0000:65:00.0")
    );
    assert_eq!(restored.plugins.len(), 1);
    assert_eq!(restored.plugins[0].name, "demo");
    assert_eq!(restored.plugins[0].enabled, Some(true));
    assert_eq!(restored.plugins[0].web_ui_enabled, Some(false));
    assert_eq!(restored.plugins[0].command.as_deref(), Some("mesh-llm"));
    assert_eq!(restored.plugins[0].args, vec!["--plugin"]);
}
