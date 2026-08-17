#[test]
fn legacy_manual_model_launch_fields_still_validate() {
    let config: MeshConfig = toml::from_str(
        r#"
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen/Qwen3-8B-GGUF:Q4_K_M"
gpu_id = "pci:0000:00:00.0"
ctx_size = 8192
batch = 256
ubatch = 128
cache_type_k = "q8_0"
cache_type_v = "q8_0"
flash_attention = "enabled"
"#,
    )
    .expect("config should parse before validation");

    let diagnostics = validate_config_diagnostics(&config);
    assert!(
        diagnostics.is_empty(),
        "legacy manual launch fields should still validate: {diagnostics:?}"
    );
    validate_config(&config).expect("legacy manual launch fields should remain valid");
}
