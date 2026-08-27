use mesh_llm_config::{
    ConfigDiagnosticSeverity, MeshConfig, built_in_config_settings, config_to_toml,
    validate_config_diagnostics,
};

const MANIFEST_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const MANIFEST_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

#[test]
fn typed_topology_round_trips_defaults_and_per_model_override() {
    let source = format!(
        r#"
[defaults.topology]
mode = "locked"
manifest_sha256 = "{MANIFEST_A}"

[[defaults.topology.stages]]
node = {{ hostname = "default-a.local" }}
layer_start = 0
layer_end = 20

[[defaults.topology.stages]]
node = {{ hostname = "default-b.local" }}
layer_start = 20
layer_end = 40

[[models]]
model = "meshllm/example@0123456789abcdef:model.gguf"

[models.topology]
manifest_sha256 = "{MANIFEST_B}"

[[models.topology.stages]]
node = {{ hostname = "model-a.local" }}
layer_start = 0
layer_end = 16

[[models.topology.stages]]
node = {{ hostname = "model-b.local" }}
layer_start = 16
layer_end = 40
"#
    );
    let config: MeshConfig = toml::from_str(&source).expect("typed topology should parse");

    let serialized = config_to_toml(&config).expect("typed topology should serialize");

    assert!(serialized.contains("[defaults.topology]"));
    assert!(serialized.contains("[[models.topology.stages]]"));
    assert!(serialized.contains(MANIFEST_B));
}

#[test]
fn topology_node_selector_requires_exactly_one_stable_path() {
    let config: MeshConfig = toml::from_str(&format!(
        r#"
[[models]]
model = "meshllm/example@0123456789abcdef:model.gguf"

[models.topology]
mode = "locked"
manifest_sha256 = "{MANIFEST_A}"

[[models.topology.stages]]
node = {{ endpoint_id = "endpoint-a", hostname = "worker.local" }}
layer_start = 0
layer_end = 20

[[models.topology.stages]]
node = {{ }}
layer_start = 20
layer_end = 40
"#
    ))
    .expect("invalid selectors should remain available to structured validation");

    let paths = validate_config_diagnostics(&config)
        .into_iter()
        .filter(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error)
        .filter_map(|diagnostic| diagnostic.path.map(|path| path.render()))
        .collect::<std::collections::BTreeSet<_>>();

    assert!(paths.contains("models[0].topology.stages[0].node"));
    assert!(paths.contains("models[0].topology.stages[1].node"));
}

#[test]
fn topology_ranges_and_manifest_are_strictly_validated() {
    let config: MeshConfig = toml::from_str(
        r#"
[[models]]
model = "mutable/catalog-model"

[models.topology]
mode = "locked"
manifest_sha256 = "not-a-sha256"

[[models.topology.stages]]
node = { hostname = "worker-a.local" }
layer_start = 1
layer_end = 20

[[models.topology.stages]]
node = { hostname = "worker-b.local" }
layer_start = 21
layer_end = 21
"#,
    )
    .expect("invalid topology should parse for structured validation");

    let paths = validate_config_diagnostics(&config)
        .into_iter()
        .filter(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error)
        .filter_map(|diagnostic| diagnostic.path.map(|path| path.render()))
        .collect::<std::collections::BTreeSet<_>>();

    assert!(paths.contains("models[0].topology.manifest_sha256"));
    assert!(paths.contains("models[0].model"));
    assert!(paths.contains("models[0].topology.stages[0].layer_start"));
    assert!(paths.contains("models[0].topology.stages[1].layer_start"));
    assert!(paths.contains("models[0].topology.stages[1].layer_end"));
}

#[test]
fn topology_schema_exports_typed_leaf_paths() {
    let paths = built_in_config_settings()
        .into_iter()
        .map(|setting| setting.path.render())
        .collect::<std::collections::BTreeSet<_>>();

    for path in [
        "defaults.topology.mode",
        "defaults.topology.manifest_sha256",
        "defaults.topology.stages",
        "models.<model-ref>.topology.mode",
        "models.<model-ref>.topology.manifest_sha256",
        "models.<model-ref>.topology.stages",
    ] {
        assert!(paths.contains(path), "missing topology schema path {path}");
    }
}
