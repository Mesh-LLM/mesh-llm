# mesh-llm-config

`mesh-llm-config` owns the shared `config.toml` schema, path resolution, file
I/O, preservation-friendly edits, and validation rules used by MeshLLM.

Use this crate when an application or SDK surface needs to read or write the
same `~/.mesh-llm/config.toml` file as the CLI without depending on the full
host runtime.

This crate includes:

- typed config data structures such as `MeshConfig`, `ModelConfigEntry`,
  `GpuConfig`, `PluginConfigEntry`, and telemetry settings
- high-level authoring APIs for configuring nodes, models, and plugins without
  hand-writing TOML
- default config path resolution, including `MESH_LLM_CONFIG`
- validated typed loading through `load_config` and `ConfigStore::load`
- atomic typed saves through `ConfigStore::save`
- typed load/edit/save through `ConfigStore::update`
- TOML parse/serialize helpers used by MeshLLM control-plane payloads

## Examples

### Load the real MeshLLM config

Use `ConfigStore::default_path()` when you want the same config file the CLI
uses. It honors `MESH_LLM_CONFIG` before falling back to
`~/.mesh-llm/config.toml`.

```rust
use mesh_llm_config::ConfigStore;

let store = ConfigStore::default_path()?;
let config = store.load()?;
println!("configured models: {}", config.models.len());
```

Use `ConfigStore::open(path)` for tests, importers, or explicit config paths.

```rust
use mesh_llm_config::ConfigStore;

let store = ConfigStore::open("/tmp/mesh-config.toml");
let config = store.load()?;
```

### Configure a local serving node

Configure a local serving node from an SDK or desktop app:

```rust
use mesh_llm_config::{ConfigStore, GpuAssignment, LocalServingNodeConfig};
use mesh_llm_types::runtime::ModelRuntimeKind;

let store = ConfigStore::default_path()?;
store.update(|config| {
    config.configure_local_serving_node(LocalServingNodeConfig {
        model: "Qwen/Qwen3-8B-GGUF:Q4_K_M".into(),
        runtime: Some(ModelRuntimeKind::Metal),
        device: Some("metal:0".into()),
        context_size: Some(8192),
        parallel: Some(2),
        gpu_assignment: Some(GpuAssignment::Auto),
        owner_control_bind: Some("127.0.0.1:0".parse()?),
        ..LocalServingNodeConfig::default()
    })?;
    Ok(())
})?;
```

This writes the canonical nested config shape and validates it before replacing
the file.

### Set shared defaults

Use defaults when an app wants all configured models to inherit the same runtime,
device, context, or throughput policy.

```rust
use mesh_llm_config::{ConfigStore, GpuAssignment};
use mesh_llm_types::runtime::ModelRuntimeKind;

let store = ConfigStore::default_path()?;
store.update(|config| {
    config
        .set_version(Some(1))
        .set_gpu_assignment(GpuAssignment::Auto)
        .set_default_runtime(ModelRuntimeKind::Metal)
        .set_default_device("auto")
        .set_default_context_size(Some(8192));
    config.defaults().parallel(Some(2));
    Ok(())
})?;
```

### Add or update a model

Model refs are stored as the same strings the CLI understands, so callers can
write catalog names, Hugging Face GGUF refs, or direct local model refs without
knowing the TOML layout.

```rust
use mesh_llm_config::ConfigStore;
use mesh_llm_types::runtime::ModelRuntimeKind;

let store = ConfigStore::default_path()?;
store.update(|config| {
    config
        .upsert_model("Qwen/Qwen3-8B-GGUF:Q4_K_M")?
        .runtime(ModelRuntimeKind::Metal)
        .device("metal:0")
        .context_size(8192)
        .parallel(2)
        .cache_types("q8_0", "q4_0")
        .max_tokens(1024)
        .temperature(0.2);
    Ok(())
})?;
```

Remove a model through the same typed editor:

```rust
use mesh_llm_config::ConfigStore;

let store = ConfigStore::default_path()?;
store.update(|config| {
    config.remove_model("Qwen/Qwen3-8B-GGUF:Q4_K_M")?;
    Ok(())
})?;
```

### Configure plugins

Use plugin helpers for common cases instead of writing `[[plugin]]` tables.

```rust
use mesh_llm_config::ConfigStore;

let store = ConfigStore::default_path()?;
store.update(|config| {
    config.enable_builtin_plugin("telemetry")?;
    config.upsert_plugin("endpoint-plugin")?
        .enabled(true)
        .url("http://localhost:8000/v1")
        .connect_timeout_secs(75)
        .init_timeout_secs(90)
        .optional(true)
        .lazy_start(true);
    config.upsert_external_plugin("custom-tool", "mesh-tool", ["--serve"])?;
    Ok(())
})?;
```

### Validate imported TOML

Apps that import or receive TOML can still use the shared parser and validation
rules before deciding whether to save.

```rust
use mesh_llm_config::{config_to_toml, parse_config_toml, ConfigStore};

let imported = parse_config_toml(raw_toml)?;
let canonical_toml = config_to_toml(&imported)?;

let store = ConfigStore::default_path()?;
store.save(&imported)?;
```

### Preserve comments for narrow edits

`ConfigStore::update` is the preferred high-level API for SDKs and apps. For a
small edit to an existing user-authored file where comments and ordering matter,
use the dedicated preserving helpers.

```rust
use mesh_llm_config::ConfigStore;

let store = ConfigStore::default_path()?;
let models = store.add_model_ref("Qwen/Qwen3-8B-GGUF:Q4_K_M")?;
println!("configured models: {models:?}");

let models = store.remove_model_ref("Qwen/Qwen3-8B-GGUF:Q4_K_M")?;
println!("configured models: {models:?}");
```

Runtime interpretation should stay outside this crate. In particular, plugin
resolution, plugin process lifecycle, model serving, live config apply
revisions, and mesh control-plane behavior belong in the host runtime or SDK
layers that consume this crate.
