# mesh-llm-sdk

`mesh-llm-sdk` is the public Rust SDK facade for Mesh LLM applications.

The default feature set intentionally depends only on publishable SDK crates:

- `mesh-llm-api-client` for client-side mesh discovery and request APIs
- `mesh-llm-runtime-install` for native runtime manifest resolution,
  downloads, cache management, and pruning

The SDK does not depend on `mesh-llm-host-runtime`. Native runtimes are release
artifacts selected and installed at runtime; Cargo does not build them from
source as part of SDK compilation.

Applications that need a full in-process Mesh LLM node should depend on
`mesh-llm-embedded-runtime` directly while the host runtime is being split into
publishable layers:

```toml
[dependencies]
mesh-llm-embedded-runtime = "0.68.0"
```

```rust,no_run
use mesh_llm_embedded_runtime::{EmbeddedMeshNodeConfig, start_embedded_node};

let node = start_embedded_node(
    EmbeddedMeshNodeConfig::builder()
        .serve()
        .model("unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
        .build(),
)
.await?;
```

## Native Runtime Install Example

```rust,no_run
use mesh_llm_sdk::native_runtime::{
    NativeRuntimeInstallOptions, RuntimeSelection, install_native_runtime,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let outcome = install_native_runtime(NativeRuntimeInstallOptions {
        selection: RuntimeSelection::Recommended,
        ..Default::default()
    })
    .await?;

    println!("runtime: {}", outcome.runtime.path.display());
    Ok(())
}
```
