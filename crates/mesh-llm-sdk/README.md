# mesh-llm-sdk

`mesh-llm-sdk` is the public Rust SDK facade for Mesh LLM applications.

The default feature set intentionally depends only on publishable SDK crates:

- `mesh-llm-api-client` for client-side mesh discovery and request APIs
- `mesh-llm-runtime-install` for native runtime manifest resolution,
  downloads, cache management, and pruning

Native runtimes are release artifacts selected and installed at runtime; Cargo
does not build them from source as part of SDK compilation.

## Embedded Node Example

```toml
[dependencies]
mesh-llm-sdk = { version = "0.68.0", features = ["serving"] }
```

```rust,no_run
use mesh_llm_sdk::MeshNode;

let node = MeshNode::builder()
    .serve()
    .model("unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
    .auto_join_public_mesh()
    .start()
    .await?;

let openai = node.openai_client();
let models = openai.models().await?;
let status = node.status().await?;

node.shutdown().await?;
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
