---
title: Rust SDK
---

# Rust SDK

Use [`mesh-llm-sdk`](https://crates.io/crates/mesh-llm-sdk) for Rust applications. The default `client` feature is lightweight; enable `serving` when the app should manage local models and run an embedded node.

## Install

Client-only application:

```toml
[dependencies]
anyhow = "1"
mesh-llm-sdk = "{{ site.sdkVersion }}"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Serving application:

```toml
[dependencies]
anyhow = "1"
mesh-llm-sdk = { version = "{{ site.sdkVersion }}", features = ["serving"] }
serde_json = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Add the `console` feature when the app also packages the web console.

## Connect as a client

The direct mesh transport avoids requiring a local OpenAI-compatible HTTP listener:

```rust,no_run
use mesh_llm_sdk::{ClientBuilder, InviteToken, OwnerKeypair};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let owner = OwnerKeypair::generate();
    let invite = std::env::var("MESH_INVITE_TOKEN")?.parse::<InviteToken>()?;
    let mut client = ClientBuilder::new(owner, invite)
        .with_direct_mesh_transport()
        .build()?;

    client.join().await?;
    let models = client.list_models().await?;
    println!("models: {}", models.len());
    // Stream chat or responses through the client inference API here.
    client.disconnect().await;
    Ok(())
}
```

For public-mesh discovery, use `select_public_mesh(PublicMeshQuery { .. })` and build the client from the selected mesh. Keep the owner keypair stable across reconnects.

## Embed local serving

Resolve or install the native runtime into an app-owned cache before starting a serving node:

```rust,no_run
use mesh_llm_sdk::native_runtime::{
    install_native_runtime, NativeRuntimeInstallOptions, RuntimeSelection,
};

let outcome = install_native_runtime(NativeRuntimeInstallOptions {
    selection: RuntimeSelection::Recommended,
    cache_dir: Some(app_cache_dir.join("mesh-llm-native-runtimes")),
    bundle_dirs: vec![app_resources.join("meshllm-native-runtime")],
    allow_download: false,
    ..Default::default()
})
.await?;
println!("runtime: {}", outcome.runtime.path.display());
```

Then embed the node. This example serves a local model and joins a public mesh; replace `auto_join_public_mesh()` with `join_token(...)` for a private mesh:

```rust,no_run
use mesh_llm_sdk::MeshNode;

let node = MeshNode::builder()
    .serve()
    .model("unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
    .auto_join_public_mesh()
    .api_port(0)
    .console_port(0)
    .start()
    .await?;

let models = node.openai_client().models().await?;
println!("served models: {models}");
node.shutdown().await?;
```

`MeshNode` also exposes `status()`, `api_base_url()`, `console_url()`, `invite_token()`, and an `openai_client()` for chat completions and responses. If the application supplies its own serving implementation, the lower-level `mesh-llm-api-server` API can bind a `ServingController`; most applications should start with `mesh-llm-sdk`.

## Console assets

Enable `console` and package the generated console directory with the application:

```rust,no_run
let console = mesh_llm_sdk::console::start_file_console(
    mesh_llm_sdk::console::ConsoleServerOptions {
        asset_dir: app_resources.join("console"),
        port: 0,
        listen_all: false,
    },
)
.await?;
println!("console: {}", console.url());
```

Keep console hosting optional so client-only applications do not carry web assets.

## Errors and shutdown

Return `MeshApiError`/`anyhow::Error` to the application boundary, report download and serving progress, and always unload a served model before shutting down when requests may still be in flight. Use an app-owned cache and avoid sharing one mutable runtime directory between concurrent nodes.
