# mesh-llm-sdk

`mesh-llm-sdk` is the public Rust SDK facade for Mesh LLM applications.

The default `client` feature intentionally depends only on publishable SDK
client crates:

- `mesh-llm-api-client` for client-side mesh discovery and request APIs

Client requests use direct mesh transport by default, so SDK consumers do not
need a local OpenAI `/v1` HTTP listener. Applications that intentionally want
to call an existing HTTP endpoint can opt in with the explicit
`ClientBuilder::with_openai_http_transport(...)` method.

Native runtime install/update APIs are exposed by the `serving` feature because
they are only needed by applications that manage local in-process serving.
Native runtimes are release artifacts selected and installed at runtime; Cargo
does not build them from source as part of SDK compilation. Runtime artifacts
are fetched from Mesh LLM release manifests by default, but compatibility is
checked against the exact Skippy ABI version.

## Client Transport Example

```toml
[dependencies]
mesh-llm-sdk = "0.68.0"
```

```rust,no_run
use mesh_llm_sdk::{ClientBuilder, InviteToken, OwnerKeypair};

let owner = OwnerKeypair::generate();
let invite = std::env::var("MESH_INVITE_TOKEN")?.parse::<InviteToken>()?;

let mut client = ClientBuilder::new(owner, invite)
    .with_direct_mesh_transport()
    .build()?;

client.join().await?;
let models = client.list_models().await?;
client.disconnect().await;
```

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

Enable `serving` to use native-runtime install/update APIs:

```toml
[dependencies]
mesh-llm-sdk = { version = "0.68.0", features = ["serving"] }
```

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
