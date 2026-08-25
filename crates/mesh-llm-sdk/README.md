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
mesh-llm-sdk = "0.76.0-rc7"
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
mesh-llm-sdk = { version = "0.76.0-rc7", features = ["serving"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
anyhow = "1"
```

```rust,no_run
use mesh_llm_sdk::MeshNode;

// Complete the explicit version-check/install flow below before starting.

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

## Packaged Provider Runtimes

Host-capable SDK applications can carry executable provider runtimes in their
own resources and pass those resource paths through typed configuration. The
shared host resolver still verifies the manifest, payload hashes, platform,
protocol, signature policy, and lifecycle; the SDK does not interpret an Apple
bundle or bind Foundation Models itself.

On Apple silicon with macOS Golden Gate, package the experimental sidecar and
point the embedded host at either the bundle itself or its parent directory:

```rust,no_run
use mesh_llm_sdk::MeshNode;

let node = MeshNode::builder()
    .serve()
    .provider_runtime_root(
        "/Applications/MyApp.app/Contents/Resources/provider-runtimes/apple",
    )
    .start()
    .await?;

let models = node.openai_client().models().await?;
node.shutdown().await?;
```

Provider-only embedded serving does not require a Skippy native runtime. Add a
native runtime only when the same embedded host also loads GGUF models through
`.model(...)`. Embedded provider discovery deliberately ignores
`MESH_LLM_PROVIDER_RUNTIME_BUNDLE_DIR`,
`MESH_LLM_PROVIDER_RUNTIME_INDEX`, and the provider download flag: SDK carriers
must supply roots, an optional release manifest/cache, and download permission
through the builder. This avoids process-global discovery collisions when more
than one SDK host exists in an application.

The complete Golden Gate example includes a versioned completion and tool call:

```bash
just apple::rust-sdk
```

## Check and Install the Required Native Runtime

Embedded serving requires a native runtime matching both the exact
`mesh-llm-sdk` release and its linked Skippy ABI. Re-run this check whenever the
SDK is upgraded: `CURRENT_MESH_VERSION` changes with the crate release, so a
runtime cached for the previous SDK version is not sufficient.

Enable `serving` to use native-runtime cache and install APIs:

```toml
[dependencies]
mesh-llm-sdk = { version = "0.76.0-rc7", features = ["serving"] }
```

```rust,no_run
use mesh_llm_sdk::native_runtime::{
    CURRENT_MESH_VERSION, NativeRuntimeInstallOptions, RuntimeSelection,
    current_skippy_abi_version, install_native_runtime, native_runtime_versions_match_current_sdk,
};
use mesh_llm_sdk::{MeshNode, initialize_host_runtime};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let required_abi = current_skippy_abi_version();
    let outcome = install_native_runtime(NativeRuntimeInstallOptions {
        mesh_version: CURRENT_MESH_VERSION.to_string(),
        skippy_abi_version: Some(required_abi),
        selection: RuntimeSelection::Recommended,
        ..Default::default()
    })
    .await?;
    let runtime = outcome.runtime;

    anyhow::ensure!(
        native_runtime_versions_match_current_sdk(
            &runtime.mesh_version,
            &runtime.manifest.runtime.skippy_abi,
        ),
        "native runtime does not match this SDK build"
    );
    runtime.load_plan()?;

    println!("runtime: {}", runtime.path.display());
    initialize_host_runtime().await?;

    let node = MeshNode::builder().serve().start().await?;
    node.shutdown().await?;
    Ok(())
}
```

`install_native_runtime` is the explicit network/install step. It resolves the
recommended runtime against the current host profile, MeshLLM version, and
Skippy ABI, returning the compatible cached runtime when already installed or
installing it otherwise. The `load_plan` check verifies its declared libraries
are present before startup. Embedded node startup only loads a compatible cached
runtime; it never downloads one. If the cache has no matching runtime, startup
reports the required MeshLLM version, Skippy ABI, cache directory, and the
install API to call before retrying.
