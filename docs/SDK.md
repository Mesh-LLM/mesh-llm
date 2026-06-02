# MeshLLM SDK Usage Guide

MeshLLM exposes two SDK roles across Rust, Swift, Kotlin, and Node.js:

- `Client` connects to an existing mesh and runs inference.
- `Node` includes the client role and adds local model management plus serving
  load/unload.

The SDK is split into two parts:

- **Language SDKs** provide the public API: Rust `mesh-llm-api-client` for
  client-only apps, Rust `mesh-llm-api-server` for node/serving apps, Swift `MeshLLM`,
  Kotlin `ai.meshllm`, and Node.js `@meshllm/sdk`.
- **Native runtime artifacts** provide local serving for a specific
  platform/runtime flavor, such as macOS Metal or Linux CUDA.

Client-only mesh inference only needs the language SDK. Local serving also
needs a matching native runtime artifact or an embedded Rust `ServingController`.

## Install

### Rust

Add the Rust SDK facade crate:

```toml
[dependencies]
mesh-llm-sdk = "0.68.0"
```

The default Rust SDK features expose client-side mesh APIs and native runtime
install/cache management without depending on the full `mesh-llm-host-runtime`
application crate.

For lower-level use, depend on `mesh-llm-api-client` directly to connect to an
existing mesh, or `mesh-llm-api-server` for platform-neutral node/model
management APIs.

For local serving in a Rust app, the node must be built with a
`ServingController`. The plain `mesh-llm-api-server` crate intentionally does
not pick a Metal, CUDA, ROCm, Vulkan, or CPU runtime for you.

### Swift

Add the repo Swift package from a tagged release:

```swift
dependencies: [
    .package(url: "https://github.com/Mesh-LLM/mesh-llm", from: "0.68.0"),
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            .product(name: "MeshLLM", package: "mesh-llm"),
        ]
    ),
]
```

Tagged releases resolve the prebuilt `MeshLLMFFI.xcframework` through SwiftPM.
For local checkout development, build the XCFramework first:

```bash
./sdk/swift/scripts/build-xcframework.sh
```

### Kotlin

The Android/Kotlin package is published to this repository's GitHub Packages
Maven registry as:

```text
ai.meshllm:meshllm-android:<version>
```

Configure the Maven repository:

```kotlin
repositories {
    maven {
        url = uri("https://maven.pkg.github.com/Mesh-LLM/mesh-llm")
        credentials {
            username = providers.gradleProperty("gpr.user")
                .orElse(System.getenv("GITHUB_ACTOR"))
                .get()
            password = providers.gradleProperty("gpr.key")
                .orElse(System.getenv("GITHUB_TOKEN"))
                .get()
        }
    }
}
```

### Node.js

Install the Node package in a Node.js or Electron app:

```json
{
  "dependencies": {
    "@meshllm/sdk": "0.68.0"
  }
}
```

When building from this repository, build the native N-API addon first:

```bash
cd sdk/node
npm run build:native
```

## Node Lifecycle

Client-only use:

```text
create or load an owner keypair
create Client with an invite token
start
list mesh models
chat or responses
stop
```

Local serving use:

```text
configure a native runtime artifact
create or load an owner keypair
create Node with an invite token
search or show a model
download the model unless it is already installed
start
load the model through serving
run inference
unload the served model or served instance
stop
```

## Rust Usage

Create a client and join a mesh:

```rust
use mesh_llm_api_client::{ClientBuilder, InviteToken, OwnerKeypair};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let owner = OwnerKeypair::generate();
    let invite = "your-invite-token".parse::<InviteToken>()?;

    let mut client = ClientBuilder::new(owner, invite)
        .build()?;

    client.join().await?;

    let models = client.list_models().await?;
    println!("models: {models:#?}");

    client.disconnect().await;
    Ok(())
}
```

Create a node when the app also needs model management or local serving:

```rust
use mesh_llm_api_server::{InviteToken, MeshNode, OwnerKeypair};

let owner = OwnerKeypair::generate();
let invite = "your-invite-token".parse::<InviteToken>()?;

let node = MeshNode::builder()
    .identity(owner)
    .join(invite)
    .build()?;
```

Model management APIs are available on `MeshNode` without local serving:

```rust
use mesh_llm_api_server::{DownloadOptions, ModelSearchQuery};

let matches = node.models().search(ModelSearchQuery {
    query: "Qwen2.5 3B instruct GGUF".to_string(),
    limit: Some(10),
}).await?;

let details = node.models().show("Qwen2.5-3B-Instruct-Q4_K_M").await?;
let downloaded = node.models()
    .download(&details.model_ref, DownloadOptions)
    .await?;
```

To serve from Rust, attach a real controller:

```rust
use mesh_llm_api_server::{DevicePolicy, LoadModelOptions, MeshNode, UnloadModelOptions};
use std::sync::Arc;

let controller: Arc<dyn mesh_llm_api_server::ServingController> = build_controller();

let node = MeshNode::builder()
    .identity(owner)
    .join(invite)
    .serving_controller(controller)
    .build()?;

let served = node.serving().load(
    "Qwen2.5-3B-Instruct-Q4_K_M",
    LoadModelOptions {
        device_policy: DevicePolicy::Auto,
    },
).await?;

node.serving()
    .unload_instance(
        served.instance_id.as_deref().unwrap_or(&served.model_id),
        UnloadModelOptions::default(),
    )
    .await?;
```

If no controller is attached, `serving.load()` returns an unsupported error.
This is intentional: `mesh-llm-api-server` is platform-neutral and does not silently
choose a native runtime.

### Rust SDK facade

`mesh-llm-sdk` re-exports `mesh-llm-api-client` on its default `client`
feature:

```rust
use mesh_llm_sdk::{ClientBuilder, InviteToken, OwnerKeypair};
```

The default `client` feature also includes native-runtime install APIs for
installing and managing version-matched native runtimes:

```rust
use mesh_llm_sdk::native_runtime::{
    NativeRuntimeInstallOptions, RuntimeSelection, install_native_runtime,
};
```

SDK packages that ship the built console as package resources can enable the
optional `console` feature and use the file-backed console server without
depending on `mesh-llm-host-runtime`:

```rust
let console = mesh_llm_sdk::console::start_file_console(
    mesh_llm_sdk::console::ConsoleServerOptions {
        asset_dir: "/path/to/packaged/console/dist".into(),
        port: 0,
        listen_all: false,
    },
).await?;
```

Enable the optional `serving` feature for an in-process full node. This pulls
in the embedded runtime and host runtime with dynamic native-runtime loading,
so Cargo still resolves native runtimes as release artifacts instead of
building them implicitly from source:

```toml
[dependencies]
mesh-llm-sdk = { version = "0.68.0", features = ["serving"] }
```

Enable `serving` and `console` together when the embedded node should also
serve bundled web console assets.

## Swift Usage

Configure a native runtime before local serving:

```swift
import MeshLLM

let runtime = try NativeRuntime.prepare()
print("using \(runtime.artifactId) from \(runtime.artifactDirectory.path)")
```

Create a client and run inference:

```swift
import MeshLLM

let ownerKeypair = generateOwnerKeypairHex()
let client = try Client(
    inviteToken: InviteToken("your-invite-token"),
    ownerKeypairBytesHex: ownerKeypair
)

try await client.start()

let models = try await client.inference.listModels()
let request = ChatRequest(model: models[0].id, messages: [
    ChatMessage(role: "user", content: "Hello!")
])

for try await event in client.inference.chat(request) {
    if case .tokenDelta(_, let delta) = event {
        print(delta, terminator: "")
    }
}

await client.stop()
```

Load and unload a local model:

```swift
let node = try Node(
    inviteToken: InviteToken("your-invite-token"),
    ownerKeypairBytesHex: ownerKeypair
)

let served = try await node.serving.load(
    "Qwen2.5-3B-Instruct-Q4_K_M",
    options: LoadModelOptions(devicePolicy: .auto)
)

if let instanceId = served.instanceId {
    try await node.serving.unloadInstance(
        instanceId,
        options: UnloadModelOptions(drainTimeoutMs: 1_000, force: false)
    )
} else {
    try await node.serving.unloadModel(
        served.modelId,
        options: UnloadModelOptions(drainTimeoutMs: 1_000, force: false)
    )
}
```

## Kotlin Usage

Configure the native runtime before any generated UniFFI symbol is used:

```kotlin
import ai.meshllm.NativeRuntime

val runtime = NativeRuntime.configure()
println("using ${runtime.artifactId} from ${runtime.artifactDir}")
```

Create a client:

```kotlin
import ai.meshllm.Client
import ai.meshllm.InviteToken
import uniffi.mesh_ffi.generateOwnerKeypairHex

val ownerKeypair = generateOwnerKeypairHex()
val client = Client(InviteToken("your-invite-token"), ownerKeypair)

client.start()
val models = client.inference.listModels()
```

Load, infer, and unload:

```kotlin
val served = node.serving.load(
    "Qwen2.5-3B-Instruct-Q4_K_M",
    LoadModelOptions(DevicePolicy.Auto),
)

try {
    val selected = node.inference.listModels().first { it.id == served.modelId }
    node.inference.chat(
        ChatRequest(selected.id, listOf(ChatMessage("user", "hello"))),
    ) { event -> println(event) }
} finally {
    val target = served.instanceId?.let { UnloadTarget.Instance(it) }
        ?: UnloadTarget.Model(served.modelId)
    node.serving.unload(
        target,
        UnloadModelOptions(drainTimeoutMs = 1_000UL, force = false),
    )
    node.stop()
}
```

## Node.js Usage

Create a client:

```js
const { Client, generateOwnerKeypairHex } = require('@meshllm/sdk')

const client = Client.create({
  ownerKeypairHex: generateOwnerKeypairHex(),
  inviteToken: 'your-invite-token'
})

await client.start()
const models = await client.inference.listModels()
await client.stop()
```

Use local serving by enabling serving and packaging a matching native runtime
artifact:

```js
const node = Node.create({
  ownerKeypairHex,
  inviteToken,
  servingEnabled: true,
  cacheDir: process.env.MESH_SDK_CACHE_DIR,
  runtimeDir: process.env.MESH_SDK_RUNTIME_DIR
})

await node.start()
await node.models.download('Qwen2.5-3B-Instruct-Q4_K_M')
const served = await node.serving.load('Qwen2.5-3B-Instruct-Q4_K_M', {
  devicePolicy: 'auto'
})
const result = await node.inference.chat({
  model: served.modelId,
  messages: [{ role: 'user', content: 'hello' }]
})
console.log(result.content)
await node.serving.unloadModel(served.modelId)
await node.stop()
```

## Native Runtime Artifacts

The accepted packaging direction is documented in
[design/NATIVE_RUNTIMES.md](design/NATIVE_RUNTIMES.md). In short: native
runtimes are release artifacts, not implicit Cargo builds, and the native
runtime version must exactly match the MeshLLM version that loads it.

Native runtime artifacts use this layout:

```text
meshllm-native-runtime-<platform>-<flavor>/
  manifest.json
  README.md
  lib/
    libllama.{dylib|so|dll}
    libggml*.{dylib|so|dll}
```

The manifest records the MeshLLM version, target triple, runtime flavor,
Skippy ABI metadata, load-order library paths, release URL, checksum, and
optional signature metadata. SDK loaders reject runtimes whose `mesh_version`
does not exactly match the running MeshLLM version.

Baseline artifact names:

| Artifact directory | Target | Flavor |
|---|---|---|
| `meshllm-native-runtime-darwin-aarch64-metal` | `aarch64-apple-darwin` | Metal |
| `meshllm-native-runtime-darwin-aarch64-cpu` | `aarch64-apple-darwin` | CPU |
| `meshllm-native-runtime-linux-x86_64-cpu` | `x86_64-unknown-linux-gnu` | CPU |
| `meshllm-native-runtime-linux-x86_64-cuda` | `x86_64-unknown-linux-gnu` | CUDA |
| `meshllm-native-runtime-linux-x86_64-vulkan` | `x86_64-unknown-linux-gnu` | Vulkan |
| `meshllm-native-runtime-linux-x86_64-rocm` | `x86_64-unknown-linux-gnu` | ROCm/HIP |
| `meshllm-native-runtime-windows-x86_64-cpu` | `x86_64-pc-windows-msvc` | CPU |
| `meshllm-native-runtime-windows-x86_64-cuda` | `x86_64-pc-windows-msvc` | CUDA |
| `meshllm-native-runtime-windows-x86_64-vulkan` | `x86_64-pc-windows-msvc` | Vulkan |
| `meshllm-native-runtime-windows-x86_64-rocm` | `x86_64-pc-windows-msvc` | ROCm/HIP |

CUDA and ROCm artifacts may include hardware-specific flavor suffixes such as
`cuda-sm80`, `cuda-blackwell`, or `rocm-gfx1100` when
`LLAMA_STAGE_CUDA_ARCHITECTURES` or
`LLAMA_STAGE_AMDGPU_TARGETS` is set.

Build and package one flavor:

```bash
scripts/package-native-runtime.sh \
  --build \
  --backend metal \
  --target aarch64-apple-darwin \
  --out dist/native-runtimes
```

Verify produced artifacts:

```bash
scripts/verify-native-runtime-package.sh dist/native-runtimes/*.tar.gz
```

## Selecting a Runtime From Cargo

Cargo dependencies provide the MeshLLM Rust SDK. Native runtimes are resolved
at install or application startup from release artifacts, not built implicitly
by Cargo.

Normal online install:

```bash
mesh-llm runtime install
```

Offline or packaged install:

```bash
mesh-llm runtime install --bundle-dir path/to/meshllm-native-runtime-darwin-aarch64-metal
```

Rust SDK consumers can use the same resolver/downloader path directly:

```rust
use mesh_llm_sdk::native_runtime::{
    NativeRuntimeInstallOptions, RuntimeSelection, install_native_runtime,
};

let outcome = install_native_runtime(NativeRuntimeInstallOptions {
    selection: RuntimeSelection::Recommended,
    cache_dir: Some(app_cache_dir.join("mesh-llm-native-runtimes")),
    bundle_dirs: vec![app_resources.join("meshllm-native-runtime")],
    progress: Some(std::sync::Arc::new(|event| {
        update_progress(event.downloaded_bytes, event.total_bytes);
    })),
    ..Default::default()
})
.await?;
```

Manifest discovery order:

1. explicit manifest path
2. explicit manifest URL
3. `MESH_LLM_NATIVE_RUNTIME_MANIFEST_URL`
4. GitHub release `native-runtimes.json` for the running MeshLLM version

Generated runtime crates are not the supported distribution story for native
runtimes in this PR. The supported path is release artifacts plus the release
manifest, shared by the CLI, SDK, and autoupdater.

At runtime, set one of these environment variables or pass the artifact
directory directly to the SDK resolver for offline packages:

```text
MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR
MESHLLM_NATIVE_RUNTIME_DIR
MESH_SDK_NATIVE_RUNTIME_DIR
```

## Examples

### Swift macOS Example

```bash
./sdk/swift/scripts/build-xcframework.sh
scripts/package-native-runtime.sh \
  --backend metal \
  --target aarch64-apple-darwin \
  --out dist/native-runtimes

MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR=dist/native-runtimes/meshllm-native-runtime-darwin-aarch64-metal \
MESH_SDK_MODEL_REF=Qwen2.5-3B-Instruct-Q4_K_M \
swift run --package-path sdk/swift/example/MeshExampleApp
```

Useful environment variables:

| Variable | Meaning |
|---|---|
| `MESH_SDK_MODEL_REF` | Catalog, Hugging Face, or local model reference to download/load. |
| `MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR` | Verified `meshllm-native-runtime-*` artifact directory for local serving. |
| `MESH_SDK_CACHE_DIR` | Hugging Face cache location. |
| `MESH_SDK_RUNTIME_DIR` | Runtime scratch directory. |
| `MESH_SDK_SKIP_DOWNLOAD=1` | Skip download when the model is already installed. |
| `MESH_SDK_PROMPT` | Prompt text for the local inference request. |

### Kotlin JVM Example

```bash
scripts/package-native-runtime.sh \
  --backend metal \
  --target aarch64-apple-darwin \
  --out dist/native-runtimes

MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR=dist/native-runtimes/meshllm-native-runtime-darwin-aarch64-metal \
MESH_SDK_MODEL_REF=Qwen2.5-3B-Instruct-Q4_K_M \
./gradlew --no-daemon run -p sdk/kotlin/example/example-jvm
```

### Node.js Example

```bash
cd sdk/node
npm run build:native
cd ../..

MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR=dist/native-runtimes/meshllm-native-runtime-linux-x86_64-cuda \
MESH_SDK_MODEL_REF=Qwen2.5-3B-Instruct-Q4_K_M \
node sdk/node/example/local-inference.js
```

## Errors

Rust APIs return `MeshApiError`. Swift exposes `MeshError`. Kotlin exposes
`MeshException`.

Common categories:

| Category | Meaning |
|---|---|
| Invalid invite token | The token is empty, malformed, or cannot be accepted. |
| Invalid owner keypair | The owner identity is empty or malformed. |
| Discovery failed | Public mesh discovery failed. |
| Model management failed | Search, show, download, install, delete, cleanup, or cache inspection failed. |
| Serving failed | Serving load, unload, status, or device policy control failed. |
| Serving unsupported | The current platform/build does not provide local serving. |
| Stream failed | Streaming inference setup or delivery failed. |
| Cancelled | A request was cancelled. |

Do not treat unsupported serving as a soft fallback. If a target cannot serve
locally, surface the typed unsupported error to the caller.

## Platform Support

| Platform/package | Mesh inference | Model management | Local serving |
|---|---:|---:|---:|
| Rust SDK on macOS | yes | yes | requires an attached `ServingController` |
| Rust SDK on Linux | yes | yes | requires an attached `ServingController` |
| Swift macOS | yes | yes | yes with a matching native runtime artifact |
| Swift Mac Catalyst | yes | yes | not currently advertised |
| Swift iOS | yes | limited by app filesystem policy | no |
| Kotlin JVM macOS | yes | yes | yes with a matching native runtime artifact |
| Kotlin JVM Linux | yes | yes | yes with a matching native runtime artifact |
| Kotlin Android | yes | yes | not currently advertised |
| Node.js macOS | yes | yes | yes with a matching native runtime artifact |
| Node.js Linux | yes | yes | yes with a matching native runtime artifact |
| Node.js Windows | yes | yes | yes with a matching native runtime artifact |

## Validation Commands

Run the SDK package checks:

```bash
scripts/check-sdk-contract.sh
scripts/verify-native-runtime-package.sh dist/native-runtimes/*.tar.gz
cargo test -p mesh-llm-ffi
swift build --package-path sdk/swift/example/MeshExampleApp
./gradlew --no-daemon compileKotlin -p sdk/kotlin/example/example-jvm
node --test sdk/node/test/*.test.js
```

Run serving smoke examples with a real model:

```bash
scripts/ci-swift-sdk-smoke.sh <mesh-llm> <bin-dir> <model.gguf>
scripts/ci-kotlin-sdk-smoke.sh <mesh-llm> <bin-dir> <model.gguf>
```
