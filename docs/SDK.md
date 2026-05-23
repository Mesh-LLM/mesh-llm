# MeshLLM SDK Usage Guide

MeshLLM exposes two SDK roles across Rust, Swift, Kotlin, and Node.js:

- `Client` connects to an existing mesh and runs inference.
- `Node` includes the client role and adds local model management plus serving
  load/unload.

The SDK is split into two parts:

- **Language SDKs** provide the public API: Rust `mesh-llm-api-client` for
  client-only apps, Rust `mesh-llm-api` for node/serving apps, Swift `MeshLLM`,
  Kotlin `ai.meshllm`, and Node.js `@meshllm/sdk`.
- **Native runtime artifacts** provide local serving for a specific
  platform/backend, such as macOS Metal or Linux CUDA.

Client-only mesh inference only needs the language SDK. Local serving also
needs a matching native runtime artifact or an embedded Rust `ServingController`.

## Install

### Rust

Add the Rust SDK crate:

```toml
[dependencies]
mesh-llm-api-client = "0.66.0" # connect to an existing mesh
mesh-llm-api = "0.66.0"
```

For client-only mesh inference, depend on `mesh-llm-api-client`. For model
management and serving, depend on `mesh-llm-api`.

For local serving in a Rust app, the node must be built with a
`ServingController`. The plain `mesh-llm-api` crate intentionally does not pick
a Metal, CUDA, ROCm, Vulkan, or CPU runtime for you.

### Swift

Add the repo Swift package from a tagged release:

```swift
dependencies: [
    .package(url: "https://github.com/Mesh-LLM/mesh-llm", from: "0.66.0"),
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
    "@meshllm/sdk": "0.66.0"
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
use mesh_llm_api::{InviteToken, MeshNode, OwnerKeypair};

let owner = OwnerKeypair::generate();
let invite = "your-invite-token".parse::<InviteToken>()?;

let node = MeshNode::builder()
    .identity(owner)
    .join(invite)
    .build()?;
```

Model management APIs are available on `MeshNode` without local serving:

```rust
use mesh_llm_api::{DownloadOptions, ModelSearchQuery};

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
use mesh_llm_api::{DevicePolicy, LoadModelOptions, MeshNode, UnloadModelOptions};
use std::sync::Arc;

let controller: Arc<dyn mesh_llm_api::ServingController> = build_controller();

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
This is intentional: `mesh-llm-api` is platform-neutral and does not silently
choose a native backend.

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

Swift and Kotlin load MeshLLM through `libmeshllm_ffi`, not through a public
`libllama` contract. Backend-specific llama.cpp builds are an implementation
detail of the native runtime artifact.

Native runtime artifacts use this layout:

```text
meshllm-native-<platform>-<flavor>/
  manifest.json
  README.md
  lib/
    libmeshllm_ffi.{dylib|so}
```

The manifest records the SDK version, target triple, backend flavor, library
checksum, llama.cpp upstream SHA, patched SHA, and patch digest. SDK loaders
verify the library checksum before loading the dynamic library.

Baseline artifact names:

| Artifact directory | Target | Backend |
|---|---|---|
| `meshllm-native-darwin-aarch64-metal` | `aarch64-apple-darwin` | Metal |
| `meshllm-native-darwin-aarch64-cpu` | `aarch64-apple-darwin` | CPU |
| `meshllm-native-linux-x86_64-cpu` | `x86_64-unknown-linux-gnu` | CPU |
| `meshllm-native-linux-x86_64-cuda` | `x86_64-unknown-linux-gnu` | CUDA |
| `meshllm-native-linux-x86_64-vulkan` | `x86_64-unknown-linux-gnu` | Vulkan |
| `meshllm-native-linux-x86_64-rocm` | `x86_64-unknown-linux-gnu` | ROCm/HIP |
| `meshllm-native-windows-x86_64-cpu` | `x86_64-pc-windows-msvc` | CPU |
| `meshllm-native-windows-x86_64-cuda` | `x86_64-pc-windows-msvc` | CUDA |
| `meshllm-native-windows-x86_64-vulkan` | `x86_64-pc-windows-msvc` | Vulkan |
| `meshllm-native-windows-x86_64-rocm` | `x86_64-pc-windows-msvc` | ROCm/HIP |

CUDA and ROCm artifacts may include hardware-specific suffixes such as
`cuda-sm80` or `rocm-gfx1100` when `LLAMA_STAGE_CUDA_ARCHITECTURES` or
`LLAMA_STAGE_AMDGPU_TARGETS` is set.

Build and package one flavor:

```bash
scripts/package-native-sdk.sh \
  --build \
  --backend metal \
  --target aarch64-apple-darwin \
  --out dist/native-sdk
```

Package an already-built `mesh-llm-ffi` library:

```bash
scripts/package-native-sdk.sh \
  --backend cpu \
  --target x86_64-unknown-linux-gnu
```

Verify produced artifacts:

```bash
scripts/verify-native-sdk-package.sh dist/native-sdk/*.tar.gz
```

## Selecting a Runtime From Cargo

Native runtime crates are generated from verified native runtime artifacts:

```bash
scripts/package-native-sdk-crate.sh \
  dist/native-sdk/meshllm-native-darwin-aarch64-metal.tar.gz
```

Generated crates contain the native runtime under `native/`, copy it into
Cargo's `OUT_DIR/native` during the crate build, and use:

```toml
links = "meshllm_native_runtime"
```

Cargo exposes these paths to dependent build scripts:

```text
DEP_MESHLLM_NATIVE_RUNTIME_ARTIFACT_ID
DEP_MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR
DEP_MESHLLM_NATIVE_RUNTIME_MANIFEST
DEP_MESHLLM_NATIVE_RUNTIME_LIB_DIR
DEP_MESHLLM_NATIVE_RUNTIME_LIBRARY
```

Rust applications that want Cargo to select a runtime should depend on exactly
one runtime crate for each target:

```toml
[target.'cfg(all(target_os = "macos", target_arch = "aarch64"))'.dependencies]
meshllm-native-darwin-aarch64-metal = "0.66.0"

[target.'cfg(all(target_os = "linux", target_arch = "x86_64"))'.dependencies]
meshllm-native-linux-x86-64-cpu = "0.66.0"
```

Because all runtime crates share `links = "meshllm_native_runtime"`, Cargo
rejects selecting more than one for the same build. Apps that need to ship
multiple backend choices in one installer should package multiple verified
tarball artifacts explicitly and choose between those artifact directories at
runtime.

An app build script can copy the selected artifact into its final bundle,
installer, container image, or package resource directory:

```rust
use std::{env, fs, path::PathBuf};

fn main() {
    let artifact_dir = PathBuf::from(
        env::var("DEP_MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR")
            .expect("meshllm native runtime dependency"),
    );
    let package_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"))
        .join("meshllm-native");
    fs::create_dir_all(&package_dir).expect("create native runtime output dir");
    // Copy artifact_dir recursively into package_dir with the app's packaging helper.
}
```

At runtime, set one of these environment variables or pass the artifact
directory directly to the SDK resolver:

```text
MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR
MESHLLM_NATIVE_RUNTIME_DIR
MESH_SDK_NATIVE_RUNTIME_DIR
```

## Examples

### Swift macOS Example

```bash
./sdk/swift/scripts/build-xcframework.sh
scripts/package-native-sdk.sh \
  --backend metal \
  --target aarch64-apple-darwin \
  --out dist/native-sdk

MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR=dist/native-sdk/meshllm-native-darwin-aarch64-metal \
MESH_SDK_MODEL_REF=Qwen2.5-3B-Instruct-Q4_K_M \
swift run --package-path sdk/swift/example/MeshExampleApp
```

Useful environment variables:

| Variable | Meaning |
|---|---|
| `MESH_SDK_MODEL_REF` | Catalog, Hugging Face, or local model reference to download/load. |
| `MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR` | Verified `meshllm-native-*` artifact directory for local serving. |
| `MESH_SDK_CACHE_DIR` | Hugging Face cache location. |
| `MESH_SDK_RUNTIME_DIR` | Runtime scratch directory. |
| `MESH_SDK_SKIP_DOWNLOAD=1` | Skip download when the model is already installed. |
| `MESH_SDK_PROMPT` | Prompt text for the local inference request. |

### Kotlin JVM Example

```bash
scripts/package-native-sdk.sh \
  --backend metal \
  --target aarch64-apple-darwin \
  --out dist/native-sdk

MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR=dist/native-sdk/meshllm-native-darwin-aarch64-metal \
MESH_SDK_MODEL_REF=Qwen2.5-3B-Instruct-Q4_K_M \
./gradlew --no-daemon run -p sdk/kotlin/example/example-jvm
```

### Node.js Example

```bash
cd sdk/node
npm run build:native
cd ../..

MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR=dist/native-sdk/meshllm-native-linux-x86_64-cuda \
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
scripts/verify-native-sdk-package.sh dist/native-sdk/*.tar.gz
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
