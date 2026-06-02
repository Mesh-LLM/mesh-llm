# MeshLLM SDK Usage Guide

MeshLLM exposes two SDK roles across Rust, Swift, Kotlin, and Node.js:

- `Client` connects to an existing mesh and runs inference.
- `Node` includes the client role and adds local model management plus serving
  load/unload.

The SDK is split into two parts:

- **Language SDKs** provide the public API: Rust `mesh-llm-sdk`, Swift
  `MeshLLM`, Kotlin `ai.meshllm`, and Node.js `@meshllm/sdk`.
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

The default Rust SDK feature exposes client-side mesh APIs without depending on
the full `mesh-llm-host-runtime` application crate or the native-runtime
installer.

Use `mesh-llm-sdk` features to opt into larger surfaces:

| Feature | Surface |
|---|---|
| `client` | client-only mesh inference, enabled by default |
| `node` | platform-neutral node/model management APIs |
| `serving` | full in-process serving plus native runtime install, cache, and prune APIs |
| `console` | embedded console server facade for packaged console assets |

For lower-level use, `mesh-llm-api-client` and `mesh-llm-api-server` remain
implementation crates, but Rust application code should prefer the SDK facade.

Published SDK packages that advertise console support include the built web
console as package resources. Source checkouts can regenerate those resources
with:

```bash
scripts/package-sdk-console-assets.sh --sdk all
scripts/verify-sdk-console-assets.sh --sdk all
```

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
resolve or install a native runtime
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

## SDK Examples

Each SDK supports two roles:

| Role | Public mesh mode | Private mesh mode |
|---|---|---|
| Client | Discover/select a public mesh, then connect and infer. | Join a caller-provided invite token, then infer. |
| Serving | Resolve a native runtime, join a public mesh invite selected by the app, load a local model, then infer. | Resolve a native runtime, join a private invite token, load a local model, then infer. |

For client-only apps, public mesh examples use discovery where the language SDK
exports it. For local serving, examples use concrete invite tokens because the
serving-enabled node needs to join the selected mesh while also attaching the
local serving controller.

### Rust Client

`Cargo.toml`:

```toml
[dependencies]
anyhow = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread", "sync"] }
mesh-llm-sdk = "0.68.0"
```

Public mesh client:

```rust
use mesh_llm_sdk::{
    ClientBuilder, OwnerKeypair, PublicMeshQuery, select_public_mesh,
};

let owner = OwnerKeypair::generate();
let public_mesh = select_public_mesh(PublicMeshQuery {
    model: Some("Qwen3".to_string()),
    ..Default::default()
})
.await?;

let mut client = ClientBuilder::from_public_mesh(owner, &public_mesh)?
    .with_direct_mesh_transport()
    .build()?;
client.join().await?;

let models = client.list_models().await?;
let model = models.first().expect("public mesh has models").id.clone();
let reply = chat_once(&client, model, "Say hello from the public mesh.").await?;
println!("{reply}");

client.disconnect().await;
```

Private mesh client:

```rust
use mesh_llm_sdk::{ClientBuilder, InviteToken, OwnerKeypair};

let owner = OwnerKeypair::generate();
let invite = std::env::var("MESH_PRIVATE_INVITE")?.parse::<InviteToken>()?;

let mut client = ClientBuilder::new(owner, invite)
    .with_direct_mesh_transport()
    .build()?;
client.join().await?;

let models = client.list_models().await?;
let model = models.first().expect("private mesh has models").id.clone();
let reply = chat_once(&client, model, "Say hello from the private mesh.").await?;
println!("{reply}");

client.disconnect().await;
```

Shared Rust client inference helper:

```rust
use mesh_llm_sdk::events::{Event, EventListener};
use mesh_llm_sdk::{ChatMessage, ChatRequest, MeshClient};
use std::sync::Arc;
use tokio::sync::mpsc;

struct Listener {
    tx: mpsc::UnboundedSender<Event>,
}

impl EventListener for Listener {
    fn on_event(&self, event: Event) {
        let _ = self.tx.send(event);
    }
}

async fn chat_once(client: &MeshClient, model: String, prompt: &str) -> anyhow::Result<String> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let request_id = client.chat(
        ChatRequest {
            model,
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
        },
        Arc::new(Listener { tx }),
    ).0;

    let mut output = String::new();
    while let Some(event) = rx.recv().await {
        match event {
            Event::TokenDelta { request_id: id, delta } if id == request_id => {
                output.push_str(&delta);
            }
            Event::Completed { request_id: id } if id == request_id => return Ok(output),
            Event::Failed { request_id: id, error } if id == request_id => anyhow::bail!(error),
            _ => {}
        }
    }
    anyhow::bail!("request ended before completion")
}
```

### Rust Serving

`Cargo.toml`:

```toml
[dependencies]
anyhow = "1"
serde_json = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
mesh-llm-sdk = { version = "0.68.0", features = ["serving"] }
```

Resolve the recommended native runtime once before starting a serving node:

```rust
use mesh_llm_sdk::native_runtime::{
    NativeRuntimeInstallOptions, RuntimeSelection, install_native_runtime,
};

let runtime = install_native_runtime(NativeRuntimeInstallOptions {
    selection: RuntimeSelection::Recommended,
    ..Default::default()
})
.await?;
println!("runtime: {}", runtime.runtime.path.display());
```

Public mesh serving:

```rust
use mesh_llm_sdk::MeshNode;

let model_ref = "unsloth/Qwen3-0.6B-GGUF:Q4_K_M";
let public_invite = std::env::var("MESH_PUBLIC_INVITE")?;

let node = MeshNode::builder()
    .serve()
    .model(model_ref)
    .join_token(public_invite)
    .start()
    .await?;

let reply = embedded_chat_once(&node, model_ref, "Say hello from a public serving node.").await?;
println!("{reply}");
node.shutdown().await?;
```

Private mesh serving:

```rust
let model_ref = "unsloth/Qwen3-0.6B-GGUF:Q4_K_M";
let private_invite = std::env::var("MESH_PRIVATE_INVITE")?;

let node = MeshNode::builder()
    .serve()
    .model(model_ref)
    .join_token(private_invite)
    .start()
    .await?;

let reply = embedded_chat_once(&node, model_ref, "Say hello from a private serving node.").await?;
println!("{reply}");
node.shutdown().await?;
```

Shared Rust serving inference helper:

```rust
use mesh_llm_sdk::MeshNode;
use serde_json::{Value, json};

async fn embedded_chat_once(node: &MeshNode, model: &str, prompt: &str) -> anyhow::Result<String> {
    let response = node.openai_client().chat_completions(json!({
        "model": model,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": 64,
        "temperature": 0,
    })).await?;

    Ok(response
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string())
}
```

### Rust SDK Facade

`mesh-llm-sdk` re-exports `mesh-llm-api-client` on its default `client`
feature:

```rust
use mesh_llm_sdk::{ClientBuilder, InviteToken, OwnerKeypair};
```

Enable the optional `serving` feature to install and manage version-matched
native runtimes for local in-process serving:

```toml
[dependencies]
mesh-llm-sdk = { version = "0.68.0", features = ["serving"] }
```

```rust
use mesh_llm_sdk::native_runtime::{
    NativeRuntimeInstallOptions, RuntimeSelection, install_native_runtime,
};
```

Rust applications that package the built console can enable the optional
`console` feature and use the file-backed console server without depending on
`mesh-llm-host-runtime`:

```rust
let console = mesh_llm_sdk::console::start_file_console(
    mesh_llm_sdk::console::ConsoleServerOptions {
        asset_dir: "/path/to/packaged/console".into(),
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

### Node.js Client

Public mesh client:

```js
const { Client, generateOwnerKeypairHex } = require('@meshllm/sdk')

const client = Client.create({
  ownerKeypairHex: generateOwnerKeypairHex(),
  inviteToken: process.env.MESH_PUBLIC_INVITE
})

await client.start()
const models = await client.inference.listModels()
const result = await client.inference.chat({
  model: models[0].id,
  messages: [{ role: 'user', content: 'Say hello from a public mesh.' }]
})
console.log(result.content)
await client.stop()
```

Private mesh client:

```js
const { Client, generateOwnerKeypairHex } = require('@meshllm/sdk')

const client = Client.create({
  ownerKeypairHex: generateOwnerKeypairHex(),
  inviteToken: process.env.MESH_PRIVATE_INVITE
})

await client.start()
const models = await client.inference.listModels()
const result = await client.inference.chat({
  model: models[0].id,
  messages: [{ role: 'user', content: 'Say hello from a private mesh.' }]
})
console.log(result.content)
await client.stop()
```

Node.js public discovery helpers are not currently exported by `@meshllm/sdk`;
use a public invite token selected by your app or service.

### Node.js Serving

Public mesh serving:

```js
const { Node, generateOwnerKeypairHex, resolveNativeRuntime } = require('@meshllm/sdk')

await resolveNativeRuntime({
  artifactDir: process.env.MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR,
  allowDownload: process.env.MESH_SDK_RUNTIME_ALLOW_DOWNLOAD === '1',
  onProgress: (event) => console.log(event)
})

const modelRef = process.env.MESH_SDK_MODEL_REF || 'Qwen2.5-3B-Instruct-Q4_K_M'
const node = Node.create({
  ownerKeypairHex: generateOwnerKeypairHex(),
  inviteToken: process.env.MESH_PUBLIC_INVITE,
  servingEnabled: true,
  cacheDir: process.env.MESH_SDK_CACHE_DIR,
  runtimeDir: process.env.MESH_SDK_RUNTIME_DIR
})

await node.start()
await node.models.download(modelRef)
const served = await node.serving.load(modelRef, { devicePolicy: 'auto' })
const result = await node.inference.chat({
  model: served.modelId,
  messages: [{ role: 'user', content: 'Say hello from a public serving node.' }]
})
console.log(result.content)
await node.serving.unloadModel(served.modelId)
await node.stop()
```

Private mesh serving uses the same lifecycle with `MESH_PRIVATE_INVITE`:

```js
const node = Node.create({
  ownerKeypairHex: generateOwnerKeypairHex(),
  inviteToken: process.env.MESH_PRIVATE_INVITE,
  servingEnabled: true,
  cacheDir: process.env.MESH_SDK_CACHE_DIR,
  runtimeDir: process.env.MESH_SDK_RUNTIME_DIR
})
```

### Swift Client

Public mesh client:

```swift
import MeshLLM

let ownerKeypair = generateOwnerKeypairHex()
let client = try await Client.connectPublic(
    ownerKeypairBytesHex: ownerKeypair,
    query: PublicMeshQuery(
        model: "Qwen3",
        minVramGb: nil,
        region: nil,
        targetName: nil,
        relays: []
    )
)

try await client.start()
let publicModels = try await client.inference.listModels()
try await printChat(
    stream: client.inference.chat(ChatRequest(model: publicModels[0].id, messages: [
        ChatMessage(role: "user", content: "Say hello from a public mesh.")
    ]))
)
await client.stop()
```

Private mesh client:

```swift
import MeshLLM

let ownerKeypair = generateOwnerKeypairHex()
let client = try Client(
    inviteToken: InviteToken(ProcessInfo.processInfo.environment["MESH_PRIVATE_INVITE"]!),
    ownerKeypairBytesHex: ownerKeypair
)

try await client.start()
let models = try await client.inference.listModels()
try await printChat(
    stream: client.inference.chat(ChatRequest(model: models[0].id, messages: [
        ChatMessage(role: "user", content: "Say hello from a private mesh.")
    ]))
)
await client.stop()
```

Shared Swift inference helper:

```swift
func printChat(stream: AsyncThrowingStream<Event, Error>) async throws {
    for try await event in stream {
        if case .tokenDelta(_, let delta) = event {
            print(delta, terminator: "")
        }
        if case .completed = event {
            print()
            return
        }
    }
}
```

### Swift Serving

Resolve or install a native runtime before local serving:

```swift
import MeshLLM

let runtime = try await NativeRuntime.resolve(
    NativeRuntimeResolveOptions(
        artifactDirectory: ProcessInfo.processInfo.environment["MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR"]
            .map(URL.init(fileURLWithPath:)),
        allowDownload: ProcessInfo.processInfo.environment["MESH_SDK_RUNTIME_ALLOW_DOWNLOAD"] == "1"
    )
)
print("using \(runtime.nativeRuntimeId) from \(runtime.path)")
```

Public mesh serving:

```swift
let ownerKeypair = generateOwnerKeypairHex()
let node = try Node(
    inviteToken: InviteToken(ProcessInfo.processInfo.environment["MESH_PUBLIC_INVITE"]!),
    ownerKeypairBytesHex: ownerKeypair
)
try await node.start()

let modelRef = ProcessInfo.processInfo.environment["MESH_SDK_MODEL_REF"] ?? "Qwen2.5-3B-Instruct-Q4_K_M"
_ = try await node.models.download(modelRef)
let served = try await node.serving.load(modelRef, options: LoadModelOptions(devicePolicy: .auto))
try await printChat(stream: node.inference.chat(ChatRequest(model: served.modelId, messages: [
    ChatMessage(role: "user", content: "Say hello from a public serving node.")
])))
try await node.serving.unloadModel(served.modelId, options: UnloadModelOptions(drainTimeoutMs: 1_000, force: false))
try await node.stop()
```

Private mesh serving uses the same lifecycle with `MESH_PRIVATE_INVITE`:

```swift
let ownerKeypair = generateOwnerKeypairHex()
let node = try Node(
    inviteToken: InviteToken(ProcessInfo.processInfo.environment["MESH_PRIVATE_INVITE"]!),
    ownerKeypairBytesHex: ownerKeypair
)
```

### Kotlin Client

Public mesh client:

```kotlin
import ai.meshllm.ChatMessage
import ai.meshllm.ChatRequest
import ai.meshllm.Client
import ai.meshllm.Event
import ai.meshllm.PublicMeshQuery
import kotlinx.coroutines.flow.collect
import uniffi.mesh_ffi.generateOwnerKeypairHex

val ownerKeypair = generateOwnerKeypairHex()
val client = Client.connectPublic(
    ownerKeypair,
    PublicMeshQuery(
        model = "Qwen3",
        minVramGb = null,
        region = null,
        targetName = null,
        relays = emptyList(),
    ),
)

client.start()
val publicModels = client.inference.listModels()
client.inference.chatFlow(
    ChatRequest(publicModels.first().id, listOf(ChatMessage("user", "Say hello from a public mesh."))),
).collect(::printToken)
client.stop()
```

Private mesh client:

```kotlin
import ai.meshllm.Client
import ai.meshllm.InviteToken
import uniffi.mesh_ffi.generateOwnerKeypairHex

val ownerKeypair = generateOwnerKeypairHex()
val client = Client(InviteToken(System.getenv("MESH_PRIVATE_INVITE")), ownerKeypair)

client.start()
val models = client.inference.listModels()
client.inference.chatFlow(
    ChatRequest(models.first().id, listOf(ChatMessage("user", "Say hello from a private mesh."))),
).collect(::printToken)
client.stop()
```

Shared Kotlin inference helper:

```kotlin
fun printToken(event: Event) {
    if (event is Event.TokenDelta) print(event.delta)
    if (event is Event.Completed) println()
}
```

### Kotlin Serving

Resolve or install the native runtime before local serving:

```kotlin
import ai.meshllm.NativeRuntime
import ai.meshllm.NativeRuntimeResolveOptions
import java.io.File

val runtime = NativeRuntime.resolve(
    NativeRuntimeResolveOptions(
        artifactDir = System.getenv("MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR")?.let(::File),
        allowDownload = System.getenv("MESH_SDK_RUNTIME_ALLOW_DOWNLOAD") == "1",
    ),
)
println("using ${runtime.nativeRuntimeId} from ${runtime.path}")
```

Public mesh serving:

```kotlin
import ai.meshllm.ChatMessage
import ai.meshllm.ChatRequest
import ai.meshllm.DevicePolicy
import ai.meshllm.InviteToken
import ai.meshllm.LoadModelOptions
import ai.meshllm.Node
import ai.meshllm.UnloadModelOptions

val ownerKeypair = generateOwnerKeypairHex()
val node = Node(InviteToken(System.getenv("MESH_PUBLIC_INVITE")), ownerKeypair)
node.start()

val modelRef = System.getenv("MESH_SDK_MODEL_REF") ?: "Qwen2.5-3B-Instruct-Q4_K_M"
node.models.download(modelRef)
val served = node.serving.load(modelRef, LoadModelOptions(DevicePolicy.Auto))
node.inference.chatFlow(
    ChatRequest(served.modelId, listOf(ChatMessage("user", "Say hello from a public serving node."))),
).collect(::printToken)
node.serving.unloadModel(served.modelId, UnloadModelOptions(drainTimeoutMs = 1_000UL, force = false))
node.stop()
```

Private mesh serving uses the same lifecycle with `MESH_PRIVATE_INVITE`:

```kotlin
val ownerKeypair = generateOwnerKeypairHex()
val node = Node(InviteToken(System.getenv("MESH_PRIVATE_INVITE")), ownerKeypair)
```

### Capability Notes

- Rust, Swift, and Kotlin expose public mesh discovery/auto-connect helpers for
  client mode.
- Node.js currently takes invite tokens directly. A public mesh is represented
  by a public invite token selected by the app or a discovery service.
- Serving examples use explicit invite tokens in every language so the local
  serving controller is attached before model load/unload calls.

## Native Runtime Artifacts

The accepted packaging direction is documented in
[design/NATIVE_RUNTIMES.md](design/NATIVE_RUNTIMES.md). In short: native
runtimes are release artifacts, not implicit Cargo builds. Runtime selection
defaults to the running MeshLLM release manifest, but compatibility is enforced
against the exact Skippy ABI version supported by the loader.

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
optional signature metadata. SDK loaders prefer manifests for the running
MeshLLM version and reject runtimes whose Skippy ABI version does not exactly
match the loader.

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
| `MESH_PUBLIC_INVITE` | Invite token selected from a public mesh listing. |
| `MESH_PRIVATE_INVITE` | Invite token for a private mesh. |
| `MESH_SDK_MODEL_REF` | Catalog, Hugging Face, or local model reference to download/load. |
| `MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR` | Verified `meshllm-native-runtime-*` artifact directory for local serving. |
| `MESH_SDK_RUNTIME_ALLOW_DOWNLOAD=1` | Allow SDK native-runtime resolution to download a matching runtime. |
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
