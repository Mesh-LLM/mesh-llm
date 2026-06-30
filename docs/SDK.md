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
needs a matching native runtime artifact or an embedded Rust
`ServingController`.

## Platform Support

Check this before wiring a serving flow. Some SDK targets can join meshes and
run inference but cannot currently serve local models.

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

## Package Sources

The SDK packages are published from MeshLLM releases:

| SDK | Package source |
|---|---|
| Rust | crates.io package `mesh-llm-sdk` |
| Node.js | npm package `@meshllm/sdk` |
| Swift | GitHub Swift package from tagged `Mesh-LLM/mesh-llm` releases |
| Kotlin/Android | GitHub Packages Maven registry for `Mesh-LLM/mesh-llm` |
| Native runtimes | GitHub release artifacts plus `native-runtimes.json` |

## Install

### Rust

Add the Rust SDK facade crate:

```toml
[dependencies]
mesh-llm-sdk = "0.72.1"
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

See [Rust SDK examples](sdk/rust.md).

### Swift

Add the repo Swift package from a tagged GitHub release:

```swift
dependencies: [
    .package(url: "https://github.com/Mesh-LLM/mesh-llm", from: "0.72.1"),
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

See [Swift SDK examples](sdk/swift.md).

### Kotlin/Android

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

See [Kotlin SDK examples](sdk/kotlin.md).

### Node.js

Install the Node package in a Node.js or Electron app:

```json
{
  "dependencies": {
    "@meshllm/sdk": "0.72.1"
  }
}
```

When building from this repository, build the native N-API addon first:

```bash
cd sdk/node
npm run build:native
```

See [Node.js SDK examples](sdk/node.md).

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

## Examples

Each language-specific page includes client and serving examples for public and
private mesh modes:

| SDK | Examples |
|---|---|
| Rust | [docs/sdk/rust.md](sdk/rust.md) |
| Node.js | [docs/sdk/node.md](sdk/node.md) |
| Swift | [docs/sdk/swift.md](sdk/swift.md) |
| Kotlin/Android | [docs/sdk/kotlin.md](sdk/kotlin.md) |

For client-only apps, public mesh examples use discovery where the language SDK
exports it. For local serving, examples use concrete invite tokens because the
serving-enabled node needs to join the selected mesh while also attaching the
local serving controller.

## Console Assets

Published SDK packages that advertise console support include the built web
console as package resources. Source checkouts can regenerate those resources
with:

```bash
scripts/package-sdk-console-assets.sh --sdk all
scripts/verify-sdk-console-assets.sh --sdk all
```

SDK users should not have to pass a raw directory in normal package usage. The
path-based native boundary exists so SwiftPM resources, Node package files, JVM
resources, and Rust app resources can all pass the packaged console directory
to the same console server.

## Native Runtime Artifacts

The accepted packaging direction is documented in
[design/NATIVE_RUNTIMES.md](design/NATIVE_RUNTIMES.md). In short: native
runtimes are release artifacts, not implicit Cargo builds. Runtime selection
defaults to the running MeshLLM release manifest, but compatibility is enforced
against the exact Skippy ABI version supported by the loader.

Native runtime artifacts use this layout:

```text
meshllm-native-runtime-<platform>-<backend-lane>/
  manifest.json
  README.md
  lib/
    libllama.{dylib|so|dll}
    libggml*.{dylib|so|dll}
```

The manifest records the MeshLLM version, exact Skippy ABI, platform,
structured backend requirements, load-order library paths, release URL,
checksum, and optional signature metadata. Runtime compatibility is exact
Skippy ABI plus platform/backend requirements; MeshLLM version remains part of
cache layout and pruning.

Baseline artifact names:

| Artifact directory | Target | Backend lane |
|---|---|---|
| `meshllm-native-runtime-darwin-aarch64-metal` | `aarch64-apple-darwin` | Metal |
| `meshllm-native-runtime-darwin-aarch64-cpu` | `aarch64-apple-darwin` | CPU |
| `meshllm-native-runtime-linux-x86_64-cpu` | `x86_64-unknown-linux-gnu` | CPU |
| `meshllm-native-runtime-linux-x86_64-cuda12` | `x86_64-unknown-linux-gnu` | CUDA 12 |
| `meshllm-native-runtime-linux-x86_64-cuda13` | `x86_64-unknown-linux-gnu` | CUDA 13 |
| `meshllm-native-runtime-linux-x86_64-cuda13-sm120` | `x86_64-unknown-linux-gnu` | CUDA 13 Blackwell |
| `meshllm-native-runtime-linux-x86_64-vulkan` | `x86_64-unknown-linux-gnu` | Vulkan |
| `meshllm-native-runtime-linux-x86_64-rocm` | `x86_64-unknown-linux-gnu` | ROCm/HIP |
| `meshllm-native-runtime-windows-x86_64-cpu` | `x86_64-pc-windows-msvc` | CPU |
| `meshllm-native-runtime-windows-x86_64-cuda12` | `x86_64-pc-windows-msvc` | CUDA 12 |
| `meshllm-native-runtime-windows-x86_64-cuda13` | `x86_64-pc-windows-msvc` | CUDA 13 |
| `meshllm-native-runtime-windows-x86_64-vulkan` | `x86_64-pc-windows-msvc` | Vulkan |
| `meshllm-native-runtime-windows-x86_64-rocm` | `x86_64-pc-windows-msvc` | ROCm/HIP |

CUDA and ROCm compatibility is encoded as structured backend metadata, not as
free-form flavor matching. CUDA runtimes declare a toolkit major and optional
SM architectures; ROCm runtimes can declare GFX targets.

Build and package one flavor:

```bash
scripts/package-native-runtime.sh \
  --build \
  --backend cuda \
  --target x86_64-unknown-linux-gnu \
  --out dist/native-runtimes
```

Set `MESH_LLM_CUDA_TOOLKIT_MAJOR=13` to emit a CUDA 13 lane. Use
`--backend cuda-blackwell` for the CUDA 13 `sm120` lane.

Verify produced artifacts:

```bash
scripts/verify-native-runtime-package.sh dist/native-runtimes/*.tar.gz
```

## Selecting a Runtime

Cargo, npm, SwiftPM, and Maven dependencies provide language SDKs. Native
runtimes are resolved at install or application startup from release artifacts,
not built implicitly by the package manager.

Normal online install:

```bash
mesh-llm runtime install
```

Explicit backend policy examples:

```bash
mesh-llm runtime install cuda12
mesh-llm runtime install cuda13
mesh-llm runtime install exact:meshllm-native-runtime-linux-x86_64-cuda13-sm120
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
