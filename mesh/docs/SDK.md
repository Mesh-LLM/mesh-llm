# Mesh LLM SDK Usage Guide

Python, Rust, Swift, Kotlin, and Node.js now expose one embedded node. Select
client-only, serve-only, or combined mode when constructing it. Client mode joins
a mesh and consumes inference without loading a serving runtime. Serve-only
mode loads and advertises local models but rejects SDK inference requests.
Combined mode does both.

| SDK | Guide |
|---|---|
| Rust | [Rust](sdk/rust.md) |
| Python | [Python](sdk/python.md) |
| Swift | [Swift](sdk/swift.md) |
| Kotlin | [Kotlin](sdk/kotlin.md) |
| Node.js | [Node.js](sdk/node.md) |

The legacy thin `Client`, its in-memory keypair hex constructor, and the Rust
facade's `client`/`node` feature split have been removed. Platform constructors
accept an optional Mesh LLM owner keystore path. This is an SDK breaking change;
the mesh wire protocol and `/v1` API are unchanged.

## Native Runtime Artifacts

The accepted packaging direction is documented in
[design/NATIVE_RUNTIMES.md](../../skippy/docs/design/NATIVE_RUNTIMES.md). In short: native
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
    cache_dir: Some(app_cache_dir.join("skippy-native-runtimes")),
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

## Validation

Check the Rust SDK, UniFFI bridge, Node.js addon, and the platform package tests
using their package-local test commands. A serve-only or combined node needs a
matching installed native runtime; see the installation guidance above.
