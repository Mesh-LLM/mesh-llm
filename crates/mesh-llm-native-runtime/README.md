# mesh-llm-native-runtime

Shared native runtime manifest, resolver, cache, and loading-plan policy for
MeshLLM.

This crate owns the code that decides which native runtime artifact should be
used for the current host. It is intentionally small and dependency-light so the
same policy can be reused by the CLI, installers, autoupdate, and future SDK
entry points.

## Concepts

A **native runtime** is a version-matched release artifact that contains the
platform-specific patched llama.cpp/Skippy shared libraries needed by MeshLLM,
such as a CPU, Metal, CUDA, CUDA Blackwell, ROCm, or Vulkan runtime.

Native runtimes are separate from the `mesh-llm` binary:

- `mesh-llm` is released once per OS/architecture.
- Native runtimes are released per MeshLLM version, OS/architecture, and
  runtime flavor.
- The native runtime `mesh_version` must match the running MeshLLM version.
- Runtime selection is based on a host profile and the release manifest, not on
  shell installer heuristics.

The resolver never builds native runtimes. It only selects, locates, installs,
or describes released artifacts.

## Artifact Manifest

Each packaged runtime directory contains `manifest.json`. The crate accepts two
equivalent shapes:

```json
{
  "artifact": {
    "native_runtime_id": "meshllm-native-runtime-linux-x86_64-cuda",
    "mesh_version": "0.68.0",
    "target_triple": "x86_64-unknown-linux-gnu",
    "os": "linux",
    "arch": "x86_64",
    "flavor": "cuda",
    "library_paths": ["lib/libggml.so", "lib/libllama-common.so", "lib/libmtmd.so", "lib/libllama.so"]
  }
}
```

Release-packaged native runtime artifacts use the direct shape:

```json
{
  "schema_version": 1,
  "artifact_id": "meshllm-native-runtime-linux-x86_64-cuda",
  "native_runtime_id": "meshllm-native-runtime-linux-x86_64-cuda",
  "mesh_version": "0.68.0",
  "target_triple": "x86_64-unknown-linux-gnu",
  "platform": "linux-x86_64",
  "os": "linux",
  "arch": "x86_64",
  "backend": "cuda",
  "flavor": "cuda",
  "library": "lib/libllama.so",
  "library_paths": ["lib/libggml.so", "lib/libllama-common.so", "lib/libmtmd.so", "lib/libllama.so"],
  "library_sha256": "7c2b...",
  "skippy_abi_version": "0.1.24",
  "requirements": []
}
```

Important fields:

- `native_runtime_id`: stable artifact ID used for cache paths and explicit
  selection.
- `mesh_version`: required exact MeshLLM version for this runtime.
- `target_triple`: optional Rust target triple. When both the artifact and host
  profile have a target triple, they must match exactly.
- `os` and `arch`: compared against `std::env::consts::OS` and
  `std::env::consts::ARCH` values from the host profile, for example `linux` /
  `x86_64` or `macos` / `aarch64`.
- `flavor`: runtime flavor. Built-in flavors are `cpu`, `metal`, `cuda`,
  `cuda-blackwell`, `rocm`, and `vulkan`. Unknown flavor strings are preserved
  as `NativeRuntimeFlavor::Other`.
- `priority`: optional rank adjustment. Higher compatible ranks win.
- `url`: optional archive URL used when the runtime is not already installed or
  supplied as a bundle directory.
- `sha256`: optional archive SHA-256 used by callers that download artifacts.
- `library_paths`: runtime-relative library paths used by the load-plan
  boundary.
- `requirements`: optional host requirements. Today this supports GPU display
  name matching through `gpu_name_contains`; compute capability and driver
  fields are reserved in the schema for stricter future checks.

## Release Manifest

Release manifests list the runtime artifacts available for a MeshLLM release:

```json
{
  "mesh_version": "0.68.0",
  "artifacts": [
    {
      "native_runtime_id": "meshllm-native-runtime-linux-x86_64-cpu",
      "mesh_version": "0.68.0",
      "target_triple": "x86_64-unknown-linux-gnu",
      "os": "linux",
      "arch": "x86_64",
      "flavor": "cpu",
      "url": "https://github.com/Mesh-LLM/mesh-llm/releases/download/v0.68.0/meshllm-native-runtime-linux-x86_64-cpu.tar.gz",
      "sha256": "2f1c...",
      "library_paths": ["lib/libggml.so", "lib/libllama-common.so", "lib/libmtmd.so", "lib/libllama.so"]
    }
  ]
}
```

`NativeRuntimeReleaseManifest::validate` enforces that every artifact has the
same `mesh_version` as the release manifest. This is the compatibility boundary:
new MeshLLM versions install matching native runtimes rather than reusing older
ones.

The release workflow generates this file as `native-runtimes.json` from native
runtime archives.

## Host Profile

Selection runs against `HostRuntimeProfile`:

```rust
use mesh_llm_native_runtime::{HostRuntimeProfile, NativeRuntimeFlavor};
use std::collections::BTreeSet;

let profile = HostRuntimeProfile {
    os: "linux".to_string(),
    arch: "x86_64".to_string(),
    target_triple: Some("x86_64-unknown-linux-gnu".to_string()),
    available_flavors: BTreeSet::from([
        NativeRuntimeFlavor::Cpu,
        NativeRuntimeFlavor::Cuda,
    ]),
    gpus: Vec::new(),
};
```

`HostRuntimeProfile::current_without_gpu_probe()` only reports CPU plus Metal on
macOS. Production callers that can inspect hardware should build a richer
profile and set `available_flavors` from detected devices.

## Resolution

Use `NativeRuntimeResolver` when the caller needs both the selected artifact and
where it should come from:

```rust
use mesh_llm_native_runtime::{
    NativeRuntimeCache, NativeRuntimeReleaseManifest, NativeRuntimeResolver,
    RuntimeSelection,
};
use std::path::PathBuf;

# fn example(
#     mesh_version: &str,
#     profile: mesh_llm_native_runtime::HostRuntimeProfile,
#     manifest: NativeRuntimeReleaseManifest,
# ) -> anyhow::Result<()> {
let cache = NativeRuntimeCache::new("/tmp/mesh-llm/native-runtimes");
let resolution = NativeRuntimeResolver::new(mesh_version, profile, manifest, cache)
    .with_bundle_dirs(vec![PathBuf::from("./dist/native-runtimes/meshllm-native-runtime-linux-x86_64-cpu")])
    .resolve(&RuntimeSelection::Recommended)?;

match resolution.source {
    mesh_llm_native_runtime::NativeRuntimeSource::Installed { path } => {
        println!("already installed at {}", path.display());
    }
    mesh_llm_native_runtime::NativeRuntimeSource::Bundle { path } => {
        println!("install from bundled runtime {}", path.display());
    }
    mesh_llm_native_runtime::NativeRuntimeSource::Download { url } => {
        println!("download from {url}");
    }
    mesh_llm_native_runtime::NativeRuntimeSource::Missing => {
        println!("selected runtime has no local source or URL");
    }
}
# Ok(())
# }
```

Use `select_native_runtime` when a caller only needs the best compatible
candidate and does not need source lookup.

Selection modes:

- `RuntimeSelection::Recommended`: choose the highest-ranked compatible runtime.
- `RuntimeSelection::Flavor(NativeRuntimeFlavor::Cuda)`: require a specific
  flavor.
- `RuntimeSelection::Id("meshllm-native-runtime-linux-x86_64-cuda".to_string())`:
  require a specific artifact ID.

Compatibility checks:

- exact MeshLLM version match
- OS match
- architecture match
- target triple match when both sides declare it
- host support for the artifact flavor
- artifact requirements, such as required GPU name fragments
- explicit flavor or ID selection match

Ranking is `artifact.priority + artifact.flavor.default_rank()`. Default flavor
ranking prefers accelerated runtimes over CPU, with CUDA Blackwell ranked above
general CUDA.

Every candidate is returned in `NativeRuntimeResolution::evaluated` with
structured `CandidateRejection` reasons, so CLIs and diagnostics can explain why
an artifact was skipped.

## Cache Layout

`NativeRuntimeCache` stores installed runtimes under:

```text
<cache-root>/<mesh_version>/<native_runtime_id>/
  manifest.json
  lib/...
```

The default cache root is chosen by callers. The `native_runtime_cache_root`
helper maps a base cache directory to:

```text
<base-cache-dir>/mesh-llm/native-runtimes
```

Important cache operations:

- `installed()`: enumerate installed runtimes.
- `find_installed(mesh_version, native_runtime_id)`: find one runtime.
- `install_from_dir(source_dir)`: copy a packaged runtime directory into the
  versioned cache, replacing any existing runtime with the same ID and version.
- `remove(mesh_version, native_runtime_id)`: delete one runtime.
- `prune(active_mesh_version, NativeRuntimePruneMode::KeepActiveAndPrevious)`:
  keep the active MeshLLM version and the most recent previous version.
- `prune(active_mesh_version, NativeRuntimePruneMode::ActiveOnly)`: remove every
  runtime that does not match the active MeshLLM version.

Install and autoupdate flows should normally prune with `ActiveOnly` after a
successful runtime install for the upgraded MeshLLM version.

## Load Plan Boundary

This crate does not load dynamic libraries itself. It exposes
`InstalledNativeRuntime` as a cache record and
`InstalledNativeRuntime::load_plan()` as the boundary consumed by the Skippy FFI
dynamic loader in the host runtime.

```rust
# fn example(installed: mesh_llm_native_runtime::InstalledNativeRuntime) -> anyhow::Result<()> {
let plan = installed.load_plan()?;
for library in plan.libraries {
    println!("load {}", library.display());
}
# Ok(())
# }
```

`load_plan()` validates that the manifest declares at least one library and that
each declared path exists under the installed runtime directory. The caller
still owns ABI checks and actual dynamic loading.

## CLI Integration

The host runtime crate wraps this API in:

```bash
mesh-llm runtime list --installed
mesh-llm runtime list --available --manifest native-runtimes.json
mesh-llm runtime install
mesh-llm runtime install cuda
mesh-llm runtime remove meshllm-native-runtime-linux-x86_64-cuda
mesh-llm runtime prune --active-only
mesh-llm doctor --json
```

The CLI owns network downloads, archive extraction, progress output, and
checksum verification. This crate owns the manifest, compatibility, cache, and
load-plan semantics those commands use.

## Current Boundaries

- Native runtimes are release artifacts. Cargo does not implicitly build them
  for consumers.
- Runtime versioning is strict: `mesh_version` must match the running MeshLLM
  version.
- The crate preserves unknown flavors so new platform-specific flavor names can
  be added without changing the manifest schema.
- Dynamic loading is intentionally outside this crate; the shipped host runtime
  consumes `load_plan()` and loads the declared libraries through `skippy-ffi`
  when built with `dynamic-native-runtime`.
