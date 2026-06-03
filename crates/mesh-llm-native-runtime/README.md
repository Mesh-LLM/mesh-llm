# mesh-llm-native-runtime

Shared native runtime manifest, host profile, resolver, cache, and load-plan
policy for MeshLLM.

This crate is the source of truth for selecting native runtimes. CLI install,
SDK serving install, dynamic loading, and autoupdate should all use this same
contract instead of carrying their own CUDA/ROCm/Vulkan detection logic.

## Native Runtimes

A native runtime is a release artifact containing patched llama.cpp/Skippy
shared libraries for one platform/backend lane. The `mesh-llm` binary can stay
one artifact per OS/architecture; native runtimes carry the backend-specific
matrix:

- `cpu`
- `metal`
- `cuda` with a CUDA toolkit major such as 12 or 13
- `rocm` with optional GFX targets
- `vulkan`

The hard compatibility boundary is exact Skippy ABI. `mesh_version` is still
recorded and used for cache/prune layout, but a runtime is selected by
`skippy_abi`, platform, and backend requirements.

## Artifact Manifest

Each packaged runtime directory contains `manifest.json`:

```json
{
  "runtime": {
    "id": "meshllm-native-runtime-linux-x86_64-cuda13-sm120",
    "mesh_version": "0.68.0",
    "skippy_abi": "0.1.25",
    "platform": {
      "os": "linux",
      "arch": "x86_64",
      "target": "x86_64-unknown-linux-gnu"
    },
    "backend": {
      "kind": "cuda",
      "cuda": {
        "toolkit_major": 13,
        "min_driver": "580.0",
        "gpu_arches": ["sm_120"]
      }
    },
    "rank": 0,
    "libraries": ["lib/libllama.so"]
  }
}
```

CPU uses:

```json
"backend": { "kind": "cpu" }
```

ROCm uses:

```json
"backend": {
  "kind": "rocm",
  "rocm": {
    "version": "6.4",
    "gpu_arches": ["gfx1100"]
  }
}
```

Important fields:

- `id`: stable runtime ID used for explicit selection and cache paths.
- `skippy_abi`: exact ABI version required by the loader.
- `platform`: OS/arch/optional Rust target triple.
- `backend`: structured backend requirements.
- `rank`: optional rank adjustment. Higher compatible ranks win.
- `libraries`: runtime-relative load-order library paths.
- `url` and `sha256`: populated in release manifests for downloads.

## Release Manifest

Release jobs publish `native-runtimes.json`:

```json
{
  "mesh_version": "0.68.0",
  "skippy_abi": "0.1.25",
  "artifacts": [
    {
      "id": "meshllm-native-runtime-linux-x86_64-cpu",
      "mesh_version": "0.68.0",
      "skippy_abi": "0.1.25",
      "platform": { "os": "linux", "arch": "x86_64" },
      "backend": { "kind": "cpu" },
      "rank": 0,
      "libraries": ["lib/libllama.so"],
      "url": "https://github.com/Mesh-LLM/mesh-llm/releases/download/v0.68.0/meshllm-native-runtime-linux-x86_64-cpu.tar.gz",
      "sha256": "2f1c..."
    }
  ]
}
```

## Host Profile

Selection evaluates artifacts against `HostRuntimeProfile`:

```rust
use mesh_llm_native_runtime::{
    HostCudaProfile, HostRuntimeProfile, NativeRuntimeBackendKind,
};
use std::collections::BTreeSet;

let profile = HostRuntimeProfile {
    os: "linux".to_string(),
    arch: "x86_64".to_string(),
    target_triple: Some("x86_64-unknown-linux-gnu".to_string()),
    available_flavors: BTreeSet::from([
        NativeRuntimeBackendKind::Cpu,
        NativeRuntimeBackendKind::Cuda,
    ]),
    gpus: Vec::new(),
    cuda: Some(HostCudaProfile {
        toolkit_majors: BTreeSet::from([12]),
        driver_version: None,
        gpu_arches: BTreeSet::from(["sm_90".to_string()]),
    }),
    rocm: None,
    vulkan: None,
};
```

`mesh-llm-hardware-profile` builds this profile for real hosts. It supports
explicit environment overrides for CI/release testing, including
`MESH_LLM_CUDA_TOOLKIT_MAJOR`, `MESH_LLM_CUDA_TOOLKIT_MAJORS`,
`MESH_LLM_CUDA_GPU_ARCHES`, `MESH_LLM_ROCM_GPU_ARCHES`, and
`MESH_LLM_VULKAN_AVAILABLE`.

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
#     profile: mesh_llm_native_runtime::HostRuntimeProfile,
#     manifest: NativeRuntimeReleaseManifest,
# ) -> anyhow::Result<()> {
let cache = NativeRuntimeCache::new("/tmp/mesh-llm/native-runtimes");
let resolution = NativeRuntimeResolver::new("0.68.0", profile, manifest, cache)
    .with_skippy_abi_version("0.1.25")
    .with_bundle_dirs(vec![PathBuf::from("./meshllm-native-runtime-linux-x86_64-cpu")])
    .resolve(&RuntimeSelection::Recommended)?;

println!("selected {}", resolution.selected.id);
# Ok(())
# }
```

Selection strings accepted by `RuntimeSelection::parse`:

- `recommended`
- `cpu`
- `metal`
- `cuda`
- `cuda12`
- `cuda13`
- `rocm`
- `vulkan`
- `exact:<runtime-id>`

Compatibility checks:

- exact Skippy ABI
- OS/arch/target triple
- backend kind support
- CUDA toolkit major
- CUDA SM architecture
- ROCm GFX architecture
- Vulkan availability
- explicit selection policy

Every candidate is returned in `NativeRuntimeResolution::evaluated` with
structured rejection reasons for `mesh-llm runtime list`, `mesh-llm doctor`, SDK
diagnostics, and support output.

## Cache Layout

Installed runtimes are stored under:

```text
<cache-root>/<mesh_version>/<runtime-id>/
  manifest.json
  lib/...
```

`mesh_version` remains part of the cache layout and prune policy so upgrading
MeshLLM can install the newly selected runtime, switch to it, and remove older
runtime caches after success.

## Load Plan Boundary

This crate does not load dynamic libraries. `InstalledNativeRuntime::load_plan`
validates `runtime.libraries` and returns absolute paths for the Skippy FFI
loader:

```rust
# fn example(installed: mesh_llm_native_runtime::InstalledNativeRuntime) -> anyhow::Result<()> {
let plan = installed.load_plan()?;
for library in plan.libraries {
    println!("load {}", library.display());
}
# Ok(())
# }
```

## Packaging

Package and verify a runtime:

```bash
scripts/package-native-runtime.sh \
  --build \
  --backend cuda \
  --target x86_64-unknown-linux-gnu \
  --out dist/native-runtimes

scripts/verify-native-runtime-package.sh dist/native-runtimes/*.tar.gz
```

CUDA lanes use `MESH_LLM_CUDA_TOOLKIT_MAJOR` to emit IDs such as `cuda12` or
`cuda13`. `--backend cuda-blackwell` defaults to `cuda13-sm120`.

Generate the release manifest:

```bash
scripts/generate-native-runtime-release-manifest.sh \
  --tag v0.68.0 \
  --out dist/native-runtimes/native-runtimes.json \
  dist/native-runtimes/*.tar.gz
```
