# Native Runtimes

Status: accepted SDK packaging direction.

## Terminology

Use **native runtime** for the platform-specific serving artifact that contains
the native libraries, metadata, and loader inputs needed for local inference.
Avoid backend terminology for these artifacts. Names such as `cuda`, `rocm`,
`metal`, `vulkan`, and `cpu` describe runtime flavors.

## Decision

Native runtimes are release artifacts, not implicit Cargo builds. A MeshLLM SDK
consumer should be able to source Rust crates from crates.io without compiling
llama.cpp or Skippy native code as a default side effect.

Every native runtime is tied to exactly one MeshLLM version. The native runtime
version must match the MeshLLM crate, SDK, or binary version that loads it. The
Skippy ABI version is useful diagnostic metadata, but it is not a substitute
for the MeshLLM version match.

Release CI owns the normal native runtime build. It builds, verifies, signs,
and publishes the runtime artifacts for supported target/flavor combinations
alongside the MeshLLM release.

## Artifact Identity

A native runtime is identified by:

- MeshLLM version, for example `0.68.0`
- target operating system and architecture
- runtime flavor, for example `cpu`, `metal`, `cuda`, `cuda-blackwell`,
  `rocm`, or `vulkan`
- optional hardware constraints, for example CUDA compute capability, ROCm GPU
  target, driver/runtime minimums, and priority

The artifact manifest should include at least:

- `mesh_version`
- `native_runtime_id`
- `target_triple`
- `os`
- `arch`
- `flavor`
- `skippy_abi_version`
- native library paths
- checksums
- signature or attestation metadata
- release URL
- ranking and compatibility metadata

The resolver must reject an artifact whose `mesh_version` does not exactly
match the running MeshLLM version.

## Resolver

Runtime selection belongs in shared code that both SDK loaders and the
autoupdater can use. The compatibility matrix should be data-driven because new
platforms, GPU families, and runtime flavors are expected to arrive over time.

The resolver flow is:

1. Detect the local OS, architecture, available GPU devices, drivers, and
   supported runtime flavors.
2. Load the signed release manifest for the exact running MeshLLM version.
3. Filter artifacts to those compatible with the host.
4. Rank compatible artifacts by flavor and hardware fit.
5. Prefer an explicitly configured artifact directory when provided.
6. Check the local cache for the selected runtime.
7. If allowed, download the missing artifact while reporting progress.
8. Verify checksum and signature before use.
9. Return a `NativeRuntime` descriptor with paths, identity, manifest metadata,
   and diagnostics.

The ranking policy is shared policy, not SDK-specific glue. For example, a
Linux NVIDIA host may rank `cuda-blackwell` above generic `cuda`, and generic
`cuda` above `vulkan` or `cpu`, when all compatibility checks pass.

## Cache And Progress

SDK consumers must be able to control where native runtimes are stored. The
default cache layout should be versioned:

```text
<cache>/mesh-llm/native-runtimes/<mesh_version>/<native_runtime_id>/
```

The resolver API must allow:

- a custom cache directory
- one or more packaged runtime directories to check before downloading
- download enable/disable
- a download progress callback
- clear diagnostics when no compatible runtime is available

Packaged application mode should not require network access. An app can provide
a bundled runtime directory, set downloads to false, and still use the same
resolver path as a downloading SDK consumer.

## Upgrade And Pruning

An updater must install and verify the new version-matched native runtime
before switching the active MeshLLM version. Old runtimes should be pruned only
after the new MeshLLM binary and native runtime are known to load together.

Default pruning keeps:

- native runtimes for the active MeshLLM version
- native runtimes for one previous MeshLLM version for rollback

Explicit prune operations may remove all native runtimes that do not match the
active MeshLLM version. The CLI and autoupdater should share this policy so
interactive cleanup and automatic cleanup behave the same way.

## Installer Bootstrap

`install.sh` and `install.ps1` should not duplicate GPU or runtime flavor
detection once native runtimes are available. Their long-term job is to detect
only enough OS/architecture information to download and install the matching
MeshLLM host binary, then delegate native runtime selection to MeshLLM:

```bash
mesh-llm runtime install
```

That command installs the recommended compatible native runtime for the active
MeshLLM version. Shell and PowerShell installers should use the same command as
manual users and autoupdate flows, so CUDA, CUDA Blackwell, ROCm, Vulkan, CPU,
verification, cache layout, and progress UX stay in one implementation.

## Consumer Shape

A crates.io SDK consumer that wants dynamic local serving should configure the
resolver instead of depending on a platform-specific source build. The API
shape should be equivalent to:

```rust
let runtime = NativeRuntimeResolver::builder()
    .cache_dir(app_cache_dir.join("mesh-llm"))
    .bundle_dir(app_resources.join("meshllm-native"))
    .allow_download(true)
    .on_progress(|event| {
        update_download_progress(event.downloaded_bytes, event.total_bytes);
    })
    .resolve_best()
    .await?;

let node = MeshNode::builder()
    .native_runtime(runtime)
    .build()?;
```

An offline packaged app uses the same path with downloads disabled:

```rust
let runtime = NativeRuntimeResolver::builder()
    .bundle_dir(app_resources.join("meshllm-native"))
    .allow_download(false)
    .resolve_best()
    .await?;
```

## Query And Management API

Consumers need explicit runtime inventory and cache management. The Rust API
should expose operations equivalent to:

- list available native runtimes for the current MeshLLM version
- list installed native runtimes in the configured cache
- install the recommended native runtime for this host
- optionally install an explicitly selected native runtime
- remove a selected native runtime
- prune runtimes for older MeshLLM versions
- diagnose why a runtime was selected or rejected

The `mesh-llm runtime` CLI should own inventory and cache management:

```bash
mesh-llm runtime list --available
mesh-llm runtime list --installed
mesh-llm runtime install
mesh-llm runtime install cuda
mesh-llm runtime remove meshllm-native-linux-x86_64-cuda
mesh-llm runtime prune
mesh-llm runtime prune --active-only
```

With no runtime argument, `mesh-llm runtime install` detects the host and
installs the recommended compatible native runtime. Explicit flavor or runtime
ID arguments are overrides for advanced users, CI, and prepared images.

Selected-runtime diagnostics belong in `mesh-llm doctor`, not in a separate
`mesh-llm runtime doctor` command. Doctor output should include the active
MeshLLM version, selected native runtime ID, runtime flavor, runtime path,
cache path, manifest version, verification status, and any rejected compatible
candidates with reasons.

The `mesh-llm runtime` commands should use the same interactive UX conventions
as the rest of the MeshLLM CLI:

- emoji status markers where other user-facing commands use them
- spinners or progress lines during hardware, driver, and runtime detection
- byte and percent progress while downloading native runtime artifacts
- clear status transitions for resolving, downloading, verifying, installing,
  pruning, and already-current outcomes
- concise success output that names the selected runtime flavor and version
- structured output for JSON/non-interactive modes without terminal control
  characters

The autoupdater should use the same inventory, ranking, install, prune, and
diagnostic code. It should not maintain a separate platform matrix.

## Cargo Features

Cargo features are compile-time capability gates. They are not the primary
dynamic hardware-selection mechanism for crates.io consumers.

Recommended SDK feature shape:

- `native-runtime`: enable native runtime loading APIs
- `native-download`: enable release artifact download support
- `native-build-cpu`: opt into source-building a CPU runtime
- `native-build-metal`: opt into source-building a Metal runtime
- `native-build-cuda`: opt into source-building a CUDA runtime
- `native-build-cuda-blackwell`: opt into source-building a Blackwell-tuned
  CUDA runtime
- `native-build-rocm`: opt into source-building a ROCm runtime
- `native-build-vulkan`: opt into source-building a Vulkan runtime

The source-build features are explicit escape hatches for developers and CI
jobs. They should not be default features. Build scripts should reject multiple
`native-build-*` features in one package build unless a crate is deliberately
designed to produce a multi-runtime artifact.

## Dynamic Loading Shape

For in-process SDK serving, MeshLLM needs a stable dynamic loading boundary for
native runtimes. The preferred shape is a version-matched shared native runtime
library loaded by the SDK resolver. A sidecar binary remains a possible
packaging option, but it is less appropriate for consumers that want an
embedded in-process node.

The loader must treat the exact MeshLLM version match as the compatibility
boundary, then use native runtime metadata for diagnostics and selection.
