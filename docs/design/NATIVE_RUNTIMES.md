# Native Runtimes

Status: accepted direction with an implemented resolver/install foundation.

## Terminology

Use **native runtime** for the platform-specific serving artifact that contains
the native libraries, metadata, and loader inputs needed for local inference.
Avoid backend terminology for these artifacts. Names such as `cuda`, `rocm`,
`metal`, `vulkan`, and `cpu` describe runtime backend lanes.

## Decision

Native runtimes are release artifacts, not implicit Cargo builds. A MeshLLM SDK
consumer should be able to source Rust crates from crates.io without compiling
llama.cpp or Skippy native code as a default side effect.

Every native runtime declares the exact Skippy ABI it supports. Exact Skippy ABI
is the compatibility boundary for loading. MeshLLM version is still recorded so
release manifests, cache layout, and pruning can prefer the current release, but
runtime loading must reject by Skippy ABI rather than by MeshLLM semver alone.

Release CI owns the normal native runtime build. It builds, verifies, and
publishes the runtime artifacts for supported target/flavor combinations
alongside the MeshLLM release. Downloaded artifacts require checksum metadata
today. The API already has a policy knob for requiring signatures, but
signature verification intentionally fails closed until signing keys and
attestation format are implemented.

## Artifact Identity

A native runtime is identified by:

- MeshLLM version, for example `0.68.0`
- Skippy ABI, for example `0.1.25`
- target operating system and architecture
- backend kind, for example `cpu`, `metal`, `cuda`, `rocm`, or `vulkan`
- backend requirements, for example CUDA toolkit major, CUDA SM architecture,
  ROCm GPU target, driver/runtime minimums, and priority

The artifact manifest should include at least:

- `id`
- `mesh_version`
- `skippy_abi`
- `platform`
- `backend`
- `libraries`
- checksums
- signature or attestation metadata
- release URL
- ranking and compatibility metadata

The resolver must reject an artifact whose `skippy_abi` does not exactly match
the running loader ABI.

## Resolver

Runtime selection belongs in shared code that both SDK loaders and the
autoupdater can use. The compatibility matrix should be data-driven because new
platforms, GPU families, and runtime flavors are expected to arrive over time.

The resolver flow is:

1. Detect the local OS, architecture, available GPU devices, drivers, and
   supported backend lanes.
2. Load the release manifest for the running MeshLLM version.
3. Filter artifacts to those compatible with the host.
4. Rank compatible artifacts by backend lane and hardware fit.
5. Prefer an explicitly configured artifact directory when provided.
6. Check the local cache for the selected runtime.
7. If allowed, download the missing artifact while reporting progress.
8. Verify the archive checksum before use, and require signature verification
   when the caller selects that stricter policy.
9. Return a `NativeRuntime` descriptor with paths, identity, manifest metadata,
   and diagnostics.

The ranking policy is shared policy, not SDK-specific glue. For example, a
Linux NVIDIA host with CUDA 13 and `sm_120` support may rank
`cuda13-sm120` above generic `cuda13`, and accelerated runtimes above `cpu`,
when all compatibility checks pass.

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

## Release Manifest Discovery

The CLI and Rust SDK install API resolve manifests in this order:

1. explicit manifest path
2. explicit manifest URL
3. `MESH_LLM_NATIVE_RUNTIME_MANIFEST_URL`
4. the default release URL:

```text
https://github.com/Mesh-LLM/mesh-llm/releases/download/v<mesh_version>/native-runtimes.json
```

Bundled runtime directories are always appended to the candidate manifest. When
only bundle directories are provided, the default release URL is not fetched.
This lets packaged apps stay offline while the same code path can still inspect
or install release artifacts for normal crates.io consumers.

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
MeshLLM version without the caller duplicating flavor detection. Shell and
PowerShell installers should use the same command as manual users and
autoupdate flows, so CUDA, CUDA Blackwell, ROCm, Vulkan, CPU, verification,
cache layout, and progress UX stay in one implementation.

## Consumer Shape

A crates.io SDK consumer that wants dynamic local serving should use
`mesh_llm::sdk::native_runtime` instead of depending on a platform-specific
source build:

```rust
use mesh_llm_sdk::native_runtime::{
    NativeRuntimeInstallOptions, RuntimeSelection, install_native_runtime,
};

let runtime = install_native_runtime(NativeRuntimeInstallOptions {
    selection: RuntimeSelection::Recommended,
    cache_dir: Some(app_cache_dir.join("mesh-llm-native-runtimes")),
    bundle_dirs: vec![app_resources.join("meshllm-native-runtime")],
    progress: Some(std::sync::Arc::new(|event| {
        update_download_progress(event.downloaded_bytes, event.total_bytes);
    })),
    ..Default::default()
})
.await?;

let node = MeshNode::builder()
    .native_runtime(runtime.runtime)
    .build()?;
```

An offline packaged app uses the same path with downloads disabled:

```rust
let runtime = install_native_runtime(NativeRuntimeInstallOptions {
    bundle_dirs: vec![app_resources.join("meshllm-native-runtime")],
    allow_download: false,
    ..Default::default()
})
.await?;
```

The API returns the installed runtime, the selected release artifact, and the
full candidate evaluation so callers can show diagnostics or record why another
runtime was rejected.

## Verification Policy

Downloaded native runtime artifacts are fail-closed:

- `sha256` metadata is required for downloads.
- the downloaded archive digest must match the manifest digest.
- `RequireChecksumAndSignature` requires signature metadata and then returns an
  explicit unsupported-signature-verification error until signature verification
  is implemented.

This avoids silently presenting unsigned downloads as stronger than they are.
Checksum-only verification is the default policy for the first release artifact
lane.

The public installers use release-archive `.sha256` sidecars as
backward-compatible rollout metadata. New release/package jobs should keep
publishing sidecars, but installers must not assume every historical, pinned, or
alternate-repository asset has one. `install.sh` and `install.ps1` should try to
download `<archive>.sha256`; when it exists, malformed checksum data or a digest
mismatch is fatal. When the sidecar is missing, installers warn and continue by
default. Setting `MESH_LLM_REQUIRE_CHECKSUM=1` opts into fail-closed behavior for
missing release-archive sidecars. Do not rely on backfilling old release assets
to make installer checksum verification safe.

## Native Runtime Event Callback Contract

This contract defines the native runtime event callback for Skippy v1. Keep it
aligned with the Rust ABI source of truth in `crates/skippy-ffi/src/lib.rs`.
The callback covers model-open lifecycle facts only: started, progress, success,
and handled failure.

Native emits facts only. Rust owns policy, state transitions, telemetry policy,
JSONL formatting, routing, supervision, and user-facing output. The callback is
an observation boundary, not a state machine.

### Ownership Boundary

- Native may report backend selection, progress, and handled native errors.
- Rust decides how those facts affect mesh state, retries, and presentation.
- The return value is authoritative.
- Callback data never overrides the return path or process outcome.
- Callbacks are not authoritative state transitions.

### Callback Rules

- Install the callback only for one active model-open operation.
- Treat the callback as synchronous and best-effort.
- Do not block, reenter Skippy, or assume thread affinity.
- Native may invoke the callback from worker threads or the open thread during
  that operation.
- V1 guarantees no callback after the `_with_events` entrypoint returns.
- Rust code that receives the callback must not unwind across FFI.
- Rust must not call back into Skippy from inside the callback.
- A panic in the Rust trampoline is a Rust bug, not a native failure signal.
- Native may finish the open call without a terminal callback.
- Callbacks cannot reliably report segfault, abort, or other process crashes.

### Memory And Layout

- Event structs are versioned and fixed width.
- Each struct carries `abi_version` and `struct_size`.
- Event kinds and categories use explicit integer values, not layout-dependent
  enum assumptions.
- Strings are `const char *` plus length, borrowed only for the callback duration, and copied immediately by Rust during the callback.
- Optional monotonic timestamps, sequence numbers, progress counters, and
  failure codes are explicit fields.

### Reconciliation Rules

- If a callback is missing, dropped, late, contradictory, or unsupported, Rust
  still derives success or failure from the normal return path or process
  outcome.
- If a callback says success but the function returns error, the return error
  wins.
- If a callback says failure but the function returns success, the return value
  still wins.
- If the process crashes, the callback boundary is gone and recovery moves to
  the supervisor or restart path.

### Compatibility Matrix

| Situation | Native callback | Rust result |
| --- | --- | --- |
| Callback delivered, return succeeds | Facts are translated into Rust events | Success comes from the return value |
| Callback delivered, return fails | Facts are translated, including handled failure | Failure comes from the return value |
| Callback missing or dropped | No reliable terminal fact | Rust falls back to the return value and process outcome |
| Callback contradicts return path | Callback is only an observation | Return value is authoritative |
| Segfault or abort | No reliable callback can be expected | Supervisor or restart handling owns recovery |

### Non-Goals

- No C++ orchestration.
- No routing policy.
- No telemetry policy.
- No event bus.
- No token or tensor stream in v1.

V1 stays scoped to model-open lifecycle facts. Later event families can extend
the append-only boundary, but they must keep the same ownership rule: native
emits facts only, and Rust owns policy.

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
mesh-llm runtime install cuda12
mesh-llm runtime install cuda13
mesh-llm runtime install exact:meshllm-native-runtime-linux-x86_64-cuda13-sm120
mesh-llm runtime remove meshllm-native-runtime-linux-x86_64-cuda13
mesh-llm runtime prune
mesh-llm runtime prune --active-only
```

With no runtime argument, `mesh-llm runtime install` detects the host and
installs the recommended compatible native runtime from the release manifest,
an explicit manifest, or bundle directories. Explicit backend policy or runtime ID
arguments are overrides for advanced users, CI, and prepared images.

Selected-runtime diagnostics belong in `mesh-llm doctor`, not in a separate
`mesh-llm runtime doctor` command. Doctor output should include the active
MeshLLM version, selected native runtime ID, backend lane, runtime path,
cache path, manifest version, verification status, and any rejected compatible
candidates with reasons.

The `mesh-llm runtime` commands should use the same interactive UX conventions
as the rest of the MeshLLM CLI:

- emoji status markers where other user-facing commands use them
- spinners or progress lines during hardware, driver, and runtime detection
- byte and percent progress while downloading native runtime artifacts
- clear status transitions for resolving, downloading, verifying, installing,
  pruning, and already-current outcomes
- concise success output that names the selected runtime backend and version
- structured output for JSON/non-interactive modes without terminal control
  characters

The autoupdater should use the same inventory, ranking, install, prune, and
diagnostic code. It should not maintain a separate platform matrix.

## Cargo Features

Cargo features are compile-time capability gates. They are not the primary
dynamic hardware-selection mechanism for crates.io consumers.

Recommended SDK feature shape:

- `client`: OpenAI/Mesh client API facade.
- `console`: embeddable console server facade.
- `serving`: full in-process node embedding with dynamic native-runtime
  loading. This also exposes native-runtime manifest selection, download,
  cache management, and pruning APIs.

`serving` and `console` together enable the embedded node plus bundled web
console assets. Client-only applications should not pull native-runtime
install/update APIs. Native runtime source builds are not SDK features; native
runtimes are release artifacts resolved at install/update time.

## Generated Runtime Crates

Generated Cargo crates are not the supported native runtime distribution story
for this PR. The supported path is release artifacts plus
`native-runtimes.json`, resolved by the CLI, SDK, and autoupdater. Generated
runtime crates should not be published as a user-facing API until they package
the same native runtime artifact format and the release team explicitly decides
Cargo-target selection is a product requirement.

## Initial Release Matrix

The initial release workflow packages native runtimes only where the current CI
environment can build and smoke them reliably:

- macOS `aarch64` Metal
- Linux `x86_64` CPU

CUDA 12, CUDA 13, Blackwell-specific CUDA, ROCm, Vulkan, Windows, and additional architecture lanes
use the same manifest/backend model but still need dedicated runner/toolchain
work before the release matrix can claim coverage. The resolver is designed so
adding these artifacts is manifest data plus release jobs, not SDK API churn.

## Dynamic Loading Shape

For in-process SDK serving, MeshLLM needs a stable dynamic loading boundary for
native runtimes. The preferred shape is a version-matched shared native runtime
library loaded by the SDK resolver. A sidecar binary remains a possible
packaging option, but it is less appropriate for consumers that want an
embedded in-process node.

The loader must treat the exact Skippy ABI match as the compatibility boundary,
then use native runtime metadata for diagnostics and selection.
