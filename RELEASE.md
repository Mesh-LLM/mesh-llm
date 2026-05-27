# Releasing mesh-llm

## Prerequisites

- `just` installed
- Rust toolchain installed
- `cmake` and a native compiler installed
- Node/npm installed for the UI build
- `gh` CLI authenticated if publishing manually

## Build

```bash
just build
```

`just build` prepares the pinned upstream `llama.cpp` checkout, applies the
Mesh-LLM ABI patch queue from `third_party/llama.cpp/patches`, builds the
patched static ABI libraries, builds the UI, and builds the `mesh-llm` binary.

The release bundle is now a single `mesh-llm` runtime binary. External
`llama-server`, `rpc-server`, and `llama-moe-*` binaries are not packaged.

## Bundle

```bash
just bundle
```

This creates `/tmp/mesh-bundle.tar.gz` containing the packaged `mesh-llm`
executable. That packaged binary is the release-attestation source of truth.
Local and dev builds from `just build` stay unstamped by default, so `missing`
is expected there.

Verify the packaged executable with `cargo run -p xtask -- release-attestation inspect --binary /tmp/test-bundle/mesh-llm --public-key-file /tmp/mesh-release-key.pub`.
`valid` means the packaged binary matches a trusted release signer, `missing`
means an unstamped build, and `invalid` means the bytes changed after packaging.
Bare `inspect --binary ...` is only sufficient for unstamped binaries that
should classify as `missing`; a stamped package requires `--public-key-file` and
otherwise reports `invalid` with an explicit error. A post-download mutation can
turn a stamped binary `invalid`, but default startup still allows it because this
is provenance and admission hardening, not runtime integrity proof.

Platform release archives are created with:

```bash
just release-build
just release-bundle v0.X.Y
```

Before manually cutting a tag that should be consumable through SwiftPM,
prepare the Swift binary target manifest on macOS and commit the resulting
`Package.swift` change:

```bash
scripts/prepare-swift-package-release.sh v0.X.Y
git add Package.swift sdk/swift/Sources/MeshLLM/Generated/mesh_ffi.swift
git commit -m "v0.X.Y: prepare Swift package artifact"
```

The release workflow rebuilds `MeshLLMFFI.xcframework.zip`, verifies the macOS
framework layout, runs a zipped-artifact SwiftPM consumer smoke, and checks that
the tagged `Package.swift` already points at the exact release URL and checksum.
If `Package.swift` still contains placeholders on a tag push, or if the
checksum does not match the artifact built in release CI, the release fails
before publishing.

For `workflow_dispatch` releases, the release workflow computes the SwiftPM
checksum from the XCFramework artifact it just built, patches `Package.swift`
in the workflow workspace, and creates the requested release tag at a
manifest-only commit before publishing.

The current GitHub Actions release workflow publishes macOS aarch64, Linux
x86_64 CPU, Linux ARM64 CPU, Linux CUDA, Linux CUDA Blackwell, Linux ROCm,
Linux Vulkan, Windows CPU, Windows CUDA, Windows ROCm, and Windows Vulkan
bundles, plus the SwiftPM `MeshLLMFFI.xcframework.zip` binary artifact. The
Linux ARM64 artifact is named
`mesh-llm-aarch64-unknown-linux-gnu.tar.gz`; CUDA lanes are named
`mesh-llm-x86_64-unknown-linux-gnu-cuda.tar.gz` and
`mesh-llm-x86_64-unknown-linux-gnu-cuda-blackwell.tar.gz`.

Windows release artifacts use the `x86_64-pc-windows-msvc` target triple and
`.zip` archives.

On native Windows, `just check-release` still runs the Rust/docs/workflow invariant checks, but it skips the Bash-only `install.sh` and `scripts/package-release.sh` parity checks.

## Smoke Test

```bash
mkdir /tmp/test-bundle
tar xzf /tmp/mesh-bundle.tar.gz -C /tmp/test-bundle --strip-components=1
/tmp/test-bundle/mesh-llm --model Qwen2.5-3B
rm -rf /tmp/test-bundle
```

Verify:

- the process starts without looking for `llama-server` or `rpc-server`;
- `/api/status` returns valid JSON;
- `/v1/models` lists the resolved model refs;
- `/v1/chat/completions` can generate through the embedded runtime.

## Publish

Push a `v*` tag to run `.github/workflows/release.yml`.

On non-prerelease tags, the release workflow also publishes the Rust SDK crate
chain to crates.io in dependency order:

```bash
scripts/publish-crates.sh --dry-run
```

The chain currently publishes:

1. `model-ref`
2. `mesh-llm-identity`
3. `mesh-llm-protocol`
4. `mesh-llm-routing`
5. `mesh-llm-types`
6. `model-artifact`
7. `model-hf`
8. `mesh-llm-client`
9. `mesh-llm-api-client`
10. `mesh-llm-node`
11. `mesh-llm-api-server`

Run the dry-run before cutting a GA tag after changing SDK crate manifests or
workspace-internal SDK dependencies. On the first release that introduces a
new internal SDK crate, the dry-run validates packages whose registry
dependencies already exist and reports downstream packages that will be fully
verified during the real sequential publish after their upstream crates land.

If crates.io rate-limits the non-prerelease publish chain after some crates
have already uploaded, rerun `scripts/publish-crates.sh` for the same checked
out release tag instead of recutting the GitHub release or moving the tag. The
script checks crates.io before each real publish, skips crate versions that are
already visible, and retries HTTP 429 new-crate rate-limit responses using the
retry time from crates.io when one is provided.
