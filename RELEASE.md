# Releasing mesh-llm

## Preferred path: dispatch from GitHub

Releases are normally cut by running the **Release** workflow
(`.github/workflows/release.yml`) from the GitHub Actions UI via
`workflow_dispatch` with the version input (for example `v0.31.0`). The
dispatched workflow bumps versions, generates and patches the SwiftPM
manifest, packages SDK console assets, creates and pushes the release tag,
builds all platform bundles, and publishes the GitHub release. After a complete
non-canary release succeeds, it dispatches `Mesh-LLM/mesh-packaging` to
package the verified release archives, publish the native package release
assets, and publish the supported GHCR image matrix. Dispatch inputs include
`skip_gpu_bundles` and `canary` (dry-run: build and smoke everything without
publishing). Releases that intentionally skip GPU bundles do not dispatch the
full packaging matrix.

The sections below document the underlying steps. They matter when releasing
manually via a tag push, debugging the workflow, or validating bundles
locally.

## Prerequisites

- `just` installed
- Rust toolchain installed
- `cmake` and a native compiler installed
- Node/npm installed for the UI build
- `gh` CLI authenticated if publishing manually
- `MESH_AGENT_IMAGES_DISPATCH_TOKEN` configured as a fine-grained repository
  token or GitHub App token with Contents write access to
  `Mesh-LLM/mesh-packaging`, which is the permission required to create a
  repository dispatch event. The legacy secret name is retained so existing
  release environments do not require a coordinated secret rename.

## Release Attestation Signing Keys

The GitHub Actions release workflow stamps packaged `mesh-llm` executables when
these repository Actions secrets are present:

- `MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE`
- `MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE`

The secret values are the full JSON contents of the release-attestation private
and public key files, not paths to files. Generate a production keypair with:

```bash
umask 077
mkdir -p /tmp/mesh-release-attestation
cargo run -q -p xtask -- release-attestation generate-keypair \
  --private-key-out /tmp/mesh-release-attestation/mesh-release-attestation-private-key.json \
  --public-key-out /tmp/mesh-release-attestation/mesh-release-attestation-public-key.json
```

Store the keypair in 1Password before adding or rotating GitHub secrets. The
production release-attestation keypair lives in the `mesh-llm` vault as
`GitHub Actions Release Attestation Signing Keys`, with fields named exactly
after the GitHub Actions secrets above.

Set or rotate the repository secrets from the generated files with:

```bash
gh secret set MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE \
  --app actions \
  < /tmp/mesh-release-attestation/mesh-release-attestation-private-key.json

gh secret set MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE \
  --app actions \
  < /tmp/mesh-release-attestation/mesh-release-attestation-public-key.json
```

After publishing, verify at least one packaged release archive by extracting it
and running:

```bash
cargo run -p xtask -- release-attestation inspect \
  --binary /tmp/test-bundle/mesh-llm \
  --public-key-file /tmp/mesh-release-attestation/mesh-release-attestation-public-key.json \
  --json
```

The reported status must be `valid`. A `missing` status means the bundle was
published without an embedded release-attestation footer. An `invalid` status
means a footer was present, but signature verification failed.

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

This creates `/tmp/mesh-llm-bundle.tar.gz` containing the packaged `mesh-llm`
executable for local deployment. Platform release archives are named by target,
such as `mesh-llm-aarch64-apple-darwin.tar.gz`.

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
x86_64 CPU, Linux ARM64 CPU, Linux ARM64 CUDA, Linux CUDA, Linux CUDA
Blackwell, Linux ROCm, Linux Vulkan, Windows CPU, Windows CUDA, Windows ROCm,
and Windows Vulkan bundles, plus the SwiftPM `MeshLLMFFI.xcframework.zip`
binary artifact. The Linux ARM64 CPU artifact is named
`mesh-llm-aarch64-unknown-linux-gnu.tar.gz`; the Linux ARM64 CUDA artifact is
named `mesh-llm-aarch64-unknown-linux-gnu-cuda.tar.gz`. x86_64 CUDA lanes are
named `mesh-llm-x86_64-unknown-linux-gnu-cuda.tar.gz` and
`mesh-llm-x86_64-unknown-linux-gnu-cuda-blackwell.tar.gz`.

Windows release artifacts use the `x86_64-pc-windows-msvc` target triple and
`.zip` archives.

On native Windows, `just check-release` still runs the Rust/docs/workflow invariant checks, but it skips the Bash-only `install.sh` and `scripts/package-release.sh` parity checks.

## Smoke Test

```bash
mkdir /tmp/test-bundle
tar xzf /tmp/mesh-llm-bundle.tar.gz -C /tmp/test-bundle --strip-components=1
/tmp/test-bundle/mesh-llm --model Qwen2.5-3B
rm -rf /tmp/test-bundle
```

Verify:

- the process starts without looking for `llama-server` or `rpc-server`;
- `/api/status` returns valid JSON;
- `/v1/models` lists the resolved model refs;
- `/v1/chat/completions` can generate through the embedded runtime.

## Publish

Push a `v*` tag to run `.github/workflows/release.yml`. The upstream release
workflow owns release archive production, but it does not publish OCI images.
`Mesh-LLM/mesh-packaging` is the canonical GHCR producer and starts only
after the GitHub release and its complete archive set have published
successfully. The upstream `docker.yml` workflow performs Dockerfile validation
only and is not a distribution channel.

On non-prerelease tags, the release workflow also publishes the Rust SDK crate
chain to crates.io in dependency order:

```bash
cargo run -p xtask -- repo-consistency publish-crates
scripts/publish-crates.sh --dry-run
```

SDK packages that expose the optional console must package the built web
console before publishing language SDK artifacts:

```bash
scripts/package-sdk-console-assets.sh --sdk all
scripts/verify-sdk-console-assets.sh --sdk all
```

The script builds `crates/mesh-llm-ui/dist` in release mode and copies it to
the canonical SDK resource locations: `sdk/node/console`,
`sdk/swift/Sources/MeshLLM/Resources/Console`, and
`sdk/kotlin/src/main/resources/mesh-llm/console`.

These generated directories are ignored during normal development. For a
manual tag push, force-add them into the release commit before tagging because
SwiftPM resolves package resources from the Git tag:

```bash
git add -f sdk/node/console sdk/swift/Sources/MeshLLM/Resources/Console sdk/kotlin/src/main/resources/mesh-llm/console
```

Workflow-dispatch releases generate and force-add these resources into the
release tag commit automatically.

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

Run the consistency check and dry-run before cutting a GA tag after changing
SDK crate manifests or workspace-internal SDK dependencies. The consistency
check keeps the scripted publish order, workspace path dependency versions,
publish metadata, bundled file includes, and CI release preflight in sync. On
the first release that introduces a new internal SDK crate, the dry-run
validates packages whose registry dependencies already exist and reports
downstream packages that will be fully verified during the real sequential
publish after their upstream crates land.

If crates.io rate-limits the non-prerelease publish chain after some crates
have already uploaded, rerun `scripts/publish-crates.sh` for the same checked
out release tag instead of recutting the GitHub release or moving the tag. The
script relies on `cargo publish` to report crate versions that were already
uploaded, continues past those already-uploaded crates, and retries HTTP 429
new-crate rate-limit responses using the retry time from crates.io when one is
provided.
