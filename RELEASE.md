# Releasing mesh-llm

## Cut, test, promote

A release is two buttons. **Cut** builds the release once. **Promote** makes
that same build the real release. Nothing is rebuilt in between, and nothing
needs a pull request first.

### 1. Cut a candidate

```bash
git pull --ff-only
just release 0.81.0
```

or run the **Release** workflow (`.github/workflows/release.yml`) from `main`
in the Actions UI with `version` set to `0.81.0`.

The workflow freezes the `main` commit it was dispatched from, stamps the
version on it with `scripts/release-version.sh`, builds and verifies every
platform bundle, pushes the tag `v0.81.0`, and publishes the GitHub release as
a **pre-release**. That pre-release is the release candidate: it already has
the final version, the final files and the final download URLs.

- The version is always plain `MAJOR.MINOR.PATCH`. There are no `-rc` or
  `-preview` versions.
- A cut never writes `main` and never reads the version recorded on `main`,
  so branch rules cannot stop it. `main` keeps whatever version it last
  recorded; the release version comes only from the workflow input.
- Nobody is moved onto a candidate. Installers and auto-update follow GitHub's
  **latest** release, and a candidate is not latest.
- SwiftPM reads git tags, so a Swift consumer who asks for `0.81.0` can resolve
  a candidate before it is promoted.

### 2. Test the candidate

```bash
curl -fsSL https://meshllm.cloud/install.sh | bash -s -- --pre-release
```

`install.ps1 -PreRelease` and `MESH_LLM_INSTALL_PRERELEASE=1` do the same.
They install the newest published pre-release.

### 3. Promote it

```bash
gh workflow run release-promote.yml --ref main --raw-field tag=v0.81.0
```

or run **Release · Promote** from `main` in the Actions UI. Promote takes
minutes. It checks that the candidate is complete (published, every file
uploaded, every CPU and GPU runtime present), marks it the latest stable
release, and then runs three independent follow-ups:

- regroup the release notes;
- dispatch `Mesh-LLM/mesh-packaging` for the native package assets, the GHCR
  image matrix and the Node SDK on npm, and wait for its receipt;
- publish the Rust SDK crate chain to crates.io.

If one follow-up fails, the release is still out. Use **Re-run failed jobs** on
that Promote run; it repeats only the failed job. Do not dispatch Promote again
for the same tag.

### One step instead of two

```bash
just release 0.81.0 --promote
```

or tick `promote` on the Release workflow. Promote then runs as soon as the
candidate is published.

### A bad candidate

Do not promote it. Fix `main` and cut the next patch version, such as
`0.81.1`. Tags and their files are never replaced. The rejected candidate
stays on the Releases page as a pre-release; delete it there if it should not
be installable with `--pre-release`.

### Other inputs

`canary` builds and smoke-tests everything without publishing a tag or a
release. `skip_gpu_bundles` cuts a candidate without the GPU bundles; such a
candidate cannot be promoted.

GitHub generates the release notes against the highest stable release below
the target version.

Do not use GitHub's bare **Draft a new release** form as an alternate release
path. It bypasses the verified artifact graph. The Release workflow is the only
supported GitHub release publisher; use the Actions UI or `just release` to
dispatch it from `main`.

The sections below document the underlying steps, workflow debugging, and local
bundle validation.

## Prerequisites

- `just` installed
- Rust toolchain installed
- `cmake` and a native compiler installed
- Node/npm installed for the UI build
- `gh` CLI authenticated if publishing manually
- `MESH_AGENT_IMAGES_DISPATCH_TOKEN` configured as a fine-grained repository
  token or GitHub App token with Contents write and Actions read access to
  `Mesh-LLM/mesh-packaging`. These permissions allow the workflow to dispatch
  packaging and read its readiness artifact. Before dispatch, the workflow
  checks repository push permission and probes the run-list endpoint without
  printing the credential. Public run-list access does not prove that the token
  can download an artifact; a denied artifact download fails the wait. The
  legacy secret name is retained so existing release environments do not
  require a coordinated secret rename.

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

`just build` first builds and packages one native Skippy runtime and the
standalone `skippy` CLI, then builds the console and backend-neutral dynamic
MeshLLM host beside them. It uses the
same host/runtime boundary as a release product. Static llama.cpp compilation
is confined to the runtime packaging primitive, never the host executable.

The source tree contains two products. Skippy has one backend-neutral CLI per
OS/architecture and a manifested native runtime per backend. MeshLLM has its
own backend-neutral host and console and consumes those Skippy runtime assets.
Release CI builds the Skippy CLI once per platform and publishes a distinct
`skippy-<version>-<target>-cli.tar.gz` archive and checksum alongside native
runtime archives. MeshLLM bundles retain their own host, selected runtime,
and product manifest. Both products currently share the workspace version and
release tag. External `llama-server`, `rpc-server`, and `llama-moe-*` binaries
are not packaged.

## Bundle

```bash
just bundle
```

This creates `/tmp/mesh-llm-bundle.tar.gz` for local deployment. Platform
release archives contain `mesh-llm`, `host-imports.json`,
`product-manifest.json`, and exactly one
`native-runtimes/<runtime-id>` directory. Backend-flavored archive names select
different runtimes while retaining byte-identical host input for an OS/arch.

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

For an explicit backend, use its release-build recipe to build the selected
runtime and Skippy CLI before the neutral MeshLLM host, then compose:

```bash
just release-build-cuda
just release-bundle v0.X.Y dist cuda
```

Before dispatching a release that should be consumable through SwiftPM,
prepare the Swift binary target manifest on macOS and land the resulting
`Package.swift` change on `main`. This is local preparation only; it publishes
nothing, and the release itself is still cut by dispatching the Release
workflow, which creates the tag:

```bash
scripts/prepare-swift-package-release.sh v0.X.Y
git add Package.swift mesh/sdk/swift/Sources/MeshLLM/Generated/mesh_ffi.swift
git commit -m "v0.X.Y: prepare Swift package artifact"
```

The release workflow invokes the shared typed Swift SDK producer in exhaustive
`full` mode to build `MeshLLMFFI.xcframework.zip`. That same producer is used
in `host-only` mode for PR iteration and `full` mode on main. It verifies the
exact platform and architecture slices plus the macOS framework layout, runs a
zipped-artifact SwiftPM consumer smoke, and checks that the tagged
`Package.swift` already points at the exact release URL and checksum. The
producer also uploads the generated `mesh_ffi.swift` as a separate immutable
companion artifact. Main release builds fail when that generated binding drifts
from the tracked source. If the generated binding is stale, or if the checksum
does not match the artifact built in release CI, the release fails before
publishing.
Producer and smoke use the pinned `macos-15` image and an explicit native/Xcode
cache epoch. Downstream Swift smoke consumes both verified producer artifacts
and never compiles an XCFramework replacement.

Native SDK release archives use the same typed `native-sdk-artifact.yml`
producer as PR and main Kotlin validation. Callers select an explicit target,
backend, Cargo profile, and bounded runner size; they cannot provide a runner
label or Depot-cache permission. Each Linux release invocation first nests the
shared `static-abi-artifact.yml` producer on the matching native architecture,
then restores its checksummed/stamped CPU ABI into the normal native-SDK
`--build` path so only the Rust FFI compilation remains. Both reusable
producers derive architecture-specific hosted/Depot placement and cache
authority from the protected workflow's repository/event/ref policy. Release
enables native runtime crate staging on the same verified archive path. The
producer keeps each
`release-native-sdk-<platform>-<backend>` artifact flat with exactly one
archive, its checksum sidecar, and its target-specific `.crate`, preserving the
published asset names while Kotlin smoke remains a no-build consumer.

For `workflow_dispatch` releases, the release workflow computes the SwiftPM
checksum from the XCFramework artifact it just built, carries both the patched
`Package.swift` and the producer's exact generated `mesh_ffi.swift` into the
release workspace, and creates the requested release tag at that prepared
source commit before publishing.

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

The Release workflow owns release archive production, but it does
not publish OCI images.
`Mesh-LLM/mesh-packaging` is the canonical package, GHCR, and npm producer. It
starts only when Promote marks a candidate with the complete CPU/GPU archive
set as the stable release. An unpromoted candidate never dispatches it. The upstream
`docker.yml` workflow performs Dockerfile validation only and is not a
distribution channel.

The dispatch binds the published tag commit and native runtime manifest digest
to a Promote run-attempt correlation ID. A separate job waits up to two hours
for the matching downstream run and successful readiness receipt. It checks
every requested publication result and fails on ambiguous or stale evidence.
Retrying the wait job does not dispatch packaging again.

If the downstream dispatch preflight fails, update the repository Actions secret
`MESH_AGENT_IMAGES_DISPATCH_TOKEN` with a fine-grained token or GitHub App token
that has Contents write and Actions read access to `Mesh-LLM/mesh-packaging`, then retry the
failed dispatch job. A repository secret's presence does not prove that its
credential can write the target repository. Do not put the token in workflow
logs or command output.

Promote also publishes the Rust SDK crate
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

The script builds `mesh/crates/mesh-llm-ui/dist` in release mode and copies it to
the canonical SDK resource locations: `mesh/sdk/node/console`,
`mesh/sdk/swift/Sources/MeshLLM/Resources/Console`, and
`mesh/sdk/kotlin/src/main/resources/mesh-llm/console`.

Workflow-dispatch releases generate and force-add these resources into the
release tag commit automatically.

The chain currently publishes:

1. `skippy-model-ref`
2. `mesh-llm-identity`
3. `mesh-llm-protocol`
4. `mesh-llm-routing`
5. `mesh-llm-types`
6. `skippy-model-artifact`
7. `skippy-model-hf`
8. `mesh-llm-client`
9. `mesh-llm-api-client`
10. `mesh-llm-node`
11. `mesh-llm-api-server`

Run the consistency check and dry-run before cutting a release after changing
SDK crate manifests or workspace-internal SDK dependencies. The consistency
check keeps the scripted publish order, workspace path dependency versions,
publish metadata, bundled file includes, and CI release preflight in sync. On
the first release that introduces a new internal SDK crate, the dry-run
validates packages whose registry dependencies already exist and reports
downstream packages that will be fully verified during the real sequential
publish after their upstream crates land.

If crates.io rate-limits the publish chain after some crates
have already uploaded, re-run the failed crates.io job of the Promote run, or
rerun `scripts/publish-crates.sh` for the same checked
out release tag, instead of recutting the GitHub release or moving the tag. The
script relies on `cargo publish` to report crate versions that were already
uploaded, continues past those already-uploaded crates, and retries HTTP 429
new-crate rate-limit responses using the retry time from crates.io when one is
provided.

## Analytics key

Release binaries report anonymous usage only when `MESH_LLM_POSTHOG_KEY` is
present at compile time. The release workflow supplies it from the
`MESH_LLM_POSTHOG_KEY` repository secret; `scripts/build-host.sh` prints
whether a key was present (never the key itself), so a release log confirms
it. A build without the secret is valid and reports nothing unless someone sets
`MESH_LLM_POSTHOG_KEY` in the environment at run time, which is the documented
self-hosting path.

Forks build without the secret and therefore never report. See
`website/src/docs/pages/analytics.md` for the collected surface and the
opt-out, and `crates/mesh-llm-analytics/README.md` for the design.
