# Releasing mesh-llm

A release is one MeshLLM host per platform, plus the Swift, Kotlin and Node
SDK artifacts. MeshLLM does not build inference runtimes: each archive carries
the default Skippy runtime for its platform, downloaded from the Skippy release
this source pins, and the host downloads a GPU runtime from Skippy when a
machine needs one.

| Platform | Archive | Bundled runtime |
|---|---|---|
| macOS arm64 | `mesh-llm-aarch64-apple-darwin.tar.gz` | Metal |
| Linux x86_64 | `mesh-llm-x86_64-unknown-linux-gnu.tar.gz` | CPU |
| Linux arm64 | `mesh-llm-aarch64-unknown-linux-gnu.tar.gz` | CPU |
| Windows x86_64 | `mesh-llm-x86_64-pc-windows-msvc.zip` | CPU |

Every archive is also published under a versioned name
(`mesh-llm-<tag>-<target>`), and each has a `.sha256` file.

## Not ready to release

`release.yml` refuses a real release in its first job, `release_readiness`,
while any of the following is open. A canary prints the list and continues.
An entry is removed in the same change that closes it.

- **Windows inference.** The packaged Windows host is version-checked but not
  run against a model.
- **Host import check.** `scripts/check-host-imports.sh` refuses a host that
  links CUDA, ROCm, Vulkan, Metal, ggml or llama, or that needs a newer glibc
  than the Linux floor, and writes `host-imports.json` into the archive with
  the SHA-256 of the signed host. A report exists only for a host that passed,
  so its `rejected_imports` is always empty. Its macOS branch has run on a real host. The
  Linux branch has run only against stubbed tools. Nothing has run on Windows,
  including the call from `package-release.ps1`.
- **Docker images and npm.** The handoff to `Mesh-LLM/mesh-packaging` is off.
  That repository reads `native-runtimes.json` and the GPU archives from the
  MeshLLM release, which no longer has them. A replacement contract is
  drafted in `mesh/docs/design/PACKAGING_HANDOFF.md`, and the receipt wait is
  back as `scripts/wait-packaging-receipt.sh`, tested against a stub. The
  dispatch itself is not wired, and mesh-packaging has not adopted the draft.
- **NVIDIA and AMD machines.** No run on real hardware shows the host
  downloading and loading a GPU runtime. Tests cover the choice only: a better
  runtime in the catalog, or one already installed, is used over the CPU
  runtime bundled in the archive. Metal was run on a Mac.
- **Updating from 0.80.0 on Windows.** The updater swaps in the new runtime
  and manifest with a PowerShell script that no test runs. The Unix path is
  covered. See [Compatibility archive names](#compatibility-archive-names).

Two more things stop a release before it publishes:

- **The pinned Skippy release must remain available.** The build downloads
  the runtime from `https://github.com/Mesh-LLM/skippy/releases` and stops
  when it cannot.
- **`runtime_download_gate`** runs the runtime download tests, including the
  check that refuses an archive whose runtime differs from its catalog entry.

## Dispatch

Merge a pull request that sets the new version first (run
`scripts/release-version.sh v0.81.0` and commit the result), as for 0.79.0 and
0.80.0. The rules on `main` require a pull request, and the workflow's token
may not bypass them, so the workflow cannot push the version commit itself.
With the version already on `main` it has nothing to commit and carries on.

```bash
just release v0.81.0            # real release, from main
just release v0.81.0 --canary   # rehearsal: builds, packages and smokes; pushes and publishes nothing
```

`scripts/release.sh` dispatches `release.yml` on `main` with two inputs,
`version` and `canary`. The workflow then:

1. **Prepares the source.** It sets the version with
   `scripts/release-version.sh`, commits `<tag>: prepare release source`, and
   pushes it to `main`. After the version pull request above there is nothing
   to commit. A canary leaves `main` alone.
2. **Builds each platform.** It downloads that platform's default runtime with
   MeshLLM's own installer, builds the host with `just release-build`, signs
   and packages it with `just release-bundle`, then unpacks the archive it will
   publish and checks the signature, the signed commit and target, that the
   host links no inference backend and stays within the Linux glibc floor,
   that `host-imports.json` in the archive describes that exact host, and
   (except on Windows) a real inference request.
3. **Builds the SDKs**, each in its own job: the Swift XCFramework and
   `Package.swift`, the Kotlin native library for three targets, and the Node
   addon for four.
4. **Publishes.** It verifies every checksum, adds the compatibility archive
   names, pushes the tag and creates the GitHub release with generated notes.

The tag is one commit past `main`. That commit adds `Package.swift` for the
published XCFramework, the generated Swift binding and the console assets for
the SDK packages, because Swift Package Manager reads them from the tagged
source.

Crates are not published to crates.io by this workflow, and there is no
rolling preview.

## Compatibility archive names

For one transition release only, the Linux and Windows host archives are also
published under the CUDA and ROCm archive names that 0.80.0 used (`-cuda-12`,
`-cuda-13`, `-rocm`). A 0.80.0 install on one of those GPU machines asks for
those names when it updates. The Vulkan alias is withheld because Skippy
v0.80.3 publishes no Vulkan runtime; Vulkan users must update manually after
building or obtaining a compatible runtime.

Each one is a byte-for-byte copy of the plain host archive. It is not a GPU
build: it bundles the CPU runtime and says so in its manifest. After the
update, the host downloads the recommended GPU runtime from Skippy when it is
online. A machine that runs offline or with downloads turned off needs a
matching GPU runtime installed first (`mesh-llm runtime install`), otherwise it
runs on CPU. The update does not keep the old GPU runtime.

Remove the step `Offer supported 0.80.0 GPU archive names`
from `release.yml` after that release.

## Prerequisites

- Repository secrets `MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE` and
  `MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE` (below).
- Repository secret `MESH_LLM_POSTHOG_KEY` (optional, below).
- A published Skippy release whose version is the pinned crates.io version
  (`skippy-native-runtime` in `Cargo.lock`). That version is the runtime
  release: hosts download its runtime and never reuse one cached by an earlier
  Skippy. A host upgraded without network access needs that runtime supplied
  first (`MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR`).
- `gh` authenticated with permission to dispatch workflows on `main`.

## Release Attestation Signing Keys

The release workflow signs each packaged `mesh-llm` executable with these
repository Actions secrets. A release fails if either is missing; a canary
packages unsigned without them.

- `MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE`
- `MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE`

The secret values are the full JSON contents of the release-attestation private
and public key files, not paths to files. Generate a production keypair with:

```bash
umask 077
mkdir -p /tmp/mesh-release-attestation
cargo run -q -p mesh-llm-release-attestation -- generate-keypair \
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
cargo run -p mesh-llm-release-attestation -- inspect \
  --binary /tmp/test-bundle/mesh-llm \
  --public-key-file /tmp/mesh-release-attestation/mesh-release-attestation-public-key.json \
  --json
```

The reported status must be `valid`, and `commit` and `target_triple` must be
the release commit and the archive's Rust target. A `missing` status means the bundle was
published without an embedded release-attestation footer. An `invalid` status
means a footer was present, but signature verification failed.

## Building a release archive locally

```bash
export MESH_LLM_NATIVE_RUNTIME_DIR=/path/to/a/skippy/runtime   # holds manifest.json
just release-build
just release-bundle v0.81.0 dist
```

Set the two `MESH_RELEASE_ATTESTATION_*_KEY_FILE` variables to key file paths,
and `MESH_RELEASE_COMMIT` to the full commit, to sign the host. Without them
the archive is unsigned. To run a model through the packaged host:

```bash
MESH_LLM_BIN=unpacked/mesh-bundle/mesh-llm \
  scripts/ci-smoke-test.sh unpacked/mesh-bundle/native-runtimes model.gguf
```

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
