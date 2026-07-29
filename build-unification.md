# Build unification: dynamic host and native-runtime composition

## Purpose and decision

This is the cross-repository implementation plan for making every supported
MeshLLM build use one product model:

1. a backend-neutral **host** (`mesh-llm`), built with the dynamic native-runtime
   feature; and
2. a selected, versioned **native runtime** bundle containing the backend-specific
   llama/ggml closure.

The user-visible product remains a single installable archive, native package,
Homebrew formula, or OCI image.  The installer composes host and runtime into a
known bundle location; startup discovers and loads that matching runtime.  This
preserves offline and GPU serving behavior while ensuring that a CUDA host can
start `--version` and `client --auto` without `libcuda.so.1`.

This plan applies to `mesh-llm` and `mesh-packaging`, plus the generated
`Mesh-LLM/tap` formula update after a release candidate exists.  It does not
change the already-published 0.74.0 artifacts.  The first release using this
contract must be a new version.

## Evidence driving the change

Distribution certification of `packaging-v0.74.0` found that these published
images exit 127 before `main` on hosts without an NVIDIA driver:

- `0.74.0-ubuntu-amd64-cuda12.9.2`
- `0.74.0-ubuntu-arm64-cuda12.9.2`
- `0.74.0-ubuntu-amd64-cuda13.1.2`
- `0.74.0-ubuntu-arm64-cuda13.1.2`
- `0.74.0-arch-amd64-cuda13.3.1`

The error is `mesh-llm: error while loading shared libraries: libcuda.so.1:
cannot open shared object file`.  The packaged CUDA binary has `DT_NEEDED
libcuda.so.1` and CUDA driver symbols.  Thus this is not Docker assembly: the
static CUDA ggml/llama linkage in the shipped host makes the dynamic loader
require the driver before CLI argument handling.

The normal release path already uses `MESH_LLM_DYNAMIC_NATIVE_RUNTIME=1`, but
the CUDA, ROCm, Vulkan, and several platform build recipes invoke backend
specific static `build-linux.sh` paths.  The divergence is the defect source
and the maintenance cost to remove.

## Target contract (contract v2)

### Host

- Exactly one release host per supported operating-system/architecture pair.
- The host is built with dynamic native-runtime support and carries no GPU
  backend objects or direct `DT_NEEDED`/import dependency on `libcuda`, ROCm,
  Vulkan, Metal framework bindings, or backend-specific llama libraries.
- `mesh-llm --version`, help, one-shot commands, `runtime list`, and
  `mesh-llm client --auto --log-format json --no-console` work on a machine
  with no GPU driver and with no runtime selected.
- A host may report that serving requires an installed compatible runtime; it
  must do so after process startup with an actionable JSON/plain error, not a
  loader error.

### Native runtime

- One immutable runtime artifact for every OS/architecture/backend/backend
  version currently supported by the release matrix.
- Each runtime has a manifest with host-version compatibility, runtime ABI,
  platform, backend, backend version, files, checksums, and optional tool
  capabilities.
- Backend libraries and driver-facing dependencies live here.  CUDA runtime
  closure may reference `libcuda.so.1`; the host closure may not.
- Runtime selection remains deterministic and validates version/ABI/platform
  before loading.  No search of the current working directory is permitted.

### Composed product bundle

```text
mesh-bundle/
  mesh-llm
  native-runtimes/
    <runtime-id>/
      manifest.json
      lib/...
      tools/...                 # only if the runtime owns helper tooling
```

The product bundle manifest/provenance must identify both immutable inputs:
the host digest and the native-runtime digest.  Existing backend-flavored
public archive names remain compatibility aliases, but are composed from the
same host bytes plus a different runtime; they are not separately compiled
applications.

Native packages install the host at the documented executable path and the
runtime below a versioned, package-owned directory.  Homebrew installs the
host in `bin` and runtimes in formula-owned `libexec/native-runtimes`.  OCI
images install a native package containing both, not a raw binary copy.

## Required implementation work

### 0. Freeze the contract with tests before changing artifact production

In `mesh-llm`:

- Add a host dependency-policy test/script that inspects ELF/Mach-O/PE imports
  and rejects GPU backend/driver dependencies from every dynamic host.  Keep a
  separate runtime policy that permits the appropriate backend dependencies.
- Add startup tests using an empty cache and explicit bundle directory.  Cover
  no-runtime client startup, matching bundle discovery, incompatible
  version/ABI rejection, and missing-runtime diagnostics.
- Add an artifact-level assertion that two backend product bundles for one
  OS/architecture contain byte-identical host executables and differ only in
  runtime/provenance.

In `mesh-packaging`:

- Define a checked-in host/runtime/product artifact schema and validate it in
  unit tests.  Record host URL/digest, runtime URL/digest, selected runtime ID,
  upstream version, backend, and backend version.
- Extend archive validation beyond the former single-file
  `mesh-bundle/mesh-llm` contract, with strict allowlists, traversal checks,
  digest verification, and composition provenance.

### 1. Make runtime discovery support installed composition

In `mesh-llm`, extend `mesh-llm-native-runtime` and
`mesh-llm-runtime-install` so startup searches, in an explicit documented
order:

1. an explicit CLI/config/environment bundle-directory override;
2. the runtime directory adjacent to the installed executable or portable
   bundle;
3. the package-owned versioned native-runtime directory;
4. the Homebrew formula `libexec` directory;
5. the existing user cache/downloaded runtime location.

The exact names and precedence are a public contract, tested on each platform.
Use canonicalized, manifest-validated directories; never infer from the
working directory.  Keep existing cache/download behavior as fallback, and
make an offline composed bundle work without networking.

### 2. Move all backend code behind the runtime boundary

Make dynamic native runtime the standard release, SDK, installer, package,
image, and developer build path.  Move every host-linked CUDA/ROCm/Vulkan/
Metal dependency into native runtimes.  In particular, refactor GPU benchmark
code that currently links CUDA (`gpu-bench`/build-script linkage) into a
runtime-provided helper or a loaded runtime API.  Apply the same rule to
`skippy-quantize`, llama-spec tools, and any shipped developer/SDK executable
that can pull backend objects into the host.

`skippy-ffi` and runtime-loading abstractions must use the dynamic boundary for
all supported build profiles.  Remove static backend build/link paths only
after the replacement and parity tests exist.  The branch-local patched
llama.cpp static build is no longer a normal development loop; native-runtime
development builds a local versioned runtime and launches the dynamic host
against it through an explicit override.

### 3. Replace divergent scripts and Just recipes with one build graph

In `mesh-llm`, create/retain three narrowly defined commands behind `just`:

| Layer | Responsibility | Output |
| --- | --- | --- |
| host | build one backend-neutral executable per OS/arch | host artifact + import report |
| runtime | build/package one native backend runtime | manifest, libraries, tools, checksums |
| product | verify and compose a host plus one runtime | portable `mesh-bundle` + provenance |

Public `just` recipes for CPU, CUDA, ROCm, Vulkan, Metal, release, debug,
SDK, and installer testing must orchestrate this graph rather than invoke
separate static compiler lanes.  Preserve ergonomic backend/profile commands,
but make them selections of a runtime.  Do not reintroduce an external
`llama-server`/`rpc-server` lane.

Local development must have a documented fast path that reuses a locally built
runtime, a fully isolated cache/runtime directory, and explicit environment
selection.  It must exercise the same host/runtime discovery and manifest
validation as release products.

### 4. Refactor MeshLLM release production and SDK packaging

- Replace backend-specific host release jobs with a host OS/architecture
  matrix, a native-runtime matrix, and a product-composition matrix.
- Upload signed/checksummed/provenance artifacts for each layer, then publish
  product aliases only after both input digests and compatibility metadata are
  verified.
- Retain expected asset names/releases for installers and external automation
  during a documented compatibility window.  Generate them from composition;
  never rebuild the host for an alias.
- Update installer code to preserve `native-runtimes/` from the portable bundle
  and configure discovery correctly after installation and upgrade.
- Update Node SDK/native addon packaging so the addon loads a dynamic host and
  its selected runtime through the same resolver.  Ensure `currentMeshVersion`
  remains the product version and add darwin-arm64 addon/runtime load coverage.

### 5. Refactor `mesh-packaging` release composition

Replace the current “download a backend-specific upstream binary, wrap it”
flow with:

1. resolve and verify one immutable host artifact for the package OS/arch;
2. resolve and verify the corresponding immutable runtime artifact;
3. compose the verified inputs under the package staging root;
4. build the native package from that staging root;
5. build the final OCI image by installing exactly that package;
6. emit SBOM, package checksum, provenance, and OCI labels naming both input
   digests and the selected backend/runtime.

Update the Homebrew generator/template to install the Apple Silicon host plus
runtime directory in `libexec`; its formula must continue to download the
upstream vN archive and carry the exact composition digest.  Update Debian,
Arch, and image QA so they prove package ownership of both host and runtime.

For every OCI image, test explicit-platform `--version` and no-device client
startup.  CUDA/ROCm/Vulkan absence is not a skip condition for that test.  A
separate hardware-qualified job may test backend load/serving with the proper
device runtime (`--gpus all` for CUDA), but must not replace no-driver image
certification.

## Documentation, skills, and agent instructions (required deliverables)

### `mesh-llm`

| Location | Required update |
| --- | --- |
| `AGENTS.md` | Replace the static debug/release split guidance with the host/runtime model, command table, local runtime override, dependency-policy checks, and no external-server rule. |
| `.github/AGENTS.md` | Update workflow routing/ownership if host, runtime, compose, installer, or SDK jobs move; keep it aligned with the root guidance. |
| `.agents/skills/manage-ci/SKILL.md` | **Before editing workflows**, update its prescribed inventory/change procedure for the three-layer matrix and artifact contract. |
| `.agents/skills/manage-ci/references/current-inventory.md` | Regenerate the complete workflow/job/artifact table after each CI transition; include host, runtime, composition, no-driver smoke, and hardware-qualified jobs. |
| `ci/ci.md` | Document required checks, dispatch inputs, artifacts, provenance handoff, and release gates. |
| `CONTRIBUTING.md` and root `README.md` | Explain developer builds, local native-runtime selection, offline bundle behavior, and user-visible install/run expectations. |
| `RELEASE.md` | Define host/runtime/product promotion, compatibility aliases, checksum/SBOM/provenance requirements, and rollback. |
| `docs/design/NATIVE_RUNTIMES.md` | Make the discovery order, directory layout, manifest/version/ABI contract, cache fallback, and security validation normative. |
| `docs/design/TESTING.md` | Add no-driver host/product smoke, GPU-qualified runtime smoke, multi-platform product checks, and cleanup/evidence expectations. |
| `docs/README.md` plus affected SDK/installer docs | Keep the documentation map and cross-links accurate; add a migration note for developers. |
| CI/release/installer/SDK tests and source comments | Remove stale statements that a static backend host is expected or that `--headless` is a quiet non-TUI mode. |

### `mesh-packaging`

| Location | Required update |
| --- | --- |
| `AGENTS.md` | Replace “final binary per matrix row” architecture guidance with host matrix + runtime matrix + product-composition pipeline, including provenance and no-driver QA. |
| `.skills/distribution-certification/SKILL.md` | Revise release inventory and certification to download/verify host and runtime inputs, assert package ownership/layout, distinguish no-driver client readiness from GPU qualification, capture composed provenance, and preserve pre-existing state. |
| `README.md` and `docs/native-packages.md` | Document composed package layout, host/runtime checksums, and package/image construction. |
| `docs/gpu-runbooks.md`, `docs/release-checklist.md`, `docs/matrix.md`, `docs/tagging.md`, `docs/publishing.md` | Remove static-host assumptions; document the split between universal client smoke and device-qualified runtime smoke. |
| `docs/packaging-readiness-gaps.md`, `docs/packaging-readiness-scorecard.md`, `docs/package-signing.md`, `docs/runner-capacity.md` | Update readiness, signing/provenance, and runner requirements for the three artifacts and two test tiers. |
| `packaging/native/README.md`, Homebrew README/template docs, Docker QA docs/scripts | Describe and enforce package-owned runtime composition; remove the single-binary extraction claim. |
| `.github/workflows/*` and tests | Implement/test host/runtime/product validation; update `images-precheck`, release workflow, archive tests, package tests, formula tests, and image QA. |

### Cross-repository documentation controls

- Before merging, run a repository-wide stale-claim audit (for example,
  `rg` for `static`, `build-linux.sh`, `backend-specific binary`, raw
  `mesh-bundle/mesh-llm`, and `libcuda`) and resolve or intentionally retain
  every match with current wording.
- Version the artifact schema/manifest and link both repositories to the same
  normative contract.  If a shared schema cannot live in one repository,
  duplicate only the machine-readable schema and add a CI drift check.
- The distribution-certification skill is a release gate: update it in the
  same pull request as the packaging workflow/contract changes, not later.

## Verification and release gates

### Required automated coverage

1. Unit/integration tests for resolver precedence, compatibility rejection,
   manifest/checksum validation, and installed-bundle discovery.
2. Host import/closure checks for Linux amd64/arm64, macOS arm64, and Windows
   where supported; failures name the unexpected library.
3. Runtime closure checks per backend, allowing only declared backend imports.
4. Portable bundle, Debian, Arch, Homebrew, npm SDK, and OCI package-layout
   tests that prove host and runtime are owned and co-located.
5. No-driver/no-device `--version`, `runtime list`, and JSON client readiness
   smoke for every product/image platform.  Assert the process remains alive,
   emits a real ready event, and stops cleanly.
6. Hardware-qualified backend tests on suitable runners for CUDA, ROCm,
   Vulkan, and Metal; they test runtime load and a minimal backend operation.
7. Existing mesh protocol, installer upgrade, SDK addon, SBOM, provenance,
   checksum, package-manager, Homebrew audit/test, npm, and Docker label
   coverage remains green.

### Manual release-candidate certification

Run the updated distribution-certification skill against published
release-candidate artifacts.  It must use unique directories/ports/container
names, not disturb pre-existing services/packages, start noninteractive CLIs
with `--log-format json`, and produce a report with artifact URLs, digests,
readiness evidence, endpoint/status, and cleanup.  Use the remote observable
process skill whenever the certification starts or supervises a remote SSH
process.  The report has separate rows for package format correctness and GPU
runtime qualification.

## Delivery sequence and pull requests

Keep changes reviewable and preserve a bisectable transition.  Suggested order:

1. **MeshLLM foundation PR:** this plan, normative runtime contract, resolver
   tests, host import policy, and docs/agent/CI-inventory updates required by
   the foundation.
2. **MeshLLM implementation PR(s):** dynamic host migration, GPU helper
   boundary, unified build graph, installer/SDK changes, and associated tests.
   Split by independently testable ownership if needed.
3. **Packaging contract PR:** schema, archive verifier, package staging,
   native/Homebrew/Docker tests, packaging `AGENTS.md`, docs, and updated
   distribution-certification skill.
4. **Release workflow PRs:** MeshLLM host/runtime/product publishing first,
   then packaging consumption/composition.  The consumer must support the
   producer contract before the producer switches a stable release.
5. **Candidate evidence/compatibility PR:** run the full certification on a
   non-published candidate or pre-release artifacts, record results, and make
   only evidence/documentation corrections.  Do not alter release services.
6. **Tap PR:** regenerate/update `Mesh-LLM/tap` only after the composed Apple
   Silicon archive is immutable and verified; preserve the formula version and
   checksum provenance.

Each PR must touch only its owning repository, have a separately staged
commit, update its relevant inventory/docs/agent instructions, and pass its
local required checks.  Never commit unrelated pre-existing working-tree
changes.

## Compatibility, rollback, and completion criteria

- Current public archive names and installer selection behavior remain valid
  during the transition.  New composition provenance makes it possible to
  diagnose host/runtime mismatches without rebuilding a host.
- Rollback is selecting the prior immutable product artifact; do not mutate or
  retag a released host/runtime.  Keep the legacy installer handling only for
  the declared support window, with a removal issue and test.
- Success requires byte-identical host artifacts across backend products,
  no-driver CUDA image client readiness, matching runtime load on qualified
  hardware, package ownership of both layers, correct npm addon load, and a
  full distribution certification PASS.  Any missing/corrupt/mislabeled layer,
  host backend linkage, readiness failure, or unexplained schema drift is a
  release failure—not a reason to fall back to the old static route.
