# MeshLLM CI inventory

Read this inventory with `SKILL.md` before editing CI. It records the
checked-in contract, not guaranteed live GitHub state. Verify live state with
the commands at the end before operational changes.

## Workflow ownership

| Workflow | Trigger | Ownership |
| --- | --- | --- |
| `pr_quality.yml` | PR, main push, dispatch | Formatting, affected-crate Clippy, UI quality, CLI/docs synchronization, quality summary |
| `pr_builds.yml` | PR, dispatch | Cross-platform build/test matrices, native backends, artifact producers, integration/smoke consumers, stable aggregate summary |
| `pr_website.yml` | PR, dispatch | Public website build canary and summary |
| `pr_cleanup.yml` | PR close via `pull_request_target`, dispatch | Positively matched PR cache/artifact cleanup only; never executes PR code |
| `pr_auto_assign.yml` | PR lifecycle via `pull_request_target` | PR metadata assignment only; never executes PR code |
| `ci.yml` | Main push, dispatch | Trusted main build/test/smoke lanes; composed host/runtime products and no-driver client readiness |
| `docker.yml` | Dispatch | Manual client Dockerfile validation; does not publish |
| `docker-precheck.yml` | Reusable call | Shared Docker validation precheck |
| `smoke.yml` | Reusable call | Artifact-based inference/OpenAI/split smoke |
| `scripted-binary-smoke.yml` | Reusable call | Artifact-based scripted/two-node smoke |
| `sdk-smoke.yml` | Reusable call | Artifact-based Rust, Kotlin, and Swift SDK smoke |
| `native-sdk-artifact.yml` | Reusable call | Typed target/backend/profile native SDK producer with protected runner/cache policy |
| `static-abi-artifact.yml` | Reusable call | Typed target/backend static llama ABI producer with protected runner/cache policy |
| `swift-sdk-artifact.yml` | Reusable call | Typed host-only/full Swift XCFramework producer |
| `node-sdk-addon-artifact.yml` | Reusable call | Five-target Node native-addon producer with fresh-install smoke, manifest, and checksum |
| `hf-download-smoke.yml` | Reusable call | Hugging Face download smoke |
| `nightly-stability.yml` | Schedule, dispatch | Nightly operator entry point |
| `nightly-stability-run.yml` | Reusable call | Stability probes and evidence |
| `llama-upstream-canary.yml` | Schedule, dispatch | Upstream llama.cpp compatibility canary |
| `depot-registry-canary.yml` | Dispatch from `main` | Five-pair fresh-runner comparison of one digest-pinned upstream image and its Depot pull-through mirror |
| `queue-unsloth-layer-packages.yml` | Schedule, dispatch | Hugging Face layer-package job queueing |
| `windows-warm-caches.yml` | Main path push, dispatch | Trusted Windows ABI cache warming |
| `website-pages.yml` | Main website path push, dispatch | Public website Pages build/deploy |
| `fly-deploy-console.yml` | Dispatch | `fly-console` environment deployment |
| `release.yml` | `v*` tag, dispatch | Release builds, attestations, publishing, stable-only downstream package/image/npm dispatch |
| `reset-caches.yml` | Confirmed dispatch | Destructive repository cache reset |
| `stale-prs.yml` | Schedule, dispatch | PR warning/closure maintenance |

`release.yml` uses the contract-v2 three-layer artifact graph:

- host matrix: one dynamic, backend-neutral executable and import report per
  supported OS/architecture;
- native-runtime matrix: one manifested/checksummed runtime per supported
  OS/architecture/backend/backend-version;
- product matrix: digest-verified composition of a host plus one runtime while
  retaining existing backend-flavored public archive names as aliases.

The host producer attests the binary, writes `host-imports.json`, and publishes
its checksum. Product consumers verify that immutable input before composition;
they must not re-stamp or otherwise mutate the host per backend alias. Release
CPU and backend product consumers also perform a noninteractive JSON client
readiness smoke from the verified host/runtime inputs before publication. The
background client must exit cleanly after bounded SIGTERM on Unix or
CTRL_BREAK_EVENT on Windows; interactive Ctrl-C coverage remains separate. CI
and packaging consumers must not rebuild either input. Portable bundles
place runtimes at `mesh-bundle/native-runtimes/<runtime-id>`; Debian/Arch
packages use `/usr/local/lib/mesh-llm/<version>/native-runtimes`; Homebrew uses
formula-owned `libexec/native-runtimes`.

Release also fans out `node-sdk-addon-artifact.yml` across Darwin ARM64/x64,
Linux ARM64/x64, and Windows x64. Each producer compiles once on its matching
GitHub-hosted platform, packs and fresh-installs the SDK, verifies
`currentMeshVersion()`, and emits a versioned addon archive with a strict
manifest and SHA-256 sidecar. The release publisher requires all five producers
and attaches those exact artifacts; downstream packaging verifies and assembles
them without recompiling native source.

Only a successful, complete stable release dispatches downstream package,
image, and npm publication. Prereleases publish their immutable GitHub Release
inputs but never invoke `mesh-packaging`; this provides a safe artifact
validation boundary without exposing prerelease inputs to production
promotion.

Merged
[`mesh-packaging#16`](https://github.com/Mesh-LLM/mesh-packaging/pull/16)
consumes that release graph without rebuilding the host, runtime, or Node
addons. It uses typed independent selectors, one per-row
product → package → install/QA → final image → exact-image QA chain,
digest-only promotion, and a canonical immutable evidence index. Complete dry
rehearsal
[30593548823](https://github.com/Mesh-LLM/mesh-packaging/actions/runs/30593548823)
passed 41 jobs with 15 intentional publication-only skips against
`v0.75.0-rc1`; default-branch precheck
[30595367445](https://github.com/Mesh-LLM/mesh-packaging/actions/runs/30595367445)
passed merge commit `76c619bcdd82773e159248a2282187b0b2973daa`.

The Windows host input also carries the checksum-protected `xtask` executable
that performed producer-side attestation. Windows product composers invoke that
prebuilt verifier for the immutable host instead of compiling workspace code.

`ci.yml` applies the same executable-product rule to trusted main validation.
Linux and macOS build immutable release-profile hosts and separately packaged
CPU or Metal runtimes, then upload complete product-v2 trees from
composition-only jobs. Linux CUDA, ROCm, and Vulkan each use an independent
runtime producer plus a thin composer that downloads the same immutable Linux
host; no backend waits on a matrix-wide fan-in. SDK consumers reuse the
producer's adjacent runtime and fail if CI would silently rebuild it. Kotlin
additionally downloads the verified native SDK runtime built by
`native-sdk-artifact.yml` after that producer restores the shared
`linux_static_abi_input`; it runs in parallel with the Linux product (debug on
PR, release on main). Release nests one `static-abi-artifact.yml` producer per
Linux native target through the same native-SDK workflow. Swift downloads an
immutable XCFramework and exact generated `mesh_ffi.swift` from the shared
`swift-sdk-artifact.yml` producer: PR uses `host-only`, while main and release
use exhaustive `full` mode, all on `macos-15`. Windows likewise builds one
immutable release-profile host, independent CPU/CUDA/ROCm/Vulkan runtime
inputs, and composition-only products. Broad main Rust changes exercise the
Windows CPU product; Windows GPU products remain limited to GPU/backend inputs
or manual dispatch. Every composed backend product requires `runtime list`
plus no-driver client readiness; hosted GPU rows neither inject a driver stub
nor skip startup because no device is present.

The exhaustive Swift producer has a 180-minute main/release cold-start budget
because it serially builds seven Apple target ABIs. PR host-only calls retain
their shorter budget. Exact native ABI and compiler caches remain responsible
for reducing the warm path; the timeout is only the reliability ceiling for an
unseeded cache.

`pr_builds.yml` uses the same split producer/composer shape for Linux CPU/GPU
and macOS Metal products while retaining debug-profile hosts for lightweight
PR iteration. Windows broad-Rust validation stays at lightweight Cargo checks;
`windows_checks` also runs focused `mesh-llm-log-store` artifact-path and
SQLite root/database/WAL/SHM privacy-ACL tests when that crate is affected or
on manual dispatch. The debug host plus CPU or GPU runtime/product graph runs
only for its platform/backend input or manual dispatch. Unsupported macOS CUDA,
ROCm, and Vulkan combinations are omitted rather than emitted as no-op jobs.
`scripts/plan-pr-build-jobs.py` converts the central change signals into one
ordered `required_jobs_json` list. Every conditional PR Builds job routes on
membership in that list and retains normal dependency-success behavior through
`needs`. Its static `PR Builds Summary` job directly needs every other
top-level job and consumes the same plan. It accepts a skipped result only for
an unplanned job and rejects required skips, failures, cancellations, unknown
results, duplicate plan entries, and required IDs outside its needs graph,
making that one non-matrix check the workflow's stable branch-protection
target.

Changes to the central PR/main/release workflow callers or to
`compute-changes` itself fail open to the SDK producer/smoke graph. This keeps
caller-owned mode, timeout, artifact, and trust-policy edits from skipping the
reusable Swift, Kotlin, or Rust SDK contracts they change.

The reusable Swift producer verifies the committed generated UniFFI binding
after both host-only and full builds for PR, main, and tag callers. Only a
dispatched release that deliberately prepares a versioned source tree may
replace the tracked binding before publication.

Local actions:

- `.github/actions/compute-changes` owns path, crate, backend, SDK, UI, website,
  Windows, and docs-only routing outputs.
- `.github/actions/select-ci-runners` routes trusted push/dispatch jobs through
  the Depot gate only for `refs/heads/main`, returns GitHub-hosted labels for
  tags and every other ref, and unconditionally returns GitHub-hosted labels
  for `pull_request` events. Repository ownership and the deprecated
  `DEPOT_PR_RUNNERS_ENABLED` variable do not alter that decision. Its cache
  permission is derived from the same typed trust decision.
- `.github/actions/configure-sccache-gha` exports ephemeral Actions cache
  credentials to the baked `sccache`, permits Depot WebDAV only for an explicit
  trusted call, uses writable job-local disk only for PR events because the
  pinned sccache makes a mixed chain wholly read-only and records rejected
  writes after misses, uses disk-only
  storage if a future pull-request trust context is ever evaluated on Depot,
  and resets counters after configuring the server.
- `.github/actions/capture-sccache-stats` validates and uploads one
  machine-readable sccache evidence artifact per instrumented job or matrix
  row. Evidence is retained for 14 days so cold/warm samples span the configured
  Depot cache-retention window.
- `.github/actions/prepare-host-input` owns Unix neutral-host build, optional
  release attestation, import-policy verification, and checksumming.
- `.github/actions/prepare-windows-host-input` owns the equivalent Windows
  debug/release neutral-host build, optional release attestation, import-policy
  verification, checksum, and verifier artifact.
- `.github/actions/prepare-native-runtime-input` owns runtime build/package
  invocation and the release-grade artifact verifier.
- `.github/actions/prepare-native-sdk-input` owns the native SDK
  prepare-llama/build-llama/mesh-llm-ffi/package chain, verifies the exact
  target/backend/profile manifest, and stages a flat immutable upload. Release
  mode adds the native runtime crate through the same path.
- `.github/actions/prepare-static-abi-input` owns the shared Linux static llama
  ABI build/stamp validation and emits a checksummed, target-described ABI v3
  archive containing only the path-normalized static link closure and portable
  OpenMP metadata. The reusable workflow caches that archive, not the local
  CMake build graph; crate tests and native SDK producers consume it.
- `.github/actions/resolve-native-toolchain-epoch` exports one cache-safe
  identity to both native build stamps and cache keys. Digest-pinned Linux
  containers use their immutable image digest; hosted macOS and Windows jobs
  use the exact runner image revision, with compiler/CMake/Ninja versions added
  where hosted or Depot Linux/macOS toolchains are not otherwise pinned.
- `.github/actions/compose-product-input` verifies producer inputs, creates one
  product-v2 tree without compiling, and runs CLI/client readiness.
- `.github/actions/restore-smoke-inputs` owns producer artifact staging and
  model restoration for smoke consumers.
- `.github/actions/restore-windows-abi-cache` owns the exact Windows CPU,
  CUDA, ROCm, and Vulkan ABI cache identity shared by the trusted warmer and
  PR/main/release runtime producers. The hosted-image epoch, architecture sets,
  and toolchain versions are compatibility boundaries; the action requires the
  key epoch to equal the build-stamp epoch, includes the publication action in
  the key hash, exports one validated absolute path for both restore and save,
  and never uses restore prefixes.
- `.github/actions/save-and-verify-actions-cache` snapshots existing exact
  key/ref cache entries before saving a trusted miss, then requires a new,
  non-empty entry to appear and performs a lookup-only restore with the same
  path/key to prove the current opaque cache version exists. Windows warmers
  therefore fail if
  `actions/cache/save` only warns about a reservation collision without any
  compatible upload becoming available.
- `.github/actions/setup-windows-rocm-sdk` owns reusable Windows ROCm setup.

Routing and test-planning scripts:

- `scripts/affected-crates.sh` computes affected crates and reverse dependents; its fail-open workspace list includes `mesh-llm-log-store`.
- `scripts/plan-pr-build-jobs.py` maps PR change signals to the single ordered
  top-level job plan consumed by both conditional PR Builds jobs and its stable
  summary gate.
- `scripts/plan-clippy-batches.sh` owns weighted Clippy sharding and retains a
  checked workspace-member list for fail-open/all-rust planning.
- `scripts/plan-test-batches.sh` owns weighted crate-test sharding. It derives
  workspace membership from `cargo metadata`; new crates must not be added to a
  workflow-owned test allowlist.
- `scripts/test-portable.sh` owns the portable non-Cargo test aggregate used by
  the local `test-all` path.
- `scripts/summarize-sccache-stats.py` aggregates downloaded sccache JSON
  evidence offline and can enforce the migration hit-rate threshold without
  GitHub or network access.

## Runner and image contract

GitHub-hosted labels currently used:

- Linux AMD64: `ubuntu-24.04`
- Linux ARM64: `ubuntu-24.04-arm`
- macOS: pinned `macos-15`
- Windows: `windows-2022`

Depot labels referenced behind the rollout gate:

- routing/summary: `depot-ubuntu-24.04`
- light build/planning: `depot-ubuntu-24.04-4`
- Rust/native build: `depot-ubuntu-24.04-8`
- measured high-parallelism native build: `depot-ubuntu-24.04-16`

Current PR jobs and non-main refs never select these labels. They use the corresponding
GitHub-hosted label regardless of repository ownership or
`DEPOT_PR_RUNNERS_ENABLED`; that variable is ignored. Trusted main/release jobs
use `DEPOT_RUNNERS_ENABLED`; a trusted
main-ref manual dispatch can set `use_depot=true`. Hardware-qualified GPU
execution is not part of the gate.

The default-branch-selected `native-sdk-artifact.yml` and
`static-abi-artifact.yml` workflows do not accept a runner label or Depot-cache
permission from callers. Each first runs a fixed `ubuntu-24.04` policy job,
validates `runner_size` as `default`, `4`, `8`, or `16`, maps the declared
target to the checked-in AMD64/ARM64 hosted and Depot labels, and grants both
the Depot runner and WebDAV cache only for exact
`Mesh-LLM/mesh-llm` `push`/`workflow_dispatch` calls on
`refs/heads/main` when `DEPOT_RUNNERS_ENABLED == 'true'` or when the immutable
main-dispatch event payload has `use_depot == 'true'`. Pull requests,
`pull_request_target`, tags, feature refs, external repositories, macOS, and a
disabled gate without that authorized canary resolve to a GitHub-hosted runner
with Depot cache permission false. The event-owned manual canary is evaluated
only under the same exact repository/main/dispatch guard and is not a
reusable-workflow input.

The Depot dashboard reports the `Mesh-LLM` GitHub connection active, and
GitHub lists both `depot-managed-runners` and `depot-code-access` installations
for all organization repositories. Live main-ref dispatches now prove that the
public `Mesh-LLM/mesh-llm` repository can allocate ephemeral Depot runners.
The available token cannot re-read organization runner-group settings (GitHub
returns 403), so this operational evidence does not prove the current
repository/workflow allowlist. Inspect that policy with organization-admin
authority before enabling the global gate. The separate `mesh-llm` group owns
the two dedicated GPU scale sets and is not the Depot group.

The manual `depot-registry-canary.yml` is the pull-through adoption boundary.
It accepts only a digest-pinned public reference and a safe relative Depot
repository name on an exact `main` dispatch. Five upstream jobs and five Depot
jobs each receive a fresh ephemeral GitHub-hosted runner. The summary rejects digest
drift and requires at least 20% and 10 seconds of median pull improvement.
`DEPOT_REGISTRY_HOST` supplies the nonsecret organization registry host;
the cached pull step uses GitHub OIDC to mint a short-lived read-only
`depot pull-token`. No stored registry secret is used, and the OIDC permission
is not available to PR code.

Bounded rollout evidence:

- cold and warm six-label canaries
  [30525111329](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30525111329)
  and
  [30525247727](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30525247727)
  passed on Intel `default`/`4`/`8`/`16` and ARM `default`/`8`;
- denied feature-ref
  [30593657371](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30593657371)
  concluded skipped with no Depot allocation; its temporary ref pointed exactly
  at main SHA `851888d0b0ce19916d6b0d7d73ce49246eef67d6` and was removed afterward;
- exhaustive prerelease
  [30586470043](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30586470043)
  completed 55 jobs successfully, including 15 Depot jobs across all six
  labels, and published the complete `v0.75.0-rc1` immutable release graph;
- warm non-GPU release canary
  [30590595090](https://github.com/Mesh-LLM/mesh-llm/actions/runs/30590595090)
  completed with 36 successes, 28 intentional skips, and zero failures. Its
  nine Depot jobs included exact static-ABI cache hits with zero compilation
  and roughly 95% sccache hits in both Linux native-SDK consumers.

Live inspection on 2026-08-02 found `DEPOT_RUNNERS_ENABLED=true`. The checked-in
selector still restricts Depot to eligible trusted `main` push/dispatch jobs,
but `main` has no classic branch protection and the available token cannot
fully inspect the organization runner-group workflow allowlist. Treat that
administrative boundary as unverified until an organization administrator
confirms it.

The checked-out local selector is not the security boundary because PRs can
modify workflow and local-action files. The Depot runner group must use
`restricted_to_workflows=true` and exact default-branch selected-workflow refs.
The initial selected set includes `native-sdk-artifact.yml@refs/heads/main` and
`static-abi-artifact.yml@refs/heads/main` because those reusable workflows
directly allocate eligible Linux runners; a caller-only `ci.yml` entry is not
sufficient.
Credential-bearing `hf-download-smoke.yml`, `smoke.yml`,
`scripted-binary-smoke.yml`, and `sdk-smoke.yml` are deliberately excluded and
fixed to bounded GitHub-hosted labels. `swift-sdk-artifact.yml` is fixed to
GitHub-hosted `macos-15`. No reusable workflow passes caller-provided
runner JSON directly to `runs-on`. PR callers pass no `HF_TOKEN`; trusted
main/release callers may pass it only on the fixed hosted smoke lanes.
Automatic Depot Cache still grants repository-scoped cache authority to the
whole job, so even a trusted reusable caller cannot safely execute untrusted PR
code while that injection is enabled. PRs remain GitHub-hosted.

Legacy/dedicated self-hosted label arrays currently referenced:

- NVIDIA AMD64: `["self-hosted","Linux","X64","amd64","gpu-nvidia"]`
- ARM64: `["self-hosted","Linux","ARM64"]`

ARC scale-set labels for the prebuilt runner rollout:

- `mesh-llm-amd64`
- `mesh-llm-arm64`

`pr_builds.yml` runs `public_runner_image_contract` in the public image when
the runner workflow, cache integration, or cache version changes, plus manual
dispatches. Trusted main `ci.yml` owns `arc_runner_image_contract` on both ARC
labels for the same change class. Untrusted PR-event jobs never request those
labels. The ARC job executes directly in each ephemeral runner pod, verifies
the self-hosted image contract and native architecture, and runs a small Rust
check. It intentionally has no hosted fallback.

Runner images are published from
[`Mesh-LLM/mesh-llm-runner-images`](https://github.com/Mesh-LLM/mesh-llm-runner-images)
as `ghcr.io/mesh-llm/mesh-llm-cuda-runner`. The source repository owns:

- `profiles/common.yml`
- `profiles/backends/{cpu,vulkan,cuda,rocm}.yml`
- `profiles/public.yml`
- `profiles/self-hosted.yml`
- CUDA/ROCm toolchain installers, manifest collection, dependency warming, and
  backend compiler-probe verification
- AMD64/ARM64 CPU, Vulkan, CUDA 12, and CUDA 13 images
- AMD64 ROCm 7.0 and ROCm 7.2 images

Production consumers must use the multi-architecture manifest digest. Tags are
discovery inputs and are mutable absent separately verified registry controls.

Merged runner-images PR
[`#9`](https://github.com/Mesh-LLM/mesh-llm-runner-images/pull/9) changed the
publication control plane without changing those production digests. PRs route
affected families plus a mandatory public CPU AMD64 contract, use BuildKit
cache read-only, and cannot stage or promote. Main pushes stage verified
candidate digests; weekly or explicit manual runs promote a retained cohort.
The reusable family workflow independently derives trusted runner/cache
authority, verifies the requested MeshLLM source revision, uses content-digest
immutable tags, and feeds one serial `latest` cohort reconciliation. Deleted
files are included in affected-family routing.

Its merge commit `4e79e68e22a5ea9bb1eedf9a2a7e7ccfc20b2bca`
completed the trusted main
[run 30522118156](https://github.com/Mesh-LLM/mesh-llm-runner-images/actions/runs/30522118156)
with 35 successful jobs, four intentional skips, and zero failures.

Its exhaustive Dockerfile-change PR
[run 30504335079](https://github.com/Mesh-LLM/mesh-llm-runner-images/actions/runs/30504335079)
completed all 20 platform rows in 6m 22s wall / 1h 13m 07s aggregate with no
Depot jobs and no PR cache export. Treat that as validation-path evidence, not
as proof of the trusted stage/promotion path.

The public repository and its GHCR package have independent visibility. Until
anonymous pull of the package succeeds, GitHub-hosted container jobs must grant
`packages: read` and provide `github.actor`/`secrets.GITHUB_TOKEN` through
`container.credentials`. Do not assume making the source repository public also
makes an existing package public.

The production rollout covers the shared public CPU environment and explicit
public Vulkan, CUDA, and ROCm overlays in `pr_builds.yml`, `ci.yml`,
`pr_quality.yml`, and Linux release jobs. Backend images standardize compilers
and SDKs; actual GPU access remains a separate runner label, node resource, and
trust-boundary contract. Do not route untrusted PR code to persistent GPU
runners merely because the same image can also run as an ARC pod.

The image family built from MeshLLM revision
`5f341d6828fc77cce2f3be43f2a6ff26f3223433` is:

| Image | Immutable index digest |
| --- | --- |
| public CPU | `sha256:8d93de6ba30173e825a16fdecf011f9c632edc6e1259df7289e491b0a05f829d` |
| public Vulkan | `sha256:ce55fed5c680cd3184b5d4770d9a77c43a702687690906e5753efd2cea27ed80` |
| public CUDA 12 | `sha256:c5b85ef527230f77cf9933ef40bcb44316f9bbcb8fd2ce0651b58acda5143dfd` |
| public CUDA 13 | `sha256:6b87598605f5d8deeafecfb1a55027e0ca9e47f4fc6f230d030487c450c31aa6` |
| public ROCm 7.0 | `sha256:0e13e5d2d2c121df265ff6c69be81e468989e09f81d6b7ff049b110cc0bb0d2b` |
| public ROCm 7.2 | `sha256:6b88ca9371ada2c507d6e36b71f0e0538fee378c6a5e2b39c17249b4b7e5088a` |
| self-hosted compatibility | `sha256:37e0ce710eae44952306c4a553cf89fdf94c009660a2a8fa04bba4d202a32baf` |

MeshLLM workflows pin the public digest. The Flux repository must independently
roll the ARC HelmReleases to the paired self-hosted digest; that cross-repository
change cannot be delivered by a MeshLLM pull request.

Public-image Rust jobs use the baked `sccache` binary. Trusted calls to
`configure-sccache-gha` may use Depot's `SCCACHE_WEBDAV_ENDPOINT` plus
`SCCACHE_WEBDAV_TOKEN`/`DEPOT_CACHE_TOKEN` in a fail-open `disk,webdav` chain.
When that permission is false and Depot is detected, the action gives the
sccache server a credential-free environment and uses job-local disk only.
GitHub-hosted trusted jobs retain `disk,gha` or explicit disk-only mode. PR
events use job-local disk only, including direct sccache-action users routed
through the configure action, while trusted main, release, warmer, and dispatch
paths may seed the GHA tier. The high-fanout main `rust_crate_tests` matrix is
explicitly disk-only on GitHub-hosted runners because its four concurrent
per-object writers caused 94% of cold-control GHA write errors; its distinct
bulk Cargo target caches retain cross-run reuse. Other producer and grouped-test
jobs remain remote-enabled. An explicitly authorized Depot call selects
`disk,webdav` before that GHA opt-out. Swift restores a
mode-independent Rust dependency cache that only trusted main pushes save.
The main and PR `rust_crate_tests` shard containing `skippy-runtime` downloads
the public Qwen3 correctness fixture and exposes `SKIPPY_CORRECTNESS_MODEL` to
the crate tests; this fixture does not require `HF_TOKEN`.
Persistent Cargo target and ABI reuse remains owned by
`Swatinem/rust-cache` and `actions/cache`. Current PR jobs use the normal
`mesh-llm` key namespace; native `actions/cache` writes remain merge-ref scoped,
and trusted main does not restore PR-written entries.
Raw native ABI caches also include the exact native toolchain epoch used by the
build stamp. Linux container jobs use the pinned OCI digest, macOS keys include
the hosted image revision and native-tool fingerprint, and Windows warmer,
PR, main, and release jobs share the same hosted image revision. A reported
cache hit is verified against the current build contract before reuse.
A future Depot PR entrypoint must instead use keys that trusted main/release
jobs never restore because Depot cache entries are repository-scoped. Key
separation alone is not a security boundary while the job receives Depot cache
authority, so no such entrypoint is currently eligible.

`USE_SELF_HOSTED` currently controls selected GPU/release routes. Unset or a
value other than the exact string `true` selects the hosted fallback. Any new
route must preserve a safe hosted fallback or document why one cannot exist.

## Repository variables referenced by workflows

All GitHub Actions variables are strings.

| Variable | Purpose and fallback |
| --- | --- |
| `USE_SELF_HOSTED` | Exact `true` selects supported self-hosted GPU/release lanes; otherwise hosted |
| `DEPOT_PR_RUNNERS_ENABLED` | Deprecated compatibility variable; the current selector ignores it and all PR jobs remain GitHub-hosted |
| `DEPOT_RUNNERS_ENABLED` | Exact `true` routes eligible trusted main/release Linux jobs to Depot; otherwise GitHub-hosted |
| `DEPOT_REGISTRY_HOST` | Depot organization registry host (`<org-id>.registry.depot.dev`) used only by the manual pull-through canary; required for that workflow |
| `CUDA_VERSION` | Windows CUDA toolkit selection; Linux CUDA lanes use digest-pinned backend images |
| `VULKAN_SDK_VERSION` | Windows Vulkan SDK; fallback `1.4.328.1` |
| `LLAMA_UPSTREAM_CANARY_SMOKE` | Enables canary smoke; fallback `1` |
| `LLAMA_WINDOWS_CACHE_RETENTION` | Windows warm-cache retention; fallback `2` |
| `PR_CACHE_CLEANUP_WORKERS` | Cleanup fan-out; default `5`, validated range `1..20` |
| `STALE_PR_DAYS` | PR close threshold; fallback `7` |
| `STALE_PR_WARNING_DAYS` | PR warning threshold; fallback `2` |
| `MESH_AGENT_BASE_URL` | Preferred agent smoke endpoint |
| `MESH_AGENT_MODEL` | Preferred agent smoke model |
| `MESH_OPENCODE_BASE_URL` | Legacy/fallback agent smoke endpoint |
| `MESH_OPENCODE_MODEL` | Legacy/fallback agent smoke model |
| `AGENT_SMOKE_LONG_PROMPT_CHARS` | Preferred long-prompt size; falls back through OpenCode value to `65536` |
| `OPENCODE_SMOKE_LONG_PROMPT_CHARS` | Legacy/fallback long-prompt size; fallback `65536` |
| `MESH_NIGHTLY_STABILITY_ENABLED` | Exact `1` enables scheduled stability; dispatch bypasses the gate |
| `MESH_NIGHTLY_STABILITY_BASE_URL` | Nightly endpoint fallback |
| `MESH_NIGHTLY_STABILITY_MODELS` | Model list; fallback `auto,mesh` |
| `MESH_NIGHTLY_STABILITY_ATTEMPTS` | Attempts per model; fallback `5` |
| `MESH_NIGHTLY_STABILITY_AGENT_SMOKES` | Optional agent CLI smoke list |
| `MESH_NIGHTLY_STABILITY_TIMEOUT` | Per-probe seconds; fallback `180` |

## Secret names referenced by workflows

Never record values in this inventory.

| Secret | Consumer and scope |
| --- | --- |
| `HF_TOKEN` | Download, smoke, queue, and release jobs that access Hugging Face |
| `FLY_API_TOKEN` | `fly-deploy-console.yml`; use an app-scoped deploy token and `fly-console` environment |
| `MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE` | Release attestation signing material |
| `MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE` | Release attestation public material |
| `MESH_AGENT_IMAGES_DISPATCH_TOKEN` | Cross-repository release dispatch to image publishing; may be organization-scoped |
| `CARGO_REGISTRY_TOKEN` | Crates.io publishing in release |

`GITHUB_TOKEN`/`github.token` is built in and governed by each workflow/job's
`permissions`. Do not create it as a repository secret.

Fork PRs do not receive repository secrets. A secret absent from the repository
list may be environment- or organization-scoped; verify scope instead of
assuming it is missing.

## Environments and privileged operations

Checked-in workflows reference:

- `fly-console` for Fly deployment
- `Public Website` for Pages deployment

GitHub may also expose platform-managed environments such as `github-pages`.
Query current protection rules before changing a deploy job. Do not remove an
environment gate to bypass approval or secret access.

Privileged/destructive paths include release publishing, Fly/Pages deployment,
cross-repository dispatch, crate publishing, PR cache cleanup, repository cache
reset, and any job with write permissions. Inspect permissions and concurrency
before modifying these paths.

## Live-state inspection commands

```bash
gh workflow list --all --repo Mesh-LLM/mesh-llm
gh run list --repo Mesh-LLM/mesh-llm --limit 30
gh variable list --repo Mesh-LLM/mesh-llm
gh secret list --repo Mesh-LLM/mesh-llm
gh api repos/Mesh-LLM/mesh-llm/environments
gh api repos/Mesh-LLM/mesh-llm/actions/runners
gh api repos/Mesh-LLM/mesh-llm/actions/permissions
gh api repos/Mesh-LLM/mesh-llm/actions/permissions/workflow
gh api repos/Mesh-LLM/mesh-llm/branches/main/protection
```

Use `gh variable list --env NAME` and `gh secret list --env NAME` for an
environment. Organization-level listing requires additional GitHub permission;
record a 403 as unverified scope, not as absence.
