# MeshLLM CI inventory

This file records checked-in CI facts and selected controlled probe evidence.
It is not a complete historical run log or live GitHub/Depot administration.
Read it with `../SKILL.md` and `ci/ci.md` before editing CI.

## Entry workflows

| Workflow | Trigger | Ownership |
| --- | --- | --- |
| `pr_quality.yml` (`PR · Quality`) | PR lifecycle | Canonical PR planning plus the protected reusable Quality lane |
| `pr_website.yml` (`PR · Website`) | PR lifecycle | Canonical PR planning plus the protected reusable Website lane |
| `pr_linux.yml` (`PR · Linux`) | PR lifecycle | Canonical PR planning plus the protected reusable Linux lane |
| `pr_macos.yml` (`PR · macOS`) | PR lifecycle | Canonical PR planning plus the protected reusable macOS lane |
| `pr_windows.yml` (`PR · Windows`) | PR lifecycle | Canonical PR planning plus the protected reusable Windows lane |
| `pr_builds.yml` | `workflow_call` only | Inert migration shim for the pre-merge protected runner-contract filename check; no PR event trigger |
| `ci-orchestrator.yml` | `workflow_call` only | Inert migration shim for the pre-merge protected runner-contract filename check; no PR event trigger or lane calls |
| `main_quality.yml` (`Main · Quality`) | push to `main` | Exhaustive main planning plus the same-commit reusable Quality lane |
| `main_website.yml` (`Main · Website`) | push to `main` | Exhaustive main planning plus the same-commit reusable Website lane |
| `main_linux.yml` (`Main · Linux`) | push to `main` | Exhaustive main planning plus the same-commit reusable Linux lane |
| `main_macos.yml` (`Main · macOS`) | push to `main` | Exhaustive main planning plus the same-commit reusable macOS lane |
| `main_windows.yml` (`Main · Windows`) | push to `main` | Exhaustive main planning plus the same-commit reusable Windows lane |
| `ci.yml` | `workflow_call` only | Inert migration shim for the former main ingress filename; no push trigger or dispatch |
| `ci-control.yml` (`CI · Manual Full`) | dispatch on default branch | Explicit operator-only full plan, bounded lane dispatch and correlated diagnostic checks |
| `release.yml` | release tags, dispatch | Canonical version synchronization, release-only signing, assets and publication |
| `website-pages.yml` | main website paths, dispatch | Public website deployment |
| `pr_cleanup.yml` | PR close, dispatch | Positively matched cleanup only |
| `pr_auto_assign.yml` | PR lifecycle | Metadata only |

Other scheduled, deployment, Docker, package, canary and cache-warming
workflows are independent of required PR readiness.

For a non-canary manual dispatch, `release.yml` runs the checked-in
`scripts/release-version.sh`, creates one linear release-source commit when the
tracked version surface changes, and fast-forwards `main` before any release
build starts. `just release` is a preflight and synchronous dispatcher for that
same workflow. A tag-push release is read-only with respect to `main` and is
accepted only when the tag is already reachable from `main` and applying the
same version script produces no tracked diff. Canary dispatches never update
`main` or publish. The publish job creates only the release-specific tag commit
for generated Swift/SDK resources and enables GitHub-generated release notes.
The comparison base is the highest stable `vMAJOR.MINOR.PATCH` tag below the
target; prerelease tags are excluded so RC and final notes use the same stable
baseline.

The five PR lifecycle rows and five main push rows above are the complete
allowed routine validation entry sets. Their separation and direct GitHub log
visibility are contractual, not a presentation preference. `pr_builds.yml`,
`ci-orchestrator.yml`, and `ci.yml` are reusable-only migration scaffolding;
they must never regain event triggers or call the five lanes. They are
removable after this branch's runner contract is active on protected main.

## Reusable workflows and slices

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
the debug host plus CPU or GPU runtime/product graph runs only for its
platform/backend input or manual dispatch. Unsupported macOS CUDA, ROCm, and
Vulkan combinations are omitted rather than emitted as no-op jobs.
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

- `scripts/affected-crates.sh` computes affected crates and reverse dependents.
- `mesh-llm-provider-runtime` is a runtime-facing SDK input: changes to its
  executable-provider artifact contract route the SDK smoke graph.
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

| Workflow | Contract |
| --- | --- |
| `ci-quality-lane.yml` | Quality and runner/cache contract graph; reusable from PRs and dispatchable for main/manual |
| `ci-website-lane.yml` | Console and website graph; reusable from PRs and dispatchable for main/manual |
| `ci-linux-lane.yml` | Linux host/runtime/product/Rust/SDK/smoke graph with one platform-local UI producer |
| `ci-macos-lane.yml` | macOS host/runtime/product/platform/Swift/Metal graph with one platform-local UI producer |
| `ci-windows-lane.yml` | Windows host/runtime/product/platform graph with one platform-local UI producer |
| `ci-quality-slice.yml` | Contracts, format, Clippy and CLI/docs guard; additive protected authority sentinel |
| `ci-web-slice.yml` | Console quality and public website build |
| `ci-ui-artifact-slice.yml` | Immutable console distribution producer |
| `static-abi-artifact.yml` | Typed static llama ABI producer with internal runner policy and an exact toolchain-epoch output |
| `ci-rust-tests-slice.yml` | Typed deterministic Cargo test batches that verify the producer-owned static ABI toolchain epoch |
| `ci-{linux,macos,windows}-host-slice.yml` | Platform-pure neutral host producers; no empty cross-platform jobs |
| `ci-{linux,macos,windows}-runtime-slice.yml` | Platform-pure native runtime producers |
| `ci-{linux,macos,windows}-product-slice.yml` | Platform-pure composition-only product consumers |
| `ci-platform-checks-slice.yml` | macOS portable/unit, Windows portable, and Windows log-store privacy ACL checks |
| `ci-linux-product-smoke-slice.yml`, `ci-macos-product-smoke-slice.yml` | Platform-local CPU, CUDA (`gpu-nvidia` self-hosted), two-node, Metal and model-download consumers; ROCm/Vulkan products remain package-verified pending eligible inference runners |
| `ci-linux-sdk-slice.yml`, `ci-macos-sdk-slice.yml` | Platform-local Rust/Kotlin/Swift smoke consumers; SDK producers are independent top-level calls |
| `ci-runner-contract-slice.yml` | Provider/cache/plan trust and main runner-image checks |
| `native-sdk-artifact.yml` | Typed native SDK producer |
| `swift-sdk-artifact.yml` | Host-only/full XCFramework producer; trusted main remains `macos-15`, while eligible same-repository PRs follow the protected Depot macOS 15 gate |
| `smoke.yml` | Artifact-based inference/OpenAI/split smoke |
| `scripted-binary-smoke.yml` | Artifact-based scripted product smoke |
| `sdk-smoke.yml` | Artifact-based SDK consumers |
| `hf-download-smoke.yml` | Hugging Face download smoke |

All workflow calls use typed, bounded semantic inputs. Credential-bearing smoke
workflows remain fixed to GitHub-hosted runners; the PR entrypoints pass no
repository secrets. The trusted main entrypoint may pass the optional
`HF_TOKEN` for public-fixture rate-limit resilience.

## Planner contract

- `scripts/plan-ci.py` is the only routing implementation.
- `ci/ownership.yml` maps paths and direct crates to semantic domains; unknown
  paths fail closed.
- `ci/slices.yml` defines profiles, slice dependencies, rows, runner roles,
  cache modes and worker budgets.
- `ci/ci-plan.schema.json` versions the machine-readable output.
- `compute-changes` supplies the complete event diff and affected Cargo
  closure; the planner owns signals and final matrix selection.
- Each `pr_*.yml` workflow checks out the default branch for canonical planning,
  projects one bounded lane, and calls its matching default-branch lane as a
  nested reusable workflow. Jobs and logs remain attached to five focused PR
  runs rather than one monolithic graph.
- Each `main_*.yml` workflow plans the exhaustive main profile at the pushed
  SHA, projects one bounded lane, and calls its matching same-commit lane as a
  nested reusable workflow. Routine main jobs and logs therefore remain
  attached to five focused main runs.
- `ci-control.yml` is manual-full only. It calls the planner once and dispatches
  bounded JSON lane projections as native inputs for explicit operator
  diagnostics; it cannot receive a push, PR, or workflow-run event.

Main/manual profiles enumerate every workspace crate exactly once and all
supported product/SDK rows. PR profiles select affected or directly owned rows
from that same catalog.

## Artifact and cache owners

- `prepare-host-input` / `prepare-windows-host-input`: neutral host bytes,
  import report and checksum.
- `prepare-native-runtime-input`: one verified native runtime archive and
  manifest.
- `prepare-static-abi-input`: portable static ABI archive.
- `compose-product-input`: exact host/runtime verification and composition.
- `restore-smoke-inputs`: product/model extraction for consumers.
- `select-ci-runners`: provider labels, cache permissions, and the
  provider-derived `allow_native_github_cache` / `allow_depot_remote_cache`
  outputs. Depot selections disable both cache paths by default. During the
  bounded approved exception, the exact PR revision and eligible trusted-main
  Depot jobs enable the GitHub Actions cache API while direct Depot remote
  cache remains disabled. Hosted PR, release, and cache-warmer selections
  retain native GitHub cache behavior.
- `configure-sccache-gha`: event/provider-derived compiler-cache setup.
- `capture-sccache-stats`: machine-readable cache evidence.

`scripts/collect-ci-metrics.py` is the read-only timing evidence collector. Its
schema-v3 report keeps workflow wall/queue, job runner queue, measured
dependency wait, job execution, runner-minutes, cancelled runner-minutes and
peak workers separate. It groups observations by provider, operating system,
architecture, semantic runner role and Depot size, and emits deterministic
queue/capacity heuristics plus an optional provider-cohort comparison. Raw
inputs and dated reports belong under `/tmp` or a tracking issue/artifact, not
under `ci/` or this inventory.

Artifacts are correctness boundaries; caches only accelerate regeneration.
PR artifacts generally retain for one day. Fork lanes cannot publish shared
trusted-main caches. Same-repository PRs normally use GitHub's ref-scoped cache;
an exact approved revision may temporarily use Depot's shared cross-branch
namespace under `ci/DEPOT_PR_RISK_EXCEPTION.md`. That namespace is treated as
untrusted input, not an authority or correctness boundary. Large Cargo target caches restore trusted-main entries but remain
restore-only on PRs. Exact Linux static ABI, Swift ABI, macOS Metal unit ABI,
and Windows native ABI caches may publish into GitHub's isolated PR merge-ref
scope for same-PR reruns. The Website slice is the sole publisher for the
shared pnpm key and owns the website npm cache; platform UI producers restore
the pnpm store without racing to save it. Trusted main owns shared publication.

PR Rust-test, host, native-runtime, product, and platform-check matrices receive
`fail_fast: true`; main/manual pass `false`. Quality matrices remain
non-fail-fast, failed producers suppress only declared consumers through
`needs`, and focused PR workflows never cancel one another.

## Providers and variables

GitHub-hosted labels are `ubuntu-24.04`, `ubuntu-24.04-arm`, `macos-15`, and
`windows-2022`. Depot labels are selected only by `select-ci-runners`; no
workflow accepts a raw provider label. Trusted main Linux requires
`DEPOT_RUNNERS_ENABLED=true`. An exact same-repository PR revision may use the
time-bounded exception only when `DEPOT_PR_RUNNERS_ENABLED=true` and both
`DEPOT_PR_APPROVED_REF` and `DEPOT_PR_APPROVED_SHA` match; it expires on
2026-09-14 UTC. Forks remain hosted. The intended permanent gate
may cover eligible build/test rows across Linux, Depot macOS 15 and Windows
2022 when equivalent images/architectures exist; planning/required summaries,
credential-bearing smokes, `gpu-nvidia` hardware and uncertified Intel macOS
rows remain exceptions. The documented `gpu-nvidia` ephemeral scale set is
the sole current uncredentialed, hardware-qualified same-repository PR
exception.

The permanent Depot PR gate is documented in `ci/DEPOT_MIGRATION.md`; the
accepted temporary findings and risks are in
`ci/DEPOT_PR_RISK_EXCEPTION.md`. Permanent activation requires cache
isolation, no PR cache/registry tokens, exact protected workflow refs,
ephemeral runners, a successful sentinel, and a tested GitHub rollback.

External administrative posture is now verified as follows: automatic Depot
Cache connectivity is disabled, automatic Registry Actions authentication is
disabled, and the Depot runner group is restricted to `Mesh-LLM/mesh-llm` and
the exact protected workflow refs. The repository token cannot independently
inspect organization runner-group settings through the API (403), so these
remain external facts rather than checked-in evidence. The two switches remove
Depot's direct `DEPOT_CACHE_TOKEN`/WebDAV build-tool preconfiguration and
Registry Actions authentication on fresh runners; they do not document or
enforce a per-connection/job/ref disable or ACL for the GitHub Actions cache
proxy/runtime-token path. The controlled
trusted-main seed [run 31816775585](https://github.com/Mesh-LLM/mesh-llm/actions/runs/31816775585)
succeeded at `main` commit `9e977e246`; the same-repository PR authority
sentinel [run 31816869128 / job 94821057215](https://github.com/Mesh-LLM/mesh-llm/actions/runs/31816869128/job/94821057215)
read and exactly validated the trusted seed, saved/cleared/restored and
exactly validated the poison, then failed its intended seed-isolation gate;
the enclosing PR run was later cancelled during cleanup. Trusted-main verify
[run 31817111471 / job 94821343605](https://github.com/Mesh-LLM/mesh-llm/actions/runs/31817111471/job/94821343605)
restored and exactly validated that poison, then failed its intended expected-
miss gate. This proves unsafe repository-scoped cross-trust authority, so it is
not a successful isolation result. The bounded exception knowingly accepts
that risk for exact ref/SHA-approved same-repository revisions to gain CI
iteration speed; it is not permanent-isolation evidence. The exact-SHA
five-lane candidate, provider comparison, and identical-SHA hosted rollback
are recorded in `.omo/specs/depot-pr-rollout-evidence.md`; Quality and Linux
had favorable queue observations but remain unclassified because execution
was cache-confounded, Website had insufficient samples, and macOS/Windows hit
the capacity rollback threshold. Fork PR validation and namespace purge/expiry
confirmation remain pending. Fork PR validation remains hosted and is the
no-Depot-authority half of the sentinel acceptance evidence; only the exact
same-repository sentinel ref may exercise the diagnostic Depot job. All three
sentinel cache phases attest the
provider-injected `ACTIONS_CACHE_URL`/`ACTIONS_RESULTS_URL` structure before
invoking pinned `actions/cache` restore/save actions. The shell attestation
does not require ambient `ACTIONS_RUNTIME_TOKEN`: GitHub's
`NodeScriptActionHandler` injects that credential into the cache actions, while
the shell `ScriptHandler` does not. Successful full restore/save is the
credential/token proof. The non-loopback check includes all IPv4 `127/8`
and IPv4-mapped IPv6 loopback spellings.
The protected PR probe clears and fully restores its saved poison key, requires
a cache hit and exact marker bytes before the trusted-seed gate, and thereby
proves the same-job Node token/write path; main verify's poison miss remains
the cross-scope proof.

The provider contract required before permanent PR placement is enabled is a documented,
server-enforced per-connection/job/ref control for the GitHub Actions cache
path. It must either leave PR jobs on GitHub-native branch-scoped
`ACTIONS_CACHE_URL`/`ACTIONS_RESULTS_URL` and runtime-token semantics with no
Depot proxy or direct cache token, or issue a PR-isolated namespace/token whose
ACL permits reads and writes only within that PR, denying reads and writes from
trusted main/release and every other PR namespace, without exposing
`DEPOT_CACHE_TOKEN`.
Key prefixes, loopback proxies,
ephemeral runners and the org switches are not equivalent controls. A fresh
same-repository PR, fork PR and trusted-main seed/verify sentinel must prove
the selected behavior before the temporary exception is removed.
Bracketed IPv6 authorities use the fixed runner's Python 3.8+ stdlib
`ipaddress` classifier; parser absence/version/invalidity fails closed.
Attestation reports only value-free variable/reason classes and fails closed
on malformed or missing backend data.

Relevant repository variable names include `DEPOT_RUNNERS_ENABLED`,
`DEPOT_PR_RUNNERS_ENABLED` (global temporary exception gate),
`DEPOT_PR_APPROVED_REF` (one exact merge ref), `DEPOT_PR_APPROVED_SHA` (the
exact lowercase PR head SHA; refresh after every push),
`DEPOT_PR_CANARY_REF` (absent by default; one exact
`refs/pull/<number>/merge` ref only), `DEPOT_PR_SENTINEL_REF` (absent by
default; one exact same-repository merge ref used only by the protected
no-checkout authority diagnostic), and `DEPOT_PR_SENTINEL_ID` (absent by
default; exactly 32 lowercase hexadecimal characters when the diagnostic is
deliberately armed). The canary and sentinel variables are bounded selectors,
not cache-isolation proofs or replacements for the global PR gate. The normal
Quality runner policy continues to use `DEPOT_PR_CANARY_REF`; the sentinel
uses a separate selector output and cannot move the normal build jobs.
The eligible five-lane Depot graph disables every native GitHub cache consumer
when `allow_native_github_cache=false`. During the bounded exception the exact
approved PR and eligible trusted-main Depot jobs set that output true for
cross-branch Depot Actions-cache reuse; direct Depot remote cache remains
false. This checked-in mode does not
prove the absence of ambient Depot/WebDAV authority, so the runtime sentinel
has recorded unsafe repository-scoped cross-trust authority and must be
redesigned and repeated successfully; no-secret/no-token, fork and provider-
parity canaries remain required. Other variables include `CUDA_VERSION`,
`VULKAN_SDK_VERSION`, smoke configuration variables, and release/deployment
variables. Secret values never belong in this inventory;
known names include `HF_TOKEN`, release-attestation keys, `CARGO_REGISTRY_TOKEN`
and deployment tokens.

## Live inspection

Use read-only commands when live state matters:

```bash
gh workflow list --all --repo Mesh-LLM/mesh-llm
gh run list --repo Mesh-LLM/mesh-llm --limit 30
gh variable list --repo Mesh-LLM/mesh-llm
gh api repos/Mesh-LLM/mesh-llm/rulesets
gh api orgs/Mesh-LLM/actions/runner-groups
```

Organization runner-group responses of `403` are unverified administrative
state, not proof that a restriction is absent.
