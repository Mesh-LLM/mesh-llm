# MeshLLM CI inventory

This file records checked-in CI facts. It is not a record of historical runs
or live GitHub/Depot administration. Read it with `../SKILL.md` and `ci/ci.md`
before editing CI.

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
| `release.yml` | release tags, dispatch | Release-only signing, assets and publication |
| `website-pages.yml` | main website paths, dispatch | Public website deployment |
| `pr_cleanup.yml` | PR close, dispatch | Positively matched cleanup only |
| `pr_auto_assign.yml` | PR lifecycle | Metadata only |

Other scheduled, deployment, Docker, package, canary and cache-warming
workflows are independent of required PR readiness.

The five PR lifecycle rows and five main push rows above are the complete
allowed routine validation entry sets. Their separation and direct GitHub log
visibility are contractual, not a presentation preference. `pr_builds.yml`,
`ci-orchestrator.yml`, and `ci.yml` are reusable-only migration scaffolding;
they must never regain event triggers or call the five lanes. They are
removable after this branch's runner contract is active on protected main.

## Reusable workflows and slices

| Workflow | Contract |
| --- | --- |
| `ci-quality-lane.yml` | Quality and runner/cache contract graph; reusable from PRs and dispatchable for main/manual |
| `ci-website-lane.yml` | Console and website graph; reusable from PRs and dispatchable for main/manual |
| `ci-linux-lane.yml` | Linux host/runtime/product/Rust/SDK/smoke graph with one platform-local UI producer |
| `ci-macos-lane.yml` | macOS host/runtime/product/platform/Swift/Metal graph with one platform-local UI producer |
| `ci-windows-lane.yml` | Windows host/runtime/product/platform graph with one platform-local UI producer |
| `ci-quality-slice.yml` | Contracts, format, Clippy and CLI/docs guard |
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
  outputs. Every Depot selection sets both outputs to false; hosted PR,
  release, and cache-warmer selections retain native GitHub cache behavior.
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
PR artifacts generally retain for one day. Protected same-repository and fork
lanes cannot publish shared trusted-main caches, and Depot cache access is
denied. The Depot namespace is intentionally unused by trusted workflows;
purge/expiry is required before PR activation, and an inert proxy is not
authority-isolation proof. Large Cargo target caches restore trusted-main entries but remain
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
`windows-2022`. Depot labels are selected only by `select-ci-runners` for
trusted main Linux when `DEPOT_RUNNERS_ENABLED` is exactly `true`; no workflow
accepts a raw provider label. Current ordinary PR execution remains hosted
until the protected PR gate is separately activated. The intended final gate
may cover eligible build/test rows across Linux, Depot macOS 15 and Windows
2022 when equivalent images/architectures exist; planning/required summaries,
credential-bearing smokes, `gpu-nvidia` hardware and uncertified Intel macOS
rows remain exceptions. The documented `gpu-nvidia` ephemeral scale set is
the sole current uncredentialed, hardware-qualified same-repository PR
exception.

The future Depot PR gate is documented in `ci/DEPOT_MIGRATION.md`. It requires
cache isolation, no PR cache/registry tokens, exact protected workflow refs,
ephemeral runners, a sentinel canary, and a tested GitHub rollback. No Depot
settings or runner groups are changed by this workflow refactor.

External administrative posture is now verified as follows: automatic Depot
Cache connectivity is disabled, automatic Registry Actions authentication is
disabled, and the Depot runner group is restricted to `Mesh-LLM/mesh-llm` and
the exact protected workflow refs. The repository token cannot independently
inspect organization runner-group settings through the API (403), so these
remain external facts rather than checked-in evidence. The protected
branch/main canary, same-repository PR canary, fork PR canary, provider-parity
comparison, and rollback run are still pending; `DEPOT_RUNNERS_ENABLED` or a
successful manual canary does not prove PR isolation.

Relevant repository variable names include `DEPOT_RUNNERS_ENABLED`,
`DEPOT_PR_RUNNERS_ENABLED` (absent/false until protected PR activation), and
`DEPOT_PR_CANARY_REF` (absent by default; one exact
`refs/pull/<number>/merge` ref only). The latter is a bounded selector
canary, not a cache-isolation proof or a replacement for the global PR gate.
The eligible five-lane Depot graph now disables every native GitHub cache
consumer (explicit cache actions, setup-* package caches, rust-cache, static /
Metal / Windows / Swift ABI caches and Windows SDK cache toggles) and Depot
remote cache when those outputs are false; cache misses rebuild normally. This checked-in mode does not
prove the absence of ambient Depot/WebDAV authority, so the runtime sentinel
and no-secret/no-token canaries remain required. Other variables include `CUDA_VERSION`,
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
