# MeshLLM CI inventory

This file records checked-in CI facts. It is not a record of historical runs
or live GitHub/Depot administration. Read it with `../SKILL.md` and `ci/ci.md`
before editing CI.

## Entry workflows

| Workflow | Trigger | Ownership |
| --- | --- | --- |
| `pr_builds.yml` | PR lifecycle, dispatch | Routes ordinary same-repository PRs to protected control; bootstraps control-plane changes, forks and migrations through `ci-orchestrator.yml` |
| `ci.yml` | main push, dispatch | Routes main pushes to protected control; manual/migration runs use `ci-orchestrator.yml` |
| `ci-control.yml` | completed `PR CI` / `Main CI` | Protected source resolution, one canonical plan, bounded lane dispatch and correlated required checks |
| `ci-orchestrator.yml` | `workflow_call` | Bootstrap planner and monolithic static slice graph with `CI Required` |
| `release.yml` | release tags, dispatch | Release-only signing, assets and publication |
| `website-pages.yml` | main website paths, dispatch | Public website deployment |
| `pr_cleanup.yml` | PR close, dispatch | Positively matched cleanup only |
| `pr_auto_assign.yml` | PR lifecycle | Metadata only |

Other scheduled, deployment, Docker, package, canary and cache-warming
workflows are independent of required PR readiness. The former
`pr_quality.yml` and `pr_website.yml` entrypoints no longer exist.

## Reusable workflows and slices

| Workflow | Contract |
| --- | --- |
| `ci-quality-lane.yml` | Quality and runner/cache contract graph; supports typed call and native dispatch |
| `ci-website-lane.yml` | Console and website graph; supports typed call and native dispatch |
| `ci-linux-lane.yml` | Linux host/runtime/product/Rust/SDK/smoke graph with one platform-local UI producer |
| `ci-macos-lane.yml` | macOS host/runtime/product/platform/Swift/Metal graph with one platform-local UI producer |
| `ci-windows-lane.yml` | Windows host/runtime/product/platform graph with one platform-local UI producer |
| `ci-quality-slice.yml` | Contracts, format, Clippy and CLI/docs guard |
| `ci-web-slice.yml` | Console quality and public website build |
| `ci-ui-artifact-slice.yml` | Immutable console distribution producer |
| `static-abi-artifact.yml` | Typed static llama ABI producer with internal runner policy and an exact toolchain-epoch output |
| `ci-rust-tests-slice.yml` | Typed deterministic Cargo test batches that verify the producer-owned static ABI toolchain epoch |
| `ci-host-slice.yml` | Neutral host producer per selected OS/architecture, invoked in independent platform lanes |
| `ci-runtime-product-slice.yml` | Separately invoked native runtime producers and composition-only products, joined within each platform lane |
| `ci-platform-checks-slice.yml` | macOS portable/unit, Windows portable, and Windows log-store privacy ACL checks |
| `ci-product-smoke-slice.yml` | CPU, CUDA (`gpu-nvidia` self-hosted), two-node, Metal and model-download consumers; ROCm/Vulkan products remain package-verified pending eligible inference runners |
| `ci-sdk-slice.yml` | Platform-local Rust/Kotlin/Swift smoke consumers; SDK producers are independent top-level calls |
| `ci-runner-contract-slice.yml` | Provider/cache/plan trust and main runner-image checks |
| `native-sdk-artifact.yml` | Typed native SDK producer |
| `swift-sdk-artifact.yml` | Fixed `macos-15` host-only/full XCFramework producer |
| `smoke.yml` | Artifact-based inference/OpenAI/split smoke |
| `scripted-binary-smoke.yml` | Artifact-based scripted product smoke |
| `sdk-smoke.yml` | Artifact-based SDK consumers |
| `hf-download-smoke.yml` | Hugging Face download smoke |

All workflow calls use typed, bounded semantic inputs. Credential-bearing smoke
workflows remain fixed to GitHub-hosted runners; the PR entrypoint passes no
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
- `ci-control.yml` calls the planner once and dispatches bounded JSON lane
  projections as native inputs. `ci-orchestrator.yml` preserves the same
  planner-once and slice-selection contracts in a monolithic bootstrap graph
  for fork/manual/migration compatibility without requesting check-write
  permission through a reusable-workflow chain.

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
- `select-ci-runners`: provider labels and Depot cache permission.
- `configure-sccache-gha`: event/provider-derived compiler-cache setup.
- `capture-sccache-stats`: machine-readable cache evidence.

Artifacts are correctness boundaries; caches only accelerate regeneration.
PR artifacts generally retain for one day. Bootstrap/fork cache writes are
merge-ref scoped; protected same-repository dispatched lanes are restore-only
despite running from the default-branch workflow ref. PR dispatches cannot
publish shared caches, and Depot cache access is denied. Trusted main may
publish shared caches.

## Providers and variables

GitHub-hosted labels are `ubuntu-24.04`, `ubuntu-24.04-arm`, `macos-15`, and
`windows-2022`. Depot labels are selected only by `select-ci-runners` for
trusted main Linux when `DEPOT_RUNNERS_ENABLED` is exactly `true`; no workflow
accepts a raw provider label. The documented `gpu-nvidia` ephemeral scale set
is the sole uncredentialed, hardware-qualified same-repository PR exception.

The future Depot PR gate is documented in `ci/DEPOT_MIGRATION.md`. It requires
cache isolation, no PR cache/registry tokens, exact protected workflow refs,
ephemeral runners, a sentinel canary, and a tested GitHub rollback. No Depot
settings or runner groups are changed by this workflow refactor.

Relevant repository variable names include `DEPOT_RUNNERS_ENABLED`,
`CUDA_VERSION`, `VULKAN_SDK_VERSION`, smoke configuration variables, and
release/deployment variables. Secret values never belong in this inventory;
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
