# PR and main CI composition plan

Status: implemented on this branch. The shared planner, protected PR composer,
main/manual controller, topic/platform lane workflows, platform-pure reusable
graphs, contracts and documentation are checked in. Ruleset migration and
Depot PR execution remain separate future work.

Owners: MeshLLM maintainers

## Outcome

PR and main CI use one versioned planner and one catalog of typed reusable
workflow slices. A PR selects representative rows from the main catalog; a
selected row uses the same workflow, commands, build profile, artifact
contract and verification that main uses. Faster PRs come from precise routing,
immutable producer reuse and bounded fan-out, not from a second lightweight
implementation or more runner capacity.

GitHub-hosted and Depot-managed runners are placement providers. Provider
selection cannot change plan membership, commands, artifacts, required results
or cache identity.

## Implemented on this branch

- One-job pr_builds.yml calls the protected default-branch PR composer, while
  ci.yml wakes the protected ci-control.yml planner. Same-repository and fork
  PRs keep nested reusable jobs in their native PR run; main and explicit
  manual-full runs use separate protected lane dispatch.
- ci/ownership.yml and ci/slices.yml define the checked ownership, dependency,
  row, runner-role, cache-mode and worker-budget catalog.
- scripts/plan-ci.py emits the versioned plan described by
  ci/ci-plan.schema.json, including direct domains, affected crates, signals,
  reasons, dependencies, matrices and budgets.
- Separate Quality, Website, Linux, macOS and Windows workflows receive bounded
  native inputs from one plan and own typed platform-local static slice
  supersets without unrelated platform placeholders. They support native
  reusable PR calls and protected main/manual dispatch.
- PRs expose native lane jobs and one native CI Required job; dispatched
  main/manual lanes retain correlated checks.
- Existing static ABI, native SDK, Swift, smoke and HF workflows remain
  lower-level reusable producers/consumers.
- Current PR routing is GitHub-hosted except for the documented uncredentialed
  CUDA smoke on the approved ephemeral GPU scale set; trusted-main Depot
  selection remains behind the existing exact-string policy gate.
- Current CI docs, inventory, skill and agent instructions describe this graph.

This branch does not change branch rulesets, required checks, Depot settings,
runner groups, secrets or external capacity.

## Graph

  PR entry --> protected reusable composer: compute-changes + plan-ci once
       |
       +--> Quality workflow --> CI / Quality ---+
       +--> Website workflow --> CI / Website ---+
       +--> Linux workflow ----> CI / Linux -----+
       +--> macOS workflow ----> CI / macOS -----+--> CI Required
       +--> Windows workflow --> CI / Windows ---+

  Main/manual entry --> protected controller --> the same five dispatched lanes

The PR composer calls a closed list of protected default-branch workflows as
nested reusable jobs, preserving native PR run/log visibility. The main/manual
controller dispatches the same list with bounded JSON inputs. Both pass the
immutable source SHA only to product checkouts; each lane contains a
platform-local static superset of typed reusable calls. Workflow YAML is never
generated, and lanes do not download a planner artifact or allocate another
planner. Fork heads are fetched through the base repository while workflow
definitions remain protected on the default branch.

## Planner contract

scripts/plan-ci.py is the only eligibility implementation. It reads the
JSON-compatible manifests and validates their schema and dependency graph.
Unknown paths and malformed inputs fail closed. CI-control and
runner-infrastructure changes fail open to control rows and all supported
product rows.

Each plan contains:

- event/profile and source/base identities;
- direct crates and affected Cargo reverse-dependency crates separately;
- ordered semantic domains and planner-owned signals;
- selected slice IDs, reasons and dependency closure;
- bounded Clippy, Rust-test, host, runtime/product, platform, smoke and SDK
  matrices;
- runner roles, cache modes and Linux/macOS/Windows/total worker budgets.

Profiles are closed and event-derived:

| Profile | Selection |
| --- | --- |
| pr-draft | Quality, affected Rust signal, directly owned web/product rows, core smoke only |
| pr-ready | Complete targeted rows for direct domains and affected Rust dependents |
| main | Every workspace crate and every supported product/platform/backend/SDK row |
| manual-full | Main-equivalent non-publishing dispatch |

The selected PR row is semantically identical to main. Only plan membership,
trust-derived cache mode, short-lived artifact namespace, provider label and
optional trusted credentials may differ.

## Slice catalog

- ci-quality-slice.yml: contracts, formatting, bounded Clippy and CLI docs.
- ci-web-slice.yml: console lint/type/test and public website build.
- ci-ui-artifact-slice.yml: one immutable console distribution producer.
- static-abi-artifact.yml: one verified portable static llama ABI producer.
- ci-rust-tests-slice.yml: deterministic affected/all-workspace Cargo batches.
- ci-{linux,macos,windows}-host-slice.yml: platform-pure neutral hosts.
- ci-{linux,macos,windows}-runtime-slice.yml: one native runtime per selected
  backend, running in parallel with the matching host.
- ci-{linux,macos,windows}-product-slice.yml: platform-local composition after
  matching host and runtime producers succeed.
- ci-platform-checks-slice.yml: macOS portable/unit and Windows checks.
- ci-linux-product-smoke-slice.yml and ci-macos-product-smoke-slice.yml:
  platform-local inference, backend, two-node, Metal and model-download
  consumers using only composed artifacts.
- ci-linux-sdk-slice.yml and ci-macos-sdk-slice.yml: platform-local
  Rust/Kotlin/Swift consumers. Swift and Kotlin SDK artifacts are independent
  producers that start from the plan and static ABI respectively, before
  product composition completes.
- ci-runner-contract-slice.yml: plan/provider/cache trust checks plus trusted
  main runner-image contracts.

Existing native-sdk-artifact.yml, swift-sdk-artifact.yml, smoke.yml,
scripted-binary-smoke.yml, sdk-smoke.yml and hf-download-smoke.yml remain
lower-level typed building blocks. Consumers never rebuild missing producers.

## Product and artifact contract

Every executable product has three layers:

1. prepared UI assets;
2. one release-profile backend-neutral host per OS/architecture;
3. one native runtime per OS/architecture/backend.

The catalog covers Linux CPU/CUDA/ROCm/Vulkan, macOS Metal and Windows
CPU/CUDA/ROCm/Vulkan where supported. Unsupported combinations are omitted by
the planner, not represented as permanent skipped jobs.

Host actions emit executable, checksum and import-policy evidence. The native
runtime action emits a manifested archive. compose-product-input verifies exact
producer bytes and performs no compilation, relinking, restamping or
substitution. Smoke and SDK consumers download those artifacts only.

Selected PR and main product rows both use release-profile hosts. Native
runtime compilation does not depend on host production, so each platform lane
starts those immutable producers concurrently and only joins them for product
composition.

## Fan-out and timing controls

Initial planner budgets are:

| Profile | Linux cap | macOS cap | Windows cap | Total planned |
| --- | ---: | ---: | ---: | ---: |
| PR draft/ready | 7 | 2 | 1 | 10 |
| main/manual | 12 | 4 | 2 | 18 |

The lane projections pass smaller PR max-parallel values to Clippy, tests,
hosts, runtimes and platform checks and wider bounded values to main. Host, ABI
and runtime producer identities are not duplicated. Separate run-scoped graphs
build one prepared UI artifact per active platform lane; that is the accepted
readability tradeoff, while UI tests remain owned by the Website graph. Every
heavy job has a timeout and a deterministic row identity.

scripts/collect-ci-metrics.py is the read-only measurement tool. Timing
experiments must use a new worktree from main. Keep queue, dependency wait,
execution and wall-clock measurements separate; retain raw evidence outside
authoritative topology docs. Capacity is not the optimization.

## Required summary

Each lane owns one stable non-matrix summary that directly needs its complete
static job superset and validates required work against its bounded plan.
The PR composer owns one native non-matrix CI Required job that directly needs
all reusable lane calls and validates required successes and unplanned skips.
ci-control.yml creates correlated lane checks for dispatched main/manual runs;
the final lane completes that aggregate only when all expected checks are
terminal. Existing branch-protection rules are not edited by this
implementation.

## GitHub and Depot

select-ci-runners resolves semantic runner roles. Pull requests, feature refs,
tags, macOS, Windows, credential-bearing smokes and hardware-qualified work
stay on their approved placement. Trusted main Linux work may use Depot only
when DEPOT_RUNNERS_ENABLED is exactly true, with a GitHub-hosted fallback.
Callers never provide raw labels or independent remote-cache permission.

Depot PR execution is not implemented. Depot's documented GitHub cache path is
repository-scoped and not branch-isolated; automatic cache redirection can
expose repository-wide cache authority to PR code. Cache-key prefixes are not
isolation.

Before a PR Depot path is enabled, an administrator must prove:

1. automatic Depot cache connectivity is disabled or provides per-PR isolation;
2. same-repository and fork PRs receive no cache, registry or repository secret
   authority;
3. hostile PRs cannot read a trusted sentinel or publish an entry restored by
   trusted main;
4. the ephemeral runner group is restricted to this repository and exact
   protected default-branch runner-owning workflow refs;
5. CI-control changes force the GitHub path and rollback is tested;
6. provider parity passes on comparable non-CI-change PRs.

The investigation and canary are defined in ci/DEPOT_MIGRATION.md. Do not
change Depot settings or runner groups in this graph PR. Start a later rollout
with remote cache disabled, then canary one non-secret Linux slice, one Rust
test slice and the selected Linux product graph. Keep credential-bearing,
macOS, Windows and hardware work hosted.

## Extension pattern

1. Read the manage-ci skill, current inventory, ci/ci.md and this spec.
2. Classify the owner and trust context.
3. Extend one typed reusable slice or local action; keep entrypoints thin.
4. Update ownership, dependencies, rows and budgets when routing changes.
5. Preserve immutable producer reachability and add the top-level call to its
   lane summary; update the bounded controller projection when necessary.
6. Keep runner and cache decisions in the central policy action.
7. Update current docs and fixtures in the same change.

Required local checks are actionlint with the repository config, git diff
--check, the full scripts/tests unittest suite, applicable shellcheck, and
the relevant xtask repo-consistency checks.

## Acceptance checklist

- docs-only, website-only, UI-only, ordinary Rust, runtime, protocol,
  split-serving, model, each SDK, each backend, macOS-only, Windows-only,
  mixed and CI-control changes produce expected plan rows and reasons;
- draft-to-ready changes select the correct closed profile;
- main covers every workspace member exactly once and all supported rows;
- each selected PR product uses the same slice and artifact contract as main;
- no duplicate host, ABI, or native-runtime producer identity is built for one
  source plan;
- every consumer has a reachable producer and no consumer fallback build;
- every lane summary passes unplanned skips and fails planned skips;
- cancellation leaves no required summary stuck in a running state;
- GitHub fallback passes all slice fixtures;
- no branch ruleset, runner group, Depot setting or external capacity change
  is included in this implementation PR.

Required-check migration and Depot PR enablement are separate follow-up changes
after these acceptance cases pass.
