---
name: manage-ci
description: Use this skill as the mandatory starting point whenever inspecting, running, debugging, defining, editing, reviewing, or documenting MeshLLM CI/CD. It governs GitHub Actions workflows and local actions, triggers and routing, runner images and worker labels, dependencies, caches and artifacts, variables, secrets, tokens and permissions, concurrency, releases, deployments, and CI-related scripts.
---

# Manage CI

Treat this skill as the canonical source of CI rules for MeshLLM. Read it before
every CI edit. Treat `ci/ci.md` as the current topology explanation and
`.github/AGENTS.md` as an entry-point pointer, not as competing rule sources.

Read [references/current-inventory.md](references/current-inventory.md) in full
before changing a workflow, action, runner, image, variable, secret, permission,
deployment, release, artifact, or cache contract. If the inventory or topology
does not match the checked-in implementation, verify the implementation and
update the skill resources in the same change.

## Required starting procedure

1. Inspect `git status`, the applicable `AGENTS.md` files, and the complete
   current workflow/action being changed. Preserve unrelated worktree changes.
2. Read `ci/ci.md` for routing and producer/consumer topology. Read the scripts,
   manifests, reusable workflows, and local actions reached by the proposed
   change; do not reason from one YAML fragment in isolation.
3. Use the inventory commands to inspect current workflow runs, repository
   variables, secret names, environments, and runners when the task depends on
   live configuration. Never infer live values from documentation.
4. Classify the change: PR quality, PR build/smoke, main CI, scheduled canary,
   release/publish, deployment, maintenance, runner infrastructure, or
   dependency image. Identify every producer, consumer, permission, and trust
   boundary affected before editing.
5. Make the smallest coherent change. Update this skill first when changing a
   CI rule; update `references/current-inventory.md` and `ci/ci.md` when their
   factual inventories or topology change.

## Authority and operational safety

- Treat inspection, log reads, YAML validation, and dry-run planning as
  read-only. Editing repository CI files is authorized by a request to change
  CI; changing GitHub settings, variables, secrets, environments, runner scale,
  or external deployments requires that state change to be in the user's scope.
- Do not dispatch, rerun, cancel, approve, or delete workflow runs merely to
  investigate. Obtain or infer authorization only when the requested outcome
  requires the operation, and target exact run IDs.
- Never run a release, publish, production deploy, cache reset, runner teardown,
  or other destructive maintenance workflow without explicit authorization.
  Prefer release canary/dry-run inputs before publishing.
- Never print, retrieve, persist, or commit secret values. Report secret names,
  scopes, and presence only. Redact tokens and credentials from logs and final
  responses.
- Do not weaken a required check, path gate, permission boundary, smoke test, or
  branch/environment protection merely to make a failing run green. Diagnose
  the cause and fix its owning source.

## Workflow ownership and triggers

- Keep pull-request workflows in `pr_*.yml`. Keep the early quality workflow
  named `PR Quality Checks` in `pr_quality.yml` and the build workflow named
  `PR Builds` in `pr_builds.yml`.
- Keep `ci.yml`, `docker.yml`, and `release.yml` free of pull-request triggers.
  They own main/dispatch, manual Docker validation, and tag/release behavior.
- Use `pull_request`, not `pull_request_target`, for untrusted PR code.
  `pull_request_target` workflows must never check out, build, execute, source,
  or interpolate PR-controlled content. `pr_cleanup.yml` may only operate on
  positively matched cache/artifact metadata; `pr_auto_assign.yml` may only
  update PR metadata.
- Give scheduled workflows a manual `workflow_dispatch` path when safe so an
  operator can reproduce them. Type and describe every dispatch/reusable input,
  validate free-form strings, and set explicit defaults where omission is safe.
- Add `paths` filters only when a central routing signal cannot express the
  ownership. Keep trigger filters, `.github/actions/compute-changes`, affected
  crate logic, and the topology document synchronized.
- Add a concurrency group for any workflow that publishes, deploys, mutates
  caches, or would wastefully overlap. Cancel superseded PR validation; do not
  cancel releases, deployments, cleanup, or cache warming unless rollback
  semantics explicitly permit it.

## PR routing and job graph

- Route PR work from `.github/actions/compute-changes`. Do not add heavy jobs
  that ignore applicable `docs_only`, `rust_changed`, `backend_changed`,
  `inference_artifact_required`, `windows_*_build_required`,
  `sdk_smoke_required`, `ui_changed`, or `website_changed` outputs.
- Model Linux, macOS, and Windows executable products as independent
  backend-neutral host and native-runtime producers followed by
  composition-only product jobs. A platform/backend matrix belongs on the
  runtime and product layers, never on a host producer. Do not add no-op macOS
  CUDA, ROCm, or Vulkan rows for unsupported combinations.
- Gate native backend lanes on backend inputs, not on every Rust change.
  Workflow-only and docs-only changes must not fan out into build, GPU,
  benchmark, or SDK smoke lanes without a matching product input.
- Gate public/ARC runner-image contract jobs on runner workflow, cache
  integration, or cache-version changes (plus manual dispatch). Do not make
  ordinary source/docs PRs pay an infrastructure canary that validates no
  changed contract.
- Keep Clippy sharding driven by `scripts/plan-clippy-batches.sh`; do not add
  hand-maintained static batches.
- Keep crate-test sharding driven by `scripts/plan-test-batches.sh`. It derives
  workspace membership from `cargo metadata`; do not add a workflow-owned test
  crate allowlist. Pull requests test affected crates and reverse dependents;
  main and manual dispatch test every workspace member exactly once.
- When adding, removing, renaming, or splitting a workspace crate, update
  `.github/actions/compute-changes`, `scripts/affected-crates.sh`,
  `scripts/plan-clippy-batches.sh`, Docker copy lists,
  `scripts/publish-crates.sh`, workflow crate lists, and xtask consistency
  expectations together. Do not add new crates to `plan-test-batches.sh`; its
  metadata-derived membership and default weight handle them automatically.
- If a consumer downloads an artifact, its producer must be reachable in the
  same workflow graph under every matching condition. Use `needs` with normal
  dependency-success semantics, or an explicit result check when status-aware
  continuation is intentional; do not rely on job ordering by file position.
- Give each PR entry workflow one stable, non-matrix summary job suitable for
  branch protection. Conditional top-level jobs and the summary must consume
  the same checked-in required-job plan so route conditions cannot drift. The
  summary must directly need every other top-level job, use
  `if: ${{ !cancelled() }}` rather than `always()`, require unconditional
  routing/planning jobs to succeed, and permit `skipped` only when that job is
  absent from the plan. Reject required skips, failure, cancellation, unknown
  results, duplicate plan entries, and required IDs outside the needs graph.
- Set `strategy.fail-fast: false` when every platform/backend result is useful.
  Use fail-fast only when later matrix results would be redundant or unsafe.

## Workflow and action definition

- Start with least privilege. Declare workflow- or job-level `permissions`
  explicitly; grant `contents: read` unless a job demonstrably needs more.
  Scope `actions: write`, `contents: write`, `packages: write`, `pages: write`,
  `pull-requests: write`, or `id-token: write` to the smallest job and event.
- Set `persist-credentials: false` on checkout unless a narrowly scoped job must
  push. Do not use a PAT when the job-scoped `GITHUB_TOKEN` can perform the
  operation.
- Pin newly introduced third-party actions to a full commit SHA and record the
  human-readable release in a comment. Do not add `@main`, `@master`, or another
  moving ref. Do not churn unrelated legacy action pins in a focused change,
  but treat moving refs as migration debt when touching that step.
- Prefer a local composite action or typed reusable workflow when logic is used
  by more than one job. Keep reusable inputs typed, defaults explicit, outputs
  documented, and secrets passed deliberately rather than inherited broadly.
- Give jobs and steps stable descriptive names. Step IDs must be unique within
  a job and change only when consumers are updated. Keep expressions out of
  shell strings when untrusted data is possible; pass values through `env` and
  quote them in the shell.
- Use `set -euo pipefail` for nontrivial Bash orchestration. Select the shell
  explicitly when container or platform defaults differ. Use PowerShell-native
  error handling on Windows.
- Add realistic `timeout-minutes` to network, integration, benchmark, and
  deployment jobs. A timeout is a failure boundary, not a substitute for fixing
  a hang.
- Every CI invocation of `mesh-llm` must include `--log-format json` so a TUI is
  never started on a noninteractive runner.

## Runners, workers, and container images

- Use the exact hosted and self-hosted labels in the inventory. Keep runner
  selection data as JSON when passed through `fromJson`; validate manual runner
  input against an allowlist before execution.
- Do not route fork-authored or otherwise untrusted code to a persistent
  self-hosted runner. Use ephemeral ARC pods with restricted service accounts,
  credentials, network access, and namespaces for untrusted workloads, or keep
  the workload on GitHub-hosted runners.
- Linux CI should converge on the multi-architecture images from
  `Mesh-LLM/mesh-llm-runner-images`. Use the public variant as a job-level
  container on GitHub-hosted runners. The ARC runner pod is already the
  self-hosted variant; do not wrap it in another job container.
- Pin production runner consumers by the multi-architecture OCI digest:
  `ghcr.io/mesh-llm/mesh-llm-cuda-runner@sha256:<digest>`. Treat timestamp,
  source-revision, and `*-latest` tags as discovery/evaluation inputs only.
  Resolve a selected tag to its digest before changing required CI or Flux.
- Preserve architecture and hardware constraints. AMD64 NVIDIA work requires
  the full GPU label set and appropriate device/runtime access. ARM64 work must
  use an ARM64 image child and label. Verify both children in the manifest list
  before rollout.
- Treat worker-count variables as capacity and API-rate controls. Validate
  integer ranges, cap fan-out, retain deterministic sharding, and consider
  runner availability, GitHub API limits, cache pressure, and cost before
  increasing parallelism.
- Do not change runner labels, `USE_SELF_HOSTED`, image digests, scale-set
  names, node selectors, resource requests, or worker counts in only one side
  of the contract. Update workflow routing, runner/GitOps configuration,
  inventory, and verification together.
- Route trusted main/release Depot jobs through the repository-wide
  `DEPOT_RUNNERS_ENABLED` exact-string gate, with a typed manual canary input
  accepted only when `github.ref == 'refs/heads/main'` and a GitHub-hosted
  fallback for tags and every other ref. Current
  `pull_request` workflows must always select GitHub-hosted runners;
  `DEPOT_PR_RUNNERS_ENABLED` is ignored.
- A default-branch-pinned reusable workflow that directly allocates Depot must
  not accept a caller-supplied runner label or Depot-cache permission. Derive
  both inside the protected workflow from the exact repository, event, main
  ref, `DEPOT_RUNNERS_ENABLED` gate or event-owned main-dispatch canary flag,
  target architecture, and a validated bounded runner-size input. Pull
  requests, tags, feature refs, external repositories, and unsupported targets
  must never resolve to a Depot label; the cache permission must be the output
  of the same decision.
- Reusable smoke workflows that receive `HF_TOKEN` must remain fixed to
  GitHub-hosted runners during the Depot rollout. They must not accept raw
  `runs_on` JSON or any other caller-provided label that can resolve to Depot or
  a dedicated runner group. Multi-platform smoke/producer APIs may expose only
  bounded GitHub-hosted labels and must fail closed before running checked-out
  source. Pull-request callers must not pass `HF_TOKEN`; use public fixtures and
  merge-ref-scoped caches for untrusted PR validation.
- Treat a checked-out repository-local selector as defense in depth, never as
  the PR runner trust boundary: pull requests can modify both their workflow and
  local action code. Before any PR uses Depot, automatic cache authority must be
  disabled and isolated, then a default-branch-pinned, narrowly typed reusable
  workflow must be restricted to its exact `@refs/heads/main` workflow ref with
  `restricted_to_workflows=true`. Do not use `pull_request_target` to build or
  execute PR content.
- Confirm the Depot GitHub App, public-repository runner-group access,
  selected-workflow restriction, and every selected runner label before
  enabling Depot. Hardware-qualified GPU execution remains on a restricted
  device runner.

## Dependencies and runner setup

- Treat MeshLLM manifests/lockfiles and the YAML profiles/installers in
  `mesh-llm-runner-images` as the dependency sources of truth.
- Never fix a Linux CI failure by adding a one-off `apt-get`, `pip`, global
  `npm`, `cargo install`, downloaded binary, `curl | sh`, setup action, or host
  bootstrap step to an individual workflow.
- Put Rust, Node, Python, Go, test, and SDK dependencies in their checked-in
  project manifests and lockfiles. Put shared Linux packages and CLIs in
  `profiles/common.yml`, backend-specific SDK packages in
  `profiles/backends/<backend>.yml`, environment-only capabilities in
  `profiles/public.yml` or `profiles/self-hosted.yml`, and vendor toolchains in
  the owning runner-image installer. Rebuild and verify every supported
  backend/architecture pair, publish, then pin the new digest.
- Locked project installation remains valid job work. The runner image warms
  dependency caches but does not replace the manifests as the contract.
- Centralized platform setup may remain on macOS or Windows where the Linux
  image is not applicable. Keep it reusable and version-pinned; do not copy it
  between jobs.
- Treat existing Linux workflow-local host setup as migration debt. Remove it
  when its lane adopts the runner image; never copy it to a new lane.
- Permit an emergency workaround only when it is explicitly temporary and has
  a reason, owner, and linked removal issue or expiry date.

## Variables, secrets, tokens, and environments

- Use a workflow input for a one-run operator choice, a repository variable for
  nonsecret repository-wide configuration, an environment variable for
  deployment-scoped nonsecret configuration, and a secret for credentials or
  private values. Never store a secret in a variable, workflow default,
  artifact, cache, summary, or committed file.
- Treat GitHub variables as strings. Normalize booleans explicitly, use
  `fromJson` only for validated JSON, validate numeric ranges, and provide safe
  checked-in defaults when absence is allowed. Fail early with the missing name
  when a value is required.
- Pass secrets only to the step that consumes them, normally through `env`.
  Do not interpolate secrets into command lines, cache keys, artifact names,
  matrices, job outputs, or debug traces. Do not expose secrets to PRs from
  forks or to untrusted reusable workflows/actions.
- Use environment protection and scoped deployment credentials for production
  publishing/deployments. Grant OIDC `id-token: write` only to the job that
  exchanges the token.
- Before adding or renaming a variable or secret, search every workflow,
  action, script, and downstream repository consumer. Update the inventory and
  document scope, owner, accepted format, default/failure behavior, and rotation
  or removal plan. Never document a secret value.
- Inspect live configuration with `gh variable list`, `gh secret list`, and the
  environment commands in the inventory. An absent repository secret may be an
  organization secret; lack of permission to list org configuration is not
  evidence that it does not exist.
- Creating, changing, or deleting a variable/secret is an external state
  mutation. Confirm scope and exact target, use `gh secret set` interactively or
  via stdin/file without echoing the value, and report only the name and scope.

## Caches, artifacts, and smoke tests

- Treat release production as a three-layer graph: one backend-neutral host per
  OS/architecture, one native-runtime artifact per
  OS/architecture/backend/backend-version, and product composition from those
  two immutable inputs. A backend alias may select a runtime but must never
  compile a distinct host.
- Host artifacts must include and pass the dynamic-import policy report.
  Runtime artifacts must include their manifest, file checksums, runtime ABI,
  MeshLLM version, platform, and backend compatibility. Product artifacts must
  record the exact host and runtime digests and preserve the
  `mesh-bundle/native-runtimes/<runtime-id>` layout.
- Use `.github/actions/prepare-host-input`,
  `.github/actions/prepare-windows-host-input`,
  `.github/actions/prepare-native-runtime-input`, and
  `.github/actions/compose-product-input` for PR/main/release producers.
  `prepare-host-input` owns Unix hosts and `prepare-windows-host-input` owns
  Windows hosts. Add release-only signing/publishing around those actions; do
  not fork their build/package/compose commands into workflow-local shell
  blocks.
- Keep host, runtime, and product matrices separate in release workflows.
  Product jobs download producer artifacts and verify compatibility and digest
  metadata before composition. Never satisfy a missing producer by rebuilding
  either layer in a consumer job.
- Host producers attest and import-check the host before upload. Consumers
  verify that producer checksum and attestation, then copy the exact host bytes;
  they must not re-stamp, relink, or otherwise mutate a host per backend alias.
- Main CI executable lanes follow the same product contract: CPU artifact
  producers upload a backend-neutral host together with its adjacent packaged
  runtime. A backend product lane consumes the unchanged, verified host from
  its OS/architecture producer and combines it with its selected runtime; it
  must not independently produce a backend-specific host. Do not upload or
  consume a raw host binary, rebuild a host after restoring a runtime cache, or
  use a driver stub to make an executable test pass.
- No-driver smoke is mandatory for every product alias, native package, and OCI
  image. CUDA/ROCm/Vulkan device absence is not a skip condition for
  `--version`, runtime discovery/listing, or client startup. Hardware-qualified
  serving tests are additional coverage.
- Namespace cache keys and include every compatibility boundary that can make
  reuse unsafe: OS, architecture, backend/toolchain, relevant lockfiles,
  `.github/cache-version.txt`, and build inputs. Do not broaden restore keys
  across incompatible or untrusted contexts.
- Windows native-runtime producers and the trusted warmer must use
  `.github/actions/restore-windows-abi-cache`. Keep CPU, CUDA, ROCm, and Vulkan
  architecture/toolchain identities exact; do not duplicate its key expression
  in individual workflows or add broad restore prefixes.
- GitHub-hosted PR jobs may share the normal key namespace with main because
  GitHub scopes PR writes to the merge ref and trusted main does not restore
  them. Do not assume that isolation applies to another cache provider.
- Keep the GitHub Actions sccache remote tier read-only for `pull_request` and
  `pull_request_target` events. PR jobs may read trusted default-branch entries
  and write their job-local disk tier, but only trusted main, release,
  scheduled warmer, or explicitly authorized dispatch paths may publish shared
  sccache entries. Apply the same event-derived mode to direct
  `mozilla-actions/sccache-action` users and to
  `.github/actions/configure-sccache-gha`; do not let a reusable workflow
  silently restore read-write PR publication.
- Depot's GitHub cache namespace is repository-scoped and has no branch
  isolation. With automatic Depot Cache enabled, its authority is injected into
  the whole runner job and cannot be contained by sccache disk-only mode or
  cache-key conventions. Current PR workflows must not use Depot, including
  through a trusted reusable caller, until automatic injection is disabled and
  complete token/API isolation is proven.
- Do not save large shared Rust caches from PR merge refs. Shared caches are
  written from trusted main/release/cache-warming paths. PR cleanup may delete
  positively matched PR caches/artifacts but must not delete workflow runs or
  logs.
- Use `retention-days: 1` for PR and smoke-only artifacts unless a documented
  debugging or release requirement needs longer retention. Release evidence
  follows the release policy, not the PR default. Sccache migration evidence is
  retained for 14 days so cold/warm samples cover the configured Depot
  cache-retention window.
- Restore producer artifacts through `.github/actions/restore-smoke-inputs`.
  Reuse `smoke.yml`, `scripted-binary-smoke.yml`, `sdk-smoke.yml`, and
  `hf-download-smoke.yml`; do not rebuild MeshLLM, native runtimes, or duplicate
  model/artifact restore blocks in consumers. SDK smokes consume the runtime
  adjacent to their staged producer binary and must fail rather than silently
  compiling a replacement in CI.
- Build Swift XCFramework inputs through the typed
  `swift-sdk-artifact.yml` reusable producer. Pull-request validation uses its
  `host-only` mode, while main and release use `full`; Swift smoke consumers
  download and verify both the immutable XCFramework and generated
  `mesh_ffi.swift` artifacts and must not invoke Cargo, llama.cpp builds,
  native-SDK packaging, or either XCFramework build script. Producer and smoke
  are fixed to `macos-15`; the native cache includes an explicit macOS/Xcode
  epoch, Rust uses `RUSTC_WRAPPER=sccache`, and the producer retains
  mode/run-attempt-unique sccache evidence. Main and tag producers must fail on
  tracked-binding drift, while a dispatched release must copy the producer
  binding into its tag commit.
- Build native SDK runtime inputs through the typed
  `native-sdk-artifact.yml` reusable producer. Pull-request, main, and release
  callers select an explicit target, backend, and Cargo profile; Kotlin smoke
  downloads and verifies that immutable producer artifact and must not invoke
  Cargo, llama.cpp preparation/builds, or native-SDK packaging. Release callers
  use the same producer with release-asset staging enabled so archive, checksum,
  and native runtime crate names remain identical to the published contract.
- Build Linux static llama ABI inputs through the typed
  `static-abi-artifact.yml` reusable producer. Its artifact must carry an exact
  target/backend manifest, pinned build-image/toolchain epoch, build-stamp
  checksum, archive checksum, the full llama/common/mtmd/ggml static link
  closure, and only the canonical minimal `build-stage-abi-static` link tree.
  Cache and artifact payloads must contain the same path-normalized archive,
  not CMake's producer-local build graph. Crate tests and native SDK producers
  restore it with `restore-static-abi-input.sh`;
  never extract it with raw `tar`, relabel it across architectures, or rebuild
  the same CPU ABI in a downstream consumer. Native SDK reuse must call the
  verification-only prebuilt path with build.rs auto-build disabled.
- Never put credentials, local absolute paths, private endpoints, or secret
  material into cache/artifact content or workflow summaries.

## Running, diagnosing, and updating CI

1. Resolve the exact workflow, ref/SHA, event, run ID, and trust context. Inspect
   recent comparable runs and changed workflow history before acting.
2. For a failure, read the failed job and step logs, then classify it as product
   code, declared dependency, runner image, worker capacity, cache/artifact,
   secret/variable, permission, external service, or workflow logic. Fix the
   owning source; do not paper over it with retries or installers.
3. Dispatch only the narrowest safe workflow on the intended branch/SHA. Record
   the run URL and input values, excluding secrets. Use canary/dry-run inputs for
   release/deploy workflows whenever available.
4. Watch the run to a terminal conclusion and inspect failed logs. Rerun failed
   jobs only for demonstrated transient infrastructure failures. Push a code or
   configuration fix for deterministic failures.
5. Do not report success while jobs are queued/in progress or because an
   unrelated run passed. State expected skips explicitly and verify required
   checks on the PR.

Use these operational commands as appropriate:

```bash
gh workflow list --repo Mesh-LLM/mesh-llm
gh workflow view WORKFLOW.yml --repo Mesh-LLM/mesh-llm --yaml
gh workflow run WORKFLOW.yml --repo Mesh-LLM/mesh-llm --ref BRANCH -f key=value
gh run list --repo Mesh-LLM/mesh-llm --workflow WORKFLOW.yml --limit 20
gh run watch RUN_ID --repo Mesh-LLM/mesh-llm --exit-status
gh run view RUN_ID --repo Mesh-LLM/mesh-llm --log-failed
gh pr checks PR_NUMBER --repo Mesh-LLM/mesh-llm
```

## Validation contract

Run the smallest applicable set and do not claim a check passed until it exits
with status 0.

For every workflow/action edit:

```bash
actionlint -config-file .github/actionlint.yaml
git diff --check
```

Also:

- Run `shellcheck` on changed shell scripts and substantial extracted Bash.
- Run `cargo run -p xtask -- repo-consistency ci-crate-lists` for PR routing,
  affected-crate, Clippy batch, workspace, or crate-list changes.
  This also verifies that generated crate-test batches cover every workspace
  member exactly once.
- Run `cargo run -p xtask -- repo-consistency release-targets` for release
  target, packaging, Docker, or release-workflow changes.
- Run `cargo run -p xtask -- repo-consistency publish-crates` for crate
  publishing changes.
- Run the owning local action/script tests for action or routing logic. Exercise
  both true and false/skip branches when changing a condition.
- Validate significant changes in GitHub Actions using a PR or an authorized
  `workflow_dispatch` on the branch. Prove docs-only skips, relevant product
  execution, expected matrix rows, artifact producer/consumer reachability,
  runner architecture, and secret/permission behavior as applicable.
- Keep `ci/ci.md` synchronized with topology changes and
  `references/current-inventory.md` synchronized with workflow, runner,
  variable, secret-name, environment, or ownership changes.

Finish by reporting changed files, operational state changes, validation run
IDs/URLs and conclusions, expected skips, unresolved risks, and any live
configuration the current GitHub permissions could not verify.
