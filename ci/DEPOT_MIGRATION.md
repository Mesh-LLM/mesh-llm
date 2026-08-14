# Depot runner transition

Status: trusted-main support exists; pull-request execution is future work and
is disabled. Automatic Depot Cache and Registry Actions connectivity are
admin-verified off, and the Depot runner group is admin-verified restricted to
this repository and the protected workflow allowlist. The checked-in graph and
collector define the remaining branch/main and same-repository/fork canaries;
those runtime checks still block PR activation.

Checked-in policy now treats Depot's cache namespace as unused: every Depot
selection disables both native GitHub Actions cache and Depot remote cache.
Purge or expiry of existing entries remains an activation prerequisite.

The complete PR/main composition design is in
`.omo/specs/pr-ci-optimization.md`. This document contains only the durable
Depot policy and activation gates. It intentionally contains no historical run
results or timing conclusions.

## Provider contract

Depot is a placement provider, not a different CI graph.

- The same reusable workflow slice, commands, profile, artifact contract and
  verification must run on GitHub and Depot.
- `.github/actions/select-ci-runners` owns current Linux provider selection.
- `DEPOT_RUNNERS_ENABLED == 'true'` permits eligible trusted `main` Linux jobs
  to select Depot. Every eligible role retains a GitHub-hosted fallback.
- `DEPOT_PR_RUNNERS_ENABLED == 'true'` is the independent same-repository PR
  placement gate. It must remain absent or `false` until the branch/main
  provider-parity, same-repository, and fork sentinel canaries below pass; a
  missing value fails closed.
- `DEPOT_PR_CANARY_REF` is an optional, exact
  `refs/pull/<number>/merge` selector for one protected same-repository canary
  ref. It does not enable the global PR gate, does not grant remote cache
  permission, and fails closed for forks, target/dispatch events, malformed or
  non-matching refs, and planner-forced hosted paths. It remains unset until
  the external isolation gates pass.
- Pull requests, feature refs, tags, credential-bearing smokes and
  hardware-qualified GPU work retain their approved non-Depot placement until
  the protected PR executor is separately activated. The intended executor may
  cover eligible build/test jobs in Linux, Depot macOS 15 and Windows 2022
  lanes when an equivalent image/architecture exists; control-plane planning
  and required summaries remain hosted, and Intel macOS without an equivalent
  remains hosted.
- Callers never provide a raw Depot label or a separate remote-cache
  permission. Runner and cache policy come from one event-derived decision.
- The PR isolation audit receives that selected-provider and cache-policy
  decision. Depot-selected PR jobs accept only GitHub-owned Actions endpoints
  or a strict loopback proxy (`http[s]://localhost|127.0.0.1|[::1]:<port>/<path>`);
  hosted PR jobs retain their approved local transport while still rejecting
  Depot credentials, `depot.dev` redirects, and URL userinfo.
- The Depot cache namespace is intentionally unused by trusted workflows:
  Depot selections disable both native GitHub cache APIs and Depot remote
  cache. Purge or expire any existing namespace entries before enabling the PR
  gate. A proxy's presence is inert transport, not proof of authority
  isolation; activation remains blocked until no trusted consumer uses Depot
  cache.

Disabling Depot must change placement only. It must not change plan membership,
commands, artifacts, smoke coverage or required checks.

The PR end state is therefore a protected execution-policy change, not a
`runs-on` label swap. The five native PR entrypoints and their matching
protected reusable lanes remain intact. A selected PR slice keeps its main
commands, profile, artifact identities, tests, `needs` edges, summaries,
fail-fast profile and required-check result. Only the event-derived provider,
cache mode and ephemeral runner allocation may differ. The runner-owning
workflow checks out the immutable PR SHA with `persist-credentials: false`,
receives no PR secrets or registry/cache credentials, and forces the hosted
path for CI-control/workflow/policy changes. The planner-owned
`signals.runner_contract_required` value is passed as `force_hosted` through
every protected lane and runner-owning slice, and the centralized selector
requires it to be false before enabling Depot. Control-plane planning/required
summaries, credential-bearing smoke, `gpu-nvidia` hardware, and any Intel macOS
row without a Depot-equivalent remain their approved provider exceptions.

## Required migration after the composable graph lands

The protected controller and split lane workflows change which workflow files
own and call eligible jobs. The first `main` push after that change lands
can select Depot immediately when `DEPOT_RUNNERS_ENABLED` is already `true`.
GitHub does not fall back to a hosted runner when a selected Depot label is
blocked by runner-group policy; the job remains queued. Complete this sequence
when landing the composable graph:

1. Before merging, use organization-admin authority to verify the runner group
   is limited to `Mesh-LLM/mesh-llm`, permits this public repository
   deliberately, has `restricted_to_workflows=true`, and contains every exact
   protected workflow ref in the allowlist below. If that cannot be verified,
   set `DEPOT_RUNNERS_ENABLED=false` before merging.
2. Merge the graph change with either the verified allowlist or the
   GitHub-hosted fallback active. Confirm that the first protected `CI · Plan`
   run after `Main CI` dispatches the separate Quality, Website, Linux, macOS
   and Windows workflows and that `CI Required` completes.
3. Run a same-repository activation PR on GitHub-hosted workers. Confirm that
   PR-origin dispatch remains GitHub-hosted even though the lane definitions run
   from `main`; `original_event_name=pull_request` must keep Depot and Depot
   remote-cache authority disabled.
4. From `main`, manually dispatch `CI · Plan` with `use_depot=true`. This
   exercises the split Quality/Linux graphs as a bounded provider canary
   without changing the plan, commands, artifacts or required summaries.
   Verify the eligible jobs report Depot labels and no Depot cache evidence (the
   namespace must remain inert) while
   macOS, Windows, credentialed smoke and GPU jobs retain their documented
   providers.
5. When the canary is green, set `DEPOT_RUNNERS_ENABLED=true` for normal trusted
   `main` pushes and verify one protected split-lane run. Quality and Linux
   slices may select Depot; PR, Website-only, macOS and Windows work must not.
6. Roll back by setting `DEPOT_RUNNERS_ENABLED=false`. Re-run the same profile
   and verify that the identical plan executes on GitHub-hosted Linux workers.

The current administrative posture has been verified outside the repository:
automatic Actions Cache connectivity to Depot is disabled, automatic Registry
Actions authentication is disabled, and the Depot runner group is restricted
to `Mesh-LLM/mesh-llm` with exact protected workflow refs. The repository token
cannot independently inspect organization runner-group settings through the
API (403), so this remains external activation state rather than checked-in
proof. The settings remove automatic credential authority; they do not prove
that a protected main build, same-repository PR, or fork PR receives the
intended provider, cache mode, or rollback behavior.

Do not migrate required checks, enable Depot for PR content, or change cache
isolation during this provider rollout. Those are independent changes with
their own acceptance gates below.

## Current intended runner-group boundary

Depot-managed runners register in a GitHub organization runner group. For a
public repository, that group must be limited to `Mesh-LLM/mesh-llm`, permit
public repository use deliberately, set `restricted_to_workflows=true`, and
allow only exact protected default-branch runner-owning workflow refs.

The current main allowlist is:

```text
Mesh-LLM/mesh-llm/.github/workflows/ci-control.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-quality-lane.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-linux-lane.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-quality-slice.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-web-slice.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-ui-artifact-slice.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-linux-host-slice.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-linux-runtime-slice.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-linux-product-slice.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-rust-tests-slice.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-macos-host-slice.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-macos-runtime-slice.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-macos-product-slice.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-windows-host-slice.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-windows-runtime-slice.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-windows-product-slice.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-platform-checks-slice.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/depot-canary.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/depot-registry-canary.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/native-sdk-artifact.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/release.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/static-abi-artifact.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/swift-sdk-artifact.yml@refs/heads/main
```

Credential-bearing `hf-download-smoke.yml`, `smoke.yml`,
`scripted-binary-smoke.yml`, and `sdk-smoke.yml` are not in the allowlist and
remain GitHub-hosted. `swift-sdk-artifact.yml` must be in the protected workflow
allowlist because it directly owns eligible PR Depot placement; its internal
main gate remains false, so release/main Swift production stays on `macos-15`.

The runner-group boundary above is an external administrative prerequisite.
Re-verify it with organization-admin authority if the group, repository, or
workflow allowlist changes. A repository token returning `403` is not an
independent proof of the configuration.

## Why PR canaries remain pending

Disabling automatic Depot Cache and Registry Actions connectivity removes the
repository-wide credential path that blocked the first canary. It does not by
itself prove provider parity, branch/main cache behavior, or that a same-
repository or fork PR cannot publish an entry later restored by trusted main.
Those runtime isolation checks remain prerequisites for PR execution on Depot.

Depot documents these GitHub Actions cache properties:

- all GitHub-cache API consumers on a Depot runner automatically use Depot
  Cache, including `actions/cache` and caching in `setup-*` actions;
- cache entries are repository-scoped;
- cache entries are not branch-isolated.

GitHub's native cache isolates PR writes to `refs/pull/<n>/merge`, so trusted
main cannot restore a PR-written entry. Depot does not document an equivalent
branch boundary.

An untrusted PR with a repository-scoped Depot cache token can ignore project
cache keys and access the service directly. Prefixes, restore-only intent,
job-local sccache and a default-branch caller do not remove that authority.
This creates a cache-poisoning path into trusted main/release consumers.

Official references:

- [Depot GitHub Actions cache](https://depot.dev/docs/cache/integrations/github-actions)
- [Depot sccache integration](https://depot.dev/docs/cache/integrations/sccache)
- [Depot runner architecture](https://depot.dev/docs/github-actions/overview)
- [GitHub dependency cache isolation](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching)
- [GitHub runner-group API](https://docs.github.com/en/rest/actions/self-hosted-runner-groups)

## Cache isolation and remaining registry investigation

Depot documents an organization setting named **Allow Actions jobs to
automatically connect to Depot Cache**. That setting and automatic Registry
Actions authentication are now admin-verified off. The negative
`depot-canary.yml` contract checks that this posture reaches fresh runners
without Depot cache, WebDAV, registry, or transparent GitHub-cache authority.

The remaining runtime questions are:

1. Does a fresh Depot runner expose any ambient Depot/WebDAV/cache authority
   to a PR job even when the checked-in native cache consumers are disabled?
2. Do hosted release and cache-warmer jobs retain their intended GitHub cache
   behavior while trusted Depot selections remain cache-inert, and have all
   existing Depot namespace entries been purged or expired?
3. Can a same-repository PR and a fork PR both run the protected canary with no
   Depot cache/registry authority and no entry that trusted main later restores?
4. Does provider parity hold for the same checked plan, commands, artifacts,
   and required results on GitHub and Depot?
5. Does the restricted runner group continue to allow only the exact protected
   workflow refs after the PR canary is enabled?

The selector now emits provider-derived `allow_native_github_cache` and
`allow_depot_remote_cache` outputs. Every Depot selection emits
`allow_native_github_cache=false` and `allow_depot_remote_cache=false`; hosted
selections retain native GitHub cache (`allow_native_github_cache=true`) while
Depot remote cache remains disabled (`allow_depot_remote_cache=false`). Every
eligible Depot-selected direct PR cache consumer is skipped or passed a
disabled cache input, while installation and build commands remain active;
cache misses therefore rebuild normally. This closes the checked-in native
GitHub-cache path but does not prove that a fresh Depot runner cannot reach an
ambient Depot/WebDAV/cache service. The canary must still target the actual
authority with an approved non-secret marker protocol and verify no read/write
or registry token access; it remains an external runtime/admin prerequisite.

Do not introduce a long-lived Depot organization token without a separate
security review and explicit authorization.

## Future protected PR Depot executor

After cache isolation is proven, the protected planner may place eligible
normal-code PR lane calls on runner-owning Depot reusable slices pinned to
`refs/heads/main`.
Every workflow whose job directly owns a Depot `runs-on` must be selected in
the runner group; allowlisting only the outer caller is insufficient. The
protected workflows must:

- own `runs-on` and cache mode;
- declare least-privilege `permissions: contents: read` unless a narrower
  documented permission is sufficient;
- accept only bounded semantic slice inputs and a source SHA;
- check out the immutable PR head SHA as untrusted code with
  `persist-credentials: false`;
- receive no repository secrets or registry credentials;
- run on ephemeral Depot instances;
- be exact selected workflows allowed by the runner group.

After every acceptance canary passes, activate PR placement with the repository
variable `DEPOT_PR_RUNNERS_ENABLED=true`. Roll back immediately by setting it to
`false` (or deleting it) and rerun the identical PR plan; this changes provider
placement only and must leave commands, matrices, artifacts, and required checks
unchanged. `DEPOT_RUNNERS_ENABLED` remains the separate trusted-main gate.

CI workflow, action, planner, ownership, runner and cache-policy changes must
remain on the local GitHub-hosted path. A PR may not modify the workflow that
grants its own Depot placement.

## Isolation acceptance canary

Use a non-secret sentinel protocol, keeping trusted-main creation separate from
the untrusted PR probe:

1. Trusted main creates a random non-secret marker through the actual approved
   Depot/WebDAV authority (never a repository secret and never a native
   GitHub-cache claim).
2. A same-repository PR and a fork PR run through the proposed Depot path.
3. Both probe `actions/cache`, Depot/WebDAV variables, automatic tool
   configuration, and direct cache access.
4. Both must fail to read the trusted sentinel.
5. Both must fail to publish an entry that trusted main later restores.
6. Both must receive no cache/registry token and no repository secret.
7. Hosted release/cache-warmer jobs must retain their intended GitHub cache
   behavior; trusted Depot selections must remain cache-inert.
8. Provider rollback must send the identical plan to GitHub-hosted runners.

Cache-key separation alone does not satisfy this test.

## Activation gate

Do not enable Depot for PR content until:

- the isolation canary passes for same-repository and fork PRs;
- live runner-group restrictions are admin-verified;
- protected runner-owning workflow refs require reviewed main changes;
- CI-control changes force GitHub-hosted execution;
- GitHub fallback passes the same slice fixtures;
- provider parity is validated on comparable non-CI-change PRs;
- rollback is documented and tested;
- maintainers explicitly authorize the external Depot/GitHub setting changes.

Start any later rollout with remote cache disabled. Canary one non-secret Linux
slice, then a Rust test slice, then the selected Linux product graph. Keep
credential-bearing, macOS, Windows and hardware work on their existing runners.

Depot PR activation must be a separate change after the composable CI graph is
complete. It must not be combined with routing, required-check, artifact or
branch-protection migration.

## Measurement and rollback evidence

Use `scripts/collect-ci-metrics.py` to monitor all five focused PR lanes and
their historical GitHub cohorts. Keep raw run/job JSON under `/tmp` or an issue
artifact. Schema-v3 reports separate workflow wall and queue, job runner queue,
measured dependency wait (otherwise `n/a`), execution, runner-minutes,
cancelled minutes and peak workers, grouped by provider, OS, architecture,
semantic role and Depot size. Use `--compare-input` only with matching plan
profile, selected slices, source/change class, image/toolchain epoch and cache
mode; the provider sets must be disjoint and job families common.

The date-independent rollout signals are deterministic: fewer than three job
queue samples is `insufficient_sample`; queue p95 over 60 seconds is `hold`;
job or terminal-job queue p95 at least 300 seconds is capacity-contaminated and
`rollback`; a candidate cohort is `eligible` only when provider separation and
all other comparability checks pass. A contaminated run may prove correctness
or artifact reuse but cannot prove provider latency. Rollback changes only the
central provider gate to GitHub-hosted and reruns the same plan; it does not
change build shape. Do not place dated conclusions or raw evidence in `ci/`.
