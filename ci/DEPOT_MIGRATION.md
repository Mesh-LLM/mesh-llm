# Depot runner transition

Status: trusted-main support exists; pull-request execution is future work and
is disabled.

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
- Pull requests, feature refs, tags, credential-bearing smokes, macOS, Windows
  and hardware-qualified GPU work retain their approved non-Depot placement
  until separately migrated.
- Callers never provide a raw Depot label or a separate remote-cache
  permission. Runner and cache policy come from one event-derived decision.

Disabling Depot must change placement only. It must not change plan membership,
commands, artifacts, smoke coverage or required checks.

## Required migration after the composable graph lands

The protected controller and split lane workflows change which workflow files
own and call eligible Linux jobs. The first `main` push after that change lands
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
   GitHub-hosted fallback active. Confirm that the first protected `Main CI`
   controller run dispatches the separate Quality, Website, Linux, macOS and
   Windows workflows and that `CI Required` completes.
3. Run a same-repository activation PR on GitHub-hosted workers. Confirm that
   PR-origin dispatch remains GitHub-hosted even though the lane definitions run
   from `main`; `original_event_name=pull_request` must keep Depot and Depot
   remote-cache authority disabled.
4. From `main`, manually dispatch `Main CI` with `use_depot=true`. This exercises
   the bootstrap graph as a bounded provider canary without changing the plan,
   commands, artifacts or required summaries. Verify the eligible jobs report
   Depot labels and Depot cache evidence while macOS, Windows, credentialed
   smoke and GPU jobs retain their documented providers.
5. When the canary is green, set `DEPOT_RUNNERS_ENABLED=true` for normal trusted
   `main` pushes and verify one protected split-lane run. Quality and Linux
   slices may select Depot; PR, Website-only, macOS and Windows work must not.
6. Roll back by setting `DEPOT_RUNNERS_ENABLED=false`. Re-run the same profile
   and verify that the identical plan executes on GitHub-hosted Linux workers.

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
Mesh-LLM/mesh-llm/.github/workflows/ci.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-orchestrator.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-quality-lane.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-linux-lane.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-quality-slice.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-host-slice.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/ci-runtime-product-slice.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/depot-canary.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/depot-registry-canary.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/native-sdk-artifact.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/release.yml@refs/heads/main
Mesh-LLM/mesh-llm/.github/workflows/static-abi-artifact.yml@refs/heads/main
```

Credential-bearing `hf-download-smoke.yml`, `smoke.yml`,
`scripted-binary-smoke.yml`, and `sdk-smoke.yml` are not in the allowlist and
remain GitHub-hosted. `swift-sdk-artifact.yml` remains on `macos-15`.

Verify the live runner-group repository and selected-workflow restrictions with
organization-admin authority before any rollout. A repository token returning
`403` is unverified state, not proof of a safe configuration.

## Why PRs cannot use Depot today

Cache isolation is the prerequisite for any future PR execution on Depot.

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

## Required future investigation

Depot documents an organization setting named **Allow Actions jobs to
automatically connect to Depot Cache**. Before using Depot for PRs, an
organization administrator must disable or isolate automatic cache connectivity
in a controlled canary and prove the resulting behavior.

Required questions:

1. Does disabling the setting remove `DEPOT_CACHE_TOKEN`,
   `SCCACHE_WEBDAV_*`, tool-specific cache configuration and transparent
   GitHub-cache API redirection from the PR job?
2. Does `actions/cache` then use GitHub's native branch-isolated cache, or must
   all remote cache actions be disabled on Depot PR runners?
3. Can Depot provide a short-lived cache token scoped to workflow/ref and
   read/write policy? Cache authentication does not currently document project
   tokens as supported.
4. If automatic connectivity is disabled organization-wide, how will trusted
   main cache safely: GitHub native cache, no remote compiler cache, an isolated
   Depot tenant/runner group, or a Depot-supported per-workflow policy?
5. Can the runner group be verified as exact-workflow restricted by an
   organization administrator?

Do not introduce a long-lived Depot organization token without a separate
security review and explicit authorization.

## Future protected PR executor

After cache isolation is proven, a normal code PR may call a protected
orchestrator and runner-owning reusable slices pinned to `refs/heads/main`.
Every workflow whose job directly owns a Depot `runs-on` must be selected in
the runner group; allowlisting only the outer caller is insufficient. The
protected workflows must:

- own `runs-on` and cache mode;
- accept only bounded semantic slice inputs and a source SHA;
- check out the PR merge SHA as untrusted code;
- receive no repository secrets or registry credentials;
- run on ephemeral Depot instances;
- be exact selected workflows allowed by the runner group.

CI workflow, action, planner, ownership, runner and cache-policy changes must
remain on the local GitHub-hosted path. A PR may not modify the workflow that
grants its own Depot placement.

## Isolation acceptance canary

Use a non-secret sentinel protocol:

1. Trusted main writes a random sentinel cache entry.
2. A same-repository PR and a fork PR run through the proposed Depot path.
3. Both probe `actions/cache`, Depot/WebDAV variables, automatic tool
   configuration, and direct cache access.
4. Both must fail to read the trusted sentinel.
5. Both must fail to publish an entry that trusted main later restores.
6. Both must receive no cache/registry token and no repository secret.
7. Trusted main must retain its intended cache behavior.
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
