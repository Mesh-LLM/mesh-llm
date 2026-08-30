# Depot runner transition

Status: trusted-main support exists. A time-bounded, explicitly accepted
cache-risk exception permits eligible same-repository PR jobs
through 2026-09-14 UTC; forks remain hosted. Automatic Depot Cache and Registry Actions connectivity are
admin-verified off. Those switches remove Depot's direct `DEPOT_CACHE_TOKEN`,
build-tool cache preconfiguration and Registry Actions authentication from
fresh runners; they do not provide a documented per-connection/job/ref ACL or
disable for the GitHub Actions cache proxy/runtime-token path. The Depot runner
group is admin-verified restricted to this repository and the protected
workflow allowlist. The controlled trusted-main/PR authority probe demonstrates
unsafe repository-scoped cross-trust authority. Maintainers have knowingly
accepted that risk temporarily to improve CI iteration speed; it remains a
blocker to permanent activation.

During the bounded exception, eligible same-repository PR and trusted-main
Depot jobs may use the GitHub Actions cache API and therefore Depot's shared,
cross-branch namespace. Direct Depot build-tool remote cache remains disabled.
The complete findings, risks, approval controls, expiry, and rollback procedure
are recorded in `ci/DEPOT_PR_RISK_EXCEPTION.md`.

The complete PR/main composition design is in
`.omo/specs/pr-ci-optimization.md`. This document contains the durable Depot
policy, activation gates and the controlled authority-probe evidence below;
it intentionally contains no timing conclusions.

## Provider contract

Depot is a placement provider, not a different CI graph.

- The same reusable workflow slice, commands, profile, artifact contract and
  verification must run on GitHub and Depot.
- `.github/actions/select-ci-runners` owns current Linux provider selection.
- `DEPOT_RUNNERS_ENABLED == 'true'` permits eligible trusted `main` Linux jobs
  to select Depot. Every eligible role retains a GitHub-hosted fallback.
- `DEPOT_PR_RUNNERS_ENABLED == 'true'` activates the bounded exception for
  eligible same-repository PR jobs. The checked-in selector expires this path
  on 2026-09-14 UTC; a missing, false, or expired value fails hosted.
- `DEPOT_PR_CANARY_REF` is an optional, exact
  `refs/pull/<number>/merge` selector for one protected same-repository canary
  ref. It does not enable the global PR gate, does not grant remote cache
  permission, and fails closed for forks, target/dispatch events, malformed or
  non-matching refs, and planner-forced hosted paths. It remains unset until
  the external isolation gates pass.
- Fork pull requests, feature refs, tags, credential-bearing smokes and
  hardware-qualified GPU work retain their approved non-Depot placement. The bounded executor may
  cover eligible build/test jobs in Linux, Depot macOS 15 and Windows 2022
  lanes when an equivalent image/architecture exists; control-plane planning
  and required summaries remain hosted, and Intel macOS without an equivalent
  remains hosted.
- Callers never provide a raw Depot label or a separate remote-cache
  permission. Runner and cache policy come from one event-derived decision.
- The PR authority audit receives that selected-provider and cache-policy
  decision. During the exception it permits the selected Depot Actions-cache
  proxy but still rejects direct Depot cache tokens, WebDAV/sccache authority,
  registry credentials, Docker auth, and URL userinfo before checkout.
- The selector emits `allow_native_github_cache=true` only for eligible
  same-repository PR and trusted-main Depot work while the global
  exception and checked-in deadline are active. Depot transparently maps that
  API to its shared repository namespace. `allow_depot_remote_cache` remains
  false. This is accepted risk, not proof of authority isolation.

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
proof. The switches remove the provider's direct Depot build-tool and registry
credential path (including automatic `DEPOT_CACHE_TOKEN`/WebDAV setup), but
they do not document or enforce a per-connection/job/ref disable or ACL for
the GitHub Actions cache proxy and its runtime-token path. The sentinel proved
that this path remains repository-scoped and crosses the trusted-main/PR
boundary; it therefore remains unsafe even with both switches unchecked.

Do not migrate required checks or broaden the PR exception during this provider
rollout. The bounded cache-risk exception is an independent, reviewed decision;
permanent activation still has the acceptance gates below.

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

## Controlled authority probe: unsafe cross-trust result

The protected seed/verify workflows and reusable PR slice came from `main`
commit `9e977e246`; the authoritative PR rerun used source head `e0c9be507`.
The bounded sentinel ID/ref repository variables were temporarily armed for
the experiment and removed immediately afterward. Its exact evidence is:

| Phase | Evidence | Outcome |
| --- | --- | --- |
| Trusted-main seed | [run 31816775585](https://github.com/Mesh-LLM/mesh-llm/actions/runs/31816775585), `main` at `9e977e246` | Successful seed publication |
| Same-repository PR authority sentinel | [run 31816869128](https://github.com/Mesh-LLM/mesh-llm/actions/runs/31816869128), [job 94821057215](https://github.com/Mesh-LLM/mesh-llm/actions/runs/31816869128/job/94821057215), PR source `e0c9be507`, protected slice `9e977e246` | The sentinel job failed after restoring and exactly validating the trusted seed, then saving, clearing, restoring and exactly validating the poison marker; the overall superseded probe run was later cancelled during cleanup |
| Trusted-main verify | [run 31817111471](https://github.com/Mesh-LLM/mesh-llm/actions/runs/31817111471), [job 94821343605](https://github.com/Mesh-LLM/mesh-llm/actions/runs/31817111471/job/94821343605) | Restored and exactly validated the PR poison marker; then failed the intended expected-miss gate |

This is an unsafe repository-scoped cross-trust authority result: the
same-repository PR path could read the trusted marker and publish a marker that
trusted main later restored. The intended failures are evidence that the gates
fired after the probes, not successful isolation. It does not justify a claim
of isolation. The temporary decision in `ci/DEPOT_PR_RISK_EXCEPTION.md`
explicitly accepts this risk for eligible same-repository PRs; a
provider-isolation redesign and a new successful sentinel remain
required before the exception can become permanent. `DEPOT_PR_CANARY_REF`,
`DEPOT_PR_SENTINEL_REF` and `DEPOT_PR_SENTINEL_ID`
remain absent. No Depot settings or runner groups were changed; the bounded
repository variables were temporarily armed and then removed. Read-only Cache
Explorer evidence verifies that both randomized non-secret entries are still
present: seed `mesh-llm-depot-authority-seed-v1-4089d6c8551820aa86b7780157729786`
(303 B) and poison
`mesh-llm-depot-authority-pr-v1-4089d6c8551820aa86b7780157729786-pr-1327`
(308 B). Purge or provider-expiry confirmation remains pending; this
docs-only change must not delete them.

The exact-SHA five-lane candidate, provider-separated comparison, and
identical-SHA hosted rollback are recorded in
`.omo/specs/depot-pr-rollout-evidence.md`. Quality and Linux had favorable
queue observations but remain unclassified because execution was
cache-confounded; Website had insufficient samples, and macOS/Windows hit the
deterministic capacity rollback threshold. The fork PR canary and namespace
purge/expiry confirmation remain pending. Fork validation must stay hosted and
remains the no-Depot-authority half of the acceptance evidence.

Disabling automatic Depot Cache and Registry Actions connectivity removes the
direct Depot build-tool/registry credential path that blocked the first canary
(including automatic `DEPOT_CACHE_TOKEN`/WebDAV preconfiguration), but it does
not disable or isolate the GitHub Actions cache proxy/runtime-token path. The
sentinel used that path and proved repository-scoped cross-trust authority, so
the switch state does not override the observed result or prove provider
parity, branch/main cache behavior, or fork isolation. Those remaining checks
and the isolation redesign are prerequisites for PR execution on Depot.

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

### Required provider isolation contract

Before permanent PR activation or any context outside the reviewed temporary
exception, Depot must expose a supported, server-enforced
per-connection/job/ref control for the GitHub Actions cache path. The control
must choose one of these safe outcomes for an untrusted PR:

- leave the job on GitHub-native branch-scoped `ACTIONS_CACHE_URL`,
  `ACTIONS_RESULTS_URL` and runtime-token semantics, with no Depot cache proxy
  or direct Depot cache token; or
- issue a PR-isolated cache namespace and token whose ACL permits reads and
  writes only within that PR, denying reads and writes from trusted
  `main`/release jobs and every other PR namespace, again without exposing a
  direct `DEPOT_CACHE_TOKEN`.

The contract must cover `actions/cache`, setup-* cache consumers and any other
GitHub Actions cache API caller, including the provider's
`ACTIONS_RUNTIME_TOKEN` injection. Cache-key prefixes, a loopback proxy,
ephemeral runners, the organization switch above, and runner-group workflow
restrictions are not substitutes for this server-enforced control. The
provider must document the control and its trust/ref semantics, and a fresh
same-repository PR, fork PR and trusted-main seed/verify sentinel must prove
the selected outcome before the exception is made permanent or a canary is
treated as isolation evidence.

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
without direct Depot build-tool/WebDAV or registry preconfiguration. It does
not establish a provider-side disable or per-PR ACL for the GitHub Actions
cache proxy/runtime-token path; the authority sentinel is the required proof
for that path and currently records unsafe repository-scoped access.

The controlled probe above answers the first question for the current runner:
the same-repository PR path exposed repository-scoped authority. The remaining
runtime questions are:

1. What provider-isolation redesign prevents that observed authority from
   crossing trust boundaries while preserving the same cache probe contract?
2. Do hosted release and cache-warmer jobs retain their intended GitHub cache
   behavior while trusted Depot selections remain cache-inert, and have all
   existing Depot namespace entries been purged or expired?
3. Can a redesigned same-repository PR path and a fork PR both run the
   protected canary with no Depot cache/registry authority and no entry that
   trusted main later restores?
4. Does provider parity hold for the same checked plan, commands, artifacts,
   and required results on GitHub and Depot?
5. Does the restricted runner group continue to allow only the exact protected
   workflow refs after the PR canary is enabled?

The selector emits provider-derived `allow_native_github_cache` and
`allow_depot_remote_cache` outputs. Depot remote build-tool cache remains
disabled. Outside the bounded exception, Depot selections keep native Actions
cache consumers disabled. During the exception, eligible same-repository PR
and trusted-main Depot jobs emit `allow_native_github_cache=true`,
deliberately exercising Depot's repository-wide Actions-cache proxy for
cross-branch reuse. This improves iteration speed but does not establish an
authority boundary. The canary must still target the actual
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

Permanent activation still requires every acceptance canary. The temporary
exception requires `DEPOT_PR_RUNNERS_ENABLED=true` and expires in checked-in
policy on 2026-09-14 UTC. Roll back immediately by deleting
`DEPOT_PR_CANARY_REF` and deleting or setting
`DEPOT_PR_RUNNERS_ENABLED=false`, then rerun the identical PR plan on hosted
placement; this changes provider placement only and must leave commands,
matrices, artifacts, and required checks unchanged. `DEPOT_RUNNERS_ENABLED`
remains the separate trusted-main placement gate.

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

The first checked-in half of this protocol is the manual `depot-canary.yml`
sentinel. On protected `main`, an operator may dispatch `mode=seed` with a
fresh lowercase 32-hex `sentinel_id` and canonical positive `pr_number`. The
workflow derives the only accepted keys internally:
`mesh-llm-depot-authority-seed-v1-<id>` and
`mesh-llm-depot-authority-pr-v1-<id>-pr-<number>`. It writes a deterministic,
non-secret marker at the fixed relative `.depot-authority-sentinel` path. Before
any cache action, all three cache phases attest that the provider-injected
`ACTIONS_CACHE_URL` and `ACTIONS_RESULTS_URL` are present, value-free
structurally validated HTTP endpoints with a non-GitHub/non-loopback authority
(including all IPv4 `127/8` and IPv4-mapped IPv6 loopback spellings), numeric
port and explicit path. The shell attestation intentionally does not require
ambient `ACTIONS_RUNTIME_TOKEN`: GitHub's `NodeScriptActionHandler` injects
that credential into the pinned Actions-cache restore/save actions, while the
shell `ScriptHandler` does not. Successful full restore/save calls provide the
credential/token proof.
Bracketed authorities use the fixed runner's Python 3.8+ stdlib `ipaddress`
classifier for every IPv6 spelling; missing, too-old or invalid parser state
fails closed with only a `parser` reason class.
The pinned Actions-cache save/restore actions then exercise that attested
remote backend. Seed and verify inputs are bound to the configured sentinel ID
and exact merge ref before cache access. Seed clears its local marker, performs
a full restore and validates the exact marker content; verify performs a full
restore and validates exact poison content before its expected-miss check. The
protected PR probe uses that same path and exact marker grammar, then clears
and fully restores its saved poison key, requiring a cache hit and exact bytes
before the trusted-seed gate; this same-job restore/save proves the PR Node
token/write path, while main verify's poison miss remains the cross-scope proof.
Cache-version metadata cannot vary across jobs. After the protected PR probe
publishes the bounded poison key, dispatch `mode=verify-pr-write` with the same
validated inputs; a hit fails because trusted main saw a PR publication. The existing
default `mode=audit` remains the negative resource/credential audit across the
full Depot matrix and is unchanged.

These diagnostic keys are deliberately outside normal build/cache policy and
must be purged or allowed to expire after each experiment.

The bounded protected PR half is now checked in as an additive diagnostic job
inside `ci-quality-slice.yml`. It has its own central-selector input variables:
`DEPOT_PR_SENTINEL_REF` selects one exact same-repository
`refs/pull/<number>/merge` ref, and `DEPOT_PR_SENTINEL_ID` supplies the same
validated 32-lowercase-hex identity used by the manual seed. The ordinary
quality runner policy continues to read `DEPOT_PR_CANARY_REF`, so enabling the
sentinel selector cannot move the normal Quality jobs or any other build row to
Depot. The diagnostic job is outside the planner's required slice graph: it
does not add a plan row, producer, consumer, artifact, matrix, or `needs` edge
to the existing build jobs, and the protected Quality lane still completes
through its existing `quality` and `runner_contract` summary dependencies.

The sentinel job runs only when the actual event is `pull_request`, the
protected lane receives `original_event_name=pull_request`, the central
selector selects Depot for the exact configured merge ref, and the event's
head repository is this repository. Fork heads, `pull_request_target`,
dispatches, planner `force_hosted`, a missing selector variable, or a
non-matching ref remain hosted/no-Depot; a malformed selector ref is rejected
by the central selector before any runner is selected. A global
`DEPOT_PR_RUNNERS_ENABLED=true` value alone does not run the diagnostic job.

This exception deliberately does not invoke `audit-depot-pr-isolation`: that
audit is designed to reject non-GitHub/loopback endpoints before a build, while
this no-checkout job must exercise the actual cache restore/save authority and
record the result without executing PR-controlled code. The job has
`permissions: {}`, performs no checkout, receives no secrets, and uses only the
fixed `.depot-authority-sentinel` path. Before the pinned cache restore it
attests the provider-injected remote backend using the same structural
non-GitHub/non-loopback HTTP contract (including all IPv4 `127/8` and
IPv4-mapped IPv6 loopback spellings); no endpoint, host, path, port or token
value is printed. The shell attestation does not require ambient
`ACTIONS_RUNTIME_TOKEN`; the pinned Actions-cache restore/save actions are Node
actions, whose `NodeScriptActionHandler` receives that credential while the
shell `ScriptHandler` does not. Successful full restore/save calls prove the
credential/token path. It validates the
repository variable and the actual `github.event.pull_request.number`,
restores the trusted seed (not lookup-only), validates exact seed marker
content on a hit, replaces the path with a deterministic non-secret poison
marker, saves the exact Stage 1 poison key, clears and fully restores that key,
requires a cache hit and exact marker bytes, then fails only if the trusted seed
was readable. A seed miss passes with a pending trusted-main
`verify-pr-write` check; a seed hit is reported only after the poison
restore/save proof has completed.

Fork PRs remain on the hosted path and provide the no-Depot-authority half of
the acceptance evidence; only the same-repository exact sentinel ref can
exercise this protected Depot authority probe. This hook remains inert until
`DEPOT_PR_SENTINEL_REF` and `DEPOT_PR_SENTINEL_ID` are deliberately set for an
operator-run experiment. It is not the global PR activation gate, does not
grant cache or registry authority, and does not authorize setting
`DEPOT_PR_RUNNERS_ENABLED`.

## Permanent activation gate

Do not remove the bounded expiry or make Depot PR execution permanent until:

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

Permanent Depot PR activation must be a separate change after the composable CI
graph is complete. It must not be combined with routing, required-check,
artifact or branch-protection migration.

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
