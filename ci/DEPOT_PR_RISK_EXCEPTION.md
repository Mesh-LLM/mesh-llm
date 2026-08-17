# Depot PR cache-risk exception

Status: **accepted, bounded exception**

- Decision date: 2026-08-14
- Automatic expiry: 2026-09-14 00:00 UTC
- Scope: same-repository `pull_request` merge refs only
- Objective: reduce CI iteration time while Depot and GitHub finish stronger
  cache-authority controls

## Decision

MeshLLM knowingly accepts repository-wide, cross-branch Depot Actions-cache
authority for individually approved same-repository pull requests during this
exception window. This is a deliberate speed-versus-isolation decision, not a
claim that Depot currently provides branch or pull-request cache isolation.

The exception does not apply to forks, `pull_request_target`, releases,
publishing, deployment, credential-bearing smoke jobs, CI-policy changes, or
jobs outside the existing eligible runner-selection set.

## Findings

Depot documents that GitHub Actions cache API consumers on Depot runners are
transparently routed to Depot Cache and that entries are repository-scoped,
without branch isolation. That includes `actions/cache` and setup actions that
use the GitHub cache API. Depot's ephemeral compute boundary does not provide a
separate cache authority boundary. See [Depot's GitHub Actions cache
integration](https://depot.dev/docs/cache/integrations/github-actions) and
[Depot runner overview](https://depot.dev/docs/github-actions/overview).

GitHub's native dependency cache has ref-based access rules, including the
synthetic merge ref used for pull requests. Those rules describe GitHub's cache
service, not Depot's repository-wide proxy. See [GitHub dependency-cache
reference](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching).

GitHub is developing finer cache access modes, but the public proposal and
implementation work do not constitute a supported Depot server-side ACL. See
[GitHub Community discussion 194493](https://github.com/orgs/community/discussions/194493),
[actions/runner PR 4538](https://github.com/actions/runner/pull/4538), and
[actions/cache PR 1781](https://github.com/actions/cache/pull/1781).

Depot has a useful untrusted-fork isolation design for remote Docker/BuildKit
builds: an OIDC flow sends fork builds to ephemeral builders without the main
cache. That product path does not cover arbitrary jobs on managed
`runs-on: depot-*` runners or their transparent Actions-cache proxy. See
[Depot's OSS fork-build design](https://depot.dev/blog/github-actions-oss-fork-builds).

The controlled MeshLLM sentinel proved the current cross-trust behavior:

- trusted seed run `31816775585` saved a marker;
- same-repository PR run `31816869128`, job `94821057215`, restored that trusted
  marker and saved then restored its own poison marker;
- trusted-main verify run `31817111471`, job `94821343605`, restored the PR
  poison marker.

The result is evidence of shared authority, not a successful isolation test.

## Risks we are accepting

An approved same-repository PR can:

- read non-secret cache data written by trusted main or another PR;
- publish a cache entry that a later trusted-main job may restore;
- publish entries visible to other PR namespaces;
- corrupt or crowd the repository cache namespace, causing failures, cache
  misses, resource consumption, or slower builds;
- exploit a future cache consumer that incorrectly treats restored files as a
  correctness or trust boundary.

Cache entries can outlive the PR and the approval that created them. Cache keys,
restore prefixes, ephemeral runners, and maintainer review reduce likelihood or
impact but do not create a security boundary.

No secret, signing key, release credential, registry credential, or privileged
deployment input may be stored in or derived from this cache. Artifacts with
explicit validation remain the correctness boundary.

## Approval and compensating controls

The repository's GitHub setting currently reports
`approval_policy=all_external_contributors`. GitHub's approval button therefore
protects workflows submitted by external contributors and forks; it does not
gate a branch pushed inside `Mesh-LLM/mesh-llm` by a collaborator. We do not
claim that setting as the same-repository approval control.

The protected runner policy implements an explicit equivalent for this
exception. Depot is selected only when all of the following are true:

1. `DEPOT_PR_RUNNERS_ENABLED` is exactly `true`.
2. `DEPOT_PR_APPROVED_REF` exactly equals the current
   `refs/pull/<number>/merge` ref.
3. `DEPOT_PR_APPROVED_SHA` exactly equals the lowercase 40-character PR head
   SHA.
4. The head and base repositories are both exactly `Mesh-LLM/mesh-llm`.
5. The protected default-branch reusable workflow owns runner selection.
6. The planner has not forced hosted execution for CI/workflow/policy changes.
7. The checked-in expiry has not been reached.

A new push changes the head SHA and automatically returns the PR to
GitHub-hosted runners until a maintainer reviews and updates the approval SHA.
Forks always remain GitHub-hosted. The audit still rejects direct Depot cache
tokens, WebDAV/sccache authority, registry credentials, Docker registry auth,
and URL userinfo before checkout.

## Cross-branch cache behavior

While the exception is active, the central policy enables the GitHub Actions
cache API for the exact approved Depot PR and for eligible trusted-main Depot
jobs. Depot maps those requests to its repository-wide namespace, allowing the
intended cross-branch reuse. The separate direct Depot build-tool remote-cache
flag remains disabled.

The shared namespace must be treated as attacker-controlled input. Cache
restore may improve speed, but a cache hit never proves provenance or
correctness.

## Operation and rollback

To approve one reviewed PR revision, a maintainer with repository-variable
authority sets the global gate to `true`, the approved merge ref, and the exact
head SHA. The approved SHA must be refreshed after every push. Remove the two
approval values after the run or when the PR closes.

Immediate rollback is any one of:

- delete or set `DEPOT_PR_RUNNERS_ENABLED=false`;
- delete `DEPOT_PR_APPROVED_REF`;
- delete `DEPOT_PR_APPROVED_SHA`.

The same workflow graph then selects GitHub-hosted runners. No source, matrix,
artifact, command, or required-summary change is needed.

The selector fails hosted at the start of 2026-09-14 UTC. Extending the window
requires a reviewed source change to the expiry and this decision record; a
repository variable cannot extend it.

## Validation record

The exact-SHA five-lane Depot canary, provider-separated schema-v3 metrics,
macOS toolchain fallback proof, and identical-SHA hosted rollback are recorded
in `.omo/specs/depot-pr-rollout-evidence.md`. That record preserves the strict
capacity classifications and the Windows serial-matrix timing caveat. A green
functional canary or rollback does not supersede the unsafe cache-authority
sentinel above and does not relax this exception's scope or expiry.

## Exit criteria

Remove this exception when Depot or GitHub offers and verifies either:

- GitHub-native branch/ref-scoped Actions-cache authority on the PR runner with
  no Depot proxy or alternate Depot cache credential; or
- a server-enforced namespace/token ACL scoped exclusively to the current PR,
  denying reads and writes from trusted main, release, and every other PR.

Before treating either control as an isolation boundary, rerun the trusted
seed, same-repository PR, fork, PR-write/self-restore, and trusted-main verify
sentinel. The expected result is no trusted read by the PR and no PR write
visible to trusted main or another PR.
