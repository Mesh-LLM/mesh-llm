# CI governance: human review and activation

The optional `cargo xtool ci-governance inspect-live-controls` command accepts
an existing read-only credential through `MESH_CI_GOVERNANCE_AUDIT_TOKEN` and
uses a fixed, bounded set of GitHub GET requests. It never mints a credential
or retries a denied request. A 403 is unknown; a 404 is ambiguous between
absence and inaccessible state. The output is deliberately partial and
unverified: numeric member identity, effective inherited rules, path-scoped requirements,
required-check sources, bypasses, environment review/branch policy, and token
permissions remain unknown. It does not authenticate its observer, establish
an atomic snapshot, report drift clearance, deliver notifications, or grant
execution or activation authority. Treat observed values as leads for the
independent human's complete readback, not as substitutes for it. Do not put
credentials or raw provider error bodies into records.

The inspector lists reviewer-team members through a fixed, paginated REST GET,
including child-team members as described by [GitHub's API](https://docs.github.com/en/rest/teams/members#list-team-members).
It reads through an empty terminal page twice and rechecks the team identity;
denial, malformed pages, duplicates, changes between samples or a pagination
limit leave the membership observation unknown. A stable login/type-set
comparison uses the proposed three-person roster, but does not pin numeric
human identities, prove continuously stable membership or authenticate the
credential. The `partial_drift_indicators` field names only observed mismatches
in a few directly comparable fields; it never turns an incomplete inspection
into a match or an authorization. In particular, legacy branch-protection
review counts cannot establish the effective file-scoped ruleset. The command
prints a local report; it does not schedule or deliver a drift notification.

The normative rule is in the [manage-ci skill](../.agents/skills/manage-ci/SKILL.md#independent-human-approval-for-ci-policy).
This procedure implements issue [#2781](https://github.com/Mesh-LLM/mesh-llm/issues/2781).
It is a local proposal until an eligible independent human reviews the exact
revision, authorized humans complete the pre-merge team bootstrap, it is merged,
and administrators activate and read back the external rules. No team,
protection, environment or administration
credential is created by this document or by local validator success.

## Review responsibility

Exactly these humans are eligible: Mic (`micspiral`), Nick (`ndizazzo`), and
James (`i386`). One of them, independent of both the author and responsible
sponsor, must approve the current revision before merge or live operation.
The proposed visible team is `@Mesh-LLM/ci-policy-reviewers`, containing exactly
those three accounts, with explicit repository write permission. Team
existence, visibility, permission and membership remain activation evidence
to obtain, not facts established by a CODEOWNERS entry.

For the initial implementation the authenticated human PR author is the
responsible sponsor. A bot-authored CI-policy PR is unsupported and fails
closed; its sponsoring human must submit a human-authored PR. Agents may write
the commits and prepare the evidence. They may not supply an approval. Body
text, labels, trailers and an agent's claim about a sponsor are not identity
evidence. Native author exclusion alone cannot identify a bot's sponsor.

Evaluate the latest effective review for each reviewer. A missing, unlisted,
self, sponsor, dismissed or stale approval fails; an older approval cannot
outvote a later changes-requested or dismissed review. Require the exact head
SHA, and rerun admission after every new commit or review change. The initial
governance change is subject to this same rule even before its automation is
available. An eligible independent person must inspect the bootstrap diff and
record their review; the agent implementing it cannot approve bootstrap.

Before merging any CODEOWNERS references to the proposed team, an authorized
human must obtain independent eligible approval for creating/verifying the
visible exact-three team and its explicit repository write access. Perform the
approved bounded operation and retain actual team ID, complete membership,
visibility, repository permission and GitHub valid-owner readbacks. A reviewer
must confirm those references resolve to an eligible owner before merge.
If direct precreation is unavailable, use a separately reviewed coordinated
valid-owner transition; do not merge a team-only suffix that names an absent,
invisible or non-write team. This manual bootstrap needs its own approval and
audit evidence even when the final-state local record validator cannot admit
the intermediate stage. Native review rules remain inactive until separate
authorized activation and effective-state readback. The proposed team itself
also remains unverified/inactive until its actual live readback is retained.

## What receives the additional review

CI control paths cover workflows and local actions, routing and test selection,
required checks, provider and runner sizing, concurrency and cancellation,
caches and artifacts, exception expiries, variables, images and runner groups,
permissions and credential access, release/deployment gates, and the policy,
roster, CODEOWNERS, validators and administration path themselves. Conservative
shared infrastructure ownership is intentional: changing a common build or
automation entrypoint can change every CI consumer.

The final CI CODEOWNERS rules must follow every earlier rule, including the
Windows entries, and name only the reviewer team. GitHub uses the last matching
pattern; multiple owners on that line allow any one of them to approve.
Validate actual matches using the complete file, including added lines.
Changes, deletions and rename source/destination paths all count. Protect a
newly introduced CI-policy owner before making it authoritative. Normal product
source, UI, SDK and product documentation outside the protected control paths
retain their existing product ownership and do not require this extra team
approval. A mixed PR receives CI review for its CI change.

The machine scope is [governance/policy.json](governance/policy.json), separate
from the build-selection catalog `ci/ownership.yml`. Its conservative shared
roots include `tools/`, `scripts/`, `just/`, Cargo/configuration and product
build/deployment script roots, alongside CI and governance files. The final
CODEOWNERS block mirrors that exact scope. New control paths require reviewed
scope updates; existing product source directories are not blanket CI scope.
Native source and patch-only changes under `skippy/llama_cpp/` retain normal
product ownership. The owning `skippy/scripts/` build and automation controls
remain in scope.
UI and website package/check-selection manifests, the exact UI lint/type/test
configuration files listed in the policy, and their `scripts/` roots also
receive CI review. Ordinary component `src/`, e2e/test content, documentation
and native patches retain product ownership; changing which checks execute or
their configuration receives the additional review.

Required-team review must be file scoped. Do not globally require the CI team
for every PR, and do not describe a repository-wide required status check as a
sixth product-validation lane: metadata admission may report not applicable
for product-only PRs while the five required validation lanes remain intact.

An active required custom approval check needs a distinct restricted GitHub
App identity bound in required-status configuration, or a verified equivalent
protected-workflow mechanism. The shared GitHub Actions app is insufficient:
same-repository PR workflows with `checks: write` can emit the same check name.
Keep the distinct publisher credential outside ordinary repository secrets and
PR execution, and verify its installation, permissions and protected custody
before activation. The offline snapshot evaluator neither authenticates its
input's origin nor publishes a trusted check. The prepared live collector uses
fixed GitHub endpoints and bounded complete file/review/member pagination,
checks numeric identity anchors and exact current-main policy, and repeats
head/main/membership/review reads before publication. Old policy that is merely
an ancestor of `main` is not current policy authority. These adapters remain
inactive until external activation and credential custody are verified.
The PR snapshot must bind `current_protected_sha`, `base_sha` and
`policy_revision` to the same fresh trusted `main` readback. Internally
consistent historical snapshots are insufficient. Those local fields remain
assertions until the trusted collector authenticates them.

## Expected state and observed state

Keep the approved expected non-secret configuration in version control. A
dated live snapshot is a separate observation, with repository/provider IDs,
operator, UTC observation time, source API/UI, and evidence links. An unknown,
403, missing permission or unsupported API result means unverified. Never
replace unknown observations with the desired value. Never retain secret
values; retain only credential names, scope, permissions and custody facts.

| Surface | Expected target | Observed evidence required before activation |
| --- | --- | --- |
| Reviewer team | Visible `ci-policy-reviewers`; exactly `micspiral`, `ndizazzo`, `i386`; explicit repository write | Team ID/slug, visibility, complete paginated membership and repository permission readback |
| CI merge review | One required team review on declared CI paths; PR required; stale reviews dismissed; author excluded | Effective branch/ruleset JSON, rule IDs, conditions, file patterns and review settings |
| Routine bypass | None for CI governance, including administrators and automation | Effective bypass actor IDs, modes, inherited rules and administrator applicability |
| Ownership | Final team-only CI matches; product ownership preserved | GitHub CODEOWNERS validity and representative last-match results from the exact protected revision |
| Approval admission | Protected-base code/policy; human author is sponsor; current head review | Required check identity/source, exact base/head, authenticated metadata, latest reviews and result |
| Administration | Protected `main` only; reviewed bounded record; live-state precondition | Workflow revision/ref restrictions, allowed operations and test/refusal evidence |
| Administration environment | One eligible independent reviewer; self-review prevented; admin bypass disabled; only protected `main` deployments | Environment ID, reviewer identities, protection flags, branch policy and actual approval/deployment history |
| Credentials | No repository/organization/provider admin credential in agent or PR execution | Human-reviewed credential permissions, installation/repository scope, short lifetime and environment custody; names only |
| Provider/runner/cache/variable/release settings | Approved exact values, enabled/disabled state, IDs, refs, image digests, expiry and access policy | Fresh API readback or authorized manual evidence for each changed setting |
| Drift reporting | Declared detection/reporting target, owner and report destination | Scheduled execution, permissions, observation timestamps, delivered report and measured latency |

The issue's 2026-11-01 snapshot of main ruleset `20090642` recorded zero
required approvals, code-owner review disabled and four always-bypass users.
That is historical evidence only. Re-read effective rules, branch protection,
organization rules and bypass actors before planning activation. Do not carry
that observation forward as current state or policy authorization.

## Local validation interfaces

The following four commands under `tools/xtask/src/ci_governance/` run through
the existing repository Cargo alias and consume local JSON data;
they do not fetch/authenticate GitHub state, create a team, publish an approval
check, mutate settings or deliver a drift notification.

```sh
cargo xtool ci-governance validate-ownership \
  --policy ci/governance/policy.json --codeowners .github/CODEOWNERS
cargo xtool ci-governance validate-pr \
  --policy ci/governance/policy.json --snapshot /trusted/pr-snapshot.json
cargo xtool ci-governance check-drift \
  --policy ci/governance/policy.json \
  --expected ci/governance/expected-state.json --observed /trusted/observation.json
cargo xtool ci-governance validate-change \
  --policy ci/governance/policy.json \
  --expected ci/governance/expected-state.json \
  --record /trusted/change-record.json --observed /trusted/observation.json
```

[expected-state.json](governance/expected-state.json) is the proposed logical
non-secret target, not raw GitHub API configuration or a current readback.
The [observation template](governance/examples/observation.template.json) and
[change-record template](governance/examples/change-record.template.json) show
the closed versioned local data shapes. The separate
[manual-readback template](governance/examples/manual-readback.template.json)
supplies the value for `manual_readback` after a human operation. They are
intentionally invalid until
fully populated: `<...>` placeholders, incomplete objects, timestamps and account
IDs must be replaced with real normalized evidence. They contain no fabricated
live snapshots or IDs. Copied claims and approval URLs do not become
authenticated evidence.
Trusted collection must supply fresh observations and protected observer/execution
contexts; callers that can forge all offline inputs can also forge local
admission. A checked-in main file is not a live observation. Local record checks
neither supply approval nor authorize or execute a change.

The digest interfaces are distinct. The PR snapshot's `policy_sha256` hashes
the raw supplied protected-base policy file bytes. Operations hashes for the
policy, before state and complete request use SHA-256 of compact typed JSON
with object keys recursively sorted lexicographically. Operations normalize
known unordered state lists (team members, bypass actors, scope lists,
environment reviewers/refs and observer permissions) and reject duplicates;
request hashes normalize their before/desired state lists. The operations policy
digest preserves policy array order. These digests do not hash pretty-printed
input file bytes. PR metadata must be complete
and no older than 300 seconds, with no future timestamp; operations observations
must be complete and no older than the declared drift target (900 seconds in
the initial expected state). These local freshness checks do not authenticate
the source or prove that an observation was collected live.

Successful PR validation reports
`evidence_status: "local-snapshot-validation-only"`. Operations return
`VALIDATED_PLAN`, `RECORDED_MANUAL_READBACK` or drift `MATCH` with
`execution_authorized: false` and `evidence_authenticated: false`. A matching
drift snapshot also reports `live_detection_latency_measured: false`; it does
not establish a deployed monitor or measured reporting latency.

These operations interfaces admit only live/current-time evidence. Completed
receipts stop validating after the observation or approval expires; no archived
receipt validator is supplied. Read-only drift snapshots require a separate
`observer` context binding repository, run/attempt, `refs/heads/main`, current
protected revision and an identified human or bot service principal. They may
set `execution: null`, allowing a future scheduled read-only observer without
inventing a human operator. `validate-change` instead requires the full human
`execution` context (operator, triggerer and rerun actor). Neither input shape
creates a deployed schedule or authenticates the observer. Change admission
requires reconciliation to the full final target and an already protected
administration environment, so it does not validate staged bootstrap records.
Use independently approved human bootstrap records for intermediate steps and
retain them as audit evidence without claiming local final-target admission.
The logical observer name and `separate-github-app` kind do not distinguish a
replacement dedicated App: exact App/installation/principal IDs and required
status source binding remain activation/readback evidence to obtain.

The first bounded operation set covers enumerated GitHub governance
reconciliation only. It does not admit arbitrary provider/runner APIs,
variables, secrets, command text, endpoints or file paths. Those broader policy
changes still require the normative independent review and authorized manual
procedure; extend the typed operation set with review before automating them.

## Prepared live metadata and administration paths

[activation.json](governance/activation.json) is deliberately `active: false`,
with null numeric identities and no native ruleset IDs. Live adapters refuse
that state. Only independently approved human activation may populate actual
repository, team, human, environment, App, installation and ruleset IDs after
live verification. `activation-status` reports source state, not verified
external protections.
The publisher reads only the immutable tracked activation scalar before
compiling, avoiding disabled-workflow build cost. This is a cost hint, not
authorization; typed Rust validation of protected activation remains mandatory.

`collect-pr <positive PR number>` accepts one canonical number and collects
current authenticated PR metadata with a separately minted read-only observer
token; its audit output retains exact protected policy text/hash and the
snapshot with `native_enforcement_verified: false`. It publishes no check or
settings change. `publish-pr <positive PR
number>` also accepts one canonical number, collects and
evaluates current metadata internally, and publishes the fixed
`CI policy / independent approval` check through the anchored App. It accepts
no caller verdict, snapshot, check name, head SHA or check ID. It creates an
in-progress check before approval evaluation, repeats complete collection for
success, and verifies created/completed check App/head/name/identity readbacks.
Collection/approval denial after creating that check normally concludes failure.
Errors before creation, missed events or final publication/readback failures can
leave an older same-head green or a pending check. Native scoped review plus
outage/reconciliation procedures remain necessary; the custom check alone does
not close those cases. App-bound check readback does not prove installation
identity; that is checked during token minting.

`ci-governance-publisher.yml` handles protected metadata-only
`pull_request_target` lifecycle events and completed runs of
`ci-governance-review-signal.yml`, plus main-push and scheduled open-PR
reconciliation. Reconciliation requires two complete matching lists of open
main-target PRs, with anchored repository/current-main checks; truncation,
movement or more than 200 PRs rejects the whole list. The 200 limit bounds
parsing and the workflow matrix; it is not qualified API or publishing capacity.
Ordinary PR, review and main-push updates remain event driven. A 15-minute cron
provides outage reconciliation; its cadence proves neither drift/report
delivery nor actual end-to-end latency.
The review signal has empty permissions, no checkout
and no content execution; it signals review submission/edit/dismissal. The
protected receiver authenticates the signal's workflow/run/PR metadata and
recollects current state rather than trusting its output. These controls remain
separate from the five product-validation lanes. The resolver derives numeric
per-PR serialization keys from authenticated metadata. `publish-event
--expected-pr N` rechecks that association against the resolver's locked PR
before any check write; the numeric input supplies no verdict, head or policy.
Credential-bearing
publisher/inspection consumers restore the exact same-run protected automation
artifact by source, artifact ID and executable digest; they execute no PR code.
Matrix jobs serialize by authenticated repository/PR, with at most four in
parallel per workflow matrix; this is not a global cap across workflow runs.
API quotas, replacement of pending concurrency jobs, queue backlog and actual
end-to-end reconciliation latency require measurement before activation.

The publisher and administration preparation use one dedicated App via the
step-scoped `CI_GOVERNANCE_APP_PRIVATE_KEY`, generating an ephemeral RS256 JWT
with protected temporary key handling. The CLI can instead take one explicit
ephemeral `CI_GOVERNANCE_APP_JWT`; it refuses both credential sources together
and does not fall back to a developer's stored GitHub identity. Provision the
key only in verified
main-only `ci-governance-publisher` and `ci-governance-administration`
environments, never ordinary repository secrets available to PR jobs. Minted
observer/read and publisher/check-write tokens are separately scoped to the
anchored repository and checked for exact permissions and expiry. The parent
key/JWT retains the App's checks-write and token-minting capabilities, potentially
for other installations; child-token readback does not prove the parent key is
globally repository restricted. App installation selection, permissions and
custody remain external activation evidence. No administration-write credential
is provided.

`ci-governance-administration.yml` dispatches only on exact current `main` and
accepts only `record_id` plus `record_sha256`. `prepare-administration` resolves
the fixed reviewed intent and authenticates repository/main/run/actor metadata,
then emits a concrete `PREPARED_PENDING_AUTHORIZATION` packet. An independently
environment-gated job runs `inspect-administration-approval` with the same
selectors; it reports `AUTHORIZATION_BLOCKED` and fails. Public environment GET
does not expose administrator bypass, so readback retains `administrator_bypass:
null` and `configuration_verified: false`. Available history also does not prove
exact-attempt/packet approval, a post-approval full live-state/replay precondition
or protected runtime attestation. This path releases no execution credential and
performs no administration mutation. Those missing proofs require separate
verified human procedures; a packet or environment name supplies no authority.

## Bounded change record

Prepare the record and diff before requesting the independent review. Use a
versioned, closed schema: reject unknown fields, duplicate identities,
unbounded strings/lists and unrecognized setting keys. Operations select fixed
typed controls and permitted values; no field is accepted as executable code
or an arbitrary API endpoint. Freeform rationale, job, test, rollout and rollback
prose receives length/shape checks only. Operators must keep secrets and command
payloads out of those strings; the validator cannot reliably identify them.
The reviewed record identifies:

- Human owner and sponsor; immutable repository revision, policy revision,
  record ID/digest and PR or change-request link.
- Rationale, affected jobs/consumers, exact before and proposed after values,
  expiry where applicable, and targeted tests with evidence links.
- Independent eligible approver identity, authenticated review ID/link and
  approved revision/digest; rollout stages, stop conditions and rollback plan.
- Authorized operator identity, UTC start/end, run/attempt and environment
  approval/deployment links, provider/resource IDs and actual before/after
  observation times and values.
- Outcome, readback/evidence links, detected drift and rollback outcome.

Use the [static intent template](governance/examples/intent.template.json) for
precommittable review. It binds the human owner/author/sponsor, policy and
desired/before-state digests, bounded controls and rationale, with a maximum
seven-day window. Store the reviewed intent only as a regular Git blob at
`ci/governance/records/<UUID>.json`; select it by canonical UUID and raw file
SHA-256. It omits its own commit SHA and future run identity, avoiding a
self-reference. Preparation resolves that exact blob and protected policy from
the authenticated current-main revision, refuses substituted/symlinked inputs,
and adds an expiring runtime envelope bound to the actual run, attempt and
actors. The prepared packet lasts at most one hour and supplies no authority.

Approval identity/link is an audit field, not proof by itself: fetch the
authenticated review and verify eligibility, independence and current revision.
Keep the approved intent immutable. Append execution observations as a
separate receipt, bound to its digest, rather than editing an approved record
to fit what happened. An operation with a changed target, value, source,
record digest or precondition needs a revised record and fresh approval.

The initial machine contract additionally binds the request's canonical typed
SHA-256, exact protected policy revision/digest, before-state canonical digest,
complete desired state, operation UUID, run/attempt/ref and actor context.
Its eligible independent approval binds the whole request digest, the same
policy revision/run and an expiry within one hour. A fresh complete observation
(at most 900 seconds old) carries the consumed-operation ledger and audit link;
reused operation UUIDs fail admission. A manual completion needs a matching
consumed-operation receipt and actual after-state readback. These fields are
claims until a trusted human/collector establishes their provenance. Retain the
approved request and subsequent receipt separately in the audit evidence.
The observation's `current_protected_main_revision` must equal its frozen
`protected_policy_revision`. Unix-second timestamps must be strictly ordered:
request < independent approval < before observation < manual operation < after
readback. Equal-second timestamps are conservatively refused; capture a fresh
readback in a later second. An older before snapshot does not satisfy the
post-approval precondition.

## Safe administration procedure

The checked-in workflow prepares and inspects records while refusing unsupported
authorization. It does not establish deployed protection or execute the steps
below. An authorized human follows this bounded procedure after obtaining the
missing independent approval and live control evidence; agents prepare inputs
and evidence without receiving administration credentials.

1. Freeze the reviewed record and immutable protected `main` source revision.
   Inspect effective GitHub/provider state and produce a non-secret before
   snapshot. Validate the record's allowed setting/value and compare that
   snapshot with the reviewed live-state precondition. Unreadable or different
   state stops the operation for review; do not silently overwrite drift.
2. Obtain one eligible independent current-revision human approval. The
   approver cannot be author or sponsor. Verify team membership/permission and
   the latest review immediately before execution. Retain the review link.
3. Use a dispatcher/controller defined and executed only on protected `main`,
   with a fixed operation enum and record ID/digest inputs. Resolve any approved
   record only from its immutable protected revision; never accept arbitrary
   shell, branch, workflow, script, API path, credential name or PR executable.
   Serialize changes to the same resource without cancelling an in-flight
   mutation, and impose explicit operation/time bounds.
4. Require a protected administration environment with one eligible independent
   human reviewer, self-review prevention, administrator bypass disabled and
   deployment branches limited to protected `main`. The native environment's
   initiator exclusion does not necessarily exclude the change's author or
   sponsor; independently check all three identities before releasing authority.
   Freshly verify the expected environment controls themselves. A missing
   control, membership mismatch or unverifiable approval stops execution.
5. Release only a short-lived credential scoped to the approved resource and
   setting operations after admission and environment approval. Keep it out of
   agent sessions, PR jobs, candidate-source processes, artifacts and caches.
   A credential capable of changing its own rules/environment is broader than
   the executor needs; isolate that human bootstrap authority and audit its use.
6. Re-read actual state after waiting for approval and immediately before the
   mutation. Require the reviewed precondition again, including policy/team
   state. Where supported use a provider version/ETag conditional update. Apply
   only the bounded approved change. If no atomic provider precondition exists,
   record that limitation and minimize the read/write interval; do not claim
   atomicity from two sequential API calls.
7. Read back effective state through a fresh provider request, compare it with
   the approved after state and retain the execution receipt. Run the approved
   targeted validation, record actual outcomes, and stop rollout on a mismatch.
   Do not report success from a request's acceptance status alone.
8. On failure use only the record's reviewed bounded rollback, with the same
   independent approval and credential/environment boundary. A different or
   unsafe rollback requires revised human review. Read back and validate the
   rollback, retain partial outcomes and hand off unresolved state to the human
   owner. Do not leave an ambiguous mutation marked successful.

For unsupported provider APIs, the authorized human performs exactly the
reviewed operation in the provider UI or approved native tool, independently
checks the precondition after approval, and records operator/time, before/after,
approval and readback evidence. Keep the same independence, bounded inputs,
credential custody, validation and rollback requirements. Unsupported APIs do
not authorize an agent to use a broader human token.

## Drift and residual capabilities

Adopt a declared operational target of detecting **and reporting** drift within
15 minutes. The current schema accepts only 900 seconds; a different bounded
target requires a reviewed code/schema change and a recorded target before
activation. Declare cadence, maximum collection/retry time, outage handling,
owner, destination and response process. The local comparator is not a deployed
schedule or notification channel. A monitor must compare observed non-secret
effective rules, team roster/write access, bypasses, environment protections,
workflow refs and controlled provider settings against approved expectations.
Treat failed/incomplete reads as unverified and report them. Drift reports do
not automatically approve or repair live settings.

Measure from a fixture's known drift-injection timestamp or an explicitly
authorized bounded live change to the actual report time. Retain source
observations, detection and report timestamps, measured latency and target
pass/fail. Test both unexpected additions and removals, including a fourth team
member, a removed eligible member, permission loss, bypass restoration, disabled
self-review protection and changed variable/provider policy. A cadence alone is
not evidence that the reporting target was met.
The current local latency fixture exercises illustrative timestamp arithmetic
and rejection only. It measures no collector, notification or report-delivery
path and supplies no operational latency evidence.

An administrator can edit rules, membership, required checks and environments
outside the proposed path. An agent using a human's token appears to the API as
that human. Check runs and membership reads are asynchronous: a successful
check can become stale after review dismissal or membership drift, and a
scheduled monitor has detection latency. Keep native review controls, stale
dismissal, trusted check-source binding, short credential lifetime, human
custody, audit logs and a fresh merge/operation readback. Record these residual
capabilities explicitly; do not claim that a green fixture prevents them.

## Activation and closure evidence

Native-rule activation is a separate authorized human operation after independent
review and merge of the initial governance change. Team creation/verification
and valid-owner readback must precede merge of the team CODEOWNERS references,
as described above. Do not close #2781 on local checks.
The activation record must link all of the following:

- Eligible independent review of the exact bootstrap revision, final merged
  SHA, applicable local validation and the approved activation record/digest.
- Before/after effective rule exports (including inherited rules and every
  bypass actor), exact file-scoped required-team rule, stale-review controls,
  trusted required-check identity/source, and fresh post-activation readback.
- Independently approved pre-merge visible team creation/readback, complete exact-three membership, explicit
  repository write access, and CODEOWNERS validity/last-match evidence after
  Windows overrides. Team changes require the same review and audit boundary.
- Main-only administration/environment controls and independent actual
  approval, denied feature-ref/self-review/admin-bypass cases, least-privilege
  credential custody evidence and approved unsupported-API manual procedures.
- Representative allow/deny results bound to the exact tested head: missing,
  unlisted, self/sponsor, stale, dismissed and superseded review rejected;
  bot-authored policy rejected; exactly one eligible independent current
  approval accepted, including a new empty/tree-identical commit invalidating
  the previous head's approval. Exercise author as each eligible human to show a
  different eligible reviewer works without a second approval.
- CI path probes for a workflow, Windows workflow/action override, provider
  gate, variable/cache policy, test roster, release gate, policy/roster,
  validator/workflow and an attempted gate deletion or rename; plus product-only
  Rust, UI and documentation probes showing no extra CI-team approval. Retain
  actual GitHub allow/deny outcomes, not just expected fixture assertions.
- Approved live-setting operation and rollback receipts with fresh precondition,
  observed before/after, operator/time, rationale, affected jobs, tests and
  independent approver links, without secret values.
- Deployed drift execution/reporting evidence and measured latency against the
  declared target; owner acknowledgement, outage handling and remaining
  administrator/human-token/asynchronous-check limitations.

Pending evidence remains explicitly pending until obtained. A PR description,
desired-state file, workflow definition or passing local fixture is preparation,
not external activation or deployed-control verification.
