---
name: release-validation
description: Use this skill when validating a MeshLLM release candidate or current HEAD against the last GitHub release, assembling the canonical feature/fix/modification inventory, testing locally built release bundles on user-approved real hosts and private meshes, deciding release readiness, or producing a formal evidence-backed release-validation report.
---

# Release Validation

Validate the candidate as a product, not merely as source code. Every claimed
change must have a disposition and direct evidence. Do not publish a release,
push a tag, alter production, or use a host the user did not put in scope.

Read [references/evidence-and-gates.md](references/evidence-and-gates.md) in
full before planning or executing validation. Copy
[assets/release-validation-report-template.md](assets/release-validation-report-template.md)
for the final report; do not weaken or omit its required tables.

## Required Inputs

Resolve these before remote execution:

- candidate ref, defaulting to the current `HEAD`;
- repository, normally `Mesh-LLM/mesh-llm`;
- exact user-approved SSH aliases or local hosts, platform/backend on each, and
  any cost or time limits;
- models suitable for each host and any feature-specific fixtures;
- output directory, defaulting to
  `target/release-validation/<UTC timestamp>-<short SHA>/`;
- whether a previous-release mixed-version node is authorized when compatibility
  testing is required.

If host selection, access, or cost authority is missing, complete the read-only
inventory and test plan, then stop before remote build or launch.

## Route To Existing Skills

Read each applicable skill completely before acting:

- `.agents/skills/remote-observable-process/SKILL.md` for every long-running or
  interactive SSH build/server session;
- `.agents/skills/deploy-macos/SKILL.md`,
  `.agents/skills/deploy-linux-gpu/SKILL.md`, or
  `.agents/skills/deploy-windows/SKILL.md` for the selected host platform;
- `.agents/skills/mesh-join/SKILL.md` for private-mesh creation and verification;
- `.agents/skills/connect-agents/SKILL.md` for agent/tool-call validation;
- `.agents/skills/manage-ci/SKILL.md` before inspecting CI, release workflows,
  runners, artifacts, or live workflow results;
- the relevant Skippy, plugin, telemetry, configuration, or benchmark skill
  when the delta touches that subsystem.

## Workflow

### 1. Freeze Provenance

1. Record the candidate SHA, branch, remotes, submodule state, dirty paths,
   toolchain versions, and current UTC time.
2. Refuse to present dirty or mismatched builds as reproducible. Either obtain
   explicit approval to validate the exact dirty tree and record its diff hash,
   or use a clean checkout at the candidate SHA.
3. Inspect the latest published GitHub release, its tag, notes, assets, checksums,
   publication time, and release workflow conclusion. Distinguish stable,
   prerelease, draft, and manually superseded releases.
4. Before collecting inventory, resolve the candidate SHA and previous-release
   tag commit and derive each release base from its merge-base with
   `origin/main`. An off-main candidate must be passed as its explicit release
   tag or have exactly one local tag pointing at its SHA; use that canonical tag
   for the subject check and fail closed if it is absent or ambiguous. Allow
   zero commits above a release base, or exactly one commit whose subject is the
   tag-specific `<tag>: prepare release source`. Require the previous-release
   base to be an ancestor of the candidate base. Fail closed if `origin/main` is
   unavailable or any tag, subject, commit-count, or base ordering check fails.
5. Run `scripts/collect-release-inventory.py` from this skill to capture a raw
   JSON evidence manifest. If its exact release tag is missing locally, verify
   the configured remote URL and fetch that tag before rerunning. The script
   does not classify changes.

### 2. Build The Canonical Delta Ledger

Use all of these sources, not release notes or commit subjects alone:

- GitHub comparison and merged PRs from the release tag through candidate SHA;
- local commit history and changed-file diff;
- PR bodies, linked issues, labels, tests, docs, migrations, and generated
  artifacts;
- user-visible CLI/API/UI/config/protocol behavior found in the code;
- release, installer, packaging, dependency, security, and observability changes.

Create one atomic ledger row per externally meaningful claim. Merge duplicate
PRs/commits into one item, but split unrelated behavior hidden in one PR.
Classify every row as `FEATURE`, `BUG_FIX`, or `REVISION`:

- `FEATURE`: a newly available user/operator/developer capability;
- `BUG_FIX`: behavior that now satisfies an existing contract or removes a
  defect/regression;
- `REVISION`: changed semantics, UX, performance, dependency, packaging,
  protocol, docs, or operational behavior that is neither of the above.

Give each row a stable ID (`RV-FEAT-###`, `RV-FIX-###`, or `RV-REV-###`), a
precise claim, source PRs/commits/files, affected surfaces, compatibility and
risk notes, and at least one positive and one relevant negative/edge test.
Record internal-only changes as revisions when they can affect release risk;
otherwise list them in the excluded/non-release-impact appendix with rationale.

### 3. Design Tests Before Building

Map every ledger item to concrete checks and an evidence destination. Cover
the common release matrix in the evidence reference plus all change-specific
paths. Mark a test `NOT_APPLICABLE` only with a written reason. `UNVERIFIED` is
not a pass.

Use risk to order work: provenance and packaging first, startup/readiness next,
then APIs/logs/UI/inference, feature claims, failure/recovery, mixed-version
compatibility, and nonfunctional checks. Do not allow one smoke test to stand
in for multiple materially different claims.

### 4. Build Canonical Products On Real Hosts

1. Confirm each host identity, OS/architecture, backend, GPU/driver/runtime,
   free disk/RAM/VRAM, toolchain, ports, and existing MeshLLM processes.
2. Transfer or check out the exact candidate source. Verify the candidate SHA
   on every host before building.
3. Use `just`; never invoke ad hoc Cargo builds as release evidence. Build the
   three-layer product with the applicable canonical recipes:

   ```bash
   just release-host-build
   just release-runtime-build <backend>
   just release-bundle <candidate-version> <output-directory>
   ```

   Use the Windows/platform-specific recipes where the Justfile requires them.
4. Record commands, exit codes, duration, output archive names and SHA-256,
   host-import policy results, product/runtime manifests, ABI/version metadata,
   binary version, and archive contents. Run `just check-release` and any
   applicable consistency checks.
5. Execute the extracted packaged object. Do not validate only
   `target/release/mesh-llm`, and do not substitute an older downloaded runtime.

### 5. Exercise A Private Mesh

Use at least two user-approved real hosts when available. Start foreground,
observable processes with JSON logging and isolated ports/data/runtime state.
Create a private mesh on one candidate bundle, join the other candidate bundle
with its invite token, and wait for explicit readiness rather than sleeping a
fixed interval.

Prove on both nodes:

- peer membership and stable readiness;
- `/api/status` and relevant management APIs;
- `/v1/models` union and local/remote model identity;
- non-streaming and streaming inference through exact model IDs and `auto`;
- `mesh` and tool-call behavior when supported or affected;
- structured logs parse as JSON, contain expected lifecycle/routing events,
  and contain no panic, secret, unexplained retry storm, or hidden fatal error;
- embedded UI loads, reflects the same state, has no blocking console/network
  errors, and completes the item-specific interactions;
- graceful stop, peer loss/recovery, restart, and cleanup.

For wire, gossip, routing, discovery, packaging, or compatibility changes, add
a separate mixed-version private-mesh check using the last released packaged
binary on one host and the candidate on another. Never replace the all-candidate
mesh with this compatibility check.

### 6. Judge Every Claim

Assign exactly one status to each ledger item:

- `PASS`: the claim is complete and directly proven in every required scope;
- `FAIL`: observed behavior contradicts the claim or creates a release blocker;
- `PARTIAL`: part works, but the claim, platform matrix, UX, docs, or recovery
  behavior is incomplete;
- `BLOCKED`: validation could not run because a named prerequisite is missing;
- `NOT_APPLICABLE`: a planned dimension truly does not apply, with rationale;
- `UNVERIFIED`: no adequate evidence was obtained.

Link immutable or locally preserved evidence: commands with exit codes, JSON
responses, redacted log excerpts, screenshots, checksums, manifests, test
outputs, and defect references. Never infer `PASS` from code inspection alone.

### 7. Produce The Formal Report

Write `release-validation-report.md` from the bundled template and store raw
evidence beside it. Include an evidence index with relative paths. Redact
tokens, credentials, private addresses when required, and customer data.

Apply the gate rules from the evidence reference. Give one decision:
`READY`, `CONDITIONALLY_READY`, `NOT_READY`, or `INCOMPLETE`. A conditional
decision requires an explicit waiver owner, rationale, expiry, and bounded
residual risk. Do not call a release ready while any required row is failed,
partial, blocked, or unverified.

## Operating Rules

- Preserve unrelated worktree and host state. Use isolated directories, ports,
  and process identifiers; stop only processes created by this run.
- Use SSH aliases supplied by the user or documented in `context/COMPUTERS.md`;
  never invent a host or use a raw IP when an alias exists.
- Do not expose invite tokens, API keys, owner keys, release signing material,
  private paths, or unredacted logs in the report.
- Do not fix discovered product defects during validation unless the user
  separately authorizes implementation. Record a reproducible defect and its
  release impact.
- Do not dispatch/cancel CI, publish artifacts, push commits/tags, create a
  release, or mutate GitHub configuration without explicit authorization.
- Report ongoing host cost and stop remote processes promptly when evidence is
  complete or the run is blocked.
