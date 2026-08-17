# MeshLLM Release Validation Report

## Document Control

| Field | Value |
|---|---|
| Candidate version | |
| Candidate commit | |
| Previous release | |
| Comparison range | |
| Validation run ID | |
| Started / completed (UTC) | |
| Validator | |
| Evidence root | |
| Decision | `READY` / `CONDITIONALLY_READY` / `NOT_READY` / `INCOMPLETE` |

## Executive Decision

State the decision, the evidence supporting it, the most important residual
risks, and the exact action required before release. Do not use ambiguous
language such as “mostly ready.”

## Scope And Exclusions

Describe validated platforms, hosts, backends, models, APIs, UI flows, and
compatibility pairs. List every excluded dimension and why it was excluded.

## Release Basis

### Previous Release

Record GitHub release URL, tag, date, release type, assets, notes, and relevant
workflow conclusion.

### Candidate Provenance

Record SHA, branch/ref, clean or dirty state, dirty-diff hash when authorized,
submodules, source-transfer method, and inventory manifest link.

## Canonical Change Inventory And Validation Ledger

| ID | Category | Release claim | Source PRs/commits/files | Affected surfaces | Risk | Planned checks | Status | Evidence / defects |
|---|---|---|---|---|---|---|---|---|
| | `FEATURE` / `BUG_FIX` / `REVISION` | | | | | | `PASS` / `FAIL` / `PARTIAL` / `BLOCKED` / `NOT_APPLICABLE` / `UNVERIFIED` | |

### Excluded Or Non-Release-Impact Changes

| Change | Reason excluded | Evidence reviewed |
|---|---|---|
| | | |

## Environment And Build Provenance

| Host alias | OS / arch | Hardware / backend | Candidate SHA | Canonical build commands | Archive | SHA-256 | Manifest/import-policy result | Status |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

## Product Validation Results

| Area | Case | Host / topology | Expected | Observed | Status | Evidence |
|---|---|---|---|---|---|---|
| Packaging | | | | | | |
| Startup / shutdown | | | | | | |
| Private mesh | | | | | | |
| Management API | | | | | | |
| OpenAI API | | | | | | |
| Logs | | | | | | |
| UI | | | | | | |
| Inference | | | | | | |
| Agent/tool calls | | | | | | |
| Failure / recovery | | | | | | |
| Compatibility | | | | | | |
| Install / upgrade | | | | | | |
| Security / privacy | | | | | | |
| Performance / resources | | | | | | |
| Documentation | | | | | | |
| CI / release lanes | | | | | | |

## Private Mesh Evidence

Document topology, isolated ports/state, invite handling and redaction, peer
counts on each node, model union, routing identity, failure/rejoin timeline,
and candidate-to-candidate results.

## Mixed-Version Compatibility

Document the previous-release/candidate topology, supported protocol paths,
gossip/routing/API/inference results, and any migration or incompatibility.
If not required, state the evidence-backed rationale.

## API, Logs, UI, And Inference Findings

Summarize contract checks, structured-log review, UI/API state parity,
screenshots and browser errors, local/remote inference, streaming, and agent
tool-call behavior. Link raw evidence.

## Defects And Anomalies

| Defect | Severity | Reproduction | Expected / actual | Scope | Workaround | Release impact | Evidence |
|---|---|---|---|---|---|---|---|
| | `S0` / `S1` / `S2` / `S3` | | | | | | |

## Risk, Waiver, And Follow-Up Register

| Risk / gap | Impact and likelihood | Mitigation | Owner | Approver | Expiry | Rollback / stop trigger | Status |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## Rollback And Operational Readiness

State rollback artifacts and commands, config/data compatibility, monitoring
signals, support diagnostics, and the conditions that require rollback.

## Final Gate Assessment

| Gate | Result | Evidence / rationale |
|---|---|---|
| All release claims dispositioned | | |
| Canonical packages proven on real hosts | | |
| APIs, logs, UI, and inference proven | | |
| Candidate private mesh proven | | |
| Required mixed-version compatibility proven | | |
| No open S0/S1 defect | | |
| Required CI/release lanes terminal | | |
| Residual risk accepted and bounded | | |

## Sign-Off

State the final decision and list validator, release owner, waiver approvers (if
any), timestamp, and the exact next action. Human approval remains required to
publish even when this report says `READY`.

## Evidence Index

Link every referenced command transcript, JSON response, redacted log,
screenshot, checksum, manifest, benchmark, CI run, and defect artifact using a
relative path.
