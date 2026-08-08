# Release Validation Evidence And Gates

Read this file before planning or executing a release-validation run.

## Contents

1. Evidence standard
2. Minimum validation matrix
3. Change-to-test routing
4. Readiness gates
5. Defect severity
6. Evidence bundle layout

## Evidence Standard

An acceptable result is reproducible and identifies:

- candidate SHA and previous-release tag;
- exact host, platform, architecture, backend, hardware, and locally built
  archive checksum;
- command or interaction, input/fixture, timestamp, exit/HTTP status, and
  observed output;
- expected outcome and a comparison of expected versus observed behavior;
- raw evidence path and any redaction performed.

Screenshots prove rendered state but not API correctness. Logs prove emitted
events but not user-visible behavior. Unit tests prove a narrow code contract
but not packaged-product behavior. Use complementary evidence.

Preserve failures as evidence. Do not overwrite the only failing log with a
successful rerun; record attempt numbers and explain environmental retries.

## Minimum Validation Matrix

Every run must disposition these areas:

| Area | Minimum evidence |
|---|---|
| Provenance | Previous release tag/URL/date, candidate SHA, clean/dirty state, compare manifest |
| Build | Canonical `just` commands, exit codes, duration, toolchains, host/runtime/product manifests |
| Packaging | Archive listing, SHA-256, host import policy, exactly one selected runtime, extracted binary version |
| Startup | Fresh isolated state, readiness transition, bound ports, graceful stop, restart |
| Private mesh | Two candidate nodes, join evidence, peers on both sides, model union, recovery after peer loss |
| Management API | `/api/status` and every changed management route, schema and error behavior |
| OpenAI API | `/v1/models`, chat non-streaming/streaming, changed completions/responses paths, malformed-input errors |
| Inference | Exact local model, exact remote model, `auto`, response model identity, stable repeated requests |
| Agent behavior | `auto` and `mesh` tool-call loop when tool use, routing, guardrails, or OpenAI reduction changed |
| Logs | JSON parseability, lifecycle/routing/model events, warnings/errors, secret scan, no panic/fatal residue |
| UI | Initial load, status/model/peer parity with API, affected interactions, responsive view, console/network errors, screenshots |
| Compatibility | Candidate-to-candidate always; prior-release-to-candidate when protocol/runtime/install compatibility may change |
| Failure/recovery | Invalid request, unavailable model/peer, peer loss/rejoin, bounded shutdown, useful diagnostics |
| Install/upgrade | Installer/update path when touched; existing data/config compatibility and rollback notes |
| Security/privacy | Credentials redacted, trust/auth boundaries retained, no new unintended listener/publication |
| Performance/resources | Startup, memory/VRAM, latency/throughput sanity for performance-sensitive changes; baseline noted |
| Documentation | User-visible behavior, CLI help, config examples, migration and release-note wording agree with product |
| CI/release health | Required checks and release lanes inspected under `manage-ci`; no unrelated run used as evidence |

## Change-To-Test Routing

Add these checks when the delta touches the named surface:

- **Mesh/gossip/protocol/election/routing:** two-node candidate mesh, mixed
  previous/candidate versions, cross-node inference both directions, peer loss,
  stale-state recovery, additive-wire review.
- **OpenAI frontend/guardrails/tool use/MoA:** schema/error compatibility,
  streaming reconstruction, repeated same-prefix tool loops, `auto` and `mesh`,
  client harness smoke.
- **Skippy/runtime/ABI/model packaging:** applicable Skippy skill, runtime ABI
  match, package materialization, exact and split inference, cache/session
  behavior, native log review, compatible runtime selection.
- **CLI/config:** help text, default and explicit precedence, invalid config,
  upgrade from prior config, JSON and human output, exit codes.
- **UI/management API:** browser interaction, API/UI state parity, responsive
  layouts, accessibility sanity, console/network errors, stale/reconnect state.
- **Installer/release/package/SDK:** release target consistency, extracted
  archive only, checksums/manifests/import policy, no-driver smoke, installer
  detection, SDK consumer smoke when applicable.
- **Plugins/MCP/security/auth/telemetry:** prior protocol support unless a
  breaking change was approved, permission boundaries, secret redaction,
  opt-in/default behavior, privacy review, failure isolation.
- **Performance/concurrency/cache:** repeat trials, warm/cold distinction,
  resource ceilings, concurrency saturation, regression baseline and tolerance.

## Readiness Gates

Use these rules without averaging failures away:

### READY

- Every release-ledger item is `PASS` or justified `NOT_APPLICABLE` in every
  required platform/surface.
- No open severity 0 or 1 defect exists.
- No required common-matrix area is `PARTIAL`, `BLOCKED`, or `UNVERIFIED`.
- All locally built product checksums and host provenance are recorded.
- Candidate private mesh, APIs, logs, UI, inference, shutdown, and applicable
  compatibility checks pass.
- Required CI/release lanes are terminal and successful, or explicitly outside
  the authorized validation scope with the report decision set to `INCOMPLETE`.

### CONDITIONALLY_READY

Use only when all functional release claims pass and the remaining issue is a
bounded severity 2 or 3 risk with:

- named waiver owner and approver;
- written rationale and affected scope;
- mitigation, monitoring, rollback trigger, and expiry/date;
- no security, data-loss, protocol-break, packaging-corruption, startup, or
  core-inference risk.

### NOT_READY

Use when any required claim fails or is partial, any severity 0/1 defect is
open, packaging/provenance is invalid, a required compatibility test fails, or
the candidate can corrupt data, expose secrets, strand existing nodes/plugins,
or fail core startup/inference.

### INCOMPLETE

Use when required evidence is blocked or unverified, hosts/platforms were not
available, the canonical delta cannot be bounded, or required CI is not at a
terminal conclusion. Incomplete is not a soft pass.

## Defect Severity

| Severity | Definition |
|---|---|
| S0 | Security compromise, secret exposure, data corruption/loss, destructive behavior, or broad production outage |
| S1 | Core startup, packaging, mesh formation, API, or inference failure; breaking compatibility without approved migration |
| S2 | Important feature incomplete or materially degraded with a bounded workaround; non-core platform regression |
| S3 | Minor UX/docs/diagnostic defect with negligible functional impact |

Every defect must include reproduction steps, expected/actual result, affected
build checksum/host, evidence path, severity rationale, and release disposition.

## Evidence Bundle Layout

Use this layout unless the user specifies another location:

```text
target/release-validation/<run-id>/
├── release-validation-report.md
├── release-inventory.json
├── evidence-index.md
├── builds/<host>/
│   ├── provenance.json
│   ├── checksums.txt
│   └── manifests/
├── api/<host>/
├── logs/<host>/
├── ui/<host>/
├── inference/<case>/
├── compatibility/
└── defects/
```

Use relative links from the report. Store secrets nowhere. If redaction changes
an artifact, retain only the redacted copy and note the redaction method.
