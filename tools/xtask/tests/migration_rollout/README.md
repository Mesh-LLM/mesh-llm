# Local rollout consistency fixtures

Task 26 preparation only. These sources and fixtures have not been compiled,
formatted, linted, or executed. Cargo and associated validation are reserved.
No protected merge, dispatch, caller cutover, or acceptance is established.

## Ownership and integration

Only `tools/xtask/src/automation/rollout/` and
`tools/xtask/tests/migration_rollout/` are new. Shared registrations are left
unchanged. Integration needs these edits, performed by the shared-file owner:

1. Add `pub(crate) mod rollout;` to `src/automation/mod.rs`.
2. Add `Rollout(&'a [String])` to `cli::CliCommand` and parse
   `automation rollout <rest>` before the generic `automation` arm.
3. Dispatch `CliCommand::Rollout(rest)` to `automation::rollout::run(rest)`.
4. Include `automation::rollout::USAGE` in help.

No dependency, manifest, workflow, catalog, or test-target registration is
needed. The module's `cfg(test)` includes `tests/migration_rollout/mod.rs`.
The focused test-name filter is `migration_rollout`.

## Existing inputs, not new authority

The validator reads the existing bootstrap `binary_path=`,
`target_directory=`, and `host=` output and the existing CI parity
`summary.json` unchanged. It neither invokes bootstrap nor reruns parity.
The bootstrap dependency closure and planner behavior remain owned by those
commands, not reimplemented here.

The input JSON is a local observation envelope, not an attestation. It has:

- `repository`: local Git object database location, relative to the input file
  or absolute. No fetch is performed; lazy fetch and replacement objects are
  disabled for local Git reads.
- `catalog_sha`, `support_sha`, `protected_sha`, `source_sha`: existing full
  lowercase 40-hex commit identities. Git must contain every commit. Catalog
  precedes support, support precedes protected, and source includes catalog.
  Equality is permitted when no separate catalog/support addition was needed.
- `planner_sha`, `workspace_sha`, `runner_policy_sha`: observed code ownership,
  each equal to `protected_sha`. This does not authenticate the observation.
- `bootstrap`: `source_sha`, `report`, and `binary`. `report` and `binary` are
  `{ "path": "...", "sha256": "<64 lowercase hex>" }` byte bindings. Paths
  resolve relative to the envelope, not the repository. The recorded bootstrap
  binary/target paths must remain locally available. The binary is hashed,
  never executed. Its source attribution still needs external build evidence.
- `predecessors`: exactly one `{ "task": N, "evidence": <byte binding> }`
  for each task 11 through 25. Existing per-task formats are opaque and remain
  unchanged. Availability is checked, not acceptance or platform coverage.
- `shadow`: `protected_sha`, `source_sha`, `required_check: false`, and
  `summary: <byte binding>`. The non-required claim must be checked against
  external execution evidence before rollout. Local input cannot grant it.

Catalog comparison reads committed mode-100644 blobs for `ci/ownership.yml`
and `ci/slices.yml`, not mutable worktree files. Both byte contents must agree
at all four selected revisions. The protected support check establishes
presence of bootstrap/planner entry files, not a successful build. Provider
policy comparisons cover `select-ci-runners/action.yml` and
`ci/runner-images.json` between these revisions. They do not claim to inspect
external variables, runner groups, all workflow behavior, or prior policy
changes. No local field can authorize a policy difference.

Shadow validation consumes the existing seven comparison groups, derives
frozen case labels from the protected revision, and requires the existing
eight real-catalog profiles. Counts must match unique complete rows. It
rejects Rust-only evidence and differences. The already documented non-JSON
parser diagnostic explanation is allowed only on that exact comparison.
Fixture expectations are authored independently of the validator, using the
existing 63-case census and observed task-10 237-row summary shape.

Every successful local result explicitly reports
`scope: local_fact_consistency_only` and
`task26_acceptance: not_established`. The observation-file digest binds the
selected identities and input hashes without adding fields to planner output
or changing the canonical plan digest. A hash authenticates no author.

## Pending validation after queue grant

After shared registration, run serially from the integrated candidate:

```sh
cargo fmt -p xtask
cargo fmt -p xtask --check
cargo check --locked -p xtask
cargo test --locked -p xtask migration_rollout -- --nocapture
cargo clippy --locked -p xtask --all-targets -- -D warnings
just no-console-print
just automation-bootstrap
```

The focused suite contains 45 tests on Unix, 44 on Windows. A zero-test run is
not evidence. Tests create isolated local Git repositories and inert binary
files; no Python, network, service, or real rollout runs. The success fixtures
do not establish any predecessor acceptance. Existing predecessor tests and
the full repository validation union remain the integration owner's gates.

Drive the registered command through a terminal with a locally prepared
observation envelope and retained existing bootstrap/parity files:

```sh
cargo xtool automation rollout --input /absolute/observations.json
cargo xtool automation rollout --approve /absolute/observations.json
```

The first must report local consistency only; the second must reject the
unknown option. The in-process command fixture exercises the same dispatcher
with inert evidence. Do not rerun legacy Python parity, dispatch dual runs, or
claim task 26 acceptance under a Cargo queue grant alone.

## Open acceptance contracts

Existing bootstrap output has no source SHA or executable checksum. The
existing shadow summary has no protected implementation identity, run/attempt,
required-check status, execution authorization, or signed provenance. Tasks
11 through 25 have no shared acceptance receipt schema. This validator checks
their supplied byte references but cannot turn those missing attestations into
truth. External merge/default-branch, build provenance, predecessor acceptance,
non-required shadow execution, and authorization remain explicit open gates.
