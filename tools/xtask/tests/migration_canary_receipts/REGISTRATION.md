# Corrected dependency wiring, 2026-09-28

The dedicated target now also includes the unchanged
`src/repository/python_text.rs` by path and exposes it under
`crate::repository::python_text`. Production code imports its existing
`is_space`; no shared repository module or manifest changed. The target's
module-level `expect(dead_code)` covers unused shared text helpers, not candidate
receipt code. Compilation and lint qualification remain pending.

The `migration_canary_receipts` filter selects 51 candidate tests: the original
36 plus 15 correction controls. Five inherited tests are outside that filter:
the string writer's original test and four existing Python-text tests.
See [CORRECTION.md](CORRECTION.md) for pending checks and current identity.

The original registration record below is retained verbatim. Its SHA-256 is
`b69640ee02c9fd848a3897b94f4615cb4d4ae4256ce1b085686b3af0b6332f00`.

# Registration handoff to bg_69c674c8

No shared module, CLI, manifest, lockfile, workflow, inventory or Just file was
edited by this candidate. No dirty prerequisite overlay is needed. The existing
HEAD dependencies serde, serde_json, sha2 and hex suffice.

The dedicated integration target is auto-discovered at
`tools/xtask/tests/migration_canary_receipts.rs`. It includes the candidate module
by path and the existing `ci_plan/plan_bytes.rs` string writer by path. It needs
no Cargo.toml registration or test-only change to a shared source file. This
also avoids asserting a production command exists before package admission is
available. The included string writer brings its existing single unit test;
use the `migration_canary_receipts` filter to count only this candidate's cases.

For later production linkage, the only module declaration needed in
`tools/xtask/src/automation/mod.rs` is:

```rust
pub(crate) mod canary_receipts;
```

Add that line only with an actual caller, otherwise the uncalled API is dead
code. There is deliberately no proposed CLI/dispatch variant for unchecked
identity JSON. The producer-package verification owner is still a prerequisite.
Its future typed adapter should construct:

```rust
let context = canary_receipts::ReceiptContext::from_verified_package(
    canary_receipts::VerifiedPackageInputs {
        identity,
        identity_sha256,
        plan,
    },
    current_workflow_run,
)?;
let report = canary_receipts::aggregate(&context, evidence_directory)?;
```

`ProducerIdentity` is only the receipt-relevant projection of the producer's
verified identity. The owner must validate schema 3, platform, candidate/base,
controller, selected source, run provenance, exact identity digest, plan bytes,
native executables, prepared llama source, oracle closure, summary and optional
candidate bundle BEFORE passing it here. Deserializing it is not verification.
`SourceFamilyPlan::parse` preserves the legacy controller checks from
`validate_plan`; it does not regenerate the plan or supply planner authority.
Its extra source fields are ignored, not rewritten. The caller must bind the
original plan bytes to the verified identity rather than hashing a projection.

Use `write_receipt(&context, evidence, WorkerResult { family, outcome, runner })`
for the existing receipt producer. It writes the same unversioned nine-field
receipt, sorted and ASCII escaped with a terminal newline. It replaces only
`evidence/receipt.json`, as legacy does; callers must give it the current
attempt's output directory, never historical artifact directories.

`aggregate` returns a diagnostic report even when workers fail. A returned
`Ok(report)` is NOT success. The adapter must print/append `report.summary()`,
require `report.is_green()`, and only then append `report.github_outputs()`.
The independent workflow `needs.family.result == success` gate is still
mandatory; complete uploaded receipts cannot replace it. No publication
eligibility is conferred by this module.

Keep legacy callers authoritative until queued compilation, tests, independent
oracle capture and review succeed. No command, workflow switch, or Python
removal is authorized by this handoff.
