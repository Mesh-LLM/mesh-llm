# Canary receipt contract validation

The earlier Python differential-driver assignment is superseded by the
maintainer decision recorded in `.omo/evidence/maintainer-decisions.md` and
the normative `.agents/skills/manage-ci/SKILL.md`. Do not add a driver that
invokes Python or reproduces interpreter formatting and parser quirks.

The maintained `migration_canary_receipts` target owns aggregation, result
validation, identity admission, duplicate-key handling, record separators,
resource limits, output writing, and the actual handoff CLI fixtures. Validate
the consumed contracts through those Rust owners. Preserve exact bytes where
a caller consumes or hashes them. A fixture success proves its own contract;
it does not certify a model, native backend, SDK, hosted runner, or publication.

Run from the repository root, with no concurrent Cargo owner:

```sh
just with-lld cargo test --offline --locked -p xtask --test migration_canary_receipts -- --test-threads=1
just with-lld cargo clippy --offline --locked -p xtask --all-targets -- -D warnings
just with-lld cargo fmt --all -- --check
just ci-validate
```

Use finite local fixtures and fresh evidence destinations. Keep historical
validation captures unchanged. Earlier candidate and provenance notes record
their original snapshots; pending differential captures in those notes are
not current acceptance requirements. No network, model download, upload,
publication, or workflow dispatch is authorized by this document.
