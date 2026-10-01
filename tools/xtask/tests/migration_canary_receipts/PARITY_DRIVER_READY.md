# Canary receipt parity driver handoff

An executable identical-input driver is still required. No driver code is
registered in `tools/xtask/tests/migration_canary_receipts.rs` yet. The existing
candidate unit/integration tests and the saved legacy Python tests remain
separate evidence, not identical-input comparisons.

## Verified inputs and safe launch

The legacy aggregate CLI is read-only for this task when called with
`aggregate --package <fixture> --identity <sha256> --evidence <fixture>` and
local `GITHUB_OUTPUT` / `GITHUB_STEP_SUMMARY` paths. Its source hash is
`89dc3f313c95482e7022b58d7a274eb724e18eb4aca20ae9a8907ae45245e5c2`.
Its existing test source hash is
`280bb2ac4b504713da600b2d11cb86ec43bf970fe6826892eef16372f63c0f05`.
The imported `scripts/lib/canary_family_memory.py` hash is
`371ef5c3c4a9b77cb0e9912a4b7d2a868c9a312ef1c5a956d483b68b32ea41fa`.
The existing bounded process supervisor
`scripts/run-command-with-timeout.py` hash is
`e7c46a677e93f82562534426a0c78cb0c03fef9601f4eab0013163adf9a63266`.
All are present in this candidate worktree; no dependency restoration is
needed. Pin these hashes before invoking the CLI. Set
`PYTHONDONTWRITEBYTECODE=1`; clear inherited `GITHUB_STEP_SUMMARY` and
`GITHUB_OUTPUT`; then provide fresh local paths for both. Invoke the legacy CLI
only through `scripts/run-command-with-timeout.py --seconds 120
--cleanup-on-exit -- python3 scripts/llama-canary-family-evidence.py aggregate
...`.

Do not call legacy `pack`, `build`, `restore`, `certify`, `publication`, or any
upload/job/model path.

## Fixture invariants

Use one immutable two-family package fixture for the entire test. Build its
synthetic artifact stubs first, use the frozen `fixtures/plan.json` exact bytes,
put that plan's SHA-256 into the synthetic identity, then serialize the identity
once and hash those final identity bytes. The verified inputs accepted by the
candidate context must derive from those same bytes. Assert identity bytes,
plan bytes, and their digests remain unchanged for every scenario.

For each comparison, make separate legacy and Rust evidence directories and
write byte-identical worker receipts and results to each. Recompute
`results_sha256` after result-byte changes, except in the explicit stale-hash
case. Never mutate the frozen fixtures in place or alter the plan to work around
a stale package digest.

Cover at least: successful two-family aggregation; newest failure with no
fallback to old success; duplicate same-attempt receipt; missing family;
mismatched source identity hash; stale result hash; invalid result with matching
hash; duplicate JSON-key last-wins cases for result family, lane status, receipt
outcome, and result digest; and raw separators `0x1c` through `0x1f` both between
JSON records and inside an object. For each case compare process status,
aggregate green decision, passed count, and `GITHUB_OUTPUT` presence/content.
Compare exact stdout and summary bytes on green success. On errors with
different Python/Rust diagnostics, compare stable status/count output and
withholding of green output, not traceback prose.

## Registration and queued commands

Place the test in `tools/xtask/tests/migration_canary_receipts.rs` or a module
registered from that target. The target is auto-discovered; no Cargo manifest,
workflow, shared module, or legacy file change is needed.

Do not run Cargo while another owner holds its queue. Once that owner assigns the
queue, from the candidate root run the focused test first:

```sh
just with-lld cargo test --offline --locked -p xtask --test migration_canary_receipts differential::migration_canary_receipts_legacy_differential_compares_identical_inputs -- --exact --nocapture --test-threads=1
```

Then run the serial xtask gates:

```sh
just with-lld cargo check --offline --locked -p xtask --all-targets
just with-lld cargo clippy --offline --locked -p xtask --all-targets -- -D warnings
cargo fmt -p xtask -- --check
```

The existing full isolated candidate test command remains:

```sh
just with-lld cargo test --offline --locked -p xtask --test migration_canary_receipts migration_canary_receipts -- --nocapture --test-threads=1
```

Optional captures must use a fresh, nonexistent path in
`MESH_CANARY_RECEIPT_PARITY_EVIDENCE`. Keep `validation-20260928-serial/`
unchanged and put all new captures/hashes in a new directory. No execution or
parity result is claimed by this handoff.
