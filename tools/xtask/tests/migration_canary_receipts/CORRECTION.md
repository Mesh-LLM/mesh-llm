# Canary receipt correction, 2026-09-28

Status: code-ready, not executed or qualified. The workload owner retains the
global Cargo queue. No Cargo, build, test, formatter, LSP, Python, canary or
network command ran during this correction. No root writes, index changes,
commits, caller changes, dependencies or shared-module edits were made.

This is a correction to the isolated candidate at
`/var/folders/5q/y9dmlwq11tqd74j_17t5p5ym0000gn/T/opencode/wt-task24-canary-receipts`,
base `a4e04070db2c6b8e2644a40f5c4f207789df6c2d`. Task 24 remains incomplete.

## C1: preserve nested duplicates, correct the claim

The legacy `read` uses `json.loads`, and `read_json_documents` uses
`json.JSONDecoder.raw_decode`. Both retain the last decoded object-key value.
The candidate's nested adapters already do the same through a boundary-local
`serde_json::Map`. They still require object-shaped rows and convert the final
values into typed models or lanes before validation. The adapters are now named
`object_last_wins` and `object_rows_last_wins` to make that behavior explicit.
Their implementations and acceptance behavior did not change.

The root maintainer record `.omo/evidence/maintainer-decisions.md`, read only,
permits malformed-input diagnostic changes and safe resource limits while
preserving intended acceptance and rejection. It does not approve blanket
duplicate-key rejection for canary receipts. Its SHA-256 at this read was
`9a6a6b3ba6f8a70337459b5ab7dbd4a2010cd54f01648d8a0777eaae2020ab5e`.
The original CANDIDATE.md duplicate-key approval statement was an overclaim,
not authorization to replace last-wins behavior with a stricter decoder.
SOURCE_REVIEW.md's C1 diagnosis is retained; its enforcement recommendation
must not be read as a maintainer decision.

These duplicate rules remain distinct:

| Input | Current candidate behavior | Relation to legacy |
| --- | --- | --- |
| Nested outcome, selected model, matrix, matrix row or shard object keys | Last decoded key value wins before typed conversion | Preserved |
| Known top-level fields in producer identity, source plan, result row or receipt provenance | Typed deserialization rejects a duplicate | Existing difference, unchanged and not approved by this correction |
| Known top-level receipt payload fields | Selected receipt is typed and rejects duplicates; superseded payload remains ignored | Existing selection boundary, unchanged |
| Duplicate family receipts in one attempt, duplicate lane rows or consolidated certification rows | Rejected by their respective semantic checks | Preserved; unrelated to object-key duplication |

Ignored fields do not acquire a new uniqueness check. This is not a recursive
duplicate-rejection policy. Whether to retain the existing top-level typed
rejection at cutover needs an explicit decision or a separately scoped parity
correction. Source evidence establishes the difference, not approval. The new
top-level controls characterize the existing candidate; they do not approve it.
No decision is needed to preserve nested last-wins or repair C2.

## C2: match document-separator whitespace

`read_json_documents` now trims with the existing
`repository::python_text::is_space` instead of `char::is_whitespace`. This adds
Python's raw `0x1c`, `0x1d`, `0x1e` and `0x1f` separators before, between and
after result documents. It does not change JSON whitespace inside an object,
the source-plan or receipt decoder, raw-byte hashing, or stored bytes.
The dedicated test target includes the shared text module explicitly, without
editing it or importing the repository dispatcher.

## Prepared controls and preserved history

All five original test/support files are byte-identical, retaining 36 named
tests and the 21-case aggregate table. New `duplicate_keys.rs` has 10 tests;
new `separators.rs` has five. The focused candidate filter therefore selects
51 tests by static count, not an observed execution count. The target also
includes five pre-existing dependency tests outside that filter.

| Added input or test | Expected distinction, still pending execution |
| --- | --- |
| `fixtures/separator-1c.jsonl` | Raw 0x1c at prefix, inter-document boundary and suffix is accepted |
| `fixtures/separator-1d.jsonl` | Raw 0x1d at the same three positions is accepted independently |
| `fixtures/separator-1e.jsonl` | Raw 0x1e at the same three positions is accepted independently |
| `fixtures/separator-1f.jsonl` | Raw 0x1f at the same three positions is accepted independently |
| Raw controls inserted immediately after the first `{` | Each of the four remains a JSON error inside an object |
| `fixtures/nested-status-pass-last.jsonl` | `fail`, then `pass` certifies with a matching raw results digest |
| `fixtures/nested-status-fail-last.jsonl` | `pass`, then `fail` reports RequiredLane and no green output with a matching digest |
| Raw plan model `class` duplicates, both orders | The final class determines causal versus workload result selection |
| Raw matrix `include` duplicates, both orders | The final list determines complete versus empty membership |
| Raw top-level duplicate controls | Existing result-row, plan, identity and receipt-provenance rejection remains visible |

The six new fixture files are hand-authored bytes, not output captured from
either implementation. Each separator file is 331 bytes, with its control byte
at offsets 0, 111 and 329, and LF at offset 330. `od -An -tx1` confirmed actual
bytes rather than textual escape sequences. Duplicate fixtures are read
directly without a JSON-value round trip. Plan/identity duplicates use raw text
replacement; the receipt-provenance control inserts duplicate text after the
ordinary fixture constructor writes its receipt.

Before edits, all 29 entries of CANDIDATE.sha256 passed read-only verification.
After edits, FROZEN.sha256 and EXPECTED.sha256 still pass. Those manifests,
the original fixture files, SOURCE_REVIEW.md and CANDIDATE.sha256 are unchanged.
The original CANDIDATE.md and REGISTRATION.md bodies are retained verbatim under
clearly marked correction headers, with their old hashes recorded there.

CORRECTION.sha256 identifies the corrected candidate and its unchanged local
dependencies. CANDIDATE.sha256 remains the original historical identity and
must not be regenerated. Its changed entries no longer describe current files;
this is not a failed parity receipt. SOURCE_REVIEW.md remains source-only
evidence, never relabelled as an executed test report.

## Pending serial checks

Only the workload validation owner may execute these after queue assignment.
Use the candidate root as the working directory, preserve fresh logs and actual
exit statuses, and read `docs/design/TESTING.md` first. Missing offline
dependencies are blockers, not authorization to install or use the network.

```sh
shasum -a 256 -c tools/xtask/tests/migration_canary_receipts/CORRECTION.sha256
rustfmt --edition 2024 --config skip_children=true --check tools/xtask/src/automation/canary_receipts/*.rs tools/xtask/tests/migration_canary_receipts.rs tools/xtask/tests/migration_canary_receipts/*.rs
just with-lld cargo test --offline --locked -p xtask --test migration_canary_receipts migration_canary_receipts -- --nocapture --test-threads=1
just with-lld cargo check --offline --locked -p xtask --tests
just with-lld cargo clippy --offline --locked -p xtask --all-targets -- -D warnings
just with-lld cargo test --offline --locked -p xtask --test migration_canary_receipts writer::migration_canary_receipts_full_local_receipt_flow_is_digest_bound -- --exact --nocapture
```

Require 51 selected candidate tests in the focused run, including all four
independently named separator tests and all original 36 cases. Require one test
in the exact local receipt-flow run. Zero selected tests is not success. The
five inherited dependency tests are outside the focused filter.

Run the exact twelve existing Python test names retained in CANDIDATE.md using
its `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest ...` command. They exercise
local fixture packages, not restore/build/certify. They do not establish parity
for the six added fixture files or raw duplicate mutations. Fresh identical-input
legacy/Rust observations remain pending for those inputs and the original
source-derived fixture table. Keep observed output/status in new files; never
rewrite frozen expectations or add a Python helper.

Also pending: show each separator test fails with the old predicate in a
disposable validation copy, and show duplicate-order controls detect first-wins
or blanket nested rejection. Retain failed mutation evidence separately.
Receipt and results exact-limit/limit+1 qualification remain pending at the
unchanged 1 MiB and 64 MiB caps. These are candidate bounds, not measured
producer maxima or newly approved limits.

After authorized integration, the owner still owes the xtask regression,
formatter, no-console-print and CI validation union. Queue-owned diagnostics
and independent review remain pending. No product build, native execution,
model qualification, package admission or publication authority follows from
this source-only correction. VerifiedPackageInputs remains a caller obligation;
the independent workflow family-job result gate remains mandatory.

## Source self-review

The production edit changes only document-separator consumption and names the
existing nested duplicate semantics. Typed domain values, arithmetic, errors,
limits, provenance, latest-attempt selection, hashing and output bytes are
unchanged. No unsafe code, production unwrap, numeric cast, new fallback,
logging, retry or one-off production helper was added. The added test files
separate duplicate-key behavior from document-separator behavior. Tests can
distinguish the targeted regressions by construction, but no red/green claim
is made until execution. Formatting and compilation are expressly pending.
