# Top-level duplicate correction, 2026-09-28

Status: code-only candidate, ready for the workload owner's serial validation
queue. No compilation, test execution, runtime pass, differential parity or
cutover approval is claimed. No queue job was launched. Task 24 is incomplete.

Candidate root:
`/var/folders/5q/y9dmlwq11tqd74j_17t5p5ym0000gn/T/opencode/wt-task24-canary-receipts`.
HEAD: `a4e04070db2c6b8e2644a40f5c4f207789df6c2d`.
Git common directory remains the isolated
`/private/var/folders/5q/y9dmlwq11tqd74j_17t5p5ym0000gn/T/opencode/task24-canary-receipts-repo/.git`.

This record supersedes the top-level rejection statements and 51-test count in
CORRECTION.md, CANDIDATE.md, REGISTRATION.md and SOURCE_CHECKS.md. Those records,
SOURCE_REVIEW.md, CANDIDATE.sha256 and CORRECTION.sha256 remain byte-identical
history. They have not been relabelled as execution evidence or regenerated.
The incoming CORRECTION.sha256 had 49 matching entries before this edit. Its
SHA-256 remains `9864d35d3899f02d5eb3399554c47c83ed375634fc99d2a576c35da1d458e34d`.

## Decoder contract

Legacy `read` calls `json.loads` at script lines 53-54; `read_json_documents`
calls `JSONDecoder.raw_decode` at lines 497-511. Both keep the last decoded key
value. Legacy policy checks consume that decoded object, not every overwritten
value. Blanket top-level duplicate rejection was a candidate restriction, not
an approved policy.

Five top-level types now use one boundary-local ordered object adapter before
their existing derived typed decoder: ProducerIdentity, PlanDocument,
ResultRow, ReceiptProvenance and WorkerReceipt. The adapter retains each key's
first position and replaces only its value. It then feeds the ordered entries
to Serde's MapDeserializer. A sorted map was rejected here because it would
change which malformed field the typed decoder sees first. Serde's
`remote = "Self"` derive keeps field declarations, defaults, flattening and
typed conversion in their owning structs instead of duplicating them.

The legacy-shaped nested adapters are unchanged. Raw JSON values stay inside
the decoding boundary; validation still receives typed values. Provenance and
attempt checks precede selection; only the newest selected receipt's payload
is validated. Failed selected outcomes, foreign identities, invalid plan
membership, mismatched raw results hashes, duplicate same-attempt receipts,
duplicate lane rows and duplicate consolidated certifications still fail their
existing checks. No validator, limit, raw-byte hash or writer logic changed.

Result document separators still use `repository::python_text::is_space`.
The four literal 0x1c-0x1f files, their five tests and the shared predicate are
unchanged. Each byte remains accepted outside result objects and rejected
inside them by the JSON decoder. This is source reasoning, not an observed run.

## Exact edit paths

All paths below are relative to the candidate root, never the primary checkout.

| Path | Change |
| --- | --- |
| `tools/xtask/src/automation/canary_receipts/boundary.rs` | Ordered last-value-wins object adapter |
| `tools/xtask/src/automation/canary_receipts/identity.rs` | ProducerIdentity decode boundary |
| `tools/xtask/src/automation/canary_receipts/plan.rs` | PlanDocument decode boundary |
| `tools/xtask/src/automation/canary_receipts/receipt.rs` | Provenance and selected receipt decode boundaries |
| `tools/xtask/src/automation/canary_receipts/results.rs` | ResultRow decode boundary |
| `tools/xtask/tests/migration_canary_receipts.rs` | Register the new test module only |
| `tools/xtask/tests/migration_canary_receipts/duplicate_keys.rs` | Correct four candidate-only expectations |
| `tools/xtask/tests/migration_canary_receipts/top_level_duplicates.rs` | Fourteen additional discriminating tests |
| `tools/xtask/tests/migration_canary_receipts/fixtures/root-family-planned-last.jsonl` | Raw foreign-then-planned result object |
| `tools/xtask/tests/migration_canary_receipts/fixtures/root-family-foreign-last.jsonl` | Raw planned-then-foreign result object |
| `tools/xtask/tests/migration_canary_receipts/TOP_LEVEL_CORRECTION.md` | This pending-verification receipt |
| `tools/xtask/tests/migration_canary_receipts/TOP_LEVEL_CORRECTION.sha256` | Fresh candidate and dependency byte identity |

TOP_LEVEL_CORRECTION.sha256 binds every current candidate source/test, all
fixtures, this receipt, historical records/manifests, unchanged local dependency
modules, Cargo manifests/lockfile and legacy script/test. It excludes itself.
Hash agreement establishes byte identity only, not behavior or package trust.

The two new file payloads are hand-authored, not captured decoder output. They
are consumed directly with include_bytes and tested through aggregation with
matching raw-byte results digests, so a hash failure cannot mask key semantics.

```text
7a8c6b076a29f63d333400ed253b73d6f82262fbe8cb45a14da8a1910ff61fcf  tools/xtask/tests/migration_canary_receipts/fixtures/root-family-planned-last.jsonl
ce1f8eaf1573f22158dccd37504ee6ac02e79532bd0da2913950c8915b3b2e72  tools/xtask/tests/migration_canary_receipts/fixtures/root-family-foreign-last.jsonl
```

## Corrected expectations and added controls

The six nested tests in duplicate_keys.rs are unchanged. Its four former
`root_*_duplicate_remains_rejected` tests kept their raw input construction but
now expect the valid last value. The tests were renamed to describe that
behavior. Their old expectations contradict the legacy calls cited above:

| Raw duplicate sequence | Corrected expectation |
| --- | --- |
| Result family `foreign`, then `dense` | Valid family results |
| Plan selected_models empty, then complete | Valid source plan |
| Producer run_id `999`, then `123` | Decoded run_id is `123` |
| Receipt run_id `999`, then `123` | Aggregate green with outputs |

No original test/support file or frozen expectation changed: inputs.rs,
results.rs, aggregation.rs, writer.rs and support.rs retain their incoming
hashes. The 36 original tests and 21-case aggregate table remain intact.

The new module adds reverse-order controls for those four boundaries. The
final foreign result fails Results; empty models fail Plan; foreign producer
and receipt runs fail ReceiptIdentity after successful decoding. Further
both-order controls cover success/failure outcomes and matching/null results
hashes. The selected failure case includes an older success and requires the
newest attempt to remain selected without fallback. A malformed overwritten
run_attempt object is ignored when a valid string follows; the reverse order
still fails typed conversion. Two field-order tests distinguish insertion
order from sorted or last-occurrence order, with and without duplicate keys.
The two new literal result files add digest-bound aggregate acceptance and
rejection controls.

Static named-test count is now 65: 36 original + 10 duplicate_keys + 5
separators + 14 top_level_duplicates. The target also includes five unchanged
dependency tests outside the focused filter. These are source counts, not
observed execution counts.

## Observed inspection and deferred diagnostics

Read SOURCE_REVIEW.md, CORRECTION.md, all candidate modules and mapped tests,
the complete legacy script/test, and applicable repository/CI guidance.
Inspected installed Serde 1.0.229 and serde_json 1.0.151 source offline for the
remote derive, MapDeserializer and flattened provenance path. No dependency,
toolchain or package was installed.

Read-only SHA-256 checks passed all incoming correction entries before editing
and all FROZEN.sha256 / EXPECTED.sha256 entries after the source edit. `od`
confirmed the unchanged separator files contain literal controls, not escape
text. Git's tracked diff against HEAD is empty because this candidate remains
untracked; the payload manifest, not that empty diff, identifies its changes.
Every Git command used GIT_MASTER=1 and optional index locks were disabled.

`lsp_status` reported zero active clients and an installed Rust server. Rust
file diagnostics were deferred because starting rust-analyzer can invoke Cargo
metadata/check/build scripts, violating the workload owner's exclusive queue.
No clean Rust diagnostics are claimed. A diagnostics request for this Markdown
receipt reported that no LSP server is configured for `.md`; no server was
installed or configured. No Cargo, build, test, formatter, Python,
runtime, network, commit, staging or push command ran. No primary checkout,
root evidence/index, workflow, caller, classification or shared-module write
was made. Live external configuration was not inspected and is not needed for
this code-only decoder correction.

## Pending serial verification

Only the assigned workload validation owner may run these after queue handoff.
Use the candidate root, read docs/design/TESTING.md first, and preserve fresh
logs and actual exit statuses. Missing offline prerequisites remain blockers,
not authorization for downloads. These commands are instructions, not results.

```sh
shasum -a 256 -c tools/xtask/tests/migration_canary_receipts/TOP_LEVEL_CORRECTION.sha256
rustfmt --edition 2024 --config skip_children=true --check tools/xtask/src/automation/canary_receipts/*.rs tools/xtask/tests/migration_canary_receipts.rs tools/xtask/tests/migration_canary_receipts/*.rs
just with-lld cargo test --offline --locked -p xtask --test migration_canary_receipts migration_canary_receipts -- --nocapture --test-threads=1
just with-lld cargo check --offline --locked -p xtask --tests
just with-lld cargo clippy --offline --locked -p xtask --all-targets -- -D warnings
just with-lld cargo test --offline --locked -p xtask --test migration_canary_receipts writer::migration_canary_receipts_full_local_receipt_flow_is_digest_bound -- --exact --nocapture
```

Require 65 candidate tests in the focused run, including the original 36 and
all separator tests; require one test in the exact local receipt-flow run.
Zero selected tests is not success. Run Rust LSP diagnostics only while owning
the queue. Run the exact twelve legacy tests recorded in CANDIDATE.md without
new Python helpers. Both suites passing would still not establish identical-
input differential parity for the new literal fixtures and inline mutations.
Capture that evidence separately and leave all frozen receipts untouched.

In a disposable validation copy, demonstrate that first-wins, blanket
duplicate rejection and sorted-field decoding each fail their corresponding
controls. Retain mutation failures separately. The prior separator mutation
checks and receipt/results exact-limit and limit+1 checks are still pending at
the unchanged 1 MiB / 64 MiB bounds. All integration gates, package admission,
the independent workflow family-job gate and publication restrictions from
CORRECTION.md remain in force. No caller switch is authorized.

## Source self-review

Changed files each retain one responsibility: boundary decoding, identity,
plan projection, receipt representation, result validation or the named test
concern. The ordered adapter is shared by five typed boundaries and has two
parameters. Field order is preserved without adding a dependency or exporting
generic JSON into policy code. No unsafe code, production unwrap/expect,
numeric cast, new fallback, retry, logging, defensive check or negative-form
flag was added. Existing exhaustive variant matches and validation order are
unchanged. Tests distinguish the requested regressions by construction, but
their failure sensitivity awaits execution. Each changed Rust file has at
most 164 nonblank/noncomment lines by the recorded source-count convention.
