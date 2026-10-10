# Current correction status, 2026-09-28

Code-ready for queued validation only. [CORRECTION.md](CORRECTION.md) records
the current contract, added controls, pending checks and correction hashes.
Nested object keys retain the last value as in legacy; the original statement
below claiming approved rejection of duplicate known keys was too broad.
Top-level typed rejection is unchanged, differs from legacy, and is not newly
approved here. The original 36 tests remain intact; 15 added controls bring the
candidate count to 51, all still unexecuted in this correction. Original
formatter observations below do not cover the correction.

The preparation record below is retained verbatim as history. Its original
SHA-256 is `d729d85d2036cf10eb5e086f654fb4fdf5d9ec4c4a7141557d47cf8a65344ea3`.
Use the correction's validation instructions rather than the historical count
or the original manifest as an identity of the corrected source.

# Canary receipt candidate, 2026-09-28

Status: source and test preparation only. Compilation, tests, diagnostics,
executable legacy captures, manual execution and independent review are PENDING.
No global Cargo queue slot was assigned or acquired. Task 24 is not complete.

## Isolation

Candidate worktree:
`/var/folders/5q/y9dmlwq11tqd74j_17t5p5ym0000gn/T/opencode/wt-task24-canary-receipts`.
Base commit: `a4e04070db2c6b8e2644a40f5c4f207789df6c2d`.
Its Git common directory belongs to a separate local shared clone at
`/var/folders/5q/y9dmlwq11tqd74j_17t5p5ym0000gn/T/opencode/task24-canary-receipts-repo`.
This avoids registering a worktree inside the read-only root's `.git` directory.
The clone uses the root object store read-only. There was no network access.
No dirty prerequisites were copied: the complete dependency closure needed by
this candidate is already in HEAD. Do not copy the root's concurrently edited
Cargo manifest, lockfile, dispatcher, process modules or other automation owners.

Root guidance and current migration plan were read, never copied or edited.
Plan SHA-256: `6560dfbed1881ee87c16fc41281d4fffa4723f3d731d21ccd55ca2336d27288d`.
Spec SHA-256: `9fb47f561fb69e94067d133813d5874202eda53f7b39aa1ee66065a27f1420d0`.
The legacy script/test identities are in `fixtures/PROVENANCE.md`.
The reused committed string writer `tools/xtask/src/ci_plan/plan_bytes.rs` hashes
to `ce47c5b8a75f1f5fe1a990e47407b0b307a5de5923c67f2a3ebf2d91ea638754`.

## Candidate scope

Nine source modules under `tools/xtask/src/automation/canary_receipts/` own the
typed input projections, result-stream decoding, receipt writing, newest-attempt
selection, result hash checks and aggregate diagnostic report. The only generic
JSON values are boundary-local object/type/truth projections; typed values enter
the validation algorithms. No new dependencies, package verifier, planner,
archive handling, workflow adapter or CLI was added.

The aggregate validates provenance and attempt bounds for every receipt, rejects
same-attempt duplicates, then chooses the greatest numeric worker attempt. It
checks only that selected receipt's outcome and results. A selected failure,
cancelled/skipped worker, malformed payload, absent results or changed hash
cannot fall back. Old corrupt result files and old malformed outcome/hash fields
are ignored when superseded, as in legacy; old provenance failures still fail
the aggregate. Run attempts compare decimal strings exactly without machine
integer overflow. Every planned family's required lanes come from the explicit
source plan, including MTP and workload oracle lanes. Global battery preflight,
consolidated causal/workload rows and causal projector cardinality are retained.

Worker receipt output remains unversioned. Producer identity schema 3 is not
copied into receipts. The writer reuses the existing ASCII string encoder and
locally escapes DEL to match Python ensure_ascii; no shared encoder was changed.
It hashes raw results bytes and writes sorted keys plus a final newline. As in
legacy, writing targets only the supplied current evidence directory's receipt;
the aggregate never writes inputs. Historical root evidence and all pre-existing
fixtures were left unchanged.

The fixture inputs/expected verdicts were frozen before source implementation.
`FROZEN.sha256` and `EXPECTED.sha256` bind those original files.
They are source-derived, not executed oracle output. The dedicated target has
36 named candidate tests, including a 21-case aggregation fixture table. This
is a static source count, NOT an executed test count. The target includes the
existing string writer's additional test, which the focused filter excludes.

## Boundaries requiring review

Malformed-input diagnostics are typed and need not reproduce Python tracebacks.
The approved typed-boundary policy is used for malformed nested arrays,
non-string outcome/class/runner fields, duplicate known keys and invalid hashes.
Valid legacy receipts have scalar strings and object-shaped rows/outcomes.
UTF-8 JSON is supported; lone surrogate escapes and Python's nonstandard NaN/
Infinity inputs are not represented as valid typed producer data. No claim of
full malformed-input differential parity is made.

Receipt reads are capped at 1 MiB and results at 64 MiB. These are explicit new
safe resource bounds under the recorded maintainer policy, not measured maxima
or legacy limits. Boundary qualification remains pending. The source plan is
borrowed input bytes and is never regenerated. Receipt directory traversal stays
one level deep, sorted by path like the legacy glob. Regular-file checks reject
obvious special files, but this is not a race-resistant hostile-filesystem
sandbox: supplied directories must be quiescent, trusted extracted artifacts.
Symlink containment and package authenticity belong to the upstream owner.

`VerifiedPackageInputs` names a caller obligation, not a cryptographic proof.
No completed Rust package-admission owner exists in the inspected root or family
planner worktree. Until one verifies the full immutable closure, this API must
not be exposed as a production CLI over untrusted identity JSON.

## Observed non-executing checks

`rustfmt --edition 2024 --config skip_children=true` completed successfully on
candidate Rust files. The matching `--check` returned 0 on the final candidate,
including the object-boundary addition.
Static inspection traces legacy receipt/read_json_documents/validate_results/
aggregate, the existing tests, producer verification and the workflow's external
family-job success gate. No package/model/certification process was launched.
Git comparison against HEAD found no change to manifests, lockfile, dispatchers,
the reused string writer or legacy script/tests. Root reads rehashed the same
legacy inputs, string writer and current plan. This is not root-wide exclusivity
proof because other authorized agents are editing it concurrently.

LSP diagnostics were deliberately not requested: rust-analyzer may invoke Cargo
and violate the assigned global queue. No Cargo metadata, fmt, check, build,
Clippy or test command ran. No tests, legacy CLI commands, Python helpers,
network operations, commits, staging, pushes or root evidence/index edits ran.

## Serial validation handoff

Only execute after an explicit global queue assignment. Use fresh candidate-local
log destinations and preserve every status. Never overwrite historical evidence.

1. Compare final `CANDIDATE.sha256` and frozen fixture hashes. Read
   `REGISTRATION.md`; no shared wiring is needed for the auto-discovered test.
2. Read `docs/design/TESTING.md`. Run
   `cargo test --locked -p xtask --test migration_canary_receipts migration_canary_receipts -- --nocapture`.
   Require exactly 36 named candidate tests to execute, plus the recorded
   21-case table within its test. Zero selected tests fails qualification.
3. Run `cargo check --locked -p xtask --tests`, then
   `cargo clippy --locked -p xtask --all-targets -- -D warnings`, serially.
   No clean-diagnostics assertion exists until these exit successfully.
4. Capture the existing legacy tests, without new Python code:
   `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest scripts.tests.test_llama_canary_family_evidence.FamilyEvidenceTests.test_complete_distributed_pass scripts.tests.test_llama_canary_family_evidence.FamilyEvidenceTests.test_partial_rerun_reuses_build_and_successful_sibling scripts.tests.test_llama_canary_family_evidence.FamilyEvidenceTests.test_aggregate_only_rerun_reuses_complete_prior_pass scripts.tests.test_llama_canary_family_evidence.FamilyEvidenceTests.test_newer_failure_never_falls_back_to_old_success scripts.tests.test_llama_canary_family_evidence.FamilyEvidenceTests.test_newer_corrupt_results_never_fall_back scripts.tests.test_llama_canary_family_evidence.FamilyEvidenceTests.test_worker_provenance_must_be_valid scripts.tests.test_llama_canary_family_evidence.FamilyEvidenceTests.test_duplicate_worker_cannot_pass scripts.tests.test_llama_canary_family_evidence.FamilyEvidenceTests.test_missing_worker_cannot_pass scripts.tests.test_llama_canary_family_evidence.FamilyEvidenceTests.test_reports_all_failed_workers_without_emitting_green scripts.tests.test_llama_canary_family_evidence.FamilyEvidenceTests.test_workload_family_requires_its_class_lanes scripts.tests.test_llama_canary_family_evidence.FamilyEvidenceTests.test_global_battery_preflight_is_validated_separately scripts.tests.test_llama_canary_family_evidence.FamilyEvidenceTests.test_mtp_family_cannot_certify_without_its_all_head_lane`.
   These construct local fixture packages, not pack/restore/build/certify runs.
   Missing legacy PyYAML/import prerequisites must be recorded, not installed.
5. Those legacy tests do not consume the new exact fixture files. Exact fixture
   differential captures remain a separate pending gate, not established by
   both test suites passing. Freeze their observed output/status in NEW files
   before any caller cutover; do not rewrite the source-derived expectations.
6. The matching local library QA is
   `cargo test --locked -p xtask --test migration_canary_receipts writer::migration_canary_receipts_full_local_receipt_flow_is_digest_bound -- --exact --nocapture`.
   Exercise the missing-lane/digest failures in the same dedicated target.
7. After integration, run the owner-required xtask regression/formatter/
   no-console-print/CI union serially. Request LSP only while owning the queue.
   No product build or native/model proof is supplied by this bounded library.

## Self-review

Source responsibilities are separated by identity, plan projection, boundary
adaptation, storage, result validation, receipt writing, aggregation and errors.
There is no unsafe, non-test unwrap/expect, numeric cast, new dependency or
untyped domain payload. Owned variants use exhaustive matches. New helpers are
used at multiple boundaries or separate substantive validation phases.
The test-only fixture constructor's four inputs describe distinct directory,
family, worker attempt and outcome coordinates; keeping those explicit makes
the duplicate/rerun cases reviewable. No logging or exception-swallowing layer
was added. Tests are prepared but their ability to fail is not yet executed.
All source files are below 200 nonblank/noncomment lines; aggregation tests and
result tests are in the 200-250 warning band and should split by test concern
before more cases are added. No completion or parity approval is claimed.
