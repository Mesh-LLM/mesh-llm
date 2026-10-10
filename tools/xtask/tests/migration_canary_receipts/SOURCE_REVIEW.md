# Canary receipts source review

Date: 2026-09-28. Verdict: corrections and queued qualification required.
This is source review, not compilation, test execution, or differential capture.
The global Cargo queue remains with bg_fda1665b. No source, caller, policy,
limit, root evidence, or root index was changed by this review.

## Reviewed identity and scope

Read CANDIDATE.md, REGISTRATION.md, all nine source modules, the dedicated test
entry, all five test/support modules, all fixture bodies, and PROVENANCE.md.
Read the complete unchanged legacy script and test module, plus the family-pass
workflow and relevant workload battery producer code.

The candidate manifest's 29 entries passed SHA-256 verification. FROZEN.sha256
and EXPECTED.sha256 also passed. These checks establish byte identity, not
behavior. The 36 named tests and 21-case table are still unexecuted here.

Legacy source matches both this candidate and root:

```text
89dc3f313c95482e7022b58d7a274eb724e18eb4aca20ae9a8907ae45245e5c2 scripts/llama-canary-family-evidence.py
280bb2ac4b504713da600b2d11cb86ec43bf970fe6826892eef16372f63c0f05 scripts/tests/test_llama_canary_family_evidence.py
```

## Findings

### C1: nested duplicate handling contradicts the declared typed boundary

Priority: P2. Source: boundary.rs:6-21, results.rs:14-23, plan.rs:18-24;
CANDIDATE.md:64-67.

Both object adapters deserialize through serde_json::Map before constructing
the typed object. Duplicate keys have already been overwritten by then, so
Serde's duplicate-field rejection cannot see them. Direct root structs reject
duplicates, while nested lane/model/matrix objects silently use the last value.

Reproducer for the validation owner: start from fixtures/dense.jsonl and replace
the chain outcome with this literal JSON, leaving the other two outcomes intact:

```json
{"name":"chain","status":"fail","status":"pass","exit_code":0}
```

The source path admits this row. For aggregation, recompute the receipt's raw
results digest so only the duplicate policy is tested. A duplicate known model
field inside selected_models has the same reduction problem.

This is not a legacy-parity finding: Python also keeps the last duplicate value.
It is an inconsistency with the candidate's declared malformed-input boundary.
Assign the canary owner to enforce the already selected duplicate policy before
map reduction and add literal-byte nested controls. Do not silently choose a
different policy or rewrite historical fixture expectations.

### C2: result-stream whitespace no longer follows the explicit Python contract

Priority: P2, narrow valid legacy-stream acceptance difference. Source:
results.rs:114 versus legacy lines 504-505.

Reproducer: prefix or suffix the unchanged fixtures/dense.jsonl bytes with one
raw byte 0x1c, 0x1d, 0x1e, or 0x1f. Python's explicit str.isspace loop consumes
each; Rust char::is_whitespace does not. Python proceeds to a passing result;
the candidate returns ErrorKind::Json before the object or on the suffix.
These separators are outside objects, not unescaped JSON string controls.

Assign the canary owner to reuse repository::python_text::is_space, which
already includes precisely these four C0 separators. Keep the dedicated test
target's dependency wiring explicit. Add four finite byte controls; no general
JSON emulator or parser replacement is needed.

## Checks without additional findings

- All receipts undergo provenance and attempt-bound validation before selection.
  Same-family/same-attempt duplicates fail even when an attempt is superseded.
- The greatest numeric decimal attempt wins. Only its outcome/results are
  checked; an old malformed payload is ignored, but old foreign provenance
  still prevents green. A syntactically broken newer receipt records an error
  even if an older valid receipt remains selected.
- Required per-model lanes, exactly one consolidated row, battery preflight,
  workload class, causal projector cardinality, and raw-byte result digests
  follow the inspected legacy algorithm for ordinary typed inputs.
- Typed malformed diagnostics can precede legacy semantic errors. For example,
  a failed outcome with an invalid hash may produce Json before WorkerOutcome.
  This is within the declared malformed-field boundary, not a valid-input
  acceptance defect. Keep failure-category claims separate from traceback parity.
- The 1 MiB receipt and 64 MiB result caps remain unchanged. They are explicit
  new bounds, not measured producer maxima. Existing tests cover only receipt
  limit+1; the validator still needs exact-limit and limit+1 cases for both paths.
- VerifiedPackageInputs is freely constructible, not proof of package admission.
  This is correctly disclosed. Receipt success cannot authenticate schema,
  controller/source, executable/plan closure, or authorize publication. The
  workflow's independent family-result gate remains mandatory.

## Next validation commands

Only the assigned validation owner may execute these, serially, in this
candidate or an explicitly prepared validation copy. Read docs/design/TESTING.md
first. Keep fresh logs and record actual exit statuses and selected test counts.

```sh
shasum -a 256 -c tools/xtask/tests/migration_canary_receipts/CANDIDATE.sha256
just with-lld cargo test --offline --locked -p xtask --test migration_canary_receipts migration_canary_receipts -- --nocapture --test-threads=1
just with-lld cargo check --offline --locked -p xtask --tests
just with-lld cargo clippy --offline --locked -p xtask --all-targets -- -D warnings
just with-lld cargo test --offline --locked -p xtask --test migration_canary_receipts writer::migration_canary_receipts_full_local_receipt_flow_is_digest_bound -- --exact --nocapture
```

Run the exact twelve existing Python test names in CANDIDATE.md:119 using
PYTHONDONTWRITEBYTECODE=1. They do not invoke real build/certify/restore work.
Missing import prerequisites are a blocker, not permission to install tools.
Their success is not a differential capture of the new fixtures. C1/C2 and the
bounded size controls need fresh identical-input observations before cutover.
After corrections, keep the original candidate hash manifest as historical
identity and record a new correction manifest rather than relabelling this review.
