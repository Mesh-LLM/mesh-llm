# Formal verification plan

What each tool would prove, what it cannot, and whether it replaces a property
test or sits beside one. Nothing here is a commitment; it is the order we
would attack in. The one implemented spike is §S at the end.

## 1. GGUF parser bounds — CBMC or ESBMC

- **Scope:** `Reader` (`read`, `skip`, `read_string_view`, `skip_string`),
  `read_tensors`, `read_extents`, `checked_tensor_bytes`, `skip_metadata`.
- **Harness:** `verification/parser_bounds_harness.c`, a C mirror of the
  reader over a symbolic byte array of bounded length (≤ 512) with a symbolic
  GGUF header; asserts "no read past `end_`" and "no wrapping product".
- **Would prove:** for every header up to K tensors and every byte pattern,
  no out-of-bounds read and no 64-bit wrap reaches `stored_length`.
- **Cannot:** anything about real mmap'd files (the harness owns the buffer),
  or type correctness of the enum domains (already typed at rung 1).
- **CI cost:** ~1-3 min per K; runs beside the property tests, replaces none
  (properties cover plausibility, CBMC covers totality to the bound).

## 2. Small pure laws — CBMC harnesses for `split_layers`, `validate_chain`, `StagePlan::tail/view`

- **Harness:** `verification/pure_laws_harness.cpp` (CBMC's C++ subset:
  the three functions extracted or included with `--unwind` bounds).
- **Would prove:** the same laws as the RapidCheck properties, for **all**
  inputs up to bound N (layers ≤ 32, arenas ≤ 64), upgrading "random inputs
  passed" to "all inputs up to the bound".
- **Cannot:** unbounded layer counts; the RCU caches; anything allocation-shaped.
- **CI cost:** seconds each. **Sits beside** the properties: the properties
  are the readable statement of the law, CBMC is the completeness witness to N.

## 3. Rust unsafe — Kani

- **Scope:** `skippy/src/lib.rs` `unsafe` blocks (`tokens_as_native`,
  `tokens_into_native`, `tokens_from_native` — slice-to-pointer-to-slice
  round trips) and the `plan_stages` envelope mapping.
- **Harness:** `#[kani::proof]` fns beside the code (`verification/` notes
  the crate feature).
- **Would prove:** no slice out-of-bounds, no slice-from-raw-parts length lie,
  for all token counts up to a bound.
- **Cannot:** anything crossing cxx into C++ (Kani stops at the FFI edge).
- **CI cost:** Kani is the heaviest tool here (~10+ min cold); nightly-only.
  Replaces nothing; the e2e keeps the behavioural pin.

## 4. Out of scope, and why

- **Through llama/ggml:** the runtime's own semantics; our boundary is the
  plan/loader contract, which the fuzzers and the qualification lanes hold.
- **The RCU `PlanStore` / `ContextPool`:** needs a concurrency model.
  TLA+ (interleavings) or GenMC (C++ memory model) are the later options;
  both are research-sized commitments for a 2-cache design atomics already
  discipline. Revisit if a third cache or a writer thread lands.
- **The cxx bridge:** the seam is generated; proving it means proving cxx.
  The `guarded<T>` in-band pattern plus the e2e suite hold it today.

## S. Spike: `StagePlan::view` clamp, arenas ≤ 64 bytes

- **Harness:** `verification/view_clamp_harness.c` (plain C mirror of the
  clamp: `at = min(offset, size)`, `length = min(length, size - at)`,
  assert `at + length <= size` while touching every viewed byte).
- **Result (2026-10-26, cbmc 6.11.0):** `VERIFICATION SUCCESSFUL` in **0.48 s**
  with `--pointer-check --unwind 66` — all 4 properties hold, no pointer
  failure: every clamped view over every arena ≤ 64 bytes is in bounds.
- **Reads as:** the property (clone views) plus this harness pin the clamp
  for every slice over every arena ≤ 64; the fuzzer holds the same boundary
  from the outside.
