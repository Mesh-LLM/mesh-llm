# Queue fold plan — port/patches-11-16 onto the landed decode line

Status: planned, not started. Execute only after `perf/skippy-metal-decode-and-prompt-reuse`
lands on main. Last diffed 2026-11-07 against port `9813d278a` and decode `70ffac666`
(merge-base `2c54e65b9`); both hashes will have moved by the time Card 1 starts.

## Why only four cards

Both patch queues were diffed before this plan was written:

- core `0013` / `0014` / `0016` are identical on both lines
- the whole `model_support` lane (`0001`–`0012`) is identical
- the remaining native deltas are exactly three:
  1. **0015 semantics** — his `0015-skippy-size-recurrent-rollback-for-speculative-verif.patch`
     is the old clamp version; the port carries the reworked
     `0015-skippy-reserve-recurrent-rollback-snapshots-only-for-verifying-sessions.patch`
     (`3742c523a`). Both patches touch an identical file set (the diff of their
     `diff --git` lines is empty), so the rework lands as an in-place update of the
     owning patch — nothing appended, no renumbering.
  2. **0004 default** — the port's `build_rs` `get_state_rows = nullptr` default vs his
     old default: in-place update of owning patch `0004`.
  3. **0017 + 0018 are new on his line only** — trace spans and concurrent sampling.
     Neither has a hit for `recurrent_rollback` / `n_rs_seq` / `verify_rollback`, so
     both replay clean over the reworked `0015`.

## Card 1 — rebranch and replay the queue

- files: `skippy/llama_cpp/patches/0004-*`, `skippy/llama_cpp/patches/0015-*`
- why: the port queue must sit on his landed main as a series of owning patches, not a
  merge; the queue is the source of truth for what the native tree is.
- steps:
  1. rebranch the port successor onto the landed main
  2. take his queue as the base; overwrite owning patch `0004` with the nullptr flip
     and owning patch `0015` with the rework (identical file sets, so the owning-patch
     rule holds: update in place, never append, never renumber)
  3. replay cold: `scripts/prepare-llama.sh pinned`
  4. regenerate the certified-queue roster: recompute `patch_queue_sha256` over the folded
     queue and update `skippy/crates/skippy-api/src/split-certified.json` in the same commit
     (#2907 review: a queue edit without a roster regen fails `split_certification`; the
     checkpoint gate now runs `just skippy-queue-cert-check`, which was the missing local
     catch)
- done-when: the cold replay applies clean, `just skippy-api-all-targets-build` plus
  `just skippy-llama-api-test` are green on the folded tree, and `cargo test -p skippy-api --lib`
  passes — the roster certifies the queue the tree ships
- commit: `chore(skippy): fold the port queue onto the landed decode line`

## Card 2 — regenerate bindings and verify the ABI

- files: generated bindings under `skippy/skippy-api-rs`
- why: the facade rename (`rollback_depth` → `verify_rollback`) is layout-neutral, but the
  fold still carries the decode line's own ABI bumps — it moved `0.1.78 → 0.1.81` before
  `70ffac666`, so the folded `skippy/common.h` will disagree with this line's constants.
- steps:
  1. `just skippy-ffi-bindgen-regen`
  2. update the `skippy-ffi` `ABI_VERSION_*` constants and the roster's `skippy_abi` in
     `skippy/crates/skippy-api/src/split-certified.json` to the folded `skippy/common.h`
     value, together
  3. confirm the constants match the generated bindings (they are const-asserted)
- done-when: skippy-ffi and skippy-runtime lib tests green and
  `cargo test -p skippy-api --lib` green; commit through `just skippy-api-checkpoint`
- commit: `chore(skippy-api-rs): regen ffi bindings on the folded queue`

## Card 3 — re-measure the whole manifest

- files: `skippy/skippy_api/benchmarks/cases.txt`, `skippy/skippy_api/benchmarks/README.md`
- why: `0017`/`0018` touch `llama-context.cpp` and `sampling.cpp` on the staged-open
  path, so every recorded row is stale on the folded tree — a number is only true of
  the tree it was measured on.
- steps: re-run every row of the manifest (`skippy_api_ffi_benchmarks`, 15 interleaved
  reps, `--benchmark_min_time=0.3s`), byte-identical gates first, then record fresh
  numbers and verdicts.
- done-when: every row re-recorded; no row carries a number from the pre-fold tree
- commit: `docs(skippy-api): re-measure the benchmark manifest on the folded tree`

## Card 4 — reconcile the two VerifyRollback types

- files: `skippy/skippy-api-rs/skippy/src/lib.rs`, the skippy-runtime config type (`0018`)
- why: two encodings of one policy now exist — the production skippy-runtime config enum
  and the facade `skippy` crate enum (both `InPlace(2)` default, `None → 0`). Two
  encodings drift.
- steps: one seam-pinning test asserting both agree (tokens, default, `None` mapping),
  or collapse into a single type. Open decision: pin or collapse.
- done-when: one encoding, or a pinned seam that fails the day the two disagree
- commit: `test(skippy-api-rs): pin the verify-rollback seam between runtime config and the facade`

## Standing notes

- His checkout (`/Users/jaredwyles/src/mesh-llm`) is never touched by the fold; all work
  happens in the port worktree.
- Push via ssh is dead (agent failure), so commits accumulate locally until the agent is
  fixed or Jared pushes.
- Positive recurrent coverage (`n_rs_seq == 0` vs `== N` opens, checkpoint-restore
  equivalence) stays fixture-blocked until a recurrent GGUF is local
  (see `skippy/skippy_api/UPSTREAM.md`).
