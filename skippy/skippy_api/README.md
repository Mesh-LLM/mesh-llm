# skippy_api

A C++23 façade over the patched llama.cpp stage runtime for the non-chat
workloads — embedding, reranking, and encoder prompt — with no chat or
generation paths in it.

```
include/            public headers; the only part consumers include
  runtime.h           RuntimeBackend concept, NativeRuntime registry, handles
  types.h             Token / SequenceId / TokenIndex, zero-copy token views
  workload.h          workload classification and pooling
  sampling_suppress.h vocab-declared token suppression policy (patches 0012/0013/0014)
  embed.h             embed() and embed_batch()
  rerank.h            rerank() and rerank_batch()
  encode.h            encode_prompt()
  llama/              the only code that includes llama.h
src/                implementations
tests/              mocked unit tests (no llama)
benchmarks/         FFI-vs-façade A/B harness and its report
```

## Rules the code follows

- **C++23, type-safe by default.** Concepts describe what a backend must
  provide; fallible calls return `std::expected`; absence is `std::optional`;
  ids are scoped enums over `int32_t` rather than implicit integers.
- **The core has no backend dependency.** Everything above `llama/` compiles
  without `llama.h`, which is what lets the unit tests run against mocks.
- **Ranges over loops**, and named algorithms over hand-written iteration.
- **Ownership is visible.** `std::unique_ptr` with a named deleter for native
  resources, move-only handles for models and sessions, no owning raw pointers.
  `shared_ptr` is not used; the registry owns, handles borrow.
- **One responsibility per file**, and new behaviour goes into the module that
  owns the capability rather than a shared bucket.

## Building and checking

```bash
just skippy-llama-api-test            # mocked unit tests (ctest prints the count)
just skippy-llama-api-format-check    # clang-format, using .clang-format
just skippy-llama-api-tidy            # clang-tidy, fails on any finding
just skippy-llama-api-ffi-bench-compare <model.gguf>
```

`clang-tidy` is not on `PATH` on every machine; set `SKIPPY_CLANG_TIDY` to its
path (Homebrew: `/opt/homebrew/opt/llvm/bin/clang-tidy`). Three checks are
disabled in `.clang-tidy` on purpose, with the reason recorded there.

Real-model behaviour is covered from the Rust side, not here:

```bash
just skippy-api-rs-e2e <embed.gguf> <reranker.gguf>
```

## Stage planning

`stage_plan.h` owns the typed plan model and its rules; `llama/stage_planner.h`
wraps the native planner and copies nothing until a caller asks. A plan carries
the per-family state annotations as data, so callers never branch on
architecture:

```bash
just skippy-llama-api-planner-probe <model.gguf> [layers] [start] [end]
```

On Qwen2.5-0.5B (24 layers, whole model) that reports 96 state effects, 48 of
them writes, across layers 0-23, `recurrent=false`. Same model, stage `[0,12)`:
48 effects and a frontier of 1 value. BERT-family models report no state
effects at all — they hold no KV — which is the correct answer, not a gap.

## Checkpoint workflow

```bash
just skippy-api-verify <embed.gguf> <reranker.gguf>   # format, tidy, tests
just skippy-api-checkpoint "feat(skippy-api): ..." <embed.gguf> <reranker.gguf>
```

`verify` is the gate CI-equivalent for this tree; `checkpoint` runs it, stages
`skippy/skippy_api`, `skippy/skippy-api-rs`, `just/skippy.just`, and
`.gitignore`, then commits with the subject you pass. Both accept the models
as optional arguments — without them the Rust tests run unit-only, and
`SKIPPY_TEST_REQUIRE_MODELS` is left unset so they skip rather than fail.

## Not covered

Patch 0013/0014's native half — the `LLAMA_TOKEN_NULL`/negative-id rejection
applied at the runtime's own sampling, verification, batch-iteration, and
MTP-propose boundaries — lives in `skippy/llama_cpp/patches` and has no
counterpart here: this library has no sampling, verification, or
speculative-decoding call sites to guard. What is ported are the rules those
call sites apply (`eligible_token` over a selection or a raw id, and the
all-or-nothing `eligible_tokens` batch boundary), for C++ callers composing
`greedy_among_allowed` with a model's declared suppress set. The
Rust surface of the capability is the model accessor (`Model::suppress_tokens`,
patch 0012); the selection and its exhaustion handling stay with the Rust
caller, and crossing the FFI per sampled token would cost more than the check
guards.

## Working on this library

See [AGENTS.md](AGENTS.md) for the code rules, the patch-porting procedure
(A/B/budget/ceiling classification), the test split between mocked C++ and
Rust end-to-end, and the traps that have already cost us a day.
