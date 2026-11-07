# skippy_api

A C++23 façade over the patched llama.cpp stage runtime for the non-chat
workloads — embedding, reranking, encoder prompt, and bounded causal
log-prob readouts — with no chat or generation paths in it.

```
include/            public headers; the only part consumers include
  runtime.h           RuntimeBackend concept, NativeRuntime registry, handles
  types.h             Token / SequenceId / TokenIndex, zero-copy token views
  workload.h          workload classification and pooling
  embed.h             embed() and embed_batch()
  rerank.h            rerank() and rerank_batch()
  encode.h            encode_prompt()
  selected_log_probs.h  bounded selected-token log-probs (patch 0011 pure half)
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

## Patch coverage

What each patch in `skippy/llama_cpp/patches` contributed here. Verdicts live
in `benchmarks/cases.txt`; "—" means nothing ports into the façade.

| patch | what it is | what ported |
|---|---|---|
| 0011 causal token score readouts | unsampled `skippy_iteration_batch_sampled` readout, `skippy_session_reset` | `include/selected_log_probs.h` (pure, 256-bound log-softmax), `llama/causal_readout.h` (`read_selected_log_probs`), `StagedSession::reset`, Rust `StagedSession::read_selected_log_probs` |
| 0012 grammar literal lazy triggers | grammar engine matches literal lazy triggers without compiling a regex | — the grammar matcher is llama-internal; the façade never links or drives it |
| 0013 routed-expert host placement | `skippy_runtime_config.host_expert_layer_start/end`, block-range → tensor-override placement | `include/host_experts.h` (pure range validation, 4096-block cap), `StageSpec::host_experts`, `StagedModel::open` threads the pair, Rust `load_stage_with_host_experts`; plain `Model` loads keep no placement (raw-llama reference path) |
| 0014 temperature-0 argmax fast path | `skippy_sampling_selects_argmax`, `skippy_prepare_argmax_plan`, `skippy_select_unique_argmax` | `include/argmax_sampling.h` (pure eligibility + bias-plan mirror, no sampler linkage), Rust surface intentionally absent - the facade never links a sampler |
| 0015 verify rollback policy | `skippy_runtime_config.verify_rollback_tokens`, `SKIPPY_VERIFY_ROLLBACK_TOKENS_DEFAULT` (2); 0 = no snapshots, verify falls back to checkpoint restore — no clamp, no env override | `include/verify_rollback.h` (`VerifyRollback`: `none()` / `in_place(n)`, `in_place(0)` rejected), `StageSpec::verify_rollback` is required input — `StagedModel::open` refuses a spec without it; Rust `VerifyRollback` enum + `load_stage_with_verify_rollback`, plain bridge lanes state `native_default()` |
| 0016 staged verify windows roll back in place | verify windows rewind recurrent state in place, `skippy_session_verify_rollback_counts` reads the rewind budget | — verify windows are driven by the serving layer's speculative loop, never by the façade; the counts entry point has no façade consumer |

## Checkpoint workflow

```bash
just skippy-api-verify <embed.gguf> <reranker.gguf> <causal.gguf>   # format, tidy, tests
just skippy-api-checkpoint "feat(skippy-api): ..." <embed.gguf> <reranker.gguf> <causal.gguf>
```

`verify` is the gate CI-equivalent for this tree; `checkpoint` runs it, stages
`skippy/skippy_api`, `skippy/skippy-api-rs`, `just/skippy.just`, and
`.gitignore`, then commits with the subject you pass. Both accept the models
as optional arguments — without them the Rust tests run unit-only, and
`SKIPPY_TEST_REQUIRE_MODELS` is left unset so they skip rather than fail.

## Working on this library

See [AGENTS.md](AGENTS.md) for the code rules, the patch-porting procedure
(A/B/budget/ceiling classification), the test split between mocked C++ and
Rust end-to-end, and the traps that have already cost us a day.
