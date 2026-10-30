# Old C ABI vs skippy_api

`ffi_vs_api_benchmark.cpp` runs the same workload twice in one process, once
through the old C ABI and once through the facade, and reports both under one
google-benchmark report.

| benchmark | old side | new side |
|---|---|---|
| `Embed/N` | `skippy_session_embed` (patch 0010 `decode_embedding`) | `skippy::embed` → `LlamaSession::embed` |
| `Tokenize/N` | `skippy_tokenize` | `LlamaSession::tokenize` |
| `BatchBuild/N` | `llama_batch_init` + fill + free, as the patch does per call | `fill_embedding_batch` into the session's batch |
| `Normalize/N` | the patch's single-accumulator loop | `skippy::detail::normalize_embedding` |
| `SessionCreate` | `skippy_session_create` (claims a lane on the model's context) | `NativeRuntime::create_session` (takes a retained context) |
| `SessionCreateCold` | as above | `create_session` with a geometry the pool cannot serve |
| `ModelLoad` | `skippy_model_open` (model + shared context) | `load_model` (model only) |
| `Rerank` | `skippy_session_rerank` | `skippy::rerank` |
| `BatchEmbed` | B separate `skippy_session_embed` calls | `embed_batch` — B sequences, one decode |
| `RerankBatch` | B separate `skippy_session_rerank` calls | `rerank_batch` — one query, B documents, one decode |
| `stage_chain` | *row pending: the gate compares identical inputs; must compare the C-ABI chain's terminal frame against `StageChain::prefill`'s before timing* | `StageChain` prefill |

Both sides link the *same* patched llama.cpp build, load the same GGUF, and run
with the same geometry (`n_ctx = n_batch = n_ubatch = 512`, one sequence, all
layers offloaded, flash-attention AUTO). The harness embeds one 64-token input
through both paths before timing and refuses to run if the vectors disagree by
more than 1e-4. A configuration drift produces an error in the raw report and
fails the comparison instead of producing a timing verdict.

## Build

```bash
cmake -S skippy/skippy_api -B target/skippy-ffi-vs-api-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSKIPPY_LLAMA_SOURCE_DIR="$PWD/.deps/llama.cpp" \
  -DSKIPPY_API_BUILD_BENCHMARKS=ON \
  -DSKIPPY_API_BUILD_FFI_BENCHMARKS=ON
cmake --build target/skippy-ffi-vs-api-build --target skippy_api_ffi_benchmarks --parallel
```

`SKIPPY_LLAMA_SOURCE_DIR` must be the patched checkout (`skippy/scripts/prepare-llama.sh pinned`),
because the old side calls `skippy_session_embed` out of `libllama.a`. The
`skippy-llama-api-ffi-bench` just recipe does all of this in one step.

## Run

One command, prints the old-vs-new table:

```bash
just skippy-llama-api-ffi-bench-compare ~/models/all-MiniLM-L6-v2.Q4_K_M.gguf \
  --benchmark_filter='^Embed/'
just skippy-llama-api-ffi-bench-compare '' --benchmark_filter='BatchBuild|Normalize'
```

Raw google-benchmark rows, with any extra benchmark flags passed through:

```bash
just skippy-llama-api-ffi-bench ~/models/all-MiniLM-L6-v2.Q4_K_M.gguf
just skippy-llama-api-ffi-bench ~/models/bge-reranker-v2-m3.Q4_K_M.gguf \
  --benchmark_filter='Rerank|SessionCreate'
```

It builds `target/skippy-llama-api-ffi-bench/` first (and prepares the pinned
llama.cpp checkout if `.deps/llama.cpp` is missing), then runs
`skippy_api_ffi_benchmarks` with interleaved repetitions.

To drive the binary yourself (repeat counts, JSON kept on disk):

```bash
SKIPPY_BENCH_MODEL=~/models/all-MiniLM-L6-v2.Q4_K_M.gguf \
  ./target/skippy-llama-api-ffi-bench/skippy_api_ffi_benchmarks \
  --benchmark_min_time=0.5s --benchmark_repetitions=9 \
  --benchmark_enable_random_interleaving=true \
  --benchmark_format=json --benchmark_out=out.json
just automation-run automation skippy-benchmark-compare out.json
```

`BatchBuild` and `Normalize` are llama-free, so they run without a model; the
rest report missing-model errors. Select a filter matching the supplied model:
for example, a reranker GGUF as `SKIPPY_BENCH_MODEL` supports the `Rerank` pair.
The comparison rejects any error row, incomplete pair, or report without paired
samples, even when the benchmark process itself exits successfully. Both sides
rejecting a correctness gate therefore fails the selected case.

The facade's own behaviour is covered by the Rust integration tests, which
need the same two files:

```bash
cd skippy/skippy-api-rs
RUSTC_WRAPPER= \
SKIPPY_TEST_MODEL=~/models/all-MiniLM-L6-v2.Q4_K_M.gguf \
SKIPPY_TEST_RERANKER=~/models/bge-reranker-v2-m3.Q4_K_M.gguf \
  cargo test
```

## Measured on this machine

Apple silicon, Metal + BLAS, `all-MiniLM-L6-v2-Q4_K_M` (22 MB, 384 dims),
median of 9 interleaved repetitions, `--benchmark_min_time=0.2s`:

| benchmark | old FFI | new API | |
|---|---:|---:|---|
| `Embed/8` | 857 µs | 868 µs | tie |
| `Embed/32` | 1.40 ms | 1.41 ms | tie |
| `Embed/128` | 1.95 ms | 1.95 ms | tie |
| `Embed/512` | 3.45 ms | 3.49 ms | tie |
| `Tokenize/64` chars | 2.48 µs | 2.52 µs | tie |
| `Tokenize/512` chars | 17.6 µs | 17.6 µs | tie |
| `Tokenize/4096` chars | 138.3 µs | 138.0 µs | tie |
| `BatchBuild/512` | 6.65 µs | 391 ns | 17.0x |
| `Normalize/384` | 217 ns | 85 ns | 2.6x |
| `Normalize/768` | 435 ns | 165 ns | 2.6x |
| `BatchEmbed/8/8` | 6.78 ms | 1.67 ms | 4.1x |
| `BatchEmbed/8/128` | 15.6 ms | 4.99 ms | 3.1x |
| `BatchEmbed/32/8` | 27.9 ms | 2.18 ms | 12.8x |
| `BatchEmbed/32/128` | 61.9 ms | 25.6 ms | 2.4x |
| `SessionCreate` | 3.3 µs | 2.4 µs | 1.4x |
| `SessionCreateCold` | 3.4 µs | 1.65 ms | old wins ~485x |
| `ModelLoad` | 18.4 ms | 18.7 ms | tie |
| `stage_materialize/6` | 1.864 us | 1.826 us | local parity after correcting the equivalent-field gate |

The corrected materialization measurement is a separate arm64 Release run on
2026-10-31 with MiniLM, 15 interleaved 0.05-second repetitions. FFI/API minima
were 1.633/1.485 us. Its setup gate compares exact strings and state-effect
counts; the timed arms consume the same fields.

`stage_validate` has no timing verdict: the facade checks previously materialized
metadata while native validation additionally checks live graph pointers,
tensor metadata and semantic hashes. Both accepting a valid fixture does not
establish equivalent validation obligations. The pending row must align those
obligations before comparison.

With `bge-reranker-v2-m3-Q4_K_M` (438 MB):

| benchmark | old FFI | new API | |
|---|---:|---:|---|
| `Rerank` (query + document) | 9.47 ms | 9.42 ms | tie |
| `RerankBatch/8` (one query, 8 docs) | 86.6 ms | 36.1 ms | 2.4x |
| `SessionCreate` | 3.5 µs | 2.3 µs | 1.5x |
| `SessionCreateCold` | 3.5 µs | 4.50 ms | old wins ~1 300x |
| `ModelLoad` | 262.9 ms | 263.4 ms | tie |

## Reading the numbers

- **Embed is a tie, by construction.** `llama_decode` is the entire cost; both
  sides make the same call. Differences of a few percent are GPU scheduling
  noise, not interface overhead. Anything larger means the two sides stopped
  being configured the same way.
- **BatchBuild is the interesting one on paper, irrelevant in practice.** The
  facade removes a `llama_batch_init`/`free` pair per call (about 18x on the
  build alone), but that is ~7 µs against a multi-millisecond decode.
- **SessionCreate is where the design differed.** The facade's
  `SessionConfig` builds a dedicated `llama_context`; at 512 tokens that is
  1.4 ms for a 22 MB model and 4.3 ms for the 438 MB reranker. The facade now
  retains released contexts, so a create/release loop costs 0.6 µs like the old
  lane claim, and only a geometry the pool has not seen pays the build.
- **Tokenize is a wash.** Both end up in `llama_tokenize`; the facade reuses
  scratch instead of reallocating, which does not show up at these sizes.

`--benchmark_repetitions` matters: single runs of `Embed/128` move by a few
percent between invocations. `automation skippy-benchmark-compare` reports the
median of the repetitions after converting the samples to nanoseconds.
Google Benchmark's `_min` aggregates remain separate labeled comparison rows;
other aggregate statistics do not enter the repetition median.

## Where the facade is slower

Measured with `--benchmark_enable_random_interleaving=true`, which removes the
order/drift bias of running all FFI instances before all API instances.

| operation | old | new | gap | beyond the noise floor? |
|---|---|---|---|---|
| `SessionCreate`, pool can serve | 3.3 µs | 2.4 µs | 1.4x faster | no (facade ahead) |
| `SessionCreate`, cold geometry | 3.5 µs | 1.65 ms (MiniLM) / 4.50 ms (reranker) | ~485–1 300x | yes |
| `Tokenize`, 64 chars | 2.47 µs | 2.48 µs | +0.4% | no (fixed: scratch re-arm removed) |
| `Tokenize`, 4096 chars | 137.2 µs | 137.4 µs | ~0 | no |
| `Embed` (fit over 1–512 tokens) | 0.890 ms + 4.97 µs/token | 0.894 ms + 4.97 µs/token | +0.4% fixed | no |
| `Rerank` | 9.52 ms | 9.66 ms | −1.4% | no |

The noise floor here is about ±1% per benchmark (min..max over 9 interleaved
repetitions spans 3–8%), which is larger than every difference the refactor
could remove from the call itself.

### The context cost was moved, not added

The old `skippy_model_open` builds the model *and* the one context every
session then shares; the facade's `load_model` builds only the model and leaves
the context to `create_session`. `ModelLoad` measures exactly that:

| | old open (model + context) | new load before the prewarm | new load, prewarmed |
|---|---:|---:|---:|
| MiniLM | 18.4 ms | 16.4 ms | 18.7 ms |
| reranker | 262.9 ms | 260.1 ms | 263.4 ms |

Before the prewarm the facade loaded 2.3 ms (MiniLM) faster, which was exactly
the context work the old open did up front. Nothing was added; the boundary
moved — to the wrong place for a server that opens a session per request. The
prewarm now builds the default geometry at load again, so the two are level and
the first request is cheap. That is a deliberate trade, not a regression: the
same context is built either way, just at a predictable moment.

### The patch: retained contexts, prewarmed load, cached model facts

Three changes in `skippy_api/src/llama/backend.cpp`:

1. **A released session hands its context back.** `LlamaModel` keeps up to two
   idle contexts keyed by `max_tokens`; `create_session` takes a matching one
   before building. `LlamaSession`'s destructor does the handover, so nothing
   changes for callers and no API surface moved. Reuse is safe because
   `process_tokens` clears the K/V memory before every decode.
2. **Model facts are derived once at load** (`LlamaModel::Traits`: declared
   pooling, classifier count, rerank prompt pieces, decoder start). Sessions
   used to re-read `general.architecture`/`<arch>.pooling_type` from GGUF
   metadata and re-assemble the rerank prompt on every create.
3. **The default context is built at load**, so the first request never pays
   the build. That moves ~1.5 ms (MiniLM) / ~4 ms (reranker) back into
   `ModelLoad`, which is why that row is now a tie instead of a facade win:
   the total is the same as the old open, and the first request is fast.

Result, in an isolated process (median of 20):

| | old | new, before the patch | new, after |
|---|---:|---:|---:|
| `SessionCreate` (pool can serve) | 0.63 µs | 1 456 µs | 0.60 µs |
| `SessionCreateCold` (MiniLM) | 3.5 µs | 1 457 µs | 1 457 µs |
| `SessionCreateCold` (reranker) | 3.5 µs | 4 282 µs | 4 337 µs |

The cold row is the honest remainder: a geometry the pool has not seen yet
still costs a full `llama_init_from_model`. One context is pre-built at load
and up to two are retained. A geometry reuses its context while that context
remains cached; eviction can require another build.

`skippy-api-rs/skippy/tests/integration.rs::released_sessions_do_not_leak_previous_state`
covers the reuse path: eight create/embed/release cycles must reproduce the
first session's vectors exactly.

### What the patch cannot fix

The K/V allocation and graph reservation inside `llama_init_from_model`
(0.73 µs/token and 1.1 ms fixed for MiniLM). A pool only avoids repeating them,
and batching only amortises them across inputs. Sharing one context between
concurrent sessions, as patch 0010 does, is the only thing that removes them
from the second session onward, and it costs a serialisation lock and a shared
sequence budget.

### Why tokenize was marginally slower, and embed is not slower at all

`LlamaSession::tokenize` used to re-arm its scratch buffer with
`native_tokens_.resize(std::max(capacity(), 1))` before every call, which
value-initialises the whole capacity (a few KB after one long document). That
was the entire ~1% gap at 64 characters; the buffer now stays at capacity and
tokenize reads 1.00x. What remains is the ~20 ns `with_session` borrow — 1.2%
of the reranker's 1.6 µs tokenize, invisible at 110 µs. The old entry point
pays `strlen(text)` on every call, which is why neither side pulls ahead.

`Embed` has no such gap because the refactor removes 6.5 µs of batch
allocation and ~0.1 µs of normalisation from a call that spends ~890 µs in
`llama_decode` — 0.2% at 512 tokens, 0.02% at 8. The fit says the facade's
fixed cost is 3.5 µs higher and its per-token cost identical, so the two
effects cancel inside the noise floor.

## Batching: where the facade pulls ahead

The old ABI decodes one text per call and pays the decode's fixed term every
time. `embed_batch` and `rerank_batch` decode B sequences in one native pass:

```cpp
auto session = model.create_session({.max_tokens = 512, .sequences = 8});  // context holds 8 x 512
std::vector<float> output(8 * dimensions);
skippy::embed_batch(runtime, session, groups, {.values = output});
```

Measured (`BatchEmbed`, same tokens, same model):

| B × tokens | old: B calls | `embed_batch` | speedup |
|---|---:|---:|---:|
| 8 × 8 | 6.78 ms | 1.67 ms | **4.07x** |
| 8 × 128 | 15.56 ms | 4.99 ms | **3.12x** |
| 32 × 8 | 27.93 ms | 2.18 ms | **12.79x** |
| 32 × 128 | 61.92 ms | 25.58 ms | **2.42x** |

and for reranking (`RerankBatch`, one query against eight ~200-character
documents):

| | old: 8 calls | `rerank_batch` | speedup |
|---|---:|---:|---:|
| 1 query × 8 documents | 86.6 ms | 36.1 ms | **2.40x** |

The gain is the fixed term amortised over the group, so it grows as inputs get
shorter and as the batch grows. Both entry points are pure throughput: a
single request is unchanged.

Correctness: `BatchEmbed` embeds one row both ways and refuses to time a
mismatch; `batch_beyond_the_session_sequence_budget_is_rejected` and
`batched_embeddings_match_single_embeddings` cover the Rust surface (rows agree
to ~1e-4, which is the kernel-path difference, not a pooling error).

## Outliers worth knowing

From a 9-repetition interleaved run on both models:

| outlier | size | cause |
|---|---|---|
| `SessionCreateCold` | 1.6 ms (MiniLM), 4.4 ms (reranker) | a geometry the pool has not seen forces `llama_init_from_model` |
| `SessionCreate`, a second geometry | 1.4–2.6 ms | the load-time prewarm covers one context only |
| `ModelLoad`, 438 MB | ±5%, sign flips between runs | page-cache and disk, not the interface (265–275 ms, ~1.7 GB/s) |
| `Tokenize`, 64 chars | +20 ns for the facade | `with_session` borrow; 1.2% of the reranker's small-text tokenize |
| `Embed/1..2` | median moves ±3–4% between runs | fixed-cost dominated; ranges overlap |

Everything else stays inside the ±1% noise floor.

## What this benchmark found

The first run showed the facade opening a reranker as `kind=0` (causal),
`pooling=0` (none), `classifier_outputs=0`, so `skippy::rerank` returned
`UnsupportedWorkload` while the old side scored it fine.

Patch 0010 treats a model with a classifier head as a reranker and forces rank
pooling for it. `LlamaBackend::create_session` only filled
`classifier_outputs` *after* pooling already read as rank, so a GGUF that
records no pooling but ships classifier labels never got there. The fix reads
the model's declared pooling from metadata (`<arch>.pooling_type`), and:

- a model that declares its pooling keeps it — a mean-pooled embedding model
  with classifier labels stays an embedder;
- a model that declares none and ships classifier labels is a reranker, so the
  one context is built with rank pooling.

Both readings are visible in the harness's first stderr line, and
`skippy-api-rs/skippy/tests/integration.rs` covers both models
(`SKIPPY_TEST_MODEL`, `SKIPPY_TEST_RERANKER`; 13/13 pass).

## Not covered

`Encode` needs an encoder-decoder GGUF, and the Rust-side overhead comparison
(`skippy-ffi` vs `skippy-api-rs` in one binary) needs a packaged native runtime
to dlopen. Neither is in this suite yet.

The retained-context pool is an internal policy with a fixed bound of two.
There is no `SessionConfig` knob for it; a caller that would rather pay the
milliseconds than hold the memory has no way to say so.
