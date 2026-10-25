# Independent decisions from causal prefill

Supported loaded causal GGUF decoders serve the existing `/systemone` and
`/v1/decisions` contracts. Mesh advertises `system_one` in its
`/v1/models` response after probing the loaded backend. Standalone Skippy and
Mesh's `--local-model-only` mode do not expose that capability field; their
request handler checks the loaded model directly. A model name alone does not
establish support.
Each question receives the same state and its own instructions and options.
Answers to earlier questions are never inserted into later prompts.

```sh
curl -sS http://127.0.0.1:9337/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "YOUR_DISCOVERED_CAUSAL_MODEL_ID",
    "state": "The customer was charged twice and requests a refund.",
    "questions": {
      "team": {"type": "choice", "instructions": "Choose the responsible team.",
        "criteria": {"B": "Billing and refunds", "S": "Shipping and delivery"}},
      "priority": {"type": "choice", "instructions": "Choose a priority.",
        "criteria": {"H": "Urgent action needed", "N": "Normal queue"}}
    }
  }'
```

## Methods and score semantics

The actual loaded tokenizer validates every label after the exact prompt.
If all labels are one token, `selected_token_logits` reads their last-prefill
scores without sampling or decoding an answer. Otherwise,
`candidate_log_likelihood` teacher-forces each candidate independently and sums
its conditional token log probabilities. There is no length normalization;
longer labels can therefore be disadvantaged. Labels that change the prompt
token boundary or tokenize identically are rejected.

Prefer **single-token answer labels** such as `A` and `B`, with the full meaning
in each option's description, especially when natural-language labels have
different lengths. Check `candidate_token_counts` for the loaded tokenizer:
even a short label is not guaranteed to be one token. Summed log probabilities
include a non-positive term for every token, so a longer label can rank lower
because of its length. Do not interpret a mixed-length ranking as a length-fair
comparison of the described answers. No mean-per-token correction is applied.

The causal prompt has a fixed shared-state instruction, JSON state, a JSON
question with descriptions and ordered options, and an answer-label suffix.
It does not reuse a conversational history or render a model-specific chat
template. Support for this computation does not establish classification
quality for a model trained on a different format.

`/systemone` adds `readouts`, keyed by question. `/v1/decisions` adds `readout`
to each answer, preserving the original question array order. Entries contain:

| Field | Meaning |
| --- | --- |
| `method` | `selected_token_logits` or `candidate_log_likelihood` |
| `normalization` | `sum_log_probabilities`, with no length penalty |
| `probability_semantics` | `relative_candidate_probability_not_calibrated_truth` |
| `candidate_log_probabilities` | Full-vocabulary conditional log probabilities, summed for a multi-token label |
| `candidate_token_counts` | Actual label token lengths in prompt context |
| `evaluated_tokens`, `cached_tokens` | This question's execution and reuse, summed over independent candidate branches |

Candidate arrays retain input option order. Returned probabilities are softmax
over the offered scores, **not calibrated probability of truth**. Wording,
option order, label tokenization, quantization, and execution batch shape can
affect scores. The existing API field named `confidence` is retained for
compatibility and measures concentration as `1 - H(p) / ln(N)`, where `H(p)` is
the entropy of the offered-candidate distribution and `N` is its size. It is
not an estimate of correctness. Validate any threshold on your own task. An
incorrect candidate set can still receive a high value.

## Scheduling, reuse and accounting

Independent sequences use the normal scheduler, request admission, token budget,
and cancellation. With prefix caching enabled, exact shared tokens are prepared
once and reused. Multi-token branches also reuse their question prefix. Misses
recompute; the final prompt row always runs because KV state does not include
cached logits. An incomplete split restore resets the whole sequence before
recomputation. Abandoned or failed requests release local sessions and discard
incomplete downstream connections. A non-cancellation preparation failure is
logged and scoring proceeds with fresh sessions, bypassing prefix lookup and
recording for that request. Other requests retain their cache settings.
Cancellation still drains admitted preparation and terminates the request.

Top-level `readout_usage` reports physical `evaluated_tokens`, `cached_tokens`,
and `cache_preparation_tokens`. Preparation is included in evaluated work;
per-question counts exclude preparation. Completed preparation steps are counted
even if a later preparation step fails. `usage.input_tokens` is evaluated plus
cached inputs across preparation and all independent candidate branches;
`/v1/decisions` reports the same reuse in `usage.input_tokens_details.cached_tokens`.
It can exceed the sum of logical question prompts, since multi-token candidates
have separate executions. `output_tokens` remains zero. Cache snapshot writes
are not measured as tokens, so `cache_write_tokens` remains zero. Physical counts describe
tokens at each stage, not tokens multiplied by the number of split workers.
They are not elapsed-time, FLOP, energy or billing measurements.

## Limits and compatibility

The causal path accepts 1–64 questions, 2–26 choice labels or 2–10 score levels,
at most 64 tokens per candidate, and at most 32,768 prompt tokens per question.
The loaded context can impose a smaller limit and must fit the longest
teacher-forced continuation. Images, thinking, sequential reads, and multiple
samples or steps remain unsupported. Existing Laya and DiffusionGemma adapters
retain their limits and response shape, without causal evidence fields. The
public Decisions usage reports zero cached inputs for those adapters.

Only loaded causal decoders are admitted; encoder-only, encoder-decoder and
diffusion models do not enter this path. The runtime validates execution
frontiers. Unsupported frontiers fail explicitly, without a chat-generation
fallback.

In split execution, the terminal worker computes scores and returns only a
bounded selected-token vector. Full vocabulary logits remain inside the native
runtime. All workers must support stage protocol generation 12, wire header
version 17 and Skippy ABI 0.1.74. Upgrade the whole chain together; older stage
peers fail negotiation. Additive `system_one` discovery does not grant this
ability to an older runtime. No persisted-data migration is required.

## Model-backed validation

Tests that require a real GGUF are ignored by ordinary unit runs and must be
invoked explicitly. Set `SKIPPY_DECISION_MODEL` to the verified model path.
The runtime reference uses the existing constrained sampler's raw logprob
reader; it does not reuse the selected-logit normalization implementation.
Serving tests exercise the HTTP router and a real loopback binary stage chain.
Missing model files fail these tests, rather than producing a passing skip.

Native batch execution drains pending backend work unconditionally after decode,
before reading host logits or copying activation exports. This includes terminal
readout-only batches, with no sampler and no activation export. The fence is
owned by `src/skippy/execution-batch.cpp` in core patch `0002`; the selected-logit
read in patch `0011` relies on it. Reading raw host logits without that fence
would be unsafe on asynchronous backends. CPU execution evidence does not
replace a separate CUDA run.

`causal_readout_reused_host_rows_match_synchronized_reference` alternates 64
readout-only requests with distinct prompts and compares each result with the
existing synchronized sampler's raw-logprob reader. It defaults to CPU. Set
`SKIPPY_DECISION_TEST_DEVICE=CUDA0` (or the exact advertised device name, such
as `MTL0` for Metal) and
load a matching native bundle to exercise GPU readback; an unavailable device
fails instead of falling back to CPU. Other parity fixtures retain their fixed
CPU arithmetic configuration.

```sh
just with-lld cargo test -p skippy-runtime \
  causal_readout_matches_constrained_reference_and_warm_batched_prefixes \
  -- --ignored
SKIPPY_DECISION_BENCHMARK_REPORT=/tmp/decision-benchmark.json \
just with-lld cargo test -p skippy-serving causal_decisions \
  -- --ignored --test-threads=1
```

Use the same configured backend and native build for the model and test binary.
Optional `SKIPPY_DECISION_REPORT` and `SKIPPY_DECISION_SPLIT_REPORT` file paths
retain the serving test's responses and cold/warm timings. Timing a small
fixture establishes execution evidence, not model-quality certification or a
general throughput advantage.

The benchmark runs one and eight independent questions through the HTTP router,
locally and over two planner-produced stages, with caching disabled and enabled.
It compares the readout with greedy one-token chat completions using an operator
template that renders only the supplied prompt, and a native grammar allowing
only the offered labels. The fixture verifies identical prompt tokens and that
the completion emits an offered label. Each method gets a fresh model/cache, one cold
request and three repetitions. Model loading is outside the timed region.
Preserve the report, model SHA-256, native build,
backend and machine configuration together. Run timings on an otherwise idle
host; they are measurements of that configuration, not a speed guarantee.

The strict native batch/cache reference uses F32 KV without flash attention.
A second reference compares F16-KV readouts with the existing sampler's raw
logprob reader at chunk sizes 1, 32, 128 and 1024. Both paths can change scores
when their chunk shape changes: exact token-prefix reuse does not imply
bit-identical floating-point results across matrix kernels.

The strict HTTP cache and split parity fixtures use ordinary F16 KV with a
single-token native batch and flash attention, keeping matrix dispatch and
attention accumulation comparable across cache layouts. Separate HTTP
ordering, scheduler batching, split-miss and benchmark fixtures use 128-token
batches. These checks establish execution correctness, not classification
accuracy or calibrated confidence.

## Local benchmark observation

Measured on an Apple M1 Max with 64 GiB RAM, CPU execution and four lanes,
using SmolLM2-135M-Instruct Q8_0. The source is
`unsloth/SmolLM2-135M-Instruct-GGUF` at revision
`9e6855bc4be717fca1ef21360a1db4b29d5c559a`; the GGUF SHA-256 is
`c4a3dd037301b6ecea31d6da37f5cd793ead920dd5ddfe6d589294628d6ce66a`.
The native source tree is `e764bce6bac1c43c2ec8b0c478499f8c2c849ed2`,
prepared from upstream `620d2225cc436ab0b001e3200e590e3b286ef9ab`.
Split execution uses two stages on the same machine over loopback TCP.

Each cell is **cold request / median of three repeats**, in milliseconds.
Model loading is excluded. With caching disabled, the repeats also recompute
their prompts. The baseline is real grammar-constrained one-token generation
with the same prompt tokens, not a score-only or logit-bias substitute.

| Route | Questions | Prefix cache | Prefill readout (ms) | Constrained generation (ms) |
| --- | ---: | --- | ---: | ---: |
| Local | 1 | Off | 252 / 230 | 325 / 176.3 |
| Local | 1 | On | 202 / 5.3 | 170 / 11.8 |
| Local | 8 | Off | 1099 / 1141.5 | 1438 / 1332.6 |
| Local | 8 | On | 360 / 30 | 482 / 64.6 |
| Two stages | 1 | Off | 476 / 265.3 | 656 / 557.9 |
| Two stages | 1 | On | 242 / 14 | 501 / 301.3 |
| Two stages | 8 | Off | 2760 / 3090.2 | 4717 / 6017.6 |
| Two stages | 8 | On | 549 / 150.1 | 3664 / 581.3 |

The benchmark fixture records questions per second as well as latency.
For eight cached questions, median throughput was 266.9 versus 123.8
questions/s locally, and 53.3 versus 13.8 questions/s over two stages.
Readout did not win every scenario. This small execution fixture does not
establish speed on other models, heterogeneous physical nodes, answer accuracy
or calibrated confidence; rerun it for the intended deployment and task.
