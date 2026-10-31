# Reusable KV validation on JY

The bounded [validation foundation](skippy/evals/kv-validation/README.md) now
provides explicit reusable commands. Complete the remaining cases below after
the current cache and performance work.

## Completed foundation

- [x] Own the exact four-model manifest and reject model/hash substitutions.
- [x] Prepare a matching-header, pinned-runtime real-model evaluator once and
  reuse one model load across the supported K/V type pairs.
- [x] Provide typed Rust pin/prepare/run/JY-plan commands and thin Just recipes,
  with bounded child supervision, cancellation and failed evidence retention.
- [x] Run a fresh c1 cold/repeat/growing/repeat generated-content check, plus the
  unchanged strict recurrent chat/growing tests for the declared hybrid rows.
- [x] Emit four independent Linux CPU/CUDA JY specs with excluded-host guards,
  immutable image/model/tool identities and no worker rebuilds. Local Metal
  uses the same native prepare/run commands; native macOS JY admission remains
  separate work.

The new command passed end to end locally on the pinned Qwen3.5-4B model with
all eight Metal formats. This does not mark the full four-model/backend suite,
long-prefix ownership, L3 restart or throughput comparisons complete.

The hybrid rows now also require a bounded native cold-eviction/runtime-restart
regression. On local Qwen3.5-4B Metal and CPU, the 64-token checkpoint survives source
lane release, cold eviction, and fresh model/runtime/cache construction with an
L3 hit, identical attention/recurrent bytes and four identical fixed-input
continuation predictions. This is one-process runtime-restart evidence; the
fresh-process server and long-prefix L3 cases below remain open. Separate native
spill tests cover queue-full, disconnect, underestimated payload and blocked /
refused writes, checking that bytes and identity claims are released.
The model-backed refusal case also passes on both backends: a one-byte L3 budget
cannot retain a checkpoint, but eviction frees native attention cells and cold
recomputation produces identical fixed-input predictions.

## Retain PR #2657's correctness requirements

Cache-owned immutable checkpoints replace the old execution-lane pinning and
leased-pointer recorder. Before closing
[PR #2657](https://github.com/Mesh-LLM/mesh-llm/pull/2657) as fully superseded:

- [ ] Port native hybrid cancellation/consumer-error, recorder saturation and
  unload/L3-restart integration assertions to cache-owned checkpoints. Keep
  exact consumed-token boundaries, immutable KV/recurrent state and transactional
  restore checks; token-count hashes cannot replace them.
- [ ] Exercise the new spill reservation under byte-budget refusal, queue-full,
  disconnect, export failure and blocked/failed writes. Reserve both attention
  and recurrent bytes before export; retain accounting through write completion
  and prove all failure paths release it without wedging request admission.
- [ ] Document and expose the current best-effort cold-eviction policy: a refused
  or failed L3 offload can release the final cached native owner. This loses reuse
  and causes later prefill; it does not make every retained prefix durable.
  Existing blob-L1 refusal tests do not cover native-owner eviction. Preserve
  explicit refusal/eviction outcomes rather than restoring the old lane-pinning
  timeout or forced direct-decode routing.

## Four-model AgentX matrix

Run the suite on all four selected models, each using **Q4_K_M**:

| Model | Category | GGUF |
|---|---|---|
| Qwen3-4B-Instruct-2507 | Dense attention | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` |
| Qwen3.5-4B | Hybrid dense | `Qwen3.5-4B-Q4_K_M.gguf` |
| Qwen3-Coder-30B-A3B-Instruct | Attention MoE | `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` |
| Qwen3.5-35B-A3B | Hybrid MoE | `Qwen3.5-35B-A3B-Q4_K_M.gguf` |

Use the exact HF repositories, revisions, tokenizer revisions, and weight hashes
from `6c0464e5531a9ff8ceb89c068ad94cd53c71e334:skippy/evals/agentx/models.json`.
The recovered manifest has SHA-256
`6a2b724cf32208186d80ac614e21a168bfc08ca72e55afdabc3dc39b338c9206`.
Small generated native fixtures supplement this matrix; they do not replace it.

## Reusable jobs

- [ ] Build matching Skippy binaries, native runtimes, and correctness tests once
  per backend; reuse the verified artifacts across model jobs.
- [ ] Validate native prefix ownership with long prefixes and multiple lanes:
  save, release the source, apply pressure, recycle lanes, restore, and fork.
  Compare attention and recurrent state bytes, fixed-input logits, and tokens.
- [ ] Compare from controlled identical state and batch geometry. Include repeat
  controls so allocation history or numerical variation cannot masquerade as a
  cache bug. Do not weaken correctness gates to accept an optimization.
- [ ] Exercise cold, warm, shared-prefix, changed-tail, and A/B/C/revisit requests
  at c1/c2/c4. Check reuse counts, actual shared-pool capacity, and lane isolation.
- [ ] Admit concurrent long prompts whose combined reservations exceed one
  lane's context but fit the loaded aggregate pool. Verify actual overlap and
  preserve each request's context limit; distinguish admission waits from
  frontend runtime-lock waits.
- [ ] Measure identical cold work after retaining additional prefixes. Compare
  attention kernel time, matrix multiplication, transfers, and mask/input work
  against standard vanilla to catch costs that grow with idle physical KV cells.
- [ ] Exercise every KV format supported by the CUDA and Metal attention paths,
  including supported mixed K/V types, in prefill and decode. Verify indexed GPU
  execution, unequal lengths, recycled physical rows, and graph reuse; CPU or
  dense-path fallback must not count as quantized indexed-attention coverage.
- [ ] Run real-model prefill and generated-token tests across all four models on
  **CUDA, Metal, and CPU**, with flash attention and unified KV explicitly
  enabled. Cover F16 and every KV format and mixed K/V combination supported by
  each backend; include cold execution, retained-prefix pressure, warm reuse,
  and checkpoint restoration. Keep the operator matrix as a separate gate;
  generated fixtures and kernel tests do not replace these runs.
- [ ] Test cache-off controls and c1 generated-content parity. Output-token counts
  and their hashes alone are not generated-token correctness evidence.
- [ ] Test the default 1 GiB checkpoint budget and an explicit larger budget,
  independently of disk caching, including rejection of entries that cannot fit.
- [ ] Force persistent L3 eviction, wait for committed writes, stop cleanly, and
  restart a fresh server. Verify real L3 hits, complete restoration, and content
  correctness. Cover refused or failed offloads and fail-open inference.
- [ ] Compare throughput, TTFT, token latency, cache reuse, and memory against
  **standard vanilla llama.cpp** on the same model and workload. Unified-KV llama
  is a separate diagnostic control and must not replace the parity target.
- [ ] Retain the fixed AgentX replay as a performance check with its pinned
  external client and corpus, including complete request/output accounting.

## Execution and evidence

- [ ] Implement typed Rust automation in the existing owning tooling, with thin
  Just entrypoints and reusable JY job definitions. Reuse existing native tests,
  cache-family measurement, stability checks, and process supervision.
- [ ] Make these explicit JY validation jobs; do not add the full GPU/model matrix
  as an always-on local or ordinary CI requirement.
- [ ] Materialize actual settings: context per lane and total pool, concurrency,
  threads, native batch/microbatch, frontend prefill chunk size/policy, KV dtypes,
  flash attention, cache budgets, and disk mode. Native microbatch size does not
  establish the frontend prefill chunk size.
- [ ] Verify source, model, binary, test binary, runtime leaf, ABI, patch digest,
  image, client, and corpus identities before measuring. Reject stale or nested
  runtime bundles and require proof of real GPU execution.
- [ ] Verify the effective native graph after execution: flash-attention nodes
  must consume the request indices and lengths and execute on the requested
  `MTL0`, `CUDA0`, or CPU backend, with the expected K/V types. Weight placement
  or requested flags alone do not prove this. Missing backend support, disabled
  FA, dense fallback, or execution on the wrong backend must fail the required
  gate rather than silently pass or skip it. CPU execution is required for CPU
  jobs and is a failure for required Metal/CUDA jobs.
- [ ] Provide test entrypoints that can set FA, unified KV, and K/V types
  explicitly even when an ordinary CLI omits those options. Keep default CPU
  reference tests separate from explicit required CPU/Metal/CUDA model runs.
  Record real generated output and preserve failed numerical comparisons without
  loosening their gates.
- [ ] Schedule independent jobs across equivalent available Sparks without
  stacking measurements on one GPU. Current excluded nodes are `spark-000280`
  (`spark-db08`) and `spark-000499` (`spark-9f06`); enforce exclusions through the
  operator profile and worker admission.
- [ ] Retain raw requests/responses, resolved settings, timing and memory samples,
  logs, and artifact hashes for successes and failures. Propagate child failures;
  classify unsupported or missing required evidence explicitly.
- [ ] Report correctness separately from performance regressions. Document the
  commands and run the reusable suite across all four models on the allowed
  machines, with CUDA, Metal, and CPU coverage recorded separately.
