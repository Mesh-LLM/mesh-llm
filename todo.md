# Reusable KV validation on JY

Implement this **after the current cache and performance work**. Turn the
experiments into repeatable JY jobs that we can run whenever KV behavior changes.

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
