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
- [ ] Schedule independent jobs across equivalent available Sparks without
  stacking measurements on one GPU. Current excluded nodes are `spark-000280`
  (`spark-db08`) and `spark-000499` (`spark-9f06`); enforce exclusions through the
  operator profile and worker admission.
- [ ] Retain raw requests/responses, resolved settings, timing and memory samples,
  logs, and artifact hashes for successes and failures. Propagate child failures;
  classify unsupported or missing required evidence explicitly.
- [ ] Report correctness separately from performance regressions. Document the
  commands and run the reusable suite across all four models on the allowed
  machines, with CUDA and Metal coverage recorded separately.
