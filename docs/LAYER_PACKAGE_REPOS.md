# Contributing Layer Package Repositories

Layer package repositories let Mesh LLM run very large models with Skippy stage
splits. A package repository is a durable Hugging Face repo containing one
`model-package.json` manifest plus GGUF fragments for shared tensors, per-layer
tensors, and optional multimodal projectors.

Use this page for contributor workflow. The exact schema lives in
[specs/layer-package-repos.md](specs/layer-package-repos.md).

## Repository shape

```text
model-package.json
shared/
  metadata.gguf
  embeddings.gguf
  output.gguf
layers/
  layer-00000.gguf
  layer-00001.gguf
  ...
projectors/
  mmproj-model-f16.gguf
README.md
```

Required rules:

- `model-package.json` must be at the repo root.
- `schema_version` must be `1`.
- `format` must be `layer-package`.
- Each manifest artifact path must be relative to the repo root.
- Paths must not be absolute and must not escape with `..`.
- Every artifact entry must include size and SHA-256.
- Production refs should use immutable `hf://namespace/repo@revision` pins.

Model-specific generation policy defaults belong under the manifest
`generation` section. Use `generation.policy` for package-validated execution
semantics and `generation.thresholds` for numeric runtime resolver hints. Do
not add model-family-specific objects such as `generation.glm_dsa`; model
families use stable policy profiles instead.

The split matters. Policy values are portable names for phase behavior:
`decode`, `short_prefill`, `long_prefill`, `verify`, and `indexshare`.
Threshold values are numbers the resolver uses to decide whether a policy is
appropriate on the current request/backend. A package should not encode Metal,
CUDA, or Skippy implementation names in either place.

Native speculation defaults belong beside those fields as
`generation.speculative_decoding`. Keep it as a strategy map, not as a backend
switch: `native-mtp` says the package has preserved native prediction tensors
and recommends that strategy; it does not force a Skippy-specific speculation
implementation.

For GLM-DSA packages, set `generation.policy.profile` to `glm-dsa-v1` and
declare phase choices such as decode `compact-flash`, short-prefill `dense`,
long-prefill `sparse-chunked`, and IndexShare `required`. The policy values
name semantic execution paths, not backend kernel implementations. Runtime
config may override these values for experiments, but the package manifest is
the source of truth for validated defaults and consumers must log any override
or fallback. See
[specs/layer-package-repos.md](specs/layer-package-repos.md#generation-defaults).

Minimal GLM-DSA shape:

```json
{
  "generation": {
    "policy": {
      "profile": "glm-dsa-v1",
      "decode": "compact-flash",
      "short_prefill": "dense",
      "long_prefill": "sparse-chunked",
      "verify": "auto",
      "indexshare": "required",
      "experimental": {
        "selected_row_flash": "evidence-gated",
        "moe_weighted_down": "evidence-gated",
        "moe_merged_shared_gate_up": "evidence-gated"
      }
    },
    "thresholds": {
      "short_prefill_max_tokens": 2048,
      "compact_flash_min_kv": 1,
      "dense_mask_max_bytes": 268435456
    },
    "speculative_decoding": {
      "default": "native-mtp-n1",
      "strategies": {
        "native-mtp-n1": {
          "type": "native-mtp",
          "prediction_depth": 1,
          "layer_indices": [78],
          "window_policy": {
            "default": "fixed",
            "initial_window": 1,
            "min_window": 1,
            "max_window": 1
          }
        }
      }
    }
  }
}
```

Authoring rule of thumb:

| Put it here | Use it for | Examples |
| --- | --- | --- |
| `generation.policy` | Stable semantic execution choices validated for the package. | `profile`, `decode`, `short_prefill`, `long_prefill`, `verify`, `indexshare` |
| `generation.policy.experimental` | Named opt-in paths that need package/backend evidence before becoming defaults. | `selected_row_flash`, `moe_weighted_down`, `moe_merged_shared_gate_up` |
| `generation.thresholds` | Numeric resolver inputs used to accept, reject, or fall back from a policy. | `short_prefill_max_tokens`, `compact_flash_min_kv`, `dense_mask_max_bytes` |
| `generation.speculative_decoding` | Package-owned native or draft speculation strategy defaults. | `native-mtp-n1`, `prediction_depth`, `layer_indices`, `window_policy` |
| GGUF metadata | Architecture correctness and tensor layout requirements. | GLM-DSA q/k/v split dimensions, IndexShare roles, MTP tensor presence |

Writers should emit a profile only after the artifact actually matches that
profile. For `glm-dsa-v1`, that means the package writer has validated the
GLM-DSA attention tensor shape, routed/shared MoE tensors, IndexShare role
evidence, and preserved native MTP tensors when present. Serving code should
then resolve the policy by phase and backend capability, not by hard-coding
model-family branches in Skippy or Mesh.

Do not put backend names, implementation flags, or model-family-specific
objects in `generation`. A GLM-DSA package should not introduce
`generation.glm_dsa`; a CUDA/Metal-specific runtime should not introduce
backend-specific manifest fields to select a kernel. Backend support is runtime
capability evidence. The manifest records the package's validated semantic
policy and the numeric thresholds needed to explain resolver decisions.

The threshold values should be grounded in tensor sizes. For example, a
GLM-DSA top-k sideband costs `tokens * top_k * 4` bytes, while a dense sparse
mask costs roughly `tokens * visible_kv * 4` bytes. With a representative
GLM-5.2 split package sideband width of `768` i32 values per token, the
IndexShare sideband is `3 KiB/token`: `384 KiB` for a 128-token chunk,
`1.5 MiB` for 512 tokens, `6 MiB` for 2048 tokens, and `384 MiB` for a full
128k-token window. At `128k` visible KV, the corresponding dense mask is
`512 KiB/token`: `64 MiB` for 128 tokens, `256 MiB` for 512 tokens, and `1 GiB`
for 2048 tokens. That size gap is why GLM-DSA packages should record
dense-mask and compact-flash thresholds instead of leaving sparse policy
implicit.

Generation fields are not a place to record every benchmark knob. For example,
q2_K routed-down quant experiments, Metal simdgroup sweeps, row-height tuning,
or `mul_mv_id` versus `mul_mm_id` dispatch choices should stay in benchmark
reports and runtime capability evidence until they become portable semantic
policy. If they do become package defaults, express them as a profile version,
a phase policy, or a numeric threshold rather than a backend flag.

For current GLM-DSA decode tuning, the llama.cpp Metal fixtures measured
compact selected-row flash at `63.40 us/run` for
`GLM_DSA_SELECTED_ROW_FLASH(kv=257,top_k=64)`, direct sparse attention at
`106.57 us/run`, and dense masked flash at `71.72 us/run` on the comparable
one-token shape. That is the evidence behind preferring
`decode: "compact-flash"` once parity is proven on the target package/backend.
For exact-boundary fixtures where selected KV equals visible KV, compact
selected-row flash measured `61.90 us` at `kv=128`, `60.71 us` at `kv=256`,
`61.24 us` at `kv=257`, and `57.99 us` at `kv=513`; direct sparse measured
`137.15 us`, `209.97 us`, `249.30 us`, and `711.10 us` for the same shapes.
The literal `top_k >= visible_kv` boundary is handled by llama.cpp's all-KV
flash bypass, but ordinary one-token decode after prefill is different:
IndexShare top-k is selected from the previous KV state while attention sees
previous+current KV. Even at short history, that route still exercises compact
selected-KV flash. That is why native GLM-DSA decode should prefer compact
flash over direct sparse by default when flash attention is available.
At GLM-5.2's configured `top_k=768` width, compact selected-row flash measured
`55.11-55.62 us/run` for `kv=1024..2048`, while direct sparse measured
`984.50-988.95 us/run` on the same shapes. That is roughly an `18x` win for
compact selected-KV flash over direct sparse once visible KV exceeds the
selected-row window.
The same fixture family measured dense masked flash much faster than direct
sparse for short phase shapes (`68.58-70.80 us/run` versus
`461.98-473.75 us/run` for 4-16 tokens), so GLM-DSA packages should keep
short prefill and verification dense by default unless a backend-specific sparse
path has its own evidence.
After those phase gates, the next measured local bottleneck is the MoE FFN
rather than top-k routing. The gated Metal MoE fixture estimates one GLM-5.2
routed FFN decode layer at `391.13 us`, with expert matmuls accounting for
`375.42 us` (`96.0%`). Routed fused SwiGLU is only `5.35 us` (`1.4%`), and
route/top-k plus weighted sum is only `10.36 us` (`2.6%`). The shared expert is
not small. A production-shaped fused GLU shared expert plus final add measured
`405.32 us`, making the routed+shared FFN estimate `796.45 us` with the shared
expert at `50.9%`. The shared fused SwiGLU row itself is cheap (`4.09 us`, or
`8.24 us` including final add); the earlier unfused `silu(gate) * up`
diagnostic row measured `291.73 us` but does not represent the normal llama.cpp
`build_ffn()` path, which already uses `ggml_swiglu_split()`. That evidence
should inform runtime optimization order, but it does not add new manifest
schema: package policy still belongs under `generation.policy`, and numeric
resolver hints still belong under `generation.thresholds`.
The combined FFN fixture keeps that conclusion intact with a cleaner
production-shaped signal than the earlier isolated rows. With
`n_embd=6144`, the q3_K routed down baseline measured `1147.21 us`; changing
decoder routed down projections to q2_K measured `817.38 us` (`1.40x` faster),
and keeping q3_K routed down while using merged shared gate/up measured
`868.59 us` (`1.32x` faster). Route/top-k plus weights measured only `3.26 us`,
weighted sum measured `6.69 us`, and routed fused SwiGLU measured `5.09 us`, so
the next practical levers are whole-graph expert matmul/layout work and a
quality-tested q2_K routed-down quant recipe. The q2_K routed-down recipe must
exclude `blk.78`, the native NextN/MTP block, until quality and speculation
tests prove that lowering the MTP block is acceptable.
The Phase E report can be run as a hard evidence gate with
`GLM52_PHASE_E_REQUIRE_GATES=1`; the gated rows require production-shaped
routed/shared MoE, q2/q3 down alternatives, and the optional kernel sweep rows
when `GLM52_PHASE_E_KERNEL_SWEEP=1` is enabled. The optional sweep rules out two
tempting kernel-policy shortcuts: forcing one-token MoE through Metal
`mul_mm_id` measured `860.96 us` for q3_K routed down versus `164.46 us` on the
default `mul_mv_id` path, while q3_K `mul_mv_id` simdgroup tuning was
noise-level (`164.46 us` default versus `163.92 us` best measured). The
row-height sweep reached the same conclusion: q3_K row-height tuning measured
`163.98 us` for default `nr0=4`, which was also the best sampled row; `nr0=1`
and `nr0=2` were around `195-196 us`, and `nr0=8` was `166.20 us`. A fixed
`k=2048` q3_K GLM-down specialization measured `164.04 us` at default `nsg=2`
and `162.67 us` at `nsg=4`, only `1.01x` faster than ordinary default q3_K
`mul_mv_id`. The next meaningful local target is therefore a deeper expert
matmul/layout specialization or a quality-tested down-projection quant change,
not generic matrix-matrix cutoff, simdgroup, row-height, or fixed-block tuning.
The earlier `TOPK_MOE_GLM_CONSUMER_SANITY` rows remain useful as harness
history: they fixed a perf-mode bug where the fixture had `run_whole_graph()`
for correctness but timed only the final `GGML_OP_MOE_WEIGHTED_SUM` during
perf. The newer combined FFN fixture supersedes those rows as the decision
gate because it keeps routed and shared experts together. Keep
`moe_weighted_down` evidence-gated rather than making it a package default; the
combined decision path now favors q2_K routed-down quality testing and merged
shared gate/up investigation.

In practice, this means the package's `generation` block is the phase-aware
contract: decode prefers compact flash, short prefill prefers dense, long
prefill avoids dense masks, verification remains `auto`, and Shared-layer
IndexShare is required unless an explicit fallback is selected and logged.

## Local package tooling

`skippy-model-package` is the local inspection and writing tool. Current
subcommands are:

```bash
skippy-model-package inspect <model.gguf>
skippy-model-package plan <model.gguf> --stages 4
skippy-model-package write <model.gguf> --layers 0..12 --out ./stage-0.gguf
skippy-model-package write-stages <model.gguf> --stages 4 --out-dir ./stages
skippy-model-package write-package <model.gguf> --out-dir ./package
skippy-model-package validate <model.gguf> ./stages/stage-*.gguf
skippy-model-package validate-package <model.gguf> ./package
skippy-model-package preflight ./package --stages 4 --verify-sha256
```

Validate before publishing:

```bash
skippy-model-package validate-package <model.gguf> ./package
skippy-model-package preflight ./package --stages 4 --verify-sha256
```

## Queue a Hugging Face package job

Mesh LLM includes a spend-bearing HF Jobs helper for package generation. It is
dry-run by default and must be confirmed explicitly before submitting jobs:

```bash
mesh-llm models package unsloth/Qwen3-8B-GGUF:Q4_K_M --dry-run
mesh-llm models package unsloth/Qwen3-8B-GGUF:Q4_K_M --confirm --follow
```

The hidden compatibility alias is `mesh-llm model-package`; prefer
`mesh-llm models package` in docs and scripts.

Important options:

- `--target <repo>`: destination Hugging Face package repo.
- `--model-id <id>`: OpenAI-facing package model id.
- `--timeout <duration>`: HF Jobs timeout, defaulting to `1h` unless raised by
  size-based estimates.
- `--dry-run`: print the resolved package plan and maximum cost without side effects.
- `--confirm`: submit the job.
- `--follow`: wait and stream job progress.
- `--status <job-id>`, `--logs <job-id>`, `--cancel <job-id>`, `--list`: inspect
  or manage submitted jobs.
- `--update-script`: refresh the bucket script when needed.

The source model should stay in colon-selector form, for example
`unsloth/Qwen3-8B-GGUF:Q4_K_M`. Do not split the quant into a separate `--quant`
argument for generated job inputs.

## Publishing flow

The HF Jobs script performs the publishing work:

1. clone mesh-llm,
2. build `skippy-model-package`,
3. run `write-package`,
4. validate the manifest,
5. upload package artifacts incrementally,
6. upload `model-package.json`,
7. write a package model card,
8. update `meshllm/catalog`,
9. print the suggested run command.

The printed run command follows this shape:

```bash
mesh-llm serve --model <target-repo> --split
```

For package refs in hand-written docs and configs, prefer the explicit package
scheme:

```text
hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers@<revision>
```

## After publishing

Run package-only certification first:

```bash
mesh-llm models certify hf://namespace/repo@revision --package-only --report-out cert.json
```

Use the immutable published ref for certification, not only the local package
directory. Fix package-local preflight diagnostics and published-ref
package-only certification failures before moving on to a live endpoint smoke.

Then run a live endpoint smoke once the mesh is serving it:

```bash
mesh-llm models certify hf://namespace/repo@revision --api-base http://127.0.0.1:9337 --json
```

If the package is intended for public meshes, keep peer artifact transfer off by
default. Enable `MESH_LLM_ARTIFACT_TRANSFER=trusted` only for same-owner or
explicitly trusted-owner deployments, and `open` only in lab environments.
