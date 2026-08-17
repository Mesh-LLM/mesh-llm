# GLM-DSA Runtime Contract

This contract describes the native llama.cpp execution surface required for
GLM-5.2-style GLM-DSA models. Skippy split serving must inherit this behavior;
it must not compensate for missing llama.cpp semantics.

## Scope

The contract covers:

- GGUF metadata required to load and execute GLM-DSA correctly.
- Tensor presence required for dense warmup, routed/shared MoE, MLA attention,
  GLM-DSA indexer producers, and native MTP preservation.
- Full and Shared IndexShare layer roles.
- Top-k sideband shape and failure behavior.
- Conversion-time validation expected from `skippy-quantize`.

Kernel policy and backend performance are intentionally out of scope here. They
belong in layer-package `generation.policy` and `generation.thresholds` once
this correctness contract is enforceable. Package generation policy may select
between valid GLM-DSA execution paths; it must not relax the metadata, tensor,
IndexShare, or sideband requirements in this contract.

In other words, this document answers whether a GLM-DSA artifact is executable
at all. The layer-package generation defaults answer which validated execution
path to prefer for decode, prefill, verification, and IndexShare once this
contract is satisfied.

## Required GGUF Metadata

For `general.architecture = glm-dsa`, conversion must emit the following keys:

- `glm-dsa.context_length`
- `glm-dsa.embedding_length`
- `glm-dsa.block_count`
- `glm-dsa.feed_forward_length`
- `glm-dsa.attention.layer_norm_rms_epsilon`
- `glm-dsa.attention.head_count`
- `glm-dsa.attention.head_count_kv`
- `glm-dsa.attention.key_length`
- `glm-dsa.attention.value_length`
- `glm-dsa.attention.q_lora_rank`
- `glm-dsa.attention.kv_lora_rank`
- `glm-dsa.rope.dimension_count`
- `glm-dsa.expert_count`
- `glm-dsa.expert_used_count`
- `glm-dsa.expert_shared_count`
- `glm-dsa.expert_feed_forward_length`
- `glm-dsa.leading_dense_block_count`
- `glm-dsa.expert_weights_scale`
- `glm-dsa.expert_weights_norm`
- `glm-dsa.attention.indexer.head_count`
- `glm-dsa.attention.indexer.key_length`
  - must be greater than `glm-dsa.rope.dimension_count`, because the native
    indexer path slices both RoPE and non-RoPE indexer channels.
- `glm-dsa.attention.indexer.top_k`

The following metadata is optional only when the checkpoint genuinely does not
declare it:

- `glm-dsa.rope.freq_base`
- `glm-dsa.expert_shared_feed_forward_length`
- `glm-dsa.nextn_predict_layers`
- `glm-dsa.attention.indexer.top_k_frequency`
- `glm-dsa.attention.indexer.skip_top_k_offset`
- `glm-dsa.attention.indexer.types`

If `glm-dsa.attention.indexer.types` is present, its length must match the
effective decoder layer count, excluding native MTP blocks, and each value must
be `full` or `shared`.

If both `glm-dsa.attention.indexer.types` and
`glm-dsa.attention.indexer.top_k_frequency` are present, they must describe the
same Full/Shared layer pattern.

`glm-dsa.block_count` is the total GGUF block count, including native MTP
blocks. The target decoder layer count is therefore:

```text
glm-dsa.block_count - glm-dsa.nextn_predict_layers
```

For GLM-5.2 this means:

- `glm-dsa.block_count = 79`
- `glm-dsa.nextn_predict_layers = 1`
- `glm-dsa.attention.indexer.types.length = 78`

If `glm-dsa.attention.indexer.types` is absent, frequency metadata may derive
the role pattern. If both are absent, llama.cpp may fall back to tensor presence,
but that is a compatibility fallback, not the preferred GLM-5.2 path. The
contract validator treats tensor-presence role inference as invalid for new
artifacts so old packages cannot accidentally pass the GLM-5.2 pre-spend gate.

## Required Tensors

Every effective decoder layer requires:

- attention norm
- Q A/B and Q A norm tensors
- KV A MQA and KV A norm tensors
- split `attn_k_b` and `attn_v_b` tensors for sparse GLM-DSA attention
- attention output projection
- FFN norm

The native GLM-DSA loader rejects unsplit `attn_kv_b` tensors. A model may not
carry both the split and unsplit KV-B layouts.

Dense warmup layers, where `layer_index < leading_dense_block_count`, require
normal dense FFN gate/up/down tensors.

Non-warmup layers require:

- routed MoE gate input
- routed expert gate/up/down tensors
- shared expert gate/up/down tensors

Full IndexShare producer layers require the complete indexer tensor group:

- `blk.N.indexer.k_norm.weight`
- `blk.N.indexer.k_norm.bias`
- `blk.N.indexer.proj.weight`
- `blk.N.indexer.attn_k.weight`
- `blk.N.indexer.attn_q_b.weight`

Shared IndexShare consumer layers must either omit the complete group or be
declared `shared` by metadata. Partial indexer tensor groups are invalid.

Native MTP tensors should be preserved for the last
`glm-dsa.nextn_predict_layers` blocks when present:

- `blk.N.nextn.eh_proj.weight`
- `blk.N.nextn.enorm.weight`
- `blk.N.nextn.hnorm.weight`
- optional `nextn.embed_tokens`, `nextn.shared_head_head`, and
  `nextn.shared_head_norm`

MTP execution is not part of the current performance path, but missing MTP
metadata or tensors must not be silently introduced by quantization.
For GLM-5.2-style checkpoints, the native MTP block can also contain the same
attention, MoE, and IndexShare tensor groups as a decoder block. The target
decoder runtime still executes only `glm-dsa.block_count -
glm-dsa.nextn_predict_layers` layers; IndexShare role metadata describes that
target decoder range, not the trailing MTP block.

## Full and Shared Roles

A Full layer is an IndexShare producer. It computes the GLM-DSA indexer, writes
indexer keys to the DSA KV cache, produces top-k indices, and updates the
current `last_top_k` value.

A Shared layer is an IndexShare consumer. It must use the current top-k indices
from the most recent compatible Full layer. It must not recompute the indexer
and must not silently substitute dense behavior unless the runtime explicitly
selects a documented dense fallback.

Role precedence is:

1. `glm-dsa.attention.indexer.types`.
2. `glm-dsa.attention.indexer.top_k_frequency` plus
   `glm-dsa.attention.indexer.skip_top_k_offset`.
3. Legacy/debug fallback via `LLAMA_GLM_DSA_INDEXSHARE_PATTERN` or
   `LLAMA_GLM_DSA_INDEXSHARE_FREQ` only when metadata is unavailable.
4. Complete indexer tensor presence as the last fallback.

The production path relies on checkpoint/GGUF metadata, not environment
overrides. When explicit `indexer.types` and frequency metadata both exist,
`indexer.types` is authoritative.

## Top-K Sideband Contract

When a runtime slice starts inside a Shared consumer group, it requires a GLM-DSA
top-k sideband. The sideband is appended after token-major F32 hidden states in
the activation frame.

The sideband format is:

- dtype: `i32`
- layout: token-major contiguous top-k indices
- width: `n_top_k` inferred from sideband byte count
- byte size: `token_count * n_top_k * sizeof(i32)`

The sideband width must be positive and must exactly match what the consuming
stage expects for the current token count and KV shape. A narrower sideband is
invalid because Shared consumers would otherwise run with a partial visibility
window for the current session position. A wider sideband is invalid because it
does not match the graph input shape. The activation frame `layer_end` must
exactly match the consumer stage `layer_start`.

GLM-DSA top-k sidebands cannot be combined with other activation sidebands.

## Failure Behavior

The runtime must fail hard, with a clear error, for:

- missing required GLM-DSA metadata during conversion
- `attention.indexer.types` length mismatch
- invalid IndexShare role values
- `attention.indexer.types` conflicting with IndexShare frequency metadata
- partial indexer tensor groups
- Full layers declared without indexer tensors
- GLM-DSA sparse attention without split `attn_k_b` and `attn_v_b`
- GLM-DSA sparse attention with stale unsplit `attn_kv_b`
- `nextn_predict_layers >= block_count`
- impossible MoE hparams, such as `expert_used_count > expert_count`
- consumer slices entered without a top-k sideband
- sideband payload size not divisible by `token_count * sizeof(i32)`
- sideband width that does not exactly match the consumer's current IndexShare
  width
- activation frame layer boundary mismatch
- GLM-DSA sideband on a non-GLM-DSA stage

The runtime may use an explicit dense/full-visible fallback only when the graph
logs or telemetry identify that fallback. Silent fallback is not acceptable for
GLM-DSA validation.

## Conversion Contract

`skippy-quantize` must treat GLM-DSA metadata as a runtime contract, not a best
effort template. It must fail before writing a GGUF when a required metadata key
cannot be derived from the Hugging Face config.

The current enforced conversion checks cover:

- indexer head count, key length, top-k
- optional but validated indexer role list
- inferred `glm-dsa.attention.indexer.types` from complete indexer tensor
  groups when the checkpoint config lacks explicit role or frequency metadata
  using the full mapped tensor list before any output split/shard selection
- native HF conversion of `self_attn.kv_b_proj.weight` into split
  `attn_k_b.weight` and `attn_v_b.weight`, with K transposed per head to match
  llama.cpp MLA tensor layout
- MLA Q and KV LoRA ranks
- MoE expert count, experts per token, shared expert count, and expert FFN size
- dense warmup layer count
- routed expert scaling and normalization policy
- RMS norm epsilon

Future quant profiles must preserve these keys and the indexer/MTP tensors even
when the tensor quantization type changes.

## Artifact Verification

The Phase A artifact gate is:

```bash
python3 scripts/glm-dsa-inventory-verifier.py \
  --checkpoint /path/to/zai-org/GLM-5.2/snapshot \
  --gguf /path/to/BF16/GGUF/or/package \
  --json
```

The preferred BF16 GGUF reference must pass this verifier before it can be used
as a source artifact for quantization or layer-package generation. A BF16 GGUF
that lacks `glm-dsa.attention.indexer.types` or still contains unsplit
`blk.N.attn_kv_b.weight` tensors for GLM-DSA sparse attention is stale and must
be rebuilt from SafeTensors with the current converter; native llama.cpp will
reject the unsplit layout at load time.
