# SafeTensors stage-local download and adaptive MLX quantization

## Status

Metadata planning, exact-range materialization, and two-stage MLX execution
proofs completed on 2026-07-17. The frontier-model measurements inspect headers
only; the SmolLM2 proof downloads and executes selected tensor payloads.

The standalone `mlx-safetensors-stage-plan` spike proves that a layer server can:

1. read `model.safetensors.index.json`;
2. select only tensors owned by its layer range;
3. fetch the 8-byte length and JSON header from each relevant SafeTensors file;
4. turn each tensor's `data_offsets` into absolute HTTP byte ranges; and
5. stream those ranges into a valid partial `model.safetensors` artifact.

It refuses a response other than HTTP `206 Partial Content`, preventing an
ignored `Range` header from silently downloading a multi-gigabyte shard.

## Bottom line

SafeTensors supports the desired model: canonical upstream weights remain on
Hugging Face, while each layer server downloads and caches only the tensors it
owns. A separately published layer-package repository is not required.

For normally ordered checkpoints, selecting whole shard files already gets
close to exact. For Inkling, tensors are heavily interleaved across source
shards, so exact tensor ranges are mandatory: whole-shard selection would turn
a 109.84 GiB four-layer stage into a 942.99 GiB download.

The small dense-model path is now proven through execution, including a host
path that streams selected tensor ranges into bounded affine-4 cache shards,
then loads the derived stage without retaining a complete BF16 stage slice.
`model-hf` exposes the backend-neutral sequential visitor that downloads each
selected range as an ephemeral, valid one-tensor SafeTensors file and deletes it
before fetching the next tensor. Its prepared session exposes the verified
config, config hash, checkpoint identity, and range plan before payload
callbacks. On macOS/Unix, advisory locks also scavenge crash-abandoned visits
without removing concurrent ones.

The pinned SmolLM2 layer-14 visitor proof fetched 9 tensors totaling 7,080,192
bytes from a 269,060,552-byte source shard. Its largest temporary one-tensor
file was 1,769,584 bytes, and the visitor cache directory was empty afterward.
Those figures bound temporary disk use for this fixture, not process memory.

## Reproduce

```bash
just mlx-safetensors-stage-plan \
  --repo thinkingmachines/Inkling \
  --revision 86b4d430ab871652a707666b89203a866888c5e5 \
  --layer-start 30 \
  --layer-end 34
```

Use `--json` for the per-shard byte ranges. Additional tensors can be assigned
with repeated `--include-prefix` arguments; for example the first stage can own
the embedding and modality towers, while the final stage owns final norm,
readout, and optional MTP tensors.

Without `--output`, the CLI fetches only the index and SafeTensors headers. With
`--output <dir>`, it fetches the selected payload ranges, writes
`model.safetensors`, `config.json`, and a reproducible `stage-plan.json`, and
still refuses any payload response other than HTTP 206.

## Small-model execution proof

`HuggingFaceTB/SmolLM2-135M-Instruct` at immutable revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac` was split unnecessarily at layer 15:

| Stage | Owned tensors | Exact payload | Whole checkpoint | Avoided locally | HTTP payload spans |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0: embedding + layers 0..15 | 136 | 155.28 MiB | 256.60 MiB | 101.28 MiB | 3 |
| 1: layers 15..30 + norm + tied embedding | 137 | 155.28 MiB | 256.60 MiB | 101.28 MiB | 4 |

The tied embedding is intentionally duplicated: stage 0 uses it for token
input, while stage 1 uses it as the tied output projection. Neither stage file
contains the complete checkpoint. A strict whole-model baseline is assembled
from the union of the two partial files, so the parity test cannot silently
fall back to a full download.

The `mlx-split-proof` harness runs layers 0..15 and 15..30 as separate MLX
stages, serializes the residual through Skippy's real `StageWireMessage` binary
codec, maintains independent per-stage KV caches, and compares against unsplit
execution. Prompt prefill plus eight greedy decode steps passed on Metal with
both F16 and F32 wire encodings:

- identical eight-token sequence: `284, 260, 2240, 314, 1343, 327, 624, 8685`;
- worst maximum absolute logit delta: `0.0` for F16 and F32; and
- F16 total stage-wire traffic: 15,584 bytes for the tested prefill and decode.

One important engine contract emerged: after decoding F16/F32 wire bytes, the
receiving MLX stage must cast the residual back to the model's compute dtype
(BF16 here) before its first block. Leaving the reconstructed array as F32
changed Metal arithmetic and immediately changed the greedy token, despite
identical numeric boundary values.

Reproduction uses the two materializer invocations followed by:

```bash
just mlx-safetensors-split-proof \
  --stage0 /tmp/mlx-split-smol/stage0 \
  --stage1 /tmp/mlx-split-smol/stage1 \
  --split 15 \
  --steps 8 \
  --wire-dtype f16
```

This original harness proved the artifact, MLX layer-range, KV-cache, and
existing binary activation-frame seams on one Mac. Subsequent commits added the
engine abstraction, two real stage processes, and explicit host
`StagePrepare`/`StageLoad` consumption. Automatic topology production and a
remote two-node MLX run remain.

## Small-model load-time quantization proof

`MlxStageEngine` now optionally constructs quantized Llama modules and calls
safemlx's strict tensor-streaming loader. For every selected dense tensor, that
loader produces the packed weight/scales/biases, calls `eval`, synchronizes the
quantization stream, installs the result, and then continues. It does not build
a BF16-stage-sized lazy quantization graph. This is not yet a one-source-copy
guarantee: `Array::try_from(TensorView)` and the subsequent stream copy can both
contribute to the physical high-water mark, and mmap pages are outside MLX
allocator counters.

On the same SmolLM2 split, affine 4-bit with group size 64 produced this
whole-model reference:

```text
[260, 2240, 314, 253, 1379, 282, 25801, 28]
```

The two separately quantized `0..15` and `15..30` processes reproduced all
eight tokens over F16 stage residuals. Each process retained 349 MLX parameters;
post-proof RSS was 87,392 KiB and 87,952 KiB. This is correctness and steady-RSS
evidence, not a peak-memory claim.

## Direct range-to-quantized artifact proof

`mlx-stage derive` now removes the complete BF16 stage slice from the workflow.
It consumes one verified range file at a time, quantizes eligible rank-2 Llama
weights, evaluates and synchronizes MLX work, copies the packed arrays into a
bounded host-side shard, and returns before `model-hf` deletes the dense source
file. Output uses pure-Rust SafeTensors serialization to coexist with the
Skippy/llama.cpp native link.

For the two 15-layer SmolLM2 halves at affine-4/group-64 and 16 MiB output
shards:

| Layers | Dense source payload | Derived artifact | Largest source temp | MLX peak active | Max RSS | macOS peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0..15` | 162,825,984 B | 45,887,848 B | 56,623,208 B | 72,548,352 B | 140,853,248 B | 240,157,440 B |
| `15..30` | 162,827,136 B | 45,889,726 B | 56,623,208 B | 72,548,352 B | 140,722,176 B | 241,550,080 B |

Each artifact contained three shards and an index. Both loaded directly as
pre-quantized partial stages and reproduced the same eight-token affine-4
reference over F16 residuals. Repeating a one-layer derivation produced a
byte-identical weight shard. The report separates its input recipe hash from an
output-content digest and per-shard hashes; runtime memory evidence is
deliberately not part of either. Whole directories are not byte-identical
because reports include the local path and measurements. Shard size is a soft
bundle target and a single packed tensor may exceed it. The artifact byte count
excludes the report. The working-disk metric is a measured source-tensor plus
artifact-payload high-water mark, not allocated filesystem blocks or lock/report
overhead. The v1 builder fails closed unless `model_type` is exactly `llama`.

The follow-on `derive-cached` path maps that recipe to a locked managed
directory and validates the report schema, recipe, aggregate artifact bytes,
output-content digest, and every shard hash before accepting a hit. On the same
pinned layer-14 slice, a cold call made 9 tensor-payload requests and the warm
call made 0, returned the identical recipe/content/shard hashes, and used
17,809,408 B max RSS. The warm path still reads lightweight config/index/header
metadata to reconstruct the strong key.

The host `StagePrepare` and `StageLoad` lifecycle now uses that cache. The
stage-load wire carries an additive `auto`/affine-4/affine-8/MXFP4 profile;
older peers that omit it select `auto`, while unknown values fail closed. On
Apple Metal, `auto` currently resolves to affine-4/group-64. A clean two-half
host test made 136 and 137 tensor-payload requests, fetched 162,825,984 and
162,827,136 bytes, and produced 45,859,713 and 45,861,308-byte artifacts. The
complete Prepare/Load/Start/generate/Stop lifecycle passed in 120.74 seconds
with the established affine-4 tokens. An immediate validated-cache run passed
in 8.87 seconds with 258,162,688 B max RSS. Cache eviction and an optional
local request-to-recipe locator remain; the host cache root can be overridden
with `MESH_MLX_DERIVED_CACHE_DIR`.

Only Prepare may fetch tensor payloads and build a missing entry. Load performs
metadata planning and full cache validation, then fails rather than deriving if
the prepared entry is absent or corrupt.

Checkpoint claims are compared with the metadata-derived identity before the
first tensor payload request. Cancellation is cooperative through cache-lock
waits and the sequential range/quantization loop; an already-running HTTP or
MLX operation finishes before cleanup. Inventory responses echo the requested
quantization profile, and preparation plus running status carry it, so readiness
for one recipe cannot silently satisfy another. Cache-hit validation reads each
shard once while checking its own hash and the aggregate content digest.
Mixed-version automatic placement must still capability-gate explicit
non-default profiles because an old receiver ignores new additive fields.

## Representative measurements

All rows select four middle transformer layers. Layer types vary within hybrid
models, so the table demonstrates storage locality rather than equal compute.
Every repository is pinned to the immutable revision shown below.

| Model / source encoding | Revision | Layers | Full tensor bytes | Selected bytes | Whole relevant shards | Avoided by exact ranges | Largest selected tensor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-235B-A22B BF16 | `8efa61729e24bd65b1d152b5ab5409052aa80e65` | 40..44 | 437.89 GiB | 18.54 GiB | 22.31 GiB | 3.78 GiB | not recorded |
| Inkling BF16 | `86b4d430ab871652a707666b89203a866888c5e5` | 30..34 | 1.73 TiB | 109.84 GiB | 942.99 GiB | 833.15 GiB | 18.00 GiB |
| Inkling official NVFP4 | `d11961f515e883e37796edb9dd6ec1bf0e0e8212` | 30..34 | 551.21 GiB | 32.22 GiB | 501.59 GiB | 469.37 GiB | not recorded |
| Inkling community MLX affine-4 | `34f92fe0879faa413071c8dad23538014f0c266b` | 30..34 | 521.97 GiB | 32.22 GiB | 40.27 GiB | 8.05 GiB | not recorded |
| Nemotron 3 Ultra 550B BF16 | `624ba927cfbef0427354998700de3d51173c8c04` | 48..52 | 1.02 TiB | 41.81 GiB | 46.44 GiB | 4.63 GiB | 548 MiB |
| Kimi K2.6 published checkpoint | `7eb5002f6aadc958aed6a9177b7ed26bb94011bb` | 28..32 | 554.27 GiB | 36.54 GiB | 36.54 GiB | 0 | 112 MiB |
| GLM-5.2 BF16 | `b4734de4facf877f85769a911abafc5283eab3d9` | 36..40 | 1.37 TiB | 73.54 GiB | 79.90 GiB | 6.36 GiB | 192 MiB |
| DeepSeek V4 Pro FP8 | `b5968e9190ef611bbf34a7229255be88a0e937c1` | 28..32 | 805.32 GiB | 51.73 GiB | 51.73 GiB | 0 | 112 MiB |

The production `model-hf` planner now reproduces a tractable `nemotron_h`
architecture case from Nemotron 3 Nano rather than relying only on this spike.
At immutable revision
`97ab8012882a655dc38df4fee47422aca9caca07`, layer `1` of NVIDIA's 30B-A3B Base
BF16 checkpoint selects 261 tensors / 2,594,936,576 bytes from a single
4,991,210,024-byte shard, with a 19,955,712-byte largest tensor and two
coalesced payload ranges. The config contains bare `Infinity`, so production
metadata parsing uses strict JSON first and a JSON5 fallback.

The next production proof now exists too. The exact layer ranges were consumed
one tensor at a time and converted into an affine4/g64 artifact without ever
constructing the dense 128-expert bank. The run quantized 258 matrices, copied
three dense tensors, and reduced 2,594,936,576 source bytes to 730,324,736
tensor bytes in 199.70 seconds. Maximum RSS was 822,165,504 bytes; the largest
ephemeral source tensor file was 19,955,848 bytes. The six routed bank tensors
use underscore companions required by safemlx, while shared expert matrices use
normal dotted affine companions. The artifact strict-loaded into the actual
Nano layer-1 block and produced finite `[1, 1, 2688]` output. The shared
`MlxStageEngine` adapter now also auto-detects and loads this one internal
stateless MoE layer. Its F32 residual output matched direct execution of the
same affine block within `atol=1e-4`, `rtol=1e-4` (max absolute
`1.1920929e-7`, max relative `1.8225228e-5` above the `atol`
reference-magnitude floor across repeated runs). Two session IDs and an
independent reset/reuse comparison of session 1 all passed. The different output
hashes show that sparse MLX executions are not bit-identical; the numerical gate
is explicit rather than claiming exactness.

This is bounded by the final packed layer rather than one-expert RAM: the six
routed buffers total 718,405,632 bytes, consistent with the measured RSS. A
disk-backed spool would lower derivation memory further. Strict loading, a
finite forward, and stage-wrapper parity prove executable assembly and the mesh
engine seam, not dense-versus-affine accuracy.

The format mechanism is stable and documented by the
[SafeTensors format](https://github.com/huggingface/safetensors#format): the
header records each tensor's dtype, shape, and byte offsets. Hugging Face's
[Xet download protocol](https://huggingface.co/docs/xet/download-protocol#range-downloads)
supports partial-file reconstruction, and the public `resolve` endpoint honored
the byte ranges in every probe above.

## Inkling as the frontier stress test

[Inkling](https://huggingface.co/thinkingmachines/Inkling) is a 975B-total,
41B-active, 66-layer multimodal MoE with:

- 256 routed experts, with 6 selected per token, plus 2 shared experts;
- 6144-wide residual states;
- relative-position attention rather than RoPE;
- a 5:1 sliding-window/global-attention pattern;
- four short-convolution states per decoder layer;
- text, vision, and audio inputs;
- a 1,048,576-token maximum context; and
- eight optional MTP predictor layers.

The source artifacts are:

| Artifact | Exact tensor bytes | Notes |
| --- | ---: | --- |
| `thinkingmachines/Inkling` | 1,904,604,285,204 | Canonical BF16; 109 weight files plus index |
| `thinkingmachines/Inkling-NVFP4` | 591,854,374,368 | Official calibrated NVFP4; intended for Blackwell-class serving stacks |
| `mlx-community/Inkling-mlx-4bit` | 560,463,783,044 | Text-only mixed MLX affine-4 experiment |

The community MLX artifact is useful size evidence, but it is not yet a runnable
or certified answer for mesh-llm. Its own model card says the custom Inkling
forward is not registered in upstream `mlx-lm`, logits are not numerically
verified, and the vision/audio towers are excluded. The repository contains
weights and tokenizer/config files but not the custom model implementation.

The pinned Rust dependency is commit
`4e53c5ecd7cbd91c0dfd0992a3c731ca2c36e9c7` ("Add Thinking Machines Inkling
support"). Its `safemlx-lm` Inkling family implements the text decoder, dMel
audio and hMLP vision towers, native SafeTensors key transforms, heterogeneous
KV/SConv cache, and ordinary generation; its loader intentionally skips MTP
weights. The authoritative parity oracle remains
[Transformers' Inkling model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/inkling/modular_inkling.py).

What is missing is narrower than a family port but still substantial: safemlx's
Inkling model internals are not exposed as a layer-range stage. Its common
`PackedSwiGluExperts` runtime already supports affine/MXFP4 grouped execution,
but Inkling constructs those expert banks with no quantization and its custom
weight-transform loader only installs dense arrays. The high-level loader's
claim that grouped quantized execution is absent is stale; the real gap is
constructor metadata plus transformed rank-3 packing/loading.

### What an Inkling MLX stage engine must implement

1. Confirm the existing whole-model text decoder against Transformers on a
   tractable fixture/reduced config before changing its visibility.
2. Expose a stage constructor that creates only `layers[start..end]`, with embeddings
   on the first stage and final norm/readout on the last.
3. Split the existing heterogeneous KV plus SConv recurrent cache by stage;
   Inkling cannot be treated as a plain paged-KV Llama family.
4. Wire the existing packed affine/MXFP4 expert runtime into Inkling's
   constructor and transformed loader/profile. Keep sensitive tensors dense
   initially.
5. Logit parity against Transformers at several layer cuts before network work.
6. Vision/audio towers on the first stage after the text chain is certified.
7. MTP as a separate optional capability after ordinary decode is correct.

The residual-stream boundary remains clean between decoder layers, so these
features make family bring-up substantial but do not invalidate pipeline
splitting.

## Hardware-adaptive quantization

The user's proposed model is viable: choose a quantization plan after topology
and hardware discovery, then quantize only each server's selected tensors during
cold load. MLX directly supports affine 2/3/4/5/6/8-bit quantization with group
sizes 32/64/128, plus MXFP4, MXFP8, and NVFP4. It also accepts a per-module
predicate, allowing sensitive modules and different layer ranges to retain more
precision. See [`mlx.core.quantize`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
and [`mlx.nn.quantize`](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.quantize.html).

For Inkling, start with the community conversion's conservative policy:

- quantize routed-expert matrices only;
- keep attention, router, shared experts, embeddings, normalization, relative
  projections, and SConv weights in BF16;
- use affine 4-bit, group size 64; and
- never derive an MLX quant from the official NVFP4 artifact when BF16 is
  available, because that would be a lossy requantization.

The measured 4-bit artifact and MLX affine storage formula imply approximately
1.870 TB of BF16 source tensors are quantizable and 34.5 GB remain BF16. Holding
the same predicate constant gives these rough storage targets:

| Routed-expert affine precision | Estimated total weights |
| --- | ---: |
| 2-bit, group 64 | 304 GiB |
| 3-bit, group 64 | 413 GiB |
| 4-bit, group 64 | 522 GiB (matches measured artifact) |
| 5-bit, group 64 | 631 GiB |
| 6-bit, group 64 | 740 GiB |
| 8-bit, group 64 | 957 GiB |

These are capacity estimates, not quality endorsements. Two- and three-bit
profiles need evaluation, and the current community 4-bit artifact itself is
not yet logit-verified. MLX's sensitivity-based dynamic quantization can produce
mixed-bit profiles, but for a frontier model the sensitivity result should be
computed and certified once per model revision, stored as a small profile, and
then applied deterministically by every stage. Recomputing sensitivity during
every cold load would be too expensive.

Different stages may use different precision when hardware differs. The chosen
per-stage profile must be part of the topology/model identity so that a cached
stage is reproducible and correctness evidence names the exact numeric model.

### Cold-load memory contract

Load-time quantization only makes small nodes viable if it is streamed. For
Inkling layers 30..33:

- BF16 input ranges: 109.84 GiB;
- resulting mixed affine-4 stage: 32.22 GiB; and
- largest single BF16 source tensor: 18.00 GiB.

A whole-stage loader would need BF16 input plus quantized output and fail on a
128 GB node. A carefully bounded loader targets the accumulated 32.22 GiB
output plus one source unit and quantization scratch, but the current
full-tensor copy path does not yet meet that contract for an 18 GiB Inkling
expert bank. Inkling may need expert/row slabs or a zero-copy managed-array
seam. The cold path should:

1. range-fetch one bounded tensor or expert slab into a temporary/mmap buffer;
2. create the MLX source array;
3. quantize according to the certified per-tensor profile;
4. evaluate and append the packed tensor/scales/biases to the derived cache;
5. release the BF16 source buffer; and
6. continue with the next tensor.

The same rule applies to disk: do not retain an entire BF16 stage unless the
operator asks for it. A derived cache can approach `quantized stage + largest
source tensor`, rather than `BF16 stage + quantized stage`.

### Approximate Inkling 4-bit deployment shapes

The 521.97 GiB text-weight artifact plus Inkling's long-context cache and load
scratch makes aggregate memory, not raw model size alone, the constraint.
At full 1M context, BF16 KV is approximately 44 GiB: eleven global-attention
layers each retain about 4 GiB, while the 55 sliding layers retain only their
512-token windows (about 220 MiB combined). SConv state is comparatively small.

Assuming balanced stages and the 18 GiB largest source tensor:

| Topology | Weight share/node | Assessment before measured runtime overhead |
| --- | ---: | --- |
| 2 × 512 GB | ~261 GiB | Comfortable capacity; simplest plausible full-context shape |
| 3 × 256 GB | ~174 GiB | Comfortable capacity |
| 4 × 192 GB | ~131 GiB | Plausible |
| 5 × 128 GB | ~104 GiB | Too tight once 18 GiB load scratch and KV are included |
| 6 × 128 GB | ~87 GiB | Plausible but needs measured allocator/kernel headroom |
| 8 × 128 GB | ~65 GiB | Safer first 128 GB-node target |
| 12 × 64 GB | ~44 GiB | Too tight during 18 GiB source-tensor quantization |
| 16 × 64 GB | ~33 GiB | Plausible capacity; stage latency may dominate |

Shorter context materially reduces the KV portion. These are feasibility
estimates, not throughput claims; MoE dispatch performance, per-stage latency,
and the MLX boundary fence still need measurement.

## Frontier-family prioritization

SafeTensors acquisition is general, but MLX execution remains family-specific.
The measured candidates suggest this order:

1. **Qwen/Llama**: finish the partial loader and two-stage correctness proof.
2. **Nemotron-H Nano 30B-A3B**: best next frontier proof because
   `safemlx-lm` has matching public layer/cache structures and an affine rank-3
   expert runtime. Its public SafeTensors path still needs bounded quantized
   ReLU2 expert-bank assembly and Mamba/recurrent boundary certification.
3. **Inkling text backbone**: safemlx already has whole-model text/multimodal
   execution; expose a partial-stage surface and use Transformers as the parity
   oracle rather than porting the family again.
4. **Inkling multimodal + MTP**: add first-stage towers and optional predictor
   layers after text correctness.
5. **Kimi K2.6, GLM-5.2, DeepSeek V4**: all are viable range-download targets,
   but each requires a new or substantially updated MLX family and native
   support for its existing compressed format or a canonical BF16 source.

Not every frontier repository should be requantized at load. Kimi K2.6's
published checkpoint is already about 554 GiB, and DeepSeek V4 Pro is already
FP8. Preserve a compatible calibrated source encoding when the local backend
supports it; use BF16-to-local-quant only when it is the cleanest compatible
source path.

Nemotron 3 Ultra is a distinct follow-up, not a larger drop-in Nano checkpoint.
Its public config describes 108 layers through `layers_block_type`, 512 experts,
and `moe_latent_size=2048`; it lacks the Nano fields consumed by the pinned
safemlx implementation. The Ultra row above proves range locality only.

## Recommended artifact identity

The durable identity should be:

```text
source repo + immutable revision
+ selected tensor names and source byte ranges
+ model-family implementation revision
+ stage range / embedding / readout / modality ownership
+ per-stage quantization profile
= derived stage cache identity
```

The cache is evictable derived data. The upstream checkpoint remains the source
of truth, and a small certified quantization/profile manifest replaces a large
published layer-package repository. Activation wire dtype belongs in topology /
deployment identity and correctness evidence, not in the weight-cache key,
because it does not alter the derived packed weights.

## Next proof

1. Add capacity/eviction ownership to the host derived-stage cache, then decide
   whether a local request-to-recipe locator should remove even the warm
   metadata probes.
2. Extend the proven single-layer Nemotron-H Nano `StageEngine` adapter into a
   hybrid stage while making recurrent/attention state explicit on the wire.
   Keep Ultra gated behind its separate latent-MoE family implementation.
3. Measure the MLX eval/readback/codec boundary fence independently at frontier
   residual widths and prefill sizes.
4. Expose the existing safemlx Inkling text decoder as one stage, prove
   Transformers parity, then add stage ranges and network execution.
