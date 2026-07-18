# MLX as a Skippy Stage Engine — Deep Dive and Plan

## Status: implemented prototype and remaining research plan

This document evaluates using Apple **MLX** (via the Rust `safemlx` / `safemlx-lm`
crates) as an alternative inference engine behind Skippy's staged execution
runtime, and proposes a phased plan.

It combines a read of the current Skippy code (`skippy-ffi`, `skippy-runtime`,
`skippy-server`, `skippy-topology`), a read of the `safemlx` fork
(`../safemlx`), a read of goose's MLX backend (`../goose/crates/goose-local-inference`),
and a second-opinion review from an external model grounded against live
MLX/mlx-lm/safemlx documentation.

**Update — a Phase-2 solo-serving spike has now run on Metal** (`spikes/mlx-solo/`,
branch `micn/mlx-redux`). It confirms the core workflow claim end to end: Qwen3-0.6B
on Apple-Silicon Metal at **321 tok/s** (bf16) and **~604 tok/s** (4-bit), where
**JIT-quantize-on-load matches a pre-quantized artifact** (604 ≈ 603 tok/s) — so
quantizing on load is free at inference time. The goose baseline (source precision)
needs **zero fork patches**; two small `safemlx-lm` fixes are only needed to go
beyond it (JIT quant + loading arbitrary mlx-community repos) and are upstream-PR
candidates. See `spikes/mlx-solo/FINDINGS.md`; results are folded into §5.3,
Phase 2, and §9.

**Update — exact stage-local SafeTensors materialization and dense split
execution are now proven**
(`spikes/mlx-safetensors-stages/`). The SafeTensors index plus per-file headers
are sufficient to map a stage to exact HTTP byte ranges, and Hugging Face honors
those range requests. This materially changes the split artifact conclusion:
nodes do not need published stage packages or even complete source shards. On
Inkling BF16, four layers contain 109.84 GiB of tensors scattered across 942.99
GiB of shard files; exact ranges avoid 833.15 GiB. A SmolLM2-135M proof then
materialized two partial files (layers 0..15 and 15..30), loaded each directly
into MLX, and matched unsplit logits exactly for prefill plus eight decode steps
through Skippy's real F16 and F32 binary activation codec. Direct range-to-
quantized artifacts is now also proven for dense Llama below. The remaining
frontier artifact gate is bounded family-specific expert transformation and a
managed cache rather than the basic SafeTensors range mechanism.
See `spikes/mlx-safetensors-stages/FINDINGS.md`.

**Update — the first engine-neutral, multi-process stage chain is now proven.**
The new `skippy-engine` crate defines a runtime-neutral `StageEngine` contract;
`skippy-server::engine_transport` carries that contract over the existing
binary stage protocol; and `MlxStageEngine` runs a partial layer range with its
own KV cache. Two real processes, each given only one 155.28 MiB partial
SmolLM2 artifact, reproduced the eight-token whole-model reference exactly over
F16 residuals. Their post-proof RSS was about 189 MiB each. This closes the
dense execution and process-boundary proof. Host topology selection, advanced
cache/session operations, additional staged families, and bounded
range-to-derived-cache quantization remain. See
`crates/skippy-engine-mlx/STAGED_EXECUTION.md`.

**Update — whole-model and explicit mesh serving are now integrated.** The
ordinary `mesh-llm serve --model` path can select MLX for a complete Hugging
Face SafeTensors model, automatically quantize eligible unquantized dense
weights to affine-4 at load time while preserving frontier/pre-quantized
representations, and serve streaming or non-streaming OpenAI chat. The explicit `--split`
path avoids the coordinator's full checkpoint download, advertises an additive
MLX stage capability, derives per-host range-only affine artifacts, and drives
the resulting chain from the same OpenAI surface. A real two-host 29/1-layer
SmolLM2 run completed with roughly 70 MB and 17 MB stage artifacts while the
remote never stored the roughly 269 MB source checkpoint. Planning now uses
header-derived per-layer estimates so large embedding/readout tensors are not
hidden by an equal-layer average. Automatic split selection and partial
executors beyond dense Llama remain open. See
`crates/skippy-engine-mlx/SERVE_INTEGRATION_STATUS.md`.

**Update — host `StagePrepare` / `StageLoad` now consumes range-only MLX
stages.** An immutable `hf-model://org/repo@<commit>` request is validated
before network work, checked against a topology-wide checkpoint identity,
derived into or reused from the recipe-keyed quantized cache on a blocking
worker, strict-loaded into `MlxStageEngine`, and served through the existing
Skippy binary wire. `StageLoad` requires that prepared entry and cannot derive
or download tensor payloads on a miss. A clean-cache SmolLM2 proof ran both
15-layer ranges through
`spawn_stage_control_loop`, reproduced the affine-4 eight-token reference, and
stopped both stages without retaining dense stage artifacts. At that
checkpoint, topology production, capability advertisement, remote two-node
proof, and cache eviction remained; the newer integrated update above
supersedes the first three items. Cache eviction still remains.

**Update — partial MLX stages now JIT-quantize one tensor at a time.** The
pinned safemlx strict loader already contains the required bounded lazy-graph
seam: visit one tensor, quantize, `eval`, synchronize, install packed parameters,
repeat. A whole-model affine-4 reference and two independently quantized
SmolLM2 stages generated the same eight tokens over F16 residuals. Post-proof
RSS was about 85.3 and 85.9 MiB per stage. This bounds the lazy graph, not
physical copies: TensorView conversion, stream copies, mmap pages, and MLX
scratch all require explicit high-water measurement. The newer direct-derived
host path below removes the intermediate BF16 stage.

**Update — exact ranges can now be consumed sequentially without a BF16 stage
artifact.** `model-hf` exposes each selected tensor as an ephemeral, valid
one-tensor SafeTensors file, verifies the pinned source identity, and removes
that file before downloading the next tensor. This engine-neutral source seam
is consumed by the Llama builder below; managed cache hits and frontier-family
transforms remain. On the pinned SmolLM2 layer-14 proof, it fetched 7,080,192
tensor bytes from a
269,060,552-byte source shard while the largest temporary file was 1,769,584
bytes; the temporary directory was empty at completion. A prepared visit makes
the verified config and checkpoint identity available before tensor callbacks,
and macOS/Unix advisory locks safely scavenge crash-abandoned visits.

**Update — production range planning now covers the `nemotron_h` architecture
used by Nemotron 3 Nano.** The planner recognizes its `backbone.layers.*`,
embedding, final-norm, and readout paths and falls back to JSON5 for Hugging
Face configs containing bare non-finite values.
For pinned NVIDIA 30B-A3B Base BF16 layer `1`, it selected 261 tensors totaling
2,594,936,576 bytes from a 4,991,210,024-byte shard; the largest tensor was
19,955,712 bytes. No tensor payload was fetched by this proof. This removes the
acquisition blocker for the next bounded expert-pack experiment.

**Update — one real Nano MoE layer now derives and executes through the shared
stage contract.** The production
builder recognizes split ReLU2 experts, quantizes one expert matrix at a time,
and writes each result into preallocated rank-3 affine banks. A two-expert test
proves this incremental layout is byte-identical to quantizing a stacked bank.
For pinned layer `1`, 2,594,936,576 source bytes became 730,324,736 affine4
tensor bytes (258 source matrices quantized, three tensors dense) in 199.70
seconds. Maximum RSS was 822,165,504 bytes; the only source payload retained at
any instant was one tensor, at most 19,955,848 bytes. A validation command then
strict-loaded safemlx's actual layer-1 `TransformerBlock` and executed a finite
nonzero `[1, 1, 2688] -> [1, 1, 2688]` forward pass. `MlxStageEngine` now
auto-detects that artifact, strict-loads only the selected block, accepts the
normal F32 residual activation, computes in BF16, and returns F32 residuals.
Its result matched a direct block execution within `atol=1e-4`, `rtol=1e-4`;
across repeated validation runs, worst observed max absolute and relative
differences were `1.1920929e-7` and `1.8225228e-5` (the relative metric excludes
reference magnitudes at or below `atol`). Two session IDs plus an independent
reset/reuse comparison of session 1 all passed. Runtime MLX active/peak memory
was 730,404,608 / 811,763,256 bytes.
The comparison command loads a direct reference and then the stage, so its
process RSS high-water is not representative of a single serving stage.
General hybrid-stage execution is still gated on recurrent/attention state and
boundary work.

**Update — that real Nano layer now crosses the Skippy binary wire.** An
intentionally unnecessary loopback proof sends a deterministic
`PrefillFinalEmbd` residual through the real affine-4 layer-1 `MlxStageEngine`
and then into a fabricated capture/final engine. The capture asserts the
forwarded execution kind, session, complete token/position sidebands, and
activation shape;
returns a sentinel prediction; and observes the forwarded session reset before
Stop/ACK completes. At 32 tokens, F32 matched the direct block with maximum
absolute error `2.3841858e-7` at `atol=1e-4`, `rtol=1e-4`. F16 produced maximum
absolute and relative errors `0.000923872` and `0.00048756658` at
`atol=5e-4`, `rtol=1e-3`. The boundary payloads were 344,064 and 172,032 bytes.
Runtime MLX active/peak memory was 730,404,608 / 820,697,688 bytes. These are
empirical thresholds for one layer and one deterministic prefill, not family
certification. The input is exactly representable in F16, so the F16 delta
primarily measures the output boundary.

This is concrete codec, forwarding, reply, and control-chain evidence around
one real frontier layer. The adjacent final stage and its three-layer topology
are synthetic harness devices. It is not evidence for two real Nemotron
stages, decode, recurrent state, host/QUIC placement, or full-model logits.

**Update — a metrics-backed frontier-width boundary-fence runner is ready.**
The release `mlx-stage bench-boundary` command separately times synthetic MLX
add completion/eval, host readback and buffer copy, production F32 or F16
activation encode, and post-receive reconstruction. It calculates real paired
per-iteration boundary and encode-plus-decode distributions, gates codec
correctness, and does not start telemetry export until timed work is complete.
It requires explicit metrics-server HTTP and OTLP endpoints, emits bounded
nonblocking spans, fails on loss or canonical count mismatch, finalizes the
run, and writes the canonical metrics-server report. This is an independent
synthetic fence and codec measurement, not model compute, message framing,
TCP, QUIC, or a complete network gate. The reproduction command and evidence
contract are in `crates/skippy-engine-mlx/STAGED_EXECUTION.md`.

**V2 result.** Commit `d381bbd3` produced 20 completed canonical runs / 1,600
spans with no telemetry loss. For 512-token boundaries at widths 4K, 8K, and
16K, F16 halved payloads while adding 4.77, 9.60, and 19.26 ms over F32 codec
work. With conversion and transfer serialized, those three cells independently
place the effective-payload break-even near 0.81 GiB/s (7.0 Gbit/s): use F16
below that measured link rate and F32 above it, subject to a real TCP/QUIC test
and any conversion/transfer overlap. The 16K cell measured 32 MiB / 0.927 ms
for F32 versus 16 MiB / 20.187 ms for F16. This makes wire dtype a hardware and
link policy choice, not a model-format constant.

The next runner, `mlx-stage bench-tcp-boundary`, now wraps the production
engine-neutral Skippy TCP server around the synthetic sink. Its end-to-end
sample covers sender activation encode through framing, loopback TCP,
engine-transport reconstruction, and predicted reply. This tests whether the
codec-only policy survives actual Skippy framing and host copies; it still does
not stand in for QUIC or a remote link.

**Loopback result.** Commit `6350e3a9` produced eight completed canonical TCP
runs / 160 spans with zero telemetry loss. At 512 tokens, F32/F16 round-trip
p50 was 2.764/8.086 ms at width 4K, 5.655/13.601 ms at 8K, and 4.936/25.398 ms
at 16K. F32 won every pair on this high-bandwidth loopback path, which is the
direction predicted by the codec-only ~7.0 Gbit/s break-even. The non-monotonic
F32 16K p50 and broad p95 make this evidence a framing/host-copy validation,
not a link-throughput estimate. A controlled remote TCP/QUIC sweep is still
required before automatic wire-dtype selection.

**Update — the two-host TCP runner is now fail-closed.** TCP boundary schema v2
can connect to a separately running production `engine_transport` sink. A
unique per-run wire session forces a fresh first-activation validation even
when the foreground sink is reused, and the sink returns the measured
exact-F32 / bounded-F16 error as an acknowledgement required by the sender.
Connect/READY and round-trip IO are bounded. Canonical telemetry records only
the neutral `external_tcp` mode, never the target address. The benchmark sink
is plaintext and unauthenticated, so it is restricted to trusted private
networks. Until sink revision is added to READY, controlled evidence must copy
and independently hash the identical release artifact on both hosts.

**Two-host result.** Commit `27bd5880` completed 10 canonical external-TCP
runs / 200 spans between an M5 Max sender and M4 Max sink with identical binary
SHA-256, explicit validation acknowledgements, and zero telemetry loss. The
receiver firewall required an SSH local forward, so the measurements include
SSH tunnelling and are not raw-LAN or QUIC evidence. F16 and F32 were effectively
tied at the real 2,688×32 Nemotron boundary; F16 reduced p50 by about 29% at
4K×512 and 31% at 8K×512. A fresh-tunnel 16K repeat favored F16 by about 40%,
but the first 16K F16 run suffered multi-second tunnel tails and lost badly.
This closes the cross-host production-framing functional proof and confirms
that payload reduction can matter on a constrained path. It does not close the
wire-dtype policy gate: direct LAN/QUIC, repeated interleaved cells, and
pipeline-overlap measurements remain.

**Update — two real stages now execute across two hosts.** The M4 Max
independently fetched only SmolLM2 layers `15..30` (137 range requests,
162,827,136 tensor bytes) and produced a 45,889,726-byte affine-4 artifact whose
three shard hashes matched the M5 Max derivation. A real M5 Max `0..15`
`MlxStageEngine` chained through an SSH local forward to that real M4 Max final
stage. Two successive F16-wire runs and one F32-wire run all reproduced
`[260, 2240, 314, 253, 1379, 282, 25801, 28]`. A copied MLX executable also
requires the build-generated `mlx.metallib` beside it (about 157 MiB in this
pinned build); `just mlx-stage-build` now exports that sibling resource. This
closes manual two-host small-Llama execution and per-host range derivation, not
mesh-managed placement, OpenAI orchestration, or raw LAN/QUIC transport.

The derivation memory bound is the final packed routed bank, not one expert:
six preallocated payload buffers total 718,405,632 bytes. Moving those buffers
to a disk-backed random-write spool is the next step if preparation RSS must
stay near the largest individual source tensor. The finite forward plus strict
parameter coverage proves artifact assembly and executability; quantization
quality still needs a dense-BF16 or Transformers reference parity oracle; this
new gate proves the stage wrapper agrees with direct execution of the same
affine artifact.

Nemotron 3 Ultra is not that next executable target. Its current public config
uses a 108-layer `layers_block_type` latent-MoE design with 512 experts and
`moe_latent_size=2048`, while the pinned safemlx `nemotron_h` implementation is
the 52-layer Nano schema driven by `num_hidden_layers` and
`hybrid_override_pattern`. Ultra range measurements remain valid storage
evidence, but execution needs separate model-family work.

**Update — direct exact-range to bounded affine stage artifacts is proven for
Llama.** `mlx-stage derive` now consumes that prepared visit, quantizes/evaluates
one rank-2 matrix at a time, serializes packed results into bounded SafeTensors
shards, and records the checkpoint/plan/quantizer identity and output hashes.
Two direct-derived SmolLM2 halves were about 45.89 MB each in three shards,
versus about 162.83 MB of dense ranges fetched per half. Largest source temp was
56.62 MB, MLX peak-active memory was 72.55 MB, max RSS was about 140.8 MB, and
macOS peak footprint was about 241 MB. The two derived stages reproduced the
established affine-4 tokens exactly. The v1 builder deliberately requires
`model_type=llama` and rejects pre-quantized sources, rank-3 weights, and
incompatible matrix dimensions. This closes the dense-Llama artifact-builder
proof, not Nemotron expert packing or Inkling's transformed rank-3 path.

**Update — the derived-stage cache seam is proven.** `mlx-stage derive-cached`
uses the derivation recipe as a destination identity, serializes competing
builders with an advisory lock, and validates the output-content plus per-shard
hashes on hits. A cold pinned layer-14 run made 9 tensor payload requests; the
warm run made 0, skipped quantization, and used about 17.8 MB max RSS. The warm
path still fetches lightweight config/index/header metadata to reconstruct the
strong recipe.

**Update — the host now owns the direct-derived load path and carries the
quantization choice.** The additive stage-load field supports `auto`, affine
4-bit, affine 8-bit, and MXFP4. Missing values from older peers map to `auto`;
unknown values fail closed. Apple Metal currently maps `auto` to affine
4-bit/group-64, and the resolved recipe remains identity-bound. A cold two-half
host lifecycle fetched 162,825,984 and 162,827,136 source tensor bytes and wrote
45,859,713 and 45,861,308-byte derived artifacts. It passed in 120.74 seconds;
the immediate validated-cache run passed in 8.87 seconds with 258,162,688 B max
RSS. `MESH_MLX_DERIVED_CACHE_DIR` overrides the host cache root. This was the
pre-integration checkpoint; eviction remains, while the newer update above
records explicit topology selection and the mesh-managed two-node proof.

Prepare is the only lifecycle operation allowed to build a missing entry;
Load validates and consumes the prepared artifact or fails closed.

The host additionally verifies the topology checkpoint claim from metadata
before any tensor payload is fetched. Prepare cancellation reaches recipe-lock
waits and the range visitor, with checks between payload requests and around
each quantization callback. In-flight HTTP or MLX work is cooperative rather
than preemptive. Inventory responses echo the requested profile, and
preparation plus running status carry it, preventing (for example) an affine-4
preparation from satisfying an affine-8 load. Old peers omit these additive
fields and therefore mean `auto`; automatic non-default placement must require
a peer that advertises the new semantics. Cache-hit validation streams each
shard once while checking both its shard hash and the aggregate content digest.

The pinned safemlx revision also already includes whole-model Inkling text,
vision, and audio execution. Earlier notes that called for porting Inkling were
stale. The remaining Inkling work is a partial-stage API plus wiring its
existing quantized grouped-expert runtime into the constructor and transformed
loader, not a model-family implementation from scratch.

---

## 1. Bottom line

MLX is a **credible second engine** for Skippy, and `safemlx-lm` is a
surprisingly good fit because it implements each model in **pure Rust as an
explicit `embed → layers[..] → norm → lm_head` loop with a per-layer KV cache**.
That is exactly the seam Skippy needs for layer-range stage splitting, and it is
Rust-facing rather than buried in C++.

It is **not** a drop-in replacement for the patched llama.cpp C ABI. The engine
boundary Skippy actually depends on is much larger than "generate tokens": it is
a **staged execution contract** (activation frames in/out, KV page
export/import, layer-range partial load, chunked prefill, single-token decode,
batched verify, trim/checkpoint, tokenizer/chat helpers).

**Two distinct reasons to adopt MLX — keep them separate:**

1. **Workflow / artifact win (biggest near-term value):** MLX loads HF
   **safetensors directly** and can **JIT-quantize on load**, so we can serve
   any supported model at a chosen bit-width *immediately* — no waiting for a
   published GGUF and no pre-run quant pipeline. This is mostly independent of
   the hard split work and pays off first in **solo serving** (§5.3).
2. **Apple-Silicon compute in a chain:** MLX runs straight to Metal and can add
   Apple-Silicon nodes to a staged split — but this is the harder, later payoff,
   gated on partial-load and boundary-fence behaviour.

**Recommendation:** introduce a Rust `StageEngine` trait, keep the existing C ABI
as the `LlamaStageEngine` adapter, and add an Apple-Silicon-gated
`MlxStageEngine`. Do **not** extend the llama.cpp C ABI to host MLX, and do
**not** invent a separate MLX network protocol. For split serving, treat the
immutable upstream SafeTensors checkpoint plus a small quantization profile as
the source of truth; range-fetch, optionally quantize, and cache only the local
stage. Published engine-specific layer packages become an optional prewarmed
optimization rather than a prerequisite. Gate execution behind the remaining
go/no-go work: **bounded-memory stage materialization / partial model loading**
and **per-token boundary fence latency**. The small-model proof also establishes
that a receiver must restore the model compute dtype after decoding the wire
dtype; numeric F32 residual values left as an F32 MLX array change downstream
Metal arithmetic for a BF16 model.

---

## 2. What the Skippy "engine" boundary actually is

Skippy's engine is not a token generator; it is a staged runtime. The contract
lives in `crates/skippy-ffi/src/lib.rs` (raw ABI) and is wrapped safely in
`crates/skippy-runtime/src/lib.rs`. The essential surface a new engine must
satisfy:

**Stage-aware load** (`RuntimeConfig`, `skippy-runtime/src/lib.rs:840`):
- `stage_index`, `layer_start`, `layer_end` — this stage owns a contiguous
  layer range only.
- `include_embeddings`, `include_output` — whether this stage owns the embedding
  table and/or the final norm + lm_head.
- `filter_tensors_on_load` — the intent that a stage should load **only** its
  tensors, not the whole model.
- backend device selection, KV cache dtype, ctx size, batch/ubatch, lanes.

**Activation frame I/O** — the wire contract between stages
(`ActivationFrame` / `ActivationDesc`, `skippy-runtime/src/lib.rs:1046` and
`:1109`):

```
ActivationDesc {
  version, dtype (F32|F16|BF16), layout (TokenMajor|Opaque),
  producer_stage_index, layer_start, layer_end,
  token_count, sequence_count, payload_bytes, flags
}
ActivationFrame { desc, payload: Vec<u8> }
```

- Stage 0 takes token IDs, runs its layers, emits an activation frame.
- Middle stages import a frame, run their layers, emit a new frame.
- Final stage runs last layers + readout, samples, returns the **predicted token
  directly** to stage 0 (generation-3 protocol, see `skippy-server/README.md`).

**Execution calls** (`skippy-runtime/src/lib.rs`):
- `prefill_chunk_frame*` (`:2998`), `decode_step_frame_sampled*` (`:3236`),
  `verify_tokens_frame*` (`:3557`), `copy_output_activation_frame` (`:3652`).
- Sampled variants carry `SamplingConfig` (penalties, logit bias, grammar).

**KV / state movement** (`skippy-runtime/src/lib.rs`):
- `export_kv_page` / `import_kv_page` (`:3866`, `:3956`) with
  `RuntimeKvPageDesc` (`:1114`) — k/v dtype, row bytes, token range, layer range.
- `export_state` / `import_state` (`:3729`), full-state and recurrent-state
  variants, `save_prefix` / `restore_prefix`, `trim_session`, checkpoint/restore.

**Tokenizer / chat / introspection**:
- tokenize/detokenize, EOG check, chat-template apply (incl. JSON tools path),
  chat-response parse, model-info tensor enumeration, GGUF slice writing.

**ABI is versioned and feature-probed** (`skippy-ffi/src/lib.rs:1`): ABI
`0.1.30`, with a feature bitmask (`RUNTIME_SLICE`, `LAYER_PACKAGE`,
`ACTIVATION_FRAME`, `BATCH_VERIFY_FRAME`, `SESSION_CHECKPOINT`,
`NATIVE_MTP_N1`, …). This is the model for how MLX capabilities should be
advertised: **probed, not assumed**.

**Key structural gap:** there is **no Rust `trait`** abstracting this today.
`skippy-server` binds the concrete `StageModel` / `StageSession` FFI structs
directly (`crates/skippy-server/src/frontend.rs:46`, `runtime_state.rs:26`).
Introducing that trait is the enabling refactor for any second engine.

---

## 3. What MLX / safemlx actually gives us (evidence)

### 3.1 The layer-split seam already exists in `safemlx-lm`

Every model in `../safemlx/safemlx-lm/src/models/*.rs` is a Rust module with the
transformer decomposed into public fields and an explicit forward loop. From
`qwen3.rs`:

```rust
pub struct Qwen3Model {
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    pub layers: Vec<TransformerBlock>,   // per-layer blocks
    pub norm: nn::RmsNorm,
}
// forward:
let mut h = self.embed_tokens.forward(inputs, stream)?;
for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
    h = layer.forward(/* x=h, mask, cache=c */, stream)?;
}
self.norm.forward(&h, stream)   // then lm_head at the Model level
```

`pub embed_tokens` / `pub layers` / `pub norm` / `pub lm_head` are exposed across
`qwen3`, `llama`, `gpt_oss`, `gemma4`, `lfm2`, `nemotron_h`, `qwen3_5_moe`, etc.
The per-layer KV cache is a `Vec<Option<C>>` threaded through the loop
(`safemlx-lm/src/lib.rs`, `cache.rs`), and blocks accept an explicit mutable
cache slot. This means running **only** `layers[start..end]` and resuming from an
imported hidden state is mechanically straightforward — no C++ surgery.

### 3.2 Activation observe / intervene hooks

`safemlx-lm/src/inspection.rs` defines `ActivationObserver` with
`observe(name, &Array)` and `intervene(name, &Array) -> Option<Array>` at block
boundaries ([inspection API](https://docs.rs/safemlx-lm/latest/safemlx_lm/inspection/)).
This is useful plumbing/debugging, **but it is not a stage ABI** — it is
name-based and per-tensor. For production we want a first-class
`forward_range()` / `resume_from_hidden()` path, not a reliance on observer
names.

### 3.3 KV cache primitives

`safemlx-lm/src/cache.rs` defines `KeyValueCache` (offset, max_size,
`update_and_fetch`), with `ConcatKeyValueCache`, `SlidingKeyValueCache`,
quantized variants, and `truncate(len)`. Cache state is MLX arrays in unified
memory. This gives us the raw material for export/import/trim — but the state
layout is MLX/model-specific and **not** interchangeable with llama.cpp's ggml
page format.

### 3.4 Loading, quant, and formats

`safemlx-lm` loads Hugging Face-style dirs (`config.json` + `tokenizer.json` +
safetensors) and also **GGUF** (`Array::load_gguf_with_metadata`,
`models/mod.rs:1123`), with a **strict loader** (`weights.rs:155`) that errors on
missing/unused tensors unless explicitly allowed. MLX quant is affine
packed-weight-in-safetensors; there is load-time quantization from unquantized
F32/F16/BF16.

### 3.5 Distributed primitives exist but are unwrapped

`../safemlx/safemlx-sys/src/mlx-c/mlx/c/distributed.h` binds
`all_gather / all_sum / all_min / all_max / send / recv / sum_scatter` and
distributed groups. **The safe `safemlx` layer does not wrap them yet.** This
matters for tensor-parallel within a node, but Skippy's cross-machine model is
its own QUIC activation-frame protocol — we do **not** want to depend on MLX
distributed for the mesh boundary.

### 3.6 goose already ships an MLX backend — what we can reuse

`../goose/crates/goose-local-inference/src/mlx.rs` (1044 lines) uses `safemlx-lm`
(`LoadedModel::load`, `generate`, Gemma4 MTP draft/speculative) as a **single
node** backend behind goose's own `LocalInferenceBackend` trait, feature-gated
`mlx` on macOS. It proves safemlx-lm is production-usable for generation.

**"Right to Metal" is accurate.** goose's MLX path pulls
`safemlx { features = ["accelerate", "metal", "safetensors"] }` and runs on
`Device::new(DeviceType::Gpu, 0)` (`mlx.rs:61,94`) — i.e.
`safemlx-sys → mlx-c → MLX → Metal`, with no llama.cpp/ggml in between. (goose
*also* keeps a separate `llama-cpp-2` Metal path; the two are independent
backends.)

**What is actually reusable, and how:**

| Asset | Reuse verdict |
| --- | --- |
| `safemlx` / `safemlx-lm` crates | **Reuse directly** — this is the real shared dependency. Both goose and Skippy just depend on the published crates. No goose code involved. |
| goose's `mlx.rs` generation flow (prompt build, sampling, MTP draft/verify, streaming, stop tokens, thinking-filter) | **Reuse as a reference template, port not lift.** It is the best worked example of driving safemlx-lm for chat + speculative, but it is coupled to goose types. |
| goose's `LocalInferenceBackend` trait (`backend.rs`, 50 lines) | **Reference only.** It is goose's shape (whole-model `load_model` + `generate`), not Skippy's staged contract. Skippy needs its own `StageEngine` (§6). |
| HF download + shard/registry (`hf_models.rs`, `goose-download-manager`) | **Reference / optional adapter.** `goose-download-manager` is cleanly separable (deps are just `reqwest`/`tokio`), but Skippy already has `model-hf` / `model-artifact`; prefer extending those. |

**Coupling is the reason it is port-not-lift.** goose's `mlx.rs` and `backend.rs`
depend on `goose_provider_types` (`Message`, `MessageContent`, `ProviderError`,
`ProviderUsage`/`Usage`, `DraftStats`), `rmcp::model::Tool`/`Role`, and
`local_model_registry::ModelSettings`. Skippy speaks `ActivationFrame`, token
IDs, `SamplingConfig`, and its OpenAI frontend types instead. So the *algorithms*
(how to prefill, sample, run MTP draft/verify against safemlx-lm) transfer
directly; the *types and trait* do not.

**License:** goose is **Apache-2.0**, so porting code with attribution is fine.

**Bottom line:** the highest-leverage reuse is simply **sharing the `safemlx-lm`
crate** (and coordinating on/​contributing the stage-aware `forward_range` /
partial-load additions the fork needs — see Phase 3), plus using goose's `mlx.rs`
as the reference implementation for the single-stage generation path in Phase 2.
It is whole-model and single-stage, so it is not a template for the staged/
KV-page work.

---

## 4. Fit analysis and sharp edges

The happy path fits:

```
tokens or imported hidden state
  → optional embedding (stage 0 only)
  → layers[start..end] with per-layer cache
  → hidden state frame  (middle/non-final)
  or → final norm + lm_head → sample → token  (final)
```

Sharp edges, in rough priority order:

1. **Partial execution must mean partial loading.** Building a whole
   `LoadedModel` and skipping layers can still materialize all weights. Skippy's
   entire value proposition is fitting a big model across small machines, so a
   stage must load only its layer range (plus embeddings/readout when it owns
   them). This requires a **stage-aware loader** in `safemlx-lm` that
   instantiates `layers` for `[start..end]` and only reads matching safetensors
   shards. **This is the #1 go/no-go item.**

2. **Every model family must define its exact residual-stream boundary.**
   Embedding scale, final norm, tied vs untied output, RoPE position accounting
   across a stage cut, attention mask construction, and any per-layer-type
   sideband cannot be inferred generically. Each family is a separate
   certification (mirrors `skippy-topology` family capability records,
   `crates/skippy-topology/src/lib.rs:182`).

3. **Hybrid / recurrent models need more than hidden states.** Mamba/RWKV/gated
   DeltaNet-style layers (`nemotron_h`, `qwen3_5_moe` / `qwen3_next`, `lfm2` in
   safemlx-lm) carry recurrent state, not page-addressable KV. Skippy already
   has `export_recurrent_state`; MLX would need the analogous opaque sideband,
   or those families are restricted to non-split.

4. **The MLX model matrix ≠ the safemlx-lm matrix ≠ the Skippy-certified
   matrix.** safemlx-lm implements models individually and is young. Start with
   **dense Llama / Qwen**; do not promise arbitrary model coverage.

5. **The boundary is more than execution.** Sampling, batched verify, trim,
   checkpoint, tokenizer/chat, and state movement are all in the ABI today
   (`crates/skippy-ffi/README.md`). The trait must cover them (some can be
   engine-agnostic and moved above the engine).

---

## 5. Hard problems (with concrete approaches)

### 5.1 Lazy evaluation — the network boundary is an eval boundary

MLX is lazy and uses unified memory. A stage boundary forces materialization.
The per-token non-final-stage sequence must be:

1. Run local layer range (lazy).
2. Cast to negotiated wire dtype (`ActivationDType::F16` first).
3. Make contiguous + token-major (`ActivationLayout::TokenMajor`).
4. **Evaluate the outgoing array AND all updated cache arrays together.**
5. Get a host-readable slice, serialize into `ActivationFrame.payload`, send.

In `safemlx`: `Array::evaluated()` materializes and `EvaluatedArray::as_slice()`
gives host access (host access / save also forces eval)
([safemlx lazy-eval source](https://docs.rs/safemlx/latest/src/safemlx/lib.rs.html),
[MLX lazy-evaluation guide](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)).

Critical details:
- **Evaluate cache state even when it does not feed the outgoing hidden state**,
  or lazy cache graphs grow unbounded across decode steps.
- Unified memory removes the explicit GPU→CPU copy but **not** GPU completion,
  sync, layout conversion, or the QUIC copy.
- Final stage: evaluate the sampled token + cache; never materialize a
  hidden-state frame.
- Evaluate params once at warmup.
- Consider separate **compiled** paths for fixed-shape decode vs bucketed
  prefill (`safemlx` [`compile_with_state`](https://docs.rs/safemlx/latest/safemlx/transforms/compile/)),
  watching recompilation from shape changes.

**The benchmark that matters** is not MLX layer time; it is
`last layer → cast/contiguous → eval fence → host view → QUIC write`, per token,
at realistic hidden widths. **This is go/no-go item #2.**

### 5.2 KV cache export/import/trim

MLX cache arrays make movement possible, but mlx-lm/safemlx cache state is not
llama.cpp page-shaped (mlx-lm caches expose array state, metadata, and tail
trimming, and prompt caches serialize as safetensors —
[mlx-lm cache.py](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/cache.py)).
Do **not** reuse `RuntimeKvPageDesc` for MLX; keep that in the llama adapter.
Define an engine-general **cache codec** with a versioned descriptor:

```
engine + model digest, architecture revision
layer range, token range + absolute position
cache kind (concat | sliding/rotating | quantized | recurrent)
segments: { role, layer, dtype, shape, strides/layout, payload }
cache-specific metadata (rotating offset, quant scales/biases)
```

- Export: slice token range, order rotating caches temporally, contiguous,
  evaluate, serialize.
- Import: validate engine/model/range/layout, rebuild arrays, restore offsets.
- Trim: offset change is cheap; reclaiming/compacting memory needs slicing.
- **Quantized KV** needs packed values + scales/biases, not just row sizes.
- **Do not promise KV interop between llama.cpp and MLX.** Import requires same
  engine + model digest + quant + arch revision + cache policy.

### 5.3 Artifact strategy: JIT safetensors vs pre-quantized layer packages

This is arguably the **strongest first reason to adopt MLX**, and it is largely
independent of the hard split work. The benefit is really **two separate
things**, and they behave very differently for solo vs split serving:

- **(A) Source freedom** — load HF **safetensors** directly instead of waiting
  for someone to publish a GGUF (or running our own GGUF quant pipeline first).
- **(B) JIT quantization** — quantize at load time (`with_quantization(Q4/…)`)
  instead of ahead of time.

(A) applies equally to solo and splits (it's just "which file do we load").
(B) is where solo and splits diverge sharply.

**What safemlx-lm actually supports (confirmed):**
- `ModelLoadOptions::with_quantization(...)` quantizes eligible dense weights
  **one tensor at a time** on load; checkpoints already carrying matching quant
  metadata load directly without requantizing
  (`../safemlx/safemlx-lm/src/models/mod.rs:137`).
- Sharded safetensors are understood via `model.safetensors.index.json`'s
  `weight_map` (tensor name → shard file), so the loader knows which shard holds
  `model.layers.<n>.*` (`../safemlx/safemlx-lm/src/weights.rs:463`). MLX quant is
  affine packed-weights-in-safetensors, matching mlx-lm's converter
  ([mlx-lm quantized loading/conversion](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/utils.py)).

#### Solo serving → pure win, lands first

Download BF16/FP16 safetensors → `with_quantization(Q4)` → serve. No wait for a
published GGUF, no pre-run of the quant pipeline. Any dense model safemlx-lm
supports is instantly servable at a chosen bit-width. **goose already does
exactly this** single-node (`../goose/crates/goose-local-inference/src/mlx.rs`).
Near-zero new distributed work; this is the cheapest, highest-value first step.

#### Splits → works, but with four real conditions

The premise of a split is that *no node holds the whole model*, which stresses
JIT quant:

1. **Stage-aware partial load + quant.** Today safemlx-lm builds
   `0..num_hidden_layers` and the strict loader expects all params, so JIT quant
   is a *whole-model* op. A split node must instantiate only `layers[start..end]`,
   read only the shards overlapping its range, and quantize only those tensors.
   This is exactly go/no-go **Spike 1**, now with a quant step folded in.
2. **Exact tensor-range download (proven).** The `weight_map` selects source
   files, and each SafeTensors header supplies exact tensor byte offsets. HTTP
   range requests therefore avoid whole-shard overfetch. This is a modest win
   for layer-ordered Qwen/Nemotron/GLM checkpoints and a requirement for
   Inkling, whose tensors for four layers are spread across 57 BF16 files.
3. **Deterministic cross-stage quant.** Every stage must quantize *identically*
   (same algo / group-size / bits / tie handling) or the split model drifts
   numerically from the solo model. Affine quant is deterministic given its
   params, so this is achievable — but the params must be pinned and folded into
   family/topology certification.
4. **Cache the quantized slice.** Re-quantizing on every launch/replan across N
   nodes is wasteful. Skippy's identity-bound materialized cache
   (`crates/skippy-runtime/src/package/materialized_cache.rs`, keyed by
   `model_id / topology_id / stage_id / layer_start / layer_end`) is the natural
   home: first launch pays the JIT cost → materialize a per-stage quantized
   artifact → reuse thereafter.

#### The tension worth naming

Skippy's existing chain (`skippy-quantize`, layer-package repos, BF16→GGUF) is
built around **pre-quantized, exactly-sliced GGUF parts** so split nodes never
quantize at runtime. Exact SafeTensors byte ranges remove the earlier coarser-
slicing disadvantage. The remaining trade is cold-start quantization time and
temporary source precision versus a prewarmed, published quant. The two paths
should **coexist**:

- **JIT safetensors = flexible coverage path** — range-fetch only the stage,
  adapt its precision to available hardware, cache the deterministic result,
  and require no weight-republishing step.
- **Pre-quantized layer packages = optimized path** — for models served
  seriously (exact slices, no runtime quant, tailored partial download).

#### One artifact identity, two physical encodings

Use **one logical package identity, not one physical weight encoding**:

```
model identity + source revision + tokenizer/config/chat metadata + topology
variants:
  llama-gguf: GGUF parts + quant          (existing skippy-model-package path)
  mlx-jit:    HF tensor ranges + per-stage quant profile (quantize on load; cache)
  mlx-packaged: pre-quantized MLX stage shards + index (optimized split path)
```

- Make **BF16/FP16 HF safetensors the canonical source**; derive all variants
  reproducibly (this repo already has `skippy-quantize`, `model-hf`,
  `model-package`, and BF16 GGUF conversion skills).
- **Never** transcode an already-quantized GGUF → MLX quant (dequant/requant
  loses quality and still rebuilds arch metadata).
- Nodes download only their selected engine/stage variant, so catalog
  duplication need not become per-node duplication.
- For true partial download in the packaged path, stage-specific MLX
  safetensors shard/index generation is needed (parallel to today's GGUF slice
  writing).

### 5.4 Cross-machine execution

Treat **MLX as the local compute engine and Skippy as the distributed runtime.**
mlx-lm's `pipeline()` / `sharded_load()` / `send`/`recv` + `all_gather` is useful
**reference**, but it uses static ranks and MLX collectives — it is not Skippy's
QUIC activation-frame protocol with independent stage lifecycle, capability
negotiation, and direct final-token return. Keep Skippy's transport; use MLX only
for compute
([MLX distributed docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html),
[mlx-lm utils](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/utils.py),
[pipeline mixin](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/pipeline.py)).

For normal Ethernet/Wi-Fi: an 8192-wide F16 activation is ~16 KiB/token/boundary
(decode is latency-bound, not bandwidth-bound); prefill 512×8192×F16 is ~8
MiB/boundary (bandwidth matters). Pipeline parallelism does **not** speed up
single-sequence decode — only concurrent sessions / speculative spans keep stages
busy. Skippy's topology wire sizing (`crates/skippy-topology/src/lib.rs:1415`)
already models F16 = `2 × hidden_width`.

### 5.5 Platform and dependency footprint

"To the metal like goose" and "lean dep" are both achievable, but the second has
a real catch: **MLX is lean at runtime and heavy at build time.**

**Runtime footprint — genuinely lean.** goose's path is
`safemlx → safemlx-sys → mlx-c → MLX → Metal`, statically linked
(`safemlx-sys/build.rs` sets `BUILD_SHARED_LIBS=OFF`), running on
`Device::new(DeviceType::Gpu, 0)`. The vendored `mlx-c` is a ~1 MB C shim; there
is no runtime service or subprocess. Mesh can depend on the same crates for the
identical to-the-metal path — the metal-ness lives in `safemlx-sys`, nothing
goose-specific.

**Build footprint — a second heavy native lane.** `safemlx-sys/build.rs` drives
**CMake**, and the bundled `CMakeLists.txt` uses `FetchContent` to **git-clone
the full MLX C++ core from `github.com/ml-explore/mlx.git` and compile it**.
Building therefore needs CMake ≥3.25, a C++20 compiler, network to fetch MLX,
and — for Metal — Apple's `metal` shader compiler (`xcrun -find metal`, producing
`mlx.metallib`). This sits alongside the existing llama.cpp patch-queue build and
becomes another native runtime artifact under the
`MESH_LLM_DYNAMIC_NATIVE_RUNTIME` packaging model.

**Keeping it lean = isolation, not intrinsic lightness.** Put the engine in its
own crate (`skippy-engine-mlx`) gated by **both** a cargo `feature = "mlx"`
**and** `cfg(target)` (Apple Silicon, optionally Linux/CUDA). Then default,
Linux-ROCm, Vulkan, and Windows builds never pull MLX or run its CMake — exactly
how goose gates it. Lean by construction, for the platforms that don't use it.

**Support matrix — broader than macOS, but not the full llama.cpp matrix**
(from `safemlx-sys/build.rs` + `safemlx-sys/README.md`):

| Target | MLX support |
| --- | --- |
| macOS Apple Silicon | ✅ Metal + Accelerate |
| iOS / tvOS / visionOS | ✅ Metal |
| Linux x86_64 / aarch64 | ✅ CPU |
| Linux + NVIDIA | ✅ CUDA (the `cuda`/`nccl` features **panic** on non-Linux) |
| Linux + AMD (ROCm) | ⏳ not today — large **active but unmerged** upstream experiment (see below) |
| Vulkan (any) | ❌ upstream *wishlist* only, no implementation |
| Windows | ❌ (some `if(WIN32)` scaffolding in vendored `mlx-c`, no working backend) |

**Coverage is expanding, and safemlx tracks it fast.** `jbg/safemlx` is very
active (103 commits, latest 2026-07-15) and pins a recent MLX core (`v0.32.0`).
It wires in new backends quickly: the `Add CUDA support` commit landed a full
`build.rs` + CMake patch + Linux CI + `cuda.rs` module + smoke test in one go,
and there is `if(WIN32)` DLL-export scaffolding in the vendored `mlx-c`. So the
matrix above is a **snapshot, not a ceiling**.

**The gaps are gated by MLX upstream, and there are *two* gates.** safemlx does
not build backends of its own — every one of its non-`main` branches is model /
runtime / quant work, not hardware work, and `forks_count`/`network_count` are 0
with no open PRs. A new backend must therefore (1) land in `ml-explore/mlx`
(C++), and only then (2) be wired through safemlx — exactly the sequence CUDA
followed (`Add CUDA support` was safemlx *exposing* an upstream backend, not
authoring one). So hardware coverage tracks upstream MLX, delayed by the safemlx
wiring step.

**ROCm is real but not bankable yet.** Upstream MLX has a large, active AMD/ROCm
effort — PR **#2300 "[Experiment] ROCm backend"** (≈449 commits, +45k lines, open
~13 months, updated as of this writing) plus issue **#2556 "Add ROCm Support for
AMD GPUs"**. It is **unmerged and `mergeable_state: dirty`**, so it is genuine
momentum, not a shipped backend. Vulkan is only an upstream *wishlist* issue with
no implementation; Windows has scaffolding but no backend. Net: the matrix is
**expanding (CPU → Metal → CUDA, ROCm being actively attempted upstream)**, so
treat it as a moving target — but do not plan around ROCm/Vulkan/Windows until
they both merge upstream **and** appear in safemlx.

**Strategic consequence.** **Today**, the ROCm / Vulkan / Windows gaps mean
**MLX cannot be Skippy's sole engine** — which reinforces (not changes) the plan:
MLX is an **additive, feature+cfg-gated second engine**, strongest on Apple
Silicon (with Linux/CUDA a real second target, and AMD plausibly later), while
llama.cpp stays the cross-platform default. Crucially, even in the optimistic
world where MLX gains ROCm/Vulkan, the durable reason to keep llama.cpp is **not**
platform coverage but its **GGUF/imatrix k-quant maturity** and the existing
**patch-queue investment** — those are the sticky arguments; hardware coverage is
the reversible one.

---

## 6. Recommended architecture

**Option (a): a second implementation behind a Rust `StageEngine` trait.**

```
Skippy protocol / skippy-server
    └── StageEngine (new trait, engine-agnostic descriptors + byte buffers)
          ├── LlamaStageEngine → existing skippy-ffi C ABI  (unchanged)
          └── MlxStageEngine   → safemlx / safemlx-lm        (Apple-Silicon-gated)
```

Trait covers: capability discovery + model inspection; stage-aware open/load;
session lifecycle; prefill / decode / batched verify; activation import/export;
trim / checkpoint / reset; opaque-or-segmented state export/import; final-stage
logits/sampling; tokenizer/chat (where not yet lifted above the engine). Backend
arrays and native handles stay private; the trait exchanges **Skippy-owned
descriptors and `Vec<u8>` payloads**.

Rejected alternatives:
- **Extend the llama.cpp C ABI for MLX** — no. It embeds GGUF/ggml dtype and
  llama loading concepts; MLX is already Rust-facing. This would degrade a good
  native adapter into a lowest-common-denominator API.
- **Separate MLX server protocol** — no, initially. It duplicates lifecycle,
  networking, and compatibility. If Metal/MLX crash isolation later becomes
  necessary, add an **optional subprocess** implementation behind the *same*
  `StageEngine` trait, reusing the existing Skippy stage protocol — not a new
  public surface.

Crate shape (fits the repo's semantic-ownership rules):
- `skippy-engine` (new): the `StageEngine` trait + shared descriptors
  (activation frame, cache codec, capability probe). Engine-neutral.
- `skippy-runtime` becomes / provides `LlamaStageEngine` implementing the trait.
- `skippy-engine-mlx` (new, `cfg(all(target_os="macos", target_arch="aarch64"))`,
  feature `mlx`): `MlxStageEngine` over `safemlx`/`safemlx-lm`.
- `skippy-server` depends on `dyn StageEngine`, not concrete `StageModel`.

Protocol compatibility: MLX support is **additive** — a new engine capability
advertised via the existing feature-probe + gossip capability mechanism, with
llama.cpp remaining the default. No gossip/stream/ABI break. A mixed-engine chain
(llama stage ↔ MLX stage) must be a **separately certified** capability with
verified residual boundary, RoPE convention, activation dtype, and model
revision — default to **engine-homogeneous chains** first.

---

## 7. Phased plan

**Phase 0 — Spikes (go/no-go, no product wiring).** Standalone binaries in
`../safemlx` or a throwaway crate. See §8. Nothing merges to Skippy until Spike 1
(partial load) and Spike 2 (boundary fence) pass.

**Phase 1 — Introduce `StageEngine` trait (llama only).** Pure refactor: define
the trait in a new `skippy-engine` crate, implement it for the existing
`skippy-runtime` FFI, and switch `skippy-server` to `dyn StageEngine`. No
behavior change; ship this independently of MLX. Validate with existing
`skippy-correctness` and `mic-lab` runs.

> **Partially implemented on this branch.** The engine-neutral crate, an
> additive reduced binary server lane, and a dense `LlamaStageEngine` adapter
> over the existing `RuntimeState` now exist; MLX uses the same contract for the
> two-process proof. The mature llama server has not yet been switched from its
> concrete `RuntimeState` path because its broader batching, cache, MTP, and
> multimodal surface still needs capability-aware migration.

**Phase 2 — Solo MLX serving + JIT quant (the workflow win; lead here).**
`MlxStageEngine` as a single-stage/whole-model engine: open/load, session,
prefill, decode-sampled, tokenizer/chat, final-stage sampling — plus the
**source-freedom + JIT-quant** path (§5.3): download HF safetensors, quantize on
load at a chosen bit-width, serve. Port goose's `mlx.rs` generation flow (§3.6)
rather than lifting it. Wire behind `--serving-backend mlx` (parallel to the
existing skippy backend selector in `docs/SKIPPY.md`). Validate against
`skippy-correctness` vs llama.cpp logits for the same model. This delivers the
"serve any supported model instantly, no wait for quant" benefit with minimal
new distributed work, and de-risks the engine before any split work.

> **Spike done on Metal (`spikes/mlx-solo/`).** The load→generate half is proven:
> Qwen3-0.6B from raw HF safetensors, in Rust, on Apple-Silicon Metal, matching
> goose's setup exactly (`["accelerate","metal","safetensors"]`, `Device::Gpu`).
> Measured decode: **321 tok/s** source precision (bf16), **~604 tok/s** at 4-bit —
> and crucially **JIT-quantize-on-load (604) ≈ a pre-quantized mlx-community repo
> (603)**, so quantizing on load is free at inference time. The source-precision
> path (goose's baseline) needs **zero fork patches**; two small `safemlx-lm` fixes
> are only needed to go beyond it (JIT quant of a tied-embedding checkpoint, and
> loading published quant repos that omit the `mode` field) — both upstream-PR
> candidates, not mesh-llm drift (see §9). CPU is not a serving path and was not
> benchmarked as one.

**Phase 3 — Streaming stage materialization + partial load + activation
frames.** Convert the proven tensor-range plan into a bounded-memory pipeline:
range-fetch one tensor, optionally quantize it, append it to a derived stage
cache, and release the source buffer. Add `forward_range` /
`resume_from_hidden` and a stage-aware model constructor to `safemlx-lm`.
Implement `prefill_chunk_frame` / `decode_step_frame` /
`copy_output_activation_frame` producing Skippy `ActivationFrame`s. Two-stage
single-machine parity first, then two Macs over the real network.

> **Dense and JIT-quantized process proofs passed.** SmolLM2-135M was split
> 15+15 using two exact-range partial SafeTensors artifacts. F16 and F32
> `StageWireMessage` boundaries matched unsplit MLX with zero measured dense
> logit delta; two real F16-wire processes also matched the whole-model
> affine-4 token reference after tensor-wise on-load quantization. Explicit host
> stage control now derives or reuses a quantized artifact directly from exact
> tensor ranges. The additive load request carries the quantization profile,
> and clean plus warm-cache host lifecycles reproduce the affine-4 reference.
> Automatic profile-bearing MLX topology production, cache eviction, and remote
> two-node execution remain.

**Phase 4 — KV/state codec + verify + trim/checkpoint.** Implement the
engine-general cache codec (§5.2), `verify_tokens_frame` for speculative decode,
trim/checkpoint/reset. Add speculative (safemlx-lm already has Gemma4 MTP draft
as a reference).

**Phase 5 — Artifact/packaging + certification.** MLX variant in the model
package (§5.3), stage-shard partial download, per-family/quant certification into
`skippy-topology` capability records and `docs/skippy/` family docs. Mixed-engine
chain certification only if warranted.

**Phase 6 — Promotion.** Only after correctness + performance parity on the
Apple-Silicon target does MLX become a default-selectable engine for
Apple-Silicon nodes. llama.cpp remains the cross-platform default.

---

## 8. Spike gates (go/no-go before Phase 3)

1. **Partial-loading proof (DENSE + LLAMA QUANT GO, FRONTIER PARTIAL):** remote exact-range
   selection is proven, including on 1.9 TB Inkling BF16. SmolLM2 partial files
   were materialized and loaded without a complete checkpoint. The live-model
   loader quantizes/evaluates one tensor at a time, and the host now streams
   exact ranges into a derived quantized cache without retaining the BF16 stage
   slice. Still required: bounded expert-bank/slab transforms and high-water
   evidence for Inkling and Nemotron-family frontier tensors.
2. **Boundary latency breakdown (GO/NO-GO):** measure layer compute, cast,
   contiguous, **eval fence**, host readback, serialize, and receive-reconstruct
   **independently**, at hidden widths 4096/8192/16384 and token counts
   1/32/512. Decode is single-sequence latency-bound; prove the fence doesn't
   dominate.
3. **Two-stage dense parity (INITIAL GO):** SmolLM2/Llama at split 15 passed F32
   and F16 through the real binary codec with zero measured logit delta for one
   prefill plus eight decode steps. Still required: multiple split points,
   chunked prefill, 128-token decode, two processes, and cross-engine comparison.
4. **Real network run:** two Macs over Wi-Fi and 1/10GbE (Thunderbolt if
   relevant); report end-to-end tok/s + p50/p95 inter-token latency, not local
   MLX throughput.
5. **KV round-trip:** export/import multiple token pages, resume decode, compare
   logits; test trim + speculative rejection; include rotating + quantized cache.
6. **Concurrency:** multiple sessions, cancellation, repeated resets; verify
   MLX stream/array ownership under the chosen Tokio / dedicated-thread model.
7. **Compilation stability:** separate fixed-shape decode vs bucketed prefill;
   watch recompilation counts, observer overhead, long-run graph/memory growth.

Spikes 1 and 2 are more decisive than any standalone token/s benchmark.

---

## 9. Risks and unknowns

- **JIT quant is free at inference time on Metal (confirmed by spike), but CPU is
  not a serving path.** On Apple-Silicon Metal, JIT 4-bit (604 tok/s) matched a
  pre-quantized mlx-community repo (603 tok/s), and source precision ran at 321
  tok/s — so the §5.3 "serve any model instantly, JIT-quantized" claim holds with
  no runtime penalty. MLX quant matmul is Metal-optimized with no fast CPU kernel,
  so JIT quant must be gated behind a Metal (or CUDA) backend; do not expose a CPU
  quant serving path. (This supersedes an earlier CPU-only measurement.)
- **Partial load may require nontrivial changes to `safemlx-lm`** (loader + model
  constructors currently build `0..num_hidden_layers`). Upstreaming to the fork
  is likely necessary. (Highest risk for the split work.)
- **Eval-fence latency** could erode the benefit of adding Apple-Silicon compute
  to a chain, especially over Wi-Fi.
- **Model coverage churn — confirmed by spike:** safemlx-lm is young; each family
  is bespoke Rust and separately certified. The spike hit two papercuts on
  Qwen3-0.6B alone: (1) the published crate hard-enables the `metal` feature, so
  a Metal-less/CI build needs a **workspace-level** `default-features = false`;
  (2) tied-embedding `lm_head.weight` fails the *quantized* strict loader
  (dense load tolerates it). The pinned fork fixes the omitted-mode loader gap;
  mesh handles the exact tied-head rejection with a narrow native-load fallback.
  Expect more per-family.
- **Recurrent/hybrid + MoE** splitting is materially harder than dense; scope
  them out of early phases.
- **Two artifact pipelines** add storage + certification cost; mitigate with a
  single canonical BF16 source and reproducible derivation.
- **safemlx supply chain — certify a git revision behind registry requirements
  (confirmed this session).** The published crates collide version strings with the fork
  HEAD: crates.io `safemlx-lm 0.4.1` is a *different, older* codebase than the
  fork's `0.4.1` (851 vs 2221 lines in `qwen3.rs`), because the fork develops on
  a fixed version without bumping. A fork-free build against published crates
  **compiled and ran but produced gibberish for Qwen3 source precision and
  crashed on a pre-quantized repo** (`rms_norm` size mismatch) — the working
  dense-Qwen3/Llama + JIT-quant code exists only in unpublished fork HEAD. The
  `skippy-engine-mlx` manifest uses normal registry requirements while this
  workspace patches them to a **specific public git commit** based on
  `jbg/safemlx`. Root patches do not propagate to crates.io consumers, and the
  registry release lacks APIs used by the engine, so standalone published MLX
  consumers are not usable yet. The current compatibility fix is proposed
  upstream in `jbg/safemlx#2`; compatible future safemlx releases can remove
  the root patch and unblock that feature shape. This makes "track upstream +
  certify + patch" a **standing cost**, not a one-off.
- **Hardware coverage is a moving target with two gates.** New backends must land
  in upstream `ml-explore/mlx` *then* be wired through safemlx (which authors no
  backends itself). ROCm is an active-but-unmerged upstream experiment (#2300);
  Vulkan is wishlist-only; Windows has scaffolding but no backend. Do not plan
  around AMD/Vulkan/Windows until both gates clear.
- **Compat discipline:** MLX must stay additive (feature-probe + gossip
  capability); homogeneous chains by default; mixed-engine only when certified.

---

## 10. Immediate next steps

1. Add capacity and eviction ownership to the host derived-stage cache, and
   decide whether a local request-to-recipe locator should eliminate warm-path
   metadata probes.
2. Extend the explicit, capability-gated dense-Llama topology into automatic
   split selection and additional certified model-family stage adapters.
3. Extend the initial metrics-backed synthetic **Spike 2 (boundary fence)**
   matrix to real model outputs and TCP/QUIC links. Preserve the separate
   eval/synchronize, host copy, codec, and network phase evidence; do not assume
   F16 wins when CPU conversion can exceed the bytes saved on a fast link.
4. Extend the proven single-layer Nemotron-H **Nano** `StageEngine` adapter into
   a hybrid staged runtime with explicit recurrent/attention boundary state. Do
   not treat Ultra as the same runtime family. Then expose safemlx's existing
   Inkling implementation as a staged text decoder and use Transformers as the
   parity oracle.

---

## Appendix — primary sources reviewed

**This repo (Skippy):**
- `crates/skippy-ffi/src/lib.rs`, `crates/skippy-ffi/README.md` — staged C ABI
- `crates/skippy-runtime/src/lib.rs` — safe stage model/session, activation
  frames, KV/state movement
- `crates/skippy-server/src/frontend*` — stage driver, generation-3 protocol
- `crates/skippy-topology/src/lib.rs` — split planning, wire sizing, family caps
- `crates/skippy-runtime/src/package/materialized_cache.rs` — identity-bound
  stage artifact cache
- `docs/design/LLAMA_STAGE_INTEGRATION_PLAN.md`, `docs/SKIPPY.md` — why the ABI
  is shaped this way; backend-selector parity

**MLX Rust fork (`../safemlx`):**
- `safemlx-lm/src/models/{qwen3,llama,gpt_oss,gemma4,...}.rs` — per-layer forward
- `safemlx-lm/src/{cache,inspection,weights}.rs`, `models/mod.rs` — KV cache,
  observer hooks, strict/sharded loading, JIT quant
- `safemlx-sys/src/mlx-c/mlx/c/distributed.h` — MLX collectives (unwrapped)

**goose (`../goose`, Apache-2.0):**
- `crates/goose-local-inference/src/{mlx,backend,hf_models}.rs`,
  `crates/goose-download-manager` — reference MLX backend + HF download

**External (grounded via web search):**
- [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- [MLX distributed](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [safemlx docs.rs](https://docs.rs/safemlx/latest/safemlx/) ·
  [safemlx-lm](https://docs.rs/safemlx-lm/latest/safemlx_lm/) ·
  [inspection](https://docs.rs/safemlx-lm/latest/safemlx_lm/inspection/) ·
  [compile](https://docs.rs/safemlx/latest/safemlx/transforms/compile/)
- mlx-lm reference:
  [utils.py](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/utils.py) ·
  [pipeline.py](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/pipeline.py) ·
  [cache.py](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/cache.py) ·
  [deepseek_v3.py](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/deepseek_v3.py)
