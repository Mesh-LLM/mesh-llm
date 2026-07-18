# MLX serve integration status

## Draft status

The `micn/mlx-redux` draft now has two connected serving proofs on Apple
Silicon. Both use mesh-llm's ordinary model/runtime and OpenAI surfaces rather
than a standalone demonstration server.

1. Whole-model serving resolves a Hugging Face SafeTensors repository through
   the normal `mesh-llm serve --model ...` path, loads it with MLX, applies
   automatic affine 4-bit/group-64 quantization while loading eligible dense
   weights, and
   serves `/v1/models` plus streaming and non-streaming chat completions.
2. Explicit `--split` serving resolves only metadata at the coordinator,
   advertises an additive MLX stage capability, plans the ordinary stage
   topology, and makes each stage fetch and derive only its assigned tensor
   ranges. The coordinator's OpenAI frontend drives the stage chain over the
   existing Skippy binary activation transport.

This is a substantial draft checkpoint, not production support for arbitrary
SafeTensors models.

## Verified whole-model proof

`HuggingFaceTB/SmolLM2-135M-Instruct` was started through the shipped command
shape:

```bash
mesh-llm serve --model HuggingFaceTB/SmolLM2-135M-Instruct
```

The normal resolver downloaded the complete SafeTensors checkpoint, selected
backend `mlx`, and served model listing, non-streaming chat, and SSE chat. This
proves useful single-node MLX serving is part of the work. It intentionally
does not prove partial downloads: a whole-model server needs all model weights.

The integrated loader now quantizes eligible, unquantized dense source tensors
to affine 4-bit as they load. Inkling, Nemotron-H, and checkpoints already
declaring a quantized representation retain their native representation. The
automatic policy also retries the native representation for quantization
incompatibility or the known benign tied-Qwen `lm_head.weight` rejection. Other
strict-loader failures remain fail-closed, as do explicit affine modes. The
earlier solo-serving measurements showed that this
load-time representation has the same steady-state generation speed as loading
an equivalent pre-quantized artifact; see `../../spikes/mlx-solo/FINDINGS.md`.

The 135M model is adequate as a serving and protocol oracle, but its weak agent
output is not evidence of Goose-quality model behavior. Larger single-node
models still need quality, memory-high-water, and agent-harness measurements.

The integrated path was therefore also exercised with `Qwen/Qwen3-0.6B` at a
16K context. Its redundant tied `lm_head.weight` is incompatible with the
pinned strict affine loader, so auto mode reported that incompatibility and
retried the native checkpoint representation. The model then answered a basic
arithmetic prompt correctly and completed a Goose OpenAI-provider run with the
requested exact response. This proves the adaptive fallback preserves useful
single-node serving rather than turning optional quantization into a startup
requirement.

The ordinary command also now accepts an unchanged published MLX-LM artifact:

```bash
mesh-llm serve --model mlx-community/Qwen3-0.6B-4bit --ctx-size 16384
```

The recorded run resolved Hugging Face revision
`73e3e38d981303bc594367cd910ea6eb48349da8`; the unpinned command above follows
the repository's current default revision.

The repository's omitted `quantization.mode` is interpreted as MLX-LM's
standard `affine` default by the pinned safemlx correction proposed upstream in
`jbg/safemlx#2`. The integrated server loaded the cached 4-bit checkpoint in
about 1.2 seconds, advertised it through `/v1/models`, returned `391` for
`23 * 17`, completed SSE with `[DONE]`, and served Goose through the same
OpenAI endpoint. A 4K-context Goose attempt was correctly rejected because its
assembled request required about 11.2K tokens; the 16K run completed.

## Verified mesh split proof

A two-host explicit split of the same 30-layer model completed through the
normal OpenAI frontend:

| Stage | Layers | Derived affine-4 artifact |
| --- | ---: | ---: |
| coordinator stage | `0..29` | about 70 MB |
| remote final stage | `29..30` | about 17 MB |

The remote host never stored the complete roughly 269 MB source checkpoint.
It fetched metadata plus the tensor ranges selected for its layer and final
boundary tensors. A non-streaming request traversed both real stages and
returned coherent text; SSE also completed with `[DONE]`.

The apparently uneven artifacts are expected: embeddings and readout tensors
are much larger than a typical transformer block. The planner now derives
conservative per-layer affine-4 estimates from SafeTensors headers and charges
boundary tensors to every stage that loads them, rather than dividing total
bytes evenly by layer count.

## Safety and lifecycle behavior

- Only immutable 40-character Hugging Face revisions identify split tensor
  ranges.
- Config, index, and shard headers are fetched before tensor payloads; exact
  HTTP ranges are identity-checked with strong ETags.
- Coordinator sidecars and derived stage artifacts are lock-protected and
  atomically published.
- Stage `Prepare` may derive a missing entry; `Load` validates and consumes an
  already prepared entry.
- Connections track every touched session. EOF, transport error, timeout, or
  normal stop resets local state and propagates `Stop` through downstream
  stages.
- Stage and generation I/O have bounded timeouts.
- MLX frontend bind succeeds before the model is advertised as ready.
- The capability advertisement is additive and is emitted only by an
  Apple-Silicon build containing the MLX feature. Older peers safely treat the
  absent field as unsupported.

## Current boundaries

- Partial-download integration is currently entered by explicit `--split`;
  automatic startup still follows the existing whole-model resolver.
- Integrated partial generation supports dense `model_type=llama` stages.
- Generation is serialized to one MLX lane and currently uses greedy sampling.
- Automatic affine-4 is the current default for eligible dense checkpoints,
  not yet a general hardware/quality policy surface.
- Cache capacity and eviction are not yet owned by this integration.
- The workspace root must patch the registry requirements to the certified
  public safemlx revision. Root patches do not propagate to crates.io
  consumers, so standalone published `skippy-engine-mlx --features mlx` support
  is not usable until compatible safemlx releases contain the required APIs.
- Frontier families require their own stage semantics. Nemotron-H has
  metadata/range planning and a one-layer execution proof, but not a complete
  hybrid-model topology. Inkling has whole-model support in the pinned safemlx
  runtime and compelling exact-range storage evidence, but no integrated
  partial-stage executor yet.

## Build and focused verification

Use the MLX recipes because they provide the full Xcode Metal toolchain and
copy the required `mlx.metallib` beside the executable:

```bash
just mlx-build
just mlx-release-build
```

Focused library checks require the same developer directory:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  cargo test -p skippy-engine-mlx --features mlx --lib
```
