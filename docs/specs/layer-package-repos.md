# Layer Package Repository Spec

Status: draft

Layer package repositories are durable model artifacts for skippy-backed stage
serving. A repository contains one `model-package.json` manifest plus GGUF
fragments that can be selected by layer range and loaded by a stage without
requiring every peer to store or materialize the full source model.

The current runtime accepts local package directories and Hugging Face package
references of the form `hf://namespace/repo`, `hf://namespace/repo:revision`,
or `hf://namespace/repo@revision`.

## Goals

- Let mesh nodes fetch only the model pieces needed for their assigned layer
  range.
- Keep package identity tied to a real source model coordinate, revision, and
  artifact set.
- Make package validation deterministic before a stage is launched.
- Keep package manifests additive so older runtimes can reject unsupported
  packages clearly instead of loading incompatible tensor layouts.
- Treat per-stage materialized GGUFs as derived cache, not as the durable
  package format.

## Repository Identity

A layer package repository SHOULD be named after the source model and
distribution it contains, for example:

```text
meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers
meshllm/DeepSeek-V3.2-UD-Q4_K_XL-layers
```

The repository identity is not enough to prove compatibility. Consumers MUST
read `model-package.json` and use the manifest fields below as the source of
truth.

Package references use `hf://` so runtime code can distinguish package repos
from model coordinates and local paths:

```text
hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers
hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers:8f4c2d1
hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers@main
```

Publishers SHOULD point production configs at an immutable commit hash or tag,
not a moving branch.

## Repository Layout

The root of the repository MUST contain `model-package.json`.

Recommended layout:

```text
model-package.json
shared/
  metadata.gguf
  embeddings.gguf
  output.gguf
layers/
  layer-00000.gguf
  layer-00001.gguf
  layer-00002.gguf
  ...
projectors/
  mmproj-model-f16.gguf
README.md
```

Required artifacts:

| Artifact | Purpose |
| --- | --- |
| `shared/metadata.gguf` | Shared GGUF metadata and tokenizer state required by every stage. |
| `shared/embeddings.gguf` | Input-boundary tensors required by the first stage. |
| `shared/output.gguf` | Output-boundary tensors required by the final stage. |
| `layers/layer-NNNNN.gguf` | Owned tensors for one transformer layer. |

Optional artifacts:

| Artifact | Purpose |
| --- | --- |
| `projectors/*.gguf` | Multimodal projector GGUFs, currently `kind: "mmproj"`, used by stage 0 or single-stage serving. |

Artifact paths in the manifest MUST be relative to the repository root. They
MUST NOT be absolute paths and MUST NOT escape the package root with `..`.
Consumers MUST reject unsafe paths.

Each owned tensor from the source model MUST appear in exactly one package
artifact. Shared metadata and tokenizer values may be repeated only where the
GGUF writer requires them to keep each fragment loadable.

Projector artifacts are package-level companions, not transformer-layer
fragments. They MUST NOT be counted as owned layer tensors and MUST NOT be
merged into per-stage GGUF materializations.

## Peer Cache Transfer

For split Skippy runs, a worker may fetch missing Hugging Face package artifacts
from the coordinating mesh node over admitted mesh `STREAM_SUBPROTOCOL`
transport before falling back to normal local/HF package resolution. This
transfer path is an optimization for already-selected split participants, not a
package discovery protocol.

Privacy and compatibility boundaries:

- Nodes advertise only `skippy-stage/2` subprotocol feature support, including
  `artifact-transfer`. They do not gossip local package inventory, artifact
  paths, cache roots, or tokens.
- Mesh owns only the subprotocol open envelope; Skippy owns the artifact
  request/response schema, authorization semantics, and byte framing.
- Peer cache transfer uses the mesh `STREAM_SUBPROTOCOL` envelope. Generation-3
  Skippy stage peers are a compatibility break, so older chained-reply
  subprotocol peers are not mixed into new split topologies.
- Only `hf://namespace/repo@revision` package refs are eligible for peer
  transfer.
- The serving node checks the active split topology and only serves artifacts
  needed by the requesting node's assigned stage range. Stage 0 may fetch input
  boundary files and projector artifacts; final stages may fetch output boundary
  files.
- `model-package.json` is capped at 16 MiB for peer transfer.
- Non-manifest artifacts must match the manifest-declared relative path, byte
  size, and SHA-256 digest.
- Received artifacts are written to a fresh hidden partial file and installed
  atomically only after size and SHA-256 verification.
- Peer artifact transfer is not advertised or served by default on public mesh
  nodes. Set `MESH_LLM_ARTIFACT_TRANSFER=trusted` to enable same-owner or
  explicitly trusted-owner transfer, or `MESH_LLM_ARTIFACT_TRANSFER=open` for
  lab deployments that intentionally allow any peer.

## Manifest Schema

The manifest file is UTF-8 JSON. The current schema version is `1`.

Minimal shape:

```json
{
  "schema_version": 1,
  "model_id": "Qwen/Qwen3-235B-A22B-GGUF:UD-Q4_K_XL",
  "source_model": {
    "path": "/cache/Qwen3-235B-A22B-UD-Q4_K_XL.gguf",
    "sha256": "<64 hex chars>",
    "repo": "Qwen/Qwen3-235B-A22B-GGUF",
    "revision": "<source commit>",
    "primary_file": "Qwen3-235B-A22B-UD-Q4_K_XL.gguf",
    "canonical_ref": "Qwen/Qwen3-235B-A22B-GGUF:UD-Q4_K_XL",
    "distribution_id": "UD-Q4_K_XL",
    "files": [
      {
        "path": "Qwen3-235B-A22B-UD-Q4_K_XL.gguf",
        "size_bytes": 123,
        "sha256": "<64 hex chars>"
      }
    ]
  },
  "format": "layer-package",
  "layer_count": 94,
  "activation_width": 8192,
  "shared": {
    "metadata": {
      "path": "shared/metadata.gguf",
      "tensor_count": 0,
      "tensor_bytes": 0,
      "artifact_bytes": 123,
      "sha256": "<64 hex chars>"
    },
    "embeddings": {
      "path": "shared/embeddings.gguf",
      "tensor_count": 4,
      "tensor_bytes": 123,
      "artifact_bytes": 123,
      "sha256": "<64 hex chars>"
    },
    "output": {
      "path": "shared/output.gguf",
      "tensor_count": 4,
      "tensor_bytes": 123,
      "artifact_bytes": 123,
      "sha256": "<64 hex chars>"
    }
  },
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
      "default": "mtp",
      "strategies": {
        "mtp": {
          "type": "native-mtp",
          "prediction_depth": 1,
          "layer_indices": [47],
          "window_policy": {
            "default": "fixed",
            "initial_window": 1,
            "min_window": 1,
            "max_window": 1
          }
        }
      }
    }
  },
  "layers": [
    {
      "layer_index": 0,
      "path": "layers/layer-00000.gguf",
      "tensor_count": 32,
      "tensor_bytes": 123,
      "artifact_bytes": 123,
      "sha256": "<64 hex chars>"
    }
  ],
  "projectors": [
    {
      "kind": "mmproj",
      "path": "projectors/mmproj-model-f16.gguf",
      "tensor_count": 128,
      "tensor_bytes": 123,
      "artifact_bytes": 123,
      "sha256": "<64 hex chars>"
    }
  ],
  "skippy_abi_version": "1.2.3",
  "created_at_unix_secs": 1790000000
}
```

Required top-level fields:

| Field | Requirement |
| --- | --- |
| `schema_version` | MUST be `1` for the current format. |
| `model_id` | MUST be a non-empty model coordinate, not a filesystem-derived name. |
| `source_model` | MUST identify the source artifact used to build the package. |
| `format` | MUST be `layer-package`. |
| `layer_count` | MUST match the source model's transformer layer count. |
| `activation_width` | SHOULD be present; routing and topology planning rely on it. |
| `generation` | MAY declare package-owned generation defaults, including native speculative decoding strategies. |
| `shared` | MUST include `metadata`, `embeddings`, and `output` artifacts. |
| `layers` | MUST include exactly one entry for each layer index `0..layer_count`. |
| `projectors` | MAY include package-level projector artifacts; currently only `kind: "mmproj"` is defined. |
| `skippy_abi_version` | MUST describe the llama/skippy ABI used to write the fragments. |
| `created_at_unix_secs` | SHOULD be set by the package writer for provenance. |

Each artifact entry MUST include:

- `path`: repository-relative artifact path.
- `tensor_count`: number of tensors in the fragment.
- `tensor_bytes`: total bytes for tensor payloads in the fragment.
- `artifact_bytes`: exact file size in bytes.
- `sha256`: lowercase or uppercase 64-character SHA-256 hex digest.

`artifact_bytes` MUST be greater than zero. `tensor_bytes` MUST be zero when
`tensor_count` is zero and greater than zero when `tensor_count` is greater
than zero.

`projectors` is optional and defaults to an empty list. When present, each
projector entry uses the same artifact fields plus:

- `kind`: projector type. The current schema defines `mmproj`; consumers MUST
  reject unknown kinds unless they explicitly support them.
- `path`: repository-relative projector GGUF path, usually under `projectors/`.

Projector entries MUST have non-empty, safe relative paths, positive
`artifact_bytes`, and valid 64-character SHA-256 digests. A package MAY declare
multiple projectors for future model variants, but the current serving default
is to use the first declared `mmproj` when no explicit projector path is
configured.

### Generation Defaults

`generation` is optional and defaults to no package-owned generation policy.
When present, it may declare package-authored runtime defaults. The package owns
defaults that are specific to the artifact distribution, such as quant layout,
preserved native tensors, validated sparse-attention paths, and native
speculative decoding strategy.

The `generation` object has two separate responsibilities:

- `generation.policy` names the semantic execution profile and phase choices
  that were validated for this artifact.
- `generation.thresholds` supplies numeric resolver hints used to decide when a
  phase choice applies.
- `generation.speculative_decoding` declares package-owned native or draft
  speculation strategy defaults, when the artifact actually contains or
  references the required tensors/models.

Use this split consistently:

| Location | Meaning | Typical fields |
| --- | --- | --- |
| `generation.policy` | Portable, semantic execution decisions for this artifact. | `profile`, `decode`, `short_prefill`, `long_prefill`, `verify`, `indexshare` |
| `generation.policy.experimental` | Named experimental policy paths that are valid only with explicit evidence and logging. | `selected_row_flash`, `moe_weighted_down`, `moe_merged_shared_gate_up` |
| `generation.thresholds` | Numeric decision inputs for the policy resolver. | token limits, byte limits, minimum KV sizes |
| `generation.speculative_decoding` | Package-owned speculation strategy defaults. | `default`, `strategies`, `native-mtp`, `prediction_depth`, `window_policy` |
| GGUF metadata | Architecture and tensor-layout correctness contract. | attention dimensions, IndexShare roles, native MTP tensor layout |

Keep these responsibilities separate. Policy fields should answer "which
execution path is intended for this phase?" Threshold fields should answer
"when should the runtime choose or reject that path?" For example,
`decode: "compact-flash"` belongs under `generation.policy`, while
`compact_flash_min_kv: 1` belongs under `generation.thresholds`.

The `generation` object is intentionally generic. Do not add
model-family-specific sub-objects such as `generation.glm_dsa`; use a stable
`generation.policy.profile` instead. This keeps the manifest shape reusable for
future sparse attention, native prediction, and verifier policies without
creating one schema branch per model family.

Keep backend tuning out of the manifest unless it can be expressed as portable
policy or threshold data. Kernel names, Metal/CUDA dispatch choices, simdgroup
tuning, row-height sweeps, and quant-kernel experiments are runtime capability
or benchmark evidence. They should not become manifest fields such as
`generation.metal` or `generation.glm_dsa`. When evidence graduates into a
package default, record it as a profile version, a phase policy, a numeric
threshold, or a speculation strategy.

Runtime config and explicit CLI/environment overrides MAY override these
defaults for experiments, but consumers SHOULD log the final resolved policy and
the package recommendation that was overridden. If a consumer cannot execute the
package-recommended path, it MUST choose a correctness-preserving fallback and
emit the fallback reason.

Generation policy is not a backend feature switch. Package authors MUST NOT use
backend-specific fields to select Metal, CUDA, CPU, Vulkan, or Skippy kernels.
Backends advertise capability and performance evidence to the resolver; the
manifest records the package-level semantic policy and thresholds that make the
resolver decision explainable.

Consumers MUST NOT silently reinterpret unknown policy values as a supported
path. Unknown profiles, phase values, or experimental policy switches should be
reported as unsupported unless the runtime has an explicit compatibility rule
for that value. Numeric threshold fields are hints and may be ignored by older
consumers, but policy fields describe execution semantics and must be handled
deliberately.

Package writers SHOULD only emit a `generation.policy.profile` after they have
validated the artifact shape that profile requires. For GLM-DSA this means the
writer has found the GLM-DSA attention tensors, routed/shared MoE tensors,
IndexShare metadata or equivalent role evidence, and any preserved native MTP
tensors before advertising `glm-dsa-v1`. The writer may infer the default
policy from GGUF metadata and tensor names, but it should not infer a more
specific policy than the artifact can actually support.

Serving resolvers SHOULD treat `generation` as input to a phase resolver, not
as direct kernel wiring. A resolver first identifies the request phase, such as
decode, short prefill, long prefill, or speculative verification. It then
combines the package policy, thresholds, backend capability evidence, and
request shape to select a path. The selected path, rejected package path, and
fallback reason should be visible in logs or telemetry whenever the resolver
does not use the package recommendation.

Package generation defaults are not a substitute for model correctness
metadata. Architecture-specific GGUF metadata and tensor layout still define
whether a runtime may execute the model at all; `generation.policy` only
chooses among valid execution paths for that artifact.

#### Execution Policy

Packages MAY declare `generation.policy` to describe the package-validated
execution profile. This profile is a model execution policy, not a backend
implementation detail. It should use stable semantic path names such as
`compact-flash` rather than Metal/CUDA kernel names.

The current proposed shape is:

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
        "selected_row_flash": "evidence-gated"
      }
    },
    "thresholds": {
      "short_prefill_max_tokens": 2048,
      "compact_flash_min_kv": 1,
      "dense_mask_max_bytes": 268435456
    }
  }
}
```

`profile` names the policy family and version. For GLM-DSA packages, use
`glm-dsa-v1` until a later profile intentionally changes the meaning of the
phase fields or thresholds. Other model families SHOULD use their own stable
profile names instead of adding model-family-specific top-level objects under
`generation`.

The profile is also the compatibility boundary for package tooling. Writers may
infer a known profile from GGUF metadata and tensors, but they must not invent
backend-specific field names for one model family. If a later GLM-DSA package
needs different phase semantics, create a new profile such as `glm-dsa-v2`
instead of changing the meaning of `glm-dsa-v1`.

Policy values are intentionally phase-specific:

- `decode`: preferred one-token generation path. For GLM-DSA this is expected
  to become `compact-flash` when compact selected-KV attention has parity on
  the package.
- `short_prefill`: preferred path below the short-prefill threshold. Packages
  MAY select `dense` when sparse/indexer overhead is known to dominate.
- `long_prefill`: preferred path above the short-prefill threshold. Packages
  SHOULD avoid policies that materialize dense sparse masks for long context.
- `verify`: preferred path for speculative verification spans. It MAY remain
  `auto` until verifier-specific parity and performance are measured.
- `indexshare`: whether Shared GLM-DSA layers require local IndexShare/top-k
  state. `required` means a consumer must not silently recompute shared-layer
  indexers unless an explicit fallback policy is selected and logged.
- `experimental.selected_row_flash`: controls selected-row flash fusion. Use
  `evidence-gated` until the package has reproducible wins for that path on the
  target backend.
- `experimental.moe_weighted_down`: controls moving MoE route weights before
  the routed down projection instead of applying them in the output weighted
  sum. Use `evidence-gated`; current evidence makes this a small graph-shape
  experiment, not a replacement for expert matmul optimization.
- `experimental.moe_merged_shared_gate_up`: controls merged shared-expert
  gate/up execution. Use `evidence-gated` until the package/backend has
  reproducible combined-FFN graph wins and parity evidence.

Suggested semantic path values are:

- `auto`: runtime chooses using package thresholds and backend capability.
- `dense`: dense attention path; useful for short prefill when measured faster.
- `direct-sparse`: direct GLM-DSA sparse attention.
- `compact-flash`: compact selected K/V followed by flash attention.
- `sparse-chunked`: chunked sparse prefill path for long prompts.
- `fallback`: named correctness fallback when a native sparse backend is not
  available.

`generation.thresholds` are package recommendations. Consumers SHOULD treat
them as input to the runtime policy resolver, not as hard schema limits. Every
policy decision SHOULD emit telemetry containing the policy profile, phase,
selected path, rejected path or fallback reason, `n_kv`, `top_k` when present,
IndexShare role when present, backend, and any dense sparse-mask allocation
avoided.

Thresholds must be named for the resolver decision they inform, not for an
implementation detail. For example, prefer `dense_mask_max_bytes` over a
backend-specific allocation flag. This keeps the same package usable across
Metal, CUDA, CPU, and future backends while still allowing each backend to make
an evidence-based decision.

Threshold units are part of the field contract:

| Threshold | Unit | Resolver use |
| --- | --- | --- |
| `short_prefill_max_tokens` | tokens | Selects the short-prefill policy when the prompt/window is at or below this size. |
| `compact_flash_min_kv` | KV rows | Rejects compact selected-KV flash below the minimum useful KV history. |
| `dense_mask_max_bytes` | bytes | Rejects dense sparse-mask materialization when the estimated mask would exceed this budget. |

#### Policy Resolution

Consumers resolve generation policy in this order:

1. Request/runtime override, when explicitly configured for an experiment.
2. Package `generation.policy` and `generation.thresholds`.
3. Runtime built-in default for the architecture.
4. Correctness fallback when the preferred path is unsupported.

The resolved policy is a runtime contract. Package tools may infer and write
the policy from GGUF tensor shape, but serving code must not re-infer a
different policy silently. For example, a GLM-DSA package with split
`attn_k_b`, `attn_v_b`, and `attn_kv_a_mqa` tensors may advertise
`glm-dsa-v1`; the runtime may still fall back from `compact-flash` to `dense`
on a backend that lacks compact selected-KV attention, but it must log that
fallback.

Policy resolution telemetry SHOULD include the package profile and the exact
decision inputs that changed the selected path: phase, backend, `n_tokens`,
`n_kv`, `top_k`, estimated dense-mask bytes, selected threshold, selected path,
fallback path, and fallback reason. This is required evidence for tuning
package defaults; without it, benchmark results cannot explain whether a run
used the intended GLM-DSA path or a correctness fallback.

For `glm-dsa-v1`, the current phase intent is:

| Phase | Recommended value | Intent |
| --- | --- | --- |
| `decode` | `compact-flash` | Avoid dense sparse-mask materialization during one-token decode. |
| `short_prefill` | `dense` | Avoid paying sparse/indexer overhead when prompts are below the package threshold. |
| `long_prefill` | `sparse-chunked` | Keep long-context prefill away from huge dense sparse masks. |
| `verify` | `auto` | Let the runtime select a verifier path until verifier-specific parity is proven. |
| `indexshare` | `required` | Reuse Full-layer top-k/index state for Shared GLM-DSA layers instead of silent recompute. |
| `experimental.selected_row_flash` | `evidence-gated` | Enable compact selected-row flash only when package/backend evidence proves parity and a win. |
| `experimental.moe_weighted_down` | `evidence-gated` | Enable weighted-down MoE graph shape only when package/backend evidence proves parity and a win. |
| `experimental.moe_merged_shared_gate_up` | `evidence-gated` | Enable merged shared-expert gate/up only when package/backend evidence proves parity and a combined-graph win. |

For `glm-dsa-v1`, the current threshold intent is:

| Threshold | Meaning |
| --- | --- |
| `short_prefill_max_tokens` | Maximum prompt/window length that should prefer the short-prefill policy. |
| `compact_flash_min_kv` | Minimum KV length where compact selected-KV flash attention is worth considering. |
| `dense_mask_max_bytes` | Maximum dense sparse-mask allocation the runtime should permit before forcing a sparse fallback. |

For GLM-DSA threshold tuning, consumers SHOULD reason from the package tensor
sizes rather than from model names alone. The relevant first-order estimates
are:

| Quantity | Formula | Example |
| --- | --- | --- |
| Hidden activation bytes | `tokens * hidden_width * activation_bytes` | GLM-5.2 width `6144`: `12 KiB/token` at f16 or `24 KiB/token` at f32. |
| IndexShare sideband bytes | `tokens * top_k * 4` | Width `768`: `3 KiB/token`, `384 KiB` for a 128-token chunk. |
| Dense sparse-mask bytes | `tokens * visible_kv * 4` | At `128k` visible KV: `512 KiB/token`, `64 MiB` for a 128-token chunk, `1 GiB` for a 2048-token chunk. |

These numbers are intentionally policy inputs, not schema requirements. A
package with a different `attention.indexer.top_k`, hidden width, activation
wire dtype, or context target will produce different thresholds. The important
contract is that packages expose enough policy and threshold information for
the runtime to explain why it selected dense, direct sparse, compact-flash, or
fallback execution.

Current GLM-DSA tuning is grounded in llama.cpp Metal backend fixtures around
`kv=257,top_k=64`. On the one-token decode shape, compact selected-row flash
measured `63.40 us/run`, direct sparse attention measured `106.57 us/run`, and
dense masked flash measured `71.72 us/run`. That makes the compact selected-row
path about `1.68x` faster than direct sparse and about `1.13x` faster than dense
masked flash for this fixture.

Exact-boundary fixtures where selected KV equals visible KV are more decisive
than the base one-token fixture. Compact selected-row flash measured
`61.90 us/run` at `kv=128,top_k=128`, `60.71 us/run` at
`kv=256,top_k=256`, `61.24 us/run` at `kv=257,top_k=257`, and
`57.99 us/run` at `kv=513,top_k=513`. Direct sparse measured
`137.15 us/run`, `209.97 us/run`, `249.30 us/run`, and `711.10 us/run` on the
same shapes. That makes compact selected-row flash about `2.2x`, `3.5x`,
`4.1x`, and `12.3x` faster than direct sparse across those exact-boundary
points. The literal `top_k >= visible_kv` boundary is a separate all-KV flash
bypass in the llama.cpp graph, but ordinary one-token decode after prefill
still exercises compact selected-KV flash because IndexShare top-k is selected
from the previous KV state while attention sees previous+current KV. This is
why `glm-dsa-v1` uses compact flash as the default decode route and reserves
direct sparse decode for explicit runtime experiments.

For the configured GLM-5.2 IndexShare width, the compact path is more
important once visible KV grows beyond the `768` selected rows. The Metal
fixtures measured compact selected-row flash at `55.11 us/run` for
`kv=1024,top_k=768` and `55.62 us/run` for `kv=2048,top_k=768`. Direct sparse
attention on the same `dk=576,dv=512,top_k=768` shapes measured
`988.95 us/run` and `984.50 us/run`. That is roughly an `18x` decode-path win
for compact selected-KV flash over direct sparse at the model-native top-k
width.

Short phase fixtures measured the opposite shape for direct sparse prefill:
dense masked flash stayed around `68.58-70.80 us/run` for 4-16 token batches,
while direct sparse measured `461.98-473.75 us/run`. That makes dense masked
flash roughly `6.5-6.9x` faster on those short phase fixtures. This is why
`glm-dsa-v1` keeps `short_prefill: "dense"` and `verify: "auto"` as the package
defaults, with direct sparse prefill reserved for explicit runtime/package
policy after backend evidence exists. Treat these as backend evidence for the
current threshold defaults, not as portable constants across every device or
quant.

Once those attention phase gates are in place, the measured local GLM-5.2
bottleneck shifts to MoE expert execution. The gated Metal MoE fixture
estimates a routed FFN decode layer at `391.13 us`: routed gate/up/down matmuls
account for `375.42 us` (`96.0%`), routed fused SwiGLU accounts for `5.35 us`
(`1.4%`), and route/top-k plus weighted sum accounts for `10.36 us` (`2.6%`).
The shared expert is equally important: a production-shaped fused GLU shared
expert plus final add measured `405.32 us`, making the routed+shared FFN
estimate `796.45 us` with the shared expert at `50.9%`. These numbers justify
prioritizing backend `MUL_MAT_ID`/expert matmul and shared-expert work after
sparse-attention correctness, but they do not require a new manifest object.
Policy remains the semantic phase contract under `generation.policy`;
performance cutoffs and byte/token limits remain numeric resolver inputs under
`generation.thresholds`.
A component breakdown also corrected a bad diagnostic path: the unfused
`silu(gate) * up` whole-graph row measured `291.73 us`, but the normal
llama.cpp shared expert path already uses `ggml_swiglu_split()`. The fused
SwiGLU split row measured only `4.09 us` (`8.24 us` including final add), so
custom activation fusion is not the next local target. Treat the fused GLU
numbers as the production-shaped evidence and keep optimization pressure on
routed q2/q3 `MUL_MAT_ID`, q2_K down quality experiments, and deeper expert
matmul/layout work.
The combined FFN fixture is the current Phase E decision gate because it keeps
routed and shared expert execution in one production-shaped graph. With
`n_embd=6144`, the q3_K routed down baseline measured `1147.21 us`; changing
decoder routed down projections to q2_K measured `817.38 us` (`1.40x` faster),
and keeping q3_K routed down while using merged shared gate/up measured
`868.59 us` (`1.32x` faster). Route/top-k plus weights measured only `3.26 us`,
weighted sum measured `6.69 us`, and routed fused SwiGLU measured `5.09 us`, so
the next practical levers are whole-graph expert matmul/layout work and a
quality-tested q2_K routed-down quant recipe. The q2_K routed-down recipe must
exclude `blk.78`, the native NextN/MTP block, until quality and speculation
tests prove that lowering the MTP block is acceptable. Treat these as
optimization-priority evidence, not as a new model-family schema branch:
quality-bearing quant changes still need separate evaluation under the normal
benchmark flow.
The Phase E report can be run with `GLM52_PHASE_E_REQUIRE_GATES=1` to fail
loudly when production-shaped MoE evidence is missing. With
`GLM52_PHASE_E_KERNEL_SWEEP=1`, the same gate also requires the dispatch-policy
sweep rows. Forcing one-token q3_K routed down through Metal `mul_mm_id`
measured `860.96 us`, compared with `164.46 us` on the default `mul_mv_id`
path. q3_K `mul_mv_id` simdgroup tuning measured `164.46 us` at the default
`nsg=2` and `163.92 us` for the best sampled row, which is noise-level rather
than a real policy win. The row-height sweep kept that conclusion intact:
default `nr0=4` measured `163.98 us`, which was also the best sampled row;
`nr0=1` and `nr0=2` were both around `195-196 us`, and `nr0=8` was `166.20 us`.
A fixed `k=2048` q3_K GLM-down specialization measured `164.04 us` at default
`nsg=2` and `162.67 us` at `nsg=4`, only `1.01x` faster than ordinary default
q3_K `mul_mv_id`. Keep q3_K down work focused on a deeper expert matmul/layout
specialization or a quality-tested q2_K down alternative, not generic
matrix-matrix cutoff, simdgroup, row-height, or fixed-block knobs.
Production-shaped routed whole-graph consumer rows remain useful as harness
history and as a guard against timing only a graph tail. They fixed an earlier
perf-mode issue where the fixture had `run_whole_graph()` for correctness but
timed only the final `GGML_OP_MOE_WEIGHTED_SUM` during perf. The newer combined
FFN fixture supersedes those rows as the policy decision gate because it keeps
routed and shared experts together. Keep
`generation.policy.experimental.moe_weighted_down` at `evidence-gated`; it is
not a package default. The q2_K down alternative is still the useful follow-up
because it produced the clearest combined-graph speedup, but it needs quality
validation before it can become a model-package quant policy.

The IndexShare numbers are why `generation.policy.indexshare` is a first-class
policy field instead of an implementation note. For GLM-5.2-style DSA with a
`top_k` width of `768`, carrying the shared top-k sideband costs `3 KiB/token`.
That is `384 KiB` for a 128-token chunk, `1.5 MiB` for 512 tokens, `6 MiB` for
2048 tokens, and `384 MiB` for a full 128k-token window. By comparison, a dense
float sparse mask over 128k visible KV costs `512 KiB/token`: `64 MiB` for 128
tokens, `256 MiB` for 512 tokens, and `1 GiB` for 2048 tokens. IndexShare still
has real bandwidth cost, but it is dramatically smaller than dense mask
materialization and preserves the model's Full-layer routing decision for
Shared layers.

Generation policy consumers should therefore treat the GLM-DSA policy as a
phase-aware resolver contract:

- Decode starts with `compact-flash` when backend support exists, because the
  compact selected-row path wins on measured one-token and post-prefill decode
  shapes.
- Short prefill starts with `dense`, because measured sparse/indexer overhead is
  worse below the package threshold.
- Long prefill starts with `sparse-chunked`, because dense masks become
  memory-hostile as visible KV grows.
- Verification starts with `auto` until verifier-specific sparse parity and
  performance are measured.
- IndexShare is `required`, because Shared layers should consume the model's
  cached top-k decision rather than recomputing silently.

#### Speculative Decoding

When present, `generation` may also declare `speculative_decoding` defaults:

- `default`: the strategy id the package recommends for this distribution.
- `proposers`: a map of reusable proposal-source ids to configuration.
- `strategies`: a map of strategy id to strategy configuration.

This object is a sibling of `policy` and `thresholds`. It is not an
experimental policy field, because speculation can be disabled, forced, or
selected independently from the attention/MoE phase resolver.

The current native MTP strategy shape is:

```json
{
  "type": "native-mtp",
  "prediction_depth": 1,
  "layer_indices": [47],
  "window_policy": {
    "default": "fixed",
    "initial_window": 1,
    "min_window": 1,
    "max_window": 1
  }
}
```

New packages SHOULD name the native MTP source under `proposers` and reference
it from the strategy. This also permits a package to add an N-gram sidecar
without changing the MTP source:

```json
{
  "default": "mtp-cache",
  "proposers": {
    "mtp": {
      "type": "native-mtp",
      "prediction_depth": 1,
      "layer_indices": [47]
    },
    "cache": {
      "type": "ngram-cache",
      "ngram_min": 2,
      "ngram_max": 4,
      "max_proposal_tokens": 10,
      "history_scope": "request"
    }
  },
  "strategies": {
    "mtp-cache": {
      "type": "composite",
      "primary": "mtp",
      "extender": "cache",
      "extension_policy": {
        "max_tokens": 8
      }
    }
  }
}
```

Supported proposer types are `native-mtp`, `ngram-cache`, and `ngram-suffix`.
Cache and suffix proposers MUST use `history_scope: "request"`.
An `ngram-cache` proposer MUST use `ngram_max` no greater than `4`, llama.cpp's
current cache match-window limit. An `ngram-suffix` proposer MUST satisfy
`3 <= ngram_min <= ngram_max <= 64`. Both contain only target-committed history
for one request and are never shared between users or sessions. A `composite`
strategy MUST use a `native-mtp` primary and an N-gram extender. Its
`extension_policy` bounds the adaptive tail; every combined candidate is still
verified by one target VerifyWindow.

The package schema separates a proposer match length from its output budget:

| Field | Applies to | Requirement |
|---|---|---|
| `prediction_depth` | `native-mtp` | Must be `1` for the current native MTP runtime. |
| `layer_indices` | `native-mtp` | Must identify the package layers that contain the model's NextN/MTP tensors. |
| `ngram_min` / `ngram_max` | N-gram proposers | Define the historical token match range. Both are required and `ngram_min <= ngram_max`. |
| `max_proposal_tokens` | N-gram proposers | Caps how many continuation tokens the proposer may return. It is independent of `ngram_max`. |
| `history_scope` | `ngram-cache`, `ngram-suffix` | Must be `"request"`; a history proposer never observes another request's tokens. |
| `initial_tokens` / `max_tokens` | composite extension policy | Bound the adaptive N-gram tail after an MTP prefix. |
| `tail_backoff_proposals` | composite extension policy | Sets how many proposals to back off after an unhelpful tail. |

`ngram-cache` is a request-local incremental lookup that starts after a
provisional MTP prefix; its
match window is intentionally capped at four tokens by the current llama.cpp
ABI. `ngram-suffix` uses a request-local exact-seed index to select the longest
earlier suffix, up to 64 tokens, and can also start after a provisional MTP
prefix. No N-gram proposer is authoritative: the target verifies the proposal
and commits the accepted prefix only.

Packages that expose the full product menu SHOULD use stable strategy ids:
`mtp`, `ngram-cache`, `ngram-suffix`, `mtp-cache`, and `mtp-suffix`. `mtp` may
reference a reusable `native-mtp` proposer instead
of repeating its prediction-depth and layer metadata. The N-gram-only names
use `proposer`; the composite names use that MTP proposer as `primary` and the
corresponding N-gram proposer as `extender`. `disabled` is a runtime/operator
baseline, not a package strategy.

Native MTP strategy rules:

- `type` MUST be `native-mtp`.
- `prediction_depth` MUST be `1` for the current Skippy native MTP path.
- `layer_indices` MUST list package layer indices containing native MTP/NextN
  tensors, usually the final `blk.N.nextn.*` block emitted by GLM GGUF
  conversion.
- `window_policy` SHOULD be fixed to `1` until runtimes support wider native
  MTP heads.
- package writers MUST NOT advertise a native MTP default unless validation has
  found the required native prediction tensors in the selected package
  artifacts.

Draft-model speculation may use the same strategy map with `type:
"draft-model"` and fields such as `draft_model` and adaptive `window_policy`.
Consumers that do not recognize a strategy type MUST ignore it unless it is the
declared default for a request they are trying to serve.

Operators may override the package recommendation in `config.toml` with
`speculative.strategy`, or for one `mesh-llm serve` invocation with
`--speculative-strategy`. `auto` uses the package/runtime default, `mtp` forces
the direct native-MTP control, and `disabled` disables speculation. A named
package strategy such as `mtp-cache` is accepted only when the selected package
declares it. Precedence is CLI invocation, selected model entry, then global
defaults. These layers may bound N-gram proposal size, extension depth,
cooldown, and VerifyWindow depth. A named package strategy cannot be invented
outside its package. For direct GGUF operation, operators may explicitly select
the request-local `ngram-cache` or `ngram-suffix` proposer by supplying valid
N-gram bounds; mesh-llm constructs
and validates that generic plan before starting Skippy. `skippy-server` receives
the resulting typed plan and does not repeat this policy resolution.

Operators may also pass the legacy `native-mtp-n1` value; the runtime normalizes it to `mtp` for backward compatibility. New configs should use `mtp`.

See [Speculative Decode Configuration](../skippy/PIPELINED_VERIFY_WINDOW.md#operator-configuration)
for the operator-side `config.toml` controls and CLI equivalents. Package
authors should publish conservative, tested values in the manifest; operators
can use configuration to choose a strategy or tighten its bounds without
changing the package's declared topology.

## Layer Selection

For a stage with `layer_start..layer_end`, consumers select:

1. `shared.metadata`
2. `shared.embeddings` only when the stage owns the input boundary
3. every `layers[]` entry with `layer_index` in `layer_start..layer_end`
4. `shared.output` only when the stage owns the final output boundary

The selected parts are sufficient to load the stage. A normal package-backed
runtime SHOULD load selected parts directly. If a runtime composes those parts
into a per-stage GGUF file, that file is derived cache and MUST NOT become the
published repository format.

Projectors are selected independently from layer parts. For a package-backed
multimodal model:

1. an explicit stage `projector_path` wins;
2. otherwise, stage 0 or a single-stage runtime uses the first declared
   `projectors[]` entry with `kind: "mmproj"`;
3. downstream stages do not load projector artifacts.

Consumers MUST NOT infer projector identity from sibling files or filename
stems. If a package needs a projector, the manifest must declare it explicitly.

## Publishing Rules

Package creation SHOULD use:

```bash
skippy-model-package write-package org/repo:distribution --out-dir model-package/
```

Multimodal packages SHOULD declare projector artifacts at write time:

```bash
skippy-model-package write-package org/repo:distribution \
  --projector mmproj-model-f16.gguf \
  --out-dir model-package/
```

The package writer copies declared projectors into `projectors/`, records their
checksums and sizes in `model-package.json`, and keeps them as durable package
artifacts.

Local source GGUF paths are allowed only with explicit provenance:

```bash
skippy-model-package write-package ./model.gguf \
  --out-dir model-package/ \
  --model-id org/repo:distribution \
  --source-revision <commit> \
  --source-file model.gguf
```

Before publishing, run package validation against the source model:

```bash
skippy-model-package validate-package /path/to/source.gguf model-package/
```

A published repository SHOULD include a short `README.md` with:

- source model coordinate and revision;
- source artifact filename and checksum;
- package manifest checksum;
- projector artifact filenames and checksums, when present;
- layer count and activation width;
- skippy ABI version used to write the package;
- package generation defaults such as native MTP strategy id and prediction
  depth, when declared;
- validation command and result;
- any model-family certification notes, such as supported activation wire dtype
  or exact-cache policy.

## Consumer Validation

Before a stage starts, consumers MUST validate:

- `model-package.json` parses as schema version `1`;
- `format` is `layer-package`;
- `skippy_abi_version` is compatible with the runtime ABI;
- `model_id` and source identity fields are non-empty;
- requested layer range is non-empty and within `layer_count`;
- requested layers exist and are not duplicated in the manifest;
- selected artifact paths are relative, safe, and files;
- selected artifact sizes match `artifact_bytes`;
- declared projector paths are relative, safe, and files;
- declared projector sizes match `artifact_bytes`.

Consumers MUST verify SHA-256 checksums for selected artifacts before using an
`hf://` layer package, including cache-hit resolutions. Local package
directories MAY keep checksum verification behind `SKIPPY_VERIFY_PACKAGE_SHA`
for development workflows.

Metadata-only inspection for inventory, identity discovery, or prepare planning
MAY validate only the manifest and shared metadata artifact, and must not
require a non-empty stage layer range. A real stage load still requires the
non-empty range and selected-artifact checks above.

Implementations MAY cache successful checksum verification results. Cache keys
and records should be derived from manifest and artifact identity plus file
metadata, and should not store raw local package paths.

Checksum verification SHOULD include declared projectors when the package is
first downloaded, when the package is validated by tooling, or when a stage is
about to use a projector.

## Compatibility

Manifest schema changes MUST be additive when possible. New optional fields may
be added to schema version `1` if old runtimes can ignore them safely.

`projectors` is an additive schema-version-1 field. Old packages without it
remain valid. Runtimes that do not understand projectors may still serve text
stages from the package, but they MUST reject multimodal serving requests that
require an undeclared or unsupported projector.

Changes that alter tensor ownership, layer indexing, path semantics, ABI
compatibility, or required fields MUST use a new schema version or a new
`format` value. Runtimes MUST reject unknown schema versions and incompatible
ABI versions rather than attempting best-effort loading.

Package repositories do not change mesh gossip or wire protocol semantics by
themselves. When package metadata is advertised through mesh status or planning,
new fields must follow the normal additive compatibility rules for mesh
protocol data.

## Open Questions

- Whether package repos should include an optional signed manifest or checksum
  sidecar for deployments that do not trust the hosting backend.
- Whether large repositories should publish one branch/tag per source revision
  and quantization, or one immutable repository per distribution.
- Whether package validation results should be machine-readable artifacts in
  the repo or remain documented in `README.md`.
