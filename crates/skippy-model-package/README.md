# skippy-model-package

Model inspection and stage-package CLI.

This tool uses llama-backed model introspection through the C ABI. GGUF writing
must go through llama.cpp writer code exposed by the ABI; Rust owns package
planning, manifests, checksums, and CLI behavior.

## Architecture Role

`skippy-model-package` prepares the per-stage model artifacts consumed by
`skippy-server` through the mesh materialization cache. Each stage owns one
contiguous layer range and loads a sparse GGUF shard or a materialized package
slice:

```mermaid
flowchart LR
    M["source model.gguf"] --> Slice["skippy-model-package"]
    Slice --> G0["stage-0.gguf<br/>layers 0..10<br/>embeddings"]
    Slice --> G1["stage-1.gguf<br/>layers 10..20"]
    Slice --> G2["stage-2.gguf<br/>layers 20..30"]
    Slice --> G3["stage-3.gguf<br/>layers 30..40<br/>output tensors"]
    G0 --> Cache["mesh materialized stage cache<br/>derived artifacts"]
    G1 --> Cache
    G2 --> Cache
    G3 --> Cache
    Cache --> S0["stage-0 server"]
    Cache --> S1["stage-1 server"]
    Cache --> S2["stage-2 server"]
    Cache --> S3["final stage server"]
```

Mesh treats these generated shards as derived cache. Package-backed models use
stable Hugging Face identity from `model-ref`/`model-hf`; direct local GGUFs are
materialized as synthetic package inputs instead of using the path stem as a
model id.

## Commands

```bash
skippy-model-package inspect model.gguf
skippy-model-package plan model.gguf --stages 4
skippy-model-package write model.gguf --layers 0..12 --out stage-0.gguf --manifest stage-0.json
skippy-model-package write-stages model.gguf --stages 4 --out-dir slices/
skippy-model-package write-package org/repo:Q4_K_M --out-dir model-package/
skippy-model-package write-package org/repo:Q4_K_M --projector mmproj-model-f16.gguf --out-dir model-package/
skippy-model-package validate model.gguf slices/stage-*.gguf
skippy-model-package validate-package model.gguf model-package/
skippy-model-package validate-glm-dsa-contract model-package/
```

`write` and `write-stages` call the llama C ABI, which uses llama.cpp GGUF
writer code for artifact metadata and streams selected tensor bytes from the
source model. The Rust CLI owns planning, manifests, file checksums, and
validation reports.

`validate` checks that every owned tensor from the source model appears exactly
once across the supplied artifact slices, with no unknown tensors and no
duplicate owned tensors. Shared metadata and tokenizer KVs are preserved by the
llama-backed writer.

`write-package` prefers model coordinates such as `org/repo:Q4_K_M`. It resolves
the coordinate through `model-ref`, `model-artifact`, and the `huggingface-hub`
backed `model-hf` adapter, downloads the resolved source artifact, and records
the resolved repo, revision, primary file, canonical ref, distribution id, and
artifact file set in `model-package.json`.

`write-package` currently records supported source-model speculation metadata
under `generation.speculative_decoding`; it does not populate GLM-DSA execution
policy or threshold fields. For GLM-DSA packages, run
`repair-glm-dsa-generation-policy model-package/ --in-place` to add
`generation.policy` and `generation.thresholds`, then validate the strict
contract. Policy uses stable semantic execution choices such as
`decode: "compact-flash"` or `indexshare: "required"`, while thresholds carry
numeric resolver inputs such as `short_prefill_max_tokens`,
`compact_flash_min_kv`, and `dense_mask_max_bytes`. Do not add
model-family-specific objects such as `generation.glm_dsa`; use a versioned
policy profile such as `glm-dsa-v1`.

Generation defaults should be grounded in gated package/backend evidence, not
single diagnostic microbench rows. For GLM-DSA MoE tuning, production-shaped
routed whole-graph consumer probes are sanity checks only unless they are
physically plausible against isolated routed FFN estimates.

Layer packages store input-boundary tensors in `shared/embeddings.gguf` and
final-boundary tensors in `shared/output.gguf`; owned tensors should appear in
exactly one package artifact.

Multimodal projectors are explicit package artifacts. Pass one or more
`--projector path/to/mmproj*.gguf` arguments to `write-package`; the CLI copies
them into `projectors/`, fingerprints them, records them as `kind: "mmproj"` in
`model-package.json`, and `validate-package` checks the declared projector
checksums and sizes. Package-backed serving uses the first declared projector
when no explicit `projector_path` is supplied by the caller.

Local paths are only accepted for package creation when the caller supplies
explicit provenance:

```bash
skippy-model-package write-package ./model.gguf \
  --out-dir model-package/ \
  --model-id org/repo:Q4_K_M \
  --source-revision abc123 \
  --source-file Qwen3-8B-Q4_K_M.gguf
```

This keeps canonical package identity tied to real model coordinates rather
than inferred from arbitrary filesystem paths.

`--transform-artifact-command` runs an in-place transformation before package
metadata is measured, so the manifest describes the transformed artifact.
`--after-artifact-command` remains suitable for upload hooks that may remove
the artifact after its metadata is captured. Pass `--resume-existing-artifacts`
to reuse existing artifacts; transform and upload hooks are still run for each
resumed artifact.

`validate-package` checks the source-model checksum, manifest artifact checksums
and sizes, declared tensor counts/bytes, layer coverage, duplicate layers, and
exact owned tensor coverage against the source model.

`validate-glm-dsa-contract` is the local pre-spend gate for GLM-5.2-style
artifacts. It checks GGUF metadata, tensor completeness, native MTP
preservation, and Full/Shared IndexShare roles. New GLM-DSA artifacts must
expose roles through `glm-dsa.attention.indexer.types` or frequency/offset
metadata; tensor-presence inference is reported as a compatibility fallback
and fails the contract gate.
