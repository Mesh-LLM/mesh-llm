---
title: Model Package Specification
---

# `model-package.json` specification

`model-package.json` is the manifest for a Skippy layer-package repository. It binds a source model identity to the GGUF artifacts needed to run contiguous layer ranges across one or more nodes.

The current manifest schema is version `1`. The manifest is the source of truth for package identity, artifact paths, layer ownership, checksums, and runtime compatibility. Repository names and README files are descriptive; a consumer must validate the manifest before loading a stage.

## Package repository

The manifest must be at the repository root. A typical package has this shape:

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

Required artifacts are `shared/metadata.gguf`, `shared/embeddings.gguf`, `shared/output.gguf`, and one `layers/layer-*.gguf` artifact for every transformer layer. `projectors/*.gguf` is optional and is currently used for multimodal `mmproj` artifacts.

Every artifact path in the manifest must be relative to the package root. An absolute path or a path containing `..` is invalid. Each owned tensor from the source model must occur in exactly one package artifact. Shared metadata may be repeated when required to keep a GGUF fragment loadable.

## Manifest shape

This example shows the complete schema shape. Values such as checksums and sizes are illustrative:

```json
{
  "schema_version": 1,
  "model_id": "Qwen/Qwen3-235B-A22B-GGUF:UD-Q4_K_XL",
  "source_model": {
    "path": "/cache/Qwen3-235B-A22B-UD-Q4_K_XL.gguf",
    "sha256": "<64 hex characters>",
    "repo": "Qwen/Qwen3-235B-A22B-GGUF",
    "revision": "<source commit>",
    "primary_file": "Qwen3-235B-A22B-UD-Q4_K_XL.gguf",
    "canonical_ref": "Qwen/Qwen3-235B-A22B-GGUF:UD-Q4_K_XL",
    "distribution_id": "UD-Q4_K_XL",
    "files": [
      {
        "path": "Qwen3-235B-A22B-UD-Q4_K_XL.gguf",
        "size_bytes": 123,
        "sha256": "<64 hex characters>"
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
      "sha256": "<64 hex characters>"
    },
    "embeddings": {
      "path": "shared/embeddings.gguf",
      "tensor_count": 4,
      "tensor_bytes": 123,
      "artifact_bytes": 123,
      "sha256": "<64 hex characters>"
    },
    "output": {
      "path": "shared/output.gguf",
      "tensor_count": 4,
      "tensor_bytes": 123,
      "artifact_bytes": 123,
      "sha256": "<64 hex characters>"
    }
  },
  "layers": [
    {
      "layer_index": 0,
      "path": "layers/layer-00000.gguf",
      "tensor_count": 32,
      "tensor_bytes": 123,
      "artifact_bytes": 123,
      "sha256": "<64 hex characters>"
    }
  ],
  "projectors": [
    {
      "kind": "mmproj",
      "path": "projectors/mmproj-model-f16.gguf",
      "tensor_count": 128,
      "tensor_bytes": 123,
      "artifact_bytes": 123,
      "sha256": "<64 hex characters>"
    }
  ],
  "skippy_abi_version": "1.2.3",
  "created_at_unix_secs": 1790000000
}
```

## Top-level fields

| Field | Required | Description |
| --- | --- | --- |
| `schema_version` | Yes | Must be `1` for this specification. |
| `model_id` | Yes | Non-empty model coordinate, including its distribution or quantization identity. |
| `source_model` | Yes | Provenance for the source GGUF model. |
| `format` | Yes | Must be `layer-package`. |
| `layer_count` | Yes | Number of transformer layers. Valid layer indices are `0` through `layer_count - 1`. |
| `activation_width` | Yes | Hidden-state width used by topology and activation planning. |
| `shared` | Yes | `metadata`, `embeddings`, and `output` artifact entries. |
| `layers` | Yes | Exactly one artifact entry for every layer index. |
| `projectors` | No | Package-level projector artifacts; currently `kind: "mmproj"`. |
| `generation` | No | Package-owned generation or speculative-decoding defaults. |
| `skippy_abi_version` | Yes | Skippy/llama ABI used to write the fragments. |
| `created_at_unix_secs` | Recommended | Unix timestamp for package provenance. |

### Source model identity

`source_model.path` and `source_model.sha256` identify the source artifact used to create the package. When the source came from a model repository, include `repo`, `revision`, `primary_file`, `canonical_ref`, and `distribution_id`. The optional `files` list records the source files, sizes, and checksums used by the package job.

The source identity is distinct from the package repository name. Consumers must not infer model compatibility from a repository name alone.

### Artifact entries

Every `shared`, `layers`, and `projectors` artifact entry contains:

| Field | Description |
| --- | --- |
| `path` | Safe, repository-relative path. |
| `tensor_count` | Number of tensors in the fragment. |
| `tensor_bytes` | Total bytes occupied by tensor payloads. |
| `artifact_bytes` | Exact file size in bytes; must be greater than zero. |
| `sha256` | 64-character SHA-256 digest of the complete artifact. |

`tensor_bytes` must be zero when `tensor_count` is zero, and greater than zero when tensors are present. A projector also has a non-empty `kind`; the current schema defines only `mmproj`. Consumers must reject an unknown projector kind unless they explicitly support it.

## Stage selection

For a stage with the half-open layer range `layer_start..layer_end`, select:

1. `shared.metadata`;
2. `shared.embeddings` when the stage owns the input boundary;
3. every `layers[]` entry whose `layer_index` is in `layer_start..layer_end`;
4. `shared.output` when the stage owns the final output boundary.

The requested range must be non-empty and must not exceed `layer_count`. A materialized per-stage GGUF is derived cache output; it is not the published package format.

Projectors are selected independently:

1. an explicit `projector_path` wins;
2. otherwise, stage 0 or a single-stage runtime uses the first `mmproj` entry;
3. downstream stages do not load projector artifacts.

Consumers must use the manifest to identify projectors rather than guessing from sibling filenames.

## Generation defaults

`generation` is optional. When present, it may declare a recommended speculative-decoding strategy:

```json
{
  "generation": {
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
  }
}
```

For the current native MTP path, `type` is `native-mtp`, `prediction_depth` is `1`, and `layer_indices` must identify package layers containing the native MTP tensors. `default` must name an entry in `strategies`. An unrecognized strategy type must be ignored unless it is the declared default for a request the runtime is trying to serve.

## Validation and integrity

Before starting a stage, a consumer must verify that:

- the file is UTF-8 JSON with `schema_version: 1` and `format: "layer-package"`;
- the model identity, source identity, and Skippy ABI version are present;
- `layer_count` and `activation_width` are valid for the requested topology;
- layer entries cover every index in `0..layer_count` exactly once;
- selected paths are safe relative paths and exist in the package;
- selected file sizes match `artifact_bytes`;
- selected artifact SHA-256 digests match the manifest;
- the runtime ABI is compatible with `skippy_abi_version`;
- any selected projector is declared, supported, and valid.

Checksum verification applies to selected artifacts, including cache-hit resolutions of `hf://` packages. A peer-transfer implementation must verify each downloaded artifact's size and SHA-256 against its manifest entry before installing it, and must install downloaded files atomically from a fresh partial file.

## Package references and publishing

Package references use the `hf://` scheme:

```text
hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers
hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers:8f4c2d1
hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers@main
```

Production configurations should use an immutable commit or tag rather than a moving branch.

Create and validate a package with the package tool:

```sh
skippy-model-package write-package org/repo:distribution --out-dir model-package/
skippy-model-package validate-package /path/to/source.gguf model-package/
```

For multimodal packages, declare the projector when writing the package:

```sh
skippy-model-package write-package org/repo:distribution \
  --projector mmproj-model-f16.gguf \
  --out-dir model-package/
```

A package README should record the source coordinate and revision, source and manifest checksums, layer count, activation width, Skippy ABI, validation result, projector checksums, and any declared generation defaults.

## Compatibility rules

Schema version `1` changes should be additive when older runtimes can safely ignore new optional fields. Packages without `projectors` remain valid.

Changes to tensor ownership, layer indexing, path semantics, ABI requirements, or required fields require a new schema version or `format` value. Runtimes must reject unknown schema versions, formats, and incompatible ABI versions; they must not attempt best-effort loading.

For the implementation-level rules and peer artifact-transfer behavior, see the [layer package repository specification](https://github.com/Mesh-LLM/mesh-llm/blob/main/docs/specs/layer-package-repos.md).
