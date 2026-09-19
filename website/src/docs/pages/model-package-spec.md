---
title: Model Package Specification
---

# `model-package.json` specification

`model-package.json` is the root manifest for a Skippy package-v2 repository. It binds an immutable source-model identity to a catalog of GGUF artifacts. Stage ownership is not encoded by filenames or fixed layer ranges: the planner selects an exact tensor closure, and package admission resolves those tensor IDs to their declared storage.

The current manifest schema is version `2`. The root manifest is intentionally small. The metadata artifact named by `source_model.metadata_artifact_id` carries the normalized model metadata and complete tensor catalog used after its bytes have been verified.

## Package repository

The manifest must be at the repository root. A typical package has this shape:

```text
model-package.json
shared/
  metadata.gguf
  common.gguf
layers/
  layer-00000.gguf
  layer-00001-part00.gguf
  layer-00001-part01.gguf
projectors/
  projector-00000.gguf
README.md
```

Artifact filenames describe physical containers only. Large common or layer groups may be split into deterministic part files. Consumers must follow `artifact_catalog.entries` and tensor storage bindings rather than infer ownership from paths.

Every artifact path must be a safe path relative to the package root. Absolute paths, parent traversal, duplicate paths, and duplicate artifact IDs are invalid.

## Root manifest shape

Values such as IDs, checksums, sizes, and timestamps are illustrative:

```json
{
  "schema_version": 2,
  "package_id": "sha256:<64 hex characters>",
  "model_id": "Qwen/Qwen3-235B-A22B-GGUF:UD-Q4_K_XL",
  "source_model": {
    "sha256": "<64 hex characters>",
    "metadata_artifact_id": "metadata",
    "repo": "Qwen/Qwen3-235B-A22B-GGUF",
    "revision": "<source commit>",
    "primary_file": "Qwen3-235B-A22B-UD-Q4_K_XL.gguf",
    "canonical_ref": "Qwen/Qwen3-235B-A22B-GGUF:UD-Q4_K_XL",
    "distribution_id": "UD-Q4_K_XL",
    "files": [
      {
        "path": "Qwen3-235B-A22B-UD-Q4_K_XL.gguf",
        "byte_size": 123,
        "sha256": "<64 hex characters>"
      }
    ]
  },
  "format": "gguf",
  "layer_count": 94,
  "artifact_catalog": {
    "entries": [
      {
        "id": "metadata",
        "path": "shared/metadata.gguf",
        "byte_size": 123,
        "sha256": "<64 hex characters>"
      },
      {
        "id": "common",
        "path": "shared/common.gguf",
        "byte_size": 123,
        "sha256": "<64 hex characters>"
      },
      {
        "id": "layer-00000",
        "path": "layers/layer-00000.gguf",
        "byte_size": 123,
        "sha256": "<64 hex characters>"
      },
      {
        "id": "projector-00000",
        "path": "projectors/projector-00000.gguf",
        "byte_size": 123,
        "sha256": "<64 hex characters>"
      }
    ]
  },
  "sidecars": [
    {
      "kind": "mmproj",
      "artifact_id": "projector-00000",
      "name": "projector-00000"
    }
  ],
  "native_abi_version": "1.2.3",
  "generator_version": "0.9.0",
  "created_at_unix_secs": 1790000000
}
```

## Root fields

| Field | Required | Description |
| --- | --- | --- |
| `schema_version` | Yes | Must be `2`. |
| `package_id` | Yes | Canonical `sha256:` identity computed from the normalized root manifest. |
| `model_id` | Yes | Non-empty model coordinate, including its distribution or quantization identity. |
| `source_model` | Yes | Provenance and complete source-file inventory. |
| `format` | Yes | Must be `gguf` for packages produced and verified by the current tool. |
| `layer_count` | Yes | Positive model layer count used to bound catalogued layer ordinals. |
| `artifact_catalog.entries` | Yes | Non-empty catalog of immutable package artifacts. |
| `sidecars` | No | Package-level sidecar bindings; the current sidecar kind is `mmproj`. |
| `generation` | No | Package-owned generation or speculative-decoding declarations. |
| `native_abi_version` | Yes | Native Skippy ABI required by the package. |
| `generator_version` | Yes | Version of the package generator. |
| `created_at_unix_secs` | Yes | Package creation time; normalized to zero when computing `package_id`. |

### Source model identity

`source_model.sha256` must equal the digest of `source_model.primary_file`. The primary file must appear in `source_model.files`, and every source-file entry contains a safe relative `path`, exact `byte_size`, and SHA-256 digest. Repository coordinates are provenance only; consumers must not infer compatibility from a repository name.

`source_model.metadata_artifact_id` names the catalog entry that carries the model metadata and tensor inventory. It must resolve to a GGUF artifact and cannot also be a sidecar.

### Artifact catalog

Each artifact entry contains:

| Field | Description |
| --- | --- |
| `id` | Stable, unique artifact identifier referenced by tensor storage and sidecars. |
| `path` | Unique, safe, repository-relative file path. |
| `byte_size` | Exact complete-file size in bytes. |
| `sha256` | SHA-256 digest of the complete artifact. |

The catalog is the only authoritative artifact inventory. Paths such as `shared/common.gguf` and `layers/layer-00000-part01.gguf` are naming conventions, not stage-selection rules.

## Metadata carrier and tensor catalog

After verifying `shared/metadata.gguf` against its artifact entry, the runtime resolves the metadata carrier and obtains:

- normalized GGUF model metadata, including `general.architecture`;
- one tensor-catalog entry for every source tensor;
- each tensor's stable ID, native name, GGML type, dimensions, and optional layer ordinal;
- an owned storage binding with `artifact_id`, aligned `data_offset`, `stored_length`, and integrity policy, or an explicit alias binding.

Tensor IDs and native names are distinct. Admission selects strictly sorted, unique tensor IDs from the graph-derived plan. The runtime resolves those IDs to artifacts and native names; it must not select tensors by filename, role, or layer range. The metadata artifact is always required in addition to the artifacts referenced by the selected tensors and sidecars.

## Generation declarations

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

Generation declarations describe package capabilities, not tensor ownership or stage-selection policy. Named defaults and proposer or strategy references must resolve within the declaration, and their numeric bounds must be valid.

## Validation and integrity

Before admitting a package or starting a stage, a consumer must verify that:

- the root is UTF-8 JSON with `schema_version: 2` and `format: "gguf"`;
- required identities, version fields, source inventory, and `artifact_catalog.entries` are present and structurally valid;
- `package_id` matches the canonical manifest identity;
- the metadata artifact ID resolves to a declared non-sidecar GGUF artifact;
- every catalogued artifact path stays within the package root, exists, and has the declared `byte_size` and SHA-256 digest;
- the verified metadata carrier resolves to non-empty model metadata and a non-empty tensor catalog;
- tensor IDs and names are unique, layer ordinals are in bounds, and owned byte ranges are aligned, non-empty, non-overlapping, and contained by their declared artifacts;
- every requested stage tensor ID and sidecar is sorted, unique, declared, and resolvable;
- the runtime native ABI is compatible with `native_abi_version`;
- independent source evidence has the same source-file identity and exact tensor inventory when certifying with `verify-package-v2`.

`verify-package-v2` is source-complete verification. It reads the package and an independently supplied source, verifies every artifact's size and digest, compares tensor metadata and payload bytes, checks exact coverage, and validates declared projector sidecars. A package must not use one of its own artifacts as that independent source.

Checksum verification also applies to cache-hit resolutions and peer-transferred artifacts. Downloads must be checked before installation and installed atomically from a fresh partial file.

## Package references and publishing

Package references use the `hf://` scheme:

```text
hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers
hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers:8f4c2d1
hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers@main
```

Production configurations should use an immutable commit or tag rather than a moving branch.

Create and validate a v2 package with the package tool:

```sh
skippy-model-package write-package org/repo:distribution --out-dir model-package/
skippy-model-package verify-package-v2 model-package/ --source /path/to/source.gguf
```

For multimodal packages, declare each independent projector source when writing and verifying the package:

```sh
skippy-model-package write-package org/repo:distribution \
  --projector mmproj-model-f16.gguf \
  --out-dir model-package/

skippy-model-package verify-package-v2 model-package/ \
  --source /path/to/source.gguf \
  --source-projector /path/to/mmproj-model-f16.gguf
```

A package README should record the immutable source coordinate, source and manifest checksums, package ID, layer count, native ABI, generator version, verification result, projector checksums, and any declared generation defaults.

## Compatibility rules

Schema-v1 layer-entry manifests and schema-v2 artifact catalogs are different formats. Current package-v2 tooling rejects v1 input rather than guessing a migration. Unknown schema versions, unknown fields, invalid formats, and incompatible native ABI versions are rejected.

Changes to identity calculation, artifact or tensor storage semantics, admission rules, or required fields require a new schema version unless old readers can safely preserve the exact contract.

For the implementation-level rules and peer artifact-transfer behavior, see the [layer package repository specification](https://github.com/Mesh-LLM/mesh-llm/blob/main/docs/specs/layer-package-repos.md).
