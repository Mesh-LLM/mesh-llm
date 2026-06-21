---
name: hf-quant-and-layer-package-jobs
description: Use when running quantization of a BF16/FP16 GGUF repo and Skippy layer-package creation as one local or Hugging Face Jobs workflow, publishing both artifacts to Hugging Face.
metadata:
  short-description: Quantize and package in one workflow
---

# HF Quant And Layer Package Jobs

Use this skill when a workflow should produce both a quantized GGUF repo and a
Skippy layer package from an existing BF16/FP16 GGUF repo. The quantization
phase must use `skippy-quantize`; do not use `llama-quantize`,
`llama-quantise`, `convert_hf_to_gguf.py`, `hf_to_gguf.py`, or the misspelled
old notes form `hf_to_gguff.py`.

## Preconditions

- Source BF16/FP16 GGUF repo is complete and has a known selector/prefix.
- Target quant repo, quant selector, tensor-type file, output basename, expected
  split count, and memory budget are known.
- Target layer-package repo is known or intentionally auto-derived by
  `mesh-llm models package`.
- The layer package phase starts only after `skippy-quantize verify-job`
  succeeds for the quantized artifact.

## Local Workflow

Quantize first:

```bash
target/release/skippy-quantize init-quant \
  --source /mnt/bf16 \
  --source-prefix BF16 \
  --target /mnt/quant \
  --target-prefix <quant-selector> \
  --output-basename <model>-<quant-selector> \
  --quant <quant-selector> \
  --tensor-type-file /mnt/recipe/tensor-types.txt \
  --window-size 1 \
  --manifest /tmp/skippy-quantize.json

target/release/skippy-quantize run-quant \
  --manifest /tmp/skippy-quantize.json \
  --backend llama-api \
  --max-memory 32G \
  --work-dir /tmp/skippy-quantize-work \
  --spool-dir /tmp/skippy-quantize-output \
  --record-dir /tmp/skippy-quantize-records

target/release/skippy-quantize verify-job \
  --manifest /tmp/skippy-quantize.json \
  --llama-load
```

Publish the quant repo if the target is not already a mounted Hub repo:

```bash
hf repo create <org>/<quant-repo> --type model --private
hf upload <org>/<quant-repo> /mnt/quant . --repo-type model
```

Package the published quant:

```bash
mesh-llm models package <org>/<quant-repo>:<quant-selector> --dry-run
mesh-llm models package <org>/<quant-repo>:<quant-selector> --confirm --follow
```

Or package locally and publish:

```bash
target/debug/skippy-model-package write-package \
  <org>/<quant-repo>:<quant-selector> \
  --out-dir /tmp/<model>-layers

target/debug/skippy-model-package preflight \
  /tmp/<model>-layers \
  --verify-sha256

hf repo create <org>/<layer-package-repo> --type model --private
hf upload <org>/<layer-package-repo> /tmp/<model>-layers . --repo-type model
```

## HF Jobs Workflow

When combining both phases in one HF Job, keep the quantized GGUF repo as the
durable boundary:

1. Mount the BF16/FP16 source repo read-only.
2. Mount the target quant repo read/write.
3. Run `skippy-quantize init-quant` if the manifest is missing.
4. Run `skippy-quantize run-quant` until complete.
5. Run `skippy-quantize verify-job`; stop if it fails.
6. Submit or run the `mesh-llm models package <quant-repo>:<selector>` package
   phase.
7. Record both the quant repo commit and the layer-package repo commit.

Template:

```bash
hf jobs uv run \
  --namespace meshllm \
  --flavor cpu-upgrade \
  --timeout 4d \
  --secrets HF_TOKEN \
  --volume hf://models/<bf16-repo>:/mnt/bf16 \
  --volume hf://models/<quant-repo>:/mnt/quant \
  --env SKIPPY_QUANTIZE_OUTPUT=json \
  --env PYTHONUNBUFFERED=1 \
  --detach \
  /path/to/skippy_quant_then_package_job.py \
  -- \
  --source /mnt/bf16 \
  --source-prefix BF16 \
  --target /mnt/quant \
  --target-prefix <quant-selector> \
  --output-basename <model>-<quant-selector> \
  --quant <quant-selector> \
  --tensor-type-file /mnt/recipe/tensor-types.txt \
  --package-ref <org>/<quant-repo>:<quant-selector> \
  --max-memory 32G
```

## Resume Rules

- If quant shards already exist, `skippy-quantize` resumes at the first missing
  shard.
- If the quant repo verifies successfully, skip quantization and run or inspect
  the package job.
- Do not delete a verified quant repo to force a clean package run. Package jobs
  should consume the published quant artifact as the source of truth.

## Validation

Before promoting the combined run, record:

- source BF16/FP16 repo revision;
- quant repo commit, quant selector, tensor recipe, split count, and verify
  output;
- layer-package job id, target repo, target commit, and package certification;
- total HF job cost and whether the combined workflow saved time or only saved
  operator steps.
