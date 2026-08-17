---
name: hf-layer-package-jobs
description: Use when changing mesh-llm automation or CLI flows that discover Hugging Face GGUF models, plan CPU Hugging Face Jobs for layer-package splitting, estimate max cost, or publish skippy layer packages/catalog entries.
metadata:
  short-description: Maintain HF layer package job automation
---

# HF Layer Package Jobs

Use this skill for the `models package` CLI, the `model-package` crate, and the
daily Unsloth queue workflow. This skill starts after a quantized GGUF artifact
exists. It does not quantize models; use `hf-gguf-quant-jobs` first or
`hf-quant-and-layer-package-jobs` when quantization and layer packaging should
run in one job.

## Workflow

1. Keep model refs in colon-selector form such as `unsloth/Qwen3-8B-GGUF:Q4_K_M`; do not split the quant into a separate `--quant` argument for generated job inputs.
2. Treat package submission as spend-bearing. The default behavior must be a dry run that prints the resolved package plan, effective timeout, selected HF Jobs hardware, and maximum cost. Require `--confirm` before submitting jobs.
3. Splitting is CPU and I/O bound. The HF Jobs hardware does not need enough RAM or VRAM to hold the full model; use CPU hardware suitable for running the splitter/build and scale timeout/cost estimates with model file size.
4. If the bucket script is stale during a confirmed submission, update it automatically before queuing jobs. Dry runs should avoid side effects.
5. The GitHub workflow should default to dry run. When confirmed, it should pass `--confirm`, submit at most the requested number of jobs, wait for every submitted HF Job, and fail if any job finishes unsuccessfully.
6. Prefer family-diverse candidate ordering after ranking by selected quant size, so one run does not consume the whole queue on a single model family.

## Commands

Preview a package job:

```bash
mesh-llm models package <gguf-repo>:<quant-selector> --dry-run
```

Submit and follow:

```bash
mesh-llm models package <gguf-repo>:<quant-selector> --confirm --follow
```

Inspect jobs:

```bash
mesh-llm models package --status <job-id>
mesh-llm models package --logs <job-id>
mesh-llm models package --list
```

For local package certification after the artifact exists:

```bash
mesh-llm models certify <layer-package-ref> --package-only --json
```

## Local Package Workflow

When the quantized GGUF is already available on the local machine, build the
package locally with `skippy-model-package`, then publish the package directory
to a Hugging Face model repo:

```bash
just build

target/debug/skippy-model-package write-package \
  <org>/<gguf-repo>:<quant-selector> \
  --out-dir /tmp/<model>-layers

target/debug/skippy-model-package preflight \
  /tmp/<model>-layers \
  --verify-sha256

hf repo create <org>/<layer-package-repo> --type model --private
hf upload <org>/<layer-package-repo> /tmp/<model>-layers . --repo-type model
```

For local GGUF paths outside the Hugging Face cache, include explicit provenance
flags on `write-package`: `--model-id`, `--source-repo`, `--source-revision`,
and `--source-file`.

## Validation

Run Rust formatting and the focused package checks before committing:

```bash
cargo fmt --all -- --check
cargo test -p model-package
cargo check -p mesh-llm-host-runtime
```

For behavior smoke tests, use a tiny dry run first:

```bash
cargo run -p model-package --bin queue-unsloth-layer-packages -- --max-jobs 1 --recent-limit 3 --popular-limit 3 --dry-run
```
