---
name: hf-bf16-gguf-conversion-jobs
description: Use when converting Hugging Face SafeTensors checkpoints into split BF16 GGUF model repos with skippy-quantize on Hugging Face Jobs or a local machine, then publishing the artifact to Hugging Face.
metadata:
  short-description: Convert HF checkpoints to BF16 GGUF repos
---

# HF BF16 GGUF Conversion Jobs

Use this skill when the source artifact is a Hugging Face checkpoint repo and
the target artifact is a split BF16 GGUF model repo. The operational tool is
`skippy-quantize`; do not use `convert_hf_to_gguf.py`, `hf_to_gguf.py`, or a
wrapper that shells out to either script. Treat `hf_to_gguff.py` as the same
forbidden path if it appears in old notes or logs.

## Preconditions

- Confirm the source checkpoint repo, revision, tokenizer files, target repo,
  output basename, expected split count, and desired split size before spending
  HF Jobs credits.
- Build the standalone binary with `just skippy-quantize-standalone-release-build`
  for local runs or in the job image/script for HF Jobs.
- Use `--output-type bf16` unless the experiment explicitly records a different
  target precision.
- Prefer a split output with `--window-size 1` for first full-model runs. Raise
  the window only after a smaller fixture proves the memory and I/O budget.
- Publish only complete windows, write per-window records, and resume from the
  first missing target shard after cancellation.

## Local Workflow

Create a manifest:

```bash
target/release/skippy-quantize init-convert \
  --source /path/to/checkpoint \
  --target /path/to/output-repo \
  --target-prefix BF16 \
  --output-basename <model>-BF16 \
  --output-type bf16 \
  --expected-splits <N> \
  --window-size 1 \
  --manifest /tmp/skippy-convert.json
```

Dry-run the next conversion window before spending I/O:

```bash
target/release/skippy-quantize convert-job \
  --source /path/to/checkpoint \
  --target /path/to/output-repo \
  --target-prefix BF16 \
  --output-basename <model>-BF16 \
  --output-type bf16 \
  --expected-splits <N> \
  --window-size 1 \
  --manifest /tmp/skippy-convert.json \
  --max-memory 32G \
  --dry-run
```

Run until complete:

```bash
target/release/skippy-quantize run-convert \
  --manifest /tmp/skippy-convert.json \
  --max-memory 32G \
  --split-max-size 50G \
  --stream-buffer-bytes 8388608 \
  --spool-dir /tmp/skippy-convert-output \
  --record-dir /tmp/skippy-convert-records \
  --json-event-file /tmp/skippy-convert-status.json \
  --json-event-interval-seconds 120 \
  --json-event-window 8
```

Validate and publish:

```bash
target/release/skippy-quantize verify-job \
  --manifest /tmp/skippy-convert.json \
  --json

hf repo create <org>/<target-repo> --type model --private
hf upload <org>/<target-repo> /path/to/output-repo . --repo-type model
```

## HF Jobs Workflow

Mount the source checkpoint and target model repo rather than downloading the
whole checkpoint into the job filesystem:

```bash
hf jobs uv run \
  --namespace meshllm \
  --flavor cpu-upgrade \
  --timeout 3d \
  --secrets HF_TOKEN \
  --volume hf://models/<source-repo>:/mnt/checkpoint \
  --volume hf://models/<target-repo>:/mnt/target \
  --env SKIPPY_QUANTIZE_OUTPUT=json \
  --env PYTHONUNBUFFERED=1 \
  --detach \
  /path/to/skippy_convert_job.py \
  -- \
  --source /mnt/checkpoint \
  --target /mnt/target \
  --target-prefix BF16 \
  --output-basename <model>-BF16 \
  --expected-splits <N> \
  --split-max-size 50G \
  --max-memory 32G
```

The job script should only build or install `skippy-quantize`, create the
manifest if missing, run `run-convert`, verify the job, and upload sidecars. It
must not call the old Python converter.

## Monitoring

Use both HF Jobs status and `skippy-quantize` status:

```bash
hf jobs inspect <job-id> --namespace meshllm
hf jobs logs <job-id> --namespace meshllm --tail 120
target/release/skippy-quantize status --manifest /tmp/skippy-convert.json --json
```

For agents, prefer polling `/tmp/skippy-convert-status.json` over ingesting full
logs. Healthy snapshots show phase movement through `running`, `publishing`,
and `complete`, with only the last few high-level events retained. Stop and
diagnose if the same window restarts without a new published shard or memory
stays pinned near the hardware limit.

## Record Keeping

Record the job id, exact command, source revision, target repo commit, split
count, split size, memory budget, tokenizer notes, and follow-ups in the
experiment card or phase iteration card before promoting the artifact.
