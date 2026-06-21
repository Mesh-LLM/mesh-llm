---
name: hf-gguf-quant-jobs
description: Use when creating, monitoring, validating, or documenting low-memory Hugging Face Jobs or local runs that quantize split BF16/FP16 GGUF model repos into custom quant GGUF repos with skippy-quantize.
---

# HF GGUF Quant Jobs

Use this skill to turn an existing split BF16/FP16 GGUF model repo into a
quantized GGUF model repo without requiring the host to hold the full model in
memory or on local disk at once. The operational tool is `skippy-quantize`; do
not use `llama-quantize`, `llama-quantise`, or wrapper scripts that shell out to
those binaries.

The supported pattern is: mount or point at the source BF16/FP16 GGUF repo,
quantize resumable split windows with `skippy-quantize`, publish completed
output shards to the target model repo, delete staged files immediately, and
resume from the first missing target shard after cancellation or failure.

## Preconditions

- Use a split BF16/FP16 GGUF repo as the source when possible. Do not re-read
  SafeTensors for requants if a BF16 GGUF artifact already exists.
- Verify the source repo is complete before spending on quantization. Count all
  expected split shards and refuse to run if any are missing.
- Use a tensor-type file for any custom recipe. Treat MTP tensors, output
  tensors, precision-sensitive tensors, and latency-sensitive layer ranges as
  explicit recipe inputs.
- Run jobs under the intended HF org and pass `HF_TOKEN` as a secret, not a
  printed environment variable.
- Prefer mounted Hub repos over full `hf download` when the job only needs to
  stream or stage one shard/window at a time.
- Build the standalone binary with `just skippy-quantize-standalone-release-build`
  for local runs or in the job image/script for HF Jobs.

## Workflow

1. Identify the source BF16/FP16 GGUF repo, target quant repo, output prefix,
   output basename, source prefix, quant type, tensor-type file, memory budget,
   and split window size.
2. Preflight both Hub and mounted source paths with `skippy-quantize status`,
   `next-window`, `validate-splits`, or a `quantize --preflight-only` run. Stop
   if the source artifact is incomplete.
3. Write or upload a `quant-plan.json` with source repo/revision, target repo,
   quant type, shard count, output prefix, tensor policy, and resume
   settings.
4. Launch the job with `--window-size 1` for the first full model run unless a
   smaller fixture proves a larger window is safe on the chosen hardware.
5. For each split window, stage only the required input shard, run
   `skippy-quantize run-quant-window` or `run-quant`, publish finished shards,
   then delete local staged input and output files.
6. Monitor for progress markers. A healthy job repeatedly emits staged source
   copies, `quant_window`, publish completion, cleanup, and increasing split
   progress.
7. Validate the target repo after completion by counting GGUF shards, checking
   the first and last shard names, and confirming `quant-plan.json` plus the
   tensor-type file are present.
8. Record the artifact in the experiment card and create an iteration card for
   the run, including job id, command, environment, repo SHA, shard count, and
   follow-up decisions.

## Launch Template

Create a quantization manifest:

```bash
target/release/skippy-quantize init-quant \
  --source /mnt/source-gguf \
  --source-prefix <source-prefix> \
  --target /mnt/target-quant \
  --target-prefix <target-prefix> \
  --output-basename <output-basename> \
  --quant <quant> \
  --tensor-type-file /mnt/recipe/tensor-types.txt \
  --window-size 1 \
  --manifest /tmp/skippy-quantize.json
```

Run until complete:

```bash
target/release/skippy-quantize run-quant \
  --manifest /tmp/skippy-quantize.json \
  --backend llama-api \
  --max-memory 32G \
  --work-dir /tmp/skippy-quantize-work \
  --spool-dir /tmp/skippy-quantize-output \
  --record-dir /tmp/skippy-quantize-records
```

For HF Jobs, mount the BF16/FP16 source repo and target quant repo, then run the
same manifest and `run-quant` commands inside the job:

```bash
hf jobs uv run \
  --namespace meshllm \
  --flavor cpu-upgrade \
  --timeout 3d \
  --secrets HF_TOKEN \
  --volume hf://models/<source-repo>:/mnt/source-gguf \
  --volume hf://models/<target-repo>:/mnt/target-quant \
  --env SKIPPY_QUANTIZE_OUTPUT=json \
  --env PYTHONUNBUFFERED=1 \
  --detach \
  /path/to/skippy_quant_job.py \
  -- \
  --source /mnt/source-gguf \
  --source-prefix <source-prefix> \
  --target /mnt/target-quant \
  --target-prefix <target-prefix> \
  --output-basename <output-basename> \
  --quant <quant> \
  --tensor-type-file /mnt/recipe/tensor-types.txt \
  --max-memory 32G
```

The job script should only build or install `skippy-quantize`, prepare the
manifest if missing, run `run-quant`, verify the job, and upload sidecars.

## Monitoring

Check status and logs:

```bash
hf jobs inspect <job-id> --namespace meshllm
hf jobs logs <job-id> --namespace meshllm --tail 120
```

Useful healthy markers:

- `Preflight QuantizeGguf with backend llama-api`
- `Source artifact is complete`
- `quant_window`
- `Published /mnt/target-quant/...`
- `Cleaned staged source`
- `split artifact ... 100.00%`

Concerning markers:

- repeated watchdog lines with no shard, tensor, upload, or cache-drop progress;
- cgroup memory pinned near the hardware limit;
- the same split window restarting repeatedly without new uploaded target files;
- fallback quant warnings for tensors that the recipe expected to preserve.

If a job stalls, cancel it before changing code or hardware. The next run should
skip already published shards and resume at the first missing output shard.

## Validation

After completion, verify the target repo with an authenticated Hub API or CLI
check. Record at least:

- target repo and commit SHA;
- privacy setting;
- total file count;
- GGUF shard count;
- first and last shard names;
- manifest/plan presence;
- tensor-type file presence.

For local smoke tests, use a small split GGUF source first and verify:

- `skippy-quantize verify-job --manifest <manifest> --llama-load` succeeds;
- `skippy-quantize validate-splits --root <target> --prefix <prefix>` succeeds;
- max RSS stays bounded compared with full-model size;
- `skippy-quantize status --manifest <manifest> --json` reports completion.

## Documentation Contract

For Jianyang-style experiments, update both records:

- the main experiment card with the promoted artifact;
- a phase iteration card with the job id, exact command, environment,
  verification output, decision, and follow-ups.

Keep post-experiment upstream notes separate from the run decision. The job can
be successful while the converter or quantizer patches still need extraction
into clean upstream PRs.
