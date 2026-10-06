---
name: hf-bf16-gguf-conversion-jobs
description: Use when converting Hugging Face SafeTensors checkpoints into split BF16 GGUF model repos with skippy-quantize on Hugging Face Jobs or a local machine, then publishing the artifact to Hugging Face.
metadata:
  short-description: Convert HF checkpoints to BF16 GGUF repos
---

# HF BF16 GGUF Conversion Jobs

New repository-side job planning and orchestration follows
`../manage-ci/SKILL.md`: no new Python tooling; use typed `tools/xtask`
commands behind thin Just recipes. From the repository root,
`cargo xtool repo-consistency ci-crate-lists` is a working alias example, not
a conversion command. Existing Hugging Face Jobs examples remain transitional
ecosystem invocations, not approval for new Python automation.

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
- Provision a pinned prepared native xtask executable and an explicit conversion manifest containing complete source pins, helper/source pins, credential-file reference and observed G3 bootstrap inputs. Set `MESH_LLM_AUTOMATION_BIN` to that absolute regular executable. The native job caller is `scripts/hf-skippy-convert-job.sh`; it does not install Python packages or provisioning tools. Normal conversion observes the canonical mesh checkout, pinned llama preparation and fresh Just-built CPU converter through G3; upload-only omits rebuilding.

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

Validate the native conversion manifest with `skippy-quantize verify-job --manifest /tmp/skippy-convert.json --json`. To publish an existing complete output through the native job caller, supply the same prepared manifest and fresh evidence directory:

```bash
MESH_LLM_AUTOMATION_BIN=/prepared/xtask scripts/hf-skippy-convert-job.sh \
  --input /prepared/conversion-job.json \
  --output-directory /data/upload-evidence \
  --source-repo <source-repo> --target-repo <target-repo> \
  --output-basename <model>-BF16 --mesh-revision <immutable-mesh-commit> \
  --upload-only --confirm-publication
```

## HF Jobs Workflow

Use the same native caller in an explicitly approved prepared Linux job image with the complete immutable checkpoint mounted at `/mnt/checkpoint` and the manifest and private credential file supplied separately. Default source/work paths are `/mnt/checkpoint` and `/data/skippy-convert`; the work parent must exist. The manifest selects exact tool/source/image/resource declarations. Tool installation is an image provisioning responsibility, not an inline job fallback.

```bash
MESH_LLM_AUTOMATION_BIN=/prepared/xtask scripts/hf-skippy-convert-job.sh \
  --input /prepared/conversion-job.json \
  --output-directory /data/conversion-evidence \
  --source-repo <source-repo> --target-repo <target-repo> \
  --output-basename <model>-BF16 --mesh-revision <immutable-mesh-commit> \
  --expected-splits <N> --split-max-size 50G --max-memory 24G \
  --confirm-publication
```

Normal execution uses the observed fresh G3 binary, exact BF16/MTP conversion, window 1, spool/per-window records/status sidecar and verifier, then provisions the public repository and publishes the entire admitted shard/sidecar roster at one immutable commit. `--dry-run` cannot combine with publication or upload-only. `--xet-high-performance` records the request and warns that this transport uses basic/multipart without Xet performance qualification. Custom mesh repositories and publisher rosters over 128 shards/32 JSON/MD/TXT sidecars are refused; do not silently substitute a different artifact set. Real source/model/tool/HF execution remains a separate acceptance gate.

The local caller executes inside an already authorized job. Native generic Jobs submission and immutable retrieval use the delivery facade below; supplying a qualified image remains a separate operator prerequisite.

## Monitoring

Use the existing bounded native Jobs monitor for the admitted job ID, and `skippy-quantize` status for the artifact. Use the native generic Jobs delivery facade below for confirmed submit and request-correlated immutable retrieval; do not substitute an inline SDK/Python wrapper.

```bash
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

## Native generic Jobs delivery

Use the prepared immutable image/runner/tool/source contracts in `docs/skippy/HF_GENERIC_CONVERSION_JOBS.md`. `model-package-generic-jobs prepare` is read-only; confirmed submit returns a sanitized acknowledgment, and collect admits only request-correlated immutable conversion receipts. The original72h work budget remains259200seconds. Upload-only uses a complete immutable mounted artifact roster without rebuilding or reconverting. Original image provisioning and real cloud/model qualification remain separate requirements; local fixtures do not authorize paid Jobs or publication.
