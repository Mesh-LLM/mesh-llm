# MoE Scripts

## `analyze_and_publish.py`

Downloads a GGUF distribution from Hugging Face, runs `llama-moe-analyze`, writes artifacts in the canonical dataset layout from [../MOE_ANALYZE_STORAGE_SPEC.md](../MOE_ANALYZE_STORAGE_SPEC.md), and can upload those artifacts to a dataset repo.

Run it with `uv`:

```bash
uv run moe/scripts/analyze_and_publish.py \
  --source-repo unsloth/GLM-5.1-GGUF \
  --source-revision main \
  --distribution-id GLM-5.1-UD-IQ2_M \
  --analyzer-source local \
  --analyzer-bin /absolute/path/to/llama-moe-analyze \
  --analyzer-id micro-v1 \
  --dataset-repo your-org/moe-rankings
```

Bootstrap `llama-moe-analyze` from GitHub releases:

```bash
uv run moe/scripts/analyze_and_publish.py \
  --source-repo unsloth/GLM-5.1-GGUF \
  --distribution-id GLM-5.1-UD-IQ2_M \
  --analyzer-source release \
  --release-repo michaelneale/mesh-llm \
  --release-tag latest \
  --analyzer-id micro-v1 \
  --dataset-repo your-org/moe-rankings
```

Dry run:

```bash
uv run moe/scripts/analyze_and_publish.py \
  --source-repo unsloth/GLM-5.1-GGUF \
  --distribution-id GLM-5.1-UD-IQ2_M \
  --analyzer-source release \
  --dry-run
```

Notes:

- `micro-v1` runs one `llama-moe-analyze` pass per prompt and combines the resulting CSVs.
- The built-in analyzer ids do not accept `--prompt-file`.
- `micro-v1` is bound to the built-in canonical prompt set.
- `full-v1` runs a single `llama-moe-analyze` pass.
- The script forces CPU execution with `-ngl 0`.
- `--analyzer-source local` uses a locally built `llama-moe-analyze`.
- `--analyzer-source release` downloads a release bundle from GitHub and extracts `llama-moe-analyze` into `.moe-cache/tools/`.
- The canonical artifact directory shape is:

```text
data/<namespace>/<repo>/<revision>/gguf/<distribution_id>/<analyzer_id>/
```

## `submit_hf_job.py`

Submits a Hugging Face Job that runs [`analyze_and_publish.py`](/Users/jdumay/.codex/worktrees/4dc4/mesh-llm/moe/scripts/analyze_and_publish.py) remotely with the same canonical dataset output contract.

Run it with `uv`:

```bash
uv run moe/scripts/submit_hf_job.py \
  --source-repo unsloth/GLM-5.1-GGUF \
  --distribution-id GLM-5.1-UD-IQ2_M \
  --analyzer-id micro-v1 \
  --release-tag latest \
  --dataset-repo meshllm/moe-rankings \
  --flavor cpu-xl \
  --timeout 1h
```

Dry run:

```bash
uv run moe/scripts/submit_hf_job.py \
  --source-repo unsloth/GLM-5.1-GGUF \
  --distribution-id GLM-5.1-UD-IQ2_M \
  --dataset-repo meshllm/moe-rankings \
  --dry-run
```

Notes:

- The job runner always uses `--analyzer-source release` inside the remote job.
- The remote job receives `HF_TOKEN` as a secret so it can create and upload to the destination dataset repo.
- The default hardware flavor is `cpu-xl`.
- The default timeout is `1h`.
- Live HF Jobs require a GitHub release that already ships `llama-moe-analyze`.
