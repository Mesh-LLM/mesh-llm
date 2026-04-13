# Same-Origin Parity Workflow

Use this workflow when a public GGUF/MLX pair is noisy and we want a cleaner
backend comparison derived from the same original checkpoint.

This is the process we used for:

- `Qwen2.5`
- `Gemma 2`
- `Gemma 3`
- `Gemma 4`

The goal is:

1. download the original checkpoint once
2. derive both `GGUF` and `MLX` artifacts from that same source
3. run the exact validation suite against those derived artifacts
4. only publish and switch the matrix when the pair is good enough to bless

## Preconditions

You need:

- `HF_TOKEN` exported with access to any gated upstream repos
- the bundled `llama.cpp` tools built via `just build`
- a Python environment with:
  - `transformers` for `convert_hf_to_gguf.py`
  - `mlx_lm` for `python -m mlx_lm.convert`

If the default `python3` does not provide those modules, use the Python
interpreter from the environment that does.

## Directory Conventions

Use these paths consistently:

| Purpose | Path pattern |
|---|---|
| original source checkpoint | `~/.cache/mesh-llm-origin/<model-slug>` |
| local derived artifacts | `~/.cache/mesh-llm-debug/<model-slug>-same-origin/` |
| local exact artifacts | `MLX_VALIDATION_RESULTS/<stamp>/...` |

Example model slugs:

- `qwen2.5-0.5b-instruct`
- `gemma-3-1b-it`
- `gemma-4-e4b-it`

## 1. Download The Original Checkpoint

First inspect the upstream repo and confirm the real weight filenames:

```bash
hf download <repo-id> --dry-run
```

For the large batch of additional parity families on `studio54`, use:

```bash
zsh scripts/download-origin-checkpoints-studio54.sh
```

To fetch only a subset:

```bash
zsh scripts/download-origin-checkpoints-studio54.sh mistral phi3 deepseek
```

The script downloads into `~/.cache/mesh-llm-origin-batch/` by default and
continues past gated or failed repos so the rest of the batch can proceed.

By default it excludes families that are too large or risky for the full
download-convert-validate workflow on `studio54`'s `128 GB` M1 Ultra:

- `mixtral`
- `cohere-command-r`
- `jamba`

Those can be added explicitly with:

```bash
zsh scripts/download-origin-checkpoints-studio54.sh --include-heavy
```

For metadata and tokenizer files, `hf download` is fine:

```bash
mkdir -p ~/.cache/mesh-llm-origin/<model-slug>
hf download <repo-id> \
  config.json generation_config.json tokenizer.json tokenizer_config.json \
  special_tokens_map.json README.md LICENSE* USE_POLICY* \
  --local-dir ~/.cache/mesh-llm-origin/<model-slug>
```

For large weight files, prefer direct resumable downloads. This avoids the
stale lock and partial-download problems we hit with `hf download`:

```bash
curl -L -C - \
  -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/<repo-id>/resolve/main/<weight-file> \
  -o ~/.cache/mesh-llm-origin/<model-slug>/<weight-file>
```

If the repo uses sharded weights, repeat that for each shard:

```bash
curl -L -C - \
  -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/<repo-id>/resolve/main/model-00001-of-00002.safetensors \
  -o ~/.cache/mesh-llm-origin/<model-slug>/model-00001-of-00002.safetensors
```

Notes:

- Download source checkpoints in serial unless there is a specific reason to do
  otherwise.
- Keep the original checkpoint intact under `~/.cache/mesh-llm-origin/`.
- Do not use third-party GGUF or MLX repos when the goal is same-origin parity.

## 2. Convert To GGUF

Create a high-fidelity GGUF first, then quantize if needed:

```bash
SRC=~/.cache/mesh-llm-origin/<model-slug>
OUT=~/.cache/mesh-llm-debug/<model-slug>-same-origin
mkdir -p "$OUT/gguf"

python3 llama.cpp/convert_hf_to_gguf.py "$SRC" \
  --outfile "$OUT/gguf/<model-slug>-f16.gguf" \
  --outtype f16
```

Quantize from that high-fidelity GGUF:

```bash
./llama.cpp/build/bin/llama-quantize \
  "$OUT/gguf/<model-slug>-f16.gguf" \
  "$OUT/gguf/<model-slug>-q8_0.gguf" \
  Q8_0
```

Or:

```bash
./llama.cpp/build/bin/llama-quantize \
  "$OUT/gguf/<model-slug>-f16.gguf" \
  "$OUT/gguf/<model-slug>-q4_k_m.gguf" \
  Q4_K_M
```

Use the quant class that best matches the MLX artifact you intend to compare.

## 3. Convert To MLX

For quantized MLX:

```bash
SRC=~/.cache/mesh-llm-origin/<model-slug>
OUT=~/.cache/mesh-llm-debug/<model-slug>-same-origin
mkdir -p "$OUT/mlx"

python3 -m mlx_lm.convert \
  --hf-path "$SRC" \
  --mlx-path "$OUT/mlx/<model-slug>-8bit" \
  -q \
  --q-bits 8
```

For high-fidelity MLX:

```bash
python3 -m mlx_lm.convert \
  --hf-path "$SRC" \
  --mlx-path "$OUT/mlx/<model-slug>-bf16" \
  --dtype bfloat16
```

If your `mlx_lm` environment uses a different Python binary, use that binary
instead of `python3`.

## 4. Run Exact Validation

Run the local exact suite against the derived artifacts before publishing
anything:

```bash
STAMP=<model-slug>-same-origin-$(date +%Y%m%d)
scripts/run-validation-matrix.py --suite exact --skip-build --cases <case-id> --stamp "$STAMP"
```

If you need a direct one-off case:

```bash
VALIDATION_RESULTS_ROOT="$PWD/MLX_VALIDATION_RESULTS" \
VALIDATION_RESULTS_STAMP="$STAMP" \
scripts/run-validation-case.sh gguf <case-id> \
python3 scripts/ci-exact-smoke.py \
  --backend gguf \
  --mesh-llm target/release/mesh-llm \
  --bin-dir llama.cpp/build/bin \
  --model "$OUT/gguf/<artifact>.gguf" \
  --prompt-suite-json "$PWD/testdata/validation/exact-prompts.json"
```

Review at minimum:

- `MLX_VALIDATION_RESULTS/<stamp>/exact-summary.tsv`
- `MLX_VALIDATION_RESULTS/<stamp>/exact-cross-backend-parity.tsv`
- per-prompt chat artifacts under:
  - `MLX_VALIDATION_RESULTS/<stamp>/exact/<case-id>/chat/`

Do not publish or switch the matrix just because a conversion succeeded.
Publish only if the derived pair gives us a cleaner parity story than the public
pair.

## 5. Publish To Hugging Face

Use `meshllm` model repos and make the pair naming explicit:

| Backend | Repo naming pattern |
|---|---|
| GGUF | `meshllm/<model-slug>-parity-<gguf-quant>-gguf` |
| MLX | `meshllm/<model-slug>-parity-<mlx-variant>-mlx` |

Examples:

- `meshllm/qwen2.5-0.5b-instruct-parity-q8_0-gguf`
- `meshllm/qwen2.5-0.5b-instruct-parity-8bit-mlx`
- `meshllm/gemma-3-1b-it-parity-f16-gguf`
- `meshllm/gemma-3-1b-it-parity-bf16-mlx`

Create a temp publish directory containing:

- the artifact
- a `README.md` with YAML front matter

Minimal `README.md` skeleton:

```md
---
license: other
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - meshllm
  - parity
  - same-origin
---

# <title>

Same-origin parity artifact derived from `<upstream-repo>`.
```

Then create the repo and upload:

```bash
hf repo create meshllm/<repo-name> --type model
hf upload-large-folder meshllm/<repo-name> /tmp/<publish-dir> --repo-type model
```

Use `upload-large-folder` for the large artifacts rather than one-off `hf upload`.

## 6. Switch The Matrix

Only after the published pair is accepted, update:

- `testdata/validation/matrix.json`
- `scripts/mlx-parity-exact.tsv`
- `.github/workflows/ci.yml`
- `.github/workflows/behavior.yml`
- `MLX_VALIDATION_MATRIX.md`

The matrix row should say:

- what upstream original checkpoint the pair was derived from
- whether behavior results are current, stale, or pending
- why this pair is better than the previous public pair

## 7. Remote Confirmation

If the pair is important, rerun it on `studio54.local` before treating it as the
canonical row.

Use the remote rules from `AGENTS.md`:

- launch long-running work in `tmux`
- use `zsh -lc` on macOS
- prefer `scp` + small remote scripts for nontrivial jobs
- verify the `tmux` session twice before calling it live

## Current Publishing Rule

Use this same-origin workflow selectively.

Good candidates:

- rows with suspicious public-pair drift
- small and medium canary models
- families where parity conclusions matter operationally

Do not publish a same-origin pair just because it exists. Publish it when it
gives us a materially better canonical parity row.
