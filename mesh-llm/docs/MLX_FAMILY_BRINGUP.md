# MLX Family Bring-Up Workflow

Use this workflow when adding a new model family to the MLX engine and proving
that the family works cleanly across both backends.

This is the end-to-end path for:

1. choosing a candidate family
2. downloading the original upstream checkpoint
3. deriving both `GGUF` and `MLX` artifacts from the same source
4. validating the family through the `llama` and `MLX` engines
5. publishing accepted artifacts to `meshllm` on Hugging Face
6. updating the checked-in validation matrix

This document is intentionally operational. It links out to the lower-level
docs when you need exact publishing or matrix details.

## Use This For

Use this workflow for dense text families first.

Good early candidates:

- `Mistral`
- `Phi-3`
- other small or medium Llama-like families with standard safetensors layouts

Avoid as first bring-up targets unless there is a strong reason:

- large MoE families
- multimodal families
- families that require custom conversion logic on both backends

## Success Criteria

Do not call a family supported just because the loader accepts its config.

A new family is only ready when all of the following are true:

1. the source checkpoint downloads cleanly
2. the source checkpoint converts to both `GGUF` and `MLX`
3. the derived `GGUF` artifact runs through the `llama` path
4. the derived `MLX` artifact runs through the `MLX` path
5. both artifacts pass the exact validation suite
6. both artifacts are healthy in the behavior suite
7. the pair is published under `meshllm/*`
8. the validation matrix is updated to pin the new pair

## Phase 1: Pick The Family And Checkpoint

Prefer a small instruct checkpoint with:

- a public Hugging Face source repo
- a known `HF -> GGUF` conversion path
- a known `HF -> MLX` conversion path
- a size that fits local or remote test hardware

Use one exact upstream source checkpoint for both derived artifacts. Do not mix
third-party `GGUF` and `MLX` repos when the goal is backend parity.

## Phase 2: Download The Original Checkpoint

Follow the same-origin rules in
[SAME_ORIGIN_PARITY_WORKFLOW.md](/Users/jdumay/code/worktrees/mesh-llm-validation/mesh-llm/docs/SAME_ORIGIN_PARITY_WORKFLOW.md).

Keep the original checkpoint intact under:

```bash
~/.cache/mesh-llm-origin/<model-slug>
```

Inspect the source repo first:

```bash
hf download <repo-id> --dry-run
```

Then download the config, tokenizer, and weight files into the origin cache.

## Phase 3: Convert To GGUF And MLX

Derive both artifacts from the same source checkpoint.

Use the exact conversion commands in
[SAME_ORIGIN_PARITY_WORKFLOW.md](/Users/jdumay/code/worktrees/mesh-llm-validation/mesh-llm/docs/SAME_ORIGIN_PARITY_WORKFLOW.md).

Recommended local layout:

```bash
~/.cache/mesh-llm-debug/<model-slug>-same-origin/
  gguf/
  mlx/
```

Typical outcome:

- `gguf/<model-slug>-f16.gguf`
- `gguf/<model-slug>-q8_0.gguf` or `gguf/<model-slug>-q4_k_m.gguf`
- `mlx/<model-slug>-bf16/` or `mlx/<model-slug>-8bit/`

## Phase 4: Make The MLX Loader Accept The Family

This is the code bring-up step.

For new MLX families, inspect:

- [mesh-llm/src/mlx/model.rs](/Users/jdumay/code/worktrees/mesh-llm-validation/mesh-llm/src/mlx/model.rs)

In practice, the first pass is usually:

1. add the family to `config_supports_mlx`
2. map the family into `model_architecture()` if it needs non-default handling
3. add any tensor transform or tokenizer patching needed for that family
4. add focused tests for acceptance and failure modes

Do not stop at config detection. A family is not real support until the derived
artifact runs through the end-to-end validation flow below.

## Phase 5: Run Validation Through Both Engines

The validation system is documented in:

- [TESTING.md](/Users/jdumay/code/worktrees/mesh-llm-validation/mesh-llm/docs/TESTING.md)
- [testdata/validation/README.md](/Users/jdumay/code/worktrees/mesh-llm-validation/testdata/validation/README.md)

Use three layers of validation.

### 1. Exact Suite

Run the deterministic exact suite against both backends:

```bash
just build
scripts/run-validation-matrix.py --suite exact --skip-build --cases <case-id> --stamp "<family>-exact"
```

This checks:

- model load and readiness
- `/v1/chat/completions`
- deterministic prompt-following
- no leaked reasoning markup when `enable_thinking=false`
- explicit reasoning mode when configured
- `/v1/models`

Review:

- `MLX_VALIDATION_RESULTS/<stamp>/exact-summary.tsv`
- `MLX_VALIDATION_RESULTS/<stamp>/exact-cross-backend-parity.tsv`
- `MLX_VALIDATION_RESULTS/<stamp>/exact/<case-id>/chat/*.json`

### 2. Behavior Suite

Run the MT-Bench-derived behavior suite:

```bash
scripts/run-validation-matrix.py --suite behavior --skip-build --cases <case-id> --stamp "<family>-behavior"
```

This is a health check, not a benchmark. It catches:

- empty outputs
- timeout and liveness failures
- leaked reasoning markup
- repeated lines
- repeated sentences
- repeated 6-grams
- low tail-token diversity

Review:

- `MLX_VALIDATION_RESULTS/<stamp>/behavior-summary.tsv`
- `MLX_VALIDATION_RESULTS/<stamp>/behavior/<case-id>/report.json`

### 3. Cross-Backend Parity Review

Treat parity as a separate review step.

Check:

1. `GGUF run` vs `GGUF baseline`
2. `MLX run` vs `MLX baseline`
3. `MLX run` vs `GGUF baseline`

For a new family, the pair should at minimum land in the same expectation
bucket on the strict exact prompts. Prefer `same-output` where realistic.

## Phase 6: Publish Accepted Artifacts To Hugging Face

Once the derived pair is good enough, publish it under `meshllm`.

Naming convention:

- `meshllm/<model-slug>-parity-<gguf-quant>-gguf`
- `meshllm/<model-slug>-parity-<mlx-variant>-mlx`

Use the publishing commands and README skeleton from
[SAME_ORIGIN_PARITY_WORKFLOW.md](/Users/jdumay/code/worktrees/mesh-llm-validation/mesh-llm/docs/SAME_ORIGIN_PARITY_WORKFLOW.md).

Prefer:

```bash
hf repo create meshllm/<repo-name> --type model
hf upload-large-folder meshllm/<repo-name> /tmp/<publish-dir> --repo-type model
```

## Phase 7: Update The Validation Matrix

Only after publishing and accepting the pair, update the checked-in matrix.

Files to update:

- [testdata/validation/matrix.json](/Users/jdumay/code/worktrees/mesh-llm-validation/testdata/validation/matrix.json)
- [scripts/mlx-parity-exact.tsv](/Users/jdumay/code/worktrees/mesh-llm-validation/scripts/mlx-parity-exact.tsv)
- [README.md](/Users/jdumay/code/worktrees/mesh-llm-validation/README.md)

Depending on the change, also update:

- `.github/workflows/ci.yml`
- `.github/workflows/behavior.yml`

Treat this as a pinned artifact change, not a routine rerun.

## Recommended Bring-Up Checklist

Use this checklist for each new family:

1. pick one upstream checkpoint
2. download the original source checkpoint
3. derive `GGUF` and `MLX` artifacts from that same source
4. add or adjust MLX family support in code
5. run exact validation on both backends
6. run behavior validation on both backends
7. review parity artifacts and logs
8. publish both artifacts to `meshllm`
9. pin the pair in `matrix.json`
10. update the backend support matrix in `README.md`

## Notes

- Prefer dense text families before MoE and multimodal families.
- Do not update `README.md` support claims before the matrix run is clean.
- Do not publish a same-origin pair just because conversion succeeded.
- If the family needs mixed-version or protocol changes elsewhere in the repo,
  treat that as a separate review stream.
