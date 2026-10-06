# Optional Granite BF16 reference evaluation

This project is solely for the existing `evals/skippy-granite-tensor-equivalence.py` model reference evaluator. It is not a prerequisite of normal build, packaging, required quality or benchmark execution. Python3.12 is deliberately selected, matching an existing repository optional compatibility project and an available local interpreter. The resolver lock is genuine; imports/API compatibility and supplied model values have not been tested.

Prepare explicitly after policy review, from the repository root:

```bash
uv sync --locked --no-python-downloads --project evals/granite-reference --python python3.12
```

Run using the isolated project interpreter, with no synchronization or implicit package installation:

```bash
evals/granite-reference/.venv/bin/python -I evals/skippy-granite-tensor-equivalence.py \
  --gguf /path/to/granite-4.0-h-1b-bf16.gguf \
  --safetensors /path/to/model.safetensors
```

The `-I` interpreter ignores ambient Python path/user site configuration. The script retains its full model-specific tensor checks, nonzero refusal and canonical source-value digest. It compares supplied local artifacts; it downloads nothing. Run outside timed benchmark windows. Do not treat dependency resolution or benchmark pins as evidence that those artifacts passed.

This optional evaluation still needs owning required-path isolation policy checks before its exact-path retention row is admitted. A prepared environment can then undergo an explicit four-import smoke check; real artifacts remain operator supplied and separately authorized. No new Python tooling is added.
