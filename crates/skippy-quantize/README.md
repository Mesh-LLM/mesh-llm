# skippy-quantize

`skippy-quantize` is the native Rust control plane for resumable GGUF
conversion and quantization jobs used by Skippy workflows. It replaces the old
Python converter and external `llama-quantize` process orchestration for this
pipeline; it does not install compatibility shims or shell out to those tools.

The crate owns:

- durable conversion and quantization manifests;
- split-GGUF progress detection and next-window planning;
- native SafeTensors-to-GGUF conversion for supported checkpoint families;
- in-process GGUF quantization through the linked llama quantization ABI;
- bounded source staging for quantization windows;
- optional output spooling with per-window publish and cleanup;
- successful-window records and JSON status/preflight output;
- tensor-type recipes for custom quant profiles such as `UD-Q3_K_S`;
- exact split artifact validation and optional llama load verification.

Build through the repo recipes:

```bash
just skippy-quantize-build
just skippy-quantize-release-build
just skippy-quantize-standalone-release-build
```

## Output

Human-readable output is the default. It uses compact emoji-labelled status
lines and progress bars for shard progress. Commands that expose `--json`
emit structured JSON instead, including `status`, `next-window`, validation,
preflight, and the resumable `run-convert[-window]` / `run-quant[-window]`
commands.

## Backends

Inspect backend capabilities:

```bash
skippy-quantize backends --json
```

`native-rust` is the HF checkpoint conversion backend. It streams tensor
payloads from SafeTensors into GGUF shards without materializing the whole model
or an output shard in memory. It currently requires tokenizer metadata from
`tokenizer.json`; checkpoints that only provide SentencePiece `tokenizer.model`
are rejected with a clear error until native SentencePiece support lands.

`llama-api` and `skippy-abi` are quantization backends. The normal
`skippy-quantize` build links the pinned llama.cpp quantization ABI into the
binary, so `llama-api` can call `llama_model_quantize` in-process without a
separate `llama-quantize` executable or a dynamic library flag:

```bash
skippy-quantize quantize \
  --backend llama-api \
  /mnt/source/BF16/model-00001-of-00002.gguf \
  /mnt/target/Q2_K/model-q2.gguf \
  Q2_K
```

`--native-runtime-library PATH` remains available for development builds that
intentionally load a dynamic llama.cpp runtime instead of using the linked ABI;
build that path with `--features dynamic-llama-quant`. Use
`--backend skippy-abi` when probing or loading the Skippy-patched runtime used
by mesh-llm.

## Convert

Create a conversion manifest:

```bash
skippy-quantize init-convert \
  --source /mnt/checkpoint \
  --target /mnt/target \
  --target-prefix BF16 \
  --output-basename GLM-5.2-BF16 \
  --output-type bf16 \
  --expected-splits 306 \
  --window-size 1 \
  --manifest /tmp/skippy-convert.json
```

Run the next missing conversion window:

```bash
skippy-quantize run-convert-window \
  --manifest /tmp/skippy-convert.json \
  --split-max-size 50G \
  --stream-buffer-bytes 8388608 \
  --spool-dir /tmp/skippy-convert-output \
  --record-dir /tmp/skippy-convert-records
```

Run conversion windows until complete:

```bash
skippy-quantize run-convert \
  --manifest /tmp/skippy-convert.json \
  --max-memory 32G \
  --stream-buffer-bytes 8388608 \
  --spool-dir /tmp/skippy-convert-output \
  --record-dir /tmp/skippy-convert-records
```

For a direct native conversion command, pass the checkpoint and desired GGUF
output path. The command derives the target prefix, output basename, manifest
path, and then runs the same resumable loop:

```bash
skippy-quantize convert \
  --output-type bf16 \
  --expected-splits 306 \
  --window-size 1 \
  --spool-dir /tmp/skippy-convert-output \
  /mnt/checkpoint \
  /mnt/target/BF16/GLM-5.2-BF16.gguf
```

Important conversion flags:

- `--output-type {auto,bf16,f16,f32}` controls the emitted GGUF tensor type.
- `--expected-splits N` declares how many output shards the job should produce.
- `--window-size N` controls how many output shards each resumable run may
  materialize.
- `--split-max-size SIZE` mirrors the intended split size in the native writer.
- `--stream-buffer-bytes BYTES` controls tensor streaming chunk size.
- `--max-memory SIZE` reduces native stream buffers and records the budget in
  job logs.
- `--mtp` writes only appended MTP draft layers where supported.
- `--no-mtp` writes the trunk and drops appended MTP draft layers.
- `--spool-dir DIR` writes window outputs to a local spool before publishing.
- `--keep-spool` keeps the spooled window after publishing.
- `--record-dir DIR` writes per-window run records.
- `--print-only` prints the planned command/report without executing.

## Quantize

Create a quantization manifest from an existing split BF16/FP16 GGUF artifact:

```bash
skippy-quantize init-quant \
  --source /mnt/bf16 \
  --source-prefix BF16 \
  --target /mnt/quant \
  --target-prefix UD-Q3_K_S \
  --output-basename GLM-5.2-UD-Q3_K_S \
  --quant UD-Q3_K_S \
  --tensor-type-file /mnt/recipe/tensor-types.txt \
  --window-size 1 \
  --manifest /tmp/skippy-quantize.json
```

Run one quantization window:

```bash
skippy-quantize run-quant-window \
  --manifest /tmp/skippy-quantize.json \
  --backend llama-api \
  --work-dir /tmp/skippy-quantize-work \
  --spool-dir /tmp/skippy-quantize-output \
  --record-dir /tmp/skippy-quantize-records
```

Run until complete:

```bash
skippy-quantize run-quant \
  --manifest /tmp/skippy-quantize.json \
  --backend llama-api \
  --max-memory 32G \
  --work-dir /tmp/skippy-quantize-work \
  --spool-dir /tmp/skippy-quantize-output
```

Important quantization flags:

- `--backend {llama-api,skippy-abi}` selects the in-process quant backend.
- `--native-runtime-library PATH` optionally loads a dynamic native runtime
  exposing `llama_model_quantize`; normal standalone builds do not need it.
- `--max-memory SIZE` is passed to the patched quantizer via
  `LLAMA_QUANTIZE_MAX_MEMORY_BYTES`.
- `--tensor-type-file PATH` applies per-tensor recipe overrides.
- `--tensor-type NAME=TYPE` adds an inline per-tensor override.
- `--imatrix PATH` loads legacy `.dat` or GGUF imatrix data.
- `--include-weights PATTERN` and `--exclude-weights PATTERN` filter imatrix
  weights.
- `--output-tensor-type TYPE` and `--token-embedding-type TYPE` override key
  tensor types.
- `--prune-layers SPEC` forwards layer-pruning metadata to the native quant
  API.
- `--override-kv KEY=TYPE:VALUE` adds GGUF metadata overrides.
- `--allow-requantize`, `--pure`, `--dry-run`, and `--leave-output-tensor`
  mirror the native llama quantization parameters.
- `--keep-split`, `--first-split`, and `--last-split` can request a manual
  split window for direct `quantize`.
- `--no-stage-source` skips local source-window staging.
- `--keep-staged-source` keeps the staged source window after success.

## Recipes

Top-level quantization modes intentionally mirror the pinned llama.cpp quant
table. Custom profile labels such as `UD-Q3_K_S` and `Q4_K_XL` are accepted as
recipe aliases when paired with `--tensor-type-file`. They resolve to the
corresponding base llama quant for backend execution while preserving the recipe
label in default output and sidecar names.

The tensor recipe format is one override per line:

```text
blk.*.ffn_gate_exps.weight=Q2_K
blk.*.ffn_down_exps.weight=Q3_K
mtp.*=Q8_0
```

Inspect supported modes and raw tensor override types:

```bash
skippy-quantize list-quants --json
skippy-quantize list-tensor-types --json
```

## Validation

Useful checks:

```bash
skippy-quantize status --manifest /tmp/skippy-quantize.json --json
skippy-quantize next-window --manifest /tmp/skippy-quantize.json --json
skippy-quantize verify-job --manifest /tmp/skippy-quantize.json --llama-load
skippy-quantize validate-tensor-types /mnt/recipe/tensor-types.txt
skippy-quantize validate-splits --root /mnt/target --prefix UD-Q3_K_S --json
```

### Reference parity smoke

Use `scripts/compare-reference-quantization.py` when changing native
conversion or quantization behavior. It compares the native Rust path against
the pinned llama.cpp reference tools:

- SafeTensors conversion: upstream `convert_hf_to_gguf.py` and
  `skippy-quantize convert` must emit the same tensor name set, shapes, types,
  and tensor payload bytes. Whole-file GGUF byte equality is not required here
  because the two writers may emit metadata and tensors in different order.
- Quantization: standalone `llama-quantize --keep-split` and
  `skippy-quantize quantize --backend llama-api` must emit byte-identical split
  GGUF outputs for every mode reported by `skippy-quantize list-quants --json`.

Conversion-only smoke:

```bash
uv run --python 3.12 \
  --with torch \
  --with transformers \
  --with numpy \
  --with sentencepiece \
  --with protobuf \
  --with gguf \
  --no-project \
  crates/skippy-quantize/scripts/compare-reference-quantization.py \
  --work-dir /tmp/skippy-quantize-conversion-parity \
  --clean \
  --skippy-quantize ./target/debug/skippy-quantize \
  --llama-quantize ./.deps/llama.cpp/build-cli/bin/llama-quantize \
  --python-converter ./.deps/llama.cpp/convert_hf_to_gguf.py \
  --checkpoint /tmp/qwen2-safetensors-fixture \
  --skip-quantization
```

All advertised quant modes:

```bash
uv run --python 3.12 \
  --with gguf \
  --with numpy \
  --no-project \
  crates/skippy-quantize/scripts/compare-reference-quantization.py \
  --work-dir /tmp/skippy-quantize-allmodes \
  --clean \
  --skippy-quantize ./target/debug/skippy-quantize \
  --llama-quantize ./.deps/llama.cpp/build-cli/bin/llama-quantize \
  --quant-input /tmp/qwen2-bf16-fixture.gguf \
  --generate-imatrix
```

`--generate-imatrix` creates a deterministic all-ones legacy imatrix from the
GGUF tensor metadata so very low-bit and IQ modes are tested instead of being
accepted as matching failures.
