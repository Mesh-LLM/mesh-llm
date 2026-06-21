# skippy-quantize

`skippy-quantize` is the native Rust control plane for resumable GGUF
conversion and quantization jobs used by Skippy workflows. It replaces the old
Python converter and external `llama-quantize` process orchestration for this
pipeline; it does not install compatibility shims or shell out to those tools.

The crate owns:

- durable conversion and quantization manifests;
- split-GGUF progress detection and next-window planning;
- native SafeTensors-to-GGUF conversion for supported checkpoint families;
- in-process GGUF quantization through the loaded llama/skippy runtime library;
- bounded source staging for quantization windows;
- optional output spooling with per-window publish and cleanup;
- successful-window records and JSON status/preflight output;
- tensor-type recipes for custom quant profiles such as `UD-Q3_K_S`;
- exact split artifact validation and optional llama load verification.

Build through the repo recipes:

```bash
just skippy-quantize-build
just skippy-quantize-release-build
```

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

`llama-api` and `skippy-abi` are quantization backends. Both call
`llama_model_quantize` in-process from a supplied native runtime library:

```bash
skippy-quantize quantize \
  --backend llama-api \
  --native-runtime-library /path/to/libllama.dylib \
  /mnt/source/BF16/model-00001-of-00002.gguf \
  /mnt/target/Q2_K/model-q2.gguf \
  Q2_K
```

Use `--backend skippy-abi` when the library is the Skippy-patched runtime used
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
  --native-runtime-library /path/to/libllama.dylib \
  --work-dir /tmp/skippy-quantize-work \
  --spool-dir /tmp/skippy-quantize-output \
  --record-dir /tmp/skippy-quantize-records
```

Run until complete:

```bash
skippy-quantize run-quant \
  --manifest /tmp/skippy-quantize.json \
  --backend llama-api \
  --native-runtime-library /path/to/libllama.dylib \
  --max-memory 32G \
  --work-dir /tmp/skippy-quantize-work \
  --spool-dir /tmp/skippy-quantize-output
```

Important quantization flags:

- `--backend {llama-api,skippy-abi}` selects the in-process quant backend.
- `--native-runtime-library PATH` loads the native runtime exposing
  `llama_model_quantize`.
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
