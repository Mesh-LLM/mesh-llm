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

### CLI output examples

Backend inspection defaults to human-readable capability summaries:

```bash
skippy-quantize backends
```

```text
✅ native-rust conversion: available
ℹ️  native-rust: resumable_windows=true low_residency_streaming=true
✅ llama-api quantization: available
ℹ️  llama-api runtime: available
⚠️  skippy-abi runtime not loaded
ℹ️  skippy-abi: model_introspection=false gguf_slice_write=false feature_mask=unknown
```

Preflight shows the job shape, backend readiness, and source/target shard
progress:

```bash
skippy-quantize quantize \
  --preflight-only \
  --backend llama-api \
  --tensor-type-file /mnt/recipe/glm-5.2-ud-q3-k-s.txt \
  /mnt/bf16/BF16/GLM-5.2-BF16-00001-of-00306.gguf \
  /mnt/quant/UD-Q3_K_S/GLM-5.2-UD-Q3_K_S.gguf \
  UD-Q3_K_S
```

```text
ℹ️  Preflight QuantizeGguf with backend llama-api
📊 target: [██░░░░░░░░░░░░░░░░░░░░░░] 28/306 shards (9.15%)
✅ Manifest is compatible
✅ Backend is ready
📊 source: [████████████████████████] 306/306 shards (100.00%)
✅ Source artifact is complete
ℹ️  Target missing ranges: 29..306
```

`status` and `next-window` are concise for operators:

```bash
skippy-quantize status --manifest /tmp/skippy-quantize.json
skippy-quantize next-window --manifest /tmp/skippy-quantize.json
```

```text
📊 job status: [██████░░░░░░░░░░░░░░░░░░] 77/306 shards (25.16%)
⚠️  Missing shards: 229
ℹ️  Missing ranges: 78..306
ℹ️  Next window: 78

ℹ️  Next window: 78
```

Conversion and quantization window runners print the selected window and the
effective command in human mode. Add `--dry-run` to direct, job, `run-*`, or
`run-*-window` commands to plan the next missing window without writing output
artifacts, creating spool/output directories, staging source shards, recording
window records, publishing shards, or running completion verification.

For long jobs, add `--json-event-file PATH` to write a compact periodic JSON
snapshot for agents to poll. The file is overwritten in place and keeps only a
bounded recent-event window, so agents do not need to ingest every log line:

```bash
skippy-quantize run-quant \
  --manifest /tmp/skippy-quantize.json \
  --backend llama-api \
  --max-memory 32G \
  --json-event-file /tmp/skippy-quantize-status.json \
  --json-event-interval-seconds 120 \
  --json-event-window 8
```

The snapshot has event type `skippy_quantize_periodic_status`, current phase,
current split window, timestamps, and the last N high-level events.

```bash
skippy-quantize run-convert-window \
  --manifest /tmp/skippy-convert.json \
  --max-memory 32G \
  --spool-dir /tmp/skippy-convert-output
```

```text
🔒 Manifest lock acquired: /tmp/skippy-convert.json.lock
🪟 convert window: 42
ℹ️  Output prefix: /tmp/skippy-convert-output/BF16/GLM-5.2-BF16.gguf
ℹ️  Command: skippy-quantize run-convert-window --backend native-rust --source /mnt/checkpoint --outfile /tmp/skippy-convert-output/BF16/GLM-5.2-BF16.gguf --first-split 42 --last-split 42 --expected-splits 306
⚠️  convert memory budget: hard cap 32.00 GiB
ℹ️  Writing native convert shard 42/306 -> /tmp/skippy-convert-output/BF16/GLM-5.2-BF16-00042-of-00306.gguf (buffer 8.00 MiB, estimated working set 16.00 MiB)
✅ Published /mnt/target/BF16/GLM-5.2-BF16-00042-of-00306.gguf (49.87 GiB)
🔓 Manifest lock released: /tmp/skippy-convert.json.lock
```

Dry-run mode stops after the same plan and memory-budget output:

```bash
skippy-quantize run-convert-window \
  --manifest /tmp/skippy-convert.json \
  --max-memory 32G \
  --spool-dir /tmp/skippy-convert-output \
  --dry-run
```

```text
🪟 convert window: 42
ℹ️  Output prefix: /tmp/skippy-convert-output/BF16/GLM-5.2-BF16.gguf
ℹ️  Command: skippy-quantize run-convert-window --backend native-rust --source /mnt/checkpoint --outfile /tmp/skippy-convert-output/BF16/GLM-5.2-BF16.gguf --first-split 42 --last-split 42 --expected-splits 306
⚠️  convert memory budget: hard cap 32.00 GiB
⚠️  convert dry run: no files were written, cleaned, recorded, or published
```

```bash
skippy-quantize run-quant-window \
  --manifest /tmp/skippy-quantize.json \
  --backend llama-api \
  --work-dir /tmp/skippy-quantize-work \
  --spool-dir /tmp/skippy-quantize-output
```

```text
🔒 Manifest lock acquired: /tmp/skippy-quantize.json.lock
📤 Copying /mnt/bf16/BF16/GLM-5.2-BF16-00001-of-00306.gguf -> /tmp/skippy-quantize-work/source-window/BF16/GLM-5.2-BF16-00001-of-00306.gguf (49.87 GiB)
ℹ️  Staged source window 1..306 at /tmp/skippy-quantize-work/source-window
🪟 quant window: 1..306
ℹ️  Staged first shard: /tmp/skippy-quantize-work/source-window/BF16/GLM-5.2-BF16-00001-of-00306.gguf
ℹ️  Output prefix: /tmp/skippy-quantize-output/UD-Q3_K_S/GLM-5.2-UD-Q3_K_S.gguf
ℹ️  Command: llama-api-quantize --tensor-type-file /mnt/recipe/glm-5.2-ud-q3-k-s.txt --keep-split /tmp/skippy-quantize-work/source-window/BF16/GLM-5.2-BF16-00001-of-00306.gguf /tmp/skippy-quantize-output/UD-Q3_K_S/GLM-5.2-UD-Q3_K_S Q3_K_S
✅ Published /mnt/quant/UD-Q3_K_S/GLM-5.2-UD-Q3_K_S-00001-of-00306.gguf (13.42 GiB)
🧹 Cleaned staged source: /tmp/skippy-quantize-work/source-window
🔓 Manifest lock released: /tmp/skippy-quantize.json.lock
```

The llama API quantization backend now uses the unpatched llama.cpp quantizer.
It can preserve split output with `--keep-split`, but it cannot process only a
partial split window. Use a quant manifest window covering all expected splits
when selecting `--backend llama-api` or `--backend skippy-abi`.

Validation commands also use the same progress-bar formatter:

```bash
skippy-quantize validate-splits \
  --root /mnt/quant \
  --prefix UD-Q3_K_S \
  --basename GLM-5.2-UD-Q3_K_S \
  --expected-splits 306
```

```text
📊 split artifact: [████████████████████████] 306/306 shards (100.00%)
✅ Split artifact is complete
```

Every workflow above can emit JSON for job automation:

```bash
skippy-quantize run-quant-window \
  --manifest /tmp/skippy-quantize.json \
  --backend llama-api \
  --json
```

```json
{
  "event": "quant_window",
  "plan": {
    "first_split": 1,
    "last_split": 306,
    "staged_first_shard": "/tmp/skippy-quantize-work/source-window/BF16/GLM-5.2-BF16-00001-of-00306.gguf",
    "output_prefix": "/mnt/quant/UD-Q3_K_S/GLM-5.2-UD-Q3_K_S.gguf",
    "command": [
      "llama-api-quantize",
      "--keep-split",
      "/tmp/skippy-quantize-work/source-window/BF16/GLM-5.2-BF16-00001-of-00306.gguf",
      "/mnt/quant/UD-Q3_K_S/GLM-5.2-UD-Q3_K_S",
      "Q3_K_S"
    ]
  }
}
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
- `--print-only` prints the planned command/report for one window without
  executing.
- `--dry-run` plans the next window without creating manifests, output/spool
  directories, records, or artifacts. Loop commands plan only the next missing
  window because no shard is written to advance progress.
- `--json-event-file PATH` writes a compact periodically refreshed status
  snapshot for agent polling.
- `--json-event-interval-seconds N` controls the refresh period, default `120`.
- `--json-event-window N` controls how many recent high-level events are kept,
  default `8`.

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
- `--max-memory SIZE` applies to native Rust conversion memory planning. The
  unpatched llama API quantization backend rejects it because llama.cpp does not
  expose a quantization memory-budget knob.
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
- `--allow-requantize`, `--pure`, and `--leave-output-tensor` mirror native
  quantization parameters.
- `--dry-run` plans the next quant window without creating manifests, output or
  spool directories, staging source shards, records, or artifacts. Loop commands
  plan only the next missing window because no shard is written to advance
  progress.
- `--json-event-file PATH` writes a compact periodically refreshed status
  snapshot for agent polling.
- `--json-event-interval-seconds N` controls the refresh period, default `120`.
- `--json-event-window N` controls how many recent high-level events are kept,
  default `8`.
- `--keep-split`, `--first-split`, and `--last-split` can request a manual
  split window for direct `quantize`; llama API quantization accepts only the
  full split range after the mesh-llm llama-quantize split-window patches were
  removed.
- `--no-stage-source` skips local source-window staging.
- `--keep-staged-source` keeps the staged source window after success.

## Recipes

Top-level quantization modes intentionally mirror the pinned llama.cpp quant
table. Custom profile names such as `Q2_K-MTP-Q8`, `UD-Q3_K_S`, or `Q4_K_XL`
belong in artifact names such as `--target-prefix` and `--output-basename`, not
in `--quant`. Pass the base llama quant with `--quant` and express any
per-tensor policy with `--tensor-type-file` or repeated `--tensor-type`.

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

## BF16 to layer package

For lab workflows, keep one reusable split BF16 GGUF artifact as the durable
source of truth, then build quantized layer packages from it. The quantized
GGUF shards can be treated as disposable staging once the package preflight
passes:

```bash
skippy-quantize quantize-layer-package \
  --source /Users/lab/glm52-work/bf16-gguf \
  --source-prefix BF16 \
  --target /Users/lab/glm52-work/quantized \
  --target-prefix Q2_K-MTP-Q8 \
  --manifest /Users/lab/glm52-work/work/q2-k-mtp-q8-package/quant-manifest.json \
  --package-dir /Users/lab/glm52-work/packages/GLM-5.2-Q2_K-MTP-Q8-layers \
  --package-model-id meshllm/GLM-5.2-Q2_K-MTP-Q8-GGUF:Q2_K-MTP-Q8 \
  --package-source-repo meshllm/GLM-5.2-Q2_K-MTP-Q8-GGUF \
  --package-source-revision local \
  --work-dir /Users/lab/glm52-work/work/q2-k-mtp-q8-package/native-work \
  --spool-dir /Users/lab/glm52-work/work/q2-k-mtp-q8-package/spool \
  --record-dir /Users/lab/glm52-work/work/q2-k-mtp-q8-package/records \
  --json-event-file /Users/lab/glm52-work/work/q2-k-mtp-q8-package/status.json \
  --quant Q2_K \
  --tensor-type-file /Users/lab/glm52-work/recipes/glm-5.2-q2-k-mtp-q8.tensor-types.txt \
  --output-basename GLM-5.2-Q2_K-MTP-Q8 \
  --stages 2 \
  --replace-package \
  --watchdog-seconds 120
```

Build prerequisites:

```bash
just skippy-quantize-standalone-release-build
cargo build --release --locked -p skippy-model-package
```

The command validates the source split, writes the package artifacts from the
BF16 GGUF source, quantizes each artifact in place, then runs package preflight.
It does not materialize a complete quantized GGUF repo first. By default, the
temporary quant scratch directory is deleted after package preflight passes;
pass `--keep-quant` to retain it. It does not pass `--max-memory` to
quantization because the unpatched llama API quant backend does not expose a
memory-budget knob.

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
