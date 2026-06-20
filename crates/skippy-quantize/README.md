# skippy-quantize

`skippy-quantize` is the Rust replacement surface for the current Python
conversion and `llama-quantize` orchestration path.

This crate is intentionally split into a resumable CLI/control plane and native
execution backends. The first implementation owns:

- durable conversion/quantization manifests;
- split-GGUF progress detection;
- resume window planning;
- resumable `convert_hf_to_gguf.py` window execution;
- resumable `llama-quantize --keep-split` window execution;
- bounded source staging for quantization windows;
- optional local output spooling with per-window publish and cleanup;
- Linux `posix_fadvise(DONTNEED)` cache drops while staging source shards;
- successful-window staged source cleanup;
- optional durable per-window run records;
- optional backend watchdog logs with elapsed time and cgroup memory;
- exact split artifact validation;
- supported quant and raw tensor type validation;
- broad `convert_hf_to_gguf.py` option passthrough for split planning,
  tokenizer/template controls, MTP/MMProj export, MoE fusion, FP8 handling,
  metadata overrides, and lazy/temp-file memory modes;
- broad `llama-quantize` option passthrough for imatrix, tensor overrides,
  pruning, metadata overrides, pure mode, and thread count;
- JSON status output suitable for Hugging Face Jobs.

The execution surface owns the resumable workflow in Rust and can run through
multiple backends. `external-process` shells out to the patched llama.cpp
converter/quantizer, `native-rust` streams SafeTensors-to-GGUF conversion
directly in Rust for supported architectures, and `llama-api` / `skippy-abi`
call the linked llama.cpp quantization API in-process behind the same
manifest/window surface.

Build the local binary through the repo `just` recipes:

```bash
just skippy-quantize-build
just skippy-quantize-release-build
```

Top-level quantization modes intentionally mirror the pinned
`llama-quantize` CLI table. Raw tensor-type recipe validation mirrors the
current GGML tensor type enum, so recipe files may use raw types such as
`MXFP4` or `NVFP4` even when those names are not exposed as whole-model
`llama-quantize` modes in the current pin.
Custom profile labels such as `UD-Q3_K_S` and `Q4_K_XL` are accepted as recipe
aliases when paired with `--tensor-type-file`. They resolve to the corresponding
base llama quant (`Q3_K_S` or `Q4_K_M`) for backend execution while preserving
the recipe label in default output and sidecar names. `list-quants --json`
reports these known recipe labels separately from direct llama modes.

## Current commands

Inspect backend capabilities:

```bash
skippy-quantize backends --json
```

Probe a specific Skippy native runtime library for ABI features:

```bash
skippy-quantize backends \
  --skippy-runtime-library /path/to/libllama.dylib \
  --json
```

Inspect supported whole-model quant modes and raw tensor override types:

```bash
skippy-quantize list-quants --json
skippy-quantize list-tensor-types --json
```

At the moment, the Skippy ABI supports staged inference/runtime operations but
does not expose HF checkpoint conversion entry points. Conversion therefore uses
either the external-process backend with patched llama.cpp tools or the native
Rust Safetensors-to-GGUF writer. GGUF quantization may use
`--backend external-process`, `--backend llama-api`, or `--backend skippy-abi`.
The `skippy-abi` quant path loads a supplied native runtime library and calls
the linked `llama_model_quantize` symbols through the same in-process path as
`llama-api`; it is not used for checkpoint conversion.
`backends --skippy-runtime-library PATH` can probe a supplied native runtime and
will report lower-level ABI features such as model introspection and GGUF slice
writing, but those are not enough by themselves to replace HF checkpoint
conversion or `llama-quantize`.

Quantization also has experimental in-process backends that call the native
`llama_model_quantize` API from supplied native runtime libraries:

```bash
skippy-quantize quantize \
  --backend llama-api \
  --native-runtime-library /path/to/libllama.dylib \
  /mnt/source/BF16/model-00001-of-00002.gguf \
  /mnt/target/Q2_K/model-q2.gguf \
  Q2_K
```

Use `--backend skippy-abi` with the same `--native-runtime-library` flag when
the library is the Skippy-patched runtime used by mesh-llm.

This path uses the same manifest, staging, spooling, publishing, and resume
logic as the external-process backend. It currently supports whole-model quant
mode, split windows, copy/dry-run/pure/requantize/output-tensor controls,
thread count, tensor override recipes, layer pruning, KV overrides, and legacy
`.dat` and GGUF imatrix files with include/exclude filters.

Native Rust Safetensors-to-GGUF conversion currently requires tokenizer metadata
from `tokenizer.json`. Checkpoints that only provide SentencePiece
`tokenizer.model` are rejected with a clear error; use the external
`convert_hf_to_gguf.py` passthrough/backend for those until native
SentencePiece support is implemented.
Native conversion rejects Python-converter-only flags such as `--remote`,
`--vocab-only`, `--mmproj`, `--metadata`, `--mistral-format`, and templating
instead of silently ignoring them. Native `--no-mtp` is supported for trunk
conversion by dropping appended MTP layers declared in `config.json`; the
emitted metadata omits `nextn_predict_layers` from the trunk block count.
Native `--mtp` is supported for draft-head conversion by writing shared
embedding/norm/head tensors plus the appended MTP layer tensors, including
known NextN tensor names such as `eh_proj.weight`, `enorm.weight`,
`hnorm.weight`, and `shared_head.norm.weight`. Qwen-style `mtp.*` tensors are
remapped to the same layer-indexed NextN names using the MTP layer boundary
from `config.json`.
Native conversion also rejects newer Qwen variants that have separate upstream
GGUF architectures, including Qwen3.5, Qwen3Next, and Qwen3VL/MoE. Those need
architecture-specific metadata, tensor mapping, and tokenizer handling before
the native Rust path can safely replace the upstream converter.

For `external-process`, backend tools are resolved in this order:

1. explicit `--converter` / `--llama-quantize`;
2. `SKIPPY_QUANTIZE_CONVERTER` / `SKIPPY_QUANTIZE_LLAMA_QUANTIZE`;
3. repo-local `.deps/llama.cpp` defaults when present.

Install compatibility shims when replacing existing scripts in a job image or
PATH. The installed names dispatch through the same direct replacement paths as
the explicit `convert-hf-to-gguf` and `llama-quantize` subcommands:

```bash
skippy-quantize install-shims \
  --dir /opt/skippy-quantize/bin \
  --binary /opt/skippy-quantize/skippy-quantize
```

The command installs `llama-quantize`, `convert_hf_to_gguf.py`,
`convert-hf-to-gguf`, `hf_to_gguf.py`, `hf_to_gguff.py`,
`skippy-quantize-llama-quantize`, and
`skippy-quantize-convert-hf-to-gguf`. Existing paths are left untouched unless
`--force` is passed.

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
  --converter /work/llama.cpp/convert_hf_to_gguf.py \
  --split-max-size 50G \
  --dry-run
```

Run conversion windows until the artifact is complete:

```bash
skippy-quantize run-convert \
  --manifest /tmp/skippy-convert.json \
  --converter /work/llama.cpp/convert_hf_to_gguf.py \
  --split-max-size 50G \
  --split-max-tensors 0 \
  --use-temp-file \
  --no-tensor-first-split \
  --model-name GLM-5.2 \
  --metadata /mnt/recipe/metadata.json \
  --no-mtp \
  --fuse-gate-up-exps \
  --fp8-as-q8 \
  --spool-dir /tmp/skippy-convert-output \
  --watchdog-seconds 120 \
  --record-dir /tmp/skippy-convert-records
```

For a more direct `convert_hf_to_gguf.py` replacement surface, pass the source
checkpoint and desired output path. The command derives the target prefix,
output basename, and durable manifest path, then runs the same resumable
low-memory conversion loop. Direct conversion defaults to a single GGUF output;
set `--expected-splits N` when intentionally producing split GGUF shards:

```bash
skippy-quantize convert \
  --converter /work/llama.cpp/convert_hf_to_gguf.py \
  --split-max-size 50G \
  --spool-dir /tmp/skippy-convert-output \
  --output-type bf16 \
  /mnt/checkpoint \
  /mnt/target/BF16/GLM-5.2-BF16.gguf
```

The same direct conversion path is also available under the compatibility
subcommand `convert-hf-to-gguf`. If the binary is invoked through a symlink or
renamed executable called `convert_hf_to_gguf.py`, `convert-hf-to-gguf`, or
`skippy-quantize-convert-hf-to-gguf`, it dispatches to this path directly.
For closer `convert_hf_to_gguf.py` compatibility, the output may be supplied as
`--outfile` instead of positional `OUTPUT`, and `--outtype` aliases
`--output-type`.
If neither `OUTPUT` nor `--outfile` is supplied for a local checkpoint, the
command derives a deterministic resumable output path under the source
directory, using `<source>/<source>-<outtype>.gguf`. With `--outtype auto`, the
local resumable path resolves the concrete output type from safetensor headers
before deriving the output and sidecar manifest paths, matching upstream's
local checkpoint behavior. Python-only shapes such as `--remote` and
`--print-supported-models` still use passthrough mode so llama.cpp owns their
exact behavior.
Templated output names such as `model-{ftype}.gguf`, `model-{outtype}.gguf`,
or `model-{}.gguf` also use passthrough mode so llama.cpp fills the template
after resolving `--outtype auto`.
The upstream shard-selection flags `--skip-output-shards-before` and
`--stop-output-shards-after` also use passthrough mode so llama.cpp's exact
metadata-planning behavior is preserved. For resumable conversion jobs, prefer
`--window-size`, `--max-windows`, and the manifest-based commands.

For HF Jobs, prefer the idempotent one-shot form. It creates the manifest on
the first run, verifies that an existing manifest matches on later runs, then
resumes from the durable target shards:

```bash
skippy-quantize convert-job \
  --source /mnt/checkpoint \
  --target /mnt/target \
  --target-prefix BF16 \
  --output-basename GLM-5.2-BF16 \
  --output-type bf16 \
  --expected-splits 306 \
  --window-size 1 \
  --manifest /tmp/skippy-convert.json \
  --converter /work/llama.cpp/convert_hf_to_gguf.py \
  --split-max-size 50G \
  --spool-dir /tmp/skippy-convert-output \
  --watchdog-seconds 120
```

Add `--preflight-only --json` to either one-shot job command to validate the
manifest shape, backend path, source/target shard state, and next window without
writing the manifest or launching the backend.
The direct `convert` and `quantize` commands support the same preflight flags
after deriving their sidecar manifest paths from the input/output paths.

One-shot jobs verify the final shard set automatically once all expected shards
exist. Use `--no-verify-on-complete` only when an operator intentionally wants
to skip that completion check.

All run paths hold a non-blocking manifest sidecar lock (`<manifest>.lock`)
while they create or resume the manifest, inspect progress, execute a backend
window, publish spooled shards, and clean temporary state. A second process
targeting the same manifest fails early instead of racing on the same shard
window.

For a more direct `llama-quantize` replacement surface, pass the first split
input shard and the desired output path. The command derives the split source,
target prefix, output basename, durable manifest path, and split count, then
runs the same resumable low-memory quantization loop:

```bash
skippy-quantize quantize \
  --llama-quantize /work/llama.cpp/build/bin/llama-quantize \
  --work-dir /tmp/skippy-quant-work \
  --spool-dir /tmp/skippy-quant-output \
  /mnt/source/BF16/GLM-5.2-BF16-00001-of-00306.gguf \
  /mnt/target/Q2_K/GLM-5.2-Q2_K.gguf \
  Q2_K \
  8
```

The same direct quantization path is also available under the compatibility
subcommand `llama-quantize`. If the binary is invoked through a symlink or
renamed executable called `llama-quantize` or `skippy-quantize-llama-quantize`,
it dispatches to this path directly.

For closer `llama-quantize` compatibility, the output path may be omitted just
like upstream. In that form the direct command derives the upstream-style
`ggml-model-<type>.gguf` output path in the input shard directory and still uses
the resumable manifest path.
Quantization type arguments accept both names such as `Q4_K` and llama.cpp's
numeric ftype ids such as `15`.
The upstream split-control flags `--keep-split`, `--first-split`, and
`--last-split` preserve llama.cpp semantics. With `--backend external-process`
they use passthrough mode so llama.cpp owns exact argument handling. With
`--backend llama-api` or `--backend skippy-abi`, the same flags select a
requested Rust-managed resumable window: missing `--first-split` defaults to
split `1`, missing `--last-split` defaults to the final source split, and
already completed shards inside the requested range are skipped on resume. For
normal Skippy/HF jobs, prefer `--window-size`, `--max-windows`, and the
manifest-based commands instead.

```bash
skippy-quantize llama-quantize \
  --llama-quantize /work/llama.cpp/build/bin/llama-quantize \
  /mnt/source/BF16/GLM-5.2-BF16-00001-of-00306.gguf \
  Q2_K \
  8
```

Direct `convert` and `quantize` accept `--max-windows N` for bounded recovery
passes or smoke tests.

Create a quantization manifest from a complete split GGUF source:

```bash
skippy-quantize init-quant \
  --source /mnt/source-gguf \
  --source-prefix BF16 \
  --target /mnt/target-quant \
  --target-prefix Q2_K \
  --output-basename GLM-5.2-Q2_K \
  --quant Q2_K \
  --tensor-type-file /mnt/recipe/tensor-types.txt \
  --window-size 1 \
  --manifest /tmp/skippy-quant.json
```

Run the next missing quantization window:

```bash
skippy-quantize run-quant-window \
  --manifest /tmp/skippy-quant.json \
  --llama-quantize /work/llama.cpp/build/bin/llama-quantize \
  --work-dir /tmp/skippy-quant-work \
  --dry-run \
  --record-dir /tmp/skippy-quant-records
```

Run quantization windows until the artifact is complete:

```bash
skippy-quantize run-quant \
  --manifest /tmp/skippy-quant.json \
  --llama-quantize /work/llama.cpp/build/bin/llama-quantize \
  --work-dir /tmp/skippy-quant-work \
  --spool-dir /tmp/skippy-quant-output \
  --max-memory 28G \
  --memory-policy hard \
  --imatrix /mnt/recipe/imatrix.dat \
  --include-weights blk. \
  --output-tensor-type Q8_0 \
  --token-embedding-type F16 \
  --tensor-type mtp_head.weight=NVFP4 \
  --override-kv general.name=str:custom-quant \
  --nthreads 8 \
  --watchdog-seconds 120 \
  --record-dir /tmp/skippy-quant-records
```

The one-shot quant form follows the same manifest create-or-resume behavior:

```bash
skippy-quantize quant-job \
  --source /mnt/source-gguf \
  --source-prefix BF16 \
  --target /mnt/target-quant \
  --target-prefix Q2_K \
  --output-basename GLM-5.2-Q2_K \
  --quant Q2_K \
  --tensor-type-file /mnt/recipe/tensor-types.txt \
  --window-size 1 \
  --manifest /tmp/skippy-quant.json \
  --llama-quantize /work/llama.cpp/build/bin/llama-quantize \
  --work-dir /tmp/skippy-quant-work \
  --spool-dir /tmp/skippy-quant-output \
  --watchdog-seconds 120
```

Use `--max-windows N` on `run-convert` or `run-quant` for bounded recovery
passes, smoke tests, or intentionally chunked HF Jobs.

`run-convert` passes through the patched converter controls used by the current
Jianyang workflow, including `--vocab-only`, `--bigendian`, `--no-lazy`,
`--model-name`, `--split-max-tensors`, `--no-tensor-first-split`, `--metadata`,
`--print-supported-models`, `--remote`, `--mmproj`, `--mtp`, `--no-mtp`,
`--mistral-format`, `--disable-mistral-community-chat-template`,
`--sentence-transformers-dense-modules`, `--fuse-gate-up-exps`, `--fp8-as-q8`,
and `--target-model-dir`.

Use `status` and `next-window` against either manifest to inspect resumability.
Use `--print-only` on `run-convert-window` or `run-quant-window` to validate the
planned command without executing the backend.

Verify the exact artifact described by a manifest after a job completes:

```bash
skippy-quantize verify-job --manifest /tmp/skippy-quant.json --json
```

Validate a tensor recipe before launching an expensive job:

```bash
skippy-quantize validate-tensor-types /mnt/recipe/tensor-types.txt --json
```

The tensor recipe format matches `llama-quantize --tensor-type-file`: each
entry is `tensor_name=ggml_type`, separated by whitespace or newlines.
Recipe values must be raw GGML tensor types; whole-model mixture labels such as
`IQ2_M`, `IQ3_XS`, `IQ3_M`, `Q3_K_S`, or `Q4_K_M` are rejected for tensor
overrides.

`run-quant` also passes through the advanced `llama-quantize` controls needed
for custom recipes: `--pure`, `--imatrix`, repeated `--include-weights` or
`--exclude-weights`, `--output-tensor-type`, `--token-embedding-type`,
repeated `--tensor-type`, `--prune-layers`, repeated `--override-kv`, and
`--nthreads`. Inline tensor type arguments are validated with the same raw
GGML tensor-type parser as tensor recipe files.

`run-quant-window` stages only the selected shard window as real local files and
symlinks the other source shards. On Linux, staged-source reads and writes are
advised out of the OS page cache as they are copied. After a successful real
quant window, the staged source directory is deleted unless
`--keep-staged-source` is set.

When `--spool-dir` is set, backend output is written under that local spool
root first. After the backend exits successfully, only the completed split
window is copied into the manifest target with bounded IO, the target shard is
atomically renamed from a `.part` file, and the local spool shard is deleted
unless `--keep-spool` is set. Progress and resume checks always scan the
manifest target, not the spool. Before a backend is launched, stale spool files
for the exact current window are removed so a failed previous attempt cannot be
published as fresh output.

Use `--watchdog-seconds N` on long-running windows to emit
`backend_watchdog` lines while the backend process is still running. On Linux
with cgroup v2, those lines include `memory.current` and `memory.peak` values;
on other systems the memory fields are reported as `unknown`.

Use `--max-memory SIZE` on conversion and quantization runners to make memory a
planned scheduling constraint instead of a best-effort log. Native conversion
derives a smaller stream buffer from the budget and rejects estimated working
sets that exceed `--max-memory` under `--memory-policy hard`. Quantization
passes `LLAMA_QUANTIZE_MAX_MEMORY_BYTES` to llama.cpp when available and the
Rust process monitor enforces the same budget around external backends.
`--memory-policy hard` kills the backend when observed memory exceeds the
budget so the manifest can resume on a smaller window or cheaper retry;
`--memory-policy advisory` only logs the budget breach. Sizes accept raw bytes
or binary suffixes such as `M`, `MiB`, `G`, and `GiB`.

Validate a completed split artifact:

```bash
skippy-quantize validate-splits \
  --root /mnt/target-quant \
  --prefix Q2_K \
  --basename GLM-5.2-Q2_K \
  --expected-splits 306 \
  --json
```
