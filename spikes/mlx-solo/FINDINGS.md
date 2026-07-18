# Spike: goose-style solo MLX serving — findings

Validates the Phase-2 claim in `docs/design/MLX_STAGE_ENGINE_PLAN.md`: load an HF
**safetensors** model in Rust via `safemlx-lm`, optionally **JIT-quantize on
load**, and generate tokens on **Metal** — with no GGUF and no ahead-of-time
quant step.

Throwaway crate with its own `[workspace]`, deliberately **not** part of the
mesh-llm cargo workspace, so MLX/CMake native deps never touch other builds. It
mirrors goose's MLX setup (`../goose/crates/goose-local-inference`) exactly:
`safemlx` with `["accelerate", "metal", "safetensors"]`, `Device::Gpu` for
compute + `Device::Cpu` for weight staging, plain `LoadedModel::load`.

## TL;DR

- **Solo serving from raw safetensors works on Metal, in Rust, end to end.** ✅
  Qwen3-0.6B, source precision (bf16): **321 tok/s** decode, coherent.
- **JIT quant on load is free at inference time.** JIT 4-bit and a pre-quantized
  mlx-community 4-bit repo both run at **~604 tok/s** — identical. So "download
  safetensors → quantize on load → serve" costs nothing versus shipping a
  pre-quantized artifact. This is the workflow win, confirmed.
- **The goose baseline needs ZERO fork patches.** Source-precision serving runs
  on a pristine `jbg/safemlx` checkout.
- **Going beyond the goose baseline hit two genuine bugs in the young fork**
  (JIT quant + loading arbitrary mlx-community repos). Both are small and fixed
  locally; they are **upstream-PR candidates for `jbg/safemlx`, not mesh-llm
  drift** — and they confirm the plan's "safemlx-lm is young, expect per-family
  papercuts" risk.

## Results (Apple Silicon, Metal)

Model: `Qwen/Qwen3-0.6B` (dense safetensors), and `mlx-community/Qwen3-0.6B-4bit`
(pre-quantized). `-n 128`, greedy.

| Mode | model source | load | ttft | decode tok/s | patches needed |
| --- | --- | --- | --- | --- | --- |
| source precision (bf16) | dense safetensors | 1.36s | 0.876s | **321** | none |
| pre-quantized 4-bit (goose's path) | mlx-community repo | 0.15s | 0.158s | **603** | mode-default fix* |
| **JIT 4-bit on load** | dense safetensors | 0.27s | 0.010s | **604** | lm_head fix* |

*The pre-quantized and JIT rows each needed one small fork fix (see below); the
source-precision row — the actual goose baseline — needed none.

Key reading: **JIT (604) ≈ pre-quantized (603)**. Quantizing on load is not a
runtime tax; it happens during the (already fast) load. So the "no wait for a
published GGUF/quant, just serve the safetensors" story holds with no inference
penalty on Metal.

> Historical note: an earlier CPU-only run (no Metal compiler installed) showed
> 18 tok/s source / 0.4 tok/s quantized. That 0.4 was a pure CPU-kernel artifact
> (MLX quant matmul is Metal-optimized). It is not representative and has been
> superseded by the Metal numbers above. CPU is not a serving path.

## Environment

| | |
| --- | --- |
| Machine | Apple Silicon (arm64), macOS 26.5 |
| Toolchain | rustc 1.97, cmake 4.4, Apple clang 21 |
| Metal | Xcode 26.6 (already installed) + `MetalToolchain` component (688 MB, pulled via `xcodebuild -downloadComponent MetalToolchain`); `metal` 32023.883. Point builds at it with `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` — no sudo, no re-install. |
| MLX backend | `metal` + `accelerate`; MLX core `v0.32.0` via safemlx-sys FetchContent |
| safemlx fork | `jbg/safemlx` @ `0502a19`; pristine for source precision, +2 tiny fixes for quant coverage |
| Models | `Qwen/Qwen3-0.6B`, `mlx-community/Qwen3-0.6B-4bit` |

## Build notes

- **MLX C++ builds cleanly with the Metal backend under cmake 4.4.** Cold build
  (compiles MLX C++ + metallib shaders) ~2m05s; warm rebuild after a Rust-only
  change ~6.7s (MLX archives cached in the target dir).
- Only harmless linker warnings (`object file … has version 26.5.x, which is
  newer than target minimum of 11.0.0`).
- The build needs `DEVELOPER_DIR` pointed at full Xcode so `safemlx-sys`'s
  `xcrun -find metal` resolves the real compiler (CommandLineTools alone is not
  enough; its `metal` is a stub that errors at runtime until the toolchain
  component is installed).

## Issues found — both upstream-PR candidates, not drift

The goose baseline (source-precision serving) needs no changes. The two fixes
below are only needed to go *beyond* goose's usage — JIT quant and loading
arbitrary published quantized repos — which is in-scope because that is the
"serve any safetensors model, quantized on load" workflow the plan proposes.
They are small, isolated, and should be PR'd to `jbg/safemlx`.

### 1. Tied-embedding `lm_head.weight` fails the *quantized* strict loader (qwen3)

`Qwen3-0.6B` sets `tie_word_embeddings: true` but the checkpoint still ships a
redundant `lm_head.weight`. The dense loader is lenient and tolerates it; the
quantized loader uses a bare `StrictLoadConfig::default()` and rejects it:

```
strict weight-load validation failed: 0 missing, 1 unused
  unused: lm_head.weight
```

Fix (`safemlx-lm/src/models/qwen3.rs`, `load_qwen3_model_quantized`):
```rust
let config = StrictLoadConfig::default().allow_unused_prefix("lm_head.");
```
Harmless when untied (then `lm_head.weight` is a loaded param, not unused).

### 2. `WeightQuantization` requires a `mode` field many mlx-community repos omit

`mlx-community/Qwen3-0.6B-4bit`'s `config.json` has
`"quantization": {"group_size": 64, "bits": 4}` with no `mode`. The fork's
`WeightQuantizationMetadata` makes `mode` mandatory, so the load fails with
`missing field 'mode'`. mlx-lm itself defaults a missing mode to `affine`.

Fix (`safemlx-lm/src/quantization.rs`):
```rust
#[serde(default = "default_affine_mode_string")]
mode: String,
// ...
fn default_affine_mode_string() -> String { "affine".to_string() }
```

## Reproduce

1. Install the Metal toolchain if `xcrun metal --version` errors:
   `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
    /Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild -downloadComponent MetalToolchain`
2. Sibling checkout of the fork at `../safemlx`, base `0502a19`, with the two
   fixes above (only needed for the quant rows).
3. Model dirs (HF safetensors) for `Qwen/Qwen3-0.6B` and, optionally,
   `mlx-community/Qwen3-0.6B-4bit`.
4. Build/run with Xcode selected for this shell:
   ```
   export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
   cd spikes/mlx-solo
   cargo build --release
   ./target/release/mlx-solo --model <DENSE_DIR>       -n 128 "..."   # source precision
   ./target/release/mlx-solo --model <PREQUANT_DIR>    -n 128 "..."   # pre-quantized
   ./target/release/mlx-solo --model <DENSE_DIR> -q 4  -n 128 "..."   # JIT 4-bit
   ```
   A shared `CARGO_TARGET_DIR` avoids rebuilding MLX across crates.

## What this de-risks / what it does not

De-risked:
- safemlx-lm as a Rust solo engine on **Metal** (321 tok/s bf16, ~604 tok/s 4-bit).
- The "no GGUF, no pre-quant, serve raw safetensors" workflow — including that
  **JIT quant is free at inference time** (≈ pre-quantized).
- MLX native build with the Metal backend under this toolchain (cmake 4.4).

Still open:
- Larger models + throughput vs the llama.cpp backend on the same hardware.
- Product staged/split integration. A follow-on SmolLM2 proof now range-
  materializes two partial SafeTensors files and matches whole-model MLX through
  Skippy's F16/F32 binary activation codec; see
  `../mlx-safetensors-stages/FINDINGS.md`. Two `skippy-server` processes, the
  boundary-fence benchmark, and bounded-memory load-time quantization remain.
- The omitted-mode compatibility fix is proposed in `jbg/safemlx#2`; the
  tied-`lm_head.weight` quantized-loader fix still needs upstream resolution.
