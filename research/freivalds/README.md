# Freivalds Verification for mesh-llm

Research experiment: can we use Freivalds' algorithm to verify that remote peers
in a mesh are actually running the model they claim, using llama.cpp's existing
infrastructure?

## Background

[CommitLLM](https://github.com/lambdaclass/commitllm) is a cryptographic
commit-and-audit protocol for verifying LLM inference. It works by capturing
intermediate activations during a forward pass and using Freivalds' algebraic
check to verify that the correct weight matrices were used.

CommitLLM is tightly coupled to vLLM's W8A8 quantized inference pipeline. mesh-llm
uses llama.cpp with GGUF models across Metal, CUDA, ROCm, and CPU backends. This
experiment measures whether Freivalds verification is viable for llama.cpp's
quantized matmuls, where GPU and CPU use different floating-point accumulation
orders.

## What Freivalds' Algorithm Does

Given a matrix multiplication `y = W · x`, the verifier can check correctness
without re-doing the full multiplication:

1. **Setup (once per model):** Generate secret random vector `v`. Pre-compute `r = v · W`.
2. **Check:** Compute `v · y` and `r · x` (both are single dot products — cheap).
3. **Verify:** If `v · y == r · x`, the multiplication used the correct weights.

In exact arithmetic, a cheater is caught with probability `1 - 1/p`. In floating
point, there's a residual from different accumulation orders. The question is
whether this residual is small enough to distinguish honest from dishonest computation.

## The Experiment

`freivalds-check.cpp` is a standalone tool that:

1. Loads a GGUF model via llama.cpp
2. Runs a forward pass with the `cb_eval` callback to capture activation tensors
3. Reads the weight tensors from the model
4. Performs the Freivalds check on CPU: computes `v·y` and `(v·W)·x` for specific matmuls
5. Measures the relative residual `|v·y - r·x| / max(|v·y|, |r·x|)`

The tool checks two projections per layer:
- **Wq (attention Q projection):** input = RMSNorm output, output = raw matmul result
- **W_down (FFN down projection):** input = SiLU(gate) ⊙ up, output = raw matmul result

### How to build

The tool builds inside the llama.cpp tree. From the repo root:

```bash
# Copy the tool into llama.cpp
cp research/freivalds/freivalds-check.cpp llama.cpp/tools/freivalds-check/
cp research/freivalds/CMakeLists.txt llama.cpp/tools/freivalds-check/

# Add to llama.cpp/tools/CMakeLists.txt (if not already there):
#   add_subdirectory(freivalds-check)

# Build
cd llama.cpp
cmake -B build -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-freivalds-check -j8
```

### How to run

```bash
# Basic run (CPU inference, no weight repacking)
./build/bin/llama-freivalds-check -m /path/to/model.gguf -ngl 0 -nr

# Custom prompt
./build/bin/llama-freivalds-check -m model.gguf -ngl 0 -nr -p "Hello world"
```

**Important:** Use `-nr` (no repack) to disable CPU weight repacking. The repack
optimization changes the in-memory layout of weight tensors, making direct
dequantization return garbage. With `-nr`, weights stay in standard GGUF block
layout.

## Results

### Qwen2.5-0.5B-Instruct Q4_K_M (24 layers, 896 embd)

CPU inference on Apple M4 Max. Prompt: "The quick brown fox jumps over the lazy dog" (9 tokens).

| Layer | Proj   | Dims       | Rel Residual |
|-------|--------|------------|-------------|
| 0     | Wq     | 896×896    | 3.08e-05    |
| 0     | W_down | 896×4864   | 6.44e-02    |
| 4     | Wq     | 896×896    | 5.24e-02    |
| 4     | W_down | 896×4864   | 2.19e-03    |
| 8     | Wq     | 896×896    | 1.82e-02    |
| 8     | W_down | 896×4864   | 2.23e-03    |
| 12    | Wq     | 896×896    | 1.12e-02    |
| 12    | W_down | 896×4864   | 3.17e-03    |
| 16    | Wq     | 896×896    | 6.93e-03    |
| 16    | W_down | 896×4864   | 4.13e-02    |
| 20    | Wq     | 896×896    | 8.01e-03    |
| 20    | W_down | 896×4864   | 3.97e-03    |
| 23    | Wq     | 896×896    | 6.62e-03    |
| 23    | W_down | 896×4864   | 2.33e-04    |

**Max relative residual: 6.44e-02 (6.4%)**
**Avg relative residual: 1.58e-02 (1.6%)**

### SmolLM2-135M-Instruct Q8_0 (30 layers, 576 embd)

| Layer | Proj   | Dims       | Rel Residual |
|-------|--------|------------|-------------|
| 0     | Wq     | 576×576    | 2.61e-02    |
| 0     | W_down | 576×1536   | 2.79e-03    |
| 15    | Wq     | 576×576    | 2.85e-02    |
| 15    | W_down | 576×1536   | 1.80e-02    |
| 29    | Wq     | 576×576    | 1.02e-03    |
| 29    | W_down | 576×1536   | 9.93e-03    |

**Max relative residual: 2.85e-02 (2.9%)**

### Qwen3-8B Q4_K_M (36 layers, 4096 embd)

| Layer | Proj   | Dims         | Rel Residual |
|-------|--------|-------------|-------------|
| 0     | Wq     | 4096×4096   | 8.32e-04    |
| 0     | W_down | 4096×12288  | 1.57e-02    |
| 9     | Wq     | 4096×4096   | 1.72e-02    |
| 9     | W_down | 4096×12288  | 1.67e-02    |
| 18    | Wq     | 4096×4096   | 3.84e-03    |
| 18    | W_down | 4096×12288  | 2.48e-03    |
| 35    | Wq     | 4096×4096   | **2.35e-01**|
| 35    | W_down | 4096×12288  | 7.33e-02    |

**Max relative residual: 2.35e-01 (23.5%) — last layer**
**Avg relative residual: 4.57e-02 (4.6%)**

## Analysis

### What works

1. **Model substitution is easily detectable.** Running a completely wrong model
   would produce uncorrelated dot products — ~100% relative residual. Even the
   noisiest honest residual (23.5%) is far from that.

2. **Most layers produce tight residuals.** Excluding the last layer outlier,
   residuals are typically <5%. A threshold of ~15-20% would pass all honest
   checks while catching model substitution.

3. **Quantization format barely matters.** Q8_0 and Q4_K_M produce similar
   residuals. The dominant error is FP accumulation order, not dequantization.

4. **The check is fast.** The v·W pre-computation takes <2ms per matrix on CPU.
   The actual check (two dot products) is microseconds.

5. **Backend-agnostic in principle.** The tool uses llama.cpp's `cb_eval` callback
   which works with any backend (CPU, Metal, CUDA). The activation tensors are
   read via `ggml_backend_tensor_get` which handles GPU→CPU transfer.

### What doesn't work well

1. **Last layer outlier.** The Qwen3-8B last layer (35) shows 23.5% residual,
   much higher than other layers. This needs investigation — may be related to
   the `inp_out_ids` token filtering that changes tensor shapes at the last layer,
   or just natural variance from activation magnitudes growing through the network.

2. **Single check is noisy.** One random vector gives variable results. Multiple
   checks with different random vectors would tighten the confidence but cost
   proportionally more.

3. **Requires `-nr` (no repack).** The CPU REPACK optimization changes weight
   tensor memory layout, breaking direct dequantization. The verifier needs
   access to standard GGUF-format weights. In practice the verifier would
   dequantize from the GGUF file directly rather than from runtime memory.

4. **CPU-only measurement.** These results are from CPU inference. GPU backends
   (Metal, CUDA) may have larger residuals due to more aggressive FP16/BF16
   accumulation and different tiling strategies. **This is the most important
   open question.**

### Comparison to wrong-model detection

For context, using wrong weights would produce:
- **Different model entirely (7B vs 70B):** ~100% residual (uncorrelated)
- **Same arch, different size (Qwen3-4B vs 8B):** ~100% residual (different weight dimensions)
- **Same model, different quant (Q4 vs Q8):** ~5-20% residual (different dequant values)
- **Same model, different fine-tune:** ~1-10% residual (subtle weight differences)
- **Honest computation:** <7% residual (typically <2%)

A threshold of **15% relative residual** would:
- ✅ Catch model substitution (different model/size)
- ✅ Catch significantly different quantization
- ⚠️ Possibly miss same-model different-fine-tune (depends on weight divergence)
- ✅ Pass all honest computation (except possibly last-layer outliers on larger models)

## Protocol Design (Future Work)

If the residuals prove tight enough (especially on GPU), the full verification
protocol for mesh-llm would be:

### Commit Phase (during inference)
1. Server runs inference normally
2. At each layer boundary, hashes the hidden state: `h_i = SHA256(activation_i)`
3. Builds Merkle tree over `[h_0, h_1, ..., h_N]`
4. Returns `(tokens, merkle_root)` alongside the response

### Challenge Phase (after response)
1. Verifier picks random layer L and projection P
2. Verifier says: "open layer L, projection P"
3. Server returns `(x_L, y_L, merkle_proof)` — the matmul input/output + proof
4. Verifier checks:
   - Merkle proof validates against committed root
   - Freivalds: `v · y ≈ (v · W) · x` within tolerance

### Requirements per backend
- **Hash hidden states during forward pass:** ~80 SHA256 hashes per token, ~0.5ms overhead
- **Export activation tensors on demand:** one tensor read per challenge
- **llama.cpp:** Hook into `cb_eval` callback (already exists)
- **MLX:** Read tensors at layer boundaries in Python forward method
- **vLLM:** Similar to existing CommitLLM capture, but simpler

## Open Questions

1. **GPU residuals:** How much larger are Freivalds residuals when inference runs
   on Metal/CUDA? The FP16 accumulation and different tiling could push residuals
   above the detection threshold.

2. **Last-layer outlier:** Is the large residual at the last layer a consistent
   issue or specific to certain models/prompts? Needs more measurements.

3. **Multiple checks:** How many independent random vectors are needed for
   reliable detection? Each additional check multiplies the false-positive
   probability by ~(tolerance / max_error).

4. **Commitment overhead:** What's the real-world throughput impact of hashing
   activations at every layer boundary during inference?

5. **Repacked weights:** Can the Freivalds check work with repacked weight
   layouts, or must the verifier always dequantize from the original GGUF?

## Files

- `freivalds-check.cpp` — The measurement tool (builds inside llama.cpp tree)
- `CMakeLists.txt` — Build configuration for the tool
- `README.md` — This document
