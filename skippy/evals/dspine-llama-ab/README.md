# DSpine vs DFlash on llama.cpp

A/B benchmark for the DSpine port in the llama.cpp patch queue
(`skippy/llama_cpp/patches/model_support/0015-models-add-dspine-block-parallel-drafter-support.patch`),
run through the jianyang lab fleet (`jy submit`) on a DGX Spark.

Three arms share one target (`Qwen/Qwen3-8B`, BF16 GGUF from unsloth), one prompt
set (77 report-split prompts from `skippy/evals/spec-prompts`), greedy decoding,
256 tokens per prompt, concurrency 1:

- `off` — no speculation
- `dflash` — `z-lab/Qwen3-8B-DFlash-b16` (plain DFlash, block 16), converted to GGUF
- `dspine` — `tencent/Qwen3-8B-DSpine-b16` (DSpine, block 16), converted to GGUF

Both draft GGUFs are converted with the patched llama.cpp tree's
`convert_hf_to_gguf.py` (the DSpine converter lives in `conversion/qwen.py`
there; prepare it with `scripts/prepare-llama.sh pinned`) using
`--target-model-dir Qwen/Qwen3-8B`, BF16, and are hosted at
`jamesdumay/qwen3-8b-spec-drafts-gguf` with SHA-256 verification in `run.sh`.

## Running

`build.sh` prepares the pinned + patched llama.cpp tree and builds `llama-server`
with CUDA. `run.sh` downloads and verifies the weights, then runs each arm with a
fresh server and prints one `DSPINE_AB {json}` summary line per arm (decode tok/s,
draft tokens proposed/accepted, per-prompt rows).

Job submission mirrors the other lab benchmarks: repo `mesh-llm` at a rev that
contains this directory, `build` = `bash mesh-llm/skippy/evals/dspine-llama-ab/build.sh`,
`command` = `bash mesh-llm/skippy/evals/dspine-llama-ab/run.sh`.

Smoke test: set `LIMIT=8` in the job env.
