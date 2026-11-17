---
license: apache-2.0
tags: [mesh-llm, dflash, speculative-decoding]
---

# DFlash pairing index for mesh-llm

Each pairing maps a target model's GGUF identity (architecture, executable layer count, width, expert
count, optional family) to the DFlash draft mesh-llm uses by default. The draft is pinned by Hugging
Face revision and SHA-256, and the pairing carries the default draft cap and the backends it is the
default on.

Every pairing here was chosen from measured decode speed on DGX Spark (CUDA) and Apple Silicon
(Metal). A pairing is the default on a backend only where it beat the model's previous default:
native MTP where the model ships it, else no speculation.

## Files

- `index/<sha256>.json`: every index a mesh-llm release has pinned, named by the SHA-256 of its bytes.
  A release fetches the file its pin names from `main` and uses it only if the bytes hash to the pin.
  These files are never changed or deleted, so old releases keep resolving.
- `index.json`: a copy of the newest index, for reading. No release reads it.

The source of truth is
[`mesh/crates/mesh-llm-skippy-adapter/data/dflash-pairings/`](https://github.com/Mesh-LLM/mesh-llm/tree/main/mesh/crates/mesh-llm-skippy-adapter/data/dflash-pairings)
in mesh-llm. To change the pairings, edit `index.json` there, set `DFLASH_PAIRING_INDEX_SHA256` to
its new hash, and merge. CI then publishes `index/<sha256>.json`, `index.json` and this README.

## Pairings

<!-- pairings:start -->
| Match | Draft | Cap | Backends |
| --- | --- | --- | --- |
| `qwen35`, 32 layers, 4096 wide | [AtomicChat/Qwen3.5-9B-DFlash-GGUF](https://huggingface.co/AtomicChat/Qwen3.5-9B-DFlash-GGUF/blob/c7c69923d77a9aa0004b410d3a7a3dbb64591d41/Qwen3.5-9B-DFlash.Q8_0.gguf) `Qwen3.5-9B-DFlash.Q8_0.gguf` | 15 | all |
| `qwen35`, 64 layers, 5120 wide | [AtomicChat/Qwen3.6-27B-DFlash-GGUF](https://huggingface.co/AtomicChat/Qwen3.6-27B-DFlash-GGUF/blob/a3d4fb1eff6f268cbbdf808042f6902cc17f5458/Qwen3.6-27B-DFlash.Q8_0.gguf) `Qwen3.6-27B-DFlash.Q8_0.gguf` | 15 | all |
| `qwen35`, 64 layers, 5120 wide, name carries `Qwen3.8` | [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/blob/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q8_0.gguf) `Qwen3.8-27B-DFlash2-Q8_0.gguf` | 7 | all |
| `qwen35moe`, 40 layers, 2048 wide, 256 experts | [aminya/Qwen3.6-35B-A3B-DFlash2-GGUF](https://huggingface.co/aminya/Qwen3.6-35B-A3B-DFlash2-GGUF/blob/1e26e3499d865b9d2fc55e019d65b3f0e8ab6846/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf) `Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf` | 7 | not Metal |
| `qwen35moe`, 40 layers, 2048 wide, 256 experts | [Alittlehammmer/Qwen3.6-35B-A3B-DFlash-GGUF-llama.cpp](https://huggingface.co/Alittlehammmer/Qwen3.6-35B-A3B-DFlash-GGUF-llama.cpp/blob/2aa57b642f211d2639083954c5b66d9c1ee503fc/Qwen3.6-35B-A3B-DFlash-Q8_0.gguf) `Qwen3.6-35B-A3B-DFlash-Q8_0.gguf` | 15 | Metal |
| `qwen3next`, 48 layers, 2048 wide, 512 experts | [AtomicChat/Qwen3-Coder-Next-DFlash-GGUF](https://huggingface.co/AtomicChat/Qwen3-Coder-Next-DFlash-GGUF/blob/f5b4c9d2511678f75a41b3d3de5d9e379dd88768/Qwen3-Coder-Next-DFlash.Q8_0.gguf) `Qwen3-Coder-Next-DFlash.Q8_0.gguf` | 15 | Metal |
| `gemma4`, 42 layers, 2560 wide, name carries `Nemotron-3.5-Bolt` | [unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF/blob/b738967d6da3441a7d1c71fcec15e49e32e16ecd/dflash-NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0.gguf) `dflash-NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0.gguf` | 7 | all |
| `nemotron_h_moe`, 52 layers, 2688 wide, 128 experts, name carries `Nemotron-3.5-Lightning` | [magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-GGUF](https://huggingface.co/magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-GGUF/blob/ed35fc990759330e325ab9e41bcf80737ab4dafc/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark.gguf) `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark.gguf` | 7 | not Metal |
| `laguna`, 48 layers, 3072 wide, 256 experts | [poolside/Laguna-S-2.1-GGUF](https://huggingface.co/poolside/Laguna-S-2.1-GGUF/blob/19036076775e9ba6758595f78af99cb976ef8ff0/laguna-s-2.1-DFlash-BF16.gguf) `laguna-s-2.1-DFlash-BF16.gguf` | 15 | not Metal |
| `deepseek4`, 43 layers, 4096 wide, 256 experts, name carries `DeepSeek-V4-Flash-0731` | [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/blob/fbbb5b93fb787c21338159b0af3318bb3f4d9768/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf) `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` | 5 | not Metal |
<!-- pairings:end -->

The cap is further held to 6 on Metal. Layer packages can instead declare their draft in
`generation.speculative_decoding.proposers` (`type: dflash`); a package declaration takes precedence
over this index.
