---
title: Config Defaults
description: Shared default model settings in ~/.mesh-llm/config.toml
---

# Config Defaults

Shared default settings applied to every model. Individual model entries can override these.

```toml
[defaults]

[defaults.model_fit]
ctx_size                = 0              # Context window (0 = auto)
batch                   = 0              # Batch size (0 = auto)
ubatch                  = 0              # Micro-batch size (0 = auto)
cache_type_k            = "f16"          # Key cache dtype
cache_type_v            = "f16"          # Value cache dtype
kv_offload              = true           # Offload KV cache to GPU
no_kv_offload           = false          # Keep KV cache on CPU
no_mmap                 = false          # Disable memory-mapped model loading
mlock                   = false          # Lock model in RAM
num_experts_per_token   = 0              # MoE: experts per token (0 = all)
expert_count            = 0              # MoE: total expert count
no_cont_batching        = false          # Disable continuous batching
max_batched_tokens      = 0              # Max tokens per batch
pooling_type            = ""             # Pooling type for embedding models

[defaults.hardware]
model_runtime = "cpu"                    # "cpu", "cuda", "vulkan", "metal", "sycl", etc.
device        = ""                       # Device ID (empty = auto)
gpu_layers    = 0                        # Layers to offload to GPU (0 = auto)
main_gpu      = 0                        # Primary GPU index
tensor_split  = ""                       # Comma-separated tensor split ratios
no_mul_mat_q  = false                    # Disable mulmat quantization
decode_only_q = false                    # Quantize only decode path
check_tensors = ""                       # Tensor check mode
temp_file     = ""                       # Temporary file path for offloading

[defaults.throughput]
parallel              = 1                # Parallel sequence count
continuous_batching   = true             # Enable continuous batching
threads               = 0                # Thread count (0 = auto)
cpu_threads           = 0                # CPU thread count
flash_attn            = false            # Enable flash attention
no_flash_attn         = false            # Disable flash attention
tensor_cores          = false            # Use tensor cores on NVIDIA GPUs
no_kv_reuse           = false            # Disable KV cache reuse across requests

[defaults.skippy]
stage_model_package  = ""                # Path or repo for skippy stage model
enable               = false             # Enable skippy stage serving
activation_dtype     = "q8_0"            # Activation wire dtype
num_stages           = 0                 # Number of stages (0 = auto)
stage_plan           = ""                # Explicit stage plan path

[defaults.speculative]
draft_model          = ""                # Path or repo for draft model
enable               = false             # Enable speculative decoding
draft_parallel       = 1                 # Draft parallel sequences
draft_gpu_layers     = 0                 # Draft GPU layers (0 = auto)
draft_device         = ""                # Draft device override

[defaults.request_defaults]
max_tokens    = 0                        # Max tokens per request (0 = model default)
temperature   = 0.0                      # Sampling temperature (0.0 = model default)
top_p         = 0.0                      # Top-p sampling
top_k         = 0                        # Top-k sampling
min_p         = 0.0                      # Min-p sampling
repeat_penalty = 0.0                     # Repeat penalty
presence_penalty = 0.0                   # Presence penalty
frequency_penalty = 0.0                  # Frequency penalty
stop          = []                       # Stop sequences
typical_p     = 0.0                      # Typical sampling

[defaults.multimodal]
clip_model_path   = ""                   # Path to CLIP model for multimodal
mmproj_path       = ""                   # Path to multimodal projection
image_main_gpu    = 0                    # GPU for image processing

[defaults.advanced]
cont_batching_init_slots = 0             # Initial slot count for continuous batching
hierarchical_slots        = false        # Enable hierarchical KV slot management
```

## Sub-config reference

| Section | Purpose |
|---|---|
| `model_fit` | Memory sizing — context, batch, cache dtype, offloading |
| `hardware` | Device assignment — runtime backend, GPU layers, tensor split |
| `throughput` | Concurrency — parallel sequences, threading, flash attention |
| `skippy` | Stage-split serving — stage packages, activation dtypes |
| `speculative` | Speculative decoding — draft model, GPU layers |
| `request_defaults` | Sampling — temperature, tokens, penalties, stop sequences |
| `multimodal` | Vision — CLIP model, projection, GPU assignment |
| `advanced` | Low-level — slot count, hierarchical slots |
