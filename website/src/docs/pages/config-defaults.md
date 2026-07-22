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
kv_offload              = "auto"        # KV-cache offload policy
prompt_cache            = "auto"        # Prompt-cache policy
flash_attention         = "auto"        # Flash-attention policy

[defaults.hardware]
model_runtime = "auto"                   # "auto", "cpu", "cuda", "rocm", "vulkan", or "metal"
device        = ""                       # Device ID (empty = auto)
gpu_layers    = "auto"                   # Layers to offload (or an integer)
main_gpu      = 0                        # Primary GPU index
tensor_split  = ""                       # Comma-separated tensor split ratios
mmap          = "auto"                   # Memory-map model loading
mlock         = false                    # Lock model pages in RAM
warmup        = "auto"                   # Run model warmup when supported

[defaults.throughput]
parallel              = 1                # Parallel sequence count
continuous_batching   = "auto"          # Enable continuous batching
threads               = 0                # Thread count (0 = auto)
threads_batch         = 0                # Batch thread count (0 = auto)
tuning_profile        = "balanced"      # "throughput", "balanced", or "saver"

[defaults.skippy]
stage_model_path      = ""               # Path or repo for a stage model
stage_role            = ""               # Stage role override
stage_topology        = ""               # Stage topology override
activation_wire_dtype = "auto"           # "auto", "f16", "f32", or "q8"
binary_stage_transport = ""              # Binary stage transport override
prefill_chunking      = "auto"           # Prefill chunking policy
prefill_chunk_size    = 0                 # Fixed prefill chunk size (0 = auto)

[defaults.speculative]
mode                 = "auto"            # "auto", "disabled", or "draft"
draft_model          = ""                # Path or repo for a draft model
draft_max_tokens     = 0                 # Maximum draft-token window
draft_min_tokens     = 0                 # Minimum draft-token window
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
mmproj            = ""                   # Path or reference to a multimodal projector
mmproj_offload    = "auto"               # Projector offload policy
image_min_tokens  = 0                     # Minimum image token budget
image_max_tokens  = 0                     # Maximum image token budget

[defaults.advanced.server]
alias = ""                               # Optional model alias
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
