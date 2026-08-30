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
binary_stage_transport = ""              # Binary stage transport override
prefill_chunking      = "auto"           # Prefill chunking policy
prefill_chunk_size    = 0                 # Fixed prefill chunk size (0 = auto)

[defaults.speculative]
mode                 = "auto"            # "auto", "disabled", or "draft"
draft_model          = ""                # Path or repo for a draft model
draft_hf_repo        = ""                # Hugging Face draft repository
draft_hf_file        = ""                # GGUF file within the draft repository
draft_selection_policy = "auto"          # "auto" sibling discovery or "manual"
draft_max_tokens     = 0                 # Maximum draft-token window
draft_min_tokens     = 0                 # Minimum draft-token window
draft_acceptance_threshold = 0.0         # Minimum accepted fraction (0.0 = no minimum threshold)
draft_split_probability = 0.0            # Deterministic probability of splitting a draft
draft_gpu_layers     = 0                 # Draft GPU layers (0 = auto)
draft_device         = ""                # Draft device override
# draft_threads      = 4                 # Optional draft thread count
draft_cache_type_k   = "f16"             # Draft key-cache dtype
draft_cache_type_v   = "f16"             # Draft value-cache dtype
spec_default         = "auto"            # Automatic speculative defaults policy

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
top_nsigma    = -1.0                     # Top-n-sigma filtering (-1 = disabled)
dynatemp_range = 0.0                     # Dynamic temperature range
dynatemp_exponent = 1.0                  # Dynamic temperature exponent
repeat_last_n = -1                       # Repetition window (-1 = context size)
mirostat_mode = "disabled"               # "disabled", 1, or 2
mirostat_entropy = 5.0                    # Mirostat target entropy
mirostat_learning_rate = 0.1              # Mirostat learning rate
samplers      = ["penalties", "dry", "top_n_sigma", "top_k", "typical_p", "top_p", "min_p", "xtc", "temperature"]
seed          = 0                         # 0 = random seed
ignore_eos    = false                     # Suppress EOS when true
reasoning_format = "auto"                # Template reasoning parser
reasoning_enabled = "auto"               # "auto", "off", or "on"
reasoning_budget = "auto"                # Token count or effort tier
jinja         = true                      # Use Jinja chat rendering
skip_chat_parsing = false                 # Return raw template output metadata
# chat_template = "..."                  # Optional inline template override
# chat_template_file = "/path/template.jinja"
# system_prompt = "You are a concise assistant."
# prefill_assistant = "The answer is"
# grammar = "root ::= ..."               # Mutually exclusive with json_schema
# json_schema = { type = "object" }       # Mutually exclusive with grammar

[defaults.request_defaults.dry]
multiplier = 0.0                          # 0 = disabled
base = 1.75
allowed_length = 2
penalty_last_n = -1
sequence_breakers = ["\n", ":", "\"", "*"]

[defaults.request_defaults.xtc]
probability = 0.0                         # 0 = disabled
threshold = 0.1

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
| `speculative` | Speculative decoding — draft source, verification policy, and runtime controls |
| `request_defaults` | Sampling — temperature, tokens, penalties, stop sequences |
| `multimodal` | Vision — CLIP model, projection, GPU assignment |
| `advanced` | Low-level — slot count, hierarchical slots |
