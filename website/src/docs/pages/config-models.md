---
title: Config Models
description: Configuring models and plugins in ~/.mesh-llm/config.toml
---

# Config Models & Plugins

## Models

The `[[models]]` array configures which models mesh-llm serves. Each entry can set the same sub-configs as `[defaults]` plus a `name` and `source`.

```toml
[[models]]
name          = "llama-3.1-8b"
source        = "hf:meta-llama/Llama-3.1-8B-Instruct-GGUF"

[models.model_fit]
ctx_size      = 8192
gpu_layers    = -1           # -1 = all layers on GPU

[models.throughput]
parallel      = 4

[models.request_defaults]
max_tokens    = 2048
temperature   = 0.7
```

The `source` field supports:
- `hf:org/model` — Hugging Face model ID
- A local GGUF file path
- A model reference or alias

The optional `runtime` field sets a preferred runtime backend (`auto`, `cpu`, `cuda`, `vulkan`, `metal`, `sycl`, etc.).

```toml
[[models]]
name    = "codellama-7b"
source  = "hf:codellama/CodeLlama-7b-Instruct-GGUF"
runtime = "cuda"

[models.hardware]
gpu_layers = -1
```

## Plugins

The `[[plugin]]` array configures plugin instances as local processes or remote services.

```toml
[[plugin]]
name    = "my-plugin"
enabled = true
command = "python3"
args    = ["-m", "my_plugin"]

[plugin.startup]
timeout_secs   = 30
restart_delay  = 2
max_restarts   = 3
```

For network-based plugins, use a `url` instead of `command`/`args`:

```toml
[[plugin]]
name = "openai-endpoint"
url  = "http://gpu-box:8000/v1"
```

### Plugin startup config

| Field | Default | Description |
|---|---|---|
| `timeout_secs` | 30 | Seconds to wait for plugin ready signal |
| `restart_delay` | 2 | Seconds between restart attempts |
| `max_restarts` | 3 | Maximum restart attempts before giving up |
