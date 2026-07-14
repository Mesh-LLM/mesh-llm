---
title: Config Models
description: Configuring models and plugins in ~/.mesh-llm/config.toml
---

# Config Models & Plugins

## Models

The `[[models]]` array configures which models mesh-llm serves. Each entry requires a `model` reference and can override the same sub-configs as `[defaults]`.

```toml
[[models]]
model         = "meta-llama/Llama-3.1-8B-Instruct-GGUF"

[models.model_fit]
ctx_size      = 8192

[models.throughput]
parallel      = 4

[models.request_defaults]
max_tokens    = 2048
temperature   = 0.7
```

The `model` field accepts a catalog id, Hugging Face reference, URL, or local model path.

```toml
[[models]]
model = "codellama/CodeLlama-7b-Instruct-GGUF"

[models.hardware]
model_runtime = "cuda"
gpu_layers = "auto"
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
