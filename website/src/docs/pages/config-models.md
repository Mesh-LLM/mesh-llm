---
title: Config Models
description: Configuring models and plugins in ~/.mesh-llm/config.toml
---

# Config Models & Plugins

## Models

The `[[models]]` array describes local model intent. Each entry requires a
`model` reference and can override the same sub-configs as `[defaults]`.

- In `serve`, each entry creates eager startup intent.
- In `on_demand`, entries are preference and candidate metadata; they do not
  load eagerly.
- In `client`, entries are never loaded locally.

Explicit `--model` and `--gguf` arguments stay eager in `on_demand`.

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
web_ui_enabled = false # Optional: hide only a declared console projection
command = "python3"
args    = ["-m", "my_plugin"]

[plugin.startup]
connect_timeout_secs = 30
init_timeout_secs    = 30
optional             = false
lazy_start           = false
```

For network-based plugins, use a `url` instead of `command`/`args`:

```toml
[[plugin]]
name = "openai-endpoint"
url  = "http://gpu-box:8000/v1"
```

An external-endpoint-only node needs no `[[models]]` entry:

```toml
[runtime]
mode = "on_demand"

[[plugin]]
name = "openai-endpoint"
url = "http://gpu-box:8000/v1"
```

Start it with bare `mesh-llm serve`. Provider models can appear in
`/v1/models` while no local model process exists. See
[Runtime Lifecycle](/docs/pages/runtime-lifecycle/) and
[Plugins](/docs/pages/plugins/#external-endpoint-only-workflow).

### Plugin startup config

| Field | Default | Description |
|---|---|---|
| `connect_timeout_secs` | host default | Seconds to wait for the plugin control connection. |
| `init_timeout_secs` | host default | Seconds to wait for initialization after connecting. |
| `optional` | `false` | Keep mesh-llm starting and report the plugin inactive if it is unavailable. |
| `lazy_start` | `false` | Start the plugin only when it is directly needed. |

`web_ui_enabled` applies only when the plugin declares a web UI. It controls
the console projection independently of `enabled`, which controls the plugin
process. Set it to `false` to hide the UI without disabling the plugin's MCP,
HTTP, inference, or capability contributions.
