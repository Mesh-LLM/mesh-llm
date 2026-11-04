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

## Managed inference providers

A model entry can select an installed inference plugin. Provider models use
provider-owned option and request schemas and do not inherit built-in LLM
hardware, sampling or context defaults.

```toml
[[plugin]]
name = "onnx"

[[models]]
model = "datalab-to/indra-ocr@v1.1.0"
provider = "onnx"
variant = "fp32"

[models.options]
threads = 4

[models.request_defaults]
dpi = 200

[models.lifecycle]
load = "on_demand"
idle_timeout_secs = 300
prefetch = true
```

The host passes only this provider's selected models to its process. It prepares
explicit asset plans through the shared Hugging Face cache; credentials stay in
Mesh. The plugin validates options and creates inference sessions on demand.
Idle eviction releases sessions without deleting cached assets. Set prefetch to
false to defer downloading too. A local source directory bypasses downloading;
relative directories resolve against the configuration file's directory.

## Plugins


The `[[plugin]]` array configures local plugin processes, which can attach to external services.

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

For the installed endpoint adapter, `url` names the upstream HTTP service. Mesh
still starts a local plugin process; `url` does not replace the adapter's
control connection. Install a host-compatible adapter first, or set `command`
to a compatible source-built adapter:

```toml
[[plugin]]
name = "openai-endpoint"
url  = "http://127.0.0.1:8000/v1"
```

An external-endpoint-only node needs no `[[models]]` entry:

```toml
[runtime]
mode = "on_demand"

[[plugin]]
name = "openai-endpoint"
url = "http://127.0.0.1:8000/v1"
```

Start it with bare `mesh-llm serve`. An enabled, resolved inference adapter
with a valid URL can serve without a native runtime when no local model startup
is requested. Mesh reports the missing runtime, advertises no native capacity,
and rejects later local-model and split-stage loads. Install a compatible native
runtime and restart to use those operations. Explicit local-model arguments and
configured `[[models]]` entries still require a native runtime. Provider models
can appear in `/v1/models` while no local model process exists. See
[Runtime Lifecycle](/docs/pages/runtime-lifecycle/) and
[provider quick start](/docs/pages/external-model-endpoints/) for Ollama, vLLM
and LM Studio, release compatibility, and HTTP/authentication limitations.

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

`web_ui_primary_tab` promotes a declared web UI page to a primary console tab
instead of the default auxiliary navigation item, but only when the page's own
manifest also requests `placement = "primary"` — the preference alone does
nothing. It defaults to off, is independent of `web_ui_enabled`, and the host
may still fall back to auxiliary placement if the primary tab row is full.

## On-demand native models

Mesh can load an unsplit native Skippy/llama model when an explicitly named
request arrives, then unload that instance after its final request has been
idle for the configured interval. The server stays running and cached disk
assets remain available for the next load.

```toml
[[models]]
model = "Qwen3-0.6B-Q4_K_M"

[models.lifecycle]
load = "on_demand"
idle_timeout_secs = 300
prefetch = true
```

Restart Mesh after changing lifecycle policy. The default is `load = "startup"`, with idle eviction disabled. Set `prefetch`
to `false` to defer disk resolution/download until the first request. With
`prefetch = true`, Mesh resolves disk assets in the background without loading
weights into GPU memory. Strict local source policies never prefetch remotely.

Concurrent requests share a load, with a five-minute wait limit. Active
responses, including streaming, hold residency leases. Routing to an already
ready peer avoids a local cold load; assigned peer requests execute locally.
Idle unloading uses the normal non-forced drain path and respects maintained
owner intents. Capacity admission includes the existing runtime memory budget;
a failed cold load returns HTTP 503 and can be retried.

This initial policy applies to uniquely configured native model identifiers and
instances loaded on demand. Configured aliases and cached public IDs resolve to the same residency.
On-demand configuration rejects `--split` and `--split-topology-lock`; use
`load = "startup"` for split serving. Explicit CLI model selections retain their
eager startup behavior. Startup models, split controllers, and manually
loaded instances retain their existing residency behavior. Automatic model
selection chooses ready models; explicitly name a cold configured model to
load it. Direct standalone Skippy serving retains its existing lifecycle.
