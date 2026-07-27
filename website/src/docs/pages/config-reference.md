---
title: Config Reference
description: Environment variables, rejected fields, and CLI commands for mesh-llm configuration
---

# Config Reference

## Environment Variables

| Variable | Override |
|---|---|
| `MESH_LLM_CONFIG` | Full path to config file (instead of `~/.mesh-llm/config.toml`) |

## Managing Config via CLI

```bash
mesh-llm config validate
mesh-llm config validate --config-path ./mesh.toml
mesh-llm config validate --config-path ./mesh.toml --json
```

`config validate` checks the TOML file without starting a node. If
`--config-path` is omitted, it uses the global `--config` path, then
`MESH_LLM_CONFIG`, then `~/.mesh-llm/config.toml`.

## Runtime fields

| Field | Default | Allowed values / range | Persistence |
|---|---|---|---|
| `runtime.mode` | `serve` | `serve`, `on_demand`, `client` | Config |
| `runtime.startup_failure_policy` | `best_effort` | `best_effort`, `fail_fast` | Config |
| `runtime.drain_timeout_secs` | `30` | 1–3600; no greater than maximum | Config |
| `runtime.drain_timeout_max_secs` | `300` | 1–3600; no less than normal timeout | Config |

## Activity fields

| Field | Default | Allowed values / range | Persistence |
|---|---|---|---|
| `runtime.activity.enabled` | `false` | Boolean | Config |
| `runtime.activity.idle_after_secs` | `300` | 30–86400 | Config |
| `runtime.activity.poll_interval_secs` | `5` | 1–60 | Config |
| `runtime.activity.resume_debounce_secs` | `30` | 0–300 | Config |
| `runtime.activity.response` | `pause_remote` | `pause_remote`, `pause_all`, `reduce_priority` | Config |
| `runtime.activity.advertisement` | `coarse_state` | `none`, `availability_only`, `coarse_state`, `private_coarse_state` | Config |
| Activity override | `auto` | `auto`, `active`, `idle` | Session only; never written to config |

See [Runtime Lifecycle](/docs/pages/runtime-lifecycle/) for mode behavior,
admission semantics, and the privacy boundary.

## Rejected Fields

The following fields are recognized by the parser but explicitly rejected. Setting them will cause a validation error:

| Section | Rejected fields |
|---|---|
| `hardware` | `rpc_backend` |
| `throughput` | `threads_http`, `sleep_idle_seconds` |
| `skippy` | `openai_frontend_mode` |
| `request_defaults` | `backend_sampling`, `grammar`, `json_schema`, `logprobs` |
| `advanced.server` | `host`, `port`, `reuse_port`, `timeout`, `metrics`, `slots`, `props`, `api_prefix` |
| `multimodal` | `embeddings`, `reranking`, `pooling`, `vocoder` |

These fields existed in the predecessor system (llama-server) and are not valid in mesh-llm config.
