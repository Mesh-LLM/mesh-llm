---
title: Config Reference
description: Environment variables, rejected fields, and CLI commands for mesh-llm configuration
---

# Config Reference

## Environment Variables

| Variable | Override |
|---|---|
| `MESH_LLM_CONFIG` | Full path to config file (instead of `~/.mesh-llm/config.toml`) |
| `MESH_LLM_PORT` | Inference API port (overrides `owner_control.bind`) |
| `MESH_LLM_CONSOLE_PORT` | Console/management API port |
| `MESH_LLM_DATA_DIR` | Data directory for models and runtime state |

## Managing Config via CLI

```bash
mesh-llm config show           # Show the loaded config
mesh-llm config show-effective # Show the effective (merged) config
```

## Documented Rejected Fields

The following fields are recognized by the parser but explicitly rejected. Setting them will cause a validation error:

| Section | Rejected fields |
|---|---|
| `model_fit` | `rpc_backend`, `threads_http`, `sleep_idle_seconds` |
| `hardware` | `backend_sampling` |
| `request_defaults` | `grammar`, `json_schema`, `logprobs` |
| `advanced.server` | `host`, `port`, `reuse_port`, `timeout`, `metrics`, `slots`, `props`, `api_prefix` |
| Top-level | `embeddings`, `reranking`, `pooling`, `vocoder` |

These fields existed in the predecessor system (llama-server) and are not valid in mesh-llm config.
