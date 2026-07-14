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
