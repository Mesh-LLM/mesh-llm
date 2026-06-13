---
title: Plugins
---

# Plugins

Plugins extend mesh-llm with features managed as separate processes. The plugin system handles lifecycle, IPC, and exposing plugin capabilities through MCP and HTTP.

## Install from the Catalog

Search available plugins:

```bash
mesh-llm plugins search
mesh-llm plugins search blackboard
```

Install by name:

```bash
mesh-llm plugins install blackboard
mesh-llm plugins install openai-endpoint
mesh-llm plugins install flash-moe
```

Available Mesh-LLM organization plugins:

| Plugin | Description |
|---|---|
| `blackboard` | Shared mesh notes |
| `openai-endpoint` | Connect external OpenAI-compatible backends (vLLM, TGI, Ollama, etc.) |
| `flash-moe` | Flash-MoE SSD backend for MoE model serving |

## Configure in `config.toml`

Add a `[[plugin]]` entry to `~/.mesh-llm/config.toml` for each plugin:

```toml
[[plugin]]
name = "blackboard"

[[plugin]]
name = "openai-endpoint"
url = "http://gpu-box:8000/v1"

[[plugin]]
name = "flash-moe"
```

Available fields:

- `name` (required): plugin identifier.
- `enabled` (optional, default `true`): whether the plugin starts on launch.
- `command` (optional): path to the plugin binary.
- `args` (optional): command-line arguments (string array).
- `url` (optional): URL for endpoint-style plugins.
- `[plugin.startup]` (optional):
  - `connect_timeout_secs`: max seconds for the initial connection.
  - `init_timeout_secs`: max seconds for initialization handshake.
  - `optional`: if `true`, startup failure is non-fatal.
  - `lazy_start`: if `true`, the plugin starts on first use instead of at launch.

## Manage Installed Plugins

```bash
mesh-llm plugins list                 # list installed plugins
mesh-llm plugins info blackboard      # show plugin details
mesh-llm plugins enable blackboard    # enable a disabled plugin
mesh-llm plugins disable blackboard   # disable without deleting
mesh-llm plugins update blackboard    # update to latest version
mesh-llm plugins delete blackboard    # remove permanently
```

## Plugin Usage: Blackboard

After installing the blackboard plugin, it runs as a managed process when mesh-llm starts. Interact through MCP tools and resources (available through the console's MCP endpoint):

- tool: `blackboard.feed`
- tool: `blackboard.post`
- resource: `blackboard://snapshot`

For detailed blackboard usage, see [github.com/mesh-llm/blackboard](https://github.com/mesh-llm/blackboard).

## Deep Dives

| Page | What it covers |
|---|---|
| [Plugin Architecture](/docs/pages/plugin-architecture/) | Design model, core principles, control session, streams, host vs plugin ownership |
| [Developing Plugins](/docs/pages/developing-plugins/) | DSL guide, author experience, Rust example, internal RPC, streaming |
| [Plugin Reference](/docs/pages/plugin-reference/) | External endpoints, MCP bindings, HTTP bindings, capabilities, mesh channels, control messages |
