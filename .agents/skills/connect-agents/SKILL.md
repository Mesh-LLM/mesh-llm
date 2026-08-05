---
name: connect-agents
description: Use this skill when connecting agent tools or OpenAI clients to mesh-llm — launching or configuring Goose, Claude Code, OpenCode, Pi, curl, or any OpenAI-compatible client against a local or remote mesh, picking a model, or validating tool-call reliability.
metadata:
  short-description: Connect agents and OpenAI clients to mesh-llm
---

# connect-agents

Use this when pointing an agent harness or any OpenAI client at a running
mesh-llm node. Full reference: `docs/AGENTS.md`.

## Mental model

- Every node serves an OpenAI-compatible API at `http://<host>:9337/v1`.
- `GET /v1/models` lists everything reachable (local + mesh peers); requests
  route by the `model` field.
- Special model ids: `auto` lets the mesh pick; `mesh` engages the
  mixture-of-agents path. Otherwise use an exact id from `/v1/models`.
- For coding agents, pick a tool-capable model. If `--model` is omitted, the
  built-in launchers pick the strongest tool-capable model available.

## Built-in launchers (preferred)

mesh-llm launches the major agent CLIs with config injected for you:

```bash
mesh-llm goose     [--model <id>]                 # writes ~/.config/goose/custom_providers/mesh.json
mesh-llm claude    [--model <id>]
mesh-llm opencode  [--model <id>] [--host <h>]    # injects OPENCODE_CONFIG_CONTENT (no file edits)
mesh-llm pi        [--model <id>] [--host <h>]    # writes ~/.pi/agent/models.json
```

- `goose`/`claude` reuse a local mesh on the chosen `--port`.
- `opencode`/`pi` target `--host` (default `127.0.0.1:9337`) and auto-start a
  local client only for loopback targets; the auto-started node is cleaned up
  when the harness exits.
- `mesh-llm pi --write` / `mesh-llm opencode --write` update config without
  launching (use `--host` for remote meshes).
- Agent launch commands also install available plugin skills for that agent
  (`mesh-llm skills install` does it standalone).

## Manual config (any OpenAI client)

Base URL `http://<host>:9337/v1`, any non-empty API key:

```bash
export GOOSE_PROVIDER=openai GOOSE_MODEL="<id-from-v1-models>"
export OPENAI_HOST="http://127.0.0.1:9337" OPENAI_API_KEY="mesh"
```

```bash
curl -s http://localhost:9337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"hello"}]}'
```

Exact manual provider JSON for OpenCode and Pi is in `docs/AGENTS.md`.

## Validating agent behavior

Direct API contract probe (tool-call forcing, streaming reconstruction):

```bash
scripts/qa-agent-tool-call-reliability.py \
  --base-url http://127.0.0.1:9337/v1 --models auto,mesh --attempts 3 \
  --output target/agent-tool-call-reliability/results.jsonl
```

Broader harness (models, chat, streaming, plus optional Goose/OpenCode/Pi
smokes): `scripts/qa-nightly-stability.py` — see `docs/AGENTS.md`. Use
`--print-plan` on either script for a side-effect-free preview.

## Blackboard (cross-mesh agent coordination)

Agents can share status/questions across the mesh via the blackboard plugin —
even from a client-only node:

```bash
mesh-llm plugins install blackboard
mesh-llm blackboard "STATUS: [org/repo branch:main] refactoring billing module"
mesh-llm blackboard --search "QUESTION"
```

MCP access: the management endpoint `http://127.0.0.1:3131/mcp` exposes
`blackboard_post`, `blackboard_search`, `blackboard_feed`. Posts are visible to
every peer — never post secrets, credentials, private paths, or customer data.

## Gotchas

- Use a base URL ending in `/v1`; prefer chat-completions over the Responses
  API unless the client documents Responses support.
- Model ids must match `/v1/models` exactly (they can contain spaces — quote
  them).
- An empty `/v1/models` usually means the model is still loading or no mesh was
  joined yet — check `/api/status` on `:3131` (see `mesh-join`).
- The response `"model"` field tells you which node/model actually answered.
