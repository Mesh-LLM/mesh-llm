---
title: Coding agents
---

# Coding agents

Use the console first. Once chat works at `http://localhost:3131`, connect an agent to the same local Mesh API.

## Console model choice

The chat console remembers your selected model in this browser. **Mesh — automatic** lets Mesh choose; selecting a specific model keeps that choice instead of silently switching back to automatic. If it becomes unavailable, it stays selected and new sends/retries wait until it returns or you choose another model. Queued prompts wait for the current model to be eligible. Changing the next request's model does not prevent stopping an active response with **Stop** or **Escape**.

The console lists the models this node can route to. Models that require a capability this node does not have (for example paid inference, provided by a separately installed plugin) are not listed; a plugin may add its own console pages for what it provides.

## Base URL

```text
http://localhost:9337/v1
```

If an agent asks for an API key, use any placeholder value such as `dummy`.

## Recommended first agent

```sh
mesh-llm goose
```

## Other launchers

```sh
mesh-llm claude
```

```sh
mesh-llm opencode --host 127.0.0.1:9337
```

```sh
mesh-llm pi --host 127.0.0.1:9337
```

The built-in launchers point the agent at Mesh for you. If `--model` is omitted, Mesh chooses from models available on the local mesh.

`mesh-llm goose` and `mesh-llm claude` also give the agent Mesh's MCP tools from the node's console. If the node runs its console on another port than 3131, pass the same port with `--console`.

## Manual setup

For tools without a Mesh launcher, configure an OpenAI-compatible provider:

| Setting | Value |
|---|---|
| Base URL | `http://localhost:9337/v1` |
| API key | `dummy` |
| Model | Any id from `/v1/models` |

List model ids:

```sh
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

## If the agent fails

Confirm console chat still works, then check the API:

```sh
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

If no models are listed, restart the serving node with a model that fits this machine.
