---
title: Coding agents
---

# Coding agents

Use the console first. Once chat works at `http://localhost:3131`, connect an agent to the same local Mesh API.

## Console model and payment choices

The chat console remembers your selected model and Free/Paid preference in this browser. **Mesh — automatic** lets Mesh choose; selecting a specific model keeps that choice instead of silently switching back to automatic. If it becomes unavailable, it stays selected and new sends/retries wait until it returns or you choose another model. Queued prompts wait for the current model to be eligible under their captured payment restriction.

**Free** restricts requests to free providers, including your own locally hosted models even when you charge remote buyers. Paid-only models are hidden. **Paid** permits paid providers; it does not require a charge when a free provider is available. Prices are advertised estimates, not binding quotes; “from” is the lowest advertised output rate. The switch appears only when the node reports automatic wallet policy. Otherwise requests stay free-only.

Model prices and wallet policy refresh periodically while chat is open. Changing the next request's model/payment choice does not prevent stopping an active response with **Stop** or **Escape**.

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
