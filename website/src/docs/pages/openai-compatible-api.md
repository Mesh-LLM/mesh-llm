---
title: OpenAI-Compatible API
---

# OpenAI-Compatible API

Mesh exposes one local OpenAI-compatible API. Clients call the local API; Mesh decides which local or peer model handles the request.

Name a model to route to it directly, or send `"model": "mesh"` to let Mesh
choose how to serve the request — see [Automatic routing](/docs/pages/automatic-routing/).
`"model": "auto"` is a deprecated alias for `"mesh"`.

## Base URL

```text
http://localhost:9337/v1
```

Use base URL `http://localhost:9337/v1` and any placeholder API key, such as `dummy`.

## List models

```sh
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

The list reflects usable local models plus models exposed through healthy
plugin and remote mesh routes:

- `ready_idle` can return an empty `data` array; the daemon is healthy but has
  no route yet.
- `ready_proxying` can return plugin or remote models with no local model
  process.
- `ready_serving` includes locally served routes.

Requesting an unknown model returns `404` with `model_not_found`. A known route
that is draining, paused by activity policy, or temporarily unable to accept
work returns `503` with `service_unavailable`. Pausing admission does not
unload the model.

## Chat completion

```sh
curl -s http://localhost:9337/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL","messages":[{"role":"user","content":"Say hello in one sentence."}]}'
```

## Streaming

Clients that support streamed OpenAI-compatible responses can use the same base URL.

## Tool calling

Tool-calling support depends on the selected model and the agent client. Start with console chat, then test the specific agent workflow you plan to use.

## Structured outputs

Structured output support depends on the model and client behavior. Treat schema enforcement as model- and tool-specific unless the catalog marks stronger guarantees.

For the state and routing model, see
[Runtime Lifecycle](/docs/pages/runtime-lifecycle/).
