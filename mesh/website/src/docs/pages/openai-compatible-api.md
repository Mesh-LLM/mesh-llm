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

The list includes available, confirmed LLMs served locally or through healthy
plugin and remote mesh routes. Multimodal LLMs remain included. Dedicated speech,
embedding, reranking, decision, and OCR models appear through
[Mesh model discovery](/docs/pages/model-discovery/) instead. Missing or unknown
legacy workload metadata is visible in that catalog but excluded here.

The OpenAI response contains standard model fields:

```json
{"object":"list","data":[{"id":"llm-model","object":"model","created":0,"owned_by":"mesh-llm"}]}
```

`created: 0` explicitly represents an unknown release timestamp. Get an exact
LLM offering with `GET /v1/models/{encoded-id}`. Non-LLM IDs return `404` from
this compatibility lookup even when Mesh catalog lookup finds them.

The same path with an `anthropic-version` header returns the Anthropic envelope:

```sh
curl -s http://localhost:9337/v1/models -H 'anthropic-version: 2023-06-01'
```

```json
{
  "data": [{"id":"llm-model","type":"model","display_name":"llm-model","created_at":"1970-01-01T00:00:00Z","lifecycle":"active"}],
  "has_more": false,
  "first_id": "llm-model",
  "last_id": "llm-model"
}
```

Anthropic lists accept `limit` (1–1000, default 20), `before_id` or `after_id`.
Do not combine both cursor directions. Compatibility discovery contains no
Mesh-specific `metadata`, capability arrays/status fields, `thinking`,
`virtual_model` or `payment`. Read those facts through `/mesh/v1/models` or the
native SDK catalog. `display_name` remains standard in the Anthropic projection.

An answering daemon can still have no eligible LLM route:

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

System One models answer typed questions through
[the Decisions API](/docs/pages/decisions-api/) at `POST /v1/decisions` or the
[native System One API](/docs/pages/system-one-api/) at `POST /systemone`.
Chat completion methods do not call either route.

## Streaming

Clients that support streamed OpenAI-compatible responses can use the same base URL.

Repeated prompt prefixes are reused automatically when the selected model
supports them. Responses report reused prompt tokens in
`usage.prompt_tokens_details.cached_tokens`. See [KV Caching](/docs/pages/kv-caching/)
for defaults, memory policy, and durable disk-cache controls.

## Tool calling

Tool-calling support depends on the selected model and the agent client. Start with console chat, then test the specific agent workflow you plan to use.

## Structured outputs

Structured output support depends on the model and client behavior. Treat schema enforcement as model- and tool-specific unless the catalog marks stronger guarantees.

For the state and routing model, see
[Runtime Lifecycle](/docs/pages/runtime-lifecycle/).
