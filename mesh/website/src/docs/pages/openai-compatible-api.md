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

The OpenAI list currently returns the full eligible LLM list and rejects query
parameters with `400`; `limit`, `order` and `after` pagination are not supported.
Clients must omit those parameters. Anthropic pagination is described below.

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

A model this node is still loading returns `503` with `model_not_ready` and
`Retry-After: 5`. That covers a request sent while the startup models resolve,
during a first load or a reload, and `auto` before any model is ready. Retry
after the stated seconds. A name that is still unknown once the startup models
resolve returns `404`.

## Chat completion

```sh
curl -s http://localhost:9337/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL","messages":[{"role":"user","content":"Say hello in one sentence."}]}'
```

System One models answer typed questions through
[the Decisions API](/docs/pages/decisions-api/) at `POST /v1/decisions` or the
[native System One API](/docs/pages/system-one-api/) at `POST /v1/systemone`.
Chat completion methods do not call either route.

## Responses API

`POST /v1/responses` implements the HTTP and SSE portions of the Open
Responses 2026-04-24 protocol. Mesh translates the request onto the selected
model backend, but the localhost ingress owns the response ID, transcript, and
stream lifecycle. A later request can send `previous_response_id` plus only its
new input; Mesh restores the earlier structured input and output before it
chooses a local or peer worker. Function calls and `function_call_output` items
keep their `call_id` across that continuation. Effective tools, tool choice,
structured-output/reasoning controls, sampling controls, and output limits are
inherited when the follow-up omits them; explicit child values replace them.
Instructions never inherit, matching the protocol's conversation-state rule.

```sh
curl -s http://localhost:9337/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"mesh","input":"Remember that my code word is cobalt."}'
```

Completed responses are retained in bounded process memory for 30 minutes
after their last use, up to 128 responses and 64 MiB of charged transcript per
caller (and 256 MiB total). A restart, expiry, or eviction makes the ID unknown
and returns `previous_response_not_found`; Mesh never treats an unknown ID as
an empty conversation. `store: false` prevents HTTP retention. Set
`MESH_RESPONSE_STORAGE=off` before startup to disable retention entirely.

HTTP continuation is deliberately limited to direct loopback clients without
`Forwarded`, `X-Forwarded-For`, or `X-Real-IP`. All local processes share that
host-local trust scope. A reverse proxy or multi-user deployment must disable
storage unless it supplies a separately authenticated ingress identity; a
body field or caller-provided header cannot select another caller's scope.

### WebSocket

Upgrade `GET /v1/responses` (`ws://localhost:9337/v1/responses`) and send JSON
`response.create` messages for persistent sequential turns. Streaming event
objects are sent directly as WebSocket text messages, without SSE framing or
`[DONE]`. The connection keeps its latest completed response even when that
turn used `store: false`; a `store: true` turn may also use the same bounded
caller-scoped process store as HTTP after reconnect.

Mesh retains the session at the original ingress and independently applies
request transforms, admission, private scope, target selection, routing, and
retry to every turn. The pending-turn queue, frames, outbound queue, and
retained transcript are bounded, and the absolute connection lifetime is 60
minutes including active work. Disconnecting cancels the active downstream
request. The startup bootstrap proxy does not support this upgrade, and a
serving peer never receives the whole WebSocket; it receives an ordinary
expanded HTTP request for a routed turn.

### Compaction

`POST /v1/responses/compact` summarizes a Responses transcript into an opaque
compaction item. Send `model` plus `input`; the compact request may also name a
stored `previous_response_id` to include that response's transcript. The
endpoint is non-streaming and returns `object: "response.compaction"` with the
original user messages followed by a `compaction` item. Feed that `output`
array into a later `/v1/responses` request and append the new input:

```sh
curl -s http://localhost:9337/v1/responses/compact \
  -H 'Content-Type: application/json' \
  -d '{"model":"mesh","input":[{"type":"message","role":"user","content":"Remember cobalt."}]}'
```

The item's `encrypted_content` authenticates the model-generated summary and
metadata for unresolved function calls. On restore, Mesh inserts those calls
immediately after the summary and before newly appended
`function_call_output` items; completed parallel calls are not retained.

Compaction encryption is deliberately process-local. An item can be restored
only by the same ingress process and exact model ID that created it. A restart,
another ingress instance, a changed model, tampering, or an oversized item
fails closed. The encrypted item is opaque provider state, not portable
end-to-end encryption for client data.

A compaction resource is not stored and has no usable
`previous_response_id`. Do not pass its `id` as `previous_response_id`; retain
and resend its `output` items instead. A compact request may consume an earlier
stored response ID, but the compact result does not create another stored
Responses turn.

The following standard behavior is available when the selected model supports
it:

- typed message history, images/files handled by the existing Mesh media path,
  developer-hosted function tools, parallel calls, and structured output via
  `text.format`;
- text, function-call, reasoning, usage, incomplete, and failed lifecycle
  events over SSE, with stable response/item IDs and one terminal response
  event;
- assistant input `phase` labels preserved through translation. Output phase
  labels are emitted only when a backend actually provides them.

Unsupported semantics fail explicitly. Mesh does not currently implement
background responses, automatic truncation, `max_tool_calls`, non-default
service tiers, text verbosity, hosted tools, or encrypted reasoning
continuation. The existing `/v1/realtime` WebSocket is a separate transcription
protocol. Compaction items are the one opaque provider-state item described
above; Mesh does not create encrypted reasoning state.

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
