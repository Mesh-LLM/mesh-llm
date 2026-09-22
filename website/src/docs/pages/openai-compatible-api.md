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

## System One: OpenJEV extension (proof of concept)

`POST /v1/systemone` uses the same HTTP server and `/v1` base URL, but it is
**not a standard OpenAI endpoint**. It accepts shared `state` and named
`questions`, returning typed answers and label probabilities rather than chat
messages or generated text. Selecting its model in an ordinary chat client
will not switch that client to System One; use an explicit HTTP request or an
OpenJEV-aware integration. Standard OpenAI SDK chat-completion methods do not
call this route.

With a complete DiffusionGemma model loaded under the `openjev-latest` alias:

```sh
curl http://localhost:9337/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openjev-latest",
    "state": "I was charged twice this month.",
    "questions": {
      "is_billing": {
        "type": "noul",
        "instructions": "Is this a billing issue?"
      }
    }
  }'
```

The response envelope is `{model, answers, usage}`. The answer for this
question is `answers.is_billing`, with `type: "noul"` and a `noul` probability
between zero and one. `choice` questions return a selected label and label
probabilities; `score` questions return a numeric score and its distribution.
These results can inform an application's next action; the endpoint does not
execute tools or run an agent loop.

The PoC supports text-only `noul`, `choice` (2–26 labels), and `score`
(2–10 criteria) questions with one read. It requires the full DiffusionGemma
model on one worker with one inference lane; split serving is not supported
for this operation. CUDA is the qualified backend; Metal is not certified.
Images, thinking, sequential reads, and multiple samples/steps are rejected.
Use an explicit loaded model ID or configured alias, not automatic model
selection. The chat guardrail wrapper does not screen System One requests.

`usage.input_tokens` counts prompt tokens, not the fixed diffusion canvas;
`usage.output_tokens` is zero because no text is generated. This is not full
compute accounting or a production OpenJEV compatibility guarantee.

See the [OpenJEV setup and validation runbook](https://github.com/Mesh-LLM/mesh-llm/blob/main/docs/design/OPENJEV_SKIPPY_POC.md)
for worker configuration and the supported subset.

## Streaming

Clients that support streamed OpenAI-compatible responses can use the same base URL.

## Tool calling

Tool-calling support depends on the selected model and the agent client. Start with console chat, then test the specific agent workflow you plan to use.

## Structured outputs

Structured output support depends on the model and client behavior. Treat schema enforcement as model- and tool-specific unless the catalog marks stronger guarantees.

For the state and routing model, see
[Runtime Lifecycle](/docs/pages/runtime-lifecycle/).
