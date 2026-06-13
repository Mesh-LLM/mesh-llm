---
title: OpenAI-Compatible API
---

# OpenAI-Compatible API

Mesh exposes one local OpenAI-compatible API. Clients call the local API; Mesh decides which local or peer model handles the request.

## Base URL

```text
http://localhost:9337/v1
```

Use base URL `http://localhost:9337/v1` and any placeholder API key, such as `dummy`.

## List models

```sh
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

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
