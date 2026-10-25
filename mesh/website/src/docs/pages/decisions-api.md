---
title: Decisions API
description: Use Mesh's Decisions endpoint with System One models
---

# Decisions API

`POST /v1/decisions` asks a System One model to answer questions about
text. A request can combine a yes/no `predicate`, a `choice` from supplied
options, and a numeric `score`. Mesh serves the request through a local or
reachable mesh model; it does not send the input to OpenAI.

The request and answer fields follow [OpenAI's Decisions API](https://developers.openai.com/api/docs/guides/decisions) for text input. User-message text parts are joined into text evidence for the local model. This backend does not yet evaluate images or produce refusal answers. Model IDs identify locally loaded System One models, rather than OpenAI's `gpt-6-luna`.

## Start and find a model

For a local first run, start the smaller Laya model:

```sh
mesh-llm serve --model meshllm/laya-multilingual-F16-GGUF
```

In another terminal, wait until it appears in `GET /v1/models`, then copy the
exact ID advertised with `system_one`:

```sh
curl -s http://127.0.0.1:9337/v1/models \
  | jq -r '.data[]
      | select(.id != "mesh" and .id != "auto")
      | select((.capabilities // []) | index("system_one"))
      | .id'
```

Use the `/v1/models` response from the same node to which you will send the
decision. A model name without the `system_one` capability is insufficient.
For backend setup and resource requirements, see the [System One API](/docs/pages/system-one-api/).

## Ask three questions

Replace `YOUR_SYSTEM_ONE_MODEL_ID` with a returned ID:

```sh
curl -sS http://127.0.0.1:9337/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "YOUR_SYSTEM_ONE_MODEL_ID",
    "input": "I was charged twice. Please refund me today.",
    "questions": [
      {"type": "predicate", "name": "urgent", "instructions": "Does this need action today?"},
      {"type": "choice", "name": "team", "instructions": "Which team should handle it?",
       "choices": [{"value": "billing", "description": "Payments and refunds"},
                   {"value": "support", "description": "Technical help"}]},
      {"type": "score", "name": "frustration", "instructions": "How frustrated is the customer?",
       "levels": [{"label": "0", "description": "Calm"},
                  {"label": "1", "description": "Frustrated"},
                  {"label": "2", "description": "Angry"}]}
    ]
  }'
```

The response contains one answer for each question, in request order. This is
an **illustrative shape**; probabilities, scores, and token counts depend on
the model and input:

```json
{
  "model": "YOUR_SYSTEM_ONE_MODEL_ID",
  "answers": [
    {"type": "predicate", "name": "urgent", "probability": 0.875},
    {"type": "choice", "name": "team", "choice": "billing", "probabilities": [
      {"value": "billing", "probability": 0.75},
      {"value": "support", "probability": 0.25}
    ], "confidence": 0.75},
    {"type": "score", "name": "frustration", "score": 1.25, "probabilities": [
      {"value": 0, "label": "0", "probability": 0.125},
      {"value": 1, "label": "1", "probability": 0.5},
      {"value": 2, "label": "2", "probability": 0.375}
    ], "confidence": 0.5}
  ],
  "usage": {
    "input_tokens": 12,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 0,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 12
  }
}
```

`probability` and `confidence` are numbers from zero to one. For a score,
`score` is numeric while each probability entry carries the option's zero-based
`value` and supplied `label`. `output_tokens` is zero because System One does
not generate text; the usage fields do not represent full compute cost.

## Ask one question

Send a single predicate when you only need a yes/no probability:

```sh
curl -sS http://127.0.0.1:9337/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{"model":"YOUR_SYSTEM_ONE_MODEL_ID","input":"Please refund my duplicate charge today.","questions":[{"type":"predicate","name":"urgent","instructions":"Does this need action today?"}]}'
```

## Request rules and limits

| Field | Requirement |
| --- | --- |
| `model` | Exact ID or alias advertised with `system_one` by this node's `/v1/models`. Automatic model selection is unsupported. |
| `input` | Text string, or an array of `user` messages with string content or `input_text` parts. Images return an unsupported error until the local backend can evaluate them. |
| `questions` | Nonempty array. Each question has a `type` of `predicate`, `choice`, or `score`. `name` is optional and echoed as `null` when omitted. |
| `instructions` | Required text for each question. |
| `choices` | Required for `choice`; the local backend accepts 2 to 26 options. Each option has a distinct string or boolean `value` and optional `description`. String `"true"` and boolean `true` are distinct. |
| `levels` | Required for `score`; the local backend accepts 2 to 10 levels. Each level has a distinct `label` and optional `description`. The response's numeric `value` is its position in this array. |
| `safety_identifier` | Optional caller-provided identifier. The local backend does not use it for inference. |

The selected System One backend also sets runtime limits. See
[System One API](/docs/pages/system-one-api/#make-a-read) for details. The
endpoint returns decisions for the supplied questions; it does not execute
tools or run an agent loop. Standard chat
completion methods in OpenAI SDKs do not call `/v1/decisions`.

## Smoke a running server

From a source checkout, after starting a System One model, run:

```sh
just decisions-smoke --base-url http://127.0.0.1:9337
```

The command discovers a capable model, sends all three question types through
HTTP, and checks the response shape. Use `--model <ID>` to select a particular
advertised model. The command shares one deadline across discovery and the
Decisions request (120 seconds by default; `--timeout <seconds>` changes it),
limits each response to 1 MiB, and refuses redirects and non-200 responses.
HTTP(S) root URLs are supported; HTTPS and non-IPv4 hosts use the existing
bounded curl prerequisite. URLs with credentials, queries, or fragments are
refused. It requires a running Mesh server with a real System One model;
fixture tests and the command's presence do not establish a live model run.
