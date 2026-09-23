---
title: System One API
---

# System One API (proof of concept)

`POST /systemone` is a separate API on the same HTTP server, outside the
OpenAI-compatible `/v1` surface. It is **not a standard OpenAI endpoint**. It accepts shared `state` and named
`questions`, returning typed answers and label probabilities rather than chat
messages or generated text. Selecting its model in an ordinary chat client
will not switch that client to System One; use an explicit HTTP request or an
OpenJEV-aware integration. Standard OpenAI SDK chat-completion methods do not
call this route.

With a complete DiffusionGemma model loaded under the `openjev-latest` alias:

```sh
curl http://localhost:9337/systemone \
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

