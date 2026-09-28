---
title: System One API
---

# System One API

`POST /systemone` is a separate API on the same HTTP server, outside the
OpenAI-compatible `/v1` surface. It is **not a standard OpenAI endpoint**. It accepts shared `state` and named
`questions`, returning typed answers and label probabilities rather than chat
messages or generated text. Selecting its model in an ordinary chat client
will not switch that client to System One; use an explicit HTTP request or an
OpenJEV-aware integration. Standard OpenAI SDK chat-completion methods do not
call this route.

## Choose a model

Start with one of these backends:

| Backend | Model to try | Best for | Requirements |
| --- | --- | --- | --- |
| Laya | `convaiinnovations/laya-multilingual`, converted to `laya-multilingual-F16.gguf` | A small first test, multilingual classification, CPU or GPU | One converted GGUF; about 829 MB including the worst-case read reserve |
| OpenJEV / DiffusionGemma | `unsloth/diffusiongemma-26B-A4B-it-GGUF:Q4_K_M` | Testing the DiffusionGemma System One implementation | CUDA, one complete-model worker, `parallel = 1`, and a micro-batch large enough for the model's answer canvas |

Laya is the simplest functional test. DiffusionGemma exercises the OpenJEV
path and is the qualified large-model configuration. Both implement the same
HTTP request and response contract.

## Run Laya

Prepare the llama.cpp converter once with `just llama-prepare`, then convert
the upstream checkpoint:

```sh
hf download convaiinnovations/laya-multilingual --local-dir /tmp/laya-multilingual
python3 .deps/llama.cpp/convert_hf_to_gguf.py /tmp/laya-multilingual \
  --outtype f16 --outfile /tmp/laya-multilingual-F16.gguf
```

Use that converted file. Other published Laya GGUFs may omit the decision head
or use an incompatible layout.

```sh
mesh-llm serve --gguf /tmp/laya-multilingual-F16.gguf --headless
```

Without an explicit device, Laya runs on CPU. Use the node's normal `--device`
or pinned-GPU configuration to place it on a GPU, or set
`MESH_LLM_LAYA_ACCELERATOR=1` to opt into the first GPU.

## Run OpenJEV / DiffusionGemma

Create `/tmp/openjev.toml`:

```toml
version = 1

[gpu]
assignment = "auto"
parallel = 1

[defaults.model_fit]
ctx_size = 8192
batch = 256
ubatch = 256

[[models]]
model = "unsloth/diffusiongemma-26B-A4B-it-GGUF:Q4_K_M"

[models.throughput]
parallel = 1

[models.advanced.server]
alias = "openjev-latest"
```

Then start a private node:

```sh
mesh-llm serve --config /tmp/openjev.toml --mesh-name openjev --headless
```

The worker must hold the complete model. Split serving is not supported for a
System One read.

## Find System One models

`GET /v1/models` advertises `system_one` only after the loaded backend proves
it implements the endpoint. Query the local node to see both local and remote
models visible through the mesh:

```sh
curl -s http://127.0.0.1:9337/v1/models \
  | jq '.data[]
      | select(.capabilities | index("system_one"))
      | {id, display_name, system_one_status, metadata}'
```

Print only model IDs that are safe to send to `/systemone`:

```sh
curl -s http://127.0.0.1:9337/v1/models \
  | jq -r '.data[]
      | select(.capabilities | index("system_one"))
      | .id'
```

Do not infer endpoint support from `metadata.workload_class` or architecture.
Laya's primary workload is `decision`; DiffusionGemma's is
`causal_generation`. The explicit `system_one` capability is the common API
contract.

## Make a read

Use the exact `id` returned by `/v1/models`. For the DiffusionGemma setup
above, that is the configured `openjev-latest` alias:

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

The API supports text-only `noul`, `choice`, and `score` questions with one
read. DiffusionGemma accepts 2–26 choice labels and 2–10 score criteria; Laya
accepts up to 16 options per question. DiffusionGemma requires the complete
model on one worker with one inference lane; split serving is not supported
for this operation. CUDA is its qualified backend; Metal is not certified.
Images, thinking, sequential reads, and multiple samples/steps are rejected.
Use an explicit loaded model ID or configured alias, not automatic model
selection. The chat guardrail wrapper does not screen System One requests.

`usage.input_tokens` counts prompt tokens, not the fixed diffusion canvas;
`usage.output_tokens` is zero because no text is generated. This is not full
compute accounting or a production OpenJEV compatibility guarantee.

For Laya, change only the `model` value to the discovered ID, normally
`laya-multilingual-F16`. Object-valued `state` is also accepted:

```sh
curl http://127.0.0.1:9337/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "laya-multilingual-F16",
    "state": {
      "from": "user@example.com",
      "body": "I was charged twice this month."
    },
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle it?",
        "criteria": {
          "billing": "charges or refunds",
          "support": "technical troubleshooting"
        }
      }
    }
  }'
```

## Backend differences

Two model families answer System One reads with the same request and response
shape:

- **DiffusionGemma** reads label probabilities from one diffusion canvas step,
  as described above.
- **Laya** (`general.architecture = "laya"`, for example a converted
  `convaiinnovations/laya-multilingual`) is a small encoder with a typed
  decision head that scores every option in one forward pass. It serves only
  `/systemone`, keeps the request order of choice options and `state` keys, and
  follows the node's configured device (`--device` or a pinned GPU); with none
  it runs on the CPU unless `MESH_LLM_LAYA_ACCELERATOR=1` puts it on the GPU.
  It can use the same configured aliases, but aliases must remain unique on a
  node. Use the discovered model ID when serving both backends.

See the [OpenJEV setup and validation runbook](https://github.com/Mesh-LLM/mesh-llm/blob/main/docs/design/OPENJEV_SKIPPY_POC.md)
for worker configuration and the supported subset.
