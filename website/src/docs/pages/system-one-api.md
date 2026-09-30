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
| Laya | `meshllm/laya-multilingual-F16-GGUF` | A small first test, multilingual classification, CPU or GPU | One automatically downloaded GGUF; about 829 MB including the worst-case read reserve |
| OpenJEV / DiffusionGemma | `unsloth/diffusiongemma-26B-A4B-it-GGUF:Q4_K_M` | Testing the DiffusionGemma System One implementation | CUDA, one complete-model worker, `parallel = 1`, and a micro-batch large enough for the model's answer canvas |

Laya is the simplest functional test. DiffusionGemma exercises the OpenJEV
path and is the qualified large-model configuration. Both implement the same
HTTP request and response contract.

## Run Laya

With mesh-llm v0.77.0 or newer installed:

```sh
mesh-llm serve --model meshllm/laya-multilingual-F16-GGUF
```

Mesh downloads the [compatible F16 GGUF](https://huggingface.co/meshllm/laya-multilingual-F16-GGUF)
when it is not cached and reuses it on subsequent runs. No source checkout,
Python installation, or manual conversion is required. Wait for the model to
be ready, then use its discovered ID with `/systemone` as shown below.

Without an explicit device, Laya runs on CPU. Use the node's normal `--device`
or pinned-GPU configuration to place it on a GPU, or set
`MESH_LLM_LAYA_ACCELERATOR=1` to opt into the first GPU.

### Alongside a chat model

Repeat `--model` to serve both on one node:

```sh
mesh-llm serve \
  --model meshllm/laya-multilingual-F16-GGUF \
  --model Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0
```

Both models must fit in the node's memory budget. With no device override,
Laya keeps its CPU default while the chat model uses normal device selection.
A global `--device` applies to both; use [per-model configuration](/docs/pages/config-models/)
when they need different explicit device assignments. Multiple `[[models]]`
entries also work with `mesh-llm serve --config /path/to/config.toml`.
Explicit `--model` or `--gguf` arguments replace the configured startup model list.

Add `--join <invite>` to serve them on an existing mesh, or `--publish` to
advertise your mesh for discovery; see [mesh setup](https://github.com/Mesh-LLM/mesh-llm/blob/main/docs/MESHES.md).
Only call `/systemone` through a client after its own `/v1/models` response
advertises `system_one` for the selected model; seeing a model name alone is
not sufficient. The chat model uses `/v1/chat/completions`. Loading both does
not automatically route chat through Laya.

### Advanced: convert your own checkpoint

The published model above is sufficient for normal use. For a custom conversion,
use a Mesh source checkout with its patched converter, the Hugging Face CLI,
and the converter's Python dependencies:

```sh
just llama-prepare
hf download convaiinnovations/laya-multilingual --local-dir /tmp/laya-multilingual
python3 .deps/llama.cpp/convert_hf_to_gguf.py /tmp/laya-multilingual \
  --outtype f16 --outfile /tmp/laya-multilingual-F16.gguf
mesh-llm serve --gguf /tmp/laya-multilingual-F16.gguf
```

Other published Laya GGUFs may omit the decision head or use an incompatible
layout; a `.gguf` extension alone does not establish compatibility.

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
      | select(.id != "mesh" and .id != "auto")
      | select((.capabilities // []) | index("system_one"))
      | {id, display_name, system_one_status, metadata}'
```

Print only model IDs that are safe to send to `/systemone`:

```sh
curl -s http://127.0.0.1:9337/v1/models \
  | jq -r '.data[]
      | select(.id != "mesh" and .id != "auto")
      | select((.capabilities // []) | index("system_one"))
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

For Laya, replace `YOUR_DISCOVERED_LAYA_ID` below with its exact ID from
`/v1/models`. Do not assume the ID is the filename or download reference; it
can be a canonical Hugging Face ID or a content-hash ID. Object-valued `state`
is also accepted:

```sh
curl http://127.0.0.1:9337/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "YOUR_DISCOVERED_LAYA_ID",
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
