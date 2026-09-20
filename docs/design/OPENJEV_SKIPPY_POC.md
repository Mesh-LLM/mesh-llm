# OpenJEV on Skippy: proof-of-concept runbook

This branch adds a Jev-compatible `POST /v1/systemone` route backed by one
read-only DiffusionGemma denoise step in the patched llama.cpp/Skippy runtime.
It is a proof of concept, not a production compatibility claim.

## Fastest route to a live proof

Use the complete Q4_K_M GGUF on one CUDA worker that participates in a MeshLLM
mesh. Do not pass `--split` for the first proof. The System One native operation
currently requires the whole model and exactly one inference lane on that
worker.

The useful upstream artifacts are:

- OpenJEV server and protocol: <https://github.com/razorback16/openjev>
- NVIDIA DiffusionGemma checkpoint: <https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4>
- llama.cpp-compatible GGUF: <https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF>
- Existing MeshLLM layer package: <https://huggingface.co/meshllm/diffusiongemma-26B-A4B-it-Q4_K_M-layers>
- Upstream llama.cpp DiffusionGemma work: <https://github.com/ggml-org/llama.cpp/pull/24423>

OpenJEV documents a 24 GB NVIDIA minimum for its approximately 18 GB NVFP4
weights. Start with a 24 GB or larger CUDA GPU for this proof. The branch builds
the port through MeshLLM's native runtime pipeline, but a live Metal result has
not been certified.

## Build the branch

On the CUDA host:

```bash
git fetch origin codexy/openjev-skippy
git switch --detach origin/codexy/openjev-skippy
just build backend=cuda
```

`just build` creates the development host at `target/debug/mesh-llm` and places
the matching patched native runtime beside it.

## Configure one full-model worker

Create `/tmp/openjev-poc.toml`:

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

The micro-batch must be at least the GGUF's fixed diffusion canvas length. The
server queries that metadata at runtime; the published Q4_K_M model uses a
larger canvas than OpenJEV's vLLM default, so the branch does not hard-code 64.

Start a private mesh first:

```bash
./target/debug/mesh-llm serve \
  --config /tmp/openjev-poc.toml \
  --mesh-name openjev-poc \
  --headless
```

After the private proof works, add `--publish` to advertise the mesh through
MeshLLM discovery. Other nodes can join for routing and other models, but this
branch executes each System One read wholly on the DiffusionGemma worker.

## Exercise the Jev-compatible endpoint

```bash
curl http://127.0.0.1:9337/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openjev-latest",
    "state": "I was charged twice this month.",
    "questions": {
      "is_billing": {
        "type": "noul",
        "instructions": "Is this a billing issue?"
      },
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

The response shape is `{model, answers, usage}`. This proof supports text-only
`noul`, `choice`, and `score` questions, one read, and up to 26 choices. It
rejects images, thinking, multiple samples/steps, and sequential reads rather
than silently changing their meaning.

`usage.input_tokens` counts the tokenized prompt only. The fixed canvas tokens
the read computes over are not reported, and `output_tokens` stays `0` because
the endpoint generates no text.

## What the branch implements

1. It ports the draft llama.cpp DiffusionGemma architecture support onto the
   repository's pinned llama.cpp revision.
2. It adds a narrow Skippy ABI operation that accepts prompt tokens, a fixed
   answer canvas, label slots, and label token IDs.
3. Native code performs one zero-self-conditioning diffusion decode and returns
   a per-slot softmax restricted to the caller's declared labels.
4. The HTTP frontend formats Jev question types and maps those probabilities to
   Jev-compatible answers. No answer text is generated or parsed.

Guardrail screening applies to the chat and completion paths; the guarded OpenAI
backend forwards System One reads to the inner backend unscreened. This is
intentional for the PoC: `state` is consumed as structured read input, and the
endpoint never generates free-form text.

## Split-serving boundary

The existing
`meshllm/diffusiongemma-26B-A4B-it-Q4_K_M-layers` package proves that the model
can be packaged for Skippy layer distribution, but this System One operation is
not yet stage-distributed. The endpoint intentionally rejects a staged model
instead of producing a partial read. The next engineering step is to carry the
diffusion canvas and zero-self-conditioning state across Skippy stages, then
compute the selected label logits on the terminal stage. Until that lands, a
full-model worker in the mesh is the shortest honest proof.
