# OpenJEV on Skippy: proof-of-concept runbook

This branch adds a Jev-compatible `POST /systemone` route backed by one
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
curl http://127.0.0.1:9337/systemone \
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

## Canary coverage

`scripts/skippy-system-one-smoke.sh` is the llama.cpp upstream canary's System
One lane. It runs in two independent parts, because they have different
preconditions.

**Contract part (always runs).** It starts `serve-openai` on the pinned
`family-qwen3-dense` fixture and drives `POST /systemone` through the
fail-closed boundaries the frontend decides, which need no diffusion model and
therefore no particular accelerator:

- an unloaded model, empty questions, and choice/score criteria outside their
  documented bounds (`2..=26` and `2..=10`) are typed `invalid_request` errors;
- `images`, more than one `steps` or `samples`, `think`, and `sequential` reads
  are typed unsupported-feature errors rather than silently changing meaning;
- a non-`POST` method gets the method-not-allowed fallback;
- a well-formed read against a non-DiffusionGemma model is refused by the
  native runtime instead of being answered.

**Full-model read part (backend qualified only).** It loads
`unsloth/diffusiongemma-26B-A4B-it-GGUF` at `Q4_K_M` on exactly one runtime
lane and asserts `noul`, `choice`, `score`, and mixed-question answers: finite
probabilities inside `[0, 1]`, label distributions that sum to one, an answer
inside the declared label set, a reported score that is the expectation of its
own distribution, positive input tokens with no generated output tokens, and
the documented alias, whose requested model string the response echoes. It
then repeats a read, interleaves a different read, and
repeats the first again: the two identical reads must agree, and the two
different reads must differ. A leaked diffusion canvas or a cached answer
breaks one of those two.

Both artifacts are resolved through the shared test-model manifest contract
(`ci/model-artifacts/manifests/skippy-system-one-smoke.json`), which enforces
the authorized cadence and verifies the pinned revision, byte size, and
SHA-256 before load. A mismatch fails; it is never a skip.

The part that matters is admission. CUDA is the only backend this proof of
concept certifies, and the canary's `family-certify` runner builds Metal, so
the read part is admitted by declaration rather than by assuming whatever
accelerator is present:

```bash
# Run the contract part only, wherever a patched native build exists.
scripts/skippy-system-one-smoke.sh

# Admit the full-model read on a qualified backend with the pin warmed.
LLAMA_STAGE_BACKEND=cuda SYSTEMONE_SMOKE_BUILD_BACKEND=cuda \
  scripts/skippy-system-one-smoke.sh

# Pre-warm plan for the pinned artifact (the runner cache is offline and
# operator-owned, so no CI step downloads it).
scripts/skippy-system-one-smoke.sh --prewarm
```

When the read part is not admitted, the smoke records `unqualified`, emits a
`NOT CERTIFIED` job annotation, and exits zero — a visible gap, never a quiet
pass. `SYSTEMONE_SMOKE_REQUIRE_QUALIFIED=1` (or the
`LLAMA_CANARY_SYSTEMONE_REQUIRE_QUALIFIED` repository variable) turns that into
a hard failure once a qualified backend joins the pool.

The smoke is wired into the unchanged-pin certification, the changed-pin repair
gates, and the independent candidate verification, and a red contract part or a
red declared-qualified read blocks publication. It deliberately adds no
`ci/llama-canary/family-certified.json` row and no
`ci/llama-canary/generated-family-map.json` entry: it proves the read this
branch introduces without claiming that the diffusion canvas is
stage-distributed or that a family profile is certified.

## Split-serving boundary

The existing
`meshllm/diffusiongemma-26B-A4B-it-Q4_K_M-layers` package proves that the model
can be packaged for Skippy layer distribution, but this System One operation is
not yet stage-distributed. The endpoint intentionally rejects a staged model
instead of producing a partial read. The next engineering step is to carry the
diffusion canvas and zero-self-conditioning state across Skippy stages, then
compute the selected label logits on the terminal stage. Until that lands, a
full-model worker in the mesh is the shortest honest proof.

## Laya backend

[Laya](https://huggingface.co/convaiinnovations/laya-multilingual) is a
second System One backend behind the same `POST /systemone` route, request
validation, aliases, and answer mapping.

Both backends implement one `skippy_runtime::DecisionModel`: they take a
`DecisionRequest` (state plus typed `noul` / `choice` / `score` questions,
options in request order, and a per-request seed) and return one
distribution per question. Each backend builds its own native input behind
that interface — the DiffusionGemma answer canvas, or one Laya encoder
sequence per question — so the HTTP frontend holds only the Jev contract.
Moving that interface into the native ABI is a follow-up, planned after the
System One image work lands. It is a 322M-parameter mmBERT encoder
with a typed decision head: each question becomes one encoder sequence, and
every option is scored at its own `[MASK]` marker in a single forward pass.
There is no chat template, answer canvas, or text generation.

The native side ports the draft upstream support
([ggml-org/llama.cpp#29363](https://github.com/ggml-org/llama.cpp/pull/29363))
in one family patch (`model_support/0006`), which also exposes it through a
narrow Skippy ABI (`skippy/laya.h`, feature bit 40). Laya does
not load through `skippy_model_open`; the host recognizes
`general.architecture = "laya"` and opens it through its own entry point,
skipping KV and context planning. The runtime keeps its weights on the node's
first accelerator and schedules with the CPU backend behind it, so an op the
accelerator lacks falls back rather than failing.

### Convert a checkpoint

The upstream converter supports the `laya-multilingual` checkpoint. From a
prepared llama.cpp checkout (`just llama-prepare`):

```bash
hf download convaiinnovations/laya-multilingual --local-dir /tmp/laya-multilingual
python3 .deps/llama.cpp/convert_hf_to_gguf.py /tmp/laya-multilingual \
  --outtype f16 --outfile /tmp/laya-multilingual-F16.gguf
```

Other published Laya GGUFs use different layouts: the `mys/laya-*-GGUF` files
are built by the `ggmlc` compiler, and `fr0stbit3/laya-gguf` contains only the
encoder with the head in a separate file. Use a GGUF written by this converter.

### Serve and read

```bash
./target/debug/mesh-llm serve --gguf /tmp/laya-multilingual-F16.gguf --headless

curl http://127.0.0.1:9337/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "laya-multilingual-F16",
    "state": {"from": "user@example.com", "body": "I was charged twice this month."},
    "questions": {
      "is_billing": {"type": "noul", "instructions": "Is this a billing issue?"},
      "team": {
        "type": "choice",
        "instructions": "Which team should handle it?",
        "criteria": {"billing": "charges or refunds", "support": "technical troubleshooting"}
      }
    }
  }'
```

Use the model ID from `GET /v1/models`, or configure an `openjev-latest` alias
as in the DiffusionGemma setup above. The node advertises the `decision` workload class, so chat, completion,
embedding, and audio requests are never routed to it; `/systemone` routes by
model name as it does for DiffusionGemma.

### How it differs from the DiffusionGemma read

- **Order.** Choice options and object-valued `state` keep the order the
  request lists them, rendered with Python `json.dumps` separators, because
  that is how the reference implementation builds its sequences. The
  DiffusionGemma read keeps its existing sorted choice order.
- **Budgets.** Each question is capped at the GGUF's `max_len` tokens (1,024
  for `laya-multilingual`) with options sharing a `head_max_len` region; long
  state is truncated, as in the reference. A question can have at most 16
  options.
- **Usage.** `usage.input_tokens` counts every question sequence, since each
  question re-reads the state.
- **Action head.** The checkpoint's act/escalate head is computed but not
  returned; the Jev response has no field for it and the model card reports it
  carries little signal.
- **Qualification.** Unit tests cover sequence assembly, rendering, and
  answer mapping against a stub tokenizer. A live read against the converted
  checkpoint and the upstream golden fixtures is not yet part of the System One
  smoke.
