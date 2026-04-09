# Virtual LLM Engine

Hooks inside llama-server's generation loop that call back to mesh-llm over localhost JSON. The inference engine can consult other models in the mesh during generation.

Related: [#183](https://github.com/michaelneale/mesh-llm/issues/183), [#165](https://github.com/michaelneale/mesh-llm/issues/165)

---

## How it works

llama-server gets four hooks. At each hook, it POSTs JSON to mesh-llm on localhost. mesh-llm decides what to do — consult another model, fetch context, verify, or do nothing. It replies with an action. llama-server executes it mechanically.

```
llama-server (C++)                      mesh-llm (Rust)
────────────────                        ───────────────
request arrives                         
  → Hook 1: POST /mesh/hook            → inspect request
                                        → "images + text model, I'll caption"
  ← { action: "inject", text: "..." }  ← routes to vision model in mesh
  inject into prompt, start prefill

prompt evaluated, first logits ready
  → Hook 2: POST /mesh/hook            → check entropy
                                        → "model confused, let me help"
  ← { action: "inject", text: "..." }  ← fetches context via mesh
  inject, re-evaluate, start generating

generation finishes
  → Hook 3: POST /mesh/hook            → verify response quality
                                        → "looks good"
  ← { action: "none" }                 
  send response to client

slot released
  → Hook 4: fire-and-forget telemetry  → log for routing feedback
```

### Why JSON over localhost

- **cpp-httplib already linked** in llama-server. axum already running in mesh-llm.
- **Latency is irrelevant** — the slow part is consulting another model (seconds), not the localhost call (~0.1ms).
- **Debuggable** — `curl localhost:3131/mesh/hook` to test.
- **Decoupled** — llama.cpp fork stays clean. No FFI, no shared memory, no Rust linking into C++.

---

## Hooks

Four hooks. One callback shape. The callback is always `POST http://localhost:{mesh_port}/mesh/hook`.

### Response format (same for all hooks)

```json
{
  "action": "none" | "inject" | "stop",
  "text": "The image shows a code snippet that...",
  "continue": false
}
```

- **`none`** — do nothing, proceed normally.
- **`inject`** — tokenize `text`, add to prompt/KV cache, continue. At Hook 3, if `continue: true`, resume generation after injection.
- **`stop`** — halt generation, release slot.

---

### Hook 1: Pre-inference

**When**: Request assigned to slot, before tokenization. Nothing has started.

**What it sends**:

```json
{
  "hook": "pre_inference",
  "slot_id": 0,
  "request_id": "chatcmpl-abc123",
  "model": "qwen3-32b",
  "messages": [
    {"role": "user", "content": "What's in this image?"},
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
    ]}
  ],
  "has_images": true,
  "has_audio": false,
  "model_capabilities": ["text", "code", "tool_use"]
}
```

The hook sends the **original messages array** (text + image/audio blocks), whether images/audio are present, and what the target model can do. mesh-llm already parses this from the request at routing time (`media_requirements()` in `router.rs`).

Images are sent as-is — the base64 data URLs from the original request. If mesh-llm needs to caption them, it forwards the image blocks to a vision model in the mesh.

**What mesh-llm can do**:
- **Caption images**: model is text-only but request has images → send images to vision model in mesh → return caption as inject text → caption becomes part of the prompt before prefill.
- **Pre-fetch context**: request mentions a topic → ask a small model in the mesh for relevant context → inject as system message.
- **Do nothing**: model can handle the request natively → return `none`.

**Cost**: Delays time-to-first-token. If mesh-llm returns `none` immediately (~0.1ms), negligible. If it captions an image, TTFT increases by the captioning time.

**C++ implementation**: In `launch_slot_with_task`, before tokenization. The original request JSON (`data`) is still available. POST it, parse response, if `inject` then prepend text to the prompt.

---

### Hook 2: Post-prefill

**When**: Prompt fully evaluated, first-token logits available, nothing streamed to client yet.

**What it sends**:

```json
{
  "hook": "post_prefill",
  "slot_id": 0,
  "request_id": "chatcmpl-abc123",
  "model": "qwen3-32b",
  "n_prompt_tokens": 847,
  "signals": {
    "first_token_entropy": 6.8,
    "first_token_margin": 0.02,
    "top_tokens": [
      {"text": "I",      "prob": 0.08},
      {"text": "The",    "prob": 0.06},
      {"text": "Based",  "prob": 0.06},
      {"text": "Sorry",  "prob": 0.05},
      {"text": "\n",     "prob": 0.04}
    ]
  }
}
```

The hook sends **first-token signals**: entropy and margin computed from the logit distribution after prompt evaluation, plus the top-N token candidates with probabilities. This tells mesh-llm whether the model knows how to start answering.

No message content sent — mesh-llm already has the request from Hook 1 (keyed by `request_id`). Just signals and token stats.

**What mesh-llm can do**:
- **High entropy (model confused)**: fetch context from another model, return inject → tokens added to KV cache, re-evaluate last position, model gets a better start.
- **Top token is refusal ("Sorry", "I")**: inject steering text → model starts differently.
- **Low entropy (model confident)**: return `none` → no delay.

**Cost**: Delays TTFT. But if entropy is high, the model was going to produce a bad response anyway.

**C++ implementation**: At the `SLOT_STATE_DONE_PROMPT → SLOT_STATE_GENERATING` transition. Compute entropy/margin from `get_token_probabilities(ctx, tok_idx)`. POST. If inject, add tokens, re-decode.

---

### Hook 3: Pre-response

**When**: Generation complete (EOS, stop word, limit). Full text available. For non-streaming: nothing sent yet. For streaming: partial chunks already sent, but final chunk not yet.

**What it sends**:

```json
{
  "hook": "pre_response",
  "slot_id": 0,
  "request_id": "chatcmpl-abc123",
  "model": "qwen3-32b",
  "generated_text": "The verify_token() function in auth.rs handles session validation by checking the JWT signature against the stored secret...",
  "n_decoded": 156,
  "stop_reason": "eos",
  "signals": {
    "mean_entropy": 2.4,
    "max_entropy": 7.1,
    "min_margin": 0.01,
    "uncertain_token_count": 12,
    "generation_time_ms": 3200
  }
}
```

The hook sends the **complete generated text** and **summary signal stats** over the full generation — mean/max entropy, min margin, how many tokens had high uncertainty, total time.

**What mesh-llm can do**:
- **Verify**: send generated text to a verifier model → if bad, return `inject` + `continue: true` → correction appended, generation resumes in the same slot.
- **Score quality**: record signal stats for routing feedback (next time, pre-fetch context for this kind of question).
- **Pass through**: return `none` → response sent as-is.

**Cost**: Delays final response. For non-streaming, the full response is held. For streaming, only the final chunk/metadata.

**C++ implementation**: In `send_final_response`, before pushing the result. POST. If inject + continue, add tokens, set `has_next_token = true`, return to generation loop.

---

### Hook 4: Complete (telemetry)

**When**: Response sent, slot released. Fire-and-forget.

**What it sends**:

```json
{
  "hook": "complete",
  "slot_id": 0,
  "request_id": "chatcmpl-abc123",
  "model": "qwen3-32b",
  "n_prompt_tokens": 847,
  "n_decoded": 156,
  "generation_time_ms": 3200,
  "hooks_fired": ["pre_inference"],
  "hooks_injected": ["pre_inference"],
  "hook_time_ms": 450,
  "signal_summary": {
    "mean_entropy": 2.4,
    "max_entropy": 7.1,
    "min_margin": 0.01
  }
}
```

**What mesh-llm can do**: Learn patterns. "Requests about auth.rs from qwen3-32b triggered captioning injection 80% of the time → pre-fetch faster next time." No action returned.

**C++ implementation**: Detached thread, `std::thread([...]{ client.Post(...); }).detach();`

---

## What about mid-generation?

No per-token hook. Token generation runs at 30-100+ tok/s — calling back every token is wasteful. Instead:

**Signal computation runs every token in C++ (cheap, no callback).** The slot maintains a rolling window of entropy and margin values. This is pure arithmetic — microseconds per token, no HTTP, no allocation.

**The signals feed into Hook 2 and Hook 3.** Post-prefill gets first-token signals. Pre-response gets summary stats over the full generation.

**If we later want mid-generation intervention**, there are two options that don't involve per-token callbacks:

1. **Threshold breakout**: C++ monitors the rolling signal window. When `uncertain_streak > N` (configurable), it fires a one-time callback to mesh-llm. This is a rare event (maybe 0-2 times per generation), not per-token. The slot pauses, mesh-llm consults, injects, slot resumes.

2. **Async poll**: Hook 1 starts an async consultation. C++ polls a lightweight endpoint (`GET /mesh/hook/async/{id}`) every K tokens (not every token — maybe every 8 or 16). When the result is ready, inject it. Between polls, pure fast-path generation.

Both are optional and can be added later. The four hooks above are the foundation.

---

## Signals

Computed per-token in C++ (no callback). Summarized and sent at Hook 2 and Hook 3.

### Entropy

```
H = -Σ p_i × log₂(p_i)
```

From the softmax distribution that `get_token_probabilities()` already computes.

- Low (~1-2 bits): confident, peaked.
- High (~8+ bits): uncertain, spread.

### Margin

```
margin = p_top1 - p_top2
```

Two lookups from the sorted probability array.

- Large (>0.3): clear winner.
- Small (<0.05): coin flip.

### Rolling window

```cpp
struct mesh_signal_window {
    static constexpr int SIZE = 16;
    float entropy[SIZE];
    float margin[SIZE];
    int   pos = 0;
    int   count = 0;

    float entropy_mean = 0;
    float entropy_max  = 0;
    float margin_min   = 1;
    int   uncertain_streak = 0;  // consecutive high-entropy tokens
    int   uncertain_count  = 0;  // total high-entropy tokens
};
```

Updated per token. Sent as summary at Hook 3 (pre-response).

### Self-consistency (request-level)

Not a per-token signal. mesh-llm sets `n_cmpl: 3` on the request. llama-server prefills once, copies KV to 3 child slots (`copy_state_to`), generates 3 completions. At Hook 3, mesh-llm gets all 3 and measures agreement. Uses existing `n_cmpl` machinery — no C++ changes.

---

## Token injection

When a hook returns `inject`:

1. Tokenize the text: `common_tokenize(vocab, text, false, true)`
2. Add tokens to batch: `common_batch_add(batch, tok, pos, {slot.id}, true)`
3. Push to `slot.prompt.tokens`
4. Evaluate: `llama_decode(ctx, batch)` — KV cache updated in-place
5. Continue from new state

At **Hook 1** (pre-inference): injection happens before tokenization — the text is prepended to the prompt as a system message. No KV manipulation needed, it's just part of the prompt.

At **Hook 2** (post-prefill): injection goes into the live KV cache. The model has already read the prompt; now it reads the injected context too. Next sample is conditioned on prompt + injection.

At **Hook 3** (pre-response): injection appended after the generated text. If `continue: true`, generation resumes — model continues from the correction.

Injected tokens are invisible to the client's SSE stream.

---

## Configuration

Hooks enabled per-request via JSON body:

```json
{
  "mesh_hooks": true,
  "mesh_port": 3131
}
```

When `mesh_hooks` is absent or false, all hook code is skipped — zero overhead.

mesh-llm sets these fields when forwarding requests to llama-server, based on its routing decision. A text model receiving a request with images gets `mesh_hooks: true`. A vision model handling the same request doesn't need hooks.

---

## C++ changes

**On `server_slot`**: Add `mesh_hook_ctx` — mesh port, httplib client, signal window, request ID.

**On `task_params`**: Add `mesh_hooks` (bool) and `mesh_port` (int). Parsed from request JSON.

**New slot state**: `SLOT_STATE_WAITING_MESH` — slot paused during sync callback. Other slots continue.

**Hook insertion points**:
1. `launch_slot_with_task` — after task assignment, before tokenization
2. `DONE_PROMPT → GENERATING` transition — after prompt eval, before first sample
3. `send_final_response` — before pushing final result
4. `slot.release` — fire-and-forget telemetry

**Signal computation**: In the generation loop after `common_sampler_sample`, update rolling window. No callback, just arithmetic.

## Rust changes

**New endpoint**: `POST /mesh/hook` on the management API (port 3131).

**New module**: `inference/virtual.rs` — decision logic for each hook type.

**`inference/launch.rs`**: Pass `--mesh-port {api_port}` to llama-server.

**`network/proxy.rs`**: Set `mesh_hooks: true` on requests that need it (images + text model, etc.).
