# Virtual LLM Engine

Callback hooks from llama-server into mesh-llm during inference. The model being served can ask the mesh for help — caption images, fetch context, verify its output — without the caller knowing.

Related: [#183](https://github.com/michaelneale/mesh-llm/issues/183), [#165](https://github.com/michaelneale/mesh-llm/issues/165)

---

## Callback protocol

llama-server POSTs JSON to mesh-llm on localhost at defined hook points during inference. mesh-llm decides what to do and replies with an action.

**Request** (llama-server → mesh-llm):
```
POST http://localhost:{mesh_port}/mesh/hook
Content-Type: application/json

{
  "hook": "pre_inference" | "post_prefill" | "pre_response" | "complete",
  ... hook-specific fields ...
}
```

**Response** (mesh-llm → llama-server):
```json
{
  "action": "none" | "inject" | "stop" | "pending",
  "text": "...",
  "continue": false,
  "async_id": "cap-001"
}
```

**Actions**:
- `none` — do nothing, continue normally
- `inject` — tokenize `text`, add to prompt or KV cache, continue
- `stop` — halt generation, release slot
- `pending` — mesh-llm has started background work. Generation proceeds without blocking. llama-server stores `async_id` and polls for the result during generation (see [Async hooks](#async-hooks))

---

## Hooks

### 1. Pre-inference

Fires when a request is assigned to a slot, before tokenization or prefill. Nothing has started. The full original request is available.

**Callback data**:
```json
{
  "hook": "pre_inference",
  "request_id": "chatcmpl-abc123",
  "model": "qwen3-32b",
  "model_capabilities": ["text", "code", "tool_use"],
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
      {"type": "text", "text": "What's in this image?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
    ]}
  ],
  "has_images": true,
  "has_audio": false
}
```

Sends the original messages array including image/audio data URLs, model capabilities, and media flags.

**Example — image captioning**:

Request has images but the model is text-only. mesh-llm detects the mismatch, forwards the image blocks to a vision model in the mesh, gets a caption back:

```json
{
  "action": "inject",
  "text": "[Image description: A Python code snippet showing a recursive fibonacci function with a bug on line 3 where the base case returns n instead of 1]"
}
```

The caption is prepended to the prompt as a system message before tokenization. The text model generates with the caption in context from the start.

**Example — context pre-fetch (sync)**:

Request asks about a specific topic. mesh-llm uses a small fast model to summarize relevant context. This blocks — generation waits for the context:

```json
{
  "action": "inject",
  "text": "Context: auth.rs contains three functions: verify_token() for JWT validation, check_session() for session lookup, and refresh_auth() for token renewal."
}
```

**Example — image captioning (async)**:

Request has images but captioning takes a few seconds. mesh-llm starts the vision model in the background and tells llama-server to proceed — the caption will arrive during generation:

```json
{
  "action": "pending",
  "async_id": "cap-001"
}
```

Generation starts immediately. llama-server polls for `cap-001` during generation (see [Async hooks](#async-hooks)). When the caption arrives, it's injected into the live KV cache.

**Example — no action needed**:

Model has vision capability and can handle the images directly:

```json
{ "action": "none" }
```

**Where in C++**: `launch_slot_with_task()`, after task assignment, before tokenization. The parsed request JSON (`data`) is still available.

**Blocking**: This hook can return `inject` (sync, blocks until response) or `pending` (async, returns immediately). mesh-llm decides based on how long the consultation will take.

---

### 2. Post-prefill

Fires after the prompt is fully evaluated but before the first token is sampled. The model has "read" the prompt — first-token logits are available. Nothing has been streamed to the client yet.

**Callback data**:
```json
{
  "hook": "post_prefill",
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

Sends first-token entropy (how spread out the distribution is), margin (gap between top two candidates), and the top-5 candidates. No message content — mesh-llm already has the request from Hook 1, keyed by `request_id`.

**Signals explained**:
- **Entropy**: `H = -Σ p_i × log₂(p_i)`. Low (~1-2 bits) = confident. High (~8+ bits) = confused.
- **Margin**: `p_top1 - p_top2`. Large (>0.3) = clear winner. Small (<0.05) = coin flip.

**Example — model is confused**:

Entropy is 6.8, margin is 0.02 — the model doesn't know how to start. mesh-llm fetches context from another model in the mesh:

```json
{
  "action": "inject",
  "text": "The user is asking about the authentication flow. The relevant code is in auth.rs which uses JWT tokens validated by the verify_token function."
}
```

Tokens are injected into the KV cache and re-evaluated. The model now has context to work with. First token sampled from the updated state.

**Example — model is about to refuse**:

Top token is "Sorry" with 0.05 probability. mesh-llm injects a steering nudge:

```json
{
  "action": "inject",
  "text": "Answer the question directly based on your knowledge."
}
```

**Example — model is confident**:

Entropy is 1.4, margin is 0.45 — model knows exactly what to say:

```json
{ "action": "none" }
```

**Where in C++**: The `SLOT_STATE_DONE_PROMPT → SLOT_STATE_GENERATING` transition. Entropy and margin computed from `get_token_probabilities(ctx, tok_idx)`.

**Blocking**: Always sync. mesh-llm just looks at the numbers and returns immediately — no model consultation at this hook, just a fast decision. ~0.1ms.

---

### 3. Pre-response

Fires when generation is complete (EOS, stop word, token limit). Full generated text available. For non-streaming requests, nothing has been sent to the client. For streaming, partial chunks already went out but the final chunk hasn't.

**Callback data**:
```json
{
  "hook": "pre_response",
  "request_id": "chatcmpl-abc123",
  "model": "qwen3-32b",
  "generated_text": "The verify_token() function in auth.rs handles session validation by checking the JWT signature...",
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

Sends the complete generated text and summary stats over the whole generation: mean and max entropy, minimum margin, how many tokens were uncertain, total time.

**Example — verification**:

mesh-llm sends the generated text to a verifier model. The verifier flags an issue. mesh-llm returns a correction:

```json
{
  "action": "inject",
  "text": "\n\nCorrection: verify_token() checks the JWT expiry, not the signature. Signature validation is handled by check_session().",
  "continue": true
}
```

With `continue: true`, llama-server injects the correction tokens, sets `has_next_token = true`, and resumes generation. The model continues from the correction and produces an updated conclusion. For streaming, this appears as more content arriving after a brief pause.

**Example — quality scoring (no intervention)**:

Signal stats look fine (low entropy, good margin). mesh-llm logs the stats for routing feedback but takes no action:

```json
{ "action": "none" }
```

**Where in C++**: `send_final_response()`, before pushing the result to the HTTP response queue.

**Blocking**: Sync — holds the final response until mesh-llm replies. If verification is enabled, this is where the cost goes. If mesh-llm returns `none`, ~0.1ms.

---

### 4. Complete (telemetry)

Fires after the response is fully sent and the slot is released. Fire-and-forget — no response expected.

**Callback data**:
```json
{
  "hook": "complete",
  "request_id": "chatcmpl-abc123",
  "model": "qwen3-32b",
  "n_prompt_tokens": 847,
  "n_decoded": 156,
  "generation_time_ms": 3200,
  "hooks_fired": ["pre_inference", "post_prefill"],
  "hooks_injected": ["pre_inference"],
  "injection_time_ms": 450,
  "signal_summary": {
    "mean_entropy": 2.4,
    "max_entropy": 7.1,
    "min_margin": 0.01,
    "uncertain_token_count": 12
  }
}
```

**What mesh-llm does**: Learns patterns. "Requests about auth.rs from qwen3-32b needed captioning 80% of the time → pre-fetch faster next time." Feeds routing decisions for future requests.

**Where in C++**: `slot.release()`. Runs in a detached thread so it doesn't block anything.

---

## Signal computation

Entropy and margin are computed per-token inside the C++ generation loop — pure arithmetic on the softmax distribution that `get_token_probabilities()` already produces. No callback, no allocation, ~1μs per token.

The slot maintains a rolling window:

```cpp
struct mesh_signal_window {
    static constexpr int SIZE = 16;
    float entropy[SIZE];
    float margin[SIZE];
    int   pos = 0;
    int   count = 0;

    // derived stats, updated on each push
    float entropy_mean;
    float entropy_max;
    float margin_min;
    int   uncertain_streak;  // consecutive tokens above entropy threshold
    int   uncertain_count;   // total uncertain tokens
};
```

These stats are sent as summaries at Hook 2 (first-token only) and Hook 3 (full generation summary). They're never sent per-token.

### Self-consistency

A request-level strategy, not a per-token signal. mesh-llm sets `n_cmpl: 3` on the request. llama-server prefills once, copies KV cache to 3 child slots via `copy_state_to()`, generates 3 independent completions. At Hook 3, mesh-llm receives all 3 and measures agreement. Convergence = correct. Divergence = uncertain.

Uses existing `n_cmpl` machinery — no new C++ code for the generation part.

---

## Token injection

How `inject` works at each hook:

**Hook 1 (pre-inference)**: Text is prepended to the prompt as a system message. Happens before tokenization. No KV cache manipulation — it's just part of the prompt.

**Hook 2 (post-prefill)**: Tokens are added to the live KV cache via `common_batch_add()` + `llama_decode()`. The model's attention now includes the injected content. Same pattern as speculative decoding (adding draft tokens) and context shifting (removing/shifting tokens). Next token is sampled from the updated state.

**Hook 3 (pre-response)**: If `continue: true`, tokens are appended after the generated text in the KV cache. `has_next_token` set to true, generation resumes — model continues from the injection.

Injected tokens are invisible to the client's SSE stream.

---

## Async hooks

When mesh-llm returns `action: "pending"` with an `async_id`, it's saying: "I've started working on something in the background. Don't wait for me — keep generating. I'll have the result ready soon."

llama-server stores the `async_id` and polls for the result during generation. Not every token — every N tokens (e.g., every 16). The poll is a lightweight GET:

```
GET http://localhost:{mesh_port}/mesh/hook/poll/{async_id}

→ 202 (not ready yet)
→ 200 { "action": "inject", "text": "..." }
```

When the result arrives (200), llama-server injects the text into the live KV cache and continues generating. The model's subsequent tokens are conditioned on the injected content.

**Example flow — async image captioning**:

```
Token  0: Hook 1 fires → mesh-llm returns { pending, async_id: "cap-001" }
                          mesh-llm starts captioning via vision model in mesh...
Token  1-15: generating normally, no poll
Token 16: poll GET /mesh/hook/poll/cap-001 → 202 (not ready)
Token 17-31: generating normally
Token 32: poll GET /mesh/hook/poll/cap-001 → 200 { inject, text: "Image shows..." }
           → inject 50 tokens into KV cache
Token 33+: generating conditioned on prompt + 32 generated tokens + caption
```

TTFT cost: zero. The model starts generating immediately. There's a brief stall at injection (evaluating the injected tokens through the model). The first 32 tokens were generated without the caption — they may be suboptimal, but the model often self-corrects once the context arrives because its attention now includes the injection.

**Poll cost**: One GET every 16 tokens. At 50 tok/s that's ~3 polls/second. Each returns 202 in <0.1ms when nothing is ready. Negligible.

**Multiple async**: A request can have multiple pending async_ids (e.g., two images being captioned). Each is polled independently.

### When to use sync vs async

| Situation | Mode | Why |
|---|---|---|
| Image caption for text-only model, fast vision model available | **Sync** at Hook 1 | Caption is essential, fast enough to not hurt TTFT much |
| Image caption, slow or remote vision model | **Async** from Hook 1 | Don't block TTFT, inject when ready |
| Context pre-fetch, small fast model | **Sync** at Hook 1 | Quick, and the context is important from token 1 |
| Context pre-fetch, large/slow model | **Async** from Hook 1 | Start generating, inject context when it arrives |
| First-token confidence check | **Sync** at Hook 2 | Just number-crunching, returns in <1ms |
| Response verification | **Sync** at Hook 3 | Need the verdict before sending to client |

---

## llama.cpp fork changes

### New fields on `server_slot`

```cpp
struct mesh_hook_ctx {
    bool enabled = false;
    int  port = 0;
    std::string request_id;
    json original_request;  // stored at Hook 1 for reference

    mesh_signal_window signals;

    // async polling state
    std::vector<std::string> pending_async_ids;
    int tokens_since_last_poll = 0;
    int poll_interval = 16;  // poll every N tokens

    std::unique_ptr<httplib::Client> client;

    void init(int mesh_port) {
        client = std::make_unique<httplib::Client>("localhost", mesh_port);
        client->set_connection_timeout(0, 100000); // 100ms connect
        client->set_read_timeout(30);              // 30s read
    }

    bool should_poll() {
        if (pending_async_ids.empty()) return false;
        return ++tokens_since_last_poll >= poll_interval;
    }
};
```

### New fields on `task_params`

```cpp
bool mesh_hooks = false;
int  mesh_port  = 3131;
```

Parsed from request JSON alongside existing fields (`cache_prompt`, `n_predict`, etc.).

### New slot state

```cpp
SLOT_STATE_WAITING_MESH  // paused during sync callback, other slots continue
```

### New CLI flag

```
--mesh-port PORT    Port for mesh hook callbacks (enables hook system)
```

When set, hooks are enabled for requests that include `mesh_hooks: true`.

### Hook insertion points

| Hook | Location | Code point |
|---|---|---|
| pre_inference | `launch_slot_with_task()` | After task assignment, before tokenization |
| post_prefill | `update_slots()` | At `DONE_PROMPT → GENERATING` transition |
| pre_response | `send_final_response()` | Before pushing result to queue |
| complete | `slot.release()` | After response sent, detached thread |

### Generation loop

In the generation loop, after `common_sampler_sample()` and `common_sampler_accept()`:

```cpp
if (slot.mesh_hook.enabled) {
    // 1. Update signal stats (always, cheap)
    auto probs = get_token_probabilities(ctx, tok_idx);
    float entropy = compute_entropy(probs);
    float margin = probs[0].p - probs[1].p;
    slot.mesh_hook.signals.push(entropy, margin);

    // 2. Poll for async results (every N tokens, lightweight GET)
    if (slot.mesh_hook.should_poll()) {
        slot.mesh_hook.tokens_since_last_poll = 0;
        for (auto it = slot.mesh_hook.pending_async_ids.begin();
             it != slot.mesh_hook.pending_async_ids.end(); ) {
            auto res = slot.mesh_hook.client->Get("/mesh/hook/poll/" + *it);
            if (res && res->status == 200) {
                auto body = json::parse(res->body);
                if (body["action"] == "inject") {
                    inject_tokens(slot, body["text"].get<std::string>());
                }
                it = slot.mesh_hook.pending_async_ids.erase(it);
            } else {
                ++it;  // 202 = not ready, keep polling
            }
        }
    }
}
```

Signal update is pure arithmetic (~1μs). Async poll fires every 16 tokens and returns in <0.1ms when nothing is ready.

---

## mesh-llm changes

### New API endpoints

On the management API port (3131):

- `POST /mesh/hook` — receives hook callbacks, runs decision logic, returns action. For `pending` responses, starts background work and stores the async_id.
- `GET /mesh/hook/poll/{async_id}` — lightweight poll. Returns 202 if not ready, 200 with action if ready.

### New module: `inference/virtual.rs`

Decision engine for each hook type:
- Pre-inference: detect media mismatches, decide sync inject vs async pending
- Post-prefill: evaluate first-token signals, decide if context needed
- Pre-response: optionally verify, score, or correct
- Complete: record telemetry, update routing heuristics

Manages async consultations: spawns tokio tasks for background model calls, stores results keyed by async_id, serves poll responses.

Consultation requests routed through existing mesh infrastructure — same model discovery, QUIC tunneling, and routing that normal requests use.

### `inference/launch.rs`

Pass `--mesh-port {api_port}` when spawning llama-server.

### `network/proxy.rs`

When forwarding a request to llama-server, set `mesh_hooks: true` in the JSON body when hooks should be active (e.g., images present + text-only model, or complex request that might benefit from verification).
