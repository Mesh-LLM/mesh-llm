# Virtual LLM Engine

Design for in-engine mesh hooks — making llama-server itself aware of the mesh so it can consult other models **during** inference, not just before or after.

Related: [#183](https://github.com/michaelneale/mesh-llm/issues/183), [#165](https://github.com/michaelneale/mesh-llm/issues/165)

---

## Why in-engine, not proxy-level

The proxy layer (`network/proxy.rs`) can rewrite requests before inference and inspect responses after. But it can't participate **during** token generation. Any proxy-level pipeline that needs to react to what the model is generating must buffer the response, cancel inference, reconstruct a new request, re-send, and stitch the output back together. This is expensive (re-prefill, multiple HTTP round-trips) and fragile (stream reconstruction, response format normalization).

By adding hooks inside llama-server's C++ token generation loop (`server-context.cpp`), the virtual LLM can inject context into the live KV cache, pause a slot while consulting the mesh, and resume generation — all without restarting inference or breaking the streaming connection.

| Capability | Proxy Level | In-Engine |
|---|---|---|
| Image captioning for text models | Rewrite request before inference (pre-flight) or cancel + re-send | Async fetch during prefill, inject caption into live KV cache. Zero restart. |
| Response verification | Buffer full response, send to verifier, potentially regenerate | Check at EOS, inject correction, continue generating in same slot. |
| Confidence-triggered help | Buffer tokens, score, cancel stream, re-route | Per-token logprob check, pause slot, consult mesh, inject, resume. Stream uninterrupted. |
| Context enrichment mid-generation | Cannot do during streaming | Inject tokens into live KV cache. Model's attention updates in-place. |
| Dynamic knowledge retrieval | Cannot do during streaming | Model generates uncertainty signal → hook detects → fetches from mesh → injects → model continues with the knowledge. |
| Seamless streaming to client | Must buffer, cancel, stitch SSE frames | Native — same slot, same stream, brief pause at most. |

## Hook points in llama-server

The token generation loop in `server-context.cpp` follows this path per slot per decode step:

```
llama_decode(batch)
  → common_sampler_sample() — produces a token
  → process_token()         — accumulates text, checks stop words, emits SSE chunk
  → stop condition check    — EOS, limit, time, stop word
  → send_partial_response() — pushes chunk to HTTP response queue
  ... or on stop:
  → send_final_response()   — pushes final result
  → slot.release()          — slot goes idle, KV cache retained
```

Hooks go at these points:

### 1. Pre-inference hook (`launch_slot_with_task`)

Fires once when a request is assigned to a slot, before any tokens are evaluated. At this point we have the full parsed request and can inspect messages for media, classify the request, and fire off async consultations.

**Available state**: full request body, messages, model capabilities, mesh peer list.

**Use cases**:
- Detect images in messages when the model has no vision → fire async caption request to mesh
- Detect request complexity → pre-fetch context from a smaller model
- Set up trigger patterns and thresholds for runtime hooks

**Callback to mesh**: `POST http://localhost:{mesh_port}/mesh/consult`

### 2. Per-token hook (`process_token`)

Fires after every sampled token. Must be fast for the common case (one branch check). Has access to the token, its probability, the full generated text so far, and the logit distribution.

**Available state**: `result.tok`, `result.prob`, `slot.generated_text`, `slot.n_decoded`, full logit distribution via `get_token_probabilities()`, KV cache via `slot.prompt`.

**Use cases**:
- Check if an async consultation (started at pre-inference) has a result ready → inject tokens into KV cache
- Monitor rolling logprob average → trigger mesh consultation on confidence drop
- Detect generated text patterns ("I cannot see the image", tool call structures) → trigger mesh consultation
- Inject retrieved context tokens into the live slot

**Callback to mesh**: same endpoint, or check `std::future` from async pre-fetch.

### 3. Pre-response hook (`send_final_response`)

Fires when generation is complete (EOS, stop word, limit) but before the response is sent to the HTTP client. Can inspect the full generated output and decide to augment, verify, or continue generating.

**Available state**: `slot.generated_text` (complete), all token probs, timings, stop reason.

**Use cases**:
- Send generated text to a verifier model in the mesh
- If verification fails: inject correction tokens, set `has_next_token = true`, continue generating
- Augment response with additional context from mesh
- Quality scoring and telemetry

**Callback to mesh**: synchronous HTTP to mesh-llm, since we're already at the end of generation.

### 4. Slot pause/resume (`SLOT_STATE_WAITING_MESH`)

A new slot state that means "this slot is alive, KV cache is warm, but we're waiting for a mesh consultation." The main loop skips paused slots during batching. Other slots continue. When the consultation result arrives, the slot transitions back to `SLOT_STATE_GENERATING`.

This lets synchronous consultations happen without blocking the entire server — only the requesting slot pauses.

## Communication: llama-server → mesh-llm

The hook makes HTTP requests back to mesh-llm on localhost. mesh-llm routes to the appropriate model anywhere in the mesh using existing routing, QUIC tunneling, and model discovery.

```
llama-server hook
  → HTTP POST localhost:{mesh_port}/mesh/consult
  → mesh-llm receives, routes to best model (local or remote peer)
  → inference happens on consulting model
  → result returned to llama-server hook
  → hook injects tokens into slot KV cache
  → generation continues
```

cpp-httplib (already linked in llama-server) handles the HTTP client side. mesh-llm exposes the consultation endpoint on its API port.

## Token injection

The core mechanism: adding tokens to a running slot's KV cache without restarting inference.

The existing codebase already does this in two places:
- **Speculative decoding**: adds draft tokens to the batch, evaluates them, rolls back rejected ones
- **Context shifting**: removes old tokens from KV cache, shifts positions, continues

Token injection for mesh hooks follows the same pattern:
1. Tokenize the consultation result (e.g., image caption text)
2. Add tokens to the slot's batch via `common_batch_add()`
3. Push tokens to `slot.prompt.tokens`
4. Evaluate the batch (`llama_decode`)
5. The KV cache now includes the injected content
6. Continue sampling — next token is conditioned on everything including the injection

The injected tokens don't appear in the SSE stream to the client. They're internal context.

## Request-level control

Hooks are enabled per-request via JSON body parameters, so the Rust side can decide which requests need virtual LLM behavior:

```json
{
  "mesh_hooks": true,
  "mesh_port": 3131,
  "mesh_caption_images": true,
  "mesh_confidence_threshold": -3.0,
  "mesh_confidence_window": 8,
  "mesh_verify_response": false,
  "mesh_triggers": ["[NEED_HELP]", "I cannot see"]
}
```

Requests that don't set `mesh_hooks: true` take the normal fast path with zero overhead.

## Changes required

### llama.cpp fork (C++)

- `server_slot`: add `mesh_hook_context` struct (mesh port, async state, trigger config, pending futures)
- `server_slot`: add `SLOT_STATE_WAITING_MESH` state
- `task_params`: parse `mesh_*` fields from request JSON
- `launch_slot_with_task`: pre-inference hook — analyze request, fire async consultations
- `process_token`: per-token hook — check async results, monitor confidence, detect triggers
- `send_final_response`: pre-response hook — optional verification/augmentation
- Token injection function using existing `common_batch_add` + `llama_decode` pattern

### mesh-llm (Rust)

- `inference/launch.rs`: pass `--mesh-port {port}` to llama-server
- New consultation API endpoint: `POST /mesh/consult` — routes to appropriate model in mesh
- Request preparation: when routing to a text model with images, set `mesh_caption_images: true`
- Virtual model configuration: which hooks are active for which request types
