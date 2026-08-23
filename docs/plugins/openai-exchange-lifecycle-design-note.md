# Design note: exposing OpenAI-exchange lifecycle to an out-of-process plugin (#1331)

## Status

Reference implementation of the narrowest real slice, staged on this branch only
(`mesh1331-lifecycle-hooks`, off `StevenMih/mesh-llm`). No upstream PR opened — this is
meant to give the #1331 discussion something concrete to react to, not a finished
feature. Scope: non-streaming chat completions only. Streaming, delegation, and the
rest of #1331's acceptance list are explicitly deferred.

## What #1331 assumed vs. what's actually there

#1331's framing (`OpenAiHookPolicy` only fires before-chat; `MeshEvent` is
topology-only; the real path is somewhere in `inference::provider()`) is close but
imprecise about the codebase, and the imprecision matters for design:

- `OpenAiHookPolicy` (`crates/openai-frontend/src/hooks.rs`) is correct: today it has
  `before_chat_completion`, `after_prefill`, `mid_generation` — no terminal hook.
- `MeshEvent` doesn't exist as a single type. There are two distinct things this could
  mean:
  - `mesh/node.rs::MeshEvent` — genuinely topology-only (peer join/leave/health).
  - `plugin::types::PluginMeshEvent` (`crates/mesh-llm-host-runtime/src/plugin/types.rs:10`)
    — the actual out-of-process plugin transport (`Channel`, `BulkTransfer`,
    `OpenStream`, carried over a Unix-domain-socket/named-pipe envelope protocol in
    `plugin/transport.rs`). This is general-purpose plugin IPC, not topology-only, but
    it has **no existing variant for an OpenAI chat-completion event** — extending it
    was out of scope for this milestone (see "Deliberately deferred").
- There is no `inference::provider()` function. The closest real things are
  `PluginManager::inference_endpoint_for_model` and `PluginManager::provider_for_capability`
  (`crates/mesh-llm-host-runtime/src/plugin/mod.rs:808,827`), used from the **raw TCP
  ingress proxy** (`crates/mesh-llm-host-runtime/src/network/openai/ingress.rs:582`).

That last point is the important correction: **there are two disjoint real dispatch
paths for an OpenAI-shaped request, and only one of them is `OpenAiHookPolicy`.**

1. **The `openai-frontend` crate path** — a typed Rust API
   (`ChatCompletionRequest`/`ChatCompletionResponse`, `OpenAiBackend` trait). This is
   what `#1331` names and what this milestone extends. It's used when a model is served
   in-process (e.g. the embedded/skippy backend, see
   `crates/mesh-llm-host-runtime/src/inference/skippy/hooks.rs`'s `MeshAutoHookPolicy`,
   a real, shipping `OpenAiHookPolicy` implementor).
2. **The raw-proxy ingress path** (`network/openai/ingress.rs`) — used when a model is
   served by a plugin's inference endpoint. This path never deserializes the body into
   `ChatCompletionRequest`; it forwards HTTP bytes directly to
   `endpoint.address` after resolving `plugin_manager.inference_endpoint_for_model(model)`.
   `OpenAiHookPolicy` is never invoked on this path, and it doesn't share a request type
   with the frontend crate.

**Any #1331 design that extends only `OpenAiHookPolicy` covers path 1 and misses path 2
entirely.** Plugin-served models — the case #1331 most plausibly cares about, since
that's who the "out-of-process plugin" observer would usually be — are dispatched by a
completely different, byte-oriented code path that has no typed request/response hook
surface today. Covering path 2 would mean instrumenting the raw proxy (likely via its
existing `OpenAiRouteObserver`/`route_selected_with_metadata`
(`crates/mesh-llm-host-runtime/src/logging/openai_lifecycle.rs:296`), which is already a
metadata-only, in-process observer wired to that exact seam) — a separate, larger design
than what's here.

## What this milestone implements

In `crates/openai-frontend/src/hooks.rs`, `OpenAiHookPolicy` gained two new default
no-op async methods (additive, so `MeshAutoHookPolicy` and any other existing
implementor keep compiling unchanged):

- `on_effective_chat_completion(&self, request: &ChatCompletionRequest, route: &ChatExchangeRoute)`
  — fires once, immediately before `HookedOpenAiBackend` dispatches to the real
  backend, with the **post-mutation** request (i.e. after `before_chat_completion`'s
  outcome has been applied) and a `ChatExchangeRoute { model }`.
- `on_chat_completion_terminal(&self, request: &ChatCompletionRequest, outcome: &ChatCompletionOutcome<'_>)`
  — fires exactly once per non-streaming call, with one of `Success { response }`,
  `Error { status, message }` (the backend failed), or `Denied { status, reason }`
  (`before_chat_completion` itself returned `Err`, so the backend was never called).

Both are wired into `HookedOpenAiBackend::chat_completion_with_context` — the one real
call site the live router (`router.rs`'s `chat_completions` handler) actually dispatches
non-streaming chat completions through. `chat_completion` already delegates to
`chat_completion_with_context`, so it's covered for free. `chat_completion_stream` is
untouched, per this milestone's scope.

### Why `route` is just a model string

`OpenAiRequestContext` (`crates/openai-frontend/src/backend.rs:66`) carries no
backend/route identity, and `HookedOpenAiBackend` wraps exactly one already-chosen
`Arc<dyn OpenAiBackend>` — there is no per-request backend selection inside
`openai-frontend` to report. `request.model` is the only route-relevant fact available
at this layer. Route/provider selection (which plugin, which endpoint) happens entirely
on path 2 above, outside this crate.

### Why this stays in-process here

`openai-frontend` has a standing architectural invariant, enforced by its own test
(`lifecycle.rs`'s `manifest_has_no_host_runtime_dependency`), that it never depends on
`mesh-llm-host-runtime`. So this crate cannot itself dial the out-of-process plugin
transport (`PluginMeshEvent`/`plugin/transport.rs`) — that bridging necessarily happens
one layer up, in `mesh-llm-host-runtime`, exactly the way
`OpenAiLifecycleLoggingAdapter` (`crates/mesh-llm-host-runtime/src/logging/openai_lifecycle.rs:388`)
already bridges the existing metadata-only `OpenAiLifecycleObserver` events to the
logging service today. A production implementation of these two new hook methods would
follow that same pattern: implement `OpenAiHookPolicy` in `mesh-llm-host-runtime`,
serialize `(request, route)` / `(request, outcome)` into a
`proto::ChannelMessage { channel: "openai.exchange.v1", body, content_type:
"application/json", .. }`, and send it via the existing `PluginMeshEvent::Channel`
transport (`plugin/types.rs:10`) to any plugin subscribed to that channel. Building that
bridge, plus the channel-declaration/subscription plumbing, is the natural milestone 2.

## Can/can't verdict: the rung-ladder response leg (acknowledged_receipt → full_bilateral)

Steven's question: could the terminal/response hook (a) write an `X-Capsule-Id`-style
value into the response, and (b) surface the terminal event, so a plugin could later
observe the client's signed ack (over `capsule_id`+`nonce`) that lifts
`acknowledged_receipt` → `full_bilateral`? This is the actual point of the response leg
— an assurance rung can't be lifted without a hook that reaches both directions (server
→ client marker, and client → server ack). Checked against source, not assumed:

**(a) Writing an `X-Capsule-Id`-style value into the response: cannot, on this seam, for
two independent reasons.**

1. `on_chat_completion_terminal`'s `Success` variant carries `response: &'a
   ChatCompletionResponse` — an **immutable** reference, by design (this milestone's hooks
   are observe-only; see "What this milestone implements" above). Even a policy that
   wanted to stamp a capsule id has no write access to the response through this call.
2. Even with a mutable reference, there is nowhere on `ChatCompletionResponse` to put it.
   The struct (`crates/openai-frontend/src/chat.rs:270-278`: `id, object, created, model,
   choices, usage, timings`) has no open field — contrast `ChatCompletionRequest`, which
   does carry a `pub extra: BTreeMap<String, Value>` bag (`chat.rs:45`) that
   `chat_mesh_hooks_enabled`/`set_chat_mesh_hooks_enabled` already read/write. Nothing
   equivalent exists on the response type today.
3. More fundamentally, a real `X-Capsule-Id` would ride as an **HTTP response header**,
   the same way `x-request-id` does — and that header is attached by
   `frontend_lifecycle_middleware` (`router.rs:849-866`), an axum middleware layer that
   wraps the *entire* `Response` **after** the handler (and therefore after
   `HookedOpenAiBackend` and every `OpenAiHookPolicy` call) has already run
   (`json_response_with_usage`, `router.rs:774`, is what turns the typed
   `ChatCompletionResponse` into that `Response`, with no hook in between). `OpenAiBackend`
   and `OpenAiHookPolicy` never see an `axum::http::Response` or its headers at all — that
   capability lives one layer up, at the same place `x-request-id` is set, not inside this
   milestone's hook surface.

**(b) Surfacing the terminal event: partially — surfaces the exchange's own completion,
not the later ack.** `on_chat_completion_terminal` already fires exactly once per
non-streaming request with `Success`/`Error`/`Denied`, so a bridged plugin would learn
"this exchange just completed" at the right moment. But it cannot surface a client ack
for two reasons that are architectural, not just unimplemented: no capsule id exists to
correlate against (per (a)), and — separately — the client's ack is necessarily a
**different, later HTTP request** (there is no wire mechanism, in this exchange's
response, for the client to attach anything to *this* call). `OpenAiHookPolicy` and
`HookedOpenAiBackend` are scoped to one request's lifecycle; nothing in `openai-frontend`
threads state from one request to a later, unrelated one. Observing the ack would need a
new endpoint (or a recognized field on an existing one) plus a correlation store, neither
of which exists.

**Verdict: can't, today, on the `OpenAiHookPolicy`/`HookedOpenAiBackend` seam this
milestone extends.** Both halves of the response leg need capability that lives outside
it: (a) needs a seam at the HTTP middleware/response-header layer (the
`frontend_lifecycle_middleware` layer, not `OpenAiBackend`), and (b) needs cross-request
correlation that no part of this crate provides. A future design for the response leg is
closer to "add a second, header-capable hook point next to
`frontend_lifecycle_middleware`, and give `ChatCompletionResponse` an extensible field the
way the request already has one" than to "extend the existing terminal hook" — that's a
materially different, larger change than this milestone, and it's the concrete next
question for whoever picks up the rung-ladder work.

## Mapping #1331's acceptance criteria

| #1331 acceptance item | This milestone |
| --- | --- |
| Effective request + selected route before dispatch | ✅ for path 1 (in-process typed hook); ❌ not available at all for path 2 (raw proxy never has a typed request) |
| Terminal event (success/error/denial) | ✅ for path 1, non-streaming; ❌ streaming (deferred); ❌ path 2 (only status-code-level metadata via `OpenAiRouteObserver`, no plugin delivery) |
| Out-of-process plugin observes it | ⚠️ not yet — this milestone proves the in-process hook seam; the out-of-process hop is designed above but not built (see "Why this stays in-process here") |
| Streaming / delegation | ❌ explicitly out of scope this milestone |
| Rung-ladder response leg (capsule-id marker + client-ack observation) | ❌ not on this seam — see verdict above; needs a header-capable response hook + response extensibility + cross-request correlation, none of which exist here |

## Deliberately deferred

- Bridging these hooks to `PluginMeshEvent::Channel` for a genuinely separate OS-process
  plugin (milestone 2, per above).
- `chat_completion_stream` terminal/effective-request hooks.
- Any instrumentation of the raw-proxy ingress path (path 2). This is a bigger, separate
  design question: should plugin-served traffic get a hook contract at all, and if so,
  does it reuse `OpenAiHookPolicy`'s shape or need its own (it can't share
  `ChatCompletionRequest`, since that path is intentionally payload-agnostic)?
- Delegation phases named in #1331's original text — not mapped onto anything real yet.
