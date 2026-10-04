# Emitter and Hook Inventory

This is the current repo inventory for hook, callback, and emitter surfaces that can emit output or call back into Rust.

There is no single central registry today. The canonical ownership is split across the docs and code paths listed below, so this page acts as the map.

## Source Of Truth

| Surface | Canonical docs | Canonical code |
|---|---|---|
| Rust `OutputEvent` and `OutputSink` | [mesh/crates/mesh-llm-events/README.md](../../crates/mesh-llm-events/README.md) | [mesh/crates/mesh-llm-events/src/lib.rs](../../crates/mesh-llm-events/src/lib.rs), [mesh/crates/mesh-llm-host-runtime/src/runtime/mod.rs](../../crates/mesh-llm-host-runtime/src/runtime/mod.rs), [mesh/crates/mesh-llm-host-runtime/src/runtime/discovery.rs](../../crates/mesh-llm-host-runtime/src/runtime/discovery.rs), [mesh/crates/mesh-llm-host-runtime/src/runtime/local.rs](../../crates/mesh-llm-host-runtime/src/runtime/local.rs) |
| Virtual LLM `/mesh/hook` callbacks | [docs/design/VIRTUAL_LLM.md](VIRTUAL_LLM.md) | [mesh/crates/mesh-llm-host-runtime/src/api/routes/mesh_hook.rs](../../crates/mesh-llm-host-runtime/src/api/routes/mesh_hook.rs), [mesh/crates/mesh-llm-host-runtime/src/inference/virtual_llm.rs](../../crates/mesh-llm-host-runtime/src/inference/virtual_llm.rs) |
| Skippy native runtime callbacks (`_with_events`) | [docs/design/NATIVE_RUNTIMES.md](../../../skippy/docs/design/NATIVE_RUNTIMES.md), [skippy/crates/skippy-ffi/README.md](../../../skippy/crates/skippy-ffi/README.md) | [skippy/crates/skippy-runtime/src/runtime_events.rs](../../../skippy/crates/skippy-runtime/src/runtime_events.rs), [mesh/crates/mesh-llm-host-runtime/src/runtime/local/native_runtime_events.rs](../../crates/mesh-llm-host-runtime/src/runtime/local/native_runtime_events.rs) |
| Upstream `llama_log_set` native logs | [skippy/crates/skippy-ffi/README.md](../../../skippy/crates/skippy-ffi/README.md) | [skippy/crates/skippy-runtime/src/lib.rs](../../../skippy/crates/skippy-runtime/src/lib.rs), [mesh/crates/mesh-llm-host-runtime/src/runtime/mod.rs](../../crates/mesh-llm-host-runtime/src/runtime/mod.rs) |
| OpenAI and Responses SSE emitters | [skippy/crates/skippy-inference-api/README.md](../../../skippy/crates/skippy-inference-api/README.md) | [skippy/crates/skippy-inference-api/src/responses.rs](../../../skippy/crates/skippy-inference-api/src/responses.rs), [skippy/crates/skippy-inference-api/src/sse.rs](../../../skippy/crates/skippy-inference-api/src/sse.rs), [mesh/crates/mesh-llm-host-runtime/src/network/openai/virtual_model/progress.rs](../../crates/mesh-llm-host-runtime/src/network/openai/virtual_model/progress.rs) |
| Plugin and mesh event helpers | [docs/plugins/README.md](../plugins/README.md) | [mesh/crates/mesh-llm-host-runtime/src/plugin/mod.rs](../../crates/mesh-llm-host-runtime/src/plugin/mod.rs), [mesh/crates/mesh-llm-host-runtime/src/mesh/mod.rs](../../crates/mesh-llm-host-runtime/src/mesh/mod.rs) |

## Readiness Boundary

Readiness is advanced only by Rust-owned state edges, not by native facts or raw logs.

The readiness events in current code are `ApiReady`, `WebserverReady`, `ModelReady`, and `RuntimeReady`, all emitted from Rust startup logic after Rust has decided the node is ready.

Native callbacks, native logs, and `/mesh/hook` callbacks are visibility only. They can inform Rust, but they do not make the runtime ready by themselves.

## What This Is Not

This is not a new contract. It is a factual inventory of the current surfaces.

Native callbacks emit facts only. Rust owns readiness, policy, output state, retries, and user-facing presentation.

## Rust `OutputEvent` Emission

| Surface | Trigger | Owner | Emits | Readiness | Fallback |
|---|---|---|---|---|---|
| Startup and runtime orchestration | Startup, shutdown, passive mode, node identity, launch planning, API and console bring-up | `mesh-llm-host-runtime/src/runtime/mod.rs` | `Startup`, `LaunchPlan`, `NodeIdentity`, `InviteToken`, `ApiReady`, `WebserverReady`, `ModelReady`, `RuntimeReady`, `ShutdownRequested`, `Shutdown` | Yes for the readiness edges, no for pure status lines | `mesh_llm_events::emit_event` no-ops when no sink is registered, and the tracing bridge falls back to stderr if event emission fails |
| Discovery and mesh state | Discovery start, mesh found, join success, join failure, peer join or leave, waiting for peers, host election | `mesh-llm-host-runtime/src/runtime/discovery.rs`, `mesh-llm-host-runtime/src/mesh/mod.rs` | `DiscoveryStarting`, `MeshFound`, `DiscoveryJoined`, `DiscoveryFailed`, `WaitingForPeers`, `PassiveMode`, `PeerJoined`, `PeerLeft`, `HostElected` | No | Same `emit_event` no-op fallback |
| Local model loading and download progress | Model load, unload, and package or model download progress | `mesh-llm-host-runtime/src/runtime/local.rs`, `mesh-llm-host-runtime/src/models/catalog.rs`, `mesh-llm-host-runtime/src/inference/skippy/materialization.rs` | `ModelLoading`, `ModelLoaded`, `ModelUnloading`, `ModelUnloaded`, `ModelDownloadProgress` | No for load progress, yes when the Rust startup path later emits `ModelReady` | Same `emit_event` no-op fallback |
| Native log bridge | Filtered native log events forwarded from `skippy-runtime` | `mesh-llm-host-runtime/src/runtime/local/native_runtime_events.rs`, `mesh-llm-host-runtime/src/runtime/mod.rs` | `LlamaNativeLog` | No | Native logs can be suppressed or redirected to file when runtime log setup is unavailable |
| OpenAI ingress bootstrap | Bootstrap proxy starts while the GPU loads | `mesh-llm-host-runtime/src/network/openai/ingress.rs` | `Info` with bootstrap progress text | No | Same `emit_event` no-op fallback |

## Virtual LLM `/mesh/hook` Callbacks

| Hook | Trigger | Owner | Emits or returns | Readiness | Fallback |
|---|---|---|---|---|---|
| `pre_inference` | The request carries images or audio the current model cannot handle | `mesh-llm-host-runtime/src/api/routes/mesh_hook.rs`, `mesh-llm-host-runtime/src/inference/virtual_llm.rs` | Returns `{"action":"inject", "text":...}` or `{"action":"none"}` | No | Invalid JSON returns 400, non-loopback callers return 403, unsupported cases return `none` |
| `post_prefill` | First-token entropy and margin cross the uncertain prompt threshold | Same as above | Returns `{"action":"inject", "text":...}` or `{"action":"none"}` | No | Same `none` fallback |
| `mid_generation` | Entropy spike, repetition loop, or surprise break during generation | Same as above | Returns `{"action":"inject", "text":...}` or `{"action":"none"}` | No | Same `none` fallback |

These callbacks are synchronous POSTs to `http://localhost:{mesh_port}/mesh/hook`. The host route only accepts loopback callers.

## Skippy Native Runtime Callbacks (`_with_events`)

| Callback kind | Trigger | Owner | Emits or returns | Readiness | Fallback |
|---|---|---|---|---|---|
| Model open started | Native model open begins | `skippy/crates/skippy-runtime/src/runtime_events.rs`, translated in `mesh/crates/mesh-llm-host-runtime/src/runtime/local/native_runtime_events.rs` | Rust converts it to `OutputEvent::Info` | No | If the runtime-event feature bit or `_with_events` symbol is missing, Rust uses the legacy no-events open path |
| Model open progress | Native model open reports progress | Same as above | `OutputEvent::Info` with progress text | No | Null reporter means no events, malformed events are dropped, and a full channel drops the event after `try_send` fails |
| Backend device selected | Native open chooses a backend device | Same as above | `OutputEvent::Info` | No | Same legacy no-events fallback |
| Model open finished | Native open finished successfully | Same as above | `OutputEvent::Info` that says Rust is still waiting for readiness | No | Same as above |
| Handled model open failure | Native open returns a handled failure | Same as above | `OutputEvent::Warning` | No | Same as above |

Native runtime callbacks are operation scoped. They are bounded to one active model-open operation and are best effort only.

## Upstream `llama_log_set` Native Logs

| Surface | Trigger | Owner | Emits or returns | Readiness | Fallback |
|---|---|---|---|---|---|
| Native llama.cpp logs | Upstream `llama.cpp`, `ggml`, or helper code writes a log line | `skippy/crates/skippy-runtime/src/lib.rs`, bridged by `mesh/crates/mesh-llm-host-runtime/src/runtime/mod.rs` | `NativeLogEvent`, then `OutputEvent::LlamaNativeLog` | No | If the instance has no runtime log dir or redirection fails, `mesh-llm` suppresses native logs instead of keeping them on stdout |

The upstream callback is installed through `llama_log_set`, with the sibling `ggml_log_set` and `mtmd_helper_log_set` hooks handled the same way in the runtime crate.

## OpenAI And Responses SSE Emitters

| Surface | Trigger | Owner | Emits or returns | Readiness | Fallback |
|---|---|---|---|---|---|
| SSE frame helpers | Any OpenAI response stream needs JSON framing | `skippy/crates/skippy-inference-api/src/sse.rs` | `Event` values or `[DONE]` | No | Failed JSON serialization emits a fixed server error payload |
| Responses stream scaffold | Responses API stream creation and completion | `skippy/crates/skippy-inference-api/src/responses.rs` | `response.created`, `response.output_item.added`, `response.content_part.added`, `response.output_text.delta`, `response.reasoning_text.delta`, `response.output_text.done`, `response.content_part.done`, `response.output_item.done`, `response.completed` | No | The helpers are pure constructors, so the caller owns any transport fallback |
| Virtual-model progress stream adapter | A streaming virtual-model turn is still running | `mesh/crates/mesh-llm-host-runtime/src/network/openai/virtual_model/progress.rs` | Emits Responses reasoning deltas, chat-completion reasoning content, or Anthropic pings, then the model response | No | On failure after headers are sent, the stream ends with a failure event plus `[DONE]` so clients do not see a silent truncation |

## Plugin And Mesh Event Helpers

| Surface | Trigger | Owner | Emits or returns | Readiness | Fallback |
|---|---|---|---|---|---|
| Plugin mesh delivery | A plugin sends a channel message, a bulk transfer message, or the host broadcasts a mesh event | `mesh/crates/mesh-llm-host-runtime/src/plugin/mod.rs` | Forwards `PluginMeshEvent::{Channel,BulkTransfer,OpenStream}` or `proto::MeshEvent` to subscribed plugins | No | Undeclared channels are dropped, unloaded plugins are skipped, and unsubscribed mesh events are not delivered |
| Mesh log helpers | Mesh code wants a simple info or warning line | `mesh/crates/mesh-llm-host-runtime/src/mesh/mod.rs` | Thin `OutputEvent::Info` and `OutputEvent::Warning` wrappers | No | Same `emit_event` no-op fallback |

## Notes

- `OutputEvent::RuntimeReady` is Rust owned and only appears after the startup reporter sees every startup model ready.
- Native callbacks and native logs can describe progress, device selection, or handled failure, but they do not advance readiness on their own.
- If you are looking for the current implementation of a surface, start with the owning code path listed in the tables above, then follow the linked doc.
