# DRAFT — for mesh-llm#1331, not yet sent

This is a draft technical note, written for the reference implementation in
this directory. It has not been posted anywhere. Sending it (and choosing any
attribution) is an operator decision, made separately from writing the code.

---

## What the private spike got right, and what it got wrong

A prior private spike found that `capsule-emit-mesh`'s own exemplar contract
(`exemplar/plugin_contract.py`, `mesh-llm#1331`) defines the plugin interface
as in-process Python with no serialization format or process boundary, and
concluded mesh-llm has **no cross-process plugin wire protocol at all**. That
premise is wrong. mesh-llm ships one, has for a while, and it works: the
published `mesh-llm-plugin` crate (crates.io), `proto/plugin.proto`, a DSL for
declaring plugin manifests, and a host runtime that loads external plugin
processes and speaks it. This repository's plugin
(`mesh/crates/mesh-llm-capsules-plugin`) builds against that real protocol and runs green
through `Initialize → Health → Shutdown` plus real HTTP dispatch — see its
interop tests.

One correction to the wire format itself: it is **not gRPC**. `mesh-llm-plugin`
depends on `prost`/`prost-build`, not `tonic` — there is no HTTP/2 framing or
gRPC service definition anywhere in the crate. The actual mechanism is a
length-prefixed protobuf `Envelope` message, framed over a Unix domain socket
(or a named pipe on Windows), with the host spawning the plugin as a
subprocess and passing the socket path via `MESH_LLM_PLUGIN_ENDPOINT`. Framing
it as "gRPC-native" overstates what's there; "Envelope-over-local-socket" is
accurate.

## The bigger correction: there is no per-exchange lifecycle hook

The exemplar contract's `Manifest.phases` field assumes a plugin can ask to be
called at named lifecycle points — `request_received`, `backend_selected`,
`exchange_finished` — for **every** OpenAI-compatible exchange, regardless of
which backend ultimately serves it. That shape does not exist in the real
protocol. `MeshEvent.Kind` — the only event enum in `plugin.proto` — is
mesh-topology-only: `PEER_UP`, `PEER_DOWN`, `PEER_UPDATED`, `LOCAL_ACCEPTING`,
`LOCAL_STANDBY`, `MESH_ID_UPDATED`. Nothing in either direction of the wire
protocol carries a per-request lifecycle event to a plugin that isn't itself
handling that request.

The real, working mechanism for a plugin to be dispatched OpenAI-compatible
traffic is registering as an **inference provider**
(`inference: [provider(id, address)]` in the DSL, which sets
`managed_by_plugin = true` on the endpoint manifest). But dispatch to it is:

- **Exact-name-scoped.** The host matches the requested `model` string exactly
  against each endpoint's served-models list (itself discovered by the host
  actively probing the plugin's own `/v1/models` — see
  `mesh-llm-host-runtime::plugin::health::probe_endpoint`). There is no
  wildcard or pattern field anywhere in `EndpointManifest`.
- **Last-resort-only.** A plugin endpoint is only consulted when no local or
  remote backend already serves that exact model name
  (`network::openai::ingress::route_missing_local_model`).

So a plugin that enforces an admission policy on the real protocol does it by
being the registered provider for every blocked model name it knows about in
advance, advertised through its own `/v1/models`. "Abstain" is realized
structurally — by simply not advertising a model — rather than as a per-call
decision a plugin returns. This is a materially different shape from the
exemplar's phase-based model, not a renaming exercise, and this repository's
plugin is built that way deliberately (see its
`blocked_models()` doc comment). From 0.1.2 that admission endpoint is
opt-in: it exists only when `CAPSULES_BLOCKED_MODELS` names models.
A node that names none registers no inference provider, declares no
`admission_policy.v1` capability and serves no models; it seals records and
serves the Evidence page, nothing else.

## `HttpBodyMode` does answer the body-access question — narrowly

`HttpBindingManifest.request_body_mode`/`response_body_mode`
(`HttpBodyMode::BUFFERED`/`STREAMED`) is real, and the host genuinely honors
it (`mesh-llm-host-runtime::api::routes::plugins::binding_transfer_mode`). But
it applies only to a plugin's **own custom HTTP routes**, declared via
`http_binding()`/`dsl::http::*` and relayed to the plugin process over the
Envelope RPC bridge — not to the inference-provider path this plugin uses.
The OpenAI-dispatch path is always fully buffered by the host before any
routing decision is made (the request is materialized into a
`BufferedHttpRequest` up front). So the spike's `BodyAccessDenied` /
evidence-unavailable state — the host granting or withholding body access —
has no reachable analog on the provider path: the host always hands the
plugin a complete body. `HttpBodyMode` is the real mechanism for the question
it was designed to answer; it just answers a different question than the one
a plugin enforcing an admission policy on the inference-provider path needs
answered.

## Suggested #1331 addendum

- Retitle the request from "no wire protocol is specified" to "the
  contract's phase-based model has no analog in the real protocol." The
  cross-process handshake itself is not the gap; the per-exchange
  observation hook is.
- If a future "observe every exchange regardless of destination backend"
  capability is wanted, treat it as new surface to design, not a
  documentation gap in the existing kit.
- Document the provider/exact-model-registration pattern as the sanctioned
  approach for plugins that enforce an admission policy — nothing in the
  current plugin docs frames `inference::provider()` as a policy mechanism,
  only as "serve this model yourself."
- A working reference implementation is this repository's plugin
  (`mesh/crates/mesh-llm-capsules-plugin/`).

## Sealing a HOST-SERVED exchange (the "all three real" closure)

The provider-path story above covers exchanges the plugin *serves itself*. A
real loaded GGUF is different: the host serves it directly (host → native
runtime), so it never reaches this plugin's HTTP handler and — before this
change — produced **no capsule at all**. The three real facts we want in one
sealed capsule (real model identity, real token usage, real hardware/GPU) all
exist on the host's `openai.exchange.v1` **terminal event**, which this plugin
already observes on the mesh channel. So the closure is *seal-on-observe*:

1. **Usage on the mirror.** The host terminal event carries the served
   backend's real `usage` (added host-side alongside `serving_provenance`). The
   plugin's mirror envelope previously **dropped** it; it now carries it through
   (`MirrorUsage`) so the sealed capsule holds the real token counts, not a
   zeroed stub.
2. **Seal-on-observe.** `ObservedLifecycleEvents::is_sealable_host_served`
   recognizes a host-served terminal (Terminal phase, 2xx, and **real
   served-model identity** — `architecture`/`model_identity_hash` present, which
   only a real GGUF has, so the plugin's own synthetic stub is never
   double-sealed) and the channel handler seals a capsule from the observed
   provenance + usage + request digest via `emit_for_observed_host_exchange`.
3. **Request-bytes binding.** The capsule's `agent_input_digest` must bind the
   REAL request bytes. The host now forwards a `request_digest` on the terminal
   event — the canonical JSON-DIGEST (RFC 8785 JCS over the profile-normalized,
   float-stringified request body) computed host-side **byte-for-byte identical**
   to this plugin's own `canonical_body_digest` (pinned by a shared
   Python-reference fixture on both sides). The plugin binds it directly. This
   is a genuine binding to what was asked, recomputable by any verifier from the
   request body alone.

4. **Response-body + tool_calls/reasoning binding (NOW CLOSED for the
   non-streamed JSON path).** The earlier revision of this note said the host
   streamed the response and only `usage` returned on the `Copy`
   `RouteDispatchOutcome`, so `agent_output_digest` could bind only the observed
   terminal facts and no `tool_calls_digest` was possible. That was true for the
   *streamed* path but **not** for the non-streamed JSON relay: at the JSON-relay
   delivery point (`response::json_adaptation` / `response::relay`) the host
   still holds the complete response body in hand — it parses `usage` from it and
   even captures it for logging. So the host now computes, at that point, three
   digests over the REAL served body and forwards them on the terminal event:

   - `response_digest`   = `json_digest(response body)`  → binds `agent_output_digest`
   - `tool_calls_digest` = `json_digest(tool_calls)`     (absent when the model emitted none)
   - `reasoning_digest`  = `json_digest(reasoning chunks)`(absent for a non-reasoning model)

   computed with **plain RFC 8785 JCS** (no float-stringify), byte-for-byte
   identical to the Python reference `agent_action_capsule.json_digest` and the
   labeled sub-digests of `capsule_ledger/conversation/exchange.py`'s
   `digest_conversation_exchange`. They ride up as a `Copy`
   `ExchangeOutputDigests` bundle (raw sha-256 bytes) on
   `RouteAttemptResult::Delivered → RouteDispatchOutcome::RespondedWithUsage`, so
   the enums stay `Copy`. The plugin binds `agent_output_digest` to the forwarded
   `response_digest` and seals `tool_calls_digest`/`reasoning_digest` (OPTIONAL,
   absent when the model had none, never fabricated) into
   `model_attestation.compute_attestation`.

   VERIFIED: a capsule sealed from a real host-served SETI@Home / `web_search`
   exchange carries `tool_calls_digest =
   f294be8a53bb9c29cd94472721f0857591f34b23fe010882de79b9fb210b1395`, equal to an
   independent Python `json_digest(tool_calls)` recompute, and verifies GREEN
   (`agent-action-capsule verify` + the detached `.cose`).

### What is NOT fully bound yet, and exactly what full closure needs

The **STREAMED** response path is still bound only to the observed terminal facts
(model + real usage), not the response body. On the streamed path
(`response::stream_translation`, the pipeline/MoA gateway) the body is teed to the
client chunk-by-chunk and never buffered whole, so no response/tool_calls/
reasoning digest is captured — the forwarded `ExchangeOutputDigests` bundle is
all-`None`, and the plugin falls back to the observed-terminal-facts
`agent_output_digest`, documented as such, never a fabricated body digest.

Full closure for the streamed path needs a **streaming hasher**: the proxy would
tee the response body through a SHA-256 as it streams (without buffering it) and
surface the resulting digest — and, for `tool_calls_digest`, an incremental
tool-call accumulator — on the dispatch outcome. That is a proxy-streaming
change, left as follow-up. The non-streamed JSON path (which is what a tool-call
exchange takes) and the request-side binding are fully closed here.
