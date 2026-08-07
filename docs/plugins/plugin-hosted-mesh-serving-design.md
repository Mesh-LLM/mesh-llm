# Design: mesh-wide serving for plugin-hosted models

## Status

Implemented in the commit immediately preceding this document on this
branch. Written after live end-to-end testing (an external-relay plugin
serving an LM Studio model through the mesh) surfaced a gap in how a
plugin-only node — one with zero local (downloaded) models, whose only
inference capacity comes from a plugin endpoint (e.g.
`inference: [openai_http(...)]`) — is treated by the rest of the mesh.

## What already worked

Two things were already correct by the time this investigation started,
and are not touched by this change:

- **This node's own local ingress.** The unified API proxy every node now
  binds unconditionally at startup (`run_auto()` → `api_proxy` →
  `network/openai/ingress.rs`) already merges `plugin_manager
  .inference_models()` into `/v1/models` and already has full
  external-endpoint forwarding for plugin-hosted models. This has been
  true since before this investigation — an earlier draft of this
  document (and an earlier commit on this branch, since dropped) mistook
  a *different*, no-longer-primary code path
  (`network/openai/transport.rs::handle_mesh_request`, reached only via
  the now-`#[expect(dead_code)]` "passive listener" lane —
  `run_auto_model_path_or_shutdown` — that a large "daemon model
  lifecycle reconciliation" refactor superseded between this fork's base
  and current `main`) for the node's primary listener. It isn't; that
  patch was reverted as a no-op once the actual current architecture was
  understood. See "History" below.
- **Gossip advertisement.** `mesh/gossip.rs`'s
  `plugin_inference_models()` already correctly merges plugin-provided
  models into what this node tells the mesh it serves.

## The actual gap

Even with both of those correct, no *other* peer could ever route a
request to a plugin-only node, for two structural reasons — both
independent of the local-ingress/gossip machinery above, and both still
present on current `main`:

### 1. Host-role eligibility

Other peers only consider a node as a candidate host for model X if
`PeerInfo::routes_http_model` returns true, which requires
`accepts_http_inference()` — `matches!(self.role, NodeRole::Host { .. })`
(`mesh/peer_state.rs`).

`NodeRole::Host { http_port: u16 }` is set in exactly one place in the
whole crate: `startup_handles.rs`, as a side effect of a local model
finishing load. A plugin-only node's role never leaves the default
`Worker`, so **every other peer's `hosts_for_model()` filters this node
out completely**, independent of what its gossip payload advertises.

### 2. Inbound tunnel routing

A request another peer *does* decide to route here arrives as a raw byte
relay: an inbound QUIC stream gets forwarded to a local TCP connection at
`tunnel::Manager`'s configured `http_port`. That port defaults to `0` and
is — like the role above — only ever set as a side effect of a local
model finishing load (`startup_handles.rs`, three call sites). If it's
still `0`, an inbound tunnel is dropped with a warning
(`network/tunnel.rs`).

## Fix

Both gaps have the same shape: a value that should be a property of "the
node's API surface is up," generalized everywhere else in `run_auto()`
since the daemon-lifecycle-reconciliation refactor, but that these two
specific call sites still only set as a side effect of local-model load
completing.

- **Tunnel port**: since `run_auto()` already unconditionally constructs
  `tunnel::Manager` and binds the real API proxy before any model
  resolution happens, set `tunnel_mgr.set_http_port(api_port)`
  immediately after construction, unconditionally. The three call sites
  in `startup_handles.rs` remain and become redundant-but-harmless (same
  node, same `api_port`, for the process's lifetime) rather than
  conflicting.
- **Host role**: add `spawn_plugin_host_role_watcher`, a small background
  task started alongside the tunnel manager (skipped for `--client`
  nodes) that polls `plugin_manager.inference_models()` on a short
  interval and promotes to `NodeRole::Host { http_port: api_port }` when
  it's non-empty, demoting back to `Worker` when it becomes empty again.
  It tracks whether *it* performed the last promotion via a task-local
  flag and only demotes when that flag is set, but this is a soft
  safeguard, not real ownership tracking — `node.set_role()` is a plain,
  unconditional overwrite with no concept of "who claimed this role and
  why," and a local model's `Host { http_port }` (`startup_handles.rs`)
  uses the identical `http_port` value this watcher does (both derive it
  from the same `api_port`), so the role value alone can't distinguish
  the two callers. The demotion path additionally checks
  `node.models_being_served()` is empty before actually changing the
  role, so a node that starts serving a local model *after* this watcher
  promoted it won't have that model's `Host` status pulled out from under
  it once the plugin's models later disappear — but this is a targeted
  patch for the one race that's currently reachable, not a general
  ownership mechanism. A node with more complex overlapping
  local-model/plugin lifecycles than "plugin promotes first, local model
  starts being served later" isn't fully covered by this. Buzz's relay
  feature (the motivating use case) never sets a local `model_id`, so it
  doesn't exercise this at all; a caller that does mix both should treat
  this watcher's demotion behavior as best-effort until a real
  ownership-aware transition mechanism exists. `inference_models()` reads
  the plugin manager's own already-debounced health state (see
  `plugin::health`'s startup-grace/failure-threshold logic) rather than
  probing anything itself, so the short poll interval doesn't add
  flapping risk beyond what that debouncing already provides.

Both changes are two small, unconditional additions at a single call site
in `run_auto()`'s existing startup path — no new machinery, no threading
through the passive-listener/model-selection context that an earlier
version of this fix (written against an older fork base, before the
daemon-lifecycle-reconciliation refactor landed upstream) needed and that
would have been dead weight added to an already-dead code path.

## Deliberately deferred

- **Real ownership tracking for `NodeRole::Host`.** Nothing in `mesh::Node`
  currently tracks *why* a node is `Host` — `set_role()` is a plain
  overwrite. The plugin-host-role watcher's task-local flag plus a
  `models_being_served()` check (see above) closes the one race that's
  reachable today, but a real fix would give `NodeRole::Host` (or a
  parallel piece of state) an explicit set of claimants — local model
  load, plugin health, potentially others later — and only clear the role
  when the last claimant releases it. That's a bigger, more invasive
  change than this fix's scope; worth doing if a node mixing local models
  and plugins becomes a real use case rather than a theoretical one.
- **Capacity/VRAM-based host ordering.** `order_remote_hosts_by_context`
  and related host-ranking logic (used when multiple peers can serve the
  same model) is built around real local VRAM/context-window numbers. A
  plugin-relayed model has no meaningful local VRAM figure — needs a
  policy decision (advertise an unbounded/synthetic capacity, or exempt
  plugin-sourced models from VRAM-based ranking and use a simpler
  round-robin/health-only ordering among plugin hosts).
- **Model descriptors.** `all_served_model_descriptors` /
  `all_model_runtime_descriptors` assume a real local runtime descriptor
  (loaded quant, context length, etc.). Plugin-sourced models need either
  synthetic descriptors or these call sites need to become Option-aware.
- **Streaming verification through the tunnel path specifically.** Local
  streaming completions through the plugin's proxy are verified working;
  the same request routed to this node over an inbound QUIC tunnel from a
  peer has not been separately verified.
- **Live multi-peer verification.** This fix compiles cleanly and doesn't
  regress the existing single-node acceptance test, but the actual
  mesh-wide behavior it targets — another peer discovering and
  successfully routing a request to a plugin-only node — has not yet been
  exercised by a live multi-peer test.

## History

An earlier version of this branch, written against this fork's original
base (before a large "daemon model lifecycle reconciliation" feature and
related refactors landed on upstream `main`), additionally patched
`network/openai/transport.rs::handle_mesh_request` to merge plugin models
into its own `/v1/models` response and to route matching requests
directly to a plugin's loopback endpoint. That function is reached via
`run_passive`, itself reached only through
`run_auto_model_path_or_shutdown` — a function upstream has since marked
`#[expect(dead_code, reason = "bridges the retained advertised-model and
passive runtime compatibility lanes")]`. The refactor made every node's
own local ingress go through the already-plugin-aware
`network/openai/ingress.rs` path unconditionally instead, which made that
patch a no-op on current `main`. It was reverted once this was confirmed,
rather than landing a fix for an unreachable code path.
