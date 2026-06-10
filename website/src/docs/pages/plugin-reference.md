---
title: Plugin Reference
---

# Plugin Reference

## External Endpoints

Plugins may register external services without proxying all traffic through the plugin process. This is a control-plane declaration, not a request proxying requirement.

- attached external MCP servers are declared in the `mcp` section
- attached or plugin-hosted inference backends are declared in the `inference` section

`mesh-llm` then talks to those services directly when appropriate. This keeps heavy data-plane traffic out of plugin IPC.

### MCP Contributions

The `mcp` section may contain both local MCP-facing items implemented by the plugin and attached external MCP servers.

Preferred external forms:

- `external_stdio(...)`
- `external_http(...)`
- `external_tcp(...)`
- `external_unix_socket(...)`

External MCP names are namespaced as `plugin_name.method`.

### Inference Contributions

The `inference` section may contain both attached external OpenAI-compatible endpoints and plugin-hosted inference providers.

Preferred forms:

- `openai_http(...)` for attached external endpoints
- `provider(...)` for plugin-hosted backends

### Why Endpoint Registration Exists

Some services already speak a protocol that `mesh-llm` knows how to use directly — a local OpenAI-compatible inference server, an external MCP server reachable over stdio or TCP, or a plugin-hosted inference runtime such as an MLX-backed local server.

In these cases, the plugin should remain the control-plane owner for discovery, lifecycle, readiness, and availability, but `mesh-llm` should own the data plane when possible.

### Health And Availability

Endpoint health is separate from plugin health.

If an endpoint health check fails:

- the endpoint becomes unavailable
- the endpoint is removed from routing or aggregation
- the plugin remains loaded and is not marked disabled
- the host keeps checking health

If health returns, the endpoint becomes available again automatically.

This matters because a plugin may be healthy while its managed service is starting, restarting, temporarily unhealthy, reloading a model, or intentionally stopped. The host should treat plugin liveness and endpoint liveness as separate concerns.

### Recommended State Model

**Plugin states**: `starting`, `running`, `degraded`, `disconnected`, `failed`

**Endpoint states**: `unknown`, `starting`, `healthy`, `unhealthy`, `unavailable`

**Routed availability states**: `advertised`, `routable`, `draining`, `unavailable`

Routing decisions should depend on endpoint health, not just plugin process health.

## MCP

MCP is implemented by the host, not by individual plugins.

The plugin author marks which services should appear in MCP:

- `tool(...)`
- `resource(...)`
- `resource_template_service(...)`
- `prompt(...)`
- `completion(...)`

The host then synthesizes `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`, and completions where applicable.

External MCP endpoints may also be aggregated into the host's MCP surface.

### MCP Naming

By default, tool, resource, and prompt names should be plugin-namespaced:

- tool: `blackboard.feed`
- tool: `blackboard.post`
- resource: `blackboard://snapshot`
- prompt: `blackboard.status_brief`

Friendly aliases may be added for bundled plugins, but the canonical identity should remain namespaced to avoid collisions.

### MCP Streaming

MCP-facing operations may be buffered, streaming input, streaming output, or streaming input and output. For streaming operations, the host uses negotiated side streams internally rather than pushing large data through the control connection.

## HTTP Bindings

Plugins may declare HTTP bindings as part of the manifest. These let a plugin feel native over HTTP without requiring custom host route code for each plugin.

### Default Mounting

Plugin-defined HTTP bindings should be mounted under a plugin-owned namespace by default:

- `/api/plugins/blackboard/feed`
- `/api/plugins/blackboard/post`
- `/api/plugins/object-store/objects`

This avoids collisions and keeps plugin-specific APIs out of the top-level product namespace unless explicitly promoted.

### Promoted Product Routes

Some routes may become stable product APIs owned by `mesh-llm`, for example `/api/objects`. These routes should be backed by named capabilities, not by hard-coded plugin IDs:

- top-level route: `/api/objects`
- required capability: `object-store.v1`
- provider plugin: whichever plugin the host resolves for that capability

This keeps product APIs stable while allowing the backing plugin to change.

External endpoints do not automatically become HTTP routes. They are service registrations that the host may use for routing or aggregation according to their endpoint kind.

### Buffered vs Streamed HTTP

HTTP bindings may be declared as buffered request / buffered response, streamed request / buffered response, buffered request / streamed response, or streamed request / streamed response. The host decides whether to keep the invocation on the control channel or negotiate a side stream based on the binding mode and payload size.

## Streams And Large Transfers

Large payloads must not ride the main control connection. Instead, the control session negotiates a short-lived stream for the transfer.

Conceptual flow:

1. host sends `OpenStream`
2. plugin accepts
3. host and plugin establish a short-lived local stream
4. request or response bytes flow on that stream
5. either side may cancel
6. stream is torn down and cleaned up

This design supports 10 GB uploads, large downloads, long-lived streaming responses, and future websocket-like or SSE-style responses without blocking health checks or other control traffic.

## Suggested Control Messages

The exact wire format is still open, but the protocol should support:

- `Initialize` / `InitializeResponse { manifest }`
- `Health`
- `Shutdown`
- `Invoke` / `InvokeResult`
- `Notify`
- `MeshEvent`
- `OpenStream` / `OpenStreamResult`
- `CancelStream`
- `StreamError`

The stream protocol itself may be raw bytes or lightly framed bytes, depending on the use case.

## Capabilities

Capabilities let core depend on behavior rather than on plugin names:

- `object-store.v1`
- `mesh-blackboard.v1`
- `artifact-cache.v1`
- `model-catalog-provider.v1`

Capabilities are used when core needs a stable product contract, multiple plugins could satisfy the same role, or the host wants to promote a route into the top-level API.

Capabilities are not required for every plugin. They are mainly for shared contracts that `mesh-llm` itself depends on.

Endpoint registration is related but distinct: capabilities express stable contracts that core may depend on, while endpoints express concrete service instances that the host can talk to directly. An endpoint may satisfy a capability, but the two ideas remain separate in the design.

## Mesh Channels

Plugins may declare mesh channels for plugin-specific peer-to-peer coordination. These use the generic plugin mesh transport rather than dedicated core stream types for individual plugins. Core should not embed plugin-specific wire protocols in the main mesh transport when the behavior can live behind the generic plugin channel mechanism.
