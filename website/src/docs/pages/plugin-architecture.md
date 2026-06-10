---
title: Plugin Architecture
---

# Plugin Architecture

The plugin architecture is built around three core pieces: one long-lived control connection per plugin process, zero or more short-lived negotiated streams for large or streaming data, and one declarative plugin manifest that the host `stapler` projects into MCP, HTTP, and optional promoted product APIs.

`mesh-llm` remains the owner of:

- plugin lifecycle
- local IPC
- stapling manifest-declared services onto host-facing protocols
- HTTP serving
- MCP serving
- capability routing
- mesh participation and peer-to-peer transport

A plugin owns:

- its own feature logic
- local state
- operation handlers
- resource handlers
- prompt handlers
- plugin-specific mesh channel semantics

Plugins do not need to implement raw MCP or raw HTTP servers. The `stapler` is the host projection layer that turns plugin manifests into exposed MCP and HTTP surfaces.

## High-Level Model

The plugin system is **projection-oriented at the DSL level** and **service-oriented at the runtime level**.

Plugin authors think in terms of the host surfaces they contribute to:

- `mcp`
- `http`
- `inference`
- `provides`

The host runtime still executes native service invocations internally, but the author-facing DSL is organized by the surface the plugin contributes to.

This means:

- local MCP tools, resources, prompts, and completions live under `mcp`
- attached external MCP servers also live under `mcp`
- local HTTP routes live under `http`
- attached or plugin-hosted inference backends live under `inference`
- stable product capabilities live under `provides`

There is no separate top-level `services` section in the preferred DSL.

## Core Principles

### 1. Bundled Plugins Are Allowed

Plugins shipped in this source tree may be auto-registered by the host.

That is acceptable coupling. What is not acceptable is embedding one plugin's runtime behavior directly into core mesh logic. Core mesh transport and state should stay generic.

### 2. One Control Connection, Many Data Streams

Each plugin process has one long-lived control connection.

Use the control connection for:

- initialize / health / shutdown
- manifest registration
- small RPC-style requests
- mesh event delivery
- stream negotiation
- cancellation

Do not use the control connection for large uploads, downloads, or long-lived streaming responses.

For large or streaming payloads, the host and plugin negotiate a short-lived side stream.

### 3. MCP Is A Host Projection

`mesh-llm` is the MCP server. Plugins do not need to implement MCP JSON-RPC directly. They declare MCP-facing services in the manifest, and the host `stapler` exposes them over MCP.

### 4. HTTP Is A Host Projection

`mesh-llm` owns the HTTP server. Plugins may declare HTTP bindings, but they do not need to run an HTTP server themselves. The host `stapler` maps HTTP requests onto plugin operations and resources.

### 5. Capabilities Are Stable Product Contracts

When `mesh-llm` wants a stable product API such as `/api/objects`, core should depend on a named capability like `object-store.v1`, not on a specific plugin ID like `blobstore`.

## Architecture

### Control Session

There is one long-lived control session between host and plugin used for:

- plugin startup and manifest exchange
- health checks
- native service invocation requests and responses
- plugin-to-host notifications
- host-to-plugin mesh events
- opening and closing streams
- cancellation and error reporting

The control session should stay responsive even while the plugin is sending or receiving large payloads.

The native runtime contract is service-oriented, not MCP-oriented. The host invokes services such as operations, prompts, resources, and completions. MCP method names like `tools/call` and `prompts/get` are projection-layer concerns.

### Streams

Streams are short-lived negotiated channels for a single request, response, or transfer. They are opened via the control session and then carry data independently.

Used for:

- large HTTP request bodies
- large HTTP responses
- streaming uploads and downloads
- server-sent events or similar long-lived responses
- future bulk data flows between host and plugin

On Unix, streams map to short-lived Unix sockets. On Windows, streams map to short-lived named pipes. The protocol concept is `stream`, not `socket`, so the transport binding remains platform-specific.

### Why Streams Exist

The single-socket framed-envelope design is vulnerable to head-of-line blocking. Even chunked transfer traffic competes with health checks, tool calls, mesh events, and other control messages on the same queue. This architecture avoids that by separating control plane traffic from bulk and streaming data traffic.

## What The Host Owns

- launching and registering bundled plugins
- validating plugin identity
- keeping the control session alive
- stream negotiation and cleanup
- request validation
- HTTP mounting and MCP exposure
- capability resolution
- route collision detection
- permissions and policy enforcement

## What Plugins Own

- declaring their manifest
- implementing handlers
- handling their own local state
- reading and writing stream payloads when invoked
- implementing plugin-specific business logic

## Non-Goals

The plugin system should not require each plugin to:

- run its own HTTP server
- run its own MCP server
- manually negotiate Unix socket paths in application code
- hard-code core route registration in `mesh-llm`

The plugin system should also avoid:

- top-level product APIs that are secretly bound to one plugin ID
- plugin-specific core mesh stream types when generic plugin channels are sufficient

## Open Questions

- exact manifest schema
- exact control protocol message shapes
- exact stream framing format
- capability provider selection when multiple plugins implement the same capability
- whether promoted product routes are configured statically or negotiated dynamically
- how auth and policy rules are expressed for plugin-defined HTTP bindings

## Architecture Baseline

- bundled plugins may be auto-registered
- core mesh logic remains plugin-agnostic
- MCP and HTTP are first-class host projections
- product APIs depend on capabilities, not plugin IDs
- large data flows use negotiated side streams, not the control socket
