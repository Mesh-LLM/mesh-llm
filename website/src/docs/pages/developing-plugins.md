---
title: Developing Plugins
---

# Developing Plugins

The primary design goal is very low boilerplate. The preferred DSL is surface-first, organized by the host projection the plugin contributes to.

## DSL Sections

- `provides` — stable capability contracts
- `mcp` — MCP tools, resources, prompts, and attached external MCP servers
- `http` — local HTTP routes
- `inference` — attached external inference endpoints or plugin-hosted providers
- `mesh` — mesh channels the plugin may send and receive on
- `events` — mesh events the host may deliver to the plugin

Lifecycle hooks stay local to the plugin definition:

- `startup_policy`
- `health`
- `on_initialized`
- `on_channel_message`
- `on_mesh_event`

Each section is self-contained. If a plugin contributes something to a host surface, it is declared in the section for that surface.

## Example Plugin

```rust
use mesh_llm_plugin::{
    capability, plugin_server_info, PluginMetadata,
    http::{get, post},
    inference::openai_http,
    mcp::{external_stdio, prompt, resource, tool},
    PluginStartupPolicy,
};

let plugin = mesh_llm_plugin::plugin! {
    metadata: PluginMetadata::new(
        "notes",
        "1.0.0",
        plugin_server_info(
            "notes",
            "1.0.0",
            "Notes",
            "Shared notes services",
            None::<String>,
        ),
    ),

    startup_policy: PluginStartupPolicy::PrivateMeshOnly,

    provides: [
        capability("notes.v1"),
        capability("search.v1"),
    ],

    mesh: [
        mesh_llm_plugin::mesh::channel("notes.v1"),
    ],

    events: [
        mesh_llm_plugin::events::peer_up(),
    ],

    mcp: [
        tool("search")
            .description("Search notes")
            .input::<SearchArgs>()
            .handle(search),

        resource("notes://latest")
            .name("Latest Notes")
            .handle(read_latest),

        prompt("summarize_notes")
            .description("Summarize recent notes")
            .handle(summarize_notes),

        external_stdio("filesystem", "npx")
            .arg("-y")
            .arg("@modelcontextprotocol/server-filesystem"),
    ],

    http: [
        get("/search")
            .description("Search notes")
            .input::<SearchArgs>()
            .handle(search),

        post("/notes")
            .description("Create a note")
            .input::<PostArgs>()
            .handle(post_note),
    ],

    inference: [
        openai_http("local-llm", "http://127.0.0.1:8080/v1")
            .managed_by_plugin(false),
    ],

    health: |_context| {
        Box::pin(async move { Ok("ok".to_string()) })
    },

    on_initialized: |context| {
        Box::pin(async move {
            context
                .send_json_channel(
                    "notes.v1",
                    String::new(),
                    "notes",
                    &NotesMessage::SyncRequest,
                )
                .await
        })
    },

    on_channel_message: |message, context| {
        Box::pin(async move {
            handle_notes_channel(message, context).await
        })
    },

    on_mesh_event: |event, context| {
        Box::pin(async move {
            handle_notes_mesh_event(event, context).await
        })
    },
};
```

### Key Design Points

- `mcp` contains both local MCP contributions and attached external MCP servers
- `http` contains local HTTP contributions
- `inference` contains both attached external inference endpoints and plugin-hosted inference providers
- `provides` declares stable capability contracts that core product routes can depend on
- `mesh` declares which mesh channels the plugin is allowed to receive and send
- `events` declares which mesh events the host may deliver to the plugin

Event delivery is allowlist-based: no `mesh` declaration means no channel delivery, no `events` declaration means no mesh events. Plugins only receive the event kinds they explicitly declare.

The runtime and `stapler` handle schema exposure, MCP projection, HTTP projection, request validation, stream negotiation, transport details, and host-side routing and aggregation.

Plugin authors should not manually implement MCP `tools/list`, MCP `tools/call`, MCP `resources/read`, HTTP routing, or control-plane socket negotiation.

## Internal RPC Plugins

Most plugins should use `plugin!`. Host-private plumbing services that need raw RPC methods rather than surfaced MCP, HTTP, or inference declarations should use `InternalRpcPluginBuilder`.

This is the escape hatch for internal-only services such as blobstore. It keeps raw host RPC separate from the normal manifest-driven plugin surface.

## Streaming In The DSL

Streaming is explicit. For HTTP bindings, the preferred modifiers are:

- `.stream_request()`
- `.stream_response()`
- `.sse()`

These declare whether the request body, response body, or response format requires side-stream transport.
