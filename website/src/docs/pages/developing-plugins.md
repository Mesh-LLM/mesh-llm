---
title: Developing Plugins
---

# Developing Plugins

Build a plugin when an integration needs its own process, dependencies, release cadence, or local state. The plugin API is Rust-first and manifest-driven: you declare what the plugin contributes, then attach typed handlers for the behavior it owns.

This guide describes the current author API in the `mesh-llm-plugin` crate. The [Plugin Architecture](/docs/pages/plugin-architecture/) page explains the runtime model behind it.

## Choose a contribution surface

Start by deciding what your plugin adds to the host:

| Surface | Use it for | Typical declarations |
| --- | --- | --- |
| `mcp` | Tools, resources, prompts, completions, or an external MCP server | `mcp::tool`, `mcp::resource`, `mcp::external_stdio` |
| `http` | Plugin-owned HTTP operations mounted by the host | `http::get`, `http::post`, `.stream_request()`, `.stream_response()` |
| `inference` | An attached OpenAI-compatible endpoint or a provider managed by the plugin | `inference::openai_http`, `inference::provider` |
| `provides` | A stable capability contract that core or another plugin can consume | `capability("object-store.v1")` |
| `mesh` | Plugin-specific peer-to-peer messages | `mesh::channel("notes.v1")` |
| `events` | Mesh lifecycle events delivered by the host | `events::peer_up()`, `events::peer_down()` |

MCP and HTTP are host projections. A plugin does not need to implement an MCP server, HTTP server, or socket protocol itself.

## Start a Rust plugin

Create a binary crate and depend on `mesh-llm-plugin` at a version compatible with the mesh-llm release you target. Keep the host and plugin protocol versions aligned; the host rejects incompatible initialization handshakes.

The smallest useful plugin has a manifest, one handler, and a runtime entrypoint:

```rust
use mesh_llm_plugin::{
    mcp, plugin, plugin_server_info, PluginMetadata, PluginRuntime,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let plugin = plugin! {
        metadata: PluginMetadata::new(
            "hello-plugin",
            "0.1.0",
            plugin_server_info(
                "hello-plugin",
                "0.1.0",
                "Hello plugin",
                "A small example plugin",
                None::<String>,
            ),
        ),

        mcp: [
            mcp::tool("ping")
                .description("Return a health response")
                .handle(|_args, _context| Box::pin(async {
                    Ok(serde_json::json!({ "status": "ok" }))
                })),
        ],
    };

    PluginRuntime::run(plugin).await?;
    Ok(())
}
```

`PluginRuntime::run` connects to the endpoint that mesh-llm provides through the plugin environment. Do not invent a socket path or start a second listener in the plugin process.

For typed input, define a serializable schema and pass it to `.input::<T>()`:

```rust
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct SearchArgs {
    query: String,
}

// ...
mcp::tool("search")
    .description("Search notes")
    .input::<SearchArgs>()
    .handle(|args, _context| Box::pin(async move {
        Ok(serde_json::json!({ "query": args.query, "matches": [] }))
    }))
```

The same typed-handler pattern is used by HTTP routes. Add `.output::<T>()` when clients benefit from an explicit response schema.

## Build the manifest

The `plugin!` macro keeps the manifest close to the handlers:

```rust
let plugin = plugin! {
    metadata: metadata,

    provides: [
        mesh_llm_plugin::capability("notes.v1"),
    ],

    mcp: [
        mcp::tool("search")
            .description("Search notes")
            .input::<SearchArgs>()
            .handle(search),
        mcp::resource("notes://latest")
            .name("Latest notes")
            .handle(read_latest),
        mcp::external_stdio("filesystem", "npx")
            .arg("-y")
            .arg("@modelcontextprotocol/server-filesystem"),
    ],

    http: [
        mesh_llm_plugin::http::post("/notes")
            .description("Create a note")
            .input::<PostArgs>()
            .handle(post_note),
    ],

    inference: [
        mesh_llm_plugin::inference::openai_http(
            "local-llm",
            "http://127.0.0.1:8080/v1",
        ),
    ],
};
```

Keep declarations narrow. A plugin receives mesh channels and events only when it explicitly declares them. Use capabilities for stable contracts, not as a second name for the plugin itself.

## Lifecycle and host context

The host launches the plugin as a child process and supplies:

| Variable | Meaning |
| --- | --- |
| `MESH_LLM_PLUGIN_ENDPOINT` | Local control endpoint to connect to |
| `MESH_LLM_PLUGIN_TRANSPORT` | Transport kind, such as `unix` or `pipe` |
| `MESH_LLM_PLUGIN_NAME` | Configured plugin name |
| `MESH_LLM_PLUGIN_URL` | Optional `[[plugin]].url` value |

Use lifecycle hooks only when the plugin needs them:

- `health` reports plugin liveness.
- `on_initialized` runs after the host accepts the manifest.
- `on_channel_message` handles declared plugin mesh channels.
- `on_mesh_event` handles declared host events.

Keep health fast and independent from long-running operations. Plugin health and the health of a registered inference endpoint are separate: an endpoint may restart while the plugin remains healthy.

For large request bodies, downloads, or streaming responses, declare the mode on the HTTP binding with `.stream_request()`, `.stream_response()`, or `.sse()`. The host then negotiates a side stream so control messages and health checks remain responsive.

## Configuration schemas

If your plugin has user-facing settings, expose a schema in its manifest and document the corresponding TOML. Settings belong under the plugin table:

```toml
[[plugin]]
name = "notes"

[plugin.settings]
storage_path = "/var/lib/mesh-notes"
retention_days = 14
```

Declare the schema in Rust with the manifest helpers. The host receives this manifest during initialization and uses it to validate `[plugin.settings]`:

```rust
use mesh_llm_plugin::{
    PluginMetadata, SimplePlugin, config_integer, config_schema, config_setting,
    plugin_manifest, plugin_server_info,
};

let manifest = plugin_manifest![
    config_schema("notes").setting(
        config_setting("retention_days", config_integer())
            .default_value(&14)
            .description("How long to retain entries."),
    ),
];

let plugin = SimplePlugin::new(
    PluginMetadata::new(
        "notes",
        "1.0.0",
        plugin_server_info("notes", "1.0.0", "Notes", "Shared notes services", None::<String>),
    )
    .with_manifest(manifest),
);
```

The same manifest can be passed to `InternalRpcPluginBuilder::with_manifest` when using the internal-RPC API. See the [`config_schema` and `config_setting` helpers](https://github.com/Mesh-LLM/mesh-llm/blob/main/crates/mesh-llm-plugin/src/manifest.rs) for the complete schema surface.

Prefer typed settings with useful descriptions, defaults, constraints, and explicit restart scope. Do not make users put plugin settings at the top level of `config.toml`.

## Package and release a plugin

The installer expects a native GitHub Release archive. Each archive should contain one directory named after the plugin:

```text
hello-plugin/
  plugin.toml
  hello-plugin
  README.md
  LICENSE
  skills/
    hello-workflow/
      SKILL.md
```

On Windows, the executable is `hello-plugin.exe`. The required files are `plugin.toml` and the executable; documentation, license files, and skills are recommended.

Publish archives using the target triple in the filename:

```text
hello-plugin-0.1.0-aarch64-apple-darwin.tar.gz
hello-plugin-0.1.0-x86_64-unknown-linux-gnu.tar.gz
hello-plugin-0.1.0-x86_64-pc-windows-msvc.zip
```

The current installer targets macOS Apple Silicon and Intel, Linux ARM64 and x86_64, and Windows ARM64 and x86_64. Release automation should build and test every archive it publishes.

## Ship Agent Skills

A plugin can bundle skills for agent clients under `skills/<skill-name>/SKILL.md`. Use portable lowercase names with single hyphens, include `name` and `description` frontmatter, and keep supporting files relative to the skill directory.

After installation, users can copy plugin skills into detected clients with:

```bash
mesh-llm skills install
```

Do not overwrite an existing user-owned skill unless the user explicitly asks for a forced install. Avoid hard-coded home directories and platform-specific paths in the skill.

## Test before publishing

At minimum, test the plugin with a running mesh-llm node and verify:

1. The plugin connects and completes initialization.
2. The manifest exposes the expected MCP, HTTP, inference, capability, channel, and event entries.
3. Invalid input returns a useful error without taking down the control session.
4. Health still responds while a handler is running.
5. Streaming and cancellation do not leak side streams.
6. A stopped endpoint becomes unavailable without incorrectly disabling the plugin.
7. The release archive extracts to the expected directory and runs on each target.
8. A mixed-version host rejects incompatible protocol versions clearly and accepts the versions you support.

For plugin-specific runtime tests, use the repository's normal Rust test workflow. For a full host check, use the [testing playbook](/docs/pages/testing/).

## Design and security checklist

- Keep the plugin process independent of mesh-llm internals; depend on the author API and protocol contract.
- Treat all plugin configuration and endpoint URLs as untrusted input.
- Do not put secrets in manifests, release archives, logs, or MCP tool descriptions.
- Prefer host-owned HTTP and MCP projections over opening an untracked listener.
- Use namespaced MCP identifiers to avoid collisions.
- Keep the control connection for lifecycle and small RPCs; use side streams for bulk data.
- Declare the smallest possible set of mesh channels and events.
- Document required network access, files, subprocesses, and permissions.
- Pin or verify third-party dependencies in release builds.

## Next topics worth documenting

The plugin section would benefit from a compatibility matrix by mesh-llm release, a configuration-schema cookbook, a third-party plugin security model, a release workflow template, and a troubleshooting page with sample `plugins info` and startup diagnostics.
