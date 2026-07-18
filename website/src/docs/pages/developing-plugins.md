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
| `web_ui` | Local console pages or a configuration section projected by the host | `web_ui`, `web_ui_bundle`, `web_ui_page`, `web_ui_config_section` |

MCP and HTTP are host projections. A plugin does not need to implement an MCP server, HTTP server, or socket protocol itself.

## Start a Rust plugin

Create a binary crate and depend on `mesh-llm-plugin` at a version compatible with the mesh-llm release you target. Keep the host and plugin protocol versions aligned; the host rejects incompatible initialization handshakes.

A minimal standalone `Cargo.toml` for the current release line is:

```toml
[package]
name = "hello-plugin"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = "1"
mesh-llm-plugin = "0.72.1"
schemars = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

During mesh-llm development, replace the published dependency with a path to
the checkout's `crates/mesh-llm-plugin`; do not import host or console crates.

The smallest useful plugin has a manifest, one handler, and a runtime entrypoint:

```rust
use mesh_llm_plugin::{
    mcp, plugin, plugin_server_info, Plugin, PluginMetadata, PluginRuntime,
};

fn build_plugin() -> impl Plugin + Clone + Sync + 'static {
    plugin! {
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
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let plugin = build_plugin();
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

Declare the schema in the same `plugin!` invocation as the plugin's real
handlers. The host receives this manifest during initialization and uses it to
validate `[plugin.settings]`:

```rust
use mesh_llm_plugin::{
    PluginMetadata, config_integer, config_schema, config_setting,
    constraint_range, mcp, plugin, plugin_server_info,
};

let plugin = plugin! {
    metadata: PluginMetadata::new(
        "notes",
        "1.0.0",
        plugin_server_info("notes", "1.0.0", "Notes", "Shared notes services", None::<String>),
    ),
    config: [
        config_schema("notes").setting(
            config_setting("retention_days", config_integer())
                .default_value(&14)
                .constraint(constraint_range(Some("1"), Some("365")))
                .description("How long to retain entries."),
        ),
    ],
    mcp: [
        mcp::tool("status")
            .description("Return plugin status")
            .handle(|_args, _context| Box::pin(async {
                Ok(serde_json::json!({ "status": "ok" }))
            })),
    ],
};
```

`plugin!` accepts `config: [...]` and `web_ui: [...]` alongside `mcp`, `http`,
`inference`, `provides`, `mesh`, and `events`; do not build a second plugin just
to add its schema or UI. The same manifest can be passed to
`InternalRpcPluginBuilder::with_manifest` for host-private internal-RPC
plugins. See the [`config_schema` and `config_setting` helpers](https://github.com/Mesh-LLM/mesh-llm/blob/main/crates/mesh-llm-plugin/src/manifest.rs) for the complete schema surface.

The macro fields are ordered. When present, use `metadata`, `startup_policy`,
`provides`, `config`, `web_ui`, `mesh`, `events`, `mcp`, `http`, `inference`,
then lifecycle hooks in that order. A field may be omitted, but later fields
cannot move ahead of earlier ones.

Prefer typed settings with useful descriptions, defaults, constraints, and explicit restart scope. Do not make users put plugin settings at the top level of `config.toml`.

## Add a host-projected web UI

Web UI is an additive, local package projection. The host serves validated
bundle files from the installed plugin package on its own origin; it does not
accept remote assets, iframe a plugin, or grant a generic event bus. Keep the
plugin process lifecycle separate from its optional UI projection:

```rust
use mesh_llm_plugin::{web_ui, web_ui_bundle, web_ui_config_section, web_ui_page};

// Add this field to the same plugin! invocation that owns the handlers.
web_ui: [
    web_ui()
        .bundle(web_ui_bundle("main", "bundle"))
        .page(
            web_ui_page("overview", "Overview", "overview", "register-mesh-plugin-ui.js")
                .bundle_id("main"),
        )
        .config_section(
            web_ui_config_section("settings", "Settings", "register-mesh-plugin-ui.js")
                .parent_tab("integrations")
                .bundle_id("main"),
        ),
],
```

For v1, declare one non-empty bundle id rooted in a directory below the package
root. Every page and config section needs a non-empty id and label/title, must
reference that bundle, and must name an existing non-empty entry script inside
it. Page `route` is a slug, not a path or URL. `parent_tab` is either omitted or
`"integrations"`.

The bundle is a browser ES module that exports `registerMeshPluginUi(host)` and returns page and
optional config-section mount handlers. Each handler returns `{ unmount() {} }`
and must release DOM content and subscriptions. Read current plugin settings
from `host.config.visible.settings`; persist plugin-owned setting changes with
`host.config.requestMutation(...)`, never by writing `config.toml` directly.

Copy the exemplar's self-contained
[`host-contract.d.ts`](https://github.com/Mesh-LLM/mesh-llm/blob/main/docs/plugins/exemplars/web-ui/bundle/host-contract.d.ts)
for TypeScript authoring. Do not import types from `crates/mesh-llm-ui`; that is
private console source, not a plugin SDK path. Ship browser-importable
JavaScript: the host does not transpile TypeScript, JSX, CommonJS, or unresolved
bare npm imports at runtime.

A directly shippable JavaScript bundle can use only browser APIs and the host
object. This example registers both declared ids, persists a setting through
the host, and returns cleanup handles:

```js
export async function registerMeshPluginUi(host) {
  return {
    pages: {
      overview({ element, page }) {
        const heading = document.createElement('h2')
        heading.textContent = page.label
        element.replaceChildren(heading)
        return { unmount() { element.replaceChildren() } }
      }
    },
    configSections: {
      settings({ element }) {
        const input = document.createElement('input')
        input.type = 'number'
        input.value = String(host.config.visible.settings.retention_days ?? 14)
        const save = document.createElement('button')
        save.textContent = 'Save retention'
        const onSave = async () => {
          const visible = await host.config.requestMutation({
            settings: { retention_days: Number(input.value) }
          })
          input.value = String(visible.settings.retention_days)
          host.notifications.show({ title: 'Retention saved', tone: 'success' })
        }
        save.addEventListener('click', onSave)
        element.replaceChildren(input, save)
        return {
          unmount() {
            save.removeEventListener('click', onSave)
            element.replaceChildren()
          }
        }
      }
    }
  }
}
```

The narrow host network helpers accept plugin-relative paths such as
`http/items?limit=2`. They reject origins, fragments, backslashes, and dot
segments so a bundle cannot escape `/api/plugins/<plugin>/...` through the
helper. `host.network.json(...)` rejects non-2xx responses; use
`host.network.fetchPlugin(...)` when you need to inspect the raw status.

Ready pages use the static host route
`/plugins/<plugin-name>/<page-id>`. Config sections appear in **Configuration →
Plugins → Integrations**. Operators can set `web_ui_enabled = false` or use the
console toggle to hide that projection without affecting the plugin's other
capabilities.

The maintained [web UI exemplar](https://github.com/Mesh-LLM/mesh-llm/tree/main/docs/plugins/exemplars/web-ui)
includes a matching author manifest, packaged metadata, config, typed bundle,
and lifecycle-state reference. Use it as the release check for this contract.

## Package and release a plugin

The installer accepts native GitHub Release archives and local release
archives. Each archive should contain one directory named after the plugin:

```text
hello-plugin/
  plugin.toml
  plugin-manifest.json
  hello-plugin
  bundle/
    register-mesh-plugin-ui.js
  README.md
  LICENSE
  skills/
    hello-workflow/
      SKILL.md
```

On Windows, the executable is `hello-plugin.exe`. The required files are `plugin.toml` and the executable. A plugin that declares a web UI must also ship
`plugin-manifest.json`, its declared bundle directory, and every declared entry
script. Documentation, license files, and skills are recommended.

`plugin.toml` identifies the archive root:

```toml
name = "hello-plugin"
```

Expose a deterministic manifest-export option from `main` so the packaged
metadata is generated by the same `build_plugin` declaration above that the
runtime registers. Replace the earlier `main` with:

```rust
use anyhow::{Context, bail};
use mesh_llm_plugin::package_manifest_json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let plugin = build_plugin();
    match std::env::args().nth(1).as_deref() {
        Some("--print-package-manifest") => {
            let manifest = plugin.manifest().context("plugin manifest")?;
            println!("{}", package_manifest_json(&manifest)?);
            Ok(())
        }
        Some(argument) => bail!("unknown option: {argument}"),
        None => PluginRuntime::run(plugin).await,
    }
}
```

Build and package the exact local archive from the crate root:

```bash
cargo build --release
rm -rf target/package
mkdir -p target/package/hello-plugin
cp target/release/hello-plugin target/package/hello-plugin/hello-plugin
cp plugin.toml target/package/hello-plugin/plugin.toml
cp -R bundle target/package/hello-plugin/bundle
target/release/hello-plugin --print-package-manifest \
  > target/package/hello-plugin/plugin-manifest.json
tar -C target/package -czf target/hello-plugin-0.1.0-local.tar.gz hello-plugin
```

On Windows, copy the `.exe` and create a `.zip` whose single top-level
directory is `hello-plugin/`.

Publish archives using the target triple in the filename:

```text
hello-plugin-0.1.0-aarch64-apple-darwin.tar.gz
hello-plugin-0.1.0-x86_64-unknown-linux-gnu.tar.gz
hello-plugin-0.1.0-x86_64-pc-windows-msvc.zip
```

The current installer targets macOS Apple Silicon and Intel, Linux ARM64 and x86_64, and Windows ARM64 and x86_64. Release automation should build and test every archive it publishes.

Before publishing, install the exact local archive through the same validation
boundary used for downloaded releases:

```bash
mesh-llm plugins install --archive ./hello-plugin-0.1.0-local.tar.gz \
  --name hello-plugin --version 0.1.0
```

`--archive` accepts `.tar.gz` or `.zip`, requires `--name`, and conflicts with
the positional GitHub/catalog reference. `--version` is optional and defaults
to `dev`. A local install is not updateable through `plugins update`; rebuild
and reinstall it.

Enable the installed plugin in a config file, initialize an owner identity,
then start a client-only node. Owner identity is required for console and API
configuration writes, including `host.config.requestMutation(...)`:

```toml
version = 1

[[plugin]]
name = "hello-plugin"
enabled = true
web_ui_enabled = true

[plugin.settings]
retention_days = 14
```

```bash
mesh-llm auth status
# For a local development identity when status reports none:
mesh-llm auth init --no-passphrase
mesh-llm client --port 9337 --console 3131 --config ./config.toml
```

Inside the mesh-llm repository, the equivalent debug-binary launch is:

```bash
MESH_LLM_BIN=target/debug/mesh-llm just mesh-client "" 9337 3131 ./config.toml
```

`just mesh-client` otherwise uses `target/release/mesh-llm`. When testing with
an isolated `HOME` or `MESH_LLM_PLUGIN_DIR` on Unix, keep the base path short:
the local plugin control socket is subject to the platform's socket-path limit.

The management console exposes projected plugin MCP tools at `/mcp`. For a raw
transport check, initialize a session, send the initialized notification, then
list and call the namespaced tool. Copy the returned `mcp-session-id` response
header into `SESSION`:

```bash
curl -i -X POST http://127.0.0.1:3131/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  --data '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"plugin-check","version":"1.0"}}}'

SESSION='<mcp-session-id response header>'
curl -i -X POST http://127.0.0.1:3131/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SESSION" \
  --data '{"jsonrpc":"2.0","method":"notifications/initialized"}'
curl -i -X POST http://127.0.0.1:3131/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SESSION" \
  --data '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
curl -i -X POST http://127.0.0.1:3131/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SESSION" \
  --data '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"hello-plugin.ping","arguments":{}}}'
```

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
2. The manifest exposes the expected MCP, HTTP, inference, capability, channel, event, and web UI entries.
3. Invalid input returns a useful error without taking down the control session.
4. Health still responds while a handler is running.
5. Streaming and cancellation do not leak side streams.
6. A stopped endpoint becomes unavailable without incorrectly disabling the plugin.
7. The release archive extracts to the expected directory and runs on each target.
8. A mixed-version host rejects incompatible protocol versions clearly and accepts the versions you support.
9. If the plugin declares web UI, test ready, disabled, invalid/missing bundle,
   and plugin-not-running states without losing non-UI capabilities.

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
