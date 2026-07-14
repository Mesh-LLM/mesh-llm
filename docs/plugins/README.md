# Plugins

Use this architecture reference to build and review `mesh-llm` plugins.

It describes the target architecture, not just the code as it exists today.

As implementation lands, this document should be updated to match the intended end state and the concrete protocol and runtime decisions that have been made.

Plugin-specific documentation:

- [Flash-MoE](flash-moe.md) - external OpenAI-compatible backend adapter for single-node SSD expert streaming
- [Telemetry](telemetry.md) - OTLP metrics-only runtime telemetry and external metrics plugin notes
- [Web UI exemplar](exemplars/web-ui/README.md) - source-owned maintainer sample for v1 plugin web UI projection, read directly by tests to catch drift

The main goals are:

- keep `mesh-llm` decoupled from specific plugins
- let bundled plugins be auto-registered without special-casing product behavior
- make MCP and HTTP first-class host projections
- support large request and response bodies without blocking control traffic
- keep plugin author boilerplate low

## Design Summary

A plugin is a local service process launched by `mesh-llm`.

The system has three core pieces:

- one long-lived control connection per plugin process
- zero or more short-lived negotiated streams for large or streaming data
- one declarative plugin manifest that the host `stapler` projects into MCP, HTTP, and optional promoted product APIs

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

Plugins do not need to implement raw MCP or raw HTTP servers.

The `stapler` is the host projection layer that turns plugin manifests into exposed MCP and HTTP surfaces.

## Launch Contract

When `mesh-llm` launches an external plugin, it provides the host connection
details through environment variables:

| Variable                    | Meaning                                     |
| --------------------------- | ------------------------------------------- |
| `MESH_LLM_PLUGIN_ENDPOINT`  | Local IPC endpoint the plugin connects to   |
| `MESH_LLM_PLUGIN_TRANSPORT` | Transport kind, such as `unix` or `pipe`    |
| `MESH_LLM_PLUGIN_NAME`      | Configured plugin name                      |
| `MESH_LLM_PLUGIN_URL`       | Optional `[[plugin]].url` value from config |

Plugin-specific configuration should live in the plugin process or use generic
plugin config fields. The host should not special-case behavior for a plugin by
repository or package name.

## Plugin Web UI Projection Contract

For the maintained source-owned sample, start with
[`docs/plugins/exemplars/web-ui/`](exemplars/web-ui/README.md). The manifest,
bundle, config, and lifecycle fixtures there are part of the contract, and the
tests read them directly so the docs and implementation stay aligned.

### Manifest Fields

The manifest's `web_ui` block is additive. It declares the local bundle roots,
page entries, and optional configuration-section entries that the host may
project.

Use the typed builders from `mesh_llm_plugin::manifest`:

- `web_ui`
- `web_ui_page`
- `web_ui_config_section`
- `web_ui_bundle`

Rules for the declared bundle paths:

- keep paths package-relative and below the package root; do not use an empty
  path or `.` as a bundle root
- declare exactly one non-empty bundle id and one bundle root for v1 whenever
  the block declares pages or config sections
- set every page and config-section `bundle_id` to that declared bundle id
- give every page and config section a non-empty id and display label/title
- keep page `route` values as slugs, not paths or URLs; do not include `/`,
  `\`, protocol syntax, or traversal-style dot prefixes
- keep non-empty page and config-section entry scripts inside that root; the
  installed package must contain each declared entry script
- reject remote URL schemes, absolute paths, and traversal segments
- treat `parent_tab = "integrations"` as the only supported parent-tab value for
  config sections, or omit `parent_tab`

The manifest proto keeps the bundle field repeated as a forward-compatible wire
shape. V1 validation intentionally permits only one bundle root so the host has
one deterministic `asset_base_url` for page and config-section imports. An
empty `web_ui` block contributes no usable console surface.

### State Matrix

The host exposes exactly these web UI state names:

| State                | Meaning                                                                                      |
| -------------------- | -------------------------------------------------------------------------------------------- |
| `none`               | The plugin does not declare a web UI projection.                                             |
| `ready`              | The manifest and installed bundle are valid, and the host may mount it.                      |
| `disabled`           | The projection is installed and valid, but the persisted `web_ui_enabled` preference is off. |
| `invalid`            | The manifest or installed bundle failed validation, or the bundle root is missing.           |
| `plugin_not_running` | The plugin process is stopped, but installed metadata still carries the projection state.    |

### API And Console Surface

The web UI API uses the existing plugin namespace and these exact routes:

- `GET /api/plugins/:plugin/web-ui`
- `PATCH /api/plugins/:plugin/web-ui/enabled`
- `GET /api/plugins/:plugin/web-ui/config`
- `PATCH /api/plugins/:plugin/web-ui/config`
- `GET /api/plugins/:plugin/web-ui/assets/*asset`

The toggle route changes only the persisted `web_ui_enabled` projection
preference. It does not start, stop, or disable the plugin process.

Asset delivery is host-owned and same-origin. It serves only validated installed
bundle assets and only when the projection is `ready`. Console mounts use the
backend-provided `asset_base_url` from the web UI state; a ready state without
that URL is treated as a host error and no bundle code is imported.
Plugin assets are revalidated by the browser rather than cached as immutable,
because local reinstall and same-version development builds may replace them.

The config route is also host-owned and plugin-scoped. `GET` returns:

```json
{
  "plugin": "example",
  "settings": { "retention_days": 30 },
  "schema": { "plugin_name": "example" }
}
```

`PATCH` accepts a settings-only mutation:

```json
{
  "plugin": "example",
  "settings": { "retention_days": 45 },
  "unset": ["old_setting"]
}
```

The `plugin` field must match the mounted plugin when present. Mutations may
only touch plugin-owned `settings` keys; host-owned fields such as `enabled`,
`web_ui_enabled`, `command`, `args`, `url`, and `startup` are rejected.
Malformed requests return `400`; schema-invalid setting values return `422`;
successful mutations return the newly visible plugin config.

The console route is static TanStack routing, not dynamic route injection:

- `/plugins/$pluginName/$pageId`

Plugin routes do not become a new primary `AppTab`. A ready plugin with one
declared page receives a direct auxiliary navigation item labeled from its page
manifest. When more than one plugin page is ready, the console groups those
entries under the auxiliary `Plugins` menu to protect header space. Disabled,
invalid, or stopped projections contribute no navigation item. The existing
Configuration `Plugins` tab owns config-section projection, and only ready
config sections in the `integrations` projection mount there.

Plugin-owned settings declared in `config_schema` continue to render through
the console's standard schema controls. A custom config-section bundle should
add plugin-specific actions or context; it should not recreate a schema field
with unstyled DOM controls or bypass the host-owned configuration save flow.

### Bundle Contract

Typed author bundles export `registerMeshPluginUi(host)` and return mount
handlers for pages and config sections.

- each mount handler returns an object with `unmount()`
- `unmount()` must tear down DOM content and detach host subscriptions
- read current plugin settings from `host.config.visible.settings`
- config sections use `host.config.requestMutation(...)`, not direct file writes
- configuration mutations require a local owner identity; initialize one with
  `mesh-llm auth init --no-passphrase` for an unencrypted development identity
- mutation errors reject the returned promise and are rendered by the host shell
- user-visible notices can be emitted with `host.notifications.show(...)`
- `host.network.fetchPlugin(path, init)` and `host.network.json(path, init)`
  accept plugin-relative API paths such as `http/items?limit=2`; origins,
  fragments, backslashes, and `.`/`..` path segments are rejected
- `host.network.json(...)` rejects non-2xx responses; use `fetchPlugin(...)`
  when the bundle needs to inspect a non-success status itself
- registrations must return a `pages` object, optional `configSections` object,
  and `{ unmount() }` from every mounted handler; malformed results surface as
  host contract errors rather than failing later during cleanup
- the host imports bundle code only after the projection is ready, enabled,
  available, has a same-origin `asset_base_url`, and the requested page or
  section exists
- ship browser-importable JavaScript; the host does not transpile TypeScript,
  JSX, CommonJS, or unresolved bare npm imports
- use the exemplar's self-contained `bundle/host-contract.d.ts` for author
  types; do not import private types from `crates/mesh-llm-ui`

For a pre-release or local package, exercise the real installer without a
GitHub release:

```bash
mesh-llm plugins install --archive ./cool-plugin-1.1.0-local.tar.gz \
  --name cool-plugin --version 1.1.0
```

`--archive` accepts `.tar.gz` or `.zip`, requires `--name`, and conflicts with
the positional catalog/GitHub reference. `--version` defaults to `dev`. Local
installs are replaced by reinstalling a rebuilt archive; `plugins update`
remains specific to GitHub release sources.

### Validation And Remediation

If the projection is `invalid` or assets are missing:

1. Check the packaged bundle root and the entry script paths in the manifest.
2. Remove remote URLs, absolute paths, and traversal segments.
3. Keep the bundle rooted under one directory and split files inside that root.
4. Confirm config sections either omit `parent_tab` or use `integrations`.
5. Reinstall or update the plugin package, then reload mesh-llm so the
   installed metadata is revalidated.

Invalid or missing web UI assets do not stop the plugin process or its
non-UI capabilities.

### Compatibility Guarantees

The web UI contract is additive and projection-only.

- older nodes can ignore unknown manifest fields
- `web_ui_enabled` stays independent from the plugin process `enabled` flag
- invalid, disabled, missing, and stopped web UIs remain visible in summary and
  API state
- disabling projection does not disable the plugin process

### Non-Goals

This contract does not include:

- iframe or sandbox isolation
- remote UI assets
- marketplace or discovery flow for plugin UIs
- RBAC specific to plugin web UIs
- a generic plugin event bus
- a second schema-driven settings editor inside plugin bundle code (installed
  plugin schemas continue to use the console's existing standard editor)
- dynamic TanStack route mutation
- disabling the plugin process when web UI projection is turned off

## High-Level Model

The plugin system is projection-oriented at the DSL level and service-oriented at the runtime level.

Plugin authors think in terms of the host surfaces they contribute to:

- `mcp`
- `http`
- `inference`
- `provides`
- `mesh`
- `events`
- `web_ui`

The host runtime still executes native service invocations internally, but the author-facing DSL is organized by the surface the plugin contributes to.

This means:

- local MCP tools, resources, prompts, and completions live under `mcp`
- attached external MCP servers also live under `mcp`
- local HTTP routes live under `http`
- attached or plugin-hosted inference backends live under `inference`
- stable product capabilities live under `provides`

There is no separate top-level `services` section in the preferred DSL.

## GitHub Native Package Releases

Third-party plugins may be installed from GitHub repositories.

Supported install references:

```bash
mesh-llm plugins install https://github.com/mesh-llm/cool-plugin
mesh-llm plugins install mesh-llm/cool-plugin
mesh-llm plugins install mesh-llm/cool-plugin@1.1.0
mesh-llm plugins install https://github.com/mesh-llm/cool-plugin@1.1.0
```

If the reference omits a version, the installer resolves the latest compatible
GitHub release. If the reference includes `@<version>`, the installer resolves
that release or tag. Installers should accept both `1.1.0` and `v1.1.0` when
matching a versioned release.

Plugins are distributed as native binary archives. Unlike the main `mesh-llm`
runtime bundles, plugin archives do not carry GPU backend flavors. The release
asset name is selected only by plugin name, version, operating system, and CPU
architecture.

Versioned asset names use:

```text
<plugin-name>-<version>-<target-triple>.<archive-ext>
```

Stable alias asset names may also be published for latest-release installs:

```text
<plugin-name>-<target-triple>.<archive-ext>
```

For a plugin named `cool-plugin`, a `v1.1.0` release may publish:

```text
cool-plugin-v1.1.0-aarch64-apple-darwin.tar.gz
cool-plugin-v1.1.0-x86_64-apple-darwin.tar.gz
cool-plugin-v1.1.0-x86_64-unknown-linux-gnu.tar.gz
cool-plugin-v1.1.0-aarch64-unknown-linux-gnu.tar.gz
cool-plugin-v1.1.0-x86_64-pc-windows-msvc.zip
cool-plugin-v1.1.0-aarch64-pc-windows-msvc.zip
```

The installer should also tolerate the same names without a leading `v` in the
version segment when the release tag itself omits `v`:

```text
cool-plugin-1.1.0-aarch64-apple-darwin.tar.gz
```

Supported target triples:

| Platform            | Target triple               | Archive   |
| ------------------- | --------------------------- | --------- |
| macOS Apple Silicon | `aarch64-apple-darwin`      | `.tar.gz` |
| macOS Intel         | `x86_64-apple-darwin`       | `.tar.gz` |
| Linux x86_64        | `x86_64-unknown-linux-gnu`  | `.tar.gz` |
| Linux ARM64         | `aarch64-unknown-linux-gnu` | `.tar.gz` |
| Windows x86_64      | `x86_64-pc-windows-msvc`    | `.zip`    |
| Windows ARM64       | `aarch64-pc-windows-msvc`   | `.zip`    |

Archive contents should be rooted under one directory named after the plugin:

```text
cool-plugin/
  plugin.toml
  cool-plugin
  README.md
  LICENSE
  skills/
    cool-workflow/
      SKILL.md
```

On Windows, the executable should use `.exe`:

```text
cool-plugin/
  plugin.toml
  cool-plugin.exe
```

Only `plugin.toml` and the native executable are required. Documentation,
license files, and skill folders are optional but recommended when the plugin
has agent-facing workflows.

## Plugin Skills

Installed plugins may expose Agent Skills by shipping skill directories under
their extracted plugin root:

```text
cool-plugin/
  skills/
    cool-workflow/
      SKILL.md
      references/
      scripts/
      assets/
```

Each skill directory name must use the portable Agent Skills naming convention:
lowercase ASCII letters, numbers, and single hyphen separators. `SKILL.md`
should include `name` and `description` YAML frontmatter and should refer to
supporting files with paths relative to the skill directory. Avoid hard-coded
home directories, OS-specific absolute paths, and shell-specific commands unless
the skill documents the required platform in its `compatibility` field.

`mesh-llm skills install` copies plugin skills into detected agent skill
directories. `mesh-llm goose`, `mesh-llm pi`, `mesh-llm opencode`, and
`mesh-llm claude` also install available plugin skills for that launched agent
before starting the session. Existing user-owned skill directories are not
overwritten unless `--force` is passed to the explicit installer.

Current install targets:

| Agent       | Target                      |
| ----------- | --------------------------- |
| Goose       | `~/.agents/skills`          |
| Codex       | `~/.agents/skills`          |
| Pi          | `~/.pi/agent/skills`        |
| OpenCode    | `~/.config/opencode/skills` |
| Claude Code | `~/.claude/skills`          |

Install selection should follow this order:

1. Parse the owner, repository, and optional version from the install reference.
2. Resolve the requested GitHub release.
3. Detect the local target triple.
4. Prefer a versioned asset matching the release tag and local target triple.
5. Fall back to the stable alias for the same target triple.
6. Fail clearly if no compatible asset exists.

Installed plugin metadata should record:

- source repository
- installed version
- target triple
- downloaded asset name
- install path
- enabled or disabled state

The lifecycle commands operate on that metadata:

```bash
mesh-llm plugins update cool-plugin
mesh-llm plugins enable cool-plugin
mesh-llm plugins disable cool-plugin
mesh-llm plugins delete cool-plugin
```

`update` re-resolves the source repository, selects the newest compatible
release asset for the local target triple, and replaces the installed archive
only when a newer compatible version exists.

`enable` marks an installed plugin loadable by the host. `disable` keeps the
plugin installed but prevents host startup from launching it. `delete` removes
the installed archive, extracted files, and local plugin metadata.

## Hugging Face Plugin Catalog

`mesh-llm` may use a simple Hugging Face Dataset as the public plugin catalog.

The catalog is metadata only. It helps users discover plugin GitHub
repositories, but GitHub releases remain the source of native plugin archives
for installs and updates.

The canonical catalog file is `plugins.jsonl`. Each line describes one plugin:

```json
{
  "name": "cool-plugin",
  "description": "Example plugin for mesh-llm.",
  "github_url": "https://github.com/mesh-llm/cool-plugin",
  "author_email": "dev@example.com",
  "author_name": "Mesh LLM"
}
```

Required fields:

| Field          | Meaning                                                                                       |
| -------------- | --------------------------------------------------------------------------------------------- |
| `name`         | Unique plugin name. This should match the plugin manifest ID and GitHub release asset prefix. |
| `description`  | Short human-readable plugin description.                                                      |
| `github_url`   | GitHub repository URL used for install and update resolution.                                 |
| `author_email` | Plugin author or maintainer email.                                                            |
| `author_name`  | Plugin author or maintainer display name.                                                     |

Catalog rules:

- one plugin per JSONL line
- `name` must be unique within the catalog
- unknown extra fields are ignored for forward compatibility
- catalog lookup never downloads native binaries from Hugging Face
- installing a catalog result uses its `github_url` and then follows the GitHub
  native package release flow

The initial public catalog entry should be `blackboard`:

```json
{
  "name": "blackboard",
  "description": "Shared mesh blackboard for agent status, findings, questions, answers, and searchable coordination notes.",
  "github_url": "https://github.com/mesh-llm/blackboard",
  "author_email": "maintainers@meshllm.cloud",
  "author_name": "Mesh LLM"
}
```

The CLI may resolve a bare plugin name through the catalog:

```bash
mesh-llm plugins install cool-plugin
```

Explicit GitHub references bypass catalog lookup:

```bash
mesh-llm plugins install mesh-llm/cool-plugin
mesh-llm plugins install https://github.com/mesh-llm/cool-plugin
```

## Core Principles

### 1. Bundled Plugins Are Allowed

Plugins shipped in this source tree may be auto-registered by the host.

That is acceptable coupling.

What is not acceptable is embedding one plugin's runtime behavior directly into core mesh logic. Core mesh transport and state should stay generic.

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

`mesh-llm` is the MCP server.

Plugins do not need to implement MCP JSON-RPC directly. They declare MCP-facing services in the manifest, and the host `stapler` exposes them over MCP.

### 4. HTTP Is A Host Projection

`mesh-llm` owns the HTTP server.

Plugins may declare HTTP bindings, but they do not need to run an HTTP server themselves. The host `stapler` maps HTTP requests onto plugin operations and resources.

### 5. Capabilities Are Stable Product Contracts

When `mesh-llm` wants a stable product API such as `/api/objects`, core should depend on a named capability like `object-store.v1`, not on a specific plugin ID like `blobstore`.

## Architecture

### Control Session

There is one long-lived control session between host and plugin.

The control session is used for:

- plugin startup and manifest exchange
- health checks
- native service invocation requests and responses
- plugin-to-host notifications
- host-to-plugin mesh events
- opening and closing streams
- cancellation and error reporting

The control session should stay responsive even while the plugin is sending or receiving large payloads.

The native runtime contract is service-oriented, not MCP-oriented.

The host invokes services such as:

- operations
- prompts
- resources
- completions

MCP method names like `tools/call` and `prompts/get` are projection-layer concerns. They are not the preferred host/plugin runtime contract.

### Streams

Streams are short-lived negotiated channels for a single request, response, or transfer.

They are opened via the control session and then carry data independently.

Streams are used for:

- large HTTP request bodies
- large HTTP responses
- streaming uploads and downloads
- server-sent events or similar long-lived responses
- future bulk data flows between host and plugin

On Unix, streams map to short-lived Unix sockets.

On Windows, streams map to short-lived named pipes.

The protocol concept is `stream`, not `socket`, so the transport binding remains platform-specific.

### Why Streams Exist

The current single-socket framed-envelope design is vulnerable to head-of-line blocking. Even chunked transfer traffic still competes with health checks, tool calls, mesh events, and other control messages on the same queue.

This architecture avoids that by separating:

- control plane traffic
- bulk and streaming data traffic

## Manifest

On startup, a plugin returns a manifest that declares what it provides to the host.

Conceptually, the manifest contains:

- plugin identity and version
- provided capabilities
- plugin configuration schemas
- host-projected web UI pages and configuration sections
- MCP contributions
- HTTP contributions
- inference contributions
- any mesh channel and event subscription declarations the plugin needs

The manifest is the source of truth for host projections.

## Plugin Author Experience

The primary design goal is very low boilerplate.

The preferred DSL is surface-first:

- `provides`
- `config`
- `web_ui`
- `mesh`
- `events`
- `mcp`
- `http`
- `inference`

Lifecycle hooks stay local to the plugin definition rather than becoming manifest items:

- `startup_policy`
- `health`
- `on_initialized`
- `on_channel_message`
- `on_mesh_event`

Each section is self-contained. If a plugin contributes something to a host surface, it is declared in the section for that surface.

The `plugin!` macro is order-sensitive. Declare `metadata`, optional
`startup_policy`, `provides`, `config`, `web_ui`, `mesh`, `events`, `mcp`,
`http`, `inference`, then lifecycle hooks. Omitted sections do not need empty
placeholders.

Example:

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

In this model:

- `mcp` contains both local MCP contributions and attached external MCP servers
- `http` contains local HTTP contributions
- `inference` contains both attached external inference endpoints and plugin-hosted inference providers
- `provides` declares stable capability contracts that core product routes can depend on
- `mesh` declares which mesh channels the plugin is allowed to receive and send
- `events` declares which mesh events the host may deliver to the plugin

Event delivery is allowlist-based:

- no `mesh` declaration means no channel delivery
- no `events` declaration means no mesh events
- plugins only receive the event kinds they explicitly declare

The runtime and `stapler` handle:

- schema exposure
- MCP projection
- HTTP projection
- request validation
- stream negotiation
- transport details
- host-side routing and aggregation

Plugin authors should not manually implement:

- MCP `tools/list`
- MCP `tools/call`
- MCP `resources/read`
- HTTP routing
- control-plane socket negotiation

## Internal RPC Plugins

Most plugins should use `plugin!`.

Host-private plumbing services that need raw RPC methods rather than surfaced MCP, HTTP, or inference declarations should use `InternalRpcPluginBuilder`.

This is the escape hatch for internal-only services such as blobstore. It keeps raw host RPC separate from the normal manifest-driven plugin surface.

### Streaming

Streaming is explicit in the DSL.

For HTTP bindings, the preferred modifiers are:

- `.stream_request()`
- `.stream_response()`
- `.sse()`

These declare whether the request body, response body, or response format requires side-stream transport.

## External Endpoints

Plugins may register external services without proxying all traffic through the plugin process.

This is a control-plane declaration, not a request proxying requirement.

In practice:

- attached external MCP servers are declared in the `mcp` section
- attached or plugin-hosted inference backends are declared in the `inference` section

`mesh-llm` then talks to those services directly when appropriate.

This keeps heavy data-plane traffic out of plugin IPC.

### MCP Contributions

The `mcp` section may contain both:

- local MCP-facing items implemented by the plugin
- attached external MCP servers

Preferred external forms include:

- `external_stdio(...)`
- `external_http(...)`
- `external_tcp(...)`
- `external_unix_socket(...)`

External MCP names are namespaced as:

- `plugin_name.method`

### Inference Contributions

The `inference` section may contain both:

- attached external OpenAI-compatible endpoints
- plugin-hosted inference providers

Preferred forms include:

- `openai_http(...)` for attached external endpoints
- `provider(...)` for plugin-hosted backends

### Why Endpoint Registration Exists

Some services already speak a protocol that `mesh-llm` knows how to use directly.

Examples:

- a local OpenAI-compatible inference server
- an external MCP server reachable over stdio, streamable HTTP, Unix socket, named pipe, or TCP
- a plugin-hosted inference runtime such as an MLX-backed local server

In these cases, the plugin should remain the control-plane owner for:

- discovery
- lifecycle
- readiness
- availability

But `mesh-llm` should own the data plane when possible.

### Health And Availability

Endpoint health is separate from plugin health.

If an endpoint health check fails:

- the endpoint becomes unavailable
- the endpoint is removed from routing or aggregation
- the plugin remains loaded
- the plugin is not marked disabled
- the host keeps checking health

If health returns:

- the endpoint becomes available again automatically

This is important because a plugin may be healthy while its managed or discovered service is:

- starting
- restarting
- temporarily unhealthy
- reloading a model
- intentionally stopped

The host should treat plugin liveness and endpoint liveness as separate concerns.

### Recommended State Model

Conceptually, the system should track at least:

- plugin state
- endpoint state
- model or route availability

Suggested plugin states:

- `starting`
- `running`
- `degraded`
- `disconnected`
- `failed`

Suggested endpoint states:

- `unknown`
- `starting`
- `healthy`
- `unhealthy`
- `unavailable`

Suggested routed availability states:

- `advertised`
- `routable`
- `draining`
- `unavailable`

Routing decisions should depend on endpoint health, not just plugin process health.

## MCP

MCP is implemented by the host, not by individual plugins.

The plugin author marks which services should appear in MCP:

- `tool(...)`
- `resource(...)`
- `resource_template_service(...)`
- `prompt(...)`
- `completion(...)`

The host then synthesizes:

- `tools/list`
- `tools/call`
- `resources/list`
- `resources/read`
- `prompts/list`
- `prompts/get`
- completions where applicable

External MCP endpoints may also be aggregated into the host's MCP surface via the `endpoints:` declarations described above.

### MCP Naming

By default, tool, resource, and prompt names should be plugin-namespaced.

Examples:

- tool: `blackboard.feed`
- tool: `blackboard.post`
- resource: `blackboard://snapshot`
- prompt: `blackboard.status_brief`

Friendly aliases may be added for bundled plugins, but the canonical identity should remain namespaced to avoid collisions.

### MCP Streaming

MCP-facing operations may be:

- buffered
- streaming input
- streaming output
- streaming input and output

For streaming operations, the host uses negotiated side streams internally rather than pushing large data through the control connection.

## HTTP Bindings

Plugins may declare HTTP bindings as part of the manifest.

These bindings let a plugin feel native over HTTP without requiring custom host route code for each plugin.

### Default Mounting

Plugin-defined HTTP bindings should be mounted under a plugin-owned namespace by default.

Examples:

- `/api/plugins/blackboard/feed`
- `/api/plugins/blackboard/post`
- `/api/plugins/object-store/objects`

This avoids collisions and keeps plugin-specific APIs out of the top-level product namespace unless explicitly promoted.

### Promoted Product Routes

Some routes may become stable product APIs owned by `mesh-llm`, for example:

- `/api/objects`

These routes should be backed by named capabilities, not by hard-coded plugin IDs.

Example:

- top-level route: `/api/objects`
- required capability: `object-store.v1`
- provider plugin: whichever plugin the host resolves for that capability

This keeps product APIs stable while allowing the backing plugin to change.

External endpoints do not automatically become HTTP routes. They are service registrations that the host may use for routing or aggregation according to their endpoint kind.

### Buffered vs Streamed HTTP

HTTP bindings may be declared as:

- buffered request / buffered response
- streamed request / buffered response
- buffered request / streamed response
- streamed request / streamed response

The host decides whether to keep the invocation on the control channel or negotiate a side stream based on the binding mode and payload size.

## Streams And Large Transfers

Large payloads must not ride the main control connection.

Instead, the control session negotiates a short-lived stream for the transfer.

Conceptual flow:

1. host sends `OpenStream`
2. plugin accepts
3. host and plugin establish a short-lived local stream
4. request or response bytes flow on that stream
5. either side may cancel
6. stream is torn down and cleaned up

This design supports:

- 10 GB uploads
- large downloads
- long-lived streaming responses
- future websocket-like or SSE-style responses

without blocking health checks or other control traffic.

## Suggested Control Messages

The exact wire format is still open, but the protocol should support concepts like:

- `Initialize`
- `InitializeResponse { manifest }`
- `Health`
- `Shutdown`
- `Invoke`
- `InvokeResult`
- `Notify`
- `MeshEvent`
- `OpenStream`
- `OpenStreamResult`
- `CancelStream`
- `StreamError`

The stream protocol itself may be raw bytes or lightly framed bytes, depending on the use case.

## Capabilities

Capabilities let core depend on behavior rather than on plugin names.

Examples:

- `object-store.v1`
- `mesh-blackboard.v1`
- `artifact-cache.v1`
- `model-catalog-provider.v1`

Capabilities are used when:

- core needs a stable product contract
- multiple plugins could satisfy the same role
- the host wants to promote a route into the top-level API

Capabilities are not required for every plugin. They are mainly for shared contracts that `mesh-llm` itself depends on.

Endpoint registration is related but distinct:

- capabilities express stable contracts that core may depend on
- endpoints express concrete service instances that the host can talk to directly

An endpoint may satisfy a capability, but the two ideas should remain separate in the design.

## Mesh Channels

Plugins may declare mesh channels for plugin-specific peer-to-peer coordination.

These should use the generic plugin mesh transport rather than dedicated core stream types for individual plugins.

Core should not embed plugin-specific wire protocols in the main mesh transport when the behavior can live behind the generic plugin channel mechanism.

## What The Host Owns

The host is responsible for:

- launching plugins
- registering bundled plugins
- validating plugin identity
- keeping the control session alive
- stream negotiation and cleanup
- request validation
- HTTP mounting
- MCP exposure
- capability resolution
- route collision detection
- permissions and policy enforcement

## What Plugins Own

A plugin is responsible for:

- declaring its manifest
- implementing handlers
- handling its own local state
- reading and writing stream payloads when invoked
- implementing any plugin-specific business logic

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

The following are intentionally left open for implementation design:

- exact control protocol message shapes
- exact stream framing format
- capability provider selection when multiple plugins implement the same capability
- whether promoted product routes are configured statically or negotiated dynamically
- how auth and policy rules are expressed for plugin-defined HTTP bindings
- how a future multi-bundle or isolated plugin UI version should be negotiated

## Architecture Baseline

- bundled plugins may be auto-registered
- core mesh logic remains plugin-agnostic
- MCP and HTTP are first-class host projections
- product APIs depend on capabilities, not plugin IDs
- large data flows use negotiated side streams, not the control socket
