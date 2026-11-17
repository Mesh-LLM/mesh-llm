---
title: Plugins
---

# Plugins

Plugins add capabilities to mesh-llm without putting every integration into the core runtime. A plugin is a native process managed by mesh-llm over a local control connection. The host owns lifecycle, MCP and HTTP exposure, routing, and policy; the plugin owns its feature logic and local state.

Use this page to install a plugin. If you are writing one, start with [Developing Plugins](/docs/pages/developing-plugins/).

## Released plugins

The [Extending Mesh directory](/plugins/) describes {{ plugins.count }} plugins with published native binaries: alternative inference engines, agent tools, evidence, observability and Lightning payments. It lists each release's platforms, prerequisites and preview limits. Mesh {{ plugins.hostVersion }} does not bundle plugins; install the integrations you need separately.

These are the release references checked on {{ plugins.checkedOn }}. Pass a GitHub reference to `mesh-llm plugins install`; use the downloaded-archive instructions where indicated. The installed identifier can differ from the repository name.

| Installed identifier / release | Pinned GitHub reference |
| --- | --- |
{% for group in plugins.groups -%}
{% for plugin in group.plugins -%}
| [`{{ plugin.name }}`](https://github.com/Mesh-LLM/{{ plugin.repository }}#readme)<br>[{{ plugin.version }}](https://github.com/Mesh-LLM/{{ plugin.repository }}/releases/tag/{{ plugin.version }}) | {% if plugin.archiveInstall %}[Download archive](#install-a-downloaded-archive){% else %}`Mesh-LLM/{{ plugin.repository }}@{{ plugin.version }}`{% endif %} |
{% endfor -%}
{% endfor %}

The catalog is a separate discovery source and may contain fewer entries or pin a different release. Search it before installing an unfamiliar integration:

```bash
mesh-llm plugins search
mesh-llm plugins search database
```

Read the repository, release notes and required permissions before installing. Plugins are trusted native code, **not sandboxed extensions**. An available archive is not a host/plugin compatibility guarantee.

## Install a plugin

Install a catalog entry's selected release for the current machine:

```bash
mesh-llm plugins install openai-endpoint
```

Install directly from GitHub when a plugin is not in the catalog, or pin a known release:

```bash
mesh-llm plugins install Mesh-LLM/openai-endpoint-plugin
mesh-llm plugins install Mesh-LLM/openai-endpoint-plugin@v0.2.0
```

Use the catalog name, such as `agents`, for catalog installs. Use the fully qualified `owner/repository` form, such as `Mesh-LLM/agents-plugin`, for GitHub installs. Check compatibility before updating an older host.

The installer selects a native release archive using the plugin name, version, operating system and CPU architecture. It does not select GPU-specific variants or negotiate the plugin protocol. If there is no archive for your target, build from the repository and configure the binary with `command`.

### Install a downloaded archive

Download the archive for your OS and CPU from the linked release page and check its accompanying checksum when provided. Then install it with its declared plugin name. For example, with DwarfStar's Apple Silicon archive:

```bash
mesh-llm plugins install --archive ./ds4-plugin-aarch64-apple-darwin.tar.gz \
  --name ds4-plugin --version v0.2.0
```

On Mesh {{ plugins.hostVersion }}, use this route for the DwarfStar, Uzu and Slotstream releases in the table. Their stable archive filenames retain the `-plugin` suffix that the GitHub installer strips from a repository name; those releases do not provide versioned filenames for its fallback matching.

`--archive` also supports locally built packages. It accepts `.tar.gz` or `.zip`, requires `--name`, and cannot be combined with a catalog/GitHub reference. `--version` is optional and defaults to `dev`. Download and reinstall an archive to update it; `plugins update` is for GitHub release sources.

After installing, inspect the recorded source, version, target and path:

```bash
mesh-llm plugins list
mesh-llm plugins info openai-endpoint
```

Installing an engine plugin does not select or download its models, and installing a wallet does not authorize spending. Follow the plugin's setup instructions before starting Mesh.

## Configure a plugin

Installed, enabled plugins are discovered automatically when Mesh starts. Add a `[[plugin]]` table to `~/.mesh-llm/config.toml` when the plugin needs arguments, settings or explicit permissions. For an external endpoint:

```toml
[[plugin]]
name = "openai-endpoint"
url = "http://127.0.0.1:8000/v1"
```

## External-endpoint-only workflow

**Start your provider → install `openai-endpoint` → set its URL → run
`mesh-llm serve`.** No GGUF, placeholder model, or separate `share` command.
Use `serve` on the provider machine and `client --join` on a consuming machine.

See [Share Ollama, vLLM or LM Studio](/docs/pages/external-model-endpoints/) for
the short setup, provider-specific URLs, a completion check and troubleshooting.

**Release compatibility:** adapter **0.2.0** uses plugin protocol 3, as does Mesh {{ plugins.hostVersion }}. Adapter 0.1.2 uses protocol 2 and cannot initialize on that host, despite installing successfully. Check the [adapter compatibility/build instructions](https://github.com/Mesh-LLM/openai-endpoint-plugin/blob/v0.2.0/README.md#compatibility) before installing; do not assume “latest” means compatible. Older protocol-2 hosts must retain a compatible adapter such as 0.1.2 rather than blindly updating.

See [Runtime Lifecycle](/docs/pages/runtime-lifecycle/) for zero-model and
on-demand behavior, and [OpenAI-Compatible API](/docs/pages/openai-compatible-api/)
for client semantics.

## Other plugin configurations

For plugins that do not need an endpoint, the name is usually enough:

```toml
[[plugin]]
name = "blackboard"

[[plugin]]
name = "metrics"

[telemetry]
enabled = true
endpoint = "https://otel.example.com"
```

For Flash-MoE, see the [Flash-MoE repository](https://github.com/Mesh-LLM/flash-moe-plugin) for both existing-endpoint and managed-process examples. For agents, see the [agents repository](https://github.com/Mesh-LLM/agents-plugin) for agent definitions and runtime configuration.

### Plugin fields

| Field | Meaning |
| --- | --- |
| `name` | Installed plugin identifier. Required. |
| `enabled` | Start the plugin with mesh-llm. Defaults to `true`. |
| `web_ui_enabled` | Show a declared plugin web UI in the console. Defaults to `true` when omitted; it does not start or stop the plugin process. |
| `web_ui_primary_tab` | Place a declared plugin web UI in the primary tab when the operator enables it. Defaults to `false` when omitted. |
| `allow_peer_blocks` | Let the plugin ask this node to stop (or resume) routing to a peer. Defaults to `false` when omitted, and requests are refused. |
| `allow_inference_observation` | Let the plugin follow what this node executes (requests, timing and token counts, never prompts or answers). Defaults to `false` when omitted, and its subscription is refused. Read when the plugin subscribes. |
| `command` | Explicit executable path or command. Useful for locally built plugins. |
| `args` | Arguments passed to the plugin process. |
| `url` | Optional endpoint passed to the plugin as `MESH_LLM_PLUGIN_URL`. |
| `models` | Model IDs this plugin's inference endpoints may share. Mesh does not advertise or route the other models the endpoint lists. Omit it to share every listed model. |
| `settings` | Plugin-specific settings declared by the installed plugin. Use `[plugin.settings]` in TOML. |
| `[plugin.startup]` | Startup and failure behavior. |

Startup options include `connect_timeout_secs`, `init_timeout_secs`, `optional`, and `lazy_start`. Mark an integration `optional = true` when mesh-llm should continue starting if that plugin is unavailable. Use `lazy_start = true` for plugins that should start only when directly used.

Plugin settings are validated against the schema exposed by the installed plugin. For example:

```toml
[[plugin]]
name = "my-plugin"

[plugin.settings]
mode = "strict"
retention_days = 14
```

### Plugin web UI

Some plugins declare local, host-projected console pages or a configuration
section. Their process lifecycle and UI projection are independent. To hide a
declared UI while retaining the plugin's MCP, HTTP, and endpoint capabilities:

```toml
[[plugin]]
name = "example-plugin"
enabled = true
web_ui_enabled = false
```

You can also change this preference in the console's **Configuration →
Plugins** tab. The toggle persists only `web_ui_enabled`; it does not install,
start, stop, or disable the plugin. A ready plugin page is available at the
static console path `/plugins/<plugin-name>/<page-id>`. Invalid or missing UI
assets leave the plugin process and non-UI capabilities available.

## Plugin storage

By default, mesh-llm stores plugin metadata under `~/.mesh-llm/plugins/` and extracted plugin files under `~/.mesh-llm/plugins/installed/<name>/`. The metadata record at `~/.mesh-llm/plugins/<name>/plugin-install.json` records the source, version, target, install path, enabled state, and packaged settings schema.

Set `MESH_LLM_PLUGIN_DIR` to use a different store root. With that override, the corresponding paths are `<root>/<name>/plugin-install.json` and `<root>/installed/<name>/`. The extracted directory contains the plugin executable, documentation, and any bundled `skills/` directories. Use `mesh-llm plugins info <name>` to confirm the resolved install path before inspecting files manually.

## Use plugin features

The host aggregates plugin capabilities into the local console and MCP endpoint. Plugin-owned names are namespaced to prevent collisions:

- Blackboard exposes tools such as `blackboard.feed` and `blackboard.post`, plus the `blackboard://snapshot` resource.
- Agents exposes tools such as `agents.get_agents`, `agents.send_message`, and `agents.get_task`.
- Endpoint plugins register inference services that mesh-llm can route to when their endpoint is healthy.

Plugins may also ship Agent Skills. After installing a plugin that includes skills, run:

```bash
mesh-llm skills install
```

The supported agent launchers (`mesh-llm goose`, `mesh-llm pi`, `mesh-llm opencode`, and `mesh-llm claude`) install available plugin skills for that client before starting it.

## Manage installed plugins

```bash
mesh-llm plugins update openai-endpoint
mesh-llm plugins enable openai-endpoint
mesh-llm plugins disable openai-endpoint
mesh-llm plugins delete openai-endpoint
```

Disabling keeps the extracted files and metadata on disk but prevents startup. Deleting removes the extracted files and local metadata. Restart mesh-llm after changing a plugin's configuration unless the plugin's own documentation says that it can be reloaded dynamically.

## Troubleshoot startup

Start with the plugin state:

```bash
mesh-llm plugins info <name>
```

Then check the following:

- `command` points to an executable that exists and can run on this machine.
- `url` includes the correct `/v1` path for OpenAI-compatible endpoints.
- The plugin is enabled and has a compatible release for the local target.
- The endpoint is reachable from the mesh node, not only from your laptop.
- A plugin with required settings is installed before those settings are added to `config.toml`.
- An endpoint can be unhealthy while its plugin process is healthy; check the endpoint itself before reinstalling the plugin.

For the protocol and host/plugin boundary, read [Plugin Architecture](/docs/pages/plugin-architecture/). For the author API and release contract, read [Developing Plugins](/docs/pages/developing-plugins/).

## Further plugin documentation

| Page | What it covers |
| --- | --- |
| [Plugin Architecture](/docs/pages/plugin-architecture/) | Control sessions, side streams, host projections, and ownership boundaries |
| [Developing Plugins](/docs/pages/developing-plugins/) | Rust author API, manifests, packaging, skills, and testing |
| [Plugin Reference](/docs/pages/plugin-reference/) | MCP, HTTP, inference, capabilities, mesh channels, and control messages |
| [Plugin Web UI exemplar](https://github.com/Mesh-LLM/mesh-llm/tree/main/mesh/docs/plugins/exemplars/web-ui) | A maintained manifest, package, config, bundle, and lifecycle reference |
