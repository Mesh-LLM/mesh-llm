# mesh-llm-plugin

`mesh-llm-plugin` owns the plugin author API and shared plugin wire protocol
types used by host-side plugin runtimes and external plugins.

This crate includes:

- the plugin protobuf schema at `proto/plugin.proto`
- generated plugin protocol types exposed through `mesh_llm_plugin::proto`
- typed helpers for plugin manifests, operations, resources, prompts, tasks,
  mesh events, HTTP bindings, MCP projections, host-projected web UI bundles,
  and side-stream I/O

Plugins that declare web UI use the `web_ui`, `web_ui_bundle`, `web_ui_page`,
and `web_ui_config_section` manifest builders. Declarative runtimes can place
config schemas under `config: [...]` and UI declarations under `web_ui: [...]`
in `plugin!`, alongside their real handlers. Macro fields remain in declaration
order: `metadata`, optional `startup_policy`, `provides`, `config`, `web_ui`,
`mesh`, `events`, `mcp`, `http`, `inference`, then lifecycle hooks. The v1
contract is local, same-origin, trusted, and projection-only; see
[`docs/plugins/README.md`](../../docs/plugins/README.md) and its maintained
[`web UI exemplar`](../../docs/plugins/exemplars/web-ui/README.md).

Host-only orchestration, process lifecycle, plugin config loading, MCP bridge
hosting, and built-in plugin wiring should remain in the host crate for now and
move later to a dedicated plugin-host crate. Keep this crate focused on the
stable API and protocol surface that plugin authors can depend on.
