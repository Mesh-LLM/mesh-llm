# Plugin Web UI Exemplar

This source-owned exemplar is the maintained reference for a v1 plugin web UI
package. Tests read these files directly so the sample manifest, bundle
contract, host-config setting, lifecycle states, and non-UI capability cannot
silently drift from the implementation.

## Files

| File | Purpose |
|---|---|
| `manifest.rs` | Author-side Rust manifest builder sample. |
| `plugin.package.json` | Package manifest JSON consumed by archive/install validation. |
| `config.toml` | Operator config sample showing `web_ui_enabled` independent from plugin process `enabled` and a plugin setting persisted through host config. |
| `bundle/register-mesh-plugin-ui.ts` | Typed author bundle source using `registerMeshPluginUi(host)` and explicit mount/unmount handles. |
| `lifecycle-states.json` | Canonical state examples for `none`, `ready`, `disabled`, `invalid`, and `plugin_not_running`. |

## Contract Summary

The manifest declares one local bundle id/root, one page, and one Integrations
config section. Bundle paths are package-relative only. Page `route` values are
slugs, not paths or URLs. Do not use remote URL schemes, absolute paths,
traversal segments, unknown `bundle_id` references, or multiple bundle roots.

The bundle exports `registerMeshPluginUi(host)` and returns handlers for:

- page id `overview`
- config section id `retention`

Both handlers return an object with `unmount()`. Unmount removes DOM content and
unsubscribes from host state updates.

The config section demonstrates the narrow host surface for persisted settings:
it reads `host.config.visible.settings.retention_days`, calls
`host.config.requestMutation(...)` with the plugin-owned setting key
`retention_days`, then updates the input from the returned visible config. The
host owns persistence and validation through `/api/plugins/:plugin/web-ui/config`
and the existing plugin config schema. The bundle does not write config files
directly.

The non-UI capability `exemplar.notes.v1` remains present in lifecycle samples
when web UI projection is disabled or invalid. Disabling web UI is projection
only; it does not stop plugin process capabilities.

## Operator Remediation

If the console or `/api/plugins/<plugin>/web-ui` reports `invalid`:

1. Inspect the plugin package and confirm the bundle root from
   `plugin.package.json` exists under the installed plugin root.
2. Confirm the page and config-section `entry_script` files exist inside that
   bundle root after the plugin is built.
3. Remove remote URL schemes, absolute paths, or `..` traversal from `web_ui`
   paths. V1 only serves local package assets.
4. Keep a single bundle root. Split files inside that root instead of declaring
   multiple roots.
5. For config sections in the console, use `parent_tab = "integrations"` or omit
   `parent_tab`.
6. Reinstall or update the plugin package, then restart or reload mesh-llm so the
   installed metadata is revalidated.

If assets are missing, the plugin can still run and advertise non-UI
capabilities. Fix the package contents rather than disabling the plugin process
unless the non-UI behavior is also broken.

## Persisted Setting Reproduction

1. Open the exemplar in the Configuration `Plugins` tab.
2. Confirm the Retention config section shows the current `retention_days`
   value from `host.config.visible.settings`.
3. Change the value and click `Save retention`.
4. Confirm the host sends `PATCH /api/plugins/web-ui-exemplar/web-ui/config`
   with only `settings.retention_days`.
5. Confirm the saved config changes `[[plugin]].settings.retention_days` while
   leaving `enabled` and `web_ui_enabled` unchanged.
6. Refresh plugin metadata and confirm non-UI capability `exemplar.notes.v1`
   remains represented.
