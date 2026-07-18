# Plugin Web UI Exemplar

This source-owned exemplar is the maintained, runnable reference for a v1
plugin web UI package. It builds as a standalone Rust plugin, generates its
package manifest from the same runtime declaration, installs from a local
archive, adds an auxiliary console navigation item, mounts an interactive page
and a page-actions section, renders its setting through the host's standard
schema controls, and keeps an MCP status tool available when the UI projection
is disabled. Tests read these files directly so the contract cannot silently
drift.

## Files

| File                                | Purpose                                                                                                                                       |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `Cargo.toml`, `src/main.rs`         | Standalone plugin crate and runtime entrypoint.                                                                                               |
| `manifest.rs`                       | Runtime manifest, config schema, MCP tool, and web UI declaration.                                                                            |
| `plugin.toml`                       | Native package marker required by the installer.                                                                                              |
| `plugin.package.json`               | Checked-in expected output for `plugin-manifest.json`.                                                                                        |
| `config.toml`                       | Operator config sample showing `web_ui_enabled` independent from plugin process `enabled` and a plugin setting persisted through host config. |
| `bundle/host-contract.d.ts`         | Self-contained public TypeScript author contract; it does not import console source.                                                          |
| `bundle/register-mesh-plugin-ui.js` | Directly shippable ES module loaded by the React host.                                                                                        |
| `bundle/register-mesh-plugin-ui.ts` | Typed author-source variant of the same bundle behavior.                                                                                      |
| `lifecycle-states.json`             | Canonical state examples for `none`, `ready`, `disabled`, `invalid`, and `plugin_not_running`.                                                |

## Build, Package, And Install Locally

Run these commands from the repository root. They use a temporary package and
plugin store below the exemplar's ignored `target/` directory, so they do not
change your normal mesh-llm installation.

```bash
EXEMPLAR=docs/plugins/exemplars/web-ui
PACKAGE_ROOT="$EXEMPLAR/target/package/web-ui-exemplar"
ARCHIVE="$EXEMPLAR/target/web-ui-exemplar-0.1.0-local.tar.gz"
STORE="$EXEMPLAR/target/plugin-store"

cargo build --release --manifest-path "$EXEMPLAR/Cargo.toml"
rm -rf "$EXEMPLAR/target/package" "$STORE"
mkdir -p "$PACKAGE_ROOT"
cp "$EXEMPLAR/target/release/web-ui-exemplar" "$PACKAGE_ROOT/web-ui-exemplar"
cp "$EXEMPLAR/plugin.toml" "$PACKAGE_ROOT/plugin.toml"
cp -R "$EXEMPLAR/bundle" "$PACKAGE_ROOT/bundle"
"$EXEMPLAR/target/release/web-ui-exemplar" --print-package-manifest \
  > "$PACKAGE_ROOT/plugin-manifest.json"
tar -C "$EXEMPLAR/target/package" -czf "$ARCHIVE" web-ui-exemplar

MESH_LLM_PLUGIN_DIR="$STORE" ./target/debug/mesh-llm plugins install \
  --archive "$ARCHIVE" --name web-ui-exemplar --version 0.1.0
MESH_LLM_PLUGIN_DIR="$STORE" ./target/debug/mesh-llm plugins info web-ui-exemplar
```

On Windows, copy `web-ui-exemplar.exe`, package a `.zip`, and pass that archive
to the same `plugins install --archive` command. Local install accepts only
`.tar.gz` and `.zip`; `--name` is required and `--version` defaults to `dev`.

## Run And Verify

Start a client-only node so validation does not require a model or GPU:

```bash
./target/debug/mesh-llm auth status
# If no local owner identity exists, create an unencrypted development identity:
./target/debug/mesh-llm auth init --no-passphrase
cp "$EXEMPLAR/config.toml" "$EXEMPLAR/target/runtime-config.toml"
MESH_LLM_PLUGIN_DIR="$STORE" MESH_LLM_BIN=target/debug/mesh-llm just mesh-client \
  "" 19337 13131 "$EXEMPLAR/target/runtime-config.toml"
```

Owner identity is required for the settings save exercised below. Use a
passphrase-protected or keychain-backed owner identity outside local validation.

In a second terminal, verify the runtime, package asset, setting, and non-UI
tool. All commands below must succeed:

```bash
curl --fail http://127.0.0.1:13131/api/plugins/web-ui-exemplar/web-ui
curl --fail http://127.0.0.1:13131/api/plugins/web-ui-exemplar/web-ui/assets/register-mesh-plugin-ui.js
curl --fail http://127.0.0.1:13131/api/plugins/web-ui-exemplar/web-ui/config
curl --fail -X POST http://127.0.0.1:13131/api/plugins/web-ui-exemplar/tools/status \
  -H 'Content-Type: application/json' -d '{}'
```

Open `http://127.0.0.1:13131/` in live data mode. Confirm `Exemplar Notes`
appears as a direct auxiliary navigation item, click it, and add a sample note.
The page must show the configured retention window and name
`exemplar.notes.v1`. Open Configuration -> Plugins, confirm the Plugin settings
banner is the first item, select the Web UI Exemplar category, and change
Retention days with the host-rendered numeric control. Save the configuration,
return through the `Exemplar Notes` navigation item, and confirm the retention
window and new sample-note message use the saved value.

Disable only the UI projection, then prove the asset is hidden while the MCP
tool is still callable:

```bash
curl --fail -X PATCH http://127.0.0.1:13131/api/plugins/web-ui-exemplar/web-ui/enabled \
  -H 'Content-Type: application/json' -d '{"enabled":false}'
curl --fail -X POST http://127.0.0.1:13131/api/plugins/web-ui-exemplar/tools/status \
  -H 'Content-Type: application/json' -d '{}'
test "$(curl -s -o /dev/null -w '%{http_code}' \
  http://127.0.0.1:13131/api/plugins/web-ui-exemplar/web-ui/assets/register-mesh-plugin-ui.js)" = 404
```

Re-enable the projection with the same PATCH body using `true`. Stop the node
with Ctrl-C when finished, then remove `docs/plugins/exemplars/web-ui/target/`
to clean every build, archive, store, and installed-plugin artifact.

## Contract Summary

The manifest declares one local bundle id/root, one page, and one Integrations
config section. The ready page becomes a direct auxiliary navigation item while
it is the only eligible plugin page. Bundle paths are package-relative only. Page `route` values are
slugs, not paths or URLs. Do not use remote URL schemes, absolute paths,
traversal segments, unknown `bundle_id` references, or multiple bundle roots.

The bundle exports `registerMeshPluginUi(host)` and returns handlers for:

- page id `overview`
- config section id `page-actions`

Both handlers return an object with `unmount()`. Unmount removes DOM content and
unsubscribes from host state updates.

The page and config section read
`host.config.visible.settings.retention_days`. The page turns that setting into
a visible retention meter and uses it in the interactive sample-note status.
The config section provides a styled shortcut to the page and points authors to
the host-rendered schema control below it. The bundle does not duplicate the
numeric setting, write config files directly, or bypass owner-control save and
validation.

The non-UI capability `exemplar.notes.v1` and MCP tool `status` remain present
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

1. Open the Configuration `Plugins` tab and confirm the Plugin settings banner
   precedes installed-plugin metadata.
2. Select the Web UI Exemplar schema category and confirm Retention days uses
   the same host numeric control as other bounded integer settings.
3. Change the value and click `Save config`.
4. Confirm the owner-control save persists the schema-backed
   `plugin.web-ui-exemplar.settings.retention_days` value.
5. Confirm the saved config changes `[[plugin]].settings.retention_days` while
   leaving `enabled` and `web_ui_enabled` unchanged.
6. Click the `Exemplar Notes` navigation item and confirm both the retention
   meter and a newly added sample note use the saved number of days.
7. Refresh plugin metadata and confirm non-UI capability `exemplar.notes.v1`
   remains represented.
