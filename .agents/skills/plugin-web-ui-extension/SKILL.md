---
name: plugin-web-ui-extension
description: Use this skill when maintaining the plugin web UI projection contract, docs, exemplar coverage, or recovery flow for mesh-llm plugin web UI work.
metadata:
  short-description: Maintain plugin web UI projection
---

# plugin-web-ui-extension

Use this skill for any follow-up on the plugin web UI projection contract,
source-owned docs, or maintainer triage.

## Baseline Checks

Start by reading the source-owned contract docs and the maintained exemplar:

- `docs/plugins/README.md`
- `docs/plugins/exemplars/web-ui/README.md`
- `docs/plugins/exemplars/web-ui/manifest.rs`
- `docs/plugins/exemplars/web-ui/plugin.package.json`
- `docs/plugins/exemplars/web-ui/config.toml`
- `docs/plugins/exemplars/web-ui/bundle/register-mesh-plugin-ui.ts`
- `docs/plugins/exemplars/web-ui/lifecycle-states.json`

Then confirm the current contract still matches the implementation:

- manifest `web_ui` remains additive
- `web_ui_enabled` stays separate from plugin process `enabled`
- the existing Configuration `Plugins` tab still owns Integrations projection
- the static route stays `/plugins/$pluginName/$pageId`
- the API stays under `/api/plugins/:plugin/web-ui`
- plugin config mutations stay under `/api/plugins/:plugin/web-ui/config`

## Create A Work Item

Before proposing implementation, turn the request into a contract hypothesis:

1. Name the author-visible behavior and the exact lifecycle states it changes.
2. Locate the current manifest, package, config, runtime, API, UI, exemplar, and
   CLI owners; record the observed baseline instead of assuming it.
3. State whether the change is additive for older plugins and hosts. Ask for
   explicit approval before any breaking protocol or package change.
4. List the happy path, invalid package path, disabled projection path, stopped
   process path, and non-UI continuity proof.
5. Identify the exact exemplar and documentation updates that make a new author
   able to reproduce the behavior without reading host source.

The implementation hypothesis is incomplete until it includes rollback,
package validation, config persistence, UI unmount behavior, CLI discoverability,
and a docs-only reproduction command.

## Ownership Boundaries

Keep the projection split clear:

- plugin process state controls process startup and shutdown
- web UI state controls only whether the UI projection mounts
- invalid or missing bundles do not change non-UI capabilities
- config-section mounting stays on the existing Configuration Plugins surface
- no new primary app tab is introduced for plugin routes

## Settings And Manifests

When editing manifests or persisted settings, keep the source of truth honest:

- update the manifest docs before describing new fields elsewhere
- keep bundle paths package-relative and rooted under one bundle directory
- require one non-empty v1 bundle id and ensure page/config `bundle_id`
  references match it
- keep page `route` values slug-only; reject path separators, protocols, and
  traversal-style dot prefixes
- preserve the exact route and DTO names already used by the backend and UI
- treat `parent_tab = "integrations"` as the only config-section tab value, or omit it
- keep authoring and persistence guidance tied to the host config schema, not direct file writes from the bundle

## Runtime Changes

If runtime behavior changes, check the whole lifecycle path:

- summary state should still surface `none`, `ready`, `disabled`, `invalid`, and `plugin_not_running`
- the toggle endpoint should continue to persist projection only
- asset serving should stay same-origin and validated
- route classification should parse `/api/plugins/:plugin/web-ui...` by exact
  suffix after the plugin name so stapled HTTP paths under
  `/api/plugins/:plugin/http/...` remain plugin HTTP
- ready mounts should use backend `asset_base_url` and refuse to import bundle
  code when it is missing or off-origin
- bundle imports should stay gated on ready-state projection eligibility
- mount and unmount logic should remain idempotent
- `host.config.visible.settings` should reflect current plugin-owned settings
- `host.config.requestMutation(...)` should persist only plugin settings and
  reject mismatched plugin names, host-owned keys, and invalid values
- `host.notifications.show(...)` should surface through the host shell without
  adding a generic event bus

## Exemplar Coverage

The exemplar under `docs/plugins/exemplars/web-ui/` is the drift guard.

- keep the README in sync with the implementation contract
- keep `lifecycle-states.json` aligned with the state matrix
- keep the sample manifest and bundle contract aligned with the typed host API
- keep the persisted `retention_days` setting proof wired through
  `host.config.visible` and `host.config.requestMutation`
- use the exemplar when adding tests, review notes, or recovery guidance
- keep it buildable as a standalone crate, generate `plugin-manifest.json` from
  its runtime manifest, install the local archive with
  `plugins install --archive`, and run its browser-importable JavaScript bundle
- keep `bundle/host-contract.d.ts` self-contained; public author examples must
  not import private console source paths
- initialize an owner identity before testing settings persistence; reserve
  `auth init --no-passphrase` for isolated development and document the normal
  protected/keychain alternative
- when the exemplar uses `just mesh-client` with the branch's debug binary,
  set `MESH_LLM_BIN=target/debug/mesh-llm` explicitly
- prove its MCP status tool still responds after `web_ui_enabled` is toggled off

## Triage And Recovery

When a report says web UI is broken, triage in this order:

1. Check whether the plugin process is running.
2. Check whether the projection is `disabled` or `invalid`.
3. Check whether the installed bundle root exists.
4. Check whether the page or config-section entry script is inside the bundle.
5. Check whether the config-section parent tab is `integrations`.
6. Check `/api/plugins/:plugin/web-ui/config` for current settings and schema
   when config sections render stale values or mutation errors.
7. Reinstall or update the plugin package and reload mesh-llm if metadata needs revalidation.

If the bundle is missing or invalid, fix the package contents first. Do not
disable the plugin process unless the non-UI behavior is also broken.

## API And Versioning Alignment

Keep compatibility additive:

- do not add breaking manifest or route changes without an explicit contract update
- keep backend and frontend DTO names synchronized
- keep the route namespace stable under `/api/plugins/:plugin/web-ui`
- do not claim sandboxing, remote assets, marketplace discovery, RBAC, or generic settings editing unless the contract has changed and the exemplar has been updated too

## Validation And Evidence

Run Cargo commands serially. At minimum validate the author and settings owners:

```bash
cargo test -p mesh-llm-plugin --lib
cargo test -p mesh-llm-plugin-manager --lib
cargo test -p mesh-llm-config --lib
cargo test -p mesh-llm-host-runtime --lib
```

Run the UI typecheck, unit tests, build, the website build when public docs
change, and finally `just test-all`. Confirm `just test-all` explicitly invokes
the plugin, plugin-manager, config, CLI, command, host, and UI test owners; a
dependency compiling is not evidence that its unit tests ran.

For release handoff, have one reviewer build a fresh plugin from the published
docs without reading `crates/**` or private UI source. They must install the
archive, start it, mount a React-hosted page and Integrations config section,
persist a setting, test a non-UI capability with projection disabled, and clean
up every process/store/package artifact.

## Handoff

When you finish, leave a short note with:

- what changed
- which source-owned docs or exemplar files moved
- which validation checks passed
- whether any state or recovery path still needs follow-up
