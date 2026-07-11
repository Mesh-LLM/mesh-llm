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

## Handoff

When you finish, leave a short note with:

- what changed
- which source-owned docs or exemplar files moved
- which validation checks passed
- whether any state or recovery path still needs follow-up
