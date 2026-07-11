# Plugin Web UI Extension 5/5 Completion Plan

This document is a follow-up plan for raising every evaluated plugin web UI
extension area to a perfect 5/5 completeness score against
`.omo/specs/plugin_extension_2.md`, `.omo/plans/plugin-web-ui-extension-plan.md`,
and the implementation committed as `e5bf3afbd`.

It is documentation only. Do not begin implementation from this file without a
separate explicit start-work request.

## Baseline

- Branch: `codex/plugin-web-ui-extension`
- Baseline commit: `e5bf3afbd Add plugin web UI extension`
- Current final evidence: F1-F4 approved under
  `.omo/evidence/plugin-web-ui-extension-plan/final/`
- Current high-level status:
  - Complete: scope fidelity, package validation, independent persistence,
    runtime state, UI data hooks, final verification evidence.
  - Nearly complete: manifest strictness, runtime lifecycle, plugin API route
    matching, console route/mount details, docs.
  - Needs hardening: Integrations config mutation, plugin author host surface,
    exemplar persistence proof.

## Completion Rubric

A major area earns 5/5 only when all of the following are true:

1. The behavior is explicitly required by the spec or plan, or is needed to
   remove an identified gap.
2. The implementation behavior is present in source, not only in docs or sample
   code.
3. The behavior has targeted automated coverage.
4. The behavior appears in the exemplar or docs when it affects plugin authors
   or operators.
5. The final verification evidence includes the relevant acceptance proof.

## Global Constraints

- Keep `.omo/specs/*` unchanged.
- Keep all plugin web UI routes under `/api/plugins/*`.
- Do not add `/api/config/*` plugin UI routes.
- Do not add remote asset loading.
- Do not add iframe isolation or sandboxing for v1.
- Do not add RBAC, marketplace/discovery flow, or a generic event bus.
- Do not add a new primary plugin tab or dynamic TanStack route mutation.
- Do not let web UI enablement stop, restart, or disable non-UI plugin runtime
  capabilities.

## Score Closure Matrix

| Area | Current Score | Target | Required Closure |
|---|---:|---:|---|
| Scope fidelity / non-goals | 5/5 | 5/5 | Preserve with explicit regression scans and route/tab tests. |
| Manifest web UI contract | 4.5/5 | 5/5 | Tighten v1 declaration validation and document proto forward-compatibility. |
| Package/install asset validation | 5/5 | 5/5 | Preserve; add references from new manifest strictness tests. |
| Independent persistence | 5/5 | 5/5 | Preserve; reuse for plugin UI config mutation. |
| Runtime state model | 5/5 | 5/5 | Preserve; ensure config mutation refreshes summaries where relevant. |
| Runtime lifecycle separation | 4.5/5 | 5/5 | Add regression proof that UI config mutations and toggles do not restart plugins. |
| Plugin API + static assets | 4.5/5 | 5/5 | Replace broad asset-route matching with exact route parsing and add stapled-route regression tests. |
| Console data/hooks | 5/5 | 5/5 | Preserve; extend hooks only where config mutation needs invalidation. |
| Console route/nav/page mount | 4.5/5 | 5/5 | Use backend `asset_base_url`, wire real host callbacks, and expand cleanup tests. |
| Integrations projection | 4/5 | 5/5 | Wire real config mutation and toast/error handling into config-section mounts. |
| Plugin author host surface | 3.5/5 | 5/5 | Define and implement real config visibility, mutation, notification, and error contracts. |
| Exemplar plugin artifact | 4/5 | 5/5 | Prove persisted setting mutation end-to-end from the exemplar UI. |
| Docs + maintainer skill | 4.5/5 | 5/5 | Document strict manifest rules, exact routes, and the real host surface contract. |
| Verification evidence | 5/5 | 5/5 | Preserve and add evidence for the new closure items. |

## Workstream A: Preserve Existing 5/5 Areas

### A1. Scope Guardrails

Actions:

- Add a final scope check that scans product source for forbidden items:
  `/api/config/*`, remote plugin UI assets, iframe/sandbox requirements, RBAC,
  marketplace flow, generic event bus, dynamic TanStack route mutation, and new
  primary plugin tabs.
- Keep docs and maintainer skill allowed to mention forbidden items only as
  non-goals or warnings.
- Add or preserve tests showing `AppTab` remains limited to the existing primary
  tabs and plugin routes map to no primary tab.

Acceptance:

- Forbidden-scope scan returns no product-source matches.
- Plugin route remains static and auxiliary.
- Existing non-web-ui plugins still produce `state = none` and no projected
  nav/config/page surfaces.

Verification:

- `git diff -- .omo/specs` has no output.
- UI route/layout tests pass.
- Scope scan output is saved in final evidence.

### A2. Preserve Package, Persistence, Runtime State, and UI Data Coverage

Actions:

- Keep package asset validation tests for valid root, missing bundle, traversal,
  remote URL, and symlink escape.
- Keep `web_ui_enabled` persistence tests for absent, explicit true, explicit
  false, and independence from plugin process `enabled`.
- Keep runtime state tests for `none`, `ready`, `disabled`, `invalid`, and
  `plugin_not_running`.
- Keep UI data adaptation tests that hide nav for non-ready states while leaving
  Integrations visibility intact.

Acceptance:

- No current 5/5 area loses coverage while lower-scored areas are improved.

Verification:

- `cargo test -p mesh-llm-plugin-manager --lib`
- `cargo test -p mesh-llm-config --lib`
- `cargo test -p mesh-llm-host-runtime --lib`
- `cd crates/mesh-llm-ui && pnpm test`

## Workstream B: Manifest Contract to 5/5

Current gap:

- The proto uses repeated `bundles` with v1 validation rejecting multiple
  bundles. That is workable, but the v1 contract should be explicit and
  fully validated.
- Page `route` is treated as a relative path. The spec calls it a slug, so v1
  should reject slash-bearing path shapes unless the contract is intentionally
  broadened and documented.
- `bundle_id` references should be checked against the declared bundle.

Actions:

1. Define the v1 manifest invariant:
   - `web_ui` with pages or config sections must declare exactly one bundle.
   - The bundle id must be stable and non-empty.
   - Every page/config section `bundle_id` must match the declared bundle id.
   - Page `route` is a slug, not a path or URL.
   - Entry scripts and icons remain package-relative paths inside the single
     bundle root.
2. Decide and document why the proto field remains repeated:
   - It is a forward-compatible wire shape.
   - v1 validation still permits only one bundle root.
3. Add manifest validation tests for:
   - missing bundle with declared page/config section,
   - empty bundle id,
   - unknown `bundle_id`,
   - route containing `/`,
   - route containing URL/protocol syntax,
   - valid slug with a valid single bundle.
4. Update docs and the maintainer skill with the exact slug and bundle-id rules.

Acceptance:

- A declaring plugin cannot pass manifest/package validation with ambiguous
  bundle references or non-slug route values.
- Existing old manifests without `web_ui` still decode and serialize without a
  manifest version bump.
- The docs explain the repeated proto field as forward-compatible but v1-limited.

Verification:

- `cargo test -p mesh-llm-plugin --lib`
- `cargo test -p mesh-llm-plugin-manager --lib`
- Focused docs grep for route slug and bundle-id language.

## Workstream C: Plugin API Route Matching to 5/5

Current gap:

- The asset classifier checks whether the path contains `/web-ui/assets/`.
  That can accidentally intercept a stapled plugin HTTP path such as
  `/api/plugins/demo/http/web-ui/assets/file.js`.

Actions:

1. Replace broad route checks with exact plugin web UI route parsing:
   - `/api/plugins/:plugin/web-ui`
   - `/api/plugins/:plugin/web-ui/enabled`
   - `/api/plugins/:plugin/web-ui/assets/*asset`
2. Ensure route parsing distinguishes the plugin name segment from the remaining
   route suffix before deciding whether a path is web UI or stapled HTTP.
3. Add tests proving:
   - real web UI metadata/toggle/asset routes still match before stapled HTTP,
   - `/api/plugins/:plugin/http/web-ui/assets/*` is handled by stapled HTTP,
   - unknown assets remain `404`,
   - disabled/no declaration remain `404`,
   - invalid/plugin_not_running remain `409`,
   - traversal and encoded traversal remain rejected before filesystem reads.
4. Keep all responses under the existing plugin API namespace.

Acceptance:

- The web UI route family is exact.
- Web UI routes are not swallowed by stapled HTTP.
- Stapled HTTP routes containing the literal string `web-ui/assets` are not
  swallowed by web UI asset handling.

Verification:

- `cargo test -p mesh-llm-host-runtime --lib`
- Focused route classifier unit tests.

## Workstream D: Plugin Author Host Surface to 5/5

Current gap:

- The typed host surface includes `config.requestMutation` and
  `notifications.show`, but they default to no-op in current callers.
- `config.visible` currently exposes web UI state rather than a real read-only
  plugin config/settings view.

Actions:

1. Define a narrow, versioned host-surface contract:
   - `plugin.name`
   - `webUi` state
   - `config.visible.settings`
   - `config.visible.schema` or equivalent setting metadata for the plugin
   - `config.requestMutation(request)`
   - `network.fetchPlugin` and `network.json`
   - `appearance` theme/tokens
   - `navigation.navigateTo` and `navigation.openPluginPage`
   - `notifications.show`
   - minimal `state` snapshot/update/subscribe helpers.
2. Define the config mutation request shape:
   - plugin name must match the mounted plugin,
   - settings patch only mutates plugin-owned settings by key,
   - optional unset/remove list if needed,
   - no direct mutation of `enabled` or `web_ui_enabled` unless a separate
     host-owned control explicitly allows it,
   - validation errors return structured messages to the plugin UI.
3. Implement real callbacks in the console host surface:
   - use the existing configuration edit/save pipeline or a plugin-scoped
     management API route under `/api/plugins/:plugin/web-ui/...`,
   - keep the route under `/api/plugins/*`,
   - invalidate plugin summary/config queries after successful mutation,
   - surface validation failures to the mounted plugin and visible UI.
4. Implement real notification behavior:
   - if the console already has a toast system, adapt to it,
   - otherwise add a minimal local notification adapter for plugin UI calls,
   - do not create a generic plugin event bus.
5. Add host-surface tests:
   - `requestMutation` sends the expected validated request,
   - plugin mismatch is rejected,
   - invalid setting key/type is rejected,
   - successful mutation refreshes visible config and summaries,
   - notification callback renders or records a visible toast.

Acceptance:

- Plugin UI authors receive a real, documented, tested host surface for config
  visibility, config mutation, navigation, notifications, network helpers,
  appearance context, and minimal state subscription.
- Config mutation is host-owned and scoped to the mounted plugin.
- No plugin bundle writes config files directly.

Verification:

- `cd crates/mesh-llm-ui && pnpm run typecheck`
- `cd crates/mesh-llm-ui && pnpm test`
- Backend tests if a plugin-scoped config mutation route is added.

## Workstream E: Integrations Projection to 5/5

Current gap:

- Integrations mounts config sections and shows status, but mounted config
  sections do not receive a real persistence callback.

Actions:

1. Pass the real host-surface config mutation callback into
   `PluginConfigSectionMount`.
2. Pass read-only plugin config values into `config.visible` so config sections
   can render current values.
3. Show mutation pending/success/error state in the Integrations config-section
   host shell without replacing plugin-owned UI.
4. Keep existing schema-driven plugin settings visible below or beside plugin UI
   config sections; do not replace them.
5. Add tests proving:
   - a config section can request a setting mutation,
   - the mutation persists through the host config pipeline,
   - `enabled` and `web_ui_enabled` are not changed by a settings-only mutation,
   - disabling web UI unmounts config sections exactly once,
   - invalid/plugin_not_running states do not import config bundle code,
   - existing schema settings still render.

Acceptance:

- Integrations satisfies identity, description, process state, web UI state,
  toggle, unavailable reasons, config-section projection, config mutation, and
  existing settings preservation.

Verification:

- `cd crates/mesh-llm-ui && pnpm run typecheck`
- `cd crates/mesh-llm-ui && pnpm test`
- Manual QA updates a config value from the exemplar config section.

## Workstream F: Console Route and Mount Surface to 5/5

Current gap:

- Page and config-section mounts construct asset URLs from helper functions
  instead of preferring the backend-provided `asset_base_url`.
- Page mounts also receive no real config mutation or notification callback.

Actions:

1. Update asset URL resolution:
   - use `webUi.asset_base_url` when available,
   - enforce same-origin after resolving the final URL,
   - reject missing `asset_base_url` in ready state with a host fallback.
2. Share one asset URL resolver between page and config-section mounts.
3. Share one host-surface factory between page and config-section mounts, with
   injected real config mutation, notification, navigation, and query refresh
   callbacks.
4. Add cleanup tests:
   - unmount once on route change,
   - unmount once on disable/refetch state change,
   - no import occurs for disabled/invalid/plugin_not_running/none states,
   - no import occurs when ready state lacks usable `asset_base_url`,
   - errors are visible but do not break the console shell.

Acceptance:

- Plugin page and config-section mounts use one consistent, backend-authorized
  asset loading path and one real host surface.

Verification:

- `cd crates/mesh-llm-ui && pnpm run typecheck`
- `cd crates/mesh-llm-ui && pnpm test`
- Manual QA screenshots for ready, disabled, invalid, and plugin_not_running.

## Workstream G: Exemplar Artifact to 5/5

Current gap:

- The exemplar calls `host.config.requestMutation(...)`, but current host
  callers do not persist that request.
- The exemplar does not yet prove a setting round-trip from mounted UI through
  host config and back to visible state.

Actions:

1. Update the exemplar bundle to read the current setting from
   `host.config.visible.settings.retention_days`.
2. Keep the exemplar mutation scoped to the plugin setting key
   `retention_days`.
3. Add an automated test or manual QA harness step that:
   - opens the exemplar Integrations config section,
   - changes `retention_days`,
   - triggers `host.config.requestMutation`,
   - verifies the host config/TOML/API state updates,
   - verifies plugin process enabled state remains unchanged,
   - verifies `web_ui_enabled` remains unchanged unless the web UI toggle is
     used.
4. Keep lifecycle samples for `none`, `ready`, `disabled`, `invalid`, and
   `plugin_not_running`.
5. Add an invalid exemplar package variant or fixture if needed to keep invalid
   remediation coverage source-owned.
6. Make the exemplar contract typecheck against the exported host contract
   without relying on brittle relative imports where feasible.

Acceptance:

- The exemplar is a reusable validation fixture for manifest, bundle, route,
  Integrations, config persistence, lifecycle states, invalid remediation, and
  non-UI capability continuity.

Verification:

- Exemplar contract tests pass.
- Manual QA records persisted setting mutation.
- Docs include the reproduction recipe.

## Workstream H: Docs and Maintainer Skill to 5/5

Current gap:

- Docs and skill mostly align, but they need to describe the stricter final
  manifest rules and real host-surface config mutation behavior.

Actions:

1. Update `docs/plugins/README.md` with:
   - exact v1 route slug rules,
   - bundle id/reference rules,
   - repeated proto field vs single v1 bundle explanation,
   - exact API routes and route matching behavior,
   - config visibility and mutation request/response contract,
   - notification behavior,
   - asset-base URL use,
   - error and remediation behavior.
2. Update `docs/plugins/exemplars/web-ui/README.md` with the setting mutation
   reproduction recipe.
3. Update `.agents/skills/plugin-web-ui-extension/SKILL.md` with:
   - route classifier audit steps,
   - config mutation audit steps,
   - host-surface checklist,
   - exemplar drift checklist,
   - exact final verification commands.
4. If public website docs are expected to mirror plugin docs, update the website
   source and run the website build.

Acceptance:

- Plugin authors can implement a v1 UI without reading source.
- Operators can diagnose each state and reproduce exemplar behavior.
- Maintainers have concrete checks for every prior sub-5 gap.

Verification:

- Docs grep confirms required terms are present.
- Unsupported-feature grep confirms non-goals are not described as supported.
- `just website-build` if website docs are touched.

## Workstream I: Final Verification and Rescore

Actions:

1. Run final backend gates serially:
   - `cargo fmt --all --check`
   - `cargo test -p mesh-llm-plugin --lib`
   - `cargo test -p mesh-llm-plugin-manager --lib`
   - `cargo test -p mesh-llm-config --lib`
   - `cargo test -p mesh-llm-host-runtime --lib`
   - `cargo clippy -p mesh-llm-plugin --all-targets -- -D warnings`
   - `cargo clippy -p mesh-llm-config --all-targets -- -D warnings`
   - `cargo clippy -p mesh-llm-host-runtime --all-targets -- -D warnings`
   - `cargo clippy -p mesh-llm --all-targets -- -D warnings`
2. Run final UI gates:
   - `cd crates/mesh-llm-ui && pnpm run typecheck`
   - `cd crates/mesh-llm-ui && pnpm test`
   - `cd crates/mesh-llm-ui && pnpm run build`
   - `just build`
3. Run docs/site gates if website docs are touched.
4. Run manual QA against the exemplar:
   - ready nav and route,
   - Integrations status/toggle/config section,
   - persisted setting mutation,
   - disabled hides projection while process/non-UI capability remains,
   - invalid bundle shows reason and preserves non-UI capability,
   - process-disabled shows `plugin_not_running`.
5. Save evidence for each new closure item.
6. Re-run the evaluation table and require every area to score 5/5.

Acceptance:

- Every row in the score closure matrix is 5/5.
- No final evidence file has a reject verdict.
- `.omo/specs/*` remains unchanged.

## Recommended Implementation Order

1. Manifest strictness and docs wording.
2. Exact plugin API route parsing.
3. Real host-surface config visibility/mutation/notifications.
4. Integrations and page mount callback wiring.
5. Exemplar setting round-trip and manual QA harness update.
6. Docs/skill updates.
7. Full verification and final rescore.

This order closes contract ambiguity first, then fixes the functional author
surface, then proves it through the exemplar and final evidence.
