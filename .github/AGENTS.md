# GitHub CI agent entry point

Before inspecting, running, defining, editing, reviewing, or documenting any
workflow, local action, runner, cache, artifact, variable, secret, permission,
release, deployment, or CI script:

1. Read `.agents/skills/manage-ci/SKILL.md` completely.
2. Read its `references/current-inventory.md` completely.
3. Read `ci/ci.md` and every workflow/action/script reached by the change.
4. For PR/main composition, routing, fan-out, or provider changes, follow
   `.omo/specs/pr-ci-optimization.md`.

Strict extension pattern:

- keep event entrypoints thin;
- implement new PR/main behavior once as a typed reusable slice;
- route it from the central checked plan using direct ownership or affected
  Rust dependencies as appropriate;
- make a selected PR slice identical to its main slice;
- derive runner and cache authority centrally—never accept raw labels;
- preserve immutable producer/consumer artifacts and a stable unique summary;
- validate GitHub fallback before any provider rollout.

The protected `ci-control.yml` planner and the separate Quality, Website,
Linux, macOS and Windows lane workflows are authoritative for PR and main
assembly. Platform lanes must call platform-pure host/runtime/product/smoke/SDK
reusables; do not reintroduce a cross-platform bootstrap graph or empty
platform placeholders. Current PR code is GitHub-hosted. Depot PR execution is
prohibited until the cache and runner-group isolation gates in
`ci/DEPOT_MIGRATION.md` pass. Do not change Depot settings or runner groups as
part of an ordinary CI
refactor.

The manage-ci skill is normative. The inventory and `ci/ci.md` describe current
implementation; the optimization specification records design, status and
acceptance criteria. Update the appropriate source in the same change and
remove superseded text instead of adding an investigation log here.
