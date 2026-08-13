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

The five `pr_{quality,website,linux,macos,windows}.yml` entry workflows,
`ci-control.yml` main/manual planner, and separate protected Quality, Website,
Linux, macOS and Windows lane workflows are authoritative for assembly. Each
PR entry calls one nested reusable lane so its jobs and logs remain visible in
a focused native PR run; main/manual lanes are separately dispatched.
Platform lanes must call platform-pure host/runtime/product/smoke/SDK reusables
without empty platform placeholders. Current PR code is GitHub-hosted. Depot PR execution is
prohibited until the cache and runner-group isolation gates in
`ci/DEPOT_MIGRATION.md` pass. Do not change Depot settings or runner groups as
part of an ordinary CI refactor.

Preserve the five-entry PR shape exactly. Do not create an all-platform PR
workflow, an all-lanes reusable composer, or a PR controller whose visible job
only dispatches detached runs. Quality, Website, Linux, macOS, and Windows must
remain separate PR-associated workflows with directly drillable nested jobs
and one stable `PR / <lane>` result each. Do not add path filters; planning owns
skips so every stable result exists.

The manage-ci skill is normative. The inventory and `ci/ci.md` describe current
implementation; the optimization specification records design, status and
acceptance criteria. Update the appropriate source in the same change and
remove superseded text instead of adding an investigation log here.
