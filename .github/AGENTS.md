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

The five `pr_{quality,website,linux,macos,windows}.yml` and five
`main_{quality,website,linux,macos,windows}.yml` entry workflows, the
manual-only `ci-control.yml` dispatcher, and separate Quality, Website, Linux,
macOS and Windows lane workflows are authoritative for assembly. Each PR/main
entry calls one nested reusable lane so its jobs and logs remain visible in a
focused native run; only an explicit manual-full run uses detached dispatch.
Platform lanes must call platform-pure host/runtime/product/smoke/SDK reusables
without empty platform placeholders. Eligible same-repository PR jobs may use
Depot only while the checked-in bounded exception and repository gate are
active; forks remain GitHub-hosted. The approved uncredentialed CUDA smoke may
use the ephemeral `gpu-nvidia` scale set through the protected default-branch
workflow. Neither exception grants secrets or broader runner-group access.
Permanent Depot PR execution still requires the cache and runner-group
isolation gates in `ci/DEPOT_MIGRATION.md`. Do not change Depot settings or
runner groups as part of an ordinary CI refactor.

Preserve the five-entry PR shape exactly. Do not create an all-platform PR
workflow, an all-lanes reusable composer, or a PR controller whose visible job
only dispatches detached runs. Quality, Website, Linux, macOS, and Windows must
remain separate PR-associated workflows with directly drillable nested jobs
and one stable `PR / <lane>` result each. Do not add path filters; planning owns
skips so every stable result exists.

The protected `workflow_run` sibling-failure monitor is control infrastructure,
not a sixth PR validation entrypoint. It may cancel only queued or in-progress
Quality, Website, Linux, macOS, and Windows runs for the same PR number, exact
head SHA, and event epoch after preserving the workflow containing the first
definitive job failure. Its Actions-write token must never enter a PR-controlled
workflow, checkout, action, or process, and it must never target main or other
workflow classes.

Preserve the five-entry main shape as well. Routine pushes to `main` must use
five focused native workflows with one stable `Main / <lane>` result each.
They must not use path filters, cancel older main revisions, call detached
dispatch, or compose all lanes into one graph. `ci-control.yml` must remain
`workflow_dispatch`-only.

The manage-ci skill is normative. The inventory and `ci/ci.md` describe current
implementation; the optimization specification records design, status and
acceptance criteria. Update the appropriate source in the same change and
remove superseded text instead of adding an investigation log here.
