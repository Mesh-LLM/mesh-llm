# GitHub CI agent entry point

Before inspecting, running, defining, editing, reviewing, or documenting any
GitHub Actions workflow, local action, runner, dependency, cache, artifact,
variable, secret, permission, release, or deployment, read
`.agents/skills/manage-ci/SKILL.md` completely and follow it.

The `manage-ci` skill is the canonical CI rule source. Its
`references/current-inventory.md` records the checked-in workflow, runner,
variable, secret-name, and environment contract. `ci/ci.md` explains the
current topology. If a rule changes, update the skill first; if implementation
or topology changes, update the inventory and `ci/ci.md` in the same change.

Do not add duplicate CI rules here. This file exists to make the skill the
mandatory starting point for every CI-related edit under `.github/`.
