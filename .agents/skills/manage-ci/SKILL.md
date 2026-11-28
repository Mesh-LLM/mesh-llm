---
name: manage-ci
description: Use this skill as the starting point whenever inspecting, running, debugging, editing, reviewing, or documenting MeshLLM CI/CD. It covers the GitHub Actions workflows, the local setup action, the scripts they call, releases, and deployments.
---

# Manage CI

MeshLLM's CI is small and plain. Skippy is built, tested and released in its
own repository, so nothing here builds an inference backend except the one CPU
runtime the smoke test needs.

Read these before changing CI:

1. `.github/AGENTS.md` for the rules in brief.
2. `references/current-inventory.md` for the checked-in workflows, runners,
   secrets and variables.
3. `RELEASE.md` when the change touches `release.yml` or a packaging script.

When this skill and the workflows disagree, the workflows are right. Fix this
skill and the inventory in the same change.

## What runs

- `ci.yml` runs on every pull request and on pushes to `main`: `just lint`,
  `just test-rust --locked --no-fail-fast` on Linux, macOS and Windows, `just test-ui`, and
  one Linux CPU inference and mesh routing smoke plus short macOS and Windows
  CPU host smokes. The latter install Skippy's published CPU package and make
  one request; they do not qualify the runtime backend.
- `release.yml` is started by hand (`just release vX.Y.Z`). It builds, signs,
  checks and publishes one host archive per platform and the Swift, Kotlin and
  Node SDK artifacts. It runs short CPU host smokes on Linux x86_64, macOS and
  Windows; Skippy qualifies its runtimes, so MeshLLM does not run a backend
  qualification matrix.
- The remaining workflows deploy the website and the Fly console, check the
  client Dockerfile, assign pull requests and close stale ones.

## Rules

- Each job calls a `just` recipe or a script under `scripts/`. Put logic there,
  where it runs locally, not in workflow YAML.
- A new script that CI depends on, or one you change, gets a test under
  `scripts/tests/` that `just lint` runs. Most existing scripts have none:
  only `check-host-imports.sh`, `ci-resolve-skippy.sh`, `ci-skippy-runtime.sh`,
  `wait-packaging-receipt.sh` and the installer asset selection are tested.
- Do not add Skippy builds, GPU runtimes, llama.cpp patches or Skippy tests.
- Pin every action to a full commit SHA, with the version in a comment.
- Give a workflow `contents: read` and raise a permission only on the job that
  needs it. Check out with `persist-credentials: false`. Two existing checkouts
  do not: `website-pages.yml` and `fly-deploy-console.yml`. Do not copy them.
- All jobs run on GitHub-hosted runners. Adding a self-hosted runner, a new
  secret, a new variable or a new deployment environment is a maintainer
  decision; ask first and record it in the inventory.
- Do not weaken or remove a required check to make a run pass. A check that is
  missing stays listed in `release.yml`'s `release_readiness` job and in
  `RELEASE.md` until it is replaced or a maintainer accepts its loss.
- There is no `tools/xtask`, `cargo xtool`, `ci/` directory, lane planner or
  rolling preview. Do not call them or restore them.

## Testing MeshLLM against a Skippy change

Lint, all three OS Rust test jobs, the website CLI generator and Linux smoke
use the same Skippy revision.
By default it is the crates.io release in `Cargo.toml`/`Cargo.lock`, with
the matching runtime source commit in `scripts/skippy-pin.sh`. For test-only
cross-repository work, select another one:

- In a pull request, add a line `Skippy-Ref: pull/123` (or a branch, tag or
  full commit SHA) to the description. Editing the description starts a run.
- From the Actions page, run `ci.yml` with the `skippy_ref` input.

The selection job parses the directive with `scripts/ci-skippy-ref.sh` and
resolves it once with `scripts/ci-resolve-skippy.sh`. For an override,
`ci-pin-skippy.sh` temporarily replaces the manifest's Skippy registry dependencies
with exact Git revisions (dropping the old release version constraints). Cargo
resolves one lockfile. The exact manifest/lockfile artifact is shared by lint,
tests, website and smoke, which still use `--locked`. Do not commit these
temporary CI files. Only smoke
builds the native CPU runtime, from that same SHA. The macOS/Windows CPU
install smokes are skipped for overrides because they need published packages;
they must pass in the fresh pinned run before merging. Ordinary runs download the
pinned release runtime and use the committed manifest and lockfile unchanged.

A branch or PR is resolved once per run; start a fresh full run after a new
Skippy push. A failed-jobs rerun can reuse the previous selection and artifact;
it does not select a new branch head. If its short-lived artifact has expired,
start a full run. Running on `main` with a Skippy ref means you trust that code.

**An override run is never merge-ready.** The existing required `PR / Quality`,
`PR / Linux`, `PR / macOS`, `PR / Windows` and `PR / Website` checks fail while
an override was selected, without stopping the test jobs. They also read the
current PR description before passing, rejecting a directive added while an
older pinned run was finishing. Blank or duplicate directives fail closed.

Before merging, land the Skippy change, update Mesh's committed pin/lockfile
to the intended supported revision/release as needed, remove the directive,
and wait for a fresh full pinned run to pass. Removing the directive cannot
turn an old override run green. Do not merge using stale checks while an edited
description's run is queued, and do not bypass the required checks. These are
CI merge checks, not an atomic server-side rule on PR-description contents.

## Validation

Before pushing a CI change:

```bash
actionlint
just lint        # fmt, clippy, ShellCheck and the script tests
```

Also run the `just` recipe or script the changed job calls, when it can run on
your machine. `actionlint` and the script tests do not run a workflow. Say so
when a job has never run on GitHub; most of `release.yml` has not.

Never run an installer, updater or packaging test against a real home
directory or install path. Use a throwaway `HOME` and stub the network
commands.

## Releases and deployments

- Never dispatch `release.yml`, push a tag, publish a release or deploy without
  a maintainer asking for exactly that. A canary (`canary: true`) builds and
  checks everything and publishes nothing, but still needs to be asked for.
- The handoff to `Mesh-LLM/mesh-packaging` is off. `release.yml`'s last job
  fails on purpose if `MESH_PACKAGING_DISPATCH` is turned on. The proposed
  contract is `mesh/docs/design/PACKAGING_HANDOFF.md`.
- Release signing keys are secrets. Never print them, and never add a step
  that uploads them.

## History

The rules and inventory for the earlier CI, which built Skippy in this
repository, are kept unchanged in
`references/retired-in-tree-skippy-ci-rules.md` and
`references/retired-in-tree-skippy-ci-inventory.md`. They are a record only.
`references/maintainer-decisions.md` holds past maintainer decisions.
