# ci-controller

`ci-controller` is the narrow CI controller: a library (`ci_controller`) and a
binary that plan CI and validate lane results without compiling full xtask. Its
normal dependencies are limited to serde, serde_json, sha2, and hex, so the
binary builds quickly on a bare GitHub-hosted runner.

It owns:

- `ci plan`: provider-neutral plan construction from `ci/ownership.yml` and
  `ci/slices.yml`.
- `ci validate-lane`: lane result admission for the lane `summary` jobs.
- `repository affected-crates`, `repository conventional-commits`, and
  `repository llama-upstream-pin`: the planner-side repository checks.
- `automation bootstrap`: the check that the executable is Cargo's build output
  and that only admitted workspace packages were compiled into it. Each binary
  passes its own `bootstrap::BootstrapTool` descriptor; this binary admits only
  the `ci-controller` package.

It does not own `ci family-plan`, `ci validate-graph`, or any product, native,
canary, or release automation. Those stay in [xtask](../xtask/README.md).

## Consumers

- CI jobs that run
  [`prepare-ci-controller`](../../.github/actions/prepare-ci-controller/action.yml):
  the entry planners through `compute-changes` and `plan-ci`, the five lane
  summaries, and Quality commit convention. They call the binary through
  `MESH_CI_CONTROLLER_BIN`.
- xtask, which links the library and delegates the same commands to it, so
  `cargo xtool` gives identical output, exit codes, and refusals.

## Usage

From the workspace root:

```sh
just ci-controller-bootstrap
cargo run -p ci-controller -- ci plan < plan-input.json
cargo run -p ci-controller -- ci validate-lane --lane-plan JSON --needs JSON
git diff --name-only main | cargo run -p ci-controller -- repository affected-crates --stdin
```

Every command accepts an optional leading `--repo-root DIR`. An unknown command
prints usage and exits 2. `just ci-automation-contracts` runs this crate's tests;
the main Rust test matrix skips it.
