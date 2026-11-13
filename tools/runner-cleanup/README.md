# runner-cleanup

`runner-cleanup` is the narrow executable for `ci-ops runner-cleanup`: a
library (`runner_cleanup`) and a binary that delete a persistent CI job's own
outputs without compiling full xtask. Its normal dependencies are
[ci-controller](../ci-controller/README.md),
[process-supervision](../process-supervision/README.md), and thiserror.

It owns:

- `ci-ops runner-cleanup`: the job profiles, the deletion boundary and
  admission checks, bounded Git worktree removal for replay, and interruption
  finalization.
- `automation bootstrap`, through `ci_controller::bootstrap` with this binary's
  own descriptor, which admits only `runner-cleanup`, `ci-controller`, and
  `process-supervision`.

It does not own any other `ci-ops` command or automation; those stay in
[xtask](../xtask/README.md).

## Consumers

- The trusted runner-image job in `ci-runner-contract-slice.yml`, which builds
  this binary through `just runner-cleanup-bootstrap`, freezes a private copy
  under the unchanged `xtask` name, and runs
  `ci-ops runner-cleanup --job runner-contract` from it.
- xtask, which links the library and delegates `ci-ops runner-cleanup` to it,
  so `cargo xtool` and the jobs that keep full xtask (Laya, smoke, release,
  replay, canaries) get identical output, exit codes, and refusals.

## Usage

From the workspace root:

```sh
just runner-cleanup-bootstrap
cargo run -p runner-cleanup -- ci-ops runner-cleanup --help
```

Every command accepts an optional leading `--repo-root DIR`. An unknown command
prints usage and exits 2. `just ci-automation-contracts` runs this crate's tests
with `--test-threads=1`, because the interruption scope is process-global; the
main Rust test matrix skips it.
