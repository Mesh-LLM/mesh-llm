# xtask

`xtask` is the workspace's internal Rust command runner for repository and
release checks. It is not published as a library. Root `just` recipes and CI
invoke it for crate-list, publish-chain, release-target, and console-output
consistency checks, and for release attestation operations.

CI planning, lane result admission, the planner-side repository checks, and
`automation bootstrap` live in [ci-controller](../ci-controller/README.md).
xtask keeps those commands and delegates them to the controller library, so
their output and exit codes are unchanged.

`models resolve`, `models restore-inputs`, `ci-ops authority-audit`,
`ci-ops pr-authority-audit`, and `artifact file-projection` live in
[authority-automation](../authority-automation/README.md), the executable the
protected authority producers publish. xtask delegates them the same way.

`ci-ops runner-cleanup` lives in [runner-cleanup](../runner-cleanup/README.md), the
executable the trusted runner-image job freezes for cleanup. xtask delegates it
the same way.

From the workspace root, run the repository checks through `just`:

```bash
just ci-crate-lists
just publish-crates
just check-release
just no-console-print
```

The canonical direct interface is `cargo xtool <domain> <command> [options]`.
For example, run `just with-lld cargo xtool repo-consistency ci-crate-lists`;
`just with-lld cargo xtool --help` lists the command syntax. The transition
interface `cargo run -p xtask --` remains compatible. Keep build and required
validation entrypoints in their owning Just recipes. New repository automation
belongs in typed Rust owners, not new Python tooling or generic shell policy.
