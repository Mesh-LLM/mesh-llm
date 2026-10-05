# xtask

`xtask` is the workspace's internal Rust command runner for repository and
release checks. It is not published as a library. Root `just` recipes and CI
invoke it for crate-list, publish-chain, release-target, and console-output
consistency checks, and for release attestation operations.

From the workspace root, run the repository checks through `just`:

```bash
just ci-crate-lists
just publish-crates
just check-release
just no-console-print
```

To list the direct command syntax, run `just with-lld cargo run -p xtask --`.
The command prints its usage when no subcommand is supplied.
