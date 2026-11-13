# authority-automation

`authority-automation` is the narrow executable the protected authority
producers publish: a library (`authority_automation`) and a binary that run the
commands their consumers need without compiling full xtask. Its normal
dependencies are [ci-controller](../ci-controller/README.md), serde_json, sha2,
hex, url (the endpoint parser `pr-authority-audit` uses), and libc on Unix, so
the binary builds quickly on a bare GitHub-hosted runner.

It owns:

- `models resolve` and `models restore-inputs`: select one cadence-authorized,
  integrity-pinned test-model artifact, with the manifest, field, JSON byte, and
  serving-entry modules they share.
- `ci-ops authority-audit` and `ci-ops pr-authority-audit`: the runner cache and
  registry authority checks.
- `artifact file-projection`: the SHA-256 and canonical-path helper
  `skippy/scripts/prepare-llama.sh` falls back to.
- `automation bootstrap`, through `ci_controller::bootstrap` with this binary's
  own descriptor, which admits only `authority-automation` and `ci-controller`.

It does not own `models generate`, `parity-download`, `projector-download`,
`dflash-draft-convert`, or any other automation. Those stay in
[xtask](../xtask/README.md), which imports the shared model modules from here.

## Consumers

- The protected authority producers (`protected-automation-artifact.yml`,
  Quality `runner_policy`, and the Depot canary) build this binary through
  [`upload-automation`](../../.github/actions/upload-automation/action.yml) with
  `source-profile: protected-clean` and stage it under the artifact member name
  `xtask`, so `restore-automation` and the inline verifiers are unchanged.
- The jobs that restore those bytes: `skippy_cpp_api` model restoration, the
  authority sentinels, and `audit-pr-authority-verified`.
- xtask, which links the library and delegates the same commands to it, so
  `cargo xtool` gives identical output, exit codes, and refusals.

## Usage

From the workspace root:

```sh
just authority-automation-bootstrap
cargo run -p authority-automation -- models resolve --cadence pull-request MANIFEST
cargo run -p authority-automation -- ci-ops authority-audit endpoint ACTIONS_CACHE_URL
```

Every command accepts an optional leading `--repo-root DIR`. An unknown command
prints usage and exits 2. `just ci-automation-contracts` runs this crate's tests;
the main Rust test matrix skips it.
