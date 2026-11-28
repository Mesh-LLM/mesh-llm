# Agent Notes

## Repository and instruction scope

This repository is MeshLLM. Its product tree is `mesh/`, and product guidance is in `mesh/AGENTS.md`. Skippy, the inference runtime, lives in its own repository and is built, tested and released there. MeshLLM depends on exact Skippy crates.io versions pinned in the root `Cargo.toml` and `Cargo.lock`; `scripts/skippy-pin.sh` names the matching source commit for runtime fallback builds. MeshLLM loads a packaged Skippy runtime at run time. Do not add Skippy source, llama.cpp patches, GPU runtime builds or Skippy tests here.

Root `Cargo.toml`, `Cargo.lock`, `.cargo/`, `just/`, `scripts/` and `.github/` are shared infrastructure. There is no `shared/` product tree. Some root scripts forward to `mesh/scripts/`. Build outputs remain at root `target/`, and CI checks Skippy out under `.deps/skippy`. The generated website output is root `docs/`; do not hand-edit it.

## Shared docs

| Doc | What it covers |
|---|---|
| `README.md` | Quickstart and documentation hub |
| `CONTRIBUTING.md` | Build from source and development workflow |
| `RELEASE.md` | Release process and the checks a release still lacks |
| `ROADMAP.md` | Future directions |
| `.github/AGENTS.md` | What the CI workflows run |
| `mesh/AGENTS.md` | MeshLLM host, protocol, UI, deployment, and crate guidance |

The canonical repo-local skills live under `.agents/skills/`; the selectable release-validation specialist is defined in `.agents/agents/release-validation.md`.

## Building

Always use `just`. Never build manually. Bare `just` (or `just build`) downloads the runtime published by the pinned Skippy release (into `target/debug/native-runtimes`), then the MeshLLM host and its console; `just mesh` builds the host alone. The runtime is built from source instead when the pinned release has no package for the machine and backend, when another Skippy source is selected, or with `MESH_LLM_RUNTIME_FROM_SOURCE=1`. `just release-build` builds the release host and no runtime.

`cargo check` and `cargo build` do not count as complete product builds: they skip the console, and `cargo check` produces no binary. See `mesh/AGENTS.md` for packaging commands and `CONTRIBUTING.md` for the full workflow.

To build against a local Skippy checkout instead of the pinned release, run `scripts/skippy-source.sh <checkout>`, or `scripts/skippy-source.sh --ref <branch|tag|commit|pull/N> [repository-url]` to have it fetched; `scripts/skippy-source.sh --pinned` goes back. The override is `.cargo/skippy-local.toml`, which is not committed. A checkout at another version temporarily changes `Cargo.toml`; do not commit it or the `Cargo.lock` it produces.

For CI cross-repository testing, put `Skippy-Ref: <branch|tag|full-SHA|pull/N>` in the PR description. All Rust checks use that revision, but it is **test-only**: required merge checks stay red until the directive is removed and fresh CI passes against the committed pin. See `.agents/skills/manage-ci/SKILL.md` before using or changing this flow.

### Distribution builds

Packagers build MeshLLM from source in a clean environment and install them as system packages. Keep that possible:

- No build step downloads when its input is already provided (a pinned source checkout, vendored crates and node modules). Packagers can supply their own tools instead of bundled binaries: a tool the build needs is taken from the environment when it is set, as `PROTOC` is; the bundled fallback stays for builds that don't set one.
- The code compiles with the newer compilers distributions ship, not only with CI's: include what you use.
- Build steps work with read-only inputs, and installed files stay read-only: keep state under the user's or service's directories. Replacing the binary in place is for release-bundle installs only.
- Every connection to an outside service a packaged node doesn't need (the update check, release and model catalogs, analytics) can be turned off by a setting, and local features work without a CA bundle.
- A distribution build can keep the runpaths of native libraries, and compatibility is checked against the libraries the process actually loads.
- Prefer platform APIs to spawning system tools (`ps`, `sysctl`, `hostname`); a minimal system may not have them.
- An empty path or endpoint variable counts as unset: service units and environment files produce them. Validated overrides keep rejecting an empty value (`MESH_LLM_NO_RUNTIME_DOWNLOAD=` fails closed), and presence flags keep their meaning.
- Anything that refers to an installed binary, such as a service unit, uses the stable command on `PATH`, not a versioned path behind it.

## Shared module design

Do not introduce generic buckets.

- Avoid directories or modules named `app`, `utils`, `misc`, `common`, or similar catch-alls.
- Name modules after the responsibility they own.

Keep shared code honest.

- If code is only used by one subsystem, keep it inside that subsystem.
- Only move code to a shared module or workspace crate when it is truly cross-domain.
- Do not create shared helpers prematurely.

Prefer semantic grouping over symmetry.

- Do not create one directory per file just for visual symmetry.
- A single `foo.rs` file is already a Rust module; use a directory only when `foo` has meaningful substructure.

Minimize crate-root re-exports.

- Root re-exports are acceptable as temporary compatibility shims during refactors.
- New code should prefer importing from the owning module directly.
- Remove transitional re-exports once call sites have been updated.

When to split a file.

- Split a file when it contains multiple separable responsibilities, when navigation becomes difficult, or when tests naturally cluster by concern.
- Do not split purely to reduce line count if the code still represents one coherent object or subsystem.

1k LoC refactoring rule.

- When touching a source file that is already over 1,000 lines, first check whether the change adds or exposes a separable responsibility.
- If it does, split that responsibility into a semantically named module as part of the change, and keep the new file under 1,000 lines.
- If a full split is too risky for the current task, make the smallest useful extraction and call out the remaining oversized file in the final summary.
- Add or move tests so the extracted module owns tests for the behavior it now owns.
- Do not create generic buckets just to reduce line count; split by domain responsibility and keep ownership obvious.

Naming rule.

- File and module names should describe responsibility, not implementation detail.
- Prefer names like `affinity`, `discovery`, `transport`, `maintenance`, `warnings`.
- Avoid vague names like `helpers`, `stuff`, `logic`, or `manager` unless the abstraction is genuinely that broad.

When to add a new workspace crate.

- Prefer adding modules inside an existing crate first.
- Add a new crate under its owning product only when the responsibility is genuinely cross-cutting (used by host and client, or host and a separate binary) or when isolating compile time / dependencies for a specific binary or FFI surface.
- New crates should be named after the responsibility they own, not the consumer.

Product crate documentation.

- Every crate under `mesh/crates/` needs a non-empty `description` in its `Cargo.toml` and a `README.md` at the crate root.
- The README should identify the crate's purpose, what it owns and does not own, and its primary consumers. Add build or usage instructions when the crate exposes a binary or a workflow that contributors run directly. Keep small crates' READMEs concise; do not add boilerplate sections merely for symmetry.
- Use working relative links for local files and neighboring crates. Neither `ci.yml` nor `just lint` checks this today, so review it by hand.

## Code Quality Rules for New Code

- Do not add Rust methods or functions over the configured Clippy line-count
  limit. Split long logic into semantically named helpers before it reaches the
  configured `too_many_lines` threshold.
- Do not add Rust source files over 2,000 lines. If a file is approaching that
  size, split it by responsibility into an owning module instead of adding more
  code to the oversized file.
- Do not add Rust code over the configured cognitive-complexity limit. Prefer
  small, named decision helpers and clear control-flow phases instead of nested
  branching.
- Treat these as design constraints for new code, not cleanup suggestions after
  the fact. CI runs Clippy with warnings denied, so configured Clippy warnings
  must be resolved before a PR can pass.

## Testing

Run Cargo commands serially. Do not run multiple Cargo commands in parallel: this workspace frequently hits package-cache and artifact-directory lock conflicts.

- For a touched workspace crate, run focused `cargo check -p <crate>` and `cargo test -p <crate> --lib` during iteration.
- Before a PR, run the full `cargo test -p <crate>` for the crates you changed and the workspace crates that depend on them. CI runs `just test-rust --locked`, which is `cargo test --workspace`, on Linux, macOS and Windows.
- `just test-all` runs what a pull request must pass except the inference smoke: `just lint`, `just test-rust` and `just test-ui`. Use it for a broad change, and when you change a `just` recipe or a script that CI calls.
- `--lib` skips the integration-test binaries and the doc tests, and CI runs both, so a `--lib`-only run can be green while CI is red.
- `cargo clippy --all-targets` type-checks test code but never executes it, so a green Clippy run is not a test run either. Treat a failure caught locally as cheap and a failure caught by pushed CI as expensive.
- CI's inference smoke (`just smoke <runtime-dir> <model.gguf>`, Unix only) needs a packaged Skippy CPU runtime and a GGUF model. `scripts/fetch-skippy-runtime.sh <dir> cpu` downloads the one the pinned Skippy release published, which is what CI does; `scripts/ci-skippy-runtime.sh` builds it from a Skippy checkout, which CI does only for a named `Skippy-Ref`.
- For broad refactors, fall back to `cargo check --workspace` (serially).
- Follow the owning product's additional test and compatibility guidance.
- When CI fails on a PR, first confirm the failing run belongs to the current head commit and
  actually failed (not `cancelled` or superseded by a newer push) before diagnosing; a rerun keeps
  the original commit SHA, so also check whether the job passed in a later attempt of the same run.
  Then read the failing job's log directly rather than polling the check rollup, which lags and
  keeps stale results from superseded runs.

Repository automation is plain `just` recipes and scripts under `scripts/`, with their tests under `scripts/tests/`. The `tools/xtask` crate and its `cargo xtool` and `cargo xtask` commands were removed with the in-tree Skippy build; do not call them or bring them back. Put logic in a script that can run locally, not in workflow YAML, and wire a new script test into `just lint`.

### Tests in clean build environments

Distribution packagers run the test suite while building MeshLLM, in a clean environment that has only the declared build dependencies, often on busy or slow machines. Every test that runs by default must pass there; a test that really needs the network or particular hardware is opt-in, behind a feature gate (as `claude-live-model-integration` is) or `#[ignore]`d with the reason.

- No network: no DNS, no outgoing connections, no CA bundle. Serve fixtures locally, as the lifecycle tests do with a local `HF_ENDPOINT`.
- Only declared tools: no fixed paths such as `/usr/bin/true` or `/bin/bash`, and no system tools a minimal environment lacks (`ps`, `sysctl`). Use the standard library, or find a tool on `PATH`.
- No usable home: `HOME` may be read-only or missing. Keep all state in the test's own temporary directories and pass those paths to child processes.
- A foreign environment: the builder may set any variable, including MeshLLM's own `MESH_LLM_*` ones, and there may be no terminal. Set or remove what a test depends on for the child process (`Command::env`, `env_remove`); otherwise follow the process-environment mutation contract in `mesh/docs/design/TESTING.md`.
- Slow and busy machines: don't let a result depend on machine speed. Wait for the event or state itself instead of sleeping, give each wait a generous bound, and retry only failures that are transient by nature, such as `ETXTBSY` right after writing an executable.
- Build options: tests also run with the options packagers use, such as the dynamic native runtime.
- Large test binaries: a debug test binary can be gigabytes, so code on the startup path doesn't read its own executable in full when it has no release attestation; verifying a stamped binary still reads it all.

## Pre-Commit Checklist

Before committing, run the local checks most likely to fail in CI for the files you touched. Do not rely on CI to catch basic formatting, compile, or stale UI build issues.

Use a clear, user-focused commit subject and PR title. Conventional Commit prefixes are optional hints for release-note classification, not a hook or CI requirement. The PR title becomes the squash-merge subject; unclassified titles appear under Other changes. See `.agents/skills/release-notes/SKILL.md`.

### Minimum validation by change type

Choose validation from the changed surface, not merely from the directory that contains the changed file. A non-Rust file under a Rust crate does not require Cargo validation only when it cannot affect Cargo metadata, build scripts, generated Rust, or the shipped binary. Treat Cargo and build configuration changes as Rust-impacting.

- Rust change — format changed Rust files and run `cargo check -p <touched-crate>` plus `cargo clippy -p <touched-crate> --all-targets -- -D warnings`. Follow the owning product's additional validation rules.
- Python SDK change — use the in-tree component under `mesh/sdk/python`. Its component command is `"$SDK_PYTHON" -I -B mesh/sdk/python/tests/test_client.py`. Binding generation uses this repository's Mesh source, the existing UniFFI 0.32.0 generator and an already built native library. `ci.yml` does not run the Python SDK checks today, so run them yourself.
- UI-only change — follow `mesh/AGENTS.md` and run `just build`.
- CI workflow or CI script change — run `actionlint` and `just lint` (ShellCheck and the script tests), plus any additional checks required by the CI section below.
- Documentation, non-build configuration, or non-CI shell-only change — run the targeted formatter, generator check, contract test, or syntax check for the changed surface, plus tests that consume the changed surface when such tests exist.
- Mixed change — run the union of the checks required for each changed surface.

Do not rerun otherwise unchanged validation solely because a commit is about to be pushed. Rerun affected checks when the validated diff or commit has changed.

### Rust changes

- The preferred Rust edition for this workspace is Rust 2024. Determine the edition from the owning crate's `Cargo.toml`; if it uses `edition.workspace = true`, read `workspace.package.edition` from the root `Cargo.toml`. Most crates inherit `edition = "2024"` from the root; any crate that opts out declares its own edition in its `Cargo.toml`.
- Format Rust files in a way that preserves the owning crate's edition metadata. Prefer `cargo fmt -p <crate> -- path/to/file.rs` for a narrow edit, or `cargo fmt --all` when changes span packages. Do not use `cargo fmt --all -- path/to/file.rs`: workspace-level file arguments can be parsed without the owning crate's Rust 2024 edition metadata and fail on let-chains.
- If you must invoke `rustfmt` directly on a standalone file, pass the edition resolved from that manifest lookup, for example `--edition 2024` for the current workspace default; otherwise use `cargo fmt` through the owning package.
- Before committing Rust changes, ensure the formatting check passes with `cargo fmt --all --check`.
- After Rust changes, run `cargo check` and `cargo clippy --all-targets -- -D warnings` for each touched crate (`-p <crate>`). Follow the owning product's additional validation rules.
- If a change is reachable from the shipped MeshLLM binary, also run `cargo check -p mesh-llm` and `cargo clippy -p mesh-llm --all-targets -- -D warnings`.
- Treat Clippy as a required local gate, not a CI-only cleanup step. `cargo check`, `just build`, and formatter success do not catch lints such as `clippy::collapsible-if`; run the warning-denying Clippy command before opening or updating a PR.
- If you touched tests, public APIs, routing, inference, gossip, plugin protocol, or CLI behavior, run the relevant tests before committing.
- Do not report a build or test step as complete until the command has actually exited with code `0`.
- Run Rust validation serially. Do not run multiple `cargo` commands at the same time.

### CI changes

Before editing CI, read `.github/AGENTS.md` and `.agents/skills/manage-ci/SKILL.md`. Both are short. The workflow list is in `.agents/skills/manage-ci/references/current-inventory.md`; update it in the same change as a workflow.

The rules for the CI that built Skippy in this repository are kept only as history under `.agents/skills/manage-ci/references/`. Do not follow them.

### Commit standard

- Do not commit if formatting has not been applied.
- Do not commit if basic local validation for your change type has not been run.
- Do not commit known warnings in code you touched.

## Warnings

Do not leave Rust compiler warnings behind in code you touched.

- Fix or remove unused code, dead code, and other warnings introduced or surfaced by your change before committing.
- Do not silence warnings with `#[allow(...)]` unless there is a clear reason and the developer has asked for that tradeoff.

## SDK parity invariant

Rust, Node.js, Python, Swift, and Kotlin must support the same public SDK
capabilities. Language conventions may change names or async syntax, but not
the supported operations, filters, pagination, metadata, availability semantics,
or error behavior. A public SDK API change is incomplete until its native bridge,
all five language surfaces, tests, product docs, and maintained website examples
are updated together. Do not silently substitute an HTTP compatibility call for
an API documented as native, or expose fewer model workloads in one binding.

Website SDK examples are part of this invariant: each supported public SDK
capability must have equivalent, current examples for Rust, Node.js, Python,
Swift, and Kotlin. Keep constructors, option defaults, request/response shapes,
pagination, streaming/error handling, and resource cleanup consistent with the
shipped APIs. An implementation change is not complete with source code alone;
update and validate the corresponding examples under `mesh/website/src/docs/`
and the owning SDK guides in the same repository change.

The invariant has a completion condition: a public SDK API change is complete
only when every language surface's own checks and maintained examples pass
against the same native API revision. Rust, Node.js, Python, Swift, and Kotlin
are all maintained inside this repository. Until that condition holds, the
change is in progress, not shippable, regardless of which surfaces have landed.

The Python SDK source, tests, locked client environments and generated bindings
are maintained under `mesh/sdk/python`. Generate bindings from the exact native
library metadata with the pinned generator; do not hand-edit generated bindings.
Document any unavoidable platform restriction explicitly, with its reason and
parity follow-up, rather than claiming parity based only on a common method name.

## Pull Requests

Pull request titles and descriptions should be user-focused by default.

- Prefer the GitHub CLI (`gh`) for GitHub operations in this repo, including inspecting issues/PRs, editing PR descriptions, pushing branches, and opening PRs. Use built-in MCP/GitHub connector tools only as a fallback or for read-only lookup when `gh` cannot provide the needed data.
- Title PRs around the user-visible change or capability, not the implementation detail.
- Start the description with what the user can now do, see, or understand after the change.
- Keep architectural refactors, internal state reshaping, and code-organization notes out of the opening summary unless they directly change user behavior.
- If there are important architectural changes, add a separate `## Architecture` section.
- If there are protocol or compatibility implications, add a separate `## Protocol` section that clearly calls out compatibility, migration, or breaking-change impact.
- If the PR changes CLI behavior or touches user-facing CLI flows, include example commands and representative output in the PR description.
- If the PR changes the UI, include at least one screenshot in the PR description.
- Validation and screenshots should stay separate from the user-facing summary.

## Credentials

Test machine IPs, SSH details, and passwords are in `~/Documents/private-note.txt` (outside the repo). **Never commit credentials to any tracked file.**
