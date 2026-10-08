# Agent Notes

## Repository and instruction scope

This is one Cargo workspace with two products: standalone Skippy in `skippy/` and MeshLLM in `mesh/`. Mesh depends on Skippy; Skippy must not depend on Mesh crates or plugin hosts. Product-specific guidance is in `skippy/AGENTS.md` and `mesh/AGENTS.md`. Read both when changing their integration.

Root `Cargo.toml`, `Cargo.lock`, `.cargo/`, `just/`, CI, and cross-workspace tooling are shared infrastructure. There is no `shared/` product tree. Product scripts and their tests live under the owning product; root script forwarders preserve CI entrypoints. Build outputs and native working caches remain at root `target/` and `.deps/`. The generated website output is root `docs/`; do not hand-edit it.

## Shared docs

| Doc | What it covers |
|---|---|
| `README.md` | Quickstart and documentation hub |
| `CONTRIBUTING.md` | Build from source and development workflow |
| `RELEASE.md` | Shared release process |
| `ROADMAP.md` | Future directions |
| `skippy/AGENTS.md` | Skippy build, native ABI, serving, and crate guidance |
| `mesh/AGENTS.md` | MeshLLM host, protocol, UI, deployment, and crate guidance |

The canonical repo-local skills live under `.agents/skills/`; the selectable release-validation specialist is defined in `.agents/agents/release-validation.md`.

Generate and check the public Skippy native API documentation through `cargo xtool automation native-generator contracts api-doc` and `cargo xtool automation native-generator contracts api-doc --check`; follow `skippy/AGENTS.md` for the owning native preparation and documentation paths.

## Building

Always use `just`. Never build manually. Bare `just` (or `just build`) builds Skippy first, then MeshLLM. Use `just skippy` or `just mesh` for one product; `just release-build` builds both release products in the same order. The products share a workspace version and release tag but have distinct deliverables.

`cargo check` and `cargo build` do not count as complete product builds: they skip native ABI preparation and the console, and `cargo check` produces no binary. Choose the owning product's `AGENTS.md` for product-specific build and packaging commands. See `CONTRIBUTING.md` for the full workflow.

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

- Every crate under `mesh/crates/` or `skippy/crates/` needs a non-empty `description` in its `Cargo.toml` and a `README.md` at the crate root.
- The README should identify the crate's purpose, what it owns and does not own, and its primary consumers. Add build or usage instructions when the crate exposes a binary or a workflow that contributors run directly. Keep small crates' READMEs concise; do not add boilerplate sections merely for symmetry.
- Use working relative links for local files and neighboring crates. The Quality contract test checks README presence, manifest descriptions, and local Markdown link destinations; it does not substitute for reviewing whether the explanation is accurate.

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
- For broad refactors, fall back to `cargo check --workspace` (serially).
- Follow the owning product's additional test and compatibility guidance.

For new repository automation, follow the Rust ownership rule in
`.agents/skills/manage-ci/SKILL.md` and spec section 7.5: do not write new
Python tooling, including temporary, inline, or skill-local helpers. Reuse or
extend typed `tools/xtask` commands and call them from thin Just recipes;
`cargo xtool repo-consistency ci-crate-lists` is an existing command from the
repository root. Do not move generic policy into shell, PowerShell, or
JavaScript. Existing Python validation and workflow entrypoints below remain
transitional until Rust or component-owned tests cover their shape and intent.
Do not commit Python emulation or differential tests that invoke Python; delete
each legacy implementation with its last caller switch and validate in normal
CI. Preserve real consumed outputs and failure semantics.

## Pre-Commit Checklist

Before committing, run the local checks most likely to fail in CI for the files you touched. Do not rely on CI to catch basic formatting, compile, or stale UI build issues.

Run `just hooks-install` once per clone before your first commit; git cannot activate a committed hook on clone, so nothing else enables it, and `just build` only enables it after a local build. Commit subjects must follow Conventional Commits v1.0.0, and `just check-commits` validates a range. The type decides which release-notes section the change lands in, and the PR title becomes the squash-merge subject, so give the PR a conventional title too. See `.agents/skills/release-notes/SKILL.md`.

### Minimum validation by change type

Choose validation from the changed surface, not merely from the directory that contains the changed file. A non-Rust file under a Rust crate does not require Cargo validation only when it cannot affect Cargo metadata, build scripts, generated Rust, or the shipped binary. Treat Cargo and build configuration changes as Rust-impacting.

- Rust change — format changed Rust files and run `cargo check -p <touched-crate>` plus `cargo clippy -p <touched-crate> --all-targets -- -D warnings`. Follow the owning product's additional validation rules.
- External Python SDK or research interface change — use the owning external component's declared interpreter, source pin and locked dependencies. The SDK component command is `"$SDK_PYTHON" -I -B "$MESH_PYTHON_SDK_SOURCE/sdk/tests/test_client.py"`; its mock FFI coverage is separate from the four genuine SDK clients and their preserved required compatibility or embedding-workload cadence. Set `MESH_PYTHON_SDK_SOURCE` to the admitted external checkout before local source contract gates. Binding generation belongs to that external SDK, with explicit Mesh source and existing UniFFI 0.32.0 generator inputs; native bridge staging requires an already built target library and has no consumer compiler fallback. Reader constructor fixtures use native `just agentic-trajectory-reader-contracts`. Generic repository policy and CI planning use `just ci-validate`, including `ci-automation-contracts` and the native `ci-legacy-contracts`; do not restore the retired generic Python unittest gate.
- UI-only change — follow `mesh/AGENTS.md` and run `just build`.
- CI workflow, planner fixture, or CI script change — run `just ci-validate` plus any additional checks required by the CI section below.
- Documentation, non-build configuration, or non-CI shell-only change — run the targeted formatter, generator check, contract test, or syntax check for the changed surface.
- Mixed change — run the union of the checks required for each changed surface.

Do not rerun otherwise unchanged validation solely because a commit is about to be pushed. Rerun affected checks when the validated diff or commit has changed.

### Rust changes

- After Rust changes, run `just no-console-print`. It forbids direct printing and terminal handles in product code, with no allowlist; use the owning product's output facilities.
- The preferred Rust edition for this workspace is Rust 2024. Determine the edition from the owning crate's `Cargo.toml`; if it uses `edition.workspace = true`, read `workspace.package.edition` from the root `Cargo.toml`. Most crates inherit `edition = "2024"` from the root; any crate that opts out declares its own edition in its `Cargo.toml`.
- Format Rust files in a way that preserves the owning crate's edition metadata. Prefer `cargo fmt -p <crate> -- path/to/file.rs` for a narrow edit, or `cargo fmt --all` when changes span packages. Do not use `cargo fmt --all -- path/to/file.rs`: workspace-level file arguments can be parsed without the owning crate's Rust 2024 edition metadata and fail on let-chains.
- If you must invoke `rustfmt` directly on a standalone file, pass the edition resolved from that manifest lookup, for example `--edition 2024` for the current workspace default; otherwise use `cargo fmt` through the owning package.
- Before committing Rust changes, ensure the formatting check passes with `cargo fmt --all --check`.
- After Rust changes, run `cargo check` and `cargo clippy --all-targets -- -D warnings` for each touched crate (`-p <crate>`). Follow the owning product's additional validation rules.
- If a change is reachable from the shipped MeshLLM binary, also run `cargo check -p mesh-llm` and `cargo clippy -p mesh-llm --all-targets -- -D warnings`, including when the changed code is owned by Skippy.
- Treat Clippy as a required local gate, not a CI-only cleanup step. `cargo check`, `just build`, and formatter success do not catch lints such as `clippy::collapsible-if`; run the warning-denying Clippy command before opening or updating a PR.
- If you touched tests, public APIs, routing, inference, gossip, plugin protocol, skippy ABI, or CLI behavior, run the relevant tests before committing.
- Do not report a build or test step as complete until the command has actually exited with code `0`.
- Run Rust validation serially. Do not run multiple `cargo` commands at the same time.

### CI changes

Before inspecting, running, defining, editing, reviewing, or documenting CI,
read `.agents/skills/manage-ci/SKILL.md` completely. The `manage-ci` skill is
the canonical source for workflow, dependency, runner/worker, image, cache,
artifact, variable, secret, permission, release, deployment, operational, and
validation rules. Start every `.github/`, CI-script, or runner-integration edit
there.

Keep `.agents/skills/manage-ci/references/current-inventory.md` synchronized
with the checked-in CI contract and `ci/ci.md` synchronized with topology. When
a CI rule changes, update the skill first rather than adding duplicate guidance
to this file or `.github/AGENTS.md`.

### Commit standard

- Do not commit if formatting has not been applied.
- Do not commit if basic local validation for your change type has not been run.
- Do not commit known warnings in code you touched.

## Warnings

Do not leave Rust compiler warnings behind in code you touched.

- Fix or remove unused code, dead code, and other warnings introduced or surfaced by your change before committing.
- Do not silence warnings with `#[allow(...)]` unless there is a clear reason and the developer has asked for that tradeoff.

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
