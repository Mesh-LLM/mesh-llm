# GitHub workflow agent rules

These rules apply when editing files under `.github/`, especially workflows,
actions, and CI instructions. Keep pull request CI fast, explicit, and easy to
reason about.

## Workflow ownership

- Keep pull request workflows in files named `pr_*.yml`.
- Keep the early quality workflow named `PR Quality Checks` in
  `pr_quality.yml`.
- Keep the PR build workflow named `PR Builds` in `pr_builds.yml`.
- Keep `ci.yml`, `docker.yml`, and `release.yml` free of pull request triggers;
  they own main, dispatch, tag, and release behavior.
- Keep `pr_cleanup.yml` safe for `pull_request_target`: never check out or run
  pull request code there.

## Routing and build shape

- Route PR work from `.github/actions/compute-changes` outputs; do not add heavy
  jobs that ignore `docs_only`, `rust_changed`, `backend_changed`, or
  `sdk_smoke_required`.
- Keep Linux, macOS, and Windows as top-level target matrices in `pr_builds.yml`.
  Linux/macOS CPU rows are the producer rows for downstream smoke artifacts.
- Keep macOS CUDA, ROCm, and Vulkan rows as explicit unsupported-backend skips.
- Gate backend lanes on backend inputs, not every Rust change.
- Keep clippy sharding driven by `scripts/plan-clippy-batches.sh`; do not
  replace it with hand-maintained static batches.
- When adding a Rust workspace crate, make sure its package name appears in the
  `WORKSPACE_MEMBERS` arrays in `scripts/affected-crates.sh` and
  `scripts/plan-clippy-batches.sh`. Normal affected-crate routing discovers new
  crates through `cargo metadata`, but the all-rust/fail-open paths and clippy
  `--all` planning still use those arrays. `cargo run -p xtask --
  repo-consistency ci-crate-lists` fails fast when they drift from the Cargo
  workspace.
- If workflow changes affect crate/test routing, update
  `tools/xtask/src/main.rs` invariants in the same change.

## Runner image and dependency policy

- Linux CI should use the multi-architecture runner images published by
  [`Mesh-LLM/mesh-llm-runner-images`](https://github.com/Mesh-LLM/mesh-llm-runner-images):
  the `public-*` variant through job-level `container:` on GitHub-hosted
  runners, and the `self-hosted-*` variant as the ARC runner pod image. Do not
  add another job container around an ARC runner pod.
- Pin production consumers by OCI digest, for example
  `ghcr.io/mesh-llm/mesh-llm-cuda-runner@sha256:<digest>`. Tags, including
  timestamp, source-revision, `public-latest`, and `self-hosted-latest`, are
  discovery or evaluation inputs only; resolve them to the published
  multi-architecture digest before updating required CI or Flux resources.
- **Never resolve a missing CI dependency by adding an ad hoc installer to an
  individual workflow.** Do not add new `apt-get install`, `pip install`,
  `npm install --global`, `cargo install`, downloaded binaries, setup scripts,
  setup actions that download an already-standardized toolchain, or equivalent
  host mutation as a one-job fix.
- Put application and test dependencies in MeshLLM's checked-in manifests and
  lockfiles. Put shared system packages and runner tools in the runner image's
  YAML profiles (`profiles/common.yml`, `profiles/public.yml`, and
  `profiles/self-hosted.yml`) or its owning installer script. Rebuild, verify,
  and publish the image, then update the workflow or Flux image reference.
- Locked project dependency installation remains valid job work. The runner
  image build discovers MeshLLM's Cargo, Node, Python, and Go manifests and
  warms their caches, but the checked-in manifest remains authoritative.
- Treat existing workflow-local host setup as migration debt. Do not copy it
  into new jobs. Remove it when the relevant lane adopts the prebuilt image.
- Temporary incident workarounds must be clearly marked with a reason, owner,
  and linked removal issue or expiry date. Review must reject an undocumented
  or permanent workflow-only package workaround.

## Artifact and cache policy

- PR and smoke-only CI artifacts must use short retention. The current policy is
  `retention-days: 1`.
- PR cleanup must delete PR merge-ref caches and artifacts from positively
  matched PR workflow runs without deleting workflow runs or logs.
- Do not save large shared Rust caches from PR merge refs; shared caches are
  written from trusted main/release paths.
- Do not reintroduce unreachable artifact consumers. If a smoke consumes an
  artifact, the producer must upload it in the same workflow graph.

## Smoke test policy

- Smoke jobs should restore producer artifacts through
  `.github/actions/restore-smoke-inputs` instead of rebuilding `mesh-llm` or
  patched llama.cpp.
- Use `smoke.yml`, `scripted-binary-smoke.yml`, `sdk-smoke.yml`, and
  `hf-download-smoke.yml` instead of copying artifact/model restore blocks into
  individual jobs.
- Producer-local smoke steps may stay in CPU rows when they validate the binary
  before upload.
- Every workflow or script invocation of `mesh-llm` must include
  `--log-format json` so CI never starts the TUI by default.

## Documentation and validation

- Keep `ci/ci.md` synchronized with workflow topology changes.
- CI workflow editing rules live here, not in `docs/CI_GUIDANCE.md`.
- Before committing workflow changes, run local validation equivalent to:
  - parse all workflow/action YAML files,
  - check duplicate step IDs,
  - confirm only `pr_*.yml` workflows contain pull request triggers,
  - run `GIT_MASTER=1 git diff --check`, and
  - run `cargo run -p xtask -- repo-consistency release-targets`.
- Validate significant workflow changes with GitHub Actions. If an existing PR
  cannot be reopened and a new PR is not desired, use `workflow_dispatch` on the
  branch and record that caveat.
