# MeshLLM CI inventory

What is checked in under `.github/`. Update this file in the same change as a
workflow. Every job runs on a GitHub-hosted runner; there are no self-hosted
runners.

## Workflows

| File | Starts on | What it does |
|---|---|---|
| `ci.yml` | pull request (opened, synchronize, reopened, edited), push to `main`, manual with optional `skippy_ref` | Lint, Rust tests, console and public website checks, CPU inference smoke |
| `release.yml` | manual with `version` and `canary` | Builds, signs, checks and publishes a release |
| `website-pages.yml` | push to `main` touching the website, installers, crates or build files; manual | Builds the website and deploys it to GitHub Pages |
| `fly-deploy-console.yml` | manual with optional `ref` | Deploys the console app to Fly |
| `docker.yml` | manual | Checks `mesh/deploy/docker/Dockerfile.client` with `docker buildx build --check` |
| `pr_auto_assign.yml` | `pull_request_target` | Assigns pull requests |
| `stale-prs.yml` | daily at 00:00 UTC; manual | Warns on and closes stale pull requests |

## `ci.yml` jobs

| Job | Runner | Command |
|---|---|---|
| Lint | `ubuntu-24.04` | `just lint` |
| Test | `ubuntu-24.04`, `macos-15`, `windows-2022` | `just test-rust --locked --no-fail-fast` |
| Console | `ubuntu-24.04` | `just test-ui` |
| Public website | `ubuntu-24.04` | `scripts/ci-website.sh` |
| Inference smoke (CPU) | `ubuntu-24.04` | `scripts/ci-resolve-skippy.sh`, `scripts/fetch-skippy-runtime.sh` (pinned release) or `scripts/ci-skippy-runtime.sh` (named ref), `just mesh`, `just smoke`, `just test-with-runtime` |

`just lint` is `cargo fmt --all -- --check`, Clippy on the workspace with
warnings denied, ShellCheck on `scripts/*.sh`, `scripts/tests/*.sh` and
`mesh/scripts/*.sh`, and the script tests in `scripts/tests/`.

The smoke job downloads the pinned release's CPU runtime, a pinned test model,
and checks the model SHA-256. A named Skippy ref instead checks source out under
`.deps/skippy` and builds its CPU runtime. The model is cached; the compiler
cache is sccache's GitHub backend.

The workflow has `contents: read` and uses no secret or variable. A pull
request cancels its own earlier run.
`PR / Website` requires the public website build and data checks.

## `release.yml` jobs

| Job | Runner | What it does |
|---|---|---|
| `release_readiness` | `ubuntu-24.04` | Lists the missing checks; fails a real release, lets a canary continue |
| `metadata` | `ubuntu-24.04` | Prepares the release source and tag (`contents: write`) |
| `runtime_download_gate` | `ubuntu-24.04` | Refuses runtimes that differ from their catalog entry |
| `build` | `macos-15`, `ubuntu-24.04`, `ubuntu-24.04-arm`, `windows-2022` | Builds, stamps, packages and checks one host archive per platform |
| `swift_sdk` | `macos-15` | Swift SDK artifact |
| `kotlin_sdk` | `macos-15`, `ubuntu-24.04`, `ubuntu-24.04-arm` | Kotlin SDK native libraries |
| `node_sdk` | `macos-15`, `ubuntu-24.04`, `ubuntu-24.04-arm`, `windows-2022` | Node SDK addons |
| `publish` | `ubuntu-24.04` | Publishes the GitHub release (`contents: write`); skipped for a canary |
| `dispatch_packaging_release` | `ubuntu-24.04` | Fails on purpose; runs only if `MESH_PACKAGING_DISPATCH` is `true` and the release is not a prerelease |

The publish job calls `scripts/create-compatibility-aliases.sh` for the
0.80.0 CUDA and ROCm archive names. The Vulkan alias is withheld until a
matching Skippy runtime is published.
The build matrix runs one packaged-host CPU inference smoke on Linux x86_64.
The other platforms validate their archives without repeating inference.
Representative CUDA and ROCm host smokes remain a release-readiness item;
Skippy owns runtime backend qualification.

Runs of this workflow do not cancel each other. See `RELEASE.md` for the steps
and for what is not yet checked. Apart from local runs of its scripts, this
workflow has not run on GitHub in its present form.

## Other workflows

- `website-pages.yml`: the workflow has `contents: read` and `packages: read`.
  `build` runs on `ubuntu-24.04` inside the container
  `ghcr.io/mesh-llm/mesh-llm-cuda-runner`, pinned by digest; a manual run
  builds only from `refs/heads/main`. `deploy` then runs with `pages: write`
  and `id-token: write` in the `Public Website` environment. Its checkout does
  not set `persist-credentials: false`.
- `fly-deploy-console.yml`: one job on `ubuntu-24.04` in the `fly-console`
  environment. Its checkout does not set `persist-credentials: false`.
- `pr_auto_assign.yml`: one job with `issues: write` and `pull-requests: write`.
- `stale-prs.yml`: one job with `issues: write` and `pull-requests: write`.

Every other checkout sets `persist-credentials: false`, including the two
`release.yml` jobs that push; they push with an explicit token URL.

## Local action

`.github/actions/setup` installs stable Rust with `rustfmt`, `clippy` and
`llvm-tools-preview`, sccache v0.16.0, `just`, the Linux build packages
(including `mold`, `lld`, `jq`), and `lld` on macOS.

## Secrets and variables

| Name | Kind | Used by |
|---|---|---|
| `MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE` | secret | `release.yml` |
| `MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE` | secret | `release.yml` |
| `MESH_LLM_POSTHOG_KEY` | secret | `release.yml` (compiled into release binaries) |
| `GITHUB_TOKEN` | built in | `release.yml` |
| `FLY_API_TOKEN` | secret | `fly-deploy-console.yml` |
| `MESH_PACKAGING_DISPATCH` | variable | `release.yml` (must stay off) |
| `STALE_PR_DAYS`, `STALE_PR_WARNING_DAYS` | variables | `stale-prs.yml` |

Names only. Never record a value here.

## Environments

`Public Website` (GitHub Pages) and `fly-console`.
