# MeshLLM CI inventory

Read this inventory with `SKILL.md` before editing CI. It records the
checked-in contract, not guaranteed live GitHub state. Verify live state with
the commands at the end before operational changes.

## Workflow ownership

| Workflow | Trigger | Ownership |
| --- | --- | --- |
| `pr_quality.yml` | PR, main push, dispatch | Formatting, affected-crate Clippy, UI quality, CLI/docs synchronization, quality summary |
| `pr_builds.yml` | PR, dispatch | Cross-platform build/test matrices, native backends, artifact producers, integration/smoke consumers |
| `pr_website.yml` | PR, dispatch | Public website build canary and summary |
| `pr_cleanup.yml` | PR close via `pull_request_target`, dispatch | Positively matched PR cache/artifact cleanup only; never executes PR code |
| `pr_auto_assign.yml` | PR lifecycle via `pull_request_target` | PR metadata assignment only; never executes PR code |
| `ci.yml` | Main push, dispatch | Trusted main CI equivalent of build/test/smoke lanes |
| `docker.yml` | Dispatch | Manual client Dockerfile validation; does not publish |
| `docker-precheck.yml` | Reusable call | Shared Docker validation precheck |
| `smoke.yml` | Reusable call | Artifact-based inference/OpenAI/split smoke |
| `scripted-binary-smoke.yml` | Reusable call | Artifact-based scripted/two-node smoke |
| `sdk-smoke.yml` | Reusable call | Native, Kotlin, and Swift SDK smoke |
| `hf-download-smoke.yml` | Reusable call | Hugging Face download smoke |
| `nightly-stability.yml` | Schedule, dispatch | Nightly operator entry point |
| `nightly-stability-run.yml` | Reusable call | Stability probes and evidence |
| `llama-upstream-canary.yml` | Schedule, dispatch | Upstream llama.cpp compatibility canary |
| `queue-unsloth-layer-packages.yml` | Schedule, dispatch | Hugging Face layer-package job queueing |
| `windows-warm-caches.yml` | Main path push, dispatch | Trusted Windows ABI cache warming |
| `website-pages.yml` | Main website path push, dispatch | Public website Pages build/deploy |
| `fly-deploy-console.yml` | Dispatch | `fly-console` environment deployment |
| `release.yml` | `v*` tag, dispatch | Release builds, attestations, publishing, downstream image dispatch |
| `reset-caches.yml` | Confirmed dispatch | Destructive repository cache reset |
| `stale-prs.yml` | Schedule, dispatch | PR warning/closure maintenance |

Local actions:

- `.github/actions/compute-changes` owns path, crate, backend, SDK, UI, website,
  Windows, and docs-only routing outputs.
- `.github/actions/restore-smoke-inputs` owns producer artifact staging and
  model restoration for smoke consumers.
- `.github/actions/setup-windows-rocm-sdk` owns reusable Windows ROCm setup.

Routing and test-planning scripts:

- `scripts/affected-crates.sh` computes affected crates and reverse dependents.
- `scripts/plan-clippy-batches.sh` owns weighted Clippy sharding and retains a
  checked workspace-member list for fail-open/all-rust planning.
- `scripts/plan-test-batches.sh` owns weighted crate-test sharding. It derives
  workspace membership from `cargo metadata`; new crates must not be added to a
  workflow-owned test allowlist.
- `scripts/test-portable.sh` owns the portable non-Cargo test aggregate used by
  the local `test-all` path.

## Runner and image contract

GitHub-hosted labels currently used:

- Linux AMD64: `ubuntu-24.04`
- Linux ARM64: `ubuntu-24.04-arm`
- macOS: `macos-15` and legacy `macos-latest`
- Windows: `windows-2022`

Legacy/dedicated self-hosted label arrays currently referenced:

- NVIDIA AMD64: `["self-hosted","Linux","X64","amd64","gpu-nvidia"]`
- ARM64: `["self-hosted","Linux","ARM64"]`

ARC scale-set labels for the prebuilt runner rollout:

- `mesh-llm-amd64`
- `mesh-llm-arm64`

`pr_builds.yml` runs `public_runner_image_contract` in the public image and
`arc_runner_image_contract` on both ARC labels for every pull request. The ARC
job executes directly in each ephemeral runner pod, verifies the self-hosted
image contract and native architecture, and runs a small Rust check. It
intentionally has no hosted fallback because its purpose is to detect an ARC,
K3s scheduling, architecture, or runner-image regression before merge.

Runner images are published from
[`Mesh-LLM/mesh-llm-runner-images`](https://github.com/Mesh-LLM/mesh-llm-runner-images)
as `ghcr.io/mesh-llm/mesh-llm-cuda-runner`. The source repository owns:

- `profiles/common.yml`
- `profiles/backends/{cpu,vulkan,cuda,rocm}.yml`
- `profiles/public.yml`
- `profiles/self-hosted.yml`
- CUDA/ROCm toolchain installers, manifest collection, dependency warming, and
  backend compiler-probe verification
- AMD64/ARM64 CPU, Vulkan, CUDA 12, and CUDA 13 images
- AMD64 ROCm 7.0 and ROCm 7.2 images

Production consumers must use the multi-architecture manifest digest. Tags are
discovery inputs and are mutable absent separately verified registry controls.

The public repository and its GHCR package have independent visibility. Until
anonymous pull of the package succeeds, GitHub-hosted container jobs must grant
`packages: read` and provide `github.actor`/`secrets.GITHUB_TOKEN` through
`container.credentials`. Do not assume making the source repository public also
makes an existing package public.

The production rollout covers the shared public CPU environment and explicit
public Vulkan, CUDA, and ROCm overlays in `pr_builds.yml`, `ci.yml`,
`pr_quality.yml`, and Linux release jobs. Backend images standardize compilers
and SDKs; actual GPU access remains a separate runner label, node resource, and
trust-boundary contract. Do not route untrusted PR code to persistent GPU
runners merely because the same image can also run as an ARC pod.

The image family built from MeshLLM revision
`5f341d6828fc77cce2f3be43f2a6ff26f3223433` is:

| Image | Immutable index digest |
| --- | --- |
| public CPU | `sha256:8d93de6ba30173e825a16fdecf011f9c632edc6e1259df7289e491b0a05f829d` |
| public Vulkan | `sha256:ce55fed5c680cd3184b5d4770d9a77c43a702687690906e5753efd2cea27ed80` |
| public CUDA 12 | `sha256:c5b85ef527230f77cf9933ef40bcb44316f9bbcb8fd2ce0651b58acda5143dfd` |
| public CUDA 13 | `sha256:6b87598605f5d8deeafecfb1a55027e0ca9e47f4fc6f230d030487c450c31aa6` |
| public ROCm 7.0 | `sha256:0e13e5d2d2c121df265ff6c69be81e468989e09f81d6b7ff049b110cc0bb0d2b` |
| public ROCm 7.2 | `sha256:6b88ca9371ada2c507d6e36b71f0e0538fee378c6a5e2b39c17249b4b7e5088a` |
| self-hosted compatibility | `sha256:37e0ce710eae44952306c4a553cf89fdf94c009660a2a8fa04bba4d202a32baf` |

MeshLLM workflows pin the public digest. The Flux repository must independently
roll the ARC HelmReleases to the paired self-hosted digest; that cross-repository
change cannot be delivered by a MeshLLM pull request.

Public-image Rust jobs use the baked `sccache` binary with
`SCCACHE_GHA_ENABLED=false`, so compiler startup does not depend on the
availability of GitHub's cache service. Persistent Cargo target and ABI reuse
remains owned by `Swatinem/rust-cache` and `actions/cache`. Do not reintroduce
the sccache download action merely to export credentials.

`USE_SELF_HOSTED` currently controls selected GPU/release routes. Unset or a
value other than the exact string `true` selects the hosted fallback. Any new
route must preserve a safe hosted fallback or document why one cannot exist.

## Repository variables referenced by workflows

All GitHub Actions variables are strings.

| Variable | Purpose and fallback |
| --- | --- |
| `USE_SELF_HOSTED` | Exact `true` selects supported self-hosted GPU/release lanes; otherwise hosted |
| `CUDA_VERSION` | Windows CUDA toolkit selection; Linux CUDA lanes use digest-pinned backend images |
| `VULKAN_SDK_VERSION` | Windows Vulkan SDK; fallback `1.4.328.1` |
| `LLAMA_UPSTREAM_CANARY_SMOKE` | Enables canary smoke; fallback `1` |
| `LLAMA_WINDOWS_CACHE_RETENTION` | Windows warm-cache retention; fallback `2` |
| `PR_CACHE_CLEANUP_WORKERS` | Cleanup fan-out; default `5`, validated range `1..20` |
| `STALE_PR_DAYS` | PR close threshold; fallback `7` |
| `STALE_PR_WARNING_DAYS` | PR warning threshold; fallback `2` |
| `MESH_AGENT_BASE_URL` | Preferred agent smoke endpoint |
| `MESH_AGENT_MODEL` | Preferred agent smoke model |
| `MESH_OPENCODE_BASE_URL` | Legacy/fallback agent smoke endpoint |
| `MESH_OPENCODE_MODEL` | Legacy/fallback agent smoke model |
| `AGENT_SMOKE_LONG_PROMPT_CHARS` | Preferred long-prompt size; falls back through OpenCode value to `65536` |
| `OPENCODE_SMOKE_LONG_PROMPT_CHARS` | Legacy/fallback long-prompt size; fallback `65536` |
| `MESH_NIGHTLY_STABILITY_ENABLED` | Exact `1` enables scheduled stability; dispatch bypasses the gate |
| `MESH_NIGHTLY_STABILITY_BASE_URL` | Nightly endpoint fallback |
| `MESH_NIGHTLY_STABILITY_MODELS` | Model list; fallback `auto,mesh` |
| `MESH_NIGHTLY_STABILITY_ATTEMPTS` | Attempts per model; fallback `5` |
| `MESH_NIGHTLY_STABILITY_AGENT_SMOKES` | Optional agent CLI smoke list |
| `MESH_NIGHTLY_STABILITY_TIMEOUT` | Per-probe seconds; fallback `180` |
| `MESH_NIGHTLY_STABILITY_RUNS_ON` | JSON runner label string/array; fallback `"ubuntu-24.04"` |

## Secret names referenced by workflows

Never record values in this inventory.

| Secret | Consumer and scope |
| --- | --- |
| `HF_TOKEN` | Download, smoke, queue, and release jobs that access Hugging Face |
| `FLY_API_TOKEN` | `fly-deploy-console.yml`; use an app-scoped deploy token and `fly-console` environment |
| `MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE` | Release attestation signing material |
| `MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE` | Release attestation public material |
| `MESH_AGENT_IMAGES_DISPATCH_TOKEN` | Cross-repository release dispatch to image publishing; may be organization-scoped |
| `CARGO_REGISTRY_TOKEN` | Crates.io publishing in release |

`GITHUB_TOKEN`/`github.token` is built in and governed by each workflow/job's
`permissions`. Do not create it as a repository secret.

Fork PRs do not receive repository secrets. A secret absent from the repository
list may be environment- or organization-scoped; verify scope instead of
assuming it is missing.

## Environments and privileged operations

Checked-in workflows reference:

- `fly-console` for Fly deployment
- `Public Website` for Pages deployment

GitHub may also expose platform-managed environments such as `github-pages`.
Query current protection rules before changing a deploy job. Do not remove an
environment gate to bypass approval or secret access.

Privileged/destructive paths include release publishing, Fly/Pages deployment,
cross-repository dispatch, crate publishing, PR cache cleanup, repository cache
reset, and any job with write permissions. Inspect permissions and concurrency
before modifying these paths.

## Live-state inspection commands

```bash
gh workflow list --all --repo Mesh-LLM/mesh-llm
gh run list --repo Mesh-LLM/mesh-llm --limit 30
gh variable list --repo Mesh-LLM/mesh-llm
gh secret list --repo Mesh-LLM/mesh-llm
gh api repos/Mesh-LLM/mesh-llm/environments
gh api repos/Mesh-LLM/mesh-llm/actions/runners
gh api repos/Mesh-LLM/mesh-llm/actions/permissions
gh api repos/Mesh-LLM/mesh-llm/actions/permissions/workflow
gh api repos/Mesh-LLM/mesh-llm/branches/main/protection
```

Use `gh variable list --env NAME` and `gh secret list --env NAME` for an
environment. Organization-level listing requires additional GitHub permission;
record a 403 as unverified scope, not as absence.
