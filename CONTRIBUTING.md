# Contributing

Join the [#mesh-llm channel on the Goose Discord](https://discord.gg/goose-oss) for discussion and questions.

This file covers local build and development workflows for this repository.

## Prerequisites

- `just`
- `cmake`
- Rust toolchain (`cargo`)
- Node.js 24 + npm (for UI development)

**macOS**: Apple Silicon. Metal is used automatically.

**Linux NVIDIA**: x86_64 with an NVIDIA GPU. Requires the CUDA toolkit (`nvcc` in your `PATH`). On Arch Linux, CUDA is typically at `/opt/cuda`; on Ubuntu/Debian it's at `/usr/local/cuda`. Auto-detection finds the right SM architecture for your GPU.

**Linux AMD**: ROCm/HIP is supported when ROCm is installed. Typical installs expose `hipcc`, `hipconfig`, and `rocm-smi` under `/opt/rocm/bin`.

**Linux Vulkan**: Vulkan is supported when the Vulkan development files and `glslc` are installed. On Ubuntu/Debian, install `libvulkan-dev glslc`. On Arch Linux, install `vulkan-headers shaderc`.

**Windows**: native runtime builds support `cuda`, `hip`/`rocm`, `vulkan`, or
`cpu`. Metal is not supported on Windows.

## Build from source

Build the normal debug development binary, patched llama.cpp ABI, and UI:

```bash
just build
```

Release and packaging work use a backend-neutral host plus a separate native
runtime. Build the host once:

```bash
just release-host-build
```

Then build the backend runtime you are changing:

```bash
just release-runtime-build cpu
just release-runtime-build metal
just release-runtime-build cuda
just release-runtime-build rocm
just release-runtime-build vulkan
```

Backend toolchains must be available for the corresponding runtime build. For
NVIDIA on Linux, put `nvcc` on `PATH`; runtime packaging detects the selected
CUDA major and architecture from the toolchain/environment.

```bash
PATH=/opt/cuda/bin:$PATH just release-runtime-build cuda
# or
PATH=/usr/local/cuda/bin:$PATH just release-runtime-build cuda
```

Exercise the exact release discovery boundary with an isolated cache:

```bash
runtime_cache="$(mktemp -d)"
MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$PWD/dist/native-runtimes" \
MESH_LLM_NATIVE_RUNTIME_CACHE_DIR="$runtime_cache" \
  ./target/release/mesh-llm runtime list
```

The resolver validates version, Skippy ABI, OS, architecture, and backend. It
does not search the current working directory or copy a matching bundled
runtime into the user cache.

Create a portable product bundle after building both layers:

```bash
just release-bundle v0.X.0 dist
```

## UI development workflow

The React console and embedded asset crate live in `crates/mesh-llm-ui/`.
The host binary serves the built assets through the management API.

Use this two-terminal flow for UI development.

Terminal A (run `mesh-llm` yourself):

```bash
mesh-llm --port 9337 --console 3131
```

If `mesh-llm` is not on your `PATH`:

```bash
./target/release/mesh-llm --port 9337 --console 3131
```

Terminal B (run Vite with HMR):

```bash
just ui-dev
```

Open:

```text
http://127.0.0.1:5173
```

`ui-dev` defaults:

- Serves on `127.0.0.1:5173`
- Proxies `/api/*` to `http://127.0.0.1:3131`

Overrides:

```bash
# Different backend API origin for /api proxy
just ui-dev http://127.0.0.1:4141

# Different Vite dev port
just ui-dev http://127.0.0.1:3131 5174
```

## Useful commands

```bash
just stop             # stop mesh/rpc/llama processes
just test             # quick test against :9337
just check-release    # release-target/docs/workflow parity check
just compat-smoke ~/.cache/huggingface/hub/<model>.gguf   # optional 2-node + 1-client Python/Node/LiteLLM smoke
just --list           # list all recipes
```

On native Windows, `just check-release` runs the host-safe Rust/doc invariant subset and skips the Bash-only `install.sh` / `package-release.sh` parity checks. Run it on macOS or Linux when you need full shell parity coverage.

## CI / GitHub Actions

CI uses [`dorny/paths-filter`](https://github.com/dorny/paths-filter) to skip jobs when unchanged areas of the repo are modified. A `changes` detection job runs first on every push and PR, then each build job gates on its output.

For the current PR build topology, see [`ci/ci.md`](ci/ci.md). Agents must start
every CI edit with the canonical
[`manage-ci` skill](.agents/skills/manage-ci/SKILL.md); `.github/AGENTS.md`
routes all GitHub workflow work there.

Linux CI uses prebuilt public and self-hosted images from
[`Mesh-LLM/mesh-llm-runner-images`](https://github.com/Mesh-LLM/mesh-llm-runner-images).
CPU, Vulkan, versioned CUDA, and versioned ROCm images share a core environment,
warm dependencies from MeshLLM's checked-in manifests, and allow container
runtimes to reuse cached layers where the runner host or K3s node retains them
instead of reinstalling host packages in every job.

If CI is missing a dependency, update the appropriate MeshLLM manifest and
lockfile or the runner image's YAML profile/installer, then publish the image
and pin its OCI digest. Do not add a one-off `apt-get`, `pip`, global `npm`,
`cargo install`, downloaded binary, or similar setup step to an individual
workflow. Existing workflow-local setup is migration debt, not a pattern for
new jobs.

### What triggers what

| Changed paths | `PR Quality Checks` | `PR Builds` CPU/artifact rows | Backend target rows |
| --- | --- | --- | --- |
| Runtime-facing Rust crates | ✅ fmt/clippy | ✅ Linux/macOS artifacts and Windows routing as needed | ⏭ skipped unless backend inputs changed |
| Rust tooling crates such as `tools/xtask/**` | ✅ fmt/clippy | ⏭ skipped unless another runtime/backend input changed | ⏭ skipped |
| `third_party/llama.cpp/**`, `crates/skippy-ffi/**`, backend build scripts, cache-version, backend-relevant `Justfile` hunks | ✅ fmt/clippy when Rust is affected | ✅ runs | ✅ CUDA/ROCm/Vulkan rows run where supported |
| Public website inputs (`website/**`, root install scripts, generated website paths) | ✅ website build canary | ⏭ skipped | ⏭ skipped |
| `crates/mesh-llm-ui/**` | ✅ React console UI quality | ✅ Linux/macOS UI artifact paths | ⏭ skipped |
| `**/*.md`, authored `docs/**`, anything docs-only | ✅ changes summary only | ⏭ skipped | ⏭ skipped |
| Manual `workflow_dispatch` | ✅ runs | ✅ runs | ✅ runs |

### Verifying path filtering works

To confirm builds are skipped on a docs-only change, open a PR and push a commit that touches only a `.md` file (e.g. add a blank line to `README.md`). All build jobs should appear as **Skipped** in the Actions tab — only the `changes` job runs.

To confirm UI-only changes skip backend jobs, push a commit touching only `crates/mesh-llm-ui/**`. UI quality and the CPU producer rows run, while Linux/Windows CUDA, ROCm, and Vulkan backend rows stay skipped.

To confirm public website changes stay separate from Rust artifacts, push a commit touching only `website/**` or public website passthrough inputs. `PR Quality Checks` should run the website build canary, while `PR Builds` should skip Linux/macOS inference artifacts and Windows backend builds unless the same PR also changes runtime/backend inputs.

### Adding new paths

If you add a new Rust crate, build script, or test directory, update `.github/actions/compute-changes`, `scripts/affected-crates.sh`, and the relevant `pr_*.yml` path filters so PR and main routing agree.

## GPU benchmark execution

GPU bandwidth benchmarks are launched through the `mesh-llm` binary itself rather than standalone benchmark executables. The public command remains:

```bash
mesh-llm gpus detect
```

Internally, mesh-llm runs a hidden `benchmark` subcommand in a narrow subprocess boundary so native backend hangs and stdout capture stay isolated from the main process.

Standard builds support benchmark execution only for the backends wired into the normal build flow:

- macOS Apple Silicon: Metal
- Linux / Windows NVIDIA: CUDA
- Linux / Windows AMD: HIP / ROCm

Intel GPU benchmark execution is not currently supported in standard `just build` flows, so runtime benchmark selection intentionally skips Intel GPUs.

## Protocol Backward Compatibility

Any change to `crates/mesh-llm-host-runtime/src/protocol/` or `crates/mesh-client/src/protocol/` requires backward-compatibility tests before merging.

Embedded clients (iOS, macOS, Android) are permanently supported. Protocol changes that break embedded client compatibility are breaking changes.

Run the protocol compatibility tests after any protocol change:

```bash
cargo test -p mesh-llm --test protocol_compat_v0_client
cargo test -p mesh-llm --test protocol_convert_matrix
```

See [`docs/design/EMBEDDED_CLIENT_ADR.md`](docs/design/EMBEDDED_CLIENT_ADR.md) for the full compatibility policy and rationale.
