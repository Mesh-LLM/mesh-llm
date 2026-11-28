# Contributing

Join the [#mesh-llm channel on the Goose Discord](https://discord.gg/goose-oss) for discussion and questions.

This file covers local build and development workflows for this repository.
See [MAINTAINERS.md](MAINTAINERS.md) for the Mesh maintainers.

## Prerequisites

- `just`
- `cmake`
- Rust toolchain (`cargo`)
- `sccache` (required by the repository Cargo configuration)
- Node.js 24 + pnpm 10 or newer (for UI development). The UI lockfile keeps
  `overrides` in `pnpm-workspace.yaml`, which pnpm 9 does not read, so pnpm 9
  cannot install it. `corepack pnpm@10` is enough if your host pnpm is older.

Install the pinned compiler cache and the platform linkers with:

```bash
just bootstrap-build-tools
```

The bootstrap installs sccache 0.16.0. On Linux it also installs mold and lld
with the detected system package manager; on macOS it installs lld with
Homebrew. On Windows it installs the Rust LLVM tools with rustup. Set
`MESH_LLM_SCCACHE_VERSION` only when deliberately testing a newer cache binary.

**macOS**: Apple Silicon. Metal is used automatically. Install the accelerated
linker with `brew install lld`; the repository probes `ld64.lld` against the
active SDK and uses Apple ld when that installed version is incompatible.

**Linux**: install `mold` and `lld`. Cargo uses mold by default and retains lld
as the diagnosed compatibility fallback. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y mold lld
```

GPU toolchains (CUDA, ROCm, Vulkan) are not needed to build MeshLLM. The
inference runtime is built and released by
[Skippy](https://github.com/Mesh-LLM/skippy); MeshLLM only loads it.

**Windows**: install the Rust lld tools with
`rustup component add llvm-tools-preview`, or install LLVM with `winget install
LLVM.LLVM`.

**Windows ARM64**: `scripts\build-windows.ps1 -TargetArch arm64` builds the host
for `aarch64-pc-windows-msvc`, cross-compiling from an x64 machine or natively
on Windows on ARM, which is also the default architecture there. It needs
`rustup target add aarch64-pc-windows-msvc` and two Visual Studio components:
the ARM64 MSVC tools (`Microsoft.VisualStudio.Component.VC.Tools.ARM64`) and the
C++ Clang compiler (`Microsoft.VisualStudio.Component.VC.Llvm.Clang`), which
builds the `ring` crate for ARM64. The host lands in
`target\aarch64-pc-windows-msvc\debug`. It runs a Skippy CPU runtime for
Windows ARM64, which Skippy does not publish yet.

## Build from source

Build the Skippy runtime, then the MeshLLM host and console:

```bash
just
```

This produces `target/debug/mesh-llm` and `target/debug/native-runtimes`. `just build` is the same thing, and
`just build-dev` is a faster build for local iteration.

`just build` also gets the inference runtime. It downloads the native runtime
that the Skippy release pinned in `Cargo.toml` published for this machine,
checks it against the release's checksums, and puts it in
`target/debug/native-runtimes`, where the host finds it with no settings.
It builds the runtime from source instead (the commit in `scripts/skippy-pin.sh`, fetched into
`.deps/skippy`) when the release has no package for this machine and backend
(including Vulkan), when you select another Skippy source (below), or when
you ask: `MESH_LLM_RUNTIME_FROM_SOURCE=1 just build`. The backend is detected (Metal on
macOS; CUDA, ROCm or Vulkan on Linux and Windows when their tools are installed; otherwise
CPU), or name it: `just build backend=cuda cuda_arch='120;86'`. Naming
`cuda_arch` or `rocm_arch` builds from source. A runtime
already there for the same pin and backend is reused. `just mesh` rebuilds the
host alone, and `just runtime` the runtime alone. On Windows the runtime step
needs Git Bash.

MeshLLM itself never links llama.cpp: it loads that packaged runtime at run
time, or one it downloads from a Skippy release.

For work that spans both repositories, point the build at a local Skippy
checkout, and go back to the pin when done:

```bash
scripts/skippy-source.sh ../skippy     # build against that checkout
scripts/skippy-source.sh --ref main    # or fetch a branch, tag, commit or pull/<number>
scripts/skippy-source.sh --ref my-branch https://github.com/me/skippy   # from a fork
scripts/skippy-source.sh               # show what the build uses
scripts/skippy-source.sh --pinned      # back to the pinned crates.io release
```

Do not commit the `Cargo.toml` or `Cargo.lock` changes that a local override produces.

Run real inference through the host you built, on a runtime directory and a
GGUF model you supply. The script uses a throwaway home and cache:

```bash
just smoke <dir-of-packaged-skippy-runtimes> <model.gguf>
```

Build the release host and package it with a runtime:

```bash
just release-build
MESH_LLM_NATIVE_RUNTIME_DIR=<unpacked-skippy-runtime> just release-bundle v0.X.0 dist
```

[RELEASE.md](RELEASE.md) covers signing and the release workflow.

## UI development workflow

The React console and embedded asset crate live in `mesh/crates/mesh-llm-ui/`.
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
just stop             # stop mesh-llm processes
just test             # quick test against :9337
just lint             # format, Clippy, and shell checks
just test-rust        # Rust tests for the whole workspace
just test-ui          # console lint, type check, unit tests, and build
just test-all         # everything a pull request must pass, except the smoke
just --list           # list all recipes
```

### Line endings on native Windows

`.gitattributes` checks every text file out with LF on every platform. A
checkout created before that landed, with `core.autocrlf=true`, keeps its CRLF
working copy until the files are re-extracted, and four host-runtime tests
still fail locally on content they read verbatim: both `config_schema`
snapshots, the `plugin::config` fixture, and `inference::skippy::topology`,
which greps its own source.

Renormalise once, after committing or stashing anything in progress, since the
second command discards uncommitted work:

```powershell
git rm --cached -r .
git reset --hard HEAD
```

`git ls-files --eol` should then report `i/lf w/lf` for every text file. A
fresh clone needs none of this.

### Testing crates on native Windows

Nothing native is compiled on any platform, so `just test-rust` and
single-crate commands work on a bare Windows checkout. MeshLLM always selects
`skippy-ffi`'s `dynamic-runtime` mode (see the Skippy entries in the root
`Cargo.toml`); it never links llama.cpp.

The `mesh-llm-system` gates, with the device code enabled:

```powershell
cargo test --locked -p mesh-llm-system --lib --features dynamic-native-runtime
cargo clippy --no-deps -p mesh-llm-system --all-targets --features dynamic-native-runtime -- -D warnings
cargo fmt --check -p mesh-llm-system
```

`--no-deps` keeps Clippy scoped to the package you are changing. `cfg`-gated
code can be dead on one platform only, so if a platform-specific warning
appears that your diff does not touch, compare the run against the same
command on `main` before attributing it to your change. The tests that call
into a native runtime need a packaged Skippy runtime and run through
`just test-with-runtime <runtime-dir>`, which is not available on Windows.
The Rust MSVC toolchain needs the Visual Studio Build Tools with the
"Desktop development with C++" workload installed.

## Commit messages

Use clear, user-focused commit subjects and PR titles. Conventional Commit
prefixes are optional release-note hints, not a hook or CI requirement.

When present, `feat:` groups an entry under Added, `fix:` under Fixed,
`perf:` under Changed, and `security:` under Security. Other titles remain
in Other changes. `Release-Notes: <Section>` can override classification.
The repository squash-merges, so the PR title becomes the subject on main.

Builds do not install Git hooks, and CI does not police titles or attribution
identities. Existing clones pointing `core.hooksPath` at `scripts/hooks` need
no migration: the retired commit-msg hook is no longer present. Custom hooks
and repository merge settings are unchanged.

See [the release-notes skill](.agents/skills/release-notes/SKILL.md) for
classification details.

## CI / GitHub Actions

Pull requests run [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

| Job | Local command |
|---|---|
| Lint | `just lint` |
| Test, per platform | `just test-rust --locked --no-fail-fast` |
| Console | `just test-ui` |
| Inference smoke (CPU) | `just smoke <runtime-dir> <model.gguf>`, then `just test-with-runtime <runtime-dir>` |

`just test-all` runs the first three. The smoke job downloads the CPU runtime
from the pinned Skippy release, builds MeshLLM, and requires a real completion.
With `Skippy-Ref` it builds the runtime and MeshLLM against that source instead.
It does not cover GPU runtimes, macOS, or Windows.

Run `just lint` after changing a shell script or a workflow; it includes
ShellCheck and the script tests under `scripts/tests/`.

## GPU benchmark execution

GPU bandwidth benchmarks are launched through the `mesh-llm` binary itself rather than standalone benchmark executables. The public command remains:

```bash
mesh-llm gpus detect
```

Internally, mesh-llm runs a hidden `benchmark` subcommand in a narrow subprocess boundary so native backend hangs and stdout capture stay isolated from the main process.

The benchmark runs through the installed Skippy native runtime, so it is available only for backends that have one:

- macOS Apple Silicon: Metal
- Linux / Windows NVIDIA: CUDA
- Linux / Windows AMD: HIP / ROCm

Intel GPUs have no published native-runtime benchmark tool, so runtime benchmark selection intentionally skips them.

## Protocol Backward Compatibility

Any change to `mesh/crates/mesh-llm-host-runtime/src/protocol/` or `mesh/crates/mesh-client/src/protocol/` requires backward-compatibility tests before merging.

Embedded clients (iOS, macOS, Android) are permanently supported. Protocol changes that break embedded client compatibility are breaking changes.

Run the protocol compatibility tests after any protocol change:

```bash
cargo test -p mesh-llm --test protocol_compat_v0_client
cargo test -p mesh-llm --test protocol_convert_matrix
```

See [`mesh/docs/design/EMBEDDED_CLIENT_ADR.md`](mesh/docs/design/EMBEDDED_CLIENT_ADR.md) for the full compatibility policy and rationale.
