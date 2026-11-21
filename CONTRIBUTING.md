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

## Build from source

Build the MeshLLM host and console:

```bash
just
```

This produces `target/debug/mesh-llm`. `just build` is the same thing, and
`just build-dev` is a faster build for local iteration.

The build does not produce an inference runtime. MeshLLM builds against the
Skippy revision pinned in `Cargo.toml` and loads a packaged Skippy runtime at
run time: one it downloads, or one you built from a Skippy checkout with
Skippy's `scripts/package-native-runtime.sh`.

For work that spans both repositories, point the build at a local Skippy
checkout, and go back to the pin when done:

```bash
scripts/skippy-source.sh ../skippy     # build against that checkout
scripts/skippy-source.sh               # show what the build uses
scripts/skippy-source.sh --pinned      # back to the pinned revision
```

Do not commit the `Cargo.lock` that a local override produces.

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
just stop             # stop mesh/rpc/llama processes
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
working copy until the files are re-extracted, and five host-runtime tests
still fail locally on content they read verbatim: both `config_schema`
snapshots, the `plugin::config` fixture, `inference::skippy::topology`, which
greps its own source, and `inference::skippy::split_certification`, whose
`build.rs` patch-queue digest is hashed from the bytes of the llama.cpp
patches and gates production split certification rather than tests alone.

Renormalise once, after committing or stashing anything in progress, since the
second command discards uncommitted work:

```powershell
git rm --cached -r .
git reset --hard HEAD
```

`git ls-files --eol` should then report `i/lf w/lf` for every text file. A
fresh clone needs none of this.

### Testing crates on native Windows

A bare Windows checkout cannot build the test targets of crates that pull in
`skippy-ffi`'s static link mode (`mesh-llm-system` does, through
`skippy-runtime-install`, which depends on `skippy-ffi` with
`default-features = false`), because `skippy-ffi/build.rs` then requires
the llama.cpp ABI archives to be prepared
(`automatic native preparation is not supported for Windows from build.rs yet`).
`just test-all` needs the same native preparation, through its Bash pipeline.

For crate suites that do not exercise the native runtime, enable
`dynamic-native-runtime`: feature unification turns on `skippy-ffi`'s
`dynamic-runtime`, whose build script returns early. The `mesh-llm-system`
gates then run on a machine without a prepared native build:

```powershell
cargo test --locked -p mesh-llm-system --lib --features dynamic-native-runtime
cargo clippy --no-deps -p mesh-llm-system --all-targets --features dynamic-native-runtime -- -D warnings
cargo fmt --check -p mesh-llm-system
```

`--no-deps` keeps Clippy scoped to the package you are changing. `cfg`-gated
code can be dead on one platform only, so if a platform-specific warning
appears that your diff does not touch, compare the run against the same
command on `main` before attributing it to your change. Running the
native-runtime suites still needs a prepared build (`LLAMA_STAGE_BUILD_DIR`
or `SKIPPY_LLAMA_BUILD_DIR` pointing at one), which this section does not cover.
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
| Test, per platform | `just test-rust --locked` |
| Console | `just test-ui` |
| Inference smoke (CPU) | `just smoke <runtime-dir> <model.gguf>` |

`just test-all` runs the first three. The smoke job builds Skippy's CPU runtime
from the pinned Skippy commit, builds MeshLLM, and requires a real completion.
It does not cover GPU runtimes, macOS, or Windows.

Run `just lint` after changing a shell script or a workflow; it includes
ShellCheck and the script tests under `scripts/tests/`.

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

Any change to `mesh/crates/mesh-llm-host-runtime/src/protocol/` or `mesh/crates/mesh-client/src/protocol/` requires backward-compatibility tests before merging.

Embedded clients (iOS, macOS, Android) are permanently supported. Protocol changes that break embedded client compatibility are breaking changes.

Run the protocol compatibility tests after any protocol change:

```bash
cargo test -p mesh-llm --test protocol_compat_v0_client
cargo test -p mesh-llm --test protocol_convert_matrix
```

See [`mesh/docs/design/EMBEDDED_CLIENT_ADR.md`](mesh/docs/design/EMBEDDED_CLIENT_ADR.md) for the full compatibility policy and rationale.
