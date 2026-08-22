# Distributed LLM Inference — build & run tasks

llama_dir := env("MESH_LLM_LLAMA_DIR", ".deps/llama.cpp")
llama_build_root := env("MESH_LLM_LLAMA_BUILD_ROOT", ".deps/llama-build")
mesh_dir := "crates/mesh-llm"
ui_dir := "crates/mesh-llm-ui"
website_dir := "website"
home_dir := if os_family() == "windows" { env("USERPROFILE") } else { env("HOME") }
xdg_cache_dir := env("XDG_CACHE_HOME", home_dir / ".cache")
hf_home := env("HF_HOME", xdg_cache_dir / "huggingface")
models_dir := env("HF_HUB_CACHE", hf_home / "hub")
model := models_dir / "GLM-4.7-Flash-Q4_K_M.gguf"

# Build for the current platform.
default: build

# Build the local product, then exercise the embedded console against a real
# isolated mesh-llm process. The harness owns and verifies its cleanup.
[unix]
qa-logging-console-e2e:
    @just build
    @scripts/qa-logging-console-e2e.sh --current-binary ./target/debug/mesh-llm

[private]
[unix]
with-lld *COMMAND:
    #!/usr/bin/env bash
    set -euo pipefail
    lld=""
    case "$(uname -s)" in
        Linux)
            if ! command -v ld.lld >/dev/null 2>&1; then
                cat >&2 <<'EOF'
    Error: LLVM ld.lld was not found.

    lld is required for faster Rust builds (measured up to 26% faster locally).

    Install lld, then rerun the just command. Common Linux packages:
      Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y lld
      Fedora:        sudo dnf install lld
      Arch Linux:    sudo pacman -S lld
      openSUSE:      sudo zypper install lld

    The build requires ld.lld to be available on PATH.
    EOF
                exit 1
            fi
            lld="lld"
            ;;
        Darwin)
            if command -v ld64.lld >/dev/null 2>&1; then
                lld="$(command -v ld64.lld)"
            elif command -v brew >/dev/null 2>&1; then
                lld_prefix="$(brew --prefix lld 2>/dev/null || true)"
                if [[ -n "$lld_prefix" && -x "$lld_prefix/bin/ld64.lld" ]]; then
                    lld="$lld_prefix/bin/ld64.lld"
                fi
            fi
            if [[ -z "$lld" ]]; then
                for candidate in /opt/homebrew/opt/lld/bin/ld64.lld /usr/local/opt/lld/bin/ld64.lld; do
                    if [[ -x "$candidate" ]]; then
                        lld="$candidate"
                        break
                    fi
                done
            fi
            if [[ -z "$lld" ]]; then
                cat >&2 <<'EOF'
    Error: LLVM ld64.lld was not found.

    lld is required for faster Rust builds (measured up to 26% faster locally).

    Install lld, then rerun the just command:
      brew install lld

    If Homebrew installed lld but it is not on PATH, Mesh-LLM also checks:
      $(brew --prefix lld)/bin/ld64.lld
      /opt/homebrew/opt/lld/bin/ld64.lld
      /usr/local/opt/lld/bin/ld64.lld
    EOF
                exit 1
            fi
            ;;
        *)
            echo "Unsupported OS for lld linker setup: $(uname -s)" >&2
            exit 1
            ;;
    esac
    export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C link-arg=-fuse-ld=$lld"
    exec {{ COMMAND }}

[private]
[windows]
with-lld *COMMAND:
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "$$ErrorActionPreference = 'Stop'; $$linker = $$null; try { $$sysroot = (& rustc --print sysroot).Trim(); foreach ($$target in @('x86_64-pc-windows-msvc', 'aarch64-pc-windows-msvc')) { $$candidate = Join-Path $$sysroot \"lib\rustlib\$$target\bin\rust-lld.exe\"; if (Test-Path $$candidate) { $$linker = $$candidate; break } } } catch {}; if (-not $$linker) { foreach ($$name in @('rust-lld.exe', 'lld-link.exe')) { $$command = Get-Command $$name -ErrorAction SilentlyContinue; if ($$command) { $$linker = $$command.Source; break } } }; if (-not $$linker) { Write-Error \"LLVM lld was not found for the Windows MSVC target.`n`nlld is required for faster Rust builds (measured up to 26% faster locally).`n`nInstall one of these, then rerun the just command:`n  rustup component add llvm-tools-preview`n`nOr install LLVM lld-link:`n  winget install LLVM.LLVM`n  choco install llvm`n`nThe build requires lld. It looks for rust-lld.exe in the active Rust sysroot first, then falls back to rust-lld.exe or lld-link.exe on PATH.\"; exit 1 }; $$env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER = $$linker; $$env:CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_LINKER = $$linker; Invoke-Expression '{{ COMMAND }}'"


# Build a local product for the current platform. This is always a
# backend-neutral dynamic host plus an adjacent packaged runtime.
[macos]
build backend="" cuda_arch="" rocm_arch="":
    @python3 scripts/manage-build-cache.py build -- scripts/build-development-product.sh --backend "{{ backend }}" --cuda-arch "{{ cuda_arch }}" --rocm-arch "{{ rocm_arch }}"

# Fast local iteration build: dynamic host + adjacent native runtime + UI.
[macos]
build-dev:
    @python3 scripts/manage-build-cache.py build -- env MESH_LLM_BUILD_PROFILE=dev scripts/build-development-product.sh --profile dev

# Linux overrides:
#   just build backend=cpu
#   just build backend=cuda cuda_arch='120;86'
#   just build backend=rocm rocm_arch='gfx942;gfx90a'
# just build backend=vulkan
[linux]
build backend="" cuda_arch="" rocm_arch="":
    @python3 scripts/manage-build-cache.py build -- scripts/build-development-product.sh --backend "{{ backend }}" --cuda-arch "{{ cuda_arch }}" --rocm-arch "{{ rocm_arch }}"

# Fast local iteration build: dynamic host + adjacent native runtime + UI.
[linux]
build-dev backend="" cuda_arch="" rocm_arch="":
    @python3 scripts/manage-build-cache.py build -- env MESH_LLM_BUILD_PROFILE=dev scripts/build-development-product.sh --profile dev --backend "{{ backend }}" --cuda-arch "{{ cuda_arch }}" --rocm-arch "{{ rocm_arch }}"

# Windows overrides:
#   just build backend=cpu
#   just build backend=cuda cuda_arch='120;86'
#   just build backend=rocm rocm_arch='gfx942;gfx90a'
# just build backend=vulkan
[windows]
build backend="" cuda_arch="" rocm_arch="":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend "{{ backend }}" -CudaArch "{{ cuda_arch }}" -RocmArch "{{ rocm_arch }}" -DynamicHost

# Fast local iteration build: dynamic host + adjacent native runtime + UI.
[windows]
build-dev backend="" cuda_arch="" rocm_arch="":
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:MESH_LLM_BUILD_PROFILE='dev'; & './scripts/build-windows.ps1' -Backend '{{ backend }}' -CudaArch '{{ cuda_arch }}' -RocmArch '{{ rocm_arch }}' -DynamicHost"

# Low-level static ABI primitive. This builds only the native runtime; it never
# builds a MeshLLM host. Use `just build` for normal development.
build-mac:
    @scripts/package-native-runtime.sh --build --backend metal

# Low-level static ABI primitive. This builds only the native runtime; it never
# builds a MeshLLM host. Use `just build` for normal development.
build-linux backend="" cuda_arch="" rocm_arch="":
    #!/usr/bin/env bash
    set -euo pipefail
    backend="{{ backend }}"
    [[ -n "$backend" ]] || backend=cpu
    LLAMA_STAGE_CUDA_ARCHITECTURES="{{ cuda_arch }}" LLAMA_STAGE_AMDGPU_TARGETS="{{ rocm_arch }}" scripts/package-native-runtime.sh --build --backend "$backend"

# Backward-compatible spelling for the explicit native-runtime primitive.
[linux]
build-runtime backend="" cuda_arch="" rocm_arch="":
    #!/usr/bin/env bash
    set -euo pipefail
    backend="{{ backend }}"
    [[ -n "$backend" ]] || backend=cpu
    LLAMA_STAGE_CUDA_ARCHITECTURES="{{ cuda_arch }}" \
      LLAMA_STAGE_AMDGPU_TARGETS="{{ rocm_arch }}" \
      scripts/package-native-runtime.sh --build --backend "$backend"

# Build release artifacts for the current platform.

# Prepare, publish, and watch a GitHub release from main.
release version *ARGS:
    @scripts/release.sh "{{ version }}" {{ ARGS }}

# Build the backend-neutral release host once for the current platform.
release-host-build:
    @scripts/build-release.sh

# Build one packageable native runtime for the current platform.
release-runtime-build backend="" target="":
    #!/usr/bin/env bash
    set -euo pipefail
    selected_backend="{{ backend }}"
    if [[ -z "$selected_backend" ]]; then
        if [[ "$(uname -s)" == Darwin ]]; then selected_backend=metal; else selected_backend=cpu; fi
    fi
    if [[ -n "{{ target }}" ]]; then
        scripts/package-native-runtime.sh --build --backend "$selected_backend" --target "{{ target }}"
    else
        scripts/package-native-runtime.sh --build --backend "$selected_backend"
    fi

# Build the backend-neutral host and the default runtime for this platform.
release-build: release-host-build release-runtime-build

# Build a Linux aarch64 CPU release artifact on a native aarch64 runner.
release-build-aarch64: release-host-build
    @scripts/package-native-runtime.sh --build --backend cpu --target aarch64-unknown-linux-gnu

# Build a Linux aarch64 CUDA release artifact (Jetson/Orin).
# SM arches selected by MESH_CUDA_VERSION env (set by CI matrix); a bare
# `just` invocation with the var unset detects the installed toolkit instead
# of assuming one (see scripts/detect-cuda-toolkit-version.sh). Thor (sm_110)
# needs toolkit >= 13 -- gate on the major, the same shape as the x86_64
# recipe's Blackwell gate, so a future CUDA 14 keeps Thor instead of falling
# back to the pre-13 list the way a `13.*` glob did.
release-build-aarch64-cuda: release-host-build
    #!/usr/bin/env bash
    set -euo pipefail
    cuda_version="${MESH_CUDA_VERSION:-$(scripts/detect-cuda-toolkit-version.sh)}"
    major="${cuda_version%%.*}"
    if [[ "$major" -ge 13 ]]; then
      arches='75;80;86;87;89;90;110'
    else
      arches='61;75;80;86;87;89;90'
    fi
    MESH_LLM_CUDA_TOOLKIT_MAJOR="${MESH_LLM_CUDA_TOOLKIT_MAJOR:-$major}" \
      LLAMA_STAGE_CUDA_ARCHITECTURES="$arches" \
      scripts/package-native-runtime.sh --build --backend cuda --target aarch64-unknown-linux-gnu

# Prepare the pinned llama.cpp checkout and apply the Mesh-LLM ABI patch queue.
llama-prepare:
    @scripts/prepare-llama.sh pinned

# Prepare llama.cpp at upstream master and apply the Mesh-LLM ABI patch queue.
llama-prepare-latest:
    @scripts/prepare-llama.sh latest

# Build the patched llama.cpp ABI static libraries.
llama-build: llama-prepare
    @scripts/build-llama.sh

release-build-windows:
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend cpu -BuildProfile release -DynamicHost

release-host-build-windows:
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -BuildProfile release -HostOnly

# Build a Linux CUDA release artifact.
# SM arches selected by MESH_CUDA_VERSION env (set by CI matrix); a bare
# `just` invocation with the var unset detects the installed toolkit instead
# of assuming one (see scripts/detect-cuda-toolkit-version.sh). Blackwell
# (sm_100/103/120/121) needs toolkit >= 12.8, the first release that shipped
# support for it -- gate on that boundary, not on the CUDA major alone.
release-build-cuda: release-host-build
    #!/usr/bin/env bash
    set -euo pipefail
    cuda_version="${MESH_CUDA_VERSION:-$(scripts/detect-cuda-toolkit-version.sh)}"
    major="${cuda_version%%.*}"
    minor="${cuda_version#*.}"
    [[ "$minor" == "$cuda_version" ]] && minor=0
    minor="${minor%%.*}"
    if [[ "$major" -gt 12 || ("$major" -eq 12 && "$minor" -ge 8) ]]; then
      arches='75;80;86;87;89;90;100;103;120;121'
    else
      arches='61;75;80;86;87;89;90'
    fi
    MESH_LLM_CUDA_TOOLKIT_MAJOR="${MESH_LLM_CUDA_TOOLKIT_MAJOR:-$major}" \
      LLAMA_STAGE_CUDA_ARCHITECTURES="$arches" \
      scripts/package-native-runtime.sh --build --backend cuda --target x86_64-unknown-linux-gnu

release-build-cuda-windows cuda_arch="61;75;80;86;87;89;90":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend cuda -CudaArch "{{ cuda_arch }}" -BuildProfile release -DynamicHost

# Build a Linux ROCm ABI release artifact with an explicit architecture list.
release-build-rocm rocm_arch="gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1103;gfx1151;gfx1200;gfx1201": release-host-build
    @LLAMA_STAGE_AMDGPU_TARGETS="{{ rocm_arch }}" \
      scripts/package-native-runtime.sh --build --backend rocm --target x86_64-unknown-linux-gnu

release-build-rocm-windows rocm_arch="gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1103;gfx1151;gfx1200;gfx1201":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend rocm -RocmArch "{{ rocm_arch }}" -BuildProfile release -DynamicHost

# Build a Linux Vulkan ABI release artifact.
release-build-vulkan: release-host-build
    @scripts/package-native-runtime.sh --build --backend vulkan --target x86_64-unknown-linux-gnu

release-build-vulkan-windows:
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend vulkan -BuildProfile release -DynamicHost

# Build the skippy benchmark/debug telemetry collector.
[unix]
metrics-server-build:
    just with-lld cargo build -p metrics-server

[windows]
metrics-server-build:
    @just with-lld cargo build -p metrics-server

# Build the binaries copied into the Skippy WAN Docker lab image.
[linux]
skippy-wan-lab-build-bins:
    cargo build --release --locked -p skippy-server -p skippy-prompt -p metrics-server -p skippy-model-package

# Build the resumable GGUF conversion/quantization replacement CLI.
[unix]
skippy-quantize-build:
    just with-lld cargo build -p skippy-quantize

[windows]
skippy-quantize-build:
    @just with-lld cargo build -p skippy-quantize

# Build the release binary used in HF conversion/quantization job images.
[unix]
skippy-quantize-release-build:
    just with-lld cargo build --release --locked -p skippy-quantize

[windows]
skippy-quantize-release-build:
    @just with-lld cargo build --release --locked -p skippy-quantize

# Build skippy-quantize as a standalone quantization binary with the pinned
# llama.cpp quantization ABI linked into the executable.
[unix]
skippy-quantize-standalone-build backend="cpu":
    scripts/prepare-llama.sh pinned
    LLAMA_STAGE_BACKEND="{{ backend }}" LLAMA_STAGE_LINK_MODE=static scripts/build-llama.sh
    LLAMA_STAGE_BACKEND="{{ backend }}" LLAMA_STAGE_LINK_MODE=static just with-lld cargo build -p skippy-quantize --no-default-features

[unix]
skippy-quantize-standalone-release-build backend="cpu":
    scripts/prepare-llama.sh pinned
    LLAMA_STAGE_BACKEND="{{ backend }}" LLAMA_STAGE_LINK_MODE=static scripts/build-llama.sh
    LLAMA_STAGE_BACKEND="{{ backend }}" LLAMA_STAGE_LINK_MODE=static just with-lld cargo build --release --locked -p skippy-quantize --no-default-features

# Generate a reproducible benchmark corpus for skippy bench tooling.
bench-corpus tier="smoke" *ARGS="":
    scripts/generate-bench-corpus.py "{{ tier }}" {{ ARGS }}

# Run skippy family certification checks.
family-certify *ARGS:
    just with-lld scripts/family-certify.sh {{ ARGS }}

# Run target/draft speculative compatibility checks.
spec-bench target draft *ARGS:
    just with-lld env LLAMA_STAGE_BUILD_DIR=".deps/llama-build/build-stage-abi-static" cargo build -p llama-spec-bench
    LLAMA_STAGE_BUILD_DIR=".deps/llama-build/build-stage-abi-static" target/debug/llama-spec-bench --target-model-path "{{ target }}" --draft-model-path "{{ draft }}" {{ ARGS }}

# Smoke a standalone skippy OpenAI frontend stage.
skippy-openai-smoke *ARGS:
    just with-lld scripts/skippy-openai-smoke.sh {{ ARGS }}

# Run the skippy benchmark/debug telemetry collector.
metrics-server db="/tmp/mesh-metrics.duckdb" http_addr="127.0.0.1:18080" otlp_addr="127.0.0.1:14317" *ARGS="": metrics-server-build
    target/debug/metrics-server serve --db "{{ db }}" --http-addr "{{ http_addr }}" --otlp-grpc-addr "{{ otlp_addr }}" {{ ARGS }}

# Download the default model (GLM-4.7-Flash Q4_K_M, 17GB)
download-model:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{ models_dir }}"
    if [ -f "{{ model }}" ]; then
        echo "Model already exists: {{ model }}"
    else
        echo "Downloading GLM-4.7-Flash Q4_K_M (~17GB)..."
        curl -L -o "{{ model }}" \
            "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf"
    fi

# ── QUIC Mesh ──────────────────────────────────────────────────

mesh_bin := env("MESH_LLM_BIN", "target/release/mesh-llm")

# Prints an invite token for other nodes to join.
mesh-worker gguf=model:
    "{{ mesh_bin }}" --model {{ gguf }}

# Join an existing mesh and serve through the embedded runtime.
mesh-join join="" port="9337" gguf=model split="":
    #!/usr/bin/env bash
    set -euo pipefail
    ARGS="--model {{ gguf }} --port {{ port }}"
    if [ -n "{{ join }}" ]; then
        ARGS="$ARGS --join {{ join }}"
    fi
    if [ -n "{{ split }}" ]; then
        ARGS="$ARGS --tensor-split {{ split }}"
    fi
    exec "{{ mesh_bin }}" $ARGS

# Create a portable product tarball containing the neutral host and default runtime.
bundle output="/tmp/mesh-llm-bundle.tar.gz": release-build
    #!/usr/bin/env bash
    set -euo pipefail
    staging_dir="$(mktemp -d)"
    trap 'rm -rf "$staging_dir"' EXIT
    version="v$("{{ mesh_bin }}" --version | awk '{print $NF}')"
    scripts/package-release.sh "$version" "$staging_dir"
    stable_archive=""
    for candidate in "$staging_dir"/mesh-llm-*.tar.gz; do
        [[ -f "$candidate" ]] || continue
        case "$(basename "$candidate")" in
            mesh-llm-"$version"-*) continue ;;
        esac
        if [[ -n "$stable_archive" ]]; then
            echo "multiple stable product archives were produced" >&2
            exit 1
        fi
        stable_archive="$candidate"
    done
    if [[ -z "$stable_archive" ]]; then
        echo "product packaging did not produce a stable archive" >&2
        exit 1
    fi
    mkdir -p "$(dirname "{{ output }}")"
    cp "$stable_archive" "{{ output }}"
    cp "$stable_archive.sha256" "{{ output }}.sha256"
    echo "Bundle: {{ output }} ($(du -sh "{{ output }}" | cut -f1))"

# Create release archive(s) for the current platform.

# `version` should be a tag like v0.30.0.
release-bundle version output="dist":
    @scripts/package-release.sh "{{ version }}" "{{ output }}"

# Create a Linux aarch64 CPU release archive on a native aarch64 runner.
release-bundle-aarch64 version output="dist":
    @scripts/package-release.sh "{{ version }}" "{{ output }}"

# Create a Linux aarch64 CUDA release archive on a native aarch64 runner.
release-bundle-aarch64-cuda version output="dist":
    MESH_RELEASE_ARCH=aarch64 MESH_RELEASE_FLAVOR=cuda scripts/package-release.sh "{{ version }}" "{{ output }}"

# Run repo-level release-target consistency checks.
[unix]
check-release:
    just with-lld cargo run -p xtask -- repo-consistency release-targets

[windows]
check-release:
    @just with-lld cargo run -p xtask -- repo-consistency release-targets

# Check the Rust process-environment mutation census and contract annotations.
check-env-mutation-contract:
    @python3 scripts/check-env-mutation-contract.py

# Run one release-attestation xtask action through the repository toolchain.
release-attestation *ARGS:
    @just with-lld cargo run -q -p xtask -- release-attestation {{ ARGS }}

release-bundle-windows version output="dist":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package-release.ps1 -Version "{{ version }}" -OutputDir "{{ output }}"

# Create Linux CUDA release archive(s).
release-bundle-cuda version output="dist":
    MESH_RELEASE_FLAVOR=cuda scripts/package-release.sh "{{ version }}" "{{ output }}"

release-bundle-cuda-windows version output="dist":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package-release.ps1 -Version "{{ version }}" -OutputDir "{{ output }}" -Flavor cuda

# Create Linux ROCm release archive(s).
release-bundle-rocm version output="dist":
    MESH_RELEASE_FLAVOR=rocm scripts/package-release.sh "{{ version }}" "{{ output }}"

release-bundle-rocm-windows version output="dist":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package-release.ps1 -Version "{{ version }}" -OutputDir "{{ output }}" -Flavor rocm

# Create Linux Vulkan release archive(s).
release-bundle-vulkan version output="dist":
    MESH_RELEASE_FLAVOR=vulkan scripts/package-release.sh "{{ version }}" "{{ output }}"

release-bundle-vulkan-windows version output="dist":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package-release.ps1 -Version "{{ version }}" -OutputDir "{{ output }}" -Flavor vulkan

# Run the UI dev server with Vite HMR, proxying /api to mesh-llm (default: http://127.0.0.1:3131)
ui-dev api="http://127.0.0.1:3131" port="5173":
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{ ui_dir }}"
    MESH_UI_API_ORIGIN="{{ api }}" VITE_API_URL="{{ api }}" pnpm run dev -- --host 0.0.0.0 --port {{ port }}

# Run the UI dev server proxying to the public meshllm.cloud API
ui-dev-public: (ui-dev "https://public.meshllm.cloud")

# Build the public website into docs/ for static hosting.
website-build:
    cd "{{ website_dir }}" && npm run clean && npm run build
    just crate-docs

# Build Rustdoc for the crates published by the release workflow.
crate-docs:
    scripts/build-crate-docs.sh

# Run the public website dev server on port 8765.
website-dev:
    cd "{{ website_dir }}" && npm run dev

# Remove generated public website output while preserving docs/ source markdown.
website-clean:
    cd "{{ website_dir }}" && npm run clean

# Run UI unit tests (vitest)
ui-test:
    cd "{{ ui_dir }}" && pnpm test

# ── Full Validation Gate ───────────────────────────────────────

# Populate the trusted Linux compiler-object seed with the dominant host graph.
[linux]
ci-sccache-seed-build:
    cargo clippy --locked -p mesh-llm --all-targets -- -D warnings

# Validate CI definitions, planner fixtures, and repository consistency.
ci-validate:
    actionlint -config-file .github/actionlint.yaml
    git diff --check
    python3 -m unittest discover -s scripts/tests -p 'test_*.py'
    just ci-crate-lists
    just check-release
    just no-console-print
    just publish-crates

# Run CI/workspace crate-list consistency checks.
ci-crate-lists:
    just with-lld cargo run -p xtask -- repo-consistency ci-crate-lists

# Run crates.io publish-chain consistency checks.
publish-crates:
    just with-lld cargo run -p xtask -- repo-consistency publish-crates

# Ratchet on println!/eprintln!/print!/eprint! in Rust files under crates/
# (including crate test targets; build.rs is skipped): every occurrence must be
# explicitly listed in tools/xtask/data/console_print_allowlist.json.
no-console-print:
    cargo run -p xtask -- repo-consistency no-console-print

# Shellcheck the explicitly supplied changed shell scripts.
ci-shellcheck *scripts:
    shellcheck {{ scripts }}

# Run all checks: repo consistency, Rust tests, author exemplars, fmt, clippy, UI/docs builds, and E2E smoke.
test-all:
    #!/usr/bin/env bash
    set -euo pipefail

    echo "=== 0/11 Test-all rust crate coverage preflight ==="
    just with-lld cargo run -p xtask -- repo-consistency test-all-rust-crate-coverage

    # A full workspace gate otherwise leaves hundreds of incompatible
    # incremental feature graphs behind. CI already builds non-incrementally;
    # use the same bounded-disk behavior here unless explicitly overridden.
    export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
    dynamic_target_dir="${MESH_TEST_ALL_DYNAMIC_TARGET_DIR:-$PWD/target/test-all-dynamic}"

    native_backend="${LLAMA_STAGE_BACKEND:-${SKIPPY_LLAMA_BACKEND:-${LLAMA_BACKEND:-}}}"
    if [[ -z "$native_backend" ]]; then
        case "$(uname -s)" in
            Darwin) native_backend="metal" ;;
            *) native_backend="cpu" ;;
        esac
    fi
    export LLAMA_STAGE_BACKEND="$native_backend"

    if [[ -z "${LLAMA_STAGE_BUILD_DIR:-}" ]]; then
        LLAMA_STAGE_BUILD_DIR="$(scripts/build-llama.sh --print-build-dir)"
        export LLAMA_STAGE_BUILD_DIR
    fi

    echo "=== Native llama.cpp ABI ($LLAMA_STAGE_BACKEND) ==="
    echo "Build dir: $LLAMA_STAGE_BUILD_DIR"
    scripts/prepare-llama.sh
    scripts/build-llama.sh
    echo ""

    # Each UI step runs in a subshell so cd doesn't leak between steps.
    echo "=== 1/11 Repo consistency ==="
    just with-lld cargo run -p xtask -- repo-consistency ci-crate-lists
    just with-lld cargo run -p xtask -- repo-consistency publish-crates
    echo ""
    echo "=== 2/11 Rust format check ==="
    just with-lld cargo fmt --all -- --check
    echo ""
    echo "=== 3/11 GPU bench crate check ==="
    just with-lld cargo check -p mesh-llm-gpu-bench
    echo ""
    echo "=== 4-5/11 Rust validation ==="
    # Keep Clippy and tests adjacent for each compatible feature graph. Switching
    # dynamic -> static -> dynamic -> static forces Cargo to relink both graphs.
    echo "--- Dynamic-runtime bindings: Clippy ---"
    CARGO_TARGET_DIR="$dynamic_target_dir" just with-lld cargo clippy \
        -p mesh-llm-ffi \
        -p mesh-llm-nodejs \
        -p skippy-ffi \
        -p skippy-quantize \
        --all-targets -- -D warnings
    echo "--- Dynamic-runtime bindings: tests ---"
    CARGO_TARGET_DIR="$dynamic_target_dir" just with-lld cargo test \
        -p mesh-llm-ffi \
        -p mesh-llm-nodejs \
        -p skippy-ffi \
        -p skippy-quantize
    echo "--- Static development workspace: Clippy ---"
    just with-lld cargo clippy --workspace --all-targets \
        --exclude mesh-llm-ffi \
        --exclude mesh-llm-nodejs \
        --exclude skippy-ffi \
        --exclude skippy-quantize \
        -- -D warnings
    echo "--- Static development workspace: tests ---"
    just with-lld cargo test --workspace \
        --exclude mesh-llm-ffi \
        --exclude mesh-llm-nodejs \
        --exclude skippy-ffi \
        --exclude skippy-quantize \
        --exclude skippy-runtime \
        --exclude skippy-server
    echo "--- Static Skippy runtime tests ---"
    just with-lld cargo test --package skippy-runtime --no-default-features --lib
    echo "--- Static Skippy server tests ---"
    just with-lld cargo test --package skippy-server --no-default-features --lib
    echo ""
    echo "=== 6/11 Plugin author exemplar ==="
    just with-lld cargo run --quiet --manifest-path docs/plugins/exemplars/web-ui/Cargo.toml -- --print-package-manifest > target/web-ui-exemplar-manifest.json
    diff -u <(jq -S . docs/plugins/exemplars/web-ui/plugin.package.json) <(jq -S . target/web-ui-exemplar-manifest.json)
    node --check docs/plugins/exemplars/web-ui/bundle/register-mesh-plugin-ui.js
    (cd "{{ ui_dir }}" && pnpm exec tsc --ignoreConfig --noEmit --target ES2022 --module ESNext --moduleResolution Bundler --lib ES2022,DOM ../../docs/plugins/exemplars/web-ui/bundle/register-mesh-plugin-ui.ts)
    echo ""
    echo "=== 7-10/11 Parallel portable checks and builds ==="
    scripts/test-portable.sh
    echo ""
    echo "=== 11/11 E2E smoke tests (Playwright) ==="
    (cd "{{ ui_dir }}" && pnpm run test:e2e)
    echo ""
    echo "All checks passed."

# Start a lite client — no GPU, no model, just a local HTTP proxy to the mesh host.

# Only needs the mesh-llm binary (no llama.cpp binaries or model).
mesh-client join="" port="9337" console="3131" config="":
    #!/usr/bin/env bash
    set -euo pipefail
    args=(client --port "{{ port }}" --console "{{ console }}")
    if [[ -n "{{ join }}" ]]; then
        args+=(--join "{{ join }}")
    fi
    if [[ -n "{{ config }}" ]]; then
        args+=(--config "{{ config }}")
    fi
    "{{ mesh_bin }}" "${args[@]}"

# Build and auto-join a mesh (discover via Nostr)
auto: build
    "{{ mesh_bin }}" --auto

# ── Utilities ──────────────────────────────────────────────────

# Measure repository-local Cargo build-cache usage (80 GiB / 14 day defaults).
[unix]
cache-status max_size="80GiB" max_age="14":
    python3 scripts/manage-build-cache.py status --max-size "{{ max_size }}" --max-age "{{ max_age }}"

# Emit Cargo's effective workspace and artifact-directory metadata as JSON.
[unix]
cache-cargo-metadata:
    @cargo metadata --no-deps --format-version 1

# Remove one package from an explicitly validated Cargo target directory.
[unix]
cache-cargo-clean:
    @test -n "${MESH_LLM_CACHE_TARGET_DIR:-}" || { echo "cache-cargo-clean: MESH_LLM_CACHE_TARGET_DIR is unset; this recipe is driven by scripts/manage-build-cache.py" >&2; exit 1; }
    @test -n "${MESH_LLM_CACHE_PACKAGE:-}" || { echo "cache-cargo-clean: MESH_LLM_CACHE_PACKAGE is unset; this recipe is driven by scripts/manage-build-cache.py" >&2; exit 1; }
    @cargo clean --target-dir "$MESH_LLM_CACHE_TARGET_DIR" -p "$MESH_LLM_CACHE_PACKAGE"

# Preview bounded, oldest-first incremental and package-aware Cargo cleanup.
[unix]
cache-prune-dry-run max_size="80GiB" max_age="14":
    python3 scripts/manage-build-cache.py prune --max-size "{{ max_size }}" --max-age "{{ max_age }}"

# Prune only repository-local Cargo artifacts; refuses during compilation.
[unix]
cache-prune max_size="80GiB" max_age="14":
    python3 scripts/manage-build-cache.py prune --execute --max-size "{{ max_size }}" --max-age "{{ max_age }}"

# Update both tracked llama.cpp pin files from the prepared checkout.
llama-update-pin:
    scripts/update-llama-pin.sh

# Render a Markdown summary for a llama.cpp upstream pin change.
llama-summary old new:
    scripts/summarize-llama-upstream.sh "{{ old }}" "{{ new }}"

# Clean Rust, llama.cpp, and UI build artifacts.
[unix]
clean:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf \
        target \
        .deps/llama.cpp/build-stage-abi-* \
        .deps/llama-build/build-stage-abi-* \
        "{{ ui_dir }}/node_modules" \
        "{{ ui_dir }}/dist"
    echo "Cleaned Rust target, llama.cpp build dirs, and UI artifacts"

[windows]
clean:
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "Remove-Item -Recurse -Force target,'.deps/llama.cpp/build-stage-abi-*','.deps/llama-build/build-stage-abi-*','{{ ui_dir }}/node_modules','{{ ui_dir }}/dist' -ErrorAction SilentlyContinue"
    echo "Cleaned Rust target, llama.cpp build dirs, and UI artifacts"

# Clean UI build artifacts (node_modules, dist). Fixes stale pnpm state.
[unix]
ui-clean:
    cd "{{ ui_dir }}" && rm -rf node_modules dist
    echo "Cleaned UI: node_modules + dist removed"

[windows]
ui-clean:
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location '{{ ui_dir }}'; Remove-Item -Recurse -Force node_modules,dist -ErrorAction SilentlyContinue"
    echo "Cleaned UI: node_modules + dist removed"
# Stop mesh-llm processes
stop:
    pkill -f "mesh-llm" 2>/dev/null || true
    echo "Stopped"

# Quick test inference (works with any running server on 8080 or 8090)
test port="9337":
    curl -s http://localhost:{{ port }}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello! Write a haiku about distributed computing."}],"max_tokens":50}' \
        | python3 -c "import sys,json; d=json.load(sys.stdin); t=d['timings']; print(d['choices'][0]['message'].get('content','')[:200]); print(f\"  prompt: {t['prompt_per_second']:.1f} tok/s  gen: {t['predicted_per_second']:.1f} tok/s ({t['predicted_n']} tok)\")"

# Show the local llama.cpp ABI patch queue
diff:
    ls -1 third_party/llama.cpp/patches

# Build the client-only Docker image
[unix]
docker-build-client tag="mesh-llm:client":
    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.client -t {{ tag }} .

[windows]
docker-build-client tag="mesh-llm:client":
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:DOCKER_BUILDKIT='1'; docker build -f docker/Dockerfile.client -t '{{ tag }}' ."

# Run the client console image locally
docker-run-client tag="mesh-llm:client":
    docker run --rm -p 3131:3131 -p 9337:9337 -e APP_MODE=console {{ tag }}
