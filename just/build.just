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
