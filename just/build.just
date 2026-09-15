# Build the local product, then exercise the embedded console against a real
# isolated mesh-llm process. The harness owns and verifies its cleanup.
[unix]
qa-logging-console-e2e:
    @just build
    @scripts/qa-logging-console-e2e.sh --current-binary ./target/debug/mesh-llm

[unix]
bootstrap-build-tools:
    @scripts/bootstrap-build-tools

[windows]
bootstrap-build-tools:
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap-build-tools.ps1

[private]
[unix]
with-lld *COMMAND:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v sccache >/dev/null 2>&1 || {
        echo "Error: sccache is required; install it and retry." >&2
        exit 1
    }
    scripts/cargo-linker --mesh-probe >/dev/null
    exec {{ COMMAND }}

[private]
[windows]
with-lld *COMMAND:
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "$$ErrorActionPreference = 'Stop'; if (-not (Get-Command sccache -ErrorAction SilentlyContinue)) { throw 'sccache is required; install it and retry.' }; Invoke-Expression '{{ COMMAND }}'"


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
