# Build release artifacts for the current platform.

# Prepare, publish, and watch a GitHub release from main.
release version *ARGS:
    @scripts/release.sh "{{ version }}" {{ ARGS }}

# Build the backend-neutral release host once for the current platform.
[unix]
release-host-build:
    @scripts/build-release.sh

[windows]
release-host-build:
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -BuildProfile release -HostOnly

# Build one packageable native runtime for the current platform.
[unix]
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

# Build the complete Skippy product, then the backend-neutral MeshLLM product.
[unix]
release-build: release-runtime-build skippy-cli-release-build release-host-build

[windows]
release-build:
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend cpu -BuildProfile release -DynamicHost

# Build a Linux aarch64 CPU release artifact on a native aarch64 runner.
release-build-aarch64:
    @scripts/package-native-runtime.sh --build --backend cpu --target aarch64-unknown-linux-gnu
    @just skippy-cli-release-build
    @just release-host-build

# Build a Linux aarch64 CUDA release artifact (Jetson/Orin).
# SM arches selected by MESH_CUDA_VERSION env (set by CI matrix); a bare
# `just` invocation with the var unset detects the installed toolkit instead
# of assuming one (see scripts/detect-cuda-toolkit-version.sh); no compiler or
# toolkit metadata is an actionable error. Thor (sm_110)
# needs toolkit >= 13 -- gate on the major, the same shape as the x86_64
# recipe's Blackwell gate, so a future CUDA 14 keeps Thor instead of falling
# back to the pre-13 list the way a `13.*` glob did.
release-build-aarch64-cuda:
    #!/usr/bin/env bash
    set -euo pipefail
    cuda_version="$(scripts/detect-cuda-toolkit-version.sh)"
    major="${cuda_version%%.*}"
    if [[ "$major" -ge 13 ]]; then
      arches='75;80;86;87;89;90;110'
    else
      arches='61;75;80;86;87;89;90'
    fi
    MESH_LLM_CUDA_TOOLKIT_MAJOR="${MESH_LLM_CUDA_TOOLKIT_MAJOR:-$major}" \
      LLAMA_STAGE_CUDA_ARCHITECTURES="$arches" \
      scripts/package-native-runtime.sh --build --backend cuda --target aarch64-unknown-linux-gnu
    just skippy-cli-release-build
    just release-host-build

# Prepare the pinned llama.cpp checkout and apply the Mesh-LLM ABI patch queue.
llama-prepare:
    @scripts/prepare-llama.sh pinned

# Prepare llama.cpp at upstream master and apply the Mesh-LLM ABI patch queue.
llama-prepare-latest:
    @scripts/prepare-llama.sh latest

# Build the patched llama.cpp ABI static libraries.
llama-build: llama-prepare
    @scripts/build-llama.sh

# Build a Linux CUDA release artifact.
# SM arches selected by MESH_CUDA_VERSION env (set by CI matrix); a bare
# `just` invocation with the var unset detects the installed toolkit instead
# of assuming one (see scripts/detect-cuda-toolkit-version.sh); no compiler or
# toolkit metadata is an actionable error. Blackwell
# (sm_100/103/120/121) needs toolkit >= 12.8, the first release that shipped
# support for it -- gate on that boundary, not on the CUDA major alone.
release-build-cuda:
    #!/usr/bin/env bash
    set -euo pipefail
    cuda_version="$(scripts/detect-cuda-toolkit-version.sh)"
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
    just skippy-cli-release-build
    just release-host-build

release-build-cuda-windows cuda_arch="61;75;80;86;87;89;90":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend cuda -CudaArch "{{ cuda_arch }}" -BuildProfile release -DynamicHost

# Build a Linux ROCm ABI release artifact with an explicit architecture list.
release-build-rocm rocm_arch="gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1103;gfx1151;gfx1200;gfx1201":
    @LLAMA_STAGE_AMDGPU_TARGETS="{{ rocm_arch }}" \
      scripts/package-native-runtime.sh --build --backend rocm --target x86_64-unknown-linux-gnu
    @just skippy-cli-release-build
    @just release-host-build

release-build-rocm-windows rocm_arch="gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1103;gfx1151;gfx1200;gfx1201":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend rocm -RocmArch "{{ rocm_arch }}" -BuildProfile release -DynamicHost

# Build a Linux Vulkan ABI release artifact.
release-build-vulkan:
    @scripts/package-native-runtime.sh --build --backend vulkan --target x86_64-unknown-linux-gnu
    @just skippy-cli-release-build
    @just release-host-build

release-build-vulkan-windows:
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend vulkan -BuildProfile release -DynamicHost
