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
