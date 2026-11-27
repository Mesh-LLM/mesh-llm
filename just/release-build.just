# Build release artifacts for the current platform. Skippy runtimes are built
# and released by the Skippy project; MeshLLM only builds its host.

# Prepare, publish, and watch a GitHub release from main.
release version *ARGS:
    @scripts/release.sh "{{ version }}" {{ ARGS }}

# Build the backend-neutral release host for the current platform.
[unix]
release-host-build:
    @scripts/build-release.sh

[windows]
release-host-build:
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -BuildProfile release

release-build: release-host-build
