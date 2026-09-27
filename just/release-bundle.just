# Create release archive(s) for the current platform.

# `version` should be a tag like v0.30.0.
[unix]
release-bundle version output="dist" flavor="" arch="":
    @MESH_RELEASE_FLAVOR="{{ flavor }}" MESH_RELEASE_ARCH="{{ arch }}" scripts/package-release.sh "{{ version }}" "{{ output }}"

[windows]
release-bundle version output="dist" flavor="":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package-release.ps1 -Version "{{ version }}" -OutputDir "{{ output }}" -Flavor "{{ flavor }}"

# Run repo-level release-target consistency checks.
[unix]
check-release:
    just with-lld cargo xtool repo-consistency release-targets

[windows]
check-release:
    @just with-lld cargo xtool repo-consistency release-targets

# Check the Rust process-environment mutation census and contract annotations.
check-env-mutation-contract:
    @python3 scripts/check-env-mutation-contract.py

# Run one release-attestation xtask action through the repository toolchain.
release-attestation *ARGS:
    @just with-lld cargo run -q -p xtask -- release-attestation {{ ARGS }}
