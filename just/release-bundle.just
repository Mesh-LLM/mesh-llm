# Create release archive(s) for the current platform.

# `version` should be a tag like v0.30.0.
[unix]
release-bundle version output="dist" flavor="" arch="":
    @MESH_RELEASE_FLAVOR="{{ flavor }}" MESH_RELEASE_ARCH="{{ arch }}" scripts/package-release.sh "{{ version }}" "{{ output }}"

[windows]
release-bundle version output="dist" flavor="":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package-release.ps1 -Version "{{ version }}" -OutputDir "{{ output }}" -Flavor "{{ flavor }}"
