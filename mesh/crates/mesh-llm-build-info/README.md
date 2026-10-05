# mesh-llm-build-info

Shared build and release version constants for Mesh LLM.

This crate is intentionally dependency-free. It lets build scripts stamp a
source build with a SHA-bearing display version while preserving the package
release version for compatibility checks, cache identity, and release metadata.

## API Shape

- `BUILD_VERSION` is the stamped display version when
  `MESH_LLM_BUILD_VERSION` is set at compile time, otherwise the package
  version.
- `RELEASE_VERSION` is always the plain Cargo package version.
- `is_sha_build(version)` recognizes source-build metadata of the form
  `+g<hex>` and `+g<hex>.dirty`.
