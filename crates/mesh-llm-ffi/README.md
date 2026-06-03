# mesh-llm-ffi

`mesh-llm-ffi` exposes the Mesh node SDK through a native FFI layer for
language bindings, including model management, inference, and serving control
when built with the host runtime feature.

This crate is the bridge used by the generated Swift and Kotlin SDKs. It should
stay thin and map the canonical Rust API from `crates/mesh-llm-sdk/` into an
FFI-safe surface.

Layering:

- `crates/mesh-client/` implements low-level client behavior
- `crates/mesh-llm-api-client/` and `crates/mesh-llm-api-server/` implement the
  lower-level Rust client and node APIs
- `crates/mesh-llm-sdk/` defines the public SDK facade and feature model
- `crates/mesh-llm-ffi/` adapts that SDK for cross-language consumers

The FFI layer should expose public model ids as the same full model refs used by
mesh and `/v1/models`; it should not derive identities from GGUF filenames.

Application code should usually depend on `crates/mesh-llm-sdk/` directly unless
it is building a non-Rust binding.
