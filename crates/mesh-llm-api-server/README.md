# mesh-llm-api-server

`mesh-llm-api-server` is the public Rust SDK crate for applications that embed a Mesh
node with model management and local serving. Client-only applications should
use `mesh-llm-api-client`.

The target public concept is `MeshNode`: one embedded node that can:

- join a mesh and consume inference
- search, inspect, download, install, delete, and clean up local models
- load and unload local models for serving
- observe connection, model, serving, and request lifecycle events

SDK layering:

- `crates/mesh-client/` implements the low-level client behavior
- `crates/mesh-llm-api-client/` exposes the client-only Rust SDK surface
- `crates/mesh-llm-api-server/` exposes the node Rust SDK surface and re-exports the
  client types for compatibility
- `crates/mesh-llm-node/` owns embeddable model management and serving
  orchestration as it is extracted from the host runtime. Serving is an
  in-process SDK boundary; REST management remains an external-daemon adapter.
- `crates/mesh-llm-host-runtime/` provides the reference `ServingController`
  implementation by attaching `MeshApi` to `MeshNodeBuilder` and forwarding
  load/unload to the runtime-control loop.
- `crates/mesh-llm-ffi/` wraps `crates/mesh-llm-api-server/` for Swift, Kotlin, and other native
  bindings

Serving APIs use model refs for load, explicit model-or-instance unload
targets, drain/force unload options, rich served-model status, and typed
high-level serving errors.

If an API is meant for client-only app integration, it belongs in
`mesh-llm-api-client`. If it requires model management or local serving, it
belongs in `mesh-llm-api-server`.
