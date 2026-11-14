# mesh-llm-ffi

UniFFI bridge for the embedded Mesh LLM node. It exposes a single `MeshNodeHandle`
with client, serve-only, and combined roles, OpenAI-compatible requests and
streams, and native runtime management. Python, Swift, and Kotlin bindings are
generated from the built library's exported UniFFI metadata with UniFFI 0.32.0.
The legacy UDL alone does not describe the complete surface or its checksums.
The legacy `MeshClientHandle` is removed.

`MeshNodeHandle::runtime_capacity(model)` forwards the local SDK observation as
an optional `RuntimeCapacityNative` record with lossless unsigned fields. It
preserves missing capacity separately from node failures, without HTTP.

See [SDK usage](../../docs/SDK.md).
