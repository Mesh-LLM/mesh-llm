# mesh-llm-ffi

UniFFI bridge for the embedded Mesh LLM node. It exposes a single `MeshNodeHandle`
with client, serve-only, and combined roles, OpenAI-compatible requests and
streams, and native runtime management. Python, Swift, and Kotlin bindings are
generated from `src/mesh_ffi.udl`. The legacy `MeshClientHandle` is removed.

See [SDK usage](../../docs/SDK.md).
