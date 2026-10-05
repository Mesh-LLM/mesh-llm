# mesh-llm-api-client

Client-only Rust SDK API for applications that want to connect to an existing
MeshLLM mesh and run inference without embedding a local serving runtime.

Use this crate when the application needs discovery, identity, status,
model listing, chat/responses, streaming events, reconnect, and cancellation.
It intentionally does not expose model downloads, local runtime control, or
serving load/unload APIs.
