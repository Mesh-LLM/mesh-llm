# Mesh LLM SDKs

The Python SDK package lives in [`python/`](python/). Other public platform SDKs
use the real embedded `MeshNode` in client, serve-only,
or combined mode. The legacy thin client and its platform `Client` wrappers
have been removed. Start with [SDK usage](../docs/SDK.md) and the language
specific guides under [docs/sdk](../docs/sdk/).

Rust native backend failures expose `inference::InferenceError`. Node.js,
Swift and Kotlin request and stream failures expose typed `details` containing
the backend's message, type, code and parameter, retaining the original status
and raw body. Use `context_length_exceeded` to identify context exhaustion;
transport or unstructured failures have no structured classification.
Python's separately maintained SDK is deferred from this addition.
