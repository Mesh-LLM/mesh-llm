---
title: SDKs
---

# Embed Mesh in your app

Rust, Node.js, Python, Swift, and Kotlin expose an embedded node with the same
public capabilities. Names and async syntax follow each language's conventions;
model workloads, filters, metadata, pagination and operation behavior match.

## Choose a node role

| Role | Rust builder | Other SDKs' mode | Behavior |
|---|---|---|---|
| Client | `.client()` | `client` | Join a mesh and consume inference without loading a serving runtime |
| Serve-only | `.serve_only()` | `serve` | Serve local models; SDK inference calls are disabled |
| Combined | `.serve()` | `combined` | Serve local models and consume mesh inference |

The old separate thin `Client` has been removed. Use `MeshNode` in Rust and
`Node` in the other SDKs. Supply an owner keystore path when owner identity is
required. A node accepts invite tokens for a private mesh or public-mesh
auto-join. Keep API and console ports isolated when embedding multiple nodes.

A normal lifecycle is:

```text
create node with role and join configuration
start
query the mesh model catalog
invoke the appropriate model operation
stop
```

Serving applications explicitly install or bundle a compatible native runtime
before startup and configure the local models to serve. Startup does not
implicitly download a runtime. Runtime artifacts are matched to the release,
platform/backend and exact Skippy ABI; a package's presence alone does not
establish that its target can serve models locally.

## Common SDK surface

| Capability | Common behavior |
|---|---|
| Lifecycle | Start, stop, status, and join by invite token |
| Native runtimes | Explicit install with progress, list, resolve/validate, remove/prune and version/ABI queries |
| Model discovery | List all primary-mesh advertisements, combine typed filters, follow bounded pages, and look up an exact ID |
| Compatible inference | LLM-only compatibility model list; protocol-preserving request and SSE stream APIs |
| Document OCR | Binary PDF/image upload, exact model ID, output-format/page options, and provider status/body |
| Mesh memberships | Add, list and remove additional isolated memberships without loading local models again |
| Live transcription | Send signed 16-bit mono PCM, commit turns, read events, and stop the session |

SDK changes must update all five language surfaces, native bridges, tests,
product docs and website examples together. Python is maintained in a separate
repository and uses a coordinated source/native checkpoint. Use matching
language packages and native artifacts; this documentation does not publish
new packages or qualify target-specific serving support.

Inference requests and OCR calls check HTTP success by default across all five
SDKs. Use Rust's `request_raw`, `stream_raw` or `document_ocr_raw` for deliberate
provider-status handling. Other SDKs offer `raiseForStatus: false` or Python's
`raise_for_status=False` on raw request methods. Raw responses retain status,
content type and body. Node.js raw responses expose `statusCode`, `contentType`,
`body` and `json()`; check `statusCode` explicitly in raw mode. Rust/Swift/Kotlin generic
requests previously returned non-success responses by default; those callers
must now select raw mode explicitly.

Live sessions start with `start_live_transcription` in Rust/Python or
`startLiveTranscription` in Node.js/Swift/Kotlin. Defaults use 16 kHz mono audio
and turn detection with threshold `0.5`, 300 ms prefix padding and 500 ms
silence. An explicit null turn detector (Python `None`) enables push-to-talk.
Sessions retain speech-start/stop, delta, committed, completed and error events.
They use the embedded realtime WebSocket, while native model queries use the
in-process discovery service.

## Discover all models; choose the right operation

The native `node.models` catalog includes local and remote LLM, speech,
embedding, reranking, decision and OCR advertisements. It is the node's observed
inventory in its primary mesh, not the downloadable catalog or a global
registry. Extra memberships do not merge into it implicitly.

Compatibility model lists include only available confirmed LLMs. Non-LLM
models retain their inference endpoints. For example, discover an OCR model
through `node.models`, then invoke document OCR with that exact ID.

Pages have `data` and `next_cursor`. List filters combine with AND; retain the
same filters when continuing a cursor and restart when its snapshot expires.
Runtime limits, capability evidence, thinking controls and payment observations
remain associated with each provider. A single replica's limits are not a
promise about all replicas. Missing metadata means unknown.

See [Mesh model discovery](/docs/pages/model-discovery/) for complete HTTP
shapes, filters and compatibility migration.

Only the native catalog query is guaranteed to avoid HTTP. Compatibility
inference and some lifecycle/session APIs use the embedded listener. The SDKs
do not require a separately supervised HTTP sidecar, but their whole lifecycle
is not HTTP-free.

## Pick a language guide

- [Rust](/docs/pages/sdk-rust/) — `MeshNode`, typed catalog queries and the OpenAI client.
- [Node.js and Electron](/docs/pages/sdk-node/) — `Node`, promises and the N-API addon.
- [Python](/docs/pages/sdk-python/) — `Node`, asyncio and generated UniFFI bindings.
- [Swift](/docs/pages/sdk-swift/) — SwiftPM, typed values and async APIs.
- [Kotlin and Android](/docs/pages/sdk-kotlin/) — Kotlin coroutines and the native bridge.

For raw HTTP clients, use [the compatible inference API](/docs/pages/openai-compatible-api/)
or [the Mesh catalog](/docs/pages/model-discovery/). Stop nodes with the host
application and handle transport failures, provider HTTP errors and unsupported
platform/runtime combinations explicitly.
