---
title: Swift and Apple SDK
---

# Swift and Apple SDK

## Install

Use the `MeshLLM` SwiftPM product in macOS, Mac Catalyst and iOS applications.
Tagged packages resolve a prebuilt XCFramework. Add the package dependency:

```swift
.package(url: "https://github.com/Mesh-LLM/mesh-llm", from: "{{ site.sdkVersion }}")
```

Use `.product(name: "MeshLLM", package: "mesh-llm")` in your target dependencies.
The SDK exposes one embedded `Node` in `.client`, `.serve` or `.combined`
mode. `.serve` is serve-only.

## Join and infer

```swift
import Foundation
import MeshLLM

let node = try Node(mode: .client, autoJoin: true)
try await node.start()
do {
    let llms = try await node.inference.listModels()
    guard let model = llms.first else {
        throw NSError(domain: "MeshLLM", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "No LLM is advertised yet."])
    }
    let response = try await node.inference.chatCompletions([
        "model": model.id,
        "messages": [["role": "user", "content": "Hello from Swift."]],
    ])
    print(try response.jsonObject())
    try await node.stop()
} catch {
    try? await node.stop()
    throw error
}
```

Use `joinTokens: [token]` for a private mesh and `ownerKeyPath` for an existing
Mesh LLM owner keystore. Client mode needs no serving runtime.

## Read local runtime capacity

For a started serving node and an exact local serving ID:

```swift
if let capacity = try await node.runtimeCapacity("exact-serving-id") {
    print(capacity.executionLanes, capacity.contextTokensPerLane,
          capacity.contextPoolTokens, capacity.capturedAtUnixMs)
}
```

`LaneCapacity` fields are `UInt64`. Each call reads the latest published
observation without HTTP. Busy lanes count toward capacity; this does not reserve
resources. Unknown, unsupported, unavailable or incomplete capacity returns `nil`;
stopped nodes throw. The timestamp is the oldest observation in Unix milliseconds;
callers choose a freshness policy for long inference. The total pool is not a
per-request context limit.

## Discover advertised models

`node.models` queries the embedded mesh directly for all advertised workloads,
including local and remote OCR, speech, embedding, decision and LLM models.

```swift
let page = try await node.models.list()
let query = MeshModelQuery(workloadClass: "document_ocr", available: true, limit: 50)
let ocr = try await node.models.list(query)
let model = try await node.models.get("org/model@revision#profile")
for provider in model.providers {
    print(provider.providerId, provider.workloadClass as Any, provider.runtime as Any)
}
if let cursor = ocr.nextCursor {
    var nextQuery = query
    nextQuery.cursor = cursor
    let next = try await node.models.list(nextQuery)
}
```

The result types are `MeshModelPage`, `MeshModel`, `MeshModelProvider` and
`MeshModelRuntime`. Optional facts remain absent when unknown; string-valued
workloads normalize unrecognized peer values to `unknown`; the original workload
string is not preserved. Kinds are string-valued. Open-ended metadata uses
`ModelJSONValue`. Filters combine with AND; preserve them when continuing a page.

Runtime limits, capability evidence, thinking controls and payment offers apply
to their provider. A model's operation list means at least one provider offers
each operation. `node.inference.listModels()` remains the LLM-only compatibility
view. Full discovery does not make an OpenAI HTTP request.

## Streaming

Use `node.inference.streamChatCompletions(body)` or `streamResponses(body)`.
A `.sse(frame)` value exposes the event name, data and raw frame; call
`frame.jsonObject()` for nonterminal data. Tool calls, reasoning, structured
outputs, usage and future fields survive the bridge.

## Serving and native runtime

Resolve a compatible runtime explicitly before starting a serving node:

```swift
let runtime = try await NativeRuntime.resolve(NativeRuntimeResolveOptions(
    artifactDirectory: URL(fileURLWithPath: "/path/to/bundled/runtime"),
    allowDownload: false
))
let servingNode = try Node(mode: .combined, models: ["<model-reference>"])
try await servingNode.start()
```

Node configuration selects loaded models. Discovery does not download or load
them. Check the release's platform support before enabling local serving.

For local source development, build the matching XCFramework with
`mesh/sdk/swift/scripts/build-xcframework.sh`. Binding generation reads the
prebuilt FFI library's actual metadata. A standalone regeneration accepts an
absolute library path:

```bash
mesh/sdk/swift/scripts/generate-swift-bindings.sh /absolute/path/libmeshllm_ffi.a
```

## Document OCR

```swift
let page = try await node.models.list(
    MeshModelQuery(workloadClass: "document_ocr", available: true))
if let model = page.data.first {
    let response = try await node.inference.documentOcr(
        Data(contentsOf: URL(fileURLWithPath: "scan.pdf")),
        options: DocumentOcrOptions(model: model.id, filename: "scan.pdf",
            contentType: "application/pdf", responseFormat: "json", maxPages: 10))
    let pages = try response.jsonObject()
}
```

OCR uploads PDF/image bytes to `/mesh/v1/documents/ocr` and preserves the provider's
status, content type and body. It raises `OpenAIRequestFailure` for non-2xx by
default; `raiseForStatus: false` returns the raw response. Generic
`inference.request` also checks status by default, matching the other SDKs.
**Migration:** pass `raiseForStatus: false` to retain the previous raw-response
behavior, then optionally call `response.errorForStatus()` later.
Plain text/Markdown responses use `response.body`; JSON uses `jsonObject()`.

## Additional mesh memberships

```swift
let team = try await node.addMembership(MeshMembershipConfig(
    name: "team", joinTokens: [inviteToken], trustedOwners: [ownerId], apiPort: 9338))
let memberships = try await node.memberships()
let removed = try await node.removeMembership("team")
```

A membership has an independent node identity, owner allowlist and peer set.
An empty owner allowlist admits nobody. Adding the same name replaces its
membership under the Rust core's validation/recovery rules. The native model
catalog queries the primary mesh and does not merge separate memberships.

## Live transcription

Embedded live transcription uses the node's locally loaded speech model through
in-process typed channels, without a loopback HTTP or WebSocket connection.
A remote-only speech model is rejected at startup; Rust callers that need mesh
relaying can explicitly use `LiveTranscription::start(api_base_url, options)`.
Load a speech model on this node before using the following example. Dropping
either native session half, closing it, or stopping the node cancels the work.

Supply signed 16-bit mono PCM samples recorded at 16 kHz. With turn detection
disabled, call `commit()` to end the turn explicitly:

```swift
let session = try await node.startLiveTranscription(LiveTranscriptionOptions(
    model: "speech-model", sampleRate: 16000, turnDetection: nil,
    stepMs: 400, tentative: true))
do {
    try await session.sendAudio(pcm16Samples)
    try await session.commit()
    if let event = try await session.nextEvent() {
        print(event)
    }
    let remainingEvents = try await session.stop()
    print(remainingEvents)
} catch {
    await session.close()
    throw error
}
```

`sendAudioF32` accepts floating point samples. Defaults use 16 kHz and server-side
turn detection. `stop()` returns remaining final events; `close()` abandons the
session. Stop the node when the application exits.

`stepMs` accepts 200 to 5000 ms; `nil` keeps the node's 1000 ms default.
Shorter steps run more transcription passes. `tentative` defaults to `false`.

Confirmed deltas are append-only. Recognition errors in confirmed text cannot
be corrected by later passes or the completed event. Shorter transcription
intervals do not guarantee better recognition accuracy.

Native transcription failures end the session after an error event. Start a
new session to retry. Request-validation errors may leave the session open.

Each `tentative` event replaces the previous unconfirmed text for its `itemId`;
empty text clears it. Append `delta` text to the confirmed transcript and let
`completed` replace the whole turn, including tentative text. Read events
concurrently with microphone input.

## Full embedded-node configuration

`Node.fromConfig` passes a `[String: ModelJSONValue]` object through to the
complete Rust `EmbeddedMeshNodeConfig` schema. It exposes the same relay, bind,
publication, trust/admission, storage, serving-limit and timeout settings.

```swift
let node = try Node.fromConfig([
    "mode": .string("client"),
    "network": .object([
        "join_tokens": .array([.string(inviteToken)]),
        "enumerate_host": .bool(false)
    ]),
    "startup_timeout": .object(["secs": .number(30), "nanos": .number(0)])
])
try await node.start()
```

The object preserves snake_case keys and native enum strings. Native defaults
apply to omitted fields and unknown keys are rejected. Full-config `mode`
values are `client`, `serve_only` and `serve` (combined); its default is
combined, matching Rust. The existing `Node(...)` convenience constructor keeps
its client default and Swift argument names.

## Structured inference failures

`OpenAIRequestFailure.details` and `OpenAIStreamFailure.details` expose typed
`InferenceErrorDetails` with `message`, `type`, `code` and `param`:

```swift
do {
    _ = try await node.inference.chatCompletions(request)
} catch let error as OpenAIRequestFailure {
    if error.details?.code == "context_length_exceeded" {
        // Compact the conversation before retrying.
    } else {
        throw error
    }
}
```

Inspect stream failures the same way. The original status and body remain
available. `details` is `nil` for transport failures or an unstructured body;
missing optional fields stay `nil`. Unknown codes are preserved.
