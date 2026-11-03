# Swift SDK

Add the tagged Swift package to `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/Mesh-LLM/mesh-llm", from: "0.78.0"),
],
targets: [
    .target(name: "YourApp", dependencies: [
        .product(name: "MeshLLM", package: "mesh-llm"),
    ]),
]
```

The release XCFramework supports arm64 macOS, Mac Catalyst, iOS devices, and
iOS simulators. Intel Apple machines are not supported for MeshLLM inference.

The Swift package exposes one embedded `Node` with `.client`, `.serve`, and
`.combined` roles. `.serve` is serve-only.

```swift
let node = try Node(mode: .client, autoJoin: true)
try await node.start()
let models = try await node.inference.listModels()
let response = try await node.inference.chatCompletions([
    "model": models[0].id,
    "messages": [["role": "user", "content": "Hello"]],
])
try await node.stop()
```

Use `joinTokens: [token]` for a private mesh. To serve local models, use
`mode: .serve` or `.combined` and supply `models: [modelRef]`. Serving requires
a matching native runtime. The previous `Client` and in-memory owner keypair
constructor are removed; `ownerKeyPath` accepts a Mesh LLM keystore path.

Streaming uses `node.inference.streamChatCompletions(...)` or
`streamResponses(...)`. See [runtime artifacts](../SDK.md#native-runtime-artifacts).

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
