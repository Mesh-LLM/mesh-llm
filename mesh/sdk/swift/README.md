# MeshLLM Swift SDK

The package exposes one real embedded `Node` with client, serve-only, and
combined roles. The legacy thin `Client` is removed. See the
[swift usage guide](../../docs/sdk/swift.md) for current constructors,
streaming, role selection, and native runtime requirements.

The `serve` role serves local models; `combined` also permits mesh inference.
Client mode starts no native serving runtime. Supply a Mesh LLM owner keystore
path through `ownerKeyPath`/`owner_key_path` if owner identity is required.

The native `models` API discovers all workloads advertised in the node's mesh:

```swift
let page = try await node.models.list()
let ocr = try await node.models.list(MeshModelQuery(workloadClass: "document_ocr", available: true))
let model = try await node.models.get("org/model@revision#profile")
if let cursor = page.nextCursor {
    let next = try await node.models.list(MeshModelQuery(cursor: cursor))
}
```

Results are typed `MeshModelPage`, `MeshModel`, and `MeshModelProvider` values.
Retain the same filters when continuing a page. Workload and kind strings
preserve future advertisements; optional metadata remains absent when unknown.
Open-ended capability, thinking, payment and plugin data uses `ModelJSONValue`.
Provider runtime observations apply to that provider. `inference.listModels()`
continues to expose only compatible LLMs. Discovery calls the embedded node
directly rather than using the OpenAI HTTP endpoint.

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
