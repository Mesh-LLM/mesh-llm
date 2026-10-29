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

## Thinking effort levels

`listModels()` returns each model with the effort strings the provider observed
in its loaded chat template:

```swift
let model = try await node.inference.listModels()
    .first { $0.thinkingEfforts?.contains("high") == true }
if let model {
    let response = try await node.inference.chatCompletions([
        "model": model.id,
        "messages": [["role": "user", "content": "Explain lock ordering."]],
        "reasoning_effort": "high",
    ])
}
```

`thinkingEfforts` is `nil` when the provider is older or has not probed the
model, and empty when the template exposes no effort levels. The list is an
observation of the template, not a guarantee that generation obeys it.
