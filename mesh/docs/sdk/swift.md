# Swift SDK

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
