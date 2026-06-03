# Swift SDK

Use the GitHub Swift package from tagged `Mesh-LLM/mesh-llm` releases.

## Install

```swift
dependencies: [
    .package(url: "https://github.com/Mesh-LLM/mesh-llm", from: "0.68.0"),
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            .product(name: "MeshLLM", package: "mesh-llm"),
        ]
    ),
]
```

Tagged releases resolve the prebuilt `MeshLLMFFI.xcframework` through SwiftPM.
For local checkout development, build the XCFramework first:

```bash
./sdk/swift/scripts/build-xcframework.sh
```

## Client: Public Mesh

```swift
import MeshLLM

let ownerKeypair = generateOwnerKeypairHex()
let client = try await Client.connectPublic(
    ownerKeypairBytesHex: ownerKeypair,
    query: PublicMeshQuery(
        model: "Qwen3",
        minVramGb: nil,
        region: nil,
        targetName: nil,
        relays: []
    )
)

try await client.start()
let publicModels = try await client.inference.listModels()
try await printChat(
    stream: client.inference.chat(ChatRequest(model: publicModels[0].id, messages: [
        ChatMessage(role: "user", content: "Say hello from a public mesh.")
    ]))
)
await client.stop()
```

## Client: Private Mesh

```swift
import MeshLLM

let ownerKeypair = generateOwnerKeypairHex()
let client = try Client(
    inviteToken: InviteToken(ProcessInfo.processInfo.environment["MESH_PRIVATE_INVITE"]!),
    ownerKeypairBytesHex: ownerKeypair
)

try await client.start()
let models = try await client.inference.listModels()
try await printChat(
    stream: client.inference.chat(ChatRequest(model: models[0].id, messages: [
        ChatMessage(role: "user", content: "Say hello from a private mesh.")
    ]))
)
await client.stop()
```

## Inference Helper

```swift
func printChat(stream: AsyncThrowingStream<Event, Error>) async throws {
    for try await event in stream {
        if case .tokenDelta(_, let delta) = event {
            print(delta, terminator: "")
        }
        if case .completed = event {
            print()
            return
        }
    }
}
```

## Serving: Install Runtime

Resolve or install a native runtime before local serving:

```swift
import MeshLLM

let runtime = try await NativeRuntime.resolve(
    NativeRuntimeResolveOptions(
        artifactDirectory: ProcessInfo.processInfo.environment["MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR"]
            .map(URL.init(fileURLWithPath:)),
        allowDownload: ProcessInfo.processInfo.environment["MESH_SDK_RUNTIME_ALLOW_DOWNLOAD"] == "1"
    )
)
print("using \(runtime.nativeRuntimeId) from \(runtime.path)")
```

## Serving: Public Mesh

```swift
let ownerKeypair = generateOwnerKeypairHex()
let node = try Node(
    inviteToken: InviteToken(ProcessInfo.processInfo.environment["MESH_PUBLIC_INVITE"]!),
    ownerKeypairBytesHex: ownerKeypair
)
try await node.start()

let modelRef = ProcessInfo.processInfo.environment["MESH_SDK_MODEL_REF"] ?? "Qwen2.5-3B-Instruct-Q4_K_M"
_ = try await node.models.download(modelRef)
let served = try await node.serving.load(modelRef, options: LoadModelOptions(devicePolicy: .auto))
try await printChat(stream: node.inference.chat(ChatRequest(model: served.modelId, messages: [
    ChatMessage(role: "user", content: "Say hello from a public serving node.")
])))
try await node.serving.unloadModel(served.modelId, options: UnloadModelOptions(drainTimeoutMs: 1_000, force: false))
try await node.stop()
```

## Serving: Private Mesh

Private mesh serving uses the same lifecycle with `MESH_PRIVATE_INVITE`:

```swift
let ownerKeypair = generateOwnerKeypairHex()
let node = try Node(
    inviteToken: InviteToken(ProcessInfo.processInfo.environment["MESH_PRIVATE_INVITE"]!),
    ownerKeypairBytesHex: ownerKeypair
)
```

## macOS Example

```bash
./sdk/swift/scripts/build-xcframework.sh
scripts/package-native-runtime.sh \
  --backend metal \
  --target aarch64-apple-darwin \
  --out dist/native-runtimes

MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR=dist/native-runtimes/meshllm-native-runtime-darwin-aarch64-metal \
MESH_SDK_MODEL_REF=Qwen2.5-3B-Instruct-Q4_K_M \
swift run --package-path sdk/swift/example/MeshExampleApp
```

## Console Assets

Tagged Swift package releases include console assets as SwiftPM resources when
console support is advertised:

```swift
let options = ConsoleOptions.packaged()
```
