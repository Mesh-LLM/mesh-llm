---
title: Swift and Apple SDK
---

# Swift and Apple SDK

Use the `MeshLLM` SwiftPM product in macOS, Mac Catalyst, and iOS applications. Tagged releases resolve the prebuilt `MeshLLMFFI.xcframework` automatically; local checkout builds use the same UniFFI-backed implementation.

## Install

Add the tagged Mesh-LLM package to `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/Mesh-LLM/mesh-llm", from: "{{ site.sdkVersion }}"),
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

For local checkout development, build the XCFramework before building or testing the Swift package:

```bash
./sdk/swift/scripts/build-xcframework.sh
```

## Connect as a client

```swift
import MeshLLM

let client = try await Client.connectPublic(
    ownerKeypairBytesHex: generateOwnerKeypairHex(),
    query: PublicMeshQuery(model: "Qwen3")
)

do {
    try await client.start()

    let models = try await client.inference.listModels()
    let request = ChatRequest(
        model: models[0].id,
        messages: [ChatMessage(role: "user", content: "Say hello from Swift.")]
    )

    for try await event in client.inference.chat(request) {
        switch event {
        case .tokenDelta(_, let delta):
            print(delta, terminator: "")
        case .completed:
            print()
        default:
            break
        }
    }
} catch {
    await client.stop()
    throw error
}

await client.stop()
```

For a private mesh, initialize `Client` with `InviteToken(...)` and the app's persisted owner keypair instead of `Client.connectPublic(...)`.

## Embed local serving

Resolve a matching native runtime before loading a local model:

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

Create a `Node`, load the model, stream inference, and unload it before stopping:

```swift
let node = try Node(
    inviteToken: InviteToken(ProcessInfo.processInfo.environment["MESH_INVITE_TOKEN"]!),
    ownerKeypairBytesHex: generateOwnerKeypairHex()
)

do {
    try await node.start()

    let modelRef = ProcessInfo.processInfo.environment["MESH_SDK_MODEL_REF"]
        ?? "Qwen2.5-3B-Instruct-Q4_K_M"
    _ = try await node.models.download(modelRef)
    let served = try await node.serving.load(
        modelRef,
        options: LoadModelOptions(devicePolicy: .auto)
    )

    let request = ChatRequest(
        model: served.modelId,
        messages: [ChatMessage(role: "user", content: "Say hello from local serving.")]
    )
    for try await event in node.inference.chat(request) {
        if case .tokenDelta(_, let delta) = event {
            print(delta, terminator: "")
        }
    }

    try await node.serving.unloadModel(
        served.modelId,
        options: UnloadModelOptions(drainTimeoutMs: 1_000, force: false)
    )
    try await node.stop()
} catch {
    try? await node.stop()
    throw error
}
```

Handle `MeshError.ServingUnsupported` for iOS and other targets without validated local serving. The current support line is macOS local serving; Mac Catalyst is under validation and iOS should use `Client` to reach another serving node.

## Console assets

Tagged packages can include the console as SwiftPM resources:

```swift
let console = try await node.startConsole(.packaged(port: 0))
print(console.url)
```

Keep the console optional and stop it with the app's lifecycle. For local package development, prepare assets with `scripts/package-sdk-console-assets.sh --sdk swift`.

## Apple integration notes

- Persist the owner keypair in Keychain or another app-owned secure store.
- Reconnect from `UIApplication.willEnterForegroundNotification` after iOS backgrounding.
- The SDK uses QUIC/TLS; configure `ITSAppUsesNonExemptEncryption` in `Info.plist` according to the app's export-compliance status.
- The distributed XCFramework includes a `PrivacyInfo.xcprivacy` manifest; verify it remains inside each framework slice when repackaging.
- Do not spawn the `mesh-llm` CLI as a sidecar. The Swift SDK uses the native bridge directly.
