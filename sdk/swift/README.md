# MeshLLM Swift SDK

Swift Package for connecting to mesh-llm meshes from iOS, Mac Catalyst, and macOS apps.

The SDK usage guide, native runtime packaging notes, examples, and platform
support matrix live in [`docs/SDK.md`](../../docs/SDK.md).

## Installation

Add to your app's `Package.swift` using a tagged release:

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

The repo root is the Swift package entrypoint. Tagged releases resolve the
prebuilt XCFramework automatically through SwiftPM.

For development from a local checkout, build the native artifact first:

```bash
./sdk/swift/scripts/build-xcframework.sh
```

That generates `sdk/swift/Generated/MeshLLMFFI.xcframework`, which the root
Swift package picks up automatically for the real UniFFI-backed implementation.
The macOS framework slice must already use the versioned `Versions/A` bundle
layout before it is passed to `xcodebuild -create-xcframework`; `xcodebuild`
wraps the input framework but does not fix a flat macOS bundle.
Release tags must contain a `Package.swift` whose remote XCFramework URL and
checksum have already been prepared with
`scripts/prepare-swift-package-release.sh`; SwiftPM reads the manifest from the
tag and cannot use values generated later by release CI.

Normal SDK builds and tests require the UniFFI-backed XCFramework. The package
does not ship a pure Swift fallback because the SDK must exercise the native
runtime it exposes.

## Usage

Apps that package a native runtime artifact can validate it before starting
local serving. The resolver accepts `MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR`,
`MESHLLM_NATIVE_RUNTIME_DIR`, `MESH_SDK_NATIVE_RUNTIME_DIR`, or an explicit URL:

```swift
import MeshLLM

let runtime = try NativeRuntime.prepare()
print("using \(runtime.artifactId) from \(runtime.artifactDirectory.path)")
```

```swift
import MeshLLM

let ownerKeypair = generateOwnerKeypairHex()
let node = try Node(
    inviteToken: InviteToken("your-invite-token"),
    ownerKeypairBytesHex: ownerKeypair
)

let recommended = try await node.models.recommended()
let serving = try await node.serving.status()
try await node.start()

let models = try await node.inference.listModels()
let request = ChatRequest(model: models[0].id, messages: [
    ChatMessage(role: "user", content: "Hello!")
])

for try await event in node.inference.chatStream(request) {
    switch event {
    case .tokenDelta(_, let delta):
        print(delta, terminator: "")
    case .completed:
        print()
    default:
        break
    }
}
```

Local serving follows the same lifecycle:

```swift
let served = try await node.serving.load(
    "Qwen2.5-3B-Instruct-Q4_K_M",
    options: LoadModelOptions(devicePolicy: .auto)
)
defer {
    Task {
        if let instanceId = served.instanceId {
            try? await node.serving.unloadInstance(
                instanceId,
                options: UnloadModelOptions(drainTimeoutMs: 1_000, force: false)
            )
        }
        try? await node.stop()
    }
}
```

Typed SDK failures are exposed as `MeshError`, an alias for the generated UniFFI
error enum. Handle `MeshError.ServingUnsupported` when local serving is not
available for the current target or native artifact.

## Local Inference Example

The Swift example uses the same real UniFFI-backed SDK path that apps use for
model management, serving load/unload, and inference. On macOS, run it directly
from the repo:

```bash
./sdk/swift/scripts/build-xcframework.sh
scripts/package-native-sdk.sh \
  --backend metal \
  --target aarch64-apple-darwin \
  --out dist/native-sdk

MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR=dist/native-sdk/meshllm-native-darwin-aarch64-metal \
MESH_SDK_MODEL_REF=Qwen2.5-3B-Instruct-Q4_K_M \
swift run --package-path sdk/swift/example/MeshExampleApp
```

Useful environment overrides:

- `MESH_SDK_MODEL_REF` — catalog, Hugging Face, or local model reference to download/load.
- `MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR` — verified `meshllm-native-*` artifact directory for packaged local serving.
- `MESH_SDK_CACHE_DIR` — Hugging Face cache location.
- `MESH_SDK_RUNTIME_DIR` — runtime scratch directory.
- `MESH_SDK_SKIP_DOWNLOAD=1` — skip `node.models.download` when the model is already installed.
- `MESH_SDK_PROMPT` — prompt text for the local inference request.

The generated XCFramework is built with embedded serving support for Apple
targets. `build-host-macos-xcframework.sh` remains as a faster macOS-only smoke
artifact for local development; it is not the platform SDK contract.

## Platform Status

| Target | Mesh inference | Model management | Local serving |
|---|---:|---:|---:|
| macOS | yes | yes | yes, validated with the host Metal framework |
| Mac Catalyst | yes | yes | planned validation |
| iOS | yes | limited by app filesystem policy | no |

Targets without validated local serving must throw `MeshError.ServingUnsupported`
instead of silently degrading to a fake implementation.

## App Store Export Compliance

### Encryption

mesh-llm uses QUIC (via iroh) for transport, which uses TLS 1.3. This constitutes use of encryption.

**Required**: Set `ITSAppUsesNonExemptEncryption = YES` in your app's `Info.plist`.

If your app qualifies for an exemption (e.g., uses only standard encryption), you may set `ITSAppUsesNonExemptEncryption = NO` and provide justification.

### Privacy Manifest

The MeshLLM FFI XCFramework includes a `PrivacyInfo.xcprivacy` manifest in each
framework slice declaring:
- `NSPrivacyTracking = false` (no tracking)
- No data collection
- Required-reason API declarations for native file metadata, disk-capacity,
  and elapsed-time APIs used by the embedded runtime

This manifest is embedded inside each `.framework` bundle in the XCFramework, satisfying Apple's requirement since Spring 2024.

### Entitlements

No special entitlements are required. mesh-llm uses standard POSIX sockets via iroh/quinn — no `com.apple.security.network.client` entitlement is needed for macOS (it's allowed by default).

For iOS, network access is allowed by default. No special entitlements needed.

### App Store Submission Checklist

- [ ] Set `ITSAppUsesNonExemptEncryption` in `Info.plist`
- [ ] Verify `PrivacyInfo.xcprivacy` is embedded in XCFramework (run `find MeshLLMFFI.xcframework -name PrivacyInfo.xcprivacy`)
- [ ] Run `scripts/verify-swift-release-artifact.sh dist/MeshLLMFFI.xcframework.zip`
- [ ] No subprocess spawning (mesh-llm SDK never calls `Process()`)
- [ ] No filesystem access for credentials (pass keys via constructor)
- [ ] Implement `reconnect()` in `UIApplication.willEnterForegroundNotification` observer

## iOS Backgrounding

Register for foreground notifications to reconnect after backgrounding:

```swift
NotificationCenter.default.addObserver(
    forName: UIApplication.willEnterForegroundNotification,
    object: nil,
    queue: .main
) { _ in
    Task {
        try? await node.reconnect()
    }
}
```
