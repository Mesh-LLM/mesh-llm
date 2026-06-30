# MeshLLM Kotlin SDK

Kotlin/Android bindings for mesh-llm model management, local serving, and mesh
inference.

The SDK usage guide, native runtime packaging notes, examples, and platform
support matrix live in [`docs/SDK.md`](../../docs/SDK.md).

## GitHub Packages

Release workflow publishes the Android AAR to this repository's GitHub Packages Maven registry as:

```text
ai.meshllm:meshllm-android:<version>
```

Add the GitHub Packages Maven repository:

```kotlin
repositories {
    maven {
        url = uri("https://maven.pkg.github.com/Mesh-LLM/mesh-llm")
        credentials {
            username = providers.gradleProperty("gpr.user").orElse(System.getenv("GITHUB_ACTOR")).get()
            password = providers.gradleProperty("gpr.key").orElse(System.getenv("GITHUB_TOKEN")).get()
        }
    }
}
```

Then depend on the SDK:

```kotlin
dependencies {
    implementation("ai.meshllm:meshllm-android:0.72.1")
}
```

## Local Development

To build the Android artifact locally:

```bash
./gradlew assembleAar
```

This writes the AAR to `sdk/kotlin/build/outputs/aar/meshllm-android.aar`.
The native libraries in the AAR are built with embedded serving enabled and
include per-ABI llama.cpp static runtime archives.

## Usage

Apps that need local serving resolve or install a native runtime before loading
a model. The resolver can use packaged artifact directories or download the
recommended runtime through the SDK native runtime manager:

```kotlin
import ai.meshllm.NativeRuntime
import ai.meshllm.NativeRuntimeResolveOptions
import java.io.File

val runtime = NativeRuntime.resolve(
    NativeRuntimeResolveOptions(
        artifactDir = System.getenv("MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR")?.let(::File),
        allowDownload = System.getenv("MESH_SDK_RUNTIME_ALLOW_DOWNLOAD") == "1",
    ),
)
println("loaded ${runtime.nativeRuntimeId} from ${runtime.path}")
```

```kotlin
import ai.meshllm.InviteToken
import ai.meshllm.Node
import uniffi.mesh_ffi.generateOwnerKeypairHex

val ownerKeypair = generateOwnerKeypairHex()
val node = Node(InviteToken("your-invite-token"), ownerKeypair)

val recommended = node.models.recommended()
val serving = node.serving.status()

node.start()
```

Local serving follows the same lifecycle:

```kotlin
val served = node.serving.load(
    "Qwen2.5-3B-Instruct-Q4_K_M",
    LoadModelOptions(DevicePolicy.Auto),
)
try {
    val models = node.inference.listModels()
    val selectedModel = models.first { it.id == served.modelId }
    node.inference.chat(
        ChatRequest(selectedModel.id, listOf(ChatMessage("user", "hello"))),
    ) { event -> println(event) }
} finally {
    val target = served.instanceId?.let { UnloadTarget.Instance(it) }
        ?: UnloadTarget.Model(served.modelId)
    node.serving.unload(target, UnloadModelOptions(drainTimeoutMs = 1_000UL, force = false))
    node.stop()
}
```

Typed SDK failures are exposed as `MeshException`, an alias for the generated
UniFFI exception hierarchy. Handle `MeshException.ServingUnsupported` when local
serving is not available for the current target or native artifact.

## Platform Status

| Target | Mesh inference | Model management | Local serving |
|---|---:|---:|---:|
| JVM macOS | yes | yes | yes with a matching `libmeshllm_ffi.dylib` |
| JVM Linux | yes | yes | yes with a matching `libmeshllm_ffi.so` |
| Android | yes | yes | planned validation |

Targets without validated local serving must throw
`MeshException.ServingUnsupported` instead of silently degrading to a fake
implementation.

## Local JVM Example

Build or download a native runtime artifact, then run the JVM example with that
artifact directory:

```bash
scripts/package-native-runtime.sh \
  --backend metal \
  --target aarch64-apple-darwin \
  --out dist/native-runtimes

MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR=dist/native-runtimes/meshllm-native-runtime-darwin-aarch64-metal \
MESH_SDK_MODEL_REF=Qwen2.5-3B-Instruct-Q4_K_M \
./gradlew --no-daemon run -p sdk/kotlin/example/example-jvm
```

## Optional Console Assets

Published Kotlin/JVM and Android packages include the built console resources
under `mesh-llm/console`. Because those resources may live inside a JAR or APK,
use `ConsoleAssets.packagedOptions()` to extract them to a filesystem directory
before starting the static console server:

```kotlin
val console = node.startConsole(ConsoleAssets.packagedOptions(port = 3131u.toUShort()))
println(console.url())
```

Release packages prepare those resources with:

```bash
scripts/package-sdk-console-assets.sh --sdk kotlin
scripts/verify-sdk-console-assets.sh --sdk kotlin
```
