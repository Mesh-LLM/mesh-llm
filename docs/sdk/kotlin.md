# Kotlin/Android SDK

Use the GitHub Packages Maven registry for `Mesh-LLM/mesh-llm`.

## Install

```text
ai.meshllm:meshllm-android:<version>
```

Configure the Maven repository:

```kotlin
repositories {
    maven {
        url = uri("https://maven.pkg.github.com/Mesh-LLM/mesh-llm")
        credentials {
            username = providers.gradleProperty("gpr.user")
                .orElse(System.getenv("GITHUB_ACTOR"))
                .get()
            password = providers.gradleProperty("gpr.key")
                .orElse(System.getenv("GITHUB_TOKEN"))
                .get()
        }
    }
}
```

## Client: Public Mesh

```kotlin
import ai.meshllm.ChatMessage
import ai.meshllm.ChatRequest
import ai.meshllm.Client
import ai.meshllm.Event
import ai.meshllm.PublicMeshQuery
import kotlinx.coroutines.flow.collect
import uniffi.mesh_ffi.generateOwnerKeypairHex

val ownerKeypair = generateOwnerKeypairHex()
val client = Client.connectPublic(
    ownerKeypair,
    PublicMeshQuery(
        model = "Qwen3",
        minVramGb = null,
        region = null,
        targetName = null,
        relays = emptyList(),
    ),
)

client.start()
val publicModels = client.inference.listModels()
client.inference.chatFlow(
    ChatRequest(publicModels.first().id, listOf(ChatMessage("user", "Say hello from a public mesh."))),
).collect(::printToken)
client.stop()
```

## Client: Private Mesh

```kotlin
import ai.meshllm.Client
import ai.meshllm.InviteToken
import uniffi.mesh_ffi.generateOwnerKeypairHex

val ownerKeypair = generateOwnerKeypairHex()
val client = Client(InviteToken(System.getenv("MESH_PRIVATE_INVITE")), ownerKeypair)

client.start()
val models = client.inference.listModels()
client.inference.chatFlow(
    ChatRequest(models.first().id, listOf(ChatMessage("user", "Say hello from a private mesh."))),
).collect(::printToken)
client.stop()
```

## Inference Helper

```kotlin
fun printToken(event: Event) {
    if (event is Event.TokenDelta) print(event.delta)
    if (event is Event.Completed) println()
}
```

## Serving: Install Runtime

Resolve or install the native runtime before local serving:

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
println("using ${runtime.nativeRuntimeId} from ${runtime.path}")
```

## Serving: Public Mesh

```kotlin
import ai.meshllm.ChatMessage
import ai.meshllm.ChatRequest
import ai.meshllm.DevicePolicy
import ai.meshllm.InviteToken
import ai.meshllm.LoadModelOptions
import ai.meshllm.Node
import ai.meshllm.UnloadModelOptions

val ownerKeypair = generateOwnerKeypairHex()
val node = Node(InviteToken(System.getenv("MESH_PUBLIC_INVITE")), ownerKeypair)
node.start()

val modelRef = System.getenv("MESH_SDK_MODEL_REF") ?: "Qwen2.5-3B-Instruct-Q4_K_M"
node.models.download(modelRef)
val served = node.serving.load(modelRef, LoadModelOptions(DevicePolicy.Auto))
node.inference.chatFlow(
    ChatRequest(served.modelId, listOf(ChatMessage("user", "Say hello from a public serving node."))),
).collect(::printToken)
node.serving.unloadModel(served.modelId, UnloadModelOptions(drainTimeoutMs = 1_000UL, force = false))
node.stop()
```

## Serving: Private Mesh

Private mesh serving uses the same lifecycle with `MESH_PRIVATE_INVITE`:

```kotlin
val ownerKeypair = generateOwnerKeypairHex()
val node = Node(InviteToken(System.getenv("MESH_PRIVATE_INVITE")), ownerKeypair)
```

## JVM Example

```bash
scripts/package-native-runtime.sh \
  --backend metal \
  --target aarch64-apple-darwin \
  --out dist/native-runtimes

MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR=dist/native-runtimes/meshllm-native-runtime-darwin-aarch64-metal \
MESH_SDK_MODEL_REF=Qwen2.5-3B-Instruct-Q4_K_M \
./gradlew --no-daemon run -p sdk/kotlin/example/example-jvm
```

## Console Assets

Published Kotlin packages that advertise console support include the built web
console as JVM resources. Use the packaged resource helper in normal package
usage:

```kotlin
val options = ConsoleAssets.packagedOptions()
```
