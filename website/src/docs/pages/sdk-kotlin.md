---
title: Java, Kotlin, and Android SDK
---

# Java, Kotlin, and Android SDK

The Android/JVM SDK is published as `ai.meshllm:meshllm-android`. The public API is authored in Kotlin and uses coroutines and `Flow`; Java applications can consume the same AAR, while the examples here use Kotlin because it matches the native API most closely.

## Install

Add the GitHub Packages Maven registry and the SDK dependency:

```kotlin
repositories {
    maven {
        url = uri("https://maven.pkg.github.com/Mesh-LLM/mesh-llm")
        credentials {
            username = providers.gradleProperty("gpr.user")
                .orElse(System.getenv("GITHUB_ACTOR")).get()
            password = providers.gradleProperty("gpr.key")
                .orElse(System.getenv("GITHUB_TOKEN")).get()
        }
    }
}

dependencies {
    implementation("ai.meshllm:meshllm-android:{{ site.sdkVersion }}")
}
```

For local development, build the AAR with `./gradlew assembleAar` from `sdk/kotlin`.

## Connect as a client

```kotlin
import ai.meshllm.ChatMessage
import ai.meshllm.ChatRequest
import ai.meshllm.Client
import ai.meshllm.Event
import ai.meshllm.InviteToken
import kotlinx.coroutines.flow.collect
import uniffi.mesh_ffi.generateOwnerKeypairHex

val client = Client(
    InviteToken(System.getenv("MESH_INVITE_TOKEN")),
    generateOwnerKeypairHex(),
)

client.start()
try {
    val models = client.inference.listModels()
    client.inference.chatFlow(
        ChatRequest(
            models.first().id,
            listOf(ChatMessage("user", "Say hello from Kotlin.")),
        ),
    ).collect { event ->
        if (event is Event.TokenDelta) print(event.delta)
        if (event is Event.Completed) println()
    }
} finally {
    client.stop()
}
```

For public mesh discovery, use `Client.connectPublic(ownerKeypair, PublicMeshQuery(...))`. Keep the owner keypair in app storage so reconnects retain the same identity.

## JVM local serving

Local serving is currently supported for validated JVM targets with a matching native runtime. Resolve a bundled or downloadable runtime first:

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

Then use `Node` for model management and serving:

```kotlin
import ai.meshllm.ChatMessage
import ai.meshllm.ChatRequest
import ai.meshllm.DevicePolicy
import ai.meshllm.InviteToken
import ai.meshllm.LoadModelOptions
import ai.meshllm.Node
import ai.meshllm.UnloadModelOptions
import uniffi.mesh_ffi.generateOwnerKeypairHex

val node = Node(
    InviteToken(System.getenv("MESH_INVITE_TOKEN")),
    generateOwnerKeypairHex(),
)

node.start()
try {
    val modelRef = System.getenv("MESH_SDK_MODEL_REF") ?: "Qwen2.5-3B-Instruct-Q4_K_M"
    node.models.download(modelRef)
    val served = node.serving.load(modelRef, LoadModelOptions(DevicePolicy.Auto))
    node.inference.chatFlow(
        ChatRequest(
            served.modelId,
            listOf(ChatMessage("user", "Say hello from JVM serving.")),
        ),
    ).collect { event ->
        if (event is Event.TokenDelta) print(event.delta)
        if (event is Event.Completed) println()
    }
    node.serving.unloadModel(
        served.modelId,
        UnloadModelOptions(drainTimeoutMs = 1_000UL, force = false),
    )
} finally {
    node.stop()
}
```

Catch `MeshException.ServingUnsupported` and show an actionable message when the selected JVM/native artifact does not provide local serving.

## Android support

Android apps can use the SDK for mesh inference and model-management APIs. Local serving is not currently advertised for Android, so an Android app should use `Client` to reach a serving node or display the typed unsupported error rather than trying to load a desktop runtime into the APK.

When Android serving is validated, the packaging contract will need an ABI-specific native runtime, app-compatible storage, lifecycle handling, and a verified memory budget. Keep those concerns behind the SDK rather than copying desktop runtime paths into the app.

## Console assets

JVM/Android packages may include the console as resources. Extract packaged resources before starting the static console server:

```kotlin
val console = node.startConsole(ConsoleAssets.packagedOptions(port = 3131u.toUShort()))
println(console.url())
```

Treat the console as optional and bind it only to an interface your app can protect.
