package ai.meshllm.example

import ai.meshllm.ChatMessage
import ai.meshllm.ChatRequest
import ai.meshllm.Event
import ai.meshllm.InviteToken
import ai.meshllm.LoadModelOptions
import ai.meshllm.NativeRuntime
import ai.meshllm.Node
import ai.meshllm.UnloadModelOptions
import kotlinx.coroutines.runBlocking
import uniffi.mesh_ffi.DevicePolicy
import uniffi.mesh_ffi.UnloadTarget
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

fun main(args: Array<String>) = runBlocking {
    val modelRef = System.getenv("MESH_SDK_MODEL_REF")
    val inviteToken = args.firstOrNull { !it.startsWith("--") } ?: modelRef?.let { "local-kotlin-example" } ?: run {
        System.err.println("Usage: ExampleMain <invite_token>")
        System.err.println("Or set MESH_SDK_MODEL_REF to run local serving.")
        System.err.println("Set MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR to a verified meshllm-native-* artifact.")
        return@runBlocking
    }

    val runtime = NativeRuntime.resolve(
        ai.meshllm.NativeRuntimeResolveOptions(
            artifactDir = System.getenv("MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR")?.let(::File),
            cacheDir = System.getenv("MESH_LLM_NATIVE_RUNTIME_CACHE_DIR")?.let(::File),
            allowDownload = System.getenv("MESH_SDK_RUNTIME_ALLOW_DOWNLOAD") == "1",
        )
    )
    println("[runtime] artifact=${runtime.nativeRuntimeId} path=${runtime.path}")
    // Generate an ephemeral owner keypair for the example. In a real app this
    // must be persisted across launches.
    val ownerKeypairHex = uniffi.mesh_ffi.generateOwnerKeypairHex()
    val node = Node(
        InviteToken(inviteToken),
        ownerKeypairHex,
        cacheDir = System.getenv("MESH_SDK_CACHE_DIR"),
        runtimeDir = System.getenv("MESH_SDK_RUNTIME_DIR"),
    )
    val recommended = node.models.recommended()
    val serving = node.serving.status()
    println("[node] recommended_models=${recommended.size} serving_enabled=${serving.enabled}")

    try {
        node.start()
        println("[connected]")

        val localUnloadTarget: UnloadTarget?
        val selectedModel = if (modelRef != null) {
            val served = runLocalServingExample(node, modelRef)
            localUnloadTarget = served.instanceId?.let { UnloadTarget.Instance(it) }
                ?: UnloadTarget.Model(served.modelId)
            served.modelId
        } else {
            localUnloadTarget = null
            val models = waitForModels(node)
            println("[models] N=${models.size}")
            check(models.isNotEmpty()) { "mesh reported no models" }
            System.getenv("MESH_SDK_MODEL_ID") ?: models.first().id
        }

        runChat(node, selectedModel)
        if (localUnloadTarget != null) {
            node.serving.unload(
                localUnloadTarget,
                UnloadModelOptions(drainTimeoutMs = 5_000UL, force = false),
            )
            println("[serving] unloaded model=$selectedModel")
        }
    } finally {
        node.stop()
        println("[disconnect] ok")
    }
}

private suspend fun runLocalServingExample(node: Node, modelRef: String): ai.meshllm.ServedModel {
    if (System.getenv("MESH_SDK_SKIP_DOWNLOAD") != "1") {
        val downloaded = node.models.download(modelRef)
        println("[download] model_ref=${downloaded.modelRef} path=${downloaded.primaryPath ?: downloaded.paths.firstOrNull()}")
    }

    val served = node.serving.load(modelRef, LoadModelOptions(DevicePolicy.Auto))
    println("[serving] model=${served.modelRef} id=${served.modelId} instance=${served.instanceId}")

    val models = waitForModels(node)
    println("[models] N=${models.size}")
    check(models.any { it.id == served.modelId }) { "loaded model was not listed for inference" }

    return served
}

private fun runChat(node: Node, selectedModel: String) {
    val chatRequest = ChatRequest(
        model = selectedModel,
        messages = listOf(ChatMessage(role = "user", content = System.getenv("MESH_SDK_PROMPT") ?: "hello")),
    )

    val latch = CountDownLatch(1)
    var firstTokenEmitted = false
    var completed = false
    var failed: String? = null
    val chatStartMs = System.currentTimeMillis()

    node.inference.chat(chatRequest) { event ->
        when (event) {
            is Event.TokenDelta -> {
                if (!firstTokenEmitted) {
                    firstTokenEmitted = true
                    val elapsedMs = System.currentTimeMillis() - chatStartMs
                    println("[chat] first_token_ms=$elapsedMs")
                }
            }
            is Event.Completed -> {
                completed = true
                latch.countDown()
            }
            is Event.Failed -> {
                failed = event.error
                latch.countDown()
            }
            else -> Unit
        }
    }

    check(latch.await(60, TimeUnit.SECONDS)) { "chat timed out waiting for completion" }
    check(failed == null) { "chat failed: $failed" }
    check(firstTokenEmitted) { "chat emitted no token deltas" }
    check(completed) { "chat never completed" }
    println("[chat] done")
}

private suspend fun waitForModels(node: Node): List<ai.meshllm.Model> {
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
    while (System.nanoTime() < deadline) {
        val models = node.inference.listModels()
        if (models.isNotEmpty()) {
            return models
        }
        Thread.sleep(250)
    }
    return emptyList()
}
