package ai.meshllm

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import uniffi.mesh_ffi.MeshNodeHandleInterface
import uniffi.mesh_ffi.OpenAiStreamEventNative
import uniffi.mesh_ffi.OpenAiStreamListener
import uniffi.mesh_ffi.createNode

/** The three supported embedded-node roles. */
enum class NodeMode(val wireValue: String) {
    CLIENT("client"),
    SERVE("serve"),
    COMBINED("combined"),
}

data class NodeStatus(
    val running: Boolean,
    val mode: NodeMode,
    val apiBaseUrl: String,
    val consoleUrl: String,
    val payloadJson: String,
)

data class Model(val id: String, val name: String, val contextLength: UInt?)

data class OpenAIResponse(
    val statusCode: UShort,
    val contentType: String?,
    val body: String,
) {
    fun json(): JsonElement = Json.parseToJsonElement(body)
}

sealed class OpenAIStreamEvent {
    data class Started(
        val requestId: String,
        val statusCode: UShort,
        val contentType: String?,
    ) : OpenAIStreamEvent()

    data class Sse(
        val requestId: String,
        val event: String?,
        val data: String,
        val raw: String,
    ) : OpenAIStreamEvent() {
        val isDone: Boolean get() = data == "[DONE]"
        fun json(): JsonElement? = if (isDone) null else Json.parseToJsonElement(data)
    }
}

class OpenAIStreamException(
    val requestId: String,
    val statusCode: UShort?,
    val responseBody: String?,
    message: String,
) : RuntimeException(message)

/** A real embedded Mesh LLM node with mesh inference and optional local serving. */
class Node internal constructor(private val handle: MeshNodeHandleInterface) {
    val inference = Inference(handle)

    constructor(
        mode: NodeMode = NodeMode.CLIENT,
        joinTokens: List<String> = emptyList(),
        models: List<String> = emptyList(),
        autoJoin: Boolean = false,
        ownerKeyPath: String? = null,
        apiPort: UShort = 9337u,
        consolePort: UShort = 3131u,
    ) : this(createNode(mode.wireValue, joinTokens, models, autoJoin, ownerKeyPath, apiPort, consolePort))

    suspend fun start(): Unit = withContext(Dispatchers.IO) { handle.start() }

    suspend fun stop(): Unit = withContext(Dispatchers.IO) { handle.stop() }

    suspend fun status(): NodeStatus = withContext(Dispatchers.IO) {
        val value = handle.status()
        NodeStatus(
            value.running,
            NodeMode.entries.first { it.wireValue == value.mode },
            value.apiBaseUrl,
            value.consoleUrl,
            value.payloadJson,
        )
    }

    suspend fun joinToken(token: String): Unit = withContext(Dispatchers.IO) { handle.joinToken(token) }

    class Inference(private val handle: MeshNodeHandleInterface) {
        suspend fun listModels(): List<Model> = withContext(Dispatchers.IO) {
            handle.inferenceListModels().map { Model(it.id, it.name, it.contextLength) }
        }

        suspend fun request(path: String, body: JsonObject): OpenAIResponse =
            requestJson(path, body.toString())

        suspend fun requestJson(path: String, bodyJson: String): OpenAIResponse =
            withContext(Dispatchers.IO) {
                val response = handle.openaiRequest(path, bodyJson)
                OpenAIResponse(response.statusCode, response.contentType, response.body)
            }

        suspend fun chatCompletions(body: JsonObject): OpenAIResponse =
            request("/v1/chat/completions", withStreamFlag(body, false))

        suspend fun responses(body: JsonObject): OpenAIResponse =
            request("/v1/responses", withStreamFlag(body, false))

        fun stream(path: String, bodyJson: String): Flow<OpenAIStreamEvent> = callbackFlow {
            fun deliver(value: OpenAIStreamEvent) {
                if (trySend(value).isFailure) {
                    close(IllegalStateException("MeshLLM stream consumer fell behind the event buffer"))
                }
            }
            val listener = object : OpenAiStreamListener {
                override fun onEvent(event: OpenAiStreamEventNative) {
                    when (event) {
                        is OpenAiStreamEventNative.Started -> deliver(
                            OpenAIStreamEvent.Started(event.requestId, event.statusCode, event.contentType)
                        )
                        is OpenAiStreamEventNative.Sse -> deliver(
                            OpenAIStreamEvent.Sse(event.requestId, event.eventType, event.data, event.raw)
                        )
                        is OpenAiStreamEventNative.Completed -> close()
                        is OpenAiStreamEventNative.Failed -> close(
                            OpenAIStreamException(event.requestId, event.statusCode, event.body, event.error)
                        )
                    }
                }
            }
            val requestId = withContext(Dispatchers.IO) {
                handle.openaiStream(path, bodyJson, listener)
            }
            awaitClose { handle.cancel(requestId) }
        }

        fun streamChatCompletions(bodyJson: String): Flow<OpenAIStreamEvent> =
            stream("/v1/chat/completions", withStreamFlag(Json.parseToJsonElement(bodyJson).jsonObject, true).toString())

        fun streamResponses(bodyJson: String): Flow<OpenAIStreamEvent> =
            stream("/v1/responses", withStreamFlag(Json.parseToJsonElement(bodyJson).jsonObject, true).toString())

        private fun withStreamFlag(body: JsonObject, enabled: Boolean): JsonObject =
            JsonObject(body + ("stream" to JsonPrimitive(enabled)))
    }
}
