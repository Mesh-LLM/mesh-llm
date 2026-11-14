package ai.meshllm

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long

// An elastic IO view leaves close capacity available even when every reader worker blocks.
@OptIn(ExperimentalCoroutinesApi::class)
private val statusCloseDispatcher = Dispatchers.IO.limitedParallelism(1)

sealed interface NodeLifecycle {
    data object Loading : NodeLifecycle
    data object Downloading : NodeLifecycle
    data object Ready : NodeLifecycle
    data object Stopped : NodeLifecycle
    data class Failed(val error: String) : NodeLifecycle
}

data class NodeDownloadProgress(val file: String, val downloadedBytes: Long, val totalBytes: Long?)
data class ModelCapacity(val model: String, val contextPerLane: List<Long>, val lastSuccessUnixMs: Long?)

data class NativeNodeStatus(
    val state: NodeLifecycle,
    val servingModels: List<String>,
    val models: List<MeshModel>,
    /** Native served descriptors preserve optional identity and metadata facts. */
    val localModels: List<JsonElement>,
    val capacities: List<ModelCapacity>,
    val download: NodeDownloadProgress?,
) {
    companion object {
        internal fun decode(source: String): NativeNodeStatus {
            val value = Json.parseToJsonElement(source).jsonObject
            val state = value.getValue("state").jsonObject
            val phase = state.getValue("phase").jsonPrimitive.content
            val lifecycle = when (phase) {
                "loading" -> NodeLifecycle.Loading
                "downloading" -> NodeLifecycle.Downloading
                "ready" -> NodeLifecycle.Ready
                "stopped" -> NodeLifecycle.Stopped
                "failed" -> NodeLifecycle.Failed(state.getValue("error").jsonPrimitive.content)
                else -> error("Unknown node lifecycle: $phase")
            }
            val download = value["download"]?.takeUnless { it == JsonNull }?.jsonObject
            return NativeNodeStatus(
                lifecycle,
                value.getValue("serving_models").jsonArray.map { it.jsonPrimitive.content },
                value.getValue("models").jsonArray.map { Models.decodeModel(it.jsonObject) },
                value.getValue("local_models").jsonArray.toList(),
                value.getValue("capacities").jsonArray.map { item ->
                    val capacity = item.jsonObject
                    ModelCapacity(capacity.getValue("model").jsonPrimitive.content,
                        capacity.getValue("context_per_lane").jsonArray.map { it.jsonPrimitive.long },
                        capacity["last_success_unix_ms"]?.takeUnless { it == JsonNull }?.jsonPrimitive?.long)
                },
                download?.let { NodeDownloadProgress(it.getValue("file").jsonPrimitive.content,
                    it.getValue("downloaded_bytes").jsonPrimitive.long,
                    it["total_bytes"]?.takeUnless { it == JsonNull }?.jsonPrimitive?.long) },
            )
        }
    }
}

/** Current state and coalesced updates for one node generation. Close when finished. */
class StatusEvents internal constructor(
    private val nextJson: suspend () -> String?,
    private val closeNative: () -> Unit,
) {
    suspend fun nextEvent(): NativeNodeStatus? = suspendCancellableCoroutine { continuation ->
        val readerScope = CoroutineScope(Dispatchers.IO)
        continuation.invokeOnCancellation {
            // Closing must run independently of the synchronous FFI reader's cancellation join.
            CoroutineScope(statusCloseDispatcher).launch {
                try { closeNative() } finally { readerScope.cancel() }
            }
        }
        readerScope.launch {
            try {
                continuation.resumeWith(runCatching { nextJson()?.let { NativeNodeStatus.decode(it) } })
            } finally { readerScope.cancel() }
        }
    }
    suspend fun close(): Unit = withContext(statusCloseDispatcher + NonCancellable) { closeNative() }

    fun events(): Flow<NativeNodeStatus> = callbackFlow {
        // A blocking native read needs close to run outside the collector's cancellation join.
        val nativeScope = CoroutineScope(Dispatchers.IO)
        nativeScope.launch {
            try {
                while (true) { send(nextEvent() ?: break) }
                channel.close()
            } catch (error: Throwable) { channel.close(error) }
        }
        awaitClose {
            CoroutineScope(statusCloseDispatcher).launch {
                try { closeNative() } finally { nativeScope.cancel() }
            }
        }
    }
}
