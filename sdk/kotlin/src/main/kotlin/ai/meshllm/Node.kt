package ai.meshllm

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext
import uniffi.mesh_ffi.ChatMessageNative
import uniffi.mesh_ffi.ChatRequestNative
import uniffi.mesh_ffi.ClientEvent
import uniffi.mesh_ffi.ClientStatus
import uniffi.mesh_ffi.EventListener as FfiEventListener
import uniffi.mesh_ffi.MeshClientHandleInterface
import uniffi.mesh_ffi.MeshNodeHandleInterface
import uniffi.mesh_ffi.ModelNative
import uniffi.mesh_ffi.ResponsesRequestNative
import uniffi.mesh_ffi.createAutoClient as ffiCreateAutoClient
import uniffi.mesh_ffi.createAutoNode as ffiCreateAutoNode
import uniffi.mesh_ffi.createClient as ffiCreateClient
import uniffi.mesh_ffi.createNode as ffiCreateNode
import uniffi.mesh_ffi.discoverPublicMeshes as ffiDiscoverPublicMeshes

typealias CapabilityLevel = uniffi.mesh_ffi.CapabilityLevel
typealias CleanupPolicy = uniffi.mesh_ffi.CleanupPolicy
typealias CleanupResult = uniffi.mesh_ffi.CleanupResult
typealias DeleteModelOptions = uniffi.mesh_ffi.DeleteModelOptions
typealias DeleteModelResult = uniffi.mesh_ffi.DeleteModelResult
typealias DevicePolicy = uniffi.mesh_ffi.DevicePolicy
typealias DownloadedModel = uniffi.mesh_ffi.DownloadedModel
typealias MeshException = uniffi.mesh_ffi.FfiException
typealias InstalledModel = uniffi.mesh_ffi.InstalledModel
typealias LoadModelOptions = uniffi.mesh_ffi.LoadModelOptions
typealias ModelCacheStatus = uniffi.mesh_ffi.ModelCacheStatus
typealias ModelCapabilities = uniffi.mesh_ffi.ModelCapabilities
typealias ModelDetails = uniffi.mesh_ffi.ModelDetails
typealias ModelKind = uniffi.mesh_ffi.ModelKind
typealias ModelSearchQuery = uniffi.mesh_ffi.ModelSearchQuery
typealias ModelSource = uniffi.mesh_ffi.ModelSource
typealias ModelSummary = uniffi.mesh_ffi.ModelSummary
typealias PrunePolicy = uniffi.mesh_ffi.PrunePolicy
typealias PruneResult = uniffi.mesh_ffi.PruneResult
typealias PublicMesh = uniffi.mesh_ffi.PublicMesh
typealias PublicMeshQuery = uniffi.mesh_ffi.PublicMeshQuery
typealias ServedModel = uniffi.mesh_ffi.ServedModel
typealias ServingModelState = uniffi.mesh_ffi.ServingModelState
typealias ServingStatus = uniffi.mesh_ffi.ServingStatus
typealias UnloadModelOptions = uniffi.mesh_ffi.UnloadModelOptions
typealias UnloadTarget = uniffi.mesh_ffi.UnloadTarget

@JvmInline
value class InviteToken(val value: String)

data class Model(val id: String, val name: String)

data class ChatMessage(val role: String, val content: String)

data class ChatRequest(val model: String, val messages: List<ChatMessage>)

data class ResponsesRequest(val model: String, val input: String)

data class Status(val connected: Boolean, val peerCount: ULong)

@JvmInline
value class RequestId(val value: String)

sealed class Event {
    object Connecting : Event()
    data class Joined(val nodeId: String) : Event()
    data class ModelsUpdated(val models: List<Model>) : Event()
    data class TokenDelta(val requestId: RequestId, val delta: String) : Event()
    data class Completed(val requestId: RequestId) : Event()
    data class Failed(val requestId: RequestId, val error: String) : Event()
    data class Disconnected(val reason: String) : Event()
}

fun interface EventListener {
    fun onEvent(event: Event)
}

class Client(private val handle: MeshClientHandleInterface) {
    val inference = Inference(handle)

    constructor(
        inviteToken: InviteToken,
        ownerKeypairBytesHex: String,
    ) : this(ffiCreateClient(ownerKeypairBytesHex, inviteToken.value))

    suspend fun start(): Unit = withContext(Dispatchers.IO) { handle.start() }

    suspend fun stop(): Unit = withContext(Dispatchers.IO) { handle.stop() }

    suspend fun reconnect(): Unit = withContext(Dispatchers.IO) { handle.reconnect() }

    suspend fun status(): Status = withContext(Dispatchers.IO) { handle.status().toStatus() }

    class Inference(private val handle: MeshClientHandleInterface) {
        suspend fun listModels(): List<Model> =
            withContext(Dispatchers.IO) { handle.inferenceListModels().map { it.toModel() } }

        fun chat(request: ChatRequest, listener: EventListener): RequestId {
            val bridge = object : FfiEventListener {
                override fun onEvent(event: ClientEvent) = listener.onEvent(event.toEvent())
            }
            return RequestId(handle.chat(request.toNative(), bridge))
        }

        fun responses(request: ResponsesRequest, listener: EventListener): RequestId {
            val bridge = object : FfiEventListener {
                override fun onEvent(event: ClientEvent) = listener.onEvent(event.toEvent())
            }
            return RequestId(handle.responses(request.toNative(), bridge))
        }

        fun cancel(requestId: RequestId) = handle.cancel(requestId.value)

        fun chatFlow(request: ChatRequest): Flow<Event> = callbackFlow {
            var requestId: RequestId? = null
            var terminalRequestId: RequestId? = null
            requestId = chat(request) { event ->
                trySend(event)
                val currentRequestId = requestId
                if (currentRequestId != null && event.isTerminalFor(currentRequestId)) {
                    terminalRequestId = currentRequestId
                    close()
                } else if (currentRequestId == null) {
                    terminalRequestId = event.terminalRequestId()
                }
            }
            if (requestId == terminalRequestId) {
                close()
            }
            awaitClose {
                val currentRequestId = requestId ?: return@awaitClose
                if (terminalRequestId != currentRequestId) {
                    cancel(currentRequestId)
                }
            }
        }

        fun responsesFlow(request: ResponsesRequest): Flow<Event> = callbackFlow {
            var requestId: RequestId? = null
            var terminalRequestId: RequestId? = null
            requestId = responses(request) { event ->
                trySend(event)
                val currentRequestId = requestId
                if (currentRequestId != null && event.isTerminalFor(currentRequestId)) {
                    terminalRequestId = currentRequestId
                    close()
                } else if (currentRequestId == null) {
                    terminalRequestId = event.terminalRequestId()
                }
            }
            if (requestId == terminalRequestId) {
                close()
            }
            awaitClose {
                val currentRequestId = requestId ?: return@awaitClose
                if (terminalRequestId != currentRequestId) {
                    cancel(currentRequestId)
                }
            }
        }
    }

    companion object {
        suspend fun connectPublic(
            ownerKeypairBytesHex: String,
            query: PublicMeshQuery = PublicMeshQuery(
                model = null,
                minVramGb = null,
                region = null,
                targetName = null,
                relays = emptyList(),
            ),
        ): Client = withContext(Dispatchers.IO) {
            Client(ffiCreateAutoClient(ownerKeypairBytesHex, query))
        }
    }
}

class Node(private val handle: MeshNodeHandleInterface) {
    val inference = Inference(handle)
    val models = Models(handle)
    val serving = Serving(handle)

    constructor(
        inviteToken: InviteToken,
        ownerKeypairBytesHex: String,
        cacheDir: String? = null,
        runtimeDir: String? = null,
    ) : this(ffiCreateNode(ownerKeypairBytesHex, inviteToken.value, cacheDir, runtimeDir, true))

    suspend fun start(): Unit = withContext(Dispatchers.IO) { handle.start() }

    suspend fun stop(): Unit = withContext(Dispatchers.IO) { handle.stop() }

    suspend fun reconnect(): Unit = withContext(Dispatchers.IO) { handle.reconnect() }

    suspend fun status(): Status = withContext(Dispatchers.IO) { handle.status().toStatus() }

    class Inference(private val handle: MeshNodeHandleInterface) {
        suspend fun listModels(): List<Model> =
            withContext(Dispatchers.IO) { handle.inferenceListModels().map { it.toModel() } }

        fun chat(request: ChatRequest, listener: EventListener): RequestId {
            val bridge = object : FfiEventListener {
                override fun onEvent(event: ClientEvent) = listener.onEvent(event.toEvent())
            }
            return RequestId(handle.chat(request.toNative(), bridge))
        }

        fun responses(request: ResponsesRequest, listener: EventListener): RequestId {
            val bridge = object : FfiEventListener {
                override fun onEvent(event: ClientEvent) = listener.onEvent(event.toEvent())
            }
            return RequestId(handle.responses(request.toNative(), bridge))
        }

        fun cancel(requestId: RequestId) = handle.cancel(requestId.value)

        fun chatFlow(request: ChatRequest): Flow<Event> = callbackFlow {
            var requestId: RequestId? = null
            var terminalRequestId: RequestId? = null
            requestId = chat(request) { event ->
                trySend(event)
                val currentRequestId = requestId
                if (currentRequestId != null && event.isTerminalFor(currentRequestId)) {
                    terminalRequestId = currentRequestId
                    close()
                } else if (currentRequestId == null) {
                    terminalRequestId = event.terminalRequestId()
                }
            }
            if (requestId == terminalRequestId) {
                close()
            }
            awaitClose {
                val currentRequestId = requestId ?: return@awaitClose
                if (terminalRequestId != currentRequestId) {
                    cancel(currentRequestId)
                }
            }
        }

        fun responsesFlow(request: ResponsesRequest): Flow<Event> = callbackFlow {
            var requestId: RequestId? = null
            var terminalRequestId: RequestId? = null
            requestId = responses(request) { event ->
                trySend(event)
                val currentRequestId = requestId
                if (currentRequestId != null && event.isTerminalFor(currentRequestId)) {
                    terminalRequestId = currentRequestId
                    close()
                } else if (currentRequestId == null) {
                    terminalRequestId = event.terminalRequestId()
                }
            }
            if (requestId == terminalRequestId) {
                close()
            }
            awaitClose {
                val currentRequestId = requestId ?: return@awaitClose
                if (terminalRequestId != currentRequestId) {
                    cancel(currentRequestId)
                }
            }
        }
    }

    class Models(private val handle: MeshNodeHandleInterface) {
        suspend fun recommended(): List<ModelSummary> =
            withContext(Dispatchers.IO) { handle.recommendedModels() }

        suspend fun search(query: ModelSearchQuery): List<ModelSummary> =
            withContext(Dispatchers.IO) { handle.searchModels(query) }

        suspend fun show(modelRef: String): ModelDetails =
            withContext(Dispatchers.IO) { handle.showModel(modelRef) }

        suspend fun installed(): List<InstalledModel> =
            withContext(Dispatchers.IO) { handle.installedModels() }

        suspend fun cacheStatus(): ModelCacheStatus =
            withContext(Dispatchers.IO) { handle.modelCacheStatus() }

        suspend fun download(modelRef: String): DownloadedModel =
            withContext(Dispatchers.IO) { handle.downloadModel(modelRef) }

        suspend fun delete(modelRef: String, options: DeleteModelOptions): DeleteModelResult =
            withContext(Dispatchers.IO) { handle.deleteModel(modelRef, options) }

        suspend fun cleanup(policy: CleanupPolicy): CleanupResult =
            withContext(Dispatchers.IO) { handle.cleanupModels(policy) }

        suspend fun pruneDerivedCache(policy: PrunePolicy): PruneResult =
            withContext(Dispatchers.IO) { handle.pruneDerivedCache(policy) }
    }

    class Serving(private val handle: MeshNodeHandleInterface) {
        suspend fun status(): ServingStatus =
            withContext(Dispatchers.IO) { handle.servingStatus() }

        suspend fun servedModels(): List<ServedModel> =
            withContext(Dispatchers.IO) { handle.servedModels() }

        suspend fun load(modelRef: String, options: LoadModelOptions): ServedModel =
            withContext(Dispatchers.IO) { handle.loadServingModel(modelRef, options) }

        suspend fun unload(target: UnloadTarget, options: UnloadModelOptions): Unit =
            withContext(Dispatchers.IO) { handle.unloadServingModel(target, options) }

        suspend fun unloadModel(modelId: String, options: UnloadModelOptions): Unit =
            withContext(Dispatchers.IO) { handle.unloadServingModelById(modelId, options) }

        suspend fun unloadInstance(instanceId: String, options: UnloadModelOptions): Unit =
            withContext(Dispatchers.IO) { handle.unloadServingInstance(instanceId, options) }

        suspend fun setDevicePolicy(policy: DevicePolicy): Unit =
            withContext(Dispatchers.IO) { handle.setDevicePolicy(policy) }
    }

    companion object {
        suspend fun discoverPublicMeshes(
            query: PublicMeshQuery = PublicMeshQuery(
                model = null,
                minVramGb = null,
                region = null,
                targetName = null,
                relays = emptyList(),
            ),
        ): List<PublicMesh> = withContext(Dispatchers.IO) {
            ffiDiscoverPublicMeshes(query)
        }

        suspend fun connectPublic(
            ownerKeypairBytesHex: String,
            query: PublicMeshQuery = PublicMeshQuery(
                model = null,
                minVramGb = null,
                region = null,
                targetName = null,
                relays = emptyList(),
            ),
        ): Node = withContext(Dispatchers.IO) {
            Node(ffiCreateAutoNode(ownerKeypairBytesHex, query))
        }
    }
}

private fun ModelNative.toModel() = Model(id = id, name = name)

private fun ClientStatus.toStatus() = Status(connected = connected, peerCount = peerCount)

private fun ChatMessage.toNative() = ChatMessageNative(role = role, content = content)

private fun ChatRequest.toNative() =
    ChatRequestNative(model = model, messages = messages.map { it.toNative() })

private fun ResponsesRequest.toNative() = ResponsesRequestNative(model = model, input = input)

private fun ClientEvent.toEvent(): Event =
    when (this) {
        is ClientEvent.Connecting -> Event.Connecting
        is ClientEvent.Joined -> Event.Joined(nodeId = nodeId)
        is ClientEvent.ModelsUpdated -> Event.ModelsUpdated(models = models.map { it.toModel() })
        is ClientEvent.TokenDelta -> Event.TokenDelta(requestId = RequestId(requestId), delta = delta)
        is ClientEvent.Completed -> Event.Completed(requestId = RequestId(requestId))
        is ClientEvent.Failed -> Event.Failed(requestId = RequestId(requestId), error = error)
        is ClientEvent.Disconnected -> Event.Disconnected(reason = reason)
    }

private fun Event.isTerminalFor(requestId: RequestId): Boolean =
    when (this) {
        is Event.Completed -> this.requestId == requestId
        is Event.Failed -> this.requestId == requestId
        else -> false
    }

private fun Event.terminalRequestId(): RequestId? =
    when (this) {
        is Event.Completed -> requestId
        is Event.Failed -> requestId
        else -> null
    }
