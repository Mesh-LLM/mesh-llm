import Foundation

public struct InviteToken: Sendable {
    public let value: String

    public init(_ value: String) {
        self.value = value
    }
}

public struct Model: Sendable {
    public let id: String
    public let name: String
}

public struct Status: Sendable {
    public let connected: Bool
    public let peerCount: Int
}

public struct RequestId: Sendable {
    public let value: String
}

public enum Event: Sendable {
    case connecting
    case joined(nodeId: String)
    case modelsUpdated(models: [Model])
    case tokenDelta(requestId: String, delta: String)
    case completed(requestId: String)
    case failed(requestId: String, error: String)
    case disconnected(reason: String)
}

public struct ChatMessage: Sendable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

public struct ChatRequest: Sendable {
    public let model: String
    public let messages: [ChatMessage]

    public init(model: String, messages: [ChatMessage]) {
        self.model = model
        self.messages = messages
    }
}

public struct ResponsesRequest: Sendable {
    public let model: String
    public let input: String

    public init(model: String, input: String) {
        self.model = model
        self.input = input
    }
}

#if canImport(MeshLLMFFI)
public typealias MeshError = FfiError

public final class Client: @unchecked Sendable {
    private let handle: MeshClientHandle

    public let inference: Inference

    public init(
        inviteToken: InviteToken,
        ownerKeypairBytesHex: String
    ) throws {
        let handle = try createClient(
            ownerKeypairBytesHex: ownerKeypairBytesHex,
            inviteToken: inviteToken.value
        )
        self.handle = handle
        self.inference = Inference(handle: handle)
    }

    public init(handle: MeshClientHandle) {
        self.handle = handle
        self.inference = Inference(handle: handle)
    }

    public static func connectPublic(
        ownerKeypairBytesHex: String,
        query: PublicMeshQuery = PublicMeshQuery(
            model: nil,
            minVramGb: nil,
            region: nil,
            targetName: nil,
            relays: []
        )
    ) async throws -> Client {
        let handle = try await runBlocking {
            try createAutoClient(ownerKeypairBytesHex: ownerKeypairBytesHex, query: query)
        }
        return Client(handle: handle)
    }

    public func start() async throws {
        let handle = self.handle
        try await runBlocking {
            try handle.start()
        }
    }

    public func stop() async {
        let handle = self.handle
        await runNonThrowing {
            handle.stop()
        }
    }

    public func reconnect() async throws {
        let handle = self.handle
        try await runBlocking {
            try handle.reconnect()
        }
    }

    public func status() async -> Status {
        let handle = self.handle
        let status = await runBlocking {
            handle.status()
        }
        return Status(connected: status.connected, peerCount: Int(clamping: status.peerCount))
    }

    public final class Inference: @unchecked Sendable {
        private let handle: MeshClientHandle

        fileprivate init(handle: MeshClientHandle) {
            self.handle = handle
        }

        public func listModels() async throws -> [Model] {
            let handle = self.handle
            let models = try await runBlocking {
                try handle.inferenceListModels()
            }
            return models.map(Node.mapModel)
        }

        public func chat(_ request: ChatRequest) -> AsyncThrowingStream<Event, Error> {
            let native = Node.mapChatRequest(request)
            return AsyncThrowingStream { continuation in
                do {
                    let bridge = EventStreamBridge(continuation: continuation) { [handle] requestId in
                        handle.cancel(requestId: requestId)
                    }
                    let requestId = try handle.chat(request: native, listener: bridge)
                    bridge.activate(requestId: requestId)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }

        public func responses(_ request: ResponsesRequest) -> AsyncThrowingStream<Event, Error> {
            let native = Node.mapResponsesRequest(request)
            return AsyncThrowingStream { continuation in
                do {
                    let bridge = EventStreamBridge(continuation: continuation) { [handle] requestId in
                        handle.cancel(requestId: requestId)
                    }
                    let requestId = try handle.responses(request: native, listener: bridge)
                    bridge.activate(requestId: requestId)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }

        public func cancel(_ requestId: RequestId) async {
            let handle = self.handle
            await runNonThrowing {
                handle.cancel(requestId: requestId.value)
            }
        }
    }
}

public final class Node: @unchecked Sendable {
    private let handle: MeshNodeHandle

    public let inference: Inference
    public let models: Models
    public let serving: Serving

    public init(
        inviteToken: InviteToken,
        ownerKeypairBytesHex: String,
        cacheDir: String? = nil,
        runtimeDir: String? = nil
    ) throws {
        let handle = try createNode(
            ownerKeypairBytesHex: ownerKeypairBytesHex,
            inviteToken: inviteToken.value,
            cacheDir: cacheDir,
            runtimeDir: runtimeDir,
            servingEnabled: true
        )
        self.handle = handle
        self.inference = Inference(handle: handle)
        self.models = Models(handle: handle)
        self.serving = Serving(handle: handle)
    }

    public init(handle: MeshNodeHandle) {
        self.handle = handle
        self.inference = Inference(handle: handle)
        self.models = Models(handle: handle)
        self.serving = Serving(handle: handle)
    }

    public static func discoverPublicMeshes(
        _ query: PublicMeshQuery = PublicMeshQuery(
            model: nil,
            minVramGb: nil,
            region: nil,
            targetName: nil,
            relays: []
        )
    ) async throws -> [PublicMesh] {
        try await runBlocking {
            try MeshLLM.discoverPublicMeshes(query: query)
        }
    }

    public static func connectPublic(
        ownerKeypairBytesHex: String,
        query: PublicMeshQuery = PublicMeshQuery(
            model: nil,
            minVramGb: nil,
            region: nil,
            targetName: nil,
            relays: []
        )
    ) async throws -> Node {
        let handle = try await runBlocking {
            try createAutoNode(ownerKeypairBytesHex: ownerKeypairBytesHex, query: query)
        }
        return Node(handle: handle)
    }

    public func start() async throws {
        let handle = self.handle
        try await runBlocking {
            try handle.start()
        }
    }

    public func stop() async throws {
        let handle = self.handle
        try await runBlocking {
            try handle.stop()
        }
    }

    public func reconnect() async throws {
        let handle = self.handle
        try await runBlocking {
            try handle.reconnect()
        }
    }

    public func status() async -> Status {
        let handle = self.handle
        let status = await runBlocking {
            handle.status()
        }
        return Status(connected: status.connected, peerCount: Int(clamping: status.peerCount))
    }

    public final class Inference: @unchecked Sendable {
        private let handle: MeshNodeHandle

        fileprivate init(handle: MeshNodeHandle) {
            self.handle = handle
        }

        public func listModels() async throws -> [Model] {
            let handle = self.handle
            let models = try await runBlocking {
                try handle.inferenceListModels()
            }
            return models.map(Node.mapModel)
        }

        public func chat(_ request: ChatRequest) -> AsyncThrowingStream<Event, Error> {
            let native = Node.mapChatRequest(request)
            return AsyncThrowingStream { continuation in
                do {
                    let bridge = EventStreamBridge(continuation: continuation) { [handle] requestId in
                        try? handle.cancel(requestId: requestId)
                    }
                    let requestId = try handle.chat(request: native, listener: bridge)
                    bridge.activate(requestId: requestId)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }

        public func responses(_ request: ResponsesRequest) -> AsyncThrowingStream<Event, Error> {
            let native = Node.mapResponsesRequest(request)
            return AsyncThrowingStream { continuation in
                do {
                    let bridge = EventStreamBridge(continuation: continuation) { [handle] requestId in
                        try? handle.cancel(requestId: requestId)
                    }
                    let requestId = try handle.responses(request: native, listener: bridge)
                    bridge.activate(requestId: requestId)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }

        public func cancel(_ requestId: RequestId) async throws {
            let handle = self.handle
            try await runBlocking {
                try handle.cancel(requestId: requestId.value)
            }
        }
    }

    public final class Models: @unchecked Sendable {
        private let handle: MeshNodeHandle

        fileprivate init(handle: MeshNodeHandle) {
            self.handle = handle
        }

        public func recommended() async throws -> [ModelSummary] {
            let handle = self.handle
            return try await runBlocking { try handle.recommendedModels() }
        }

        public func search(_ query: ModelSearchQuery) async throws -> [ModelSummary] {
            let handle = self.handle
            return try await runBlocking { try handle.searchModels(query: query) }
        }

        public func show(_ modelRef: String) async throws -> ModelDetails {
            let handle = self.handle
            return try await runBlocking { try handle.showModel(modelRef: modelRef) }
        }

        public func installed() async throws -> [InstalledModel] {
            let handle = self.handle
            return try await runBlocking { try handle.installedModels() }
        }

        public func cacheStatus() async throws -> ModelCacheStatus {
            let handle = self.handle
            return try await runBlocking { try handle.modelCacheStatus() }
        }

        public func download(_ modelRef: String) async throws -> DownloadedModel {
            let handle = self.handle
            return try await runBlocking { try handle.downloadModel(modelRef: modelRef) }
        }

        public func delete(_ modelRef: String, options: DeleteModelOptions) async throws -> DeleteModelResult {
            let handle = self.handle
            return try await runBlocking { try handle.deleteModel(modelRef: modelRef, options: options) }
        }

        public func cleanup(_ policy: CleanupPolicy) async throws -> CleanupResult {
            let handle = self.handle
            return try await runBlocking { try handle.cleanupModels(policy: policy) }
        }

        public func pruneDerivedCache(_ policy: PrunePolicy) async throws -> PruneResult {
            let handle = self.handle
            return try await runBlocking { try handle.pruneDerivedCache(policy: policy) }
        }
    }

    public final class Serving: @unchecked Sendable {
        private let handle: MeshNodeHandle

        fileprivate init(handle: MeshNodeHandle) {
            self.handle = handle
        }

        public func status() async throws -> ServingStatus {
            let handle = self.handle
            return try await runBlocking { try handle.servingStatus() }
        }

        public func servedModels() async throws -> [ServedModel] {
            let handle = self.handle
            return try await runBlocking { try handle.servedModels() }
        }

        public func load(_ modelRef: String, options: LoadModelOptions) async throws -> ServedModel {
            let handle = self.handle
            return try await runBlocking { try handle.loadServingModel(modelRef: modelRef, options: options) }
        }

        public func unload(_ target: UnloadTarget, options: UnloadModelOptions) async throws {
            let handle = self.handle
            return try await runBlocking { try handle.unloadServingModel(target: target, options: options) }
        }

        public func unloadModel(_ modelId: String, options: UnloadModelOptions) async throws {
            let handle = self.handle
            return try await runBlocking { try handle.unloadServingModelById(modelId: modelId, options: options) }
        }

        public func unloadInstance(_ instanceId: String, options: UnloadModelOptions) async throws {
            let handle = self.handle
            return try await runBlocking { try handle.unloadServingInstance(instanceId: instanceId, options: options) }
        }

        public func setDevicePolicy(_ policy: DevicePolicy) async throws {
            let handle = self.handle
            return try await runBlocking { try handle.setDevicePolicy(policy: policy) }
        }
    }

    fileprivate static func mapModel(_ native: ModelNative) -> Model {
        Model(id: native.id, name: native.name)
    }

    static func mapEvent(_ native: ClientEvent) -> Event {
        switch native {
        case .connecting:
            return .connecting
        case .joined(let nodeId):
            return .joined(nodeId: nodeId)
        case .modelsUpdated(let models):
            return .modelsUpdated(models: models.map(mapModel))
        case .tokenDelta(let requestId, let delta):
            return .tokenDelta(requestId: requestId, delta: delta)
        case .completed(let requestId):
            return .completed(requestId: requestId)
        case .failed(let requestId, let error):
            return .failed(requestId: requestId, error: error)
        case .disconnected(let reason):
            return .disconnected(reason: reason)
        }
    }

    fileprivate static func mapChatRequest(_ request: ChatRequest) -> ChatRequestNative {
        ChatRequestNative(
            model: request.model,
            messages: request.messages.map {
                ChatMessageNative(role: $0.role, content: $0.content)
            }
        )
    }

    fileprivate static func mapResponsesRequest(_ request: ResponsesRequest) -> ResponsesRequestNative {
        ResponsesRequestNative(model: request.model, input: request.input)
    }
}
#else
#error("MeshLLM Swift SDK requires MeshLLMFFI.xcframework. Build it with sdk/swift/scripts/build-xcframework.sh before building, testing, or integrating the SDK.")
#endif

private func runBlocking<T>(_ work: @escaping () throws -> T) async throws -> T {
    try await withCheckedThrowingContinuation { continuation in
        DispatchQueue.global().async(flags: .inheritQoS) {
            do {
                continuation.resume(returning: try work())
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}

private func runNonThrowing<T>(_ work: @escaping () -> T) async -> T {
    await withCheckedContinuation { continuation in
        DispatchQueue.global().async(flags: .inheritQoS) {
            continuation.resume(returning: work())
        }
    }
}

private func runBlocking<T>(_ work: @escaping () -> T) async -> T {
    await withCheckedContinuation { continuation in
        DispatchQueue.global().async(flags: .inheritQoS) {
            continuation.resume(returning: work())
        }
    }
}
