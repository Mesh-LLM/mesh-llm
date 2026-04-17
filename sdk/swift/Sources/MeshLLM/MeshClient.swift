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

public struct MeshStatus: Sendable {
    public let connected: Bool
    public let peerCount: Int
}

public struct RequestId: Sendable {
    public let value: String
}

public enum MeshEvent: Sendable {
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
public final class MeshClient: @unchecked Sendable {
    private let inviteToken: InviteToken
    private let ownerKeypairBytesHex: String
    private let stateLock = NSLock()
    private var handle: MeshClientHandle?

    public init(inviteToken: InviteToken, ownerKeypairBytesHex: String = "") {
        self.inviteToken = inviteToken
        self.ownerKeypairBytesHex = ownerKeypairBytesHex
    }

    public func join() async throws {
        let handle = try ensureHandle()
        try await runBlocking {
            try handle.join()
        }
    }

    public func listModels() async throws -> [Model] {
        let handle = try ensureHandle()
        let models = try await runBlocking {
            try handle.listModels()
        }
        return models.map(Self.mapModel)
    }

    public func chat(_ request: ChatRequest) -> AsyncThrowingStream<MeshEvent, Error> {
        let requestDto = Self.mapChatRequest(request)
        return AsyncThrowingStream { continuation in
            do {
                let handle = try self.ensureHandle()
                let bridge = EventStreamBridge(continuation: continuation) { [weak self] requestId in
                    self?.cancel(RequestId(value: requestId))
                }
                let requestId = handle.chat(request: requestDto, listener: bridge)
                bridge.activate(requestId: requestId)
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }

    public func responses(_ request: ResponsesRequest) -> AsyncThrowingStream<MeshEvent, Error> {
        let requestDto = Self.mapResponsesRequest(request)
        return AsyncThrowingStream { continuation in
            do {
                let handle = try self.ensureHandle()
                let bridge = EventStreamBridge(continuation: continuation) { [weak self] requestId in
                    self?.cancel(RequestId(value: requestId))
                }
                let requestId = handle.responses(request: requestDto, listener: bridge)
                bridge.activate(requestId: requestId)
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }

    public func cancel(_ requestId: RequestId) {
        guard let handle = currentHandle() else {
            return
        }
        handle.cancel(requestId: requestId.value)
    }

    public func status() async -> MeshStatus {
        guard let handle = currentHandle() else {
            return MeshStatus(connected: false, peerCount: 0)
        }
        let status = await runBlocking {
            handle.status()
        }
        return MeshStatus(
            connected: status.connected,
            peerCount: Int(clamping: status.peerCount)
        )
    }

    public func disconnect() async {
        guard let handle = currentHandle() else {
            return
        }
        await runBlocking {
            handle.disconnect()
        }
    }

    public func reconnect() async throws {
        let handle = try ensureHandle()
        try await runBlocking {
            try handle.reconnect()
        }
    }

    private func ensureHandle() throws -> MeshClientHandle {
        stateLock.lock()
        defer { stateLock.unlock() }

        if let handle {
            return handle
        }

        let created = try createClient(
            ownerKeypairBytesHex: ownerKeypairBytesHex,
            inviteToken: inviteToken.value
        )
        handle = created
        return created
    }

    private func currentHandle() -> MeshClientHandle? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return handle
    }

    private static func mapModel(_ dto: ModelDto) -> Model {
        Model(id: dto.id, name: dto.name)
    }

    static func mapEvent(_ dto: EventDto) -> MeshEvent {
        switch dto {
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

    private static func mapChatRequest(_ request: ChatRequest) -> ChatRequestDto {
        ChatRequestDto(
            model: request.model,
            messages: request.messages.map {
                ChatMessageDto(role: $0.role, content: $0.content)
            }
        )
    }

    private static func mapResponsesRequest(_ request: ResponsesRequest) -> ResponsesRequestDto {
        ResponsesRequestDto(model: request.model, input: request.input)
    }
}
#else
public final class MeshClient: @unchecked Sendable {
    private let inviteToken: InviteToken
    private var isConnected: Bool = false

    public init(inviteToken: InviteToken, ownerKeypairBytesHex _: String = "") {
        self.inviteToken = inviteToken
    }

    public func join() async throws {
        isConnected = true
    }

    public func listModels() async throws -> [Model] {
        []
    }

    public func chat(_ request: ChatRequest) -> AsyncThrowingStream<MeshEvent, Error> {
        let requestId = UUID().uuidString
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { [weak self] _ in
                self?.cancel(RequestId(value: requestId))
            }
            continuation.yield(.completed(requestId: requestId))
            continuation.finish()
        }
    }

    public func responses(_ request: ResponsesRequest) -> AsyncThrowingStream<MeshEvent, Error> {
        let requestId = UUID().uuidString
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { [weak self] _ in
                self?.cancel(RequestId(value: requestId))
            }
            continuation.yield(.completed(requestId: requestId))
            continuation.finish()
        }
    }

    public func cancel(_ requestId: RequestId) {}

    public func status() async -> MeshStatus {
        MeshStatus(connected: isConnected, peerCount: 0)
    }

    public func disconnect() async {
        isConnected = false
    }

    public func reconnect() async throws {
        await disconnect()
        try await join()
    }
}
#endif

private func runBlocking<T>(_ work: @escaping () throws -> T) async throws -> T {
    try await withCheckedThrowingContinuation { continuation in
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                continuation.resume(returning: try work())
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}

private func runBlocking<T>(_ work: @escaping () -> T) async -> T {
    await withCheckedContinuation { continuation in
        DispatchQueue.global(qos: .userInitiated).async {
            continuation.resume(returning: work())
        }
    }
}
