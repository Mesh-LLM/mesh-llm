import Foundation

public enum NodeMode: String, Sendable {
    case client
    case serve
    case combined
}

public struct NodeStatus: Sendable {
    public let running: Bool
    public let mode: NodeMode
    public let apiBaseURL: String
    public let consoleURL: String
    public let payloadJSON: String
}

public struct Model: Sendable {
    public let id: String
    public let name: String
    public let contextLength: UInt32?
    /// Ordered effort strings the serving provider observed in the loaded chat
    /// template. `nil` means unknown (an older provider or an unprobed model);
    /// an empty array means the template exposes no effort levels.
    public let thinkingEfforts: [String]?
}

public struct OpenAIResponse: Sendable {
    public let statusCode: UInt16
    public let contentType: String?
    public let body: String

    public func jsonObject() throws -> Any {
        try JSONSerialization.jsonObject(with: Data(body.utf8))
    }

    /// Return this response, throwing a typed failure for non-success statuses.
    public func errorForStatus() throws -> OpenAIResponse {
        guard (200..<300).contains(statusCode) else {
            throw OpenAIRequestFailure(statusCode: statusCode, body: body)
        }
        return self
    }
}

public struct OpenAIRequestFailure: Error, Sendable, CustomStringConvertible {
    public let statusCode: UInt16
    public let body: String
    public var description: String { "OpenAI-compatible request failed with HTTP \(statusCode): \(body)" }
}

public struct OpenAISSEEvent: Sendable {
    public let requestId: String
    public let event: String?
    public let data: String
    public let raw: String

    public var isDone: Bool { data == "[DONE]" }

    public func jsonObject() throws -> Any? {
        guard !isDone else { return nil }
        return try JSONSerialization.jsonObject(with: Data(data.utf8))
    }
}

public enum OpenAIStreamEvent: Sendable {
    case started(requestId: String, statusCode: UInt16, contentType: String?)
    case sse(OpenAISSEEvent)
}

public struct OpenAIStreamFailure: Error, Sendable, CustomStringConvertible {
    public let requestId: String
    public let statusCode: UInt16?
    public let message: String
    public let body: String?

    public var description: String {
        if let statusCode {
            return "OpenAI-compatible stream failed with HTTP \(statusCode): \(message)"
        }
        return message
    }
}

#if canImport(MeshLLMFFI)
public typealias MeshError = FfiError

public final class Node: @unchecked Sendable {
    private let handle: MeshNodeHandle
    public let inference: Inference
    public let models: Models
    private let membershipControls: Memberships

    /// Construct using the full Rust config schema and its native defaults.
    /// Keys remain snake_case and unknown fields are rejected by the core.
    public static func fromConfig(_ config: [String: ModelJSONValue]) throws -> Node {
        try fromConfig(config, create: { try createNodeConfigJson(configJson: $0) })
    }

    internal static func fromConfig(_ config: [String: ModelJSONValue],
                                    create: (String) throws -> MeshNodeHandle) throws -> Node {
        let json = String(decoding: try JSONEncoder().encode(config), as: UTF8.self)
        return Node(handle: try create(json))
    }

    public init(
        mode: NodeMode = .client,
        joinTokens: [String] = [],
        models: [String] = [],
        autoJoin: Bool = false,
        ownerKeyPath: String? = nil,
        apiPort: UInt16 = 9337,
        consolePort: UInt16 = 3131
    ) throws {
        let handle = try createNode(
            mode: mode.rawValue,
            joinTokens: joinTokens,
            models: models,
            autoJoin: autoJoin,
            ownerKeyPath: ownerKeyPath,
            apiPort: apiPort,
            consolePort: consolePort
        )
        self.handle = handle
        self.inference = Inference(handle: handle)
        self.models = Self.modelDiscovery(handle: handle)
        self.membershipControls = Self.memberships(handle: handle)
    }

    internal init(handle: MeshNodeHandle) {
        self.handle = handle
        self.inference = Inference(handle: handle)
        self.models = Self.modelDiscovery(handle: handle)
        self.membershipControls = Self.memberships(handle: handle)
    }

    private static func modelDiscovery(handle: MeshNodeHandle) -> Models {
        Models(
            listJSON: { query in try await runBlocking { try handle.meshModelsJson(queryJson: query) } },
            modelJSON: { id in try await runBlocking { try handle.meshModelJson(id: id) } }
        )
    }

    private static func memberships(handle: MeshNodeHandle) -> Memberships {
        Memberships(
            addJSON: { config in try await runBlocking { try handle.addMembershipJson(configJson: config) } },
            removeNative: { name in try await runBlocking { try handle.removeMembership(name: name) } },
            listJSON: { try await runBlocking { try handle.membershipsJson() } }
        )
    }

    public func addMembership(_ config: MeshMembershipConfig) async throws -> MeshMembershipStatus {
        try await membershipControls.add(config)
    }

    public func removeMembership(_ name: String) async throws -> Bool {
        try await membershipControls.remove(name)
    }

    public func memberships() async throws -> [MeshMembershipStatus] {
        try await membershipControls.list()
    }

    public func start() async throws {
        let handle = self.handle
        try await runBlocking { try handle.start() }
    }

    public func stop() async throws {
        let handle = self.handle
        try await runBlocking { try handle.stop() }
    }

    public func status() async throws -> NodeStatus {
        let handle = self.handle
        let value = try await runBlocking { try handle.status() }
        return NodeStatus(
            running: value.running,
            mode: NodeMode(rawValue: value.mode) ?? .client,
            apiBaseURL: value.apiBaseUrl,
            consoleURL: value.consoleUrl,
            payloadJSON: value.payloadJson
        )
    }

    public func joinToken(_ token: String) async throws {
        let handle = self.handle
        try await runBlocking { try handle.joinToken(token: token) }
    }

    public func startLiveTranscription(_ options: LiveTranscriptionOptions = LiveTranscriptionOptions()) async throws -> LiveTranscription {
        let handle = self.handle
        let optionsJSON = try options.json()
        let live = try await runBlocking { try handle.startLiveTranscriptionJson(optionsJson: optionsJSON) }
        return LiveTranscription(
            audio: { samples in try await runBlocking { try live.sendAudio(samples: samples) } },
            floatAudio: { samples in try await runBlocking { try live.sendAudioF32(samples: samples) } },
            commit: { try await runBlocking { try live.commit() } },
            eventJSON: { try await runBlocking { try live.nextEventJson() } },
            stopJSON: { try await runBlocking { try live.stopJson() } },
            close: { await withCheckedContinuation { continuation in
                DispatchQueue.global().async { live.closeAudio(); continuation.resume() }
            } }
        )
    }

    public final class Inference: @unchecked Sendable {
        private let handle: MeshNodeHandle

        fileprivate init(handle: MeshNodeHandle) {
            self.handle = handle
        }

        public func listModels() async throws -> [Model] {
            let handle = self.handle
            return try await runBlocking { try handle.inferenceListModels() }
                .map {
                    Model(
                        id: $0.id,
                        name: $0.name,
                        contextLength: $0.contextLength,
                        thinkingEfforts: $0.thinkingEfforts
                    )
                }
        }

        /// Submit PDF or image bytes to a mesh-advertised document OCR model.
        public func documentOcr(_ file: Data, options: DocumentOcrOptions,
                                raiseForStatus: Bool = true) async throws -> OpenAIResponse {
            let optionsJSON = try options.json()
            let handle = self.handle
            let response = try await runBlocking {
                try handle.documentOcr(file: file, optionsJson: optionsJSON)
            }
            let result = OpenAIResponse(statusCode: response.statusCode,
                contentType: response.contentType, body: response.body)
            return try raiseForStatus ? result.errorForStatus() : result
        }

        public func request(path: String, body: [String: Any], raiseForStatus: Bool = true) async throws -> OpenAIResponse {
            let bodyJson = try encodeOpenAIObject(body)
            let handle = self.handle
            let response = try await runBlocking {
                try handle.openaiRequest(path: path, bodyJson: bodyJson)
            }
            let result = OpenAIResponse(
                statusCode: response.statusCode,
                contentType: response.contentType,
                body: response.body
            )
            return try raiseForStatus ? result.errorForStatus() : result
        }

        public func chatCompletions(_ body: [String: Any]) async throws -> OpenAIResponse {
            var request = body
            request["stream"] = false
            return try await self.request(path: "/v1/chat/completions", body: request)
        }

        public func responses(_ body: [String: Any]) async throws -> OpenAIResponse {
            var request = body
            request["stream"] = false
            return try await self.request(path: "/v1/responses", body: request)
        }

        public func stream(path: String, body: [String: Any]) -> AsyncThrowingStream<OpenAIStreamEvent, Error> {
            do {
                var request = body
                request["stream"] = true
                let bodyJson = try encodeOpenAIObject(request)
                return AsyncThrowingStream(bufferingPolicy: .bufferingOldest(256)) { continuation in
                    do {
                        let bridge = OpenAIStreamBridge(continuation: continuation) { [handle] requestId in
                            handle.cancel(requestId: requestId)
                        }
                        let requestId = try handle.openaiStream(
                            path: path,
                            bodyJson: bodyJson,
                            listener: bridge
                        )
                        bridge.activate(requestId: requestId)
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
            } catch {
                return AsyncThrowingStream { $0.finish(throwing: error) }
            }
        }

        public func streamChatCompletions(_ body: [String: Any]) -> AsyncThrowingStream<OpenAIStreamEvent, Error> {
            stream(path: "/v1/chat/completions", body: body)
        }

        public func streamResponses(_ body: [String: Any]) -> AsyncThrowingStream<OpenAIStreamEvent, Error> {
            stream(path: "/v1/responses", body: body)
        }
    }
}
#else
#error("MeshLLM Swift SDK requires MeshLLMFFI.xcframework.")
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

private func encodeOpenAIObject(_ body: [String: Any]) throws -> String {
    guard JSONSerialization.isValidJSONObject(body) else {
        throw EncodingError.invalidValue(body, .init(codingPath: [], debugDescription: "OpenAI request body must be a JSON object"))
    }
    let data = try JSONSerialization.data(withJSONObject: body)
    guard let result = String(data: data, encoding: .utf8) else {
        throw EncodingError.invalidValue(body, .init(codingPath: [], debugDescription: "OpenAI request body must be UTF-8"))
    }
    return result
}
