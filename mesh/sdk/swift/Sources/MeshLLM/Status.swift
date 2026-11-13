import Foundation

public enum NodeLifecycle: Decodable, Sendable, Equatable {
    case loading, downloading, ready, stopped
    case failed(String)

    private enum CodingKeys: String, CodingKey { case phase, error }
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let phase = try values.decode(String.self, forKey: .phase)
        switch phase {
        case "loading": self = .loading
        case "downloading": self = .downloading
        case "ready": self = .ready
        case "stopped": self = .stopped
        case "failed": self = .failed(try values.decode(String.self, forKey: .error))
        default: throw DecodingError.dataCorruptedError(forKey: .phase, in: values, debugDescription: "Unknown node lifecycle: \(phase)")
        }
    }
}

public struct NodeDownloadProgress: Decodable, Sendable {
    public let file: String
    public let downloadedBytes: UInt64
    public let totalBytes: UInt64?
}

public struct ModelCapacity: Decodable, Sendable {
    public let model: String
    public let contextPerLane: [UInt64]
    public let lastSuccessUnixMs: UInt64?
}

public struct NativeNodeStatus: Decodable, Sendable {
    public let state: NodeLifecycle
    public let servingModels: [String]
    public let models: [MeshModel]
    /// Native served descriptors retain optional identity and metadata facts.
    public let localModels: [ModelJSONValue]
    public let capacities: [ModelCapacity]
    public let download: NodeDownloadProgress?

    internal static func decode(_ json: String) throws -> NativeNodeStatus {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Self.self, from: Data(json.utf8))
    }
}

/// Current state followed by coalesced updates for one node generation.
/// Close when finished; node stop also closes the native subscription.
public final class StatusEvents: Sendable {
    private let nextJSON: @Sendable () async throws -> String?
    private let closeNative: @Sendable () async -> Void

    internal init(nextJSON: @escaping @Sendable () async throws -> String?,
                  close: @escaping @Sendable () async -> Void) {
        self.nextJSON = nextJSON
        self.closeNative = close
    }

    public func nextEvent() async throws -> NativeNodeStatus? {
        try await withTaskCancellationHandler {
            guard let json = try await nextJSON() else { return nil }
            return try NativeNodeStatus.decode(json)
        } onCancel: {
            Task { await self.close() }
        }
    }

    public func close() async { await closeNative() }

    public func events() -> AsyncThrowingStream<NativeNodeStatus, Error> {
        AsyncThrowingStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            let task = Task {
                do {
                    while let status = try await nextEvent() { continuation.yield(status) }
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { _ in
                task.cancel()
                Task { await self.close() }
            }
        }
    }
}
