import Foundation

public struct LiveTurnDetection: Encodable, Sendable {
    public var threshold: Float
    public var prefixPaddingMs: UInt32
    public var silenceDurationMs: UInt32
    public init(threshold: Float = 0.5, prefixPaddingMs: UInt32 = 300, silenceDurationMs: UInt32 = 500) {
        self.threshold = threshold
        self.prefixPaddingMs = prefixPaddingMs
        self.silenceDurationMs = silenceDurationMs
    }
}

public struct LiveTranscriptionOptions: Encodable, Sendable {
    public var model: String?
    public var language: String?
    public var prompt: String?
    public var sampleRate: UInt32
    /// nil uses explicit commit calls instead of server-side turn detection.
    public var turnDetection: LiveTurnDetection?
    public init(model: String? = nil, language: String? = nil, prompt: String? = nil,
                sampleRate: UInt32 = 16000, turnDetection: LiveTurnDetection? = LiveTurnDetection()) {
        self.model = model; self.language = language; self.prompt = prompt
        self.sampleRate = sampleRate; self.turnDetection = turnDetection
    }
    private enum CodingKeys: String, CodingKey { case model, language, prompt, sampleRate, turnDetection }
    public func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encodeIfPresent(model, forKey: .model)
        try values.encodeIfPresent(language, forKey: .language)
        try values.encodeIfPresent(prompt, forKey: .prompt)
        try values.encode(sampleRate, forKey: .sampleRate)
        try values.encode(turnDetection, forKey: .turnDetection)
    }
    internal func json() throws -> String {
        let encoder = JSONEncoder(); encoder.keyEncodingStrategy = .convertToSnakeCase
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }
}

/// Event types are speech_started, speech_stopped, delta, committed, completed and error.
public struct LiveTranscriptionEvent: Decodable, Sendable {
    public let type: String
    public let itemId: String?
    public let audioStartMs: UInt64?
    public let audioEndMs: UInt64?
    public let text: String?
    public let transcript: String?
    public let message: String?
}

/// Audio writes and event reads may run concurrently. Close or stop when finished.
public final class LiveTranscription: @unchecked Sendable {
    private let audio: @Sendable ([Int16]) async throws -> Void
    private let floatAudio: @Sendable ([Float]) async throws -> Void
    private let commitNative: @Sendable () async throws -> Void
    private let eventJSON: @Sendable () async throws -> String?
    private let stopJSON: @Sendable () async throws -> String
    private let closeNative: @Sendable () async -> Void

    internal init(audio: @escaping @Sendable ([Int16]) async throws -> Void,
                  floatAudio: @escaping @Sendable ([Float]) async throws -> Void,
                  commit: @escaping @Sendable () async throws -> Void,
                  eventJSON: @escaping @Sendable () async throws -> String?,
                  stopJSON: @escaping @Sendable () async throws -> String,
                  close: @escaping @Sendable () async -> Void) {
        self.audio = audio; self.floatAudio = floatAudio; self.commitNative = commit
        self.eventJSON = eventJSON; self.stopJSON = stopJSON; self.closeNative = close
    }
    public func sendAudio(_ samples: [Int16]) async throws { try await audio(samples) }
    public func sendAudioF32(_ samples: [Float]) async throws { try await floatAudio(samples) }
    public func commit() async throws { try await commitNative() }
    public func nextEvent() async throws -> LiveTranscriptionEvent? {
        guard let json = try await eventJSON() else { return nil }
        return try Self.decode(LiveTranscriptionEvent.self, json)
    }
    public func stop() async throws -> [LiveTranscriptionEvent] {
        try Self.decode([LiveTranscriptionEvent].self, await stopJSON())
    }
    public func close() async { await closeNative() }
    public func events() -> AsyncThrowingStream<LiveTranscriptionEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    while let event = try await nextEvent() { continuation.yield(event) }
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { _ in
                task.cancel()
                Task { await self.close() }
            }
        }
    }
    private static func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(type, from: Data(json.utf8))
    }
}
