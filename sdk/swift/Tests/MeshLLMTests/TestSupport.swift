import Foundation
import XCTest
@testable import MeshLLM

func makeOwnerKeypairBytesHex() -> String {
    generateOwnerKeypairHex()
}

func makeTestNode() throws -> Node {
    Node(handle: TestMeshNodeHandle())
}

final class TestMeshNodeHandle: MeshNodeHandle, @unchecked Sendable {
    private let requestId = "test-request"
    private let lock = NSLock()
    private var cancelledRequestIdsStorage: [String] = []
    private var connected = false
    private var lastOpenAIPathStorage: String?
    private var lastOpenAIBodyStorage: String?

    var cancelledRequestIds: [String] {
        lock.lock()
        defer { lock.unlock() }
        return cancelledRequestIdsStorage
    }

    var lastOpenAIPath: String? {
        lock.lock()
        defer { lock.unlock() }
        return lastOpenAIPathStorage
    }

    var lastOpenAIBody: String? {
        lock.lock()
        defer { lock.unlock() }
        return lastOpenAIBodyStorage
    }

    init() {
        super.init(noHandle: MeshNodeHandle.NoHandle())
    }

    required init(unsafeFromHandle handle: UInt64) {
        super.init(unsafeFromHandle: handle)
    }

    override func chat(request: ChatRequestNative, listener: EventListener) throws -> String {
        listener.onEvent(event: .tokenDelta(requestId: requestId, delta: "hello"))
        listener.onEvent(event: .completed(requestId: requestId))
        return requestId
    }

    override func responses(request: ResponsesRequestNative, listener: EventListener) throws -> String {
        listener.onEvent(event: .tokenDelta(requestId: requestId, delta: "hello"))
        listener.onEvent(event: .completed(requestId: requestId))
        return requestId
    }

    override func openaiRequest(path: String, bodyJson: String) throws -> OpenAiResponseNative {
        lock.lock()
        lastOpenAIPathStorage = path
        lastOpenAIBodyStorage = bodyJson
        lock.unlock()
        return OpenAiResponseNative(
            statusCode: 200,
            contentType: "application/json",
            body: #"{"choices":[{"message":{"tool_calls":[{"id":"call-1"}]}}]}"#
        )
    }

    override func openaiStream(
        path: String,
        bodyJson: String,
        listener: OpenAiStreamListener
    ) throws -> String {
        lock.lock()
        lastOpenAIPathStorage = path
        lastOpenAIBodyStorage = bodyJson
        lock.unlock()
        listener.onEvent(event: .started(
            requestId: requestId,
            statusCode: 200,
            contentType: "text/event-stream"
        ))
        let data = #"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\"Syd"}}]}}]}"#
        listener.onEvent(event: .sse(
            requestId: requestId,
            eventType: path == "/v1/responses" ? "response.function_call_arguments.delta" : nil,
            data: data,
            raw: "data: \(data)\n\n"
        ))
        listener.onEvent(event: .sse(
            requestId: requestId,
            eventType: nil,
            data: "[DONE]",
            raw: "data: [DONE]\n\n"
        ))
        listener.onEvent(event: .completed(requestId: requestId))
        return requestId
    }

    override func cancel(requestId: String) throws {
        lock.lock()
        cancelledRequestIdsStorage.append(requestId)
        lock.unlock()
    }

    override func reconnect() throws {
        lock.lock()
        connected = true
        lock.unlock()
    }

    override func start() throws {
        lock.lock()
        connected = true
        lock.unlock()
    }

    override func status() -> ClientStatus {
        lock.lock()
        let isConnected = connected
        lock.unlock()
        return ClientStatus(connected: isConnected, peerCount: isConnected ? 1 : 0)
    }

    override func stop() throws {
        lock.lock()
        connected = false
        lock.unlock()
    }
}

final class OpenStreamMeshNodeHandle: MeshNodeHandle, @unchecked Sendable {
    private let requestId = "open-request"
    private let lock = NSLock()
    private var cancelledRequestIdsStorage: [String] = []
    private var chatListener: EventListener?

    var cancelledRequestIds: [String] {
        lock.lock()
        defer { lock.unlock() }
        return cancelledRequestIdsStorage
    }

    init() {
        super.init(noHandle: MeshNodeHandle.NoHandle())
    }

    required init(unsafeFromHandle handle: UInt64) {
        super.init(unsafeFromHandle: handle)
    }

    override func chat(request: ChatRequestNative, listener: EventListener) throws -> String {
        lock.lock()
        chatListener = listener
        lock.unlock()
        listener.onEvent(event: .tokenDelta(requestId: requestId, delta: "hello"))
        return requestId
    }

    override func responses(request: ResponsesRequestNative, listener: EventListener) throws -> String {
        listener.onEvent(event: .tokenDelta(requestId: requestId, delta: "hello"))
        return requestId
    }

    override func openaiStream(
        path: String,
        bodyJson: String,
        listener: OpenAiStreamListener
    ) throws -> String {
        listener.onEvent(event: .started(
            requestId: requestId,
            statusCode: 200,
            contentType: "text/event-stream"
        ))
        listener.onEvent(event: .sse(
            requestId: requestId,
            eventType: nil,
            data: #"{"choices":[{"delta":{"content":"hello"}}]}"#,
            raw: #"data: {"choices":[{"delta":{"content":"hello"}}]}"# + "\n\n"
        ))
        return requestId
    }

    override func cancel(requestId: String) throws {
        lock.lock()
        cancelledRequestIdsStorage.append(requestId)
        lock.unlock()
    }

    func completeChat() {
        lock.lock()
        let listener = chatListener
        lock.unlock()
        listener?.onEvent(event: .completed(requestId: requestId))
    }
}

final class FailedOpenAIStreamMeshNodeHandle: MeshNodeHandle, @unchecked Sendable {
    init() {
        super.init(noHandle: MeshNodeHandle.NoHandle())
    }

    required init(unsafeFromHandle handle: UInt64) {
        super.init(unsafeFromHandle: handle)
    }

    override func openaiStream(
        path: String,
        bodyJson: String,
        listener: OpenAiStreamListener
    ) throws -> String {
        listener.onEvent(event: .failed(
            requestId: "failed-request",
            statusCode: 429,
            error: "rate limited",
            body: #"{"error":"slow down"}"#
        ))
        return "failed-request"
    }
}

func waitUntil(
    timeout: Duration = .seconds(2),
    _ condition: @escaping () -> Bool
) async throws {
    let start = ContinuousClock.now
    while !condition() {
        if start.duration(to: .now) > timeout {
            XCTFail("condition was not met before timeout")
            return
        }
        try await Task.sleep(for: .milliseconds(10))
    }
}
