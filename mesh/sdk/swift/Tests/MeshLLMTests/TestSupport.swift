import Foundation
import XCTest
@testable import MeshLLM

final class TestMeshNodeHandle: MeshNodeHandle, @unchecked Sendable {
    private let lock = NSLock()
    private var lastPathStorage: String?
    private var lastBodyStorage: String?

    var lastOpenAIPath: String? {
        lock.lock()
        defer { lock.unlock() }
        return lastPathStorage
    }

    var lastOpenAIBody: String? {
        lock.lock()
        defer { lock.unlock() }
        return lastBodyStorage
    }

    init() {
        super.init(noHandle: MeshNodeHandle.NoHandle())
    }

    required init(unsafeFromHandle handle: UInt64) {
        super.init(unsafeFromHandle: handle)
    }

    override func status() throws -> NodeStatusNative {
        NodeStatusNative(
            running: false,
            mode: "client",
            apiBaseUrl: "",
            consoleUrl: "",
            payloadJson: "null"
        )
    }

    override func openaiRequest(path: String, bodyJson: String) throws -> OpenAiResponseNative {
        lock.lock()
        lastPathStorage = path
        lastBodyStorage = bodyJson
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
        lastPathStorage = path
        lastBodyStorage = bodyJson
        lock.unlock()
        listener.onEvent(event: .started(requestId: "test-request", statusCode: 200, contentType: "text/event-stream"))
        listener.onEvent(event: .sse(
            requestId: "test-request",
            eventType: "response.function_call_arguments.delta",
            data: #"{"delta":{"tool_calls":[{"index":0}]}}"#,
            raw: "data: {\"delta\":{\"tool_calls\":[{\"index\":0}]}}\n\n"
        ))
        listener.onEvent(event: .sse(requestId: "test-request", eventType: nil, data: "[DONE]", raw: "data: [DONE]\n\n"))
        listener.onEvent(event: .completed(requestId: "test-request"))
        return "test-request"
    }

    override func cancel(requestId: String) {}
}
