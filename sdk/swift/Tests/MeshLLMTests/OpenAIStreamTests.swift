import Foundation
import XCTest
@testable import MeshLLM

final class OpenAIStreamTests: XCTestCase {
    private let toolRequest: [String: Any] = [
        "model": "test-model",
        "messages": [["role": "user", "content": "weather?"]],
        "tools": [[
            "type": "function",
            "function": ["name": "weather"],
        ]],
    ]

    func testBufferedRequestPreservesAgentPayloadAndResponse() async throws {
        let handle = TestMeshNodeHandle()
        let node = Node(handle: handle)

        let response = try await node.inference.chatCompletions(toolRequest)
        let requestBody = try XCTUnwrap(handle.lastOpenAIBody)
        let request = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(requestBody.utf8)) as? [String: Any]
        )
        let payload = try XCTUnwrap(response.jsonObject() as? [String: Any])

        XCTAssertEqual(handle.lastOpenAIPath, "/v1/chat/completions")
        XCTAssertEqual(request["stream"] as? Bool, false)
        XCTAssertNotNil(request["tools"])
        XCTAssertNotNil(payload["choices"])
    }

    func testResponsesIsTheCanonicalRichResponsesAPI() async throws {
        let handle = TestMeshNodeHandle()
        let node = Node(handle: handle)

        _ = try await node.inference.responses([
            "model": "test-model",
            "input": "weather?",
        ])

        XCTAssertEqual(handle.lastOpenAIPath, "/v1/responses")
        let requestBody = try XCTUnwrap(handle.lastOpenAIBody)
        let request = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(requestBody.utf8)) as? [String: Any]
        )
        XCTAssertEqual(request["stream"] as? Bool, false)
    }

    func testChatStreamPreservesToolCallDeltaAndRawFrame() async throws {
        let handle = TestMeshNodeHandle()
        let node = Node(handle: handle)
        var events: [OpenAIStreamEvent] = []

        for try await event in node.inference.streamChatCompletions(toolRequest) {
            events.append(event)
        }

        let requestBody = try XCTUnwrap(handle.lastOpenAIBody)
        let request = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(requestBody.utf8)) as? [String: Any]
        )
        XCTAssertEqual(request["stream"] as? Bool, true)
        XCTAssertEqual(events.count, 3)
        guard case .sse(let toolDelta) = events[1] else {
            return XCTFail("expected tool-call SSE event")
        }
        let payload = try XCTUnwrap(toolDelta.jsonObject() as? [String: Any])
        XCTAssertNotNil(payload["choices"])
        XCTAssertTrue(toolDelta.raw.hasPrefix("data: "))
        guard case .sse(let done) = events[2] else {
            return XCTFail("expected done SSE event")
        }
        XCTAssertTrue(done.isDone)
    }

    func testResponsesStreamPreservesNamedEvent() async throws {
        let node = Node(handle: TestMeshNodeHandle())

        for try await event in node.inference.streamResponses([
            "model": "test-model",
            "input": "weather?",
        ]) {
            if case .sse(let sse) = event, !sse.isDone {
                XCTAssertEqual(sse.event, "response.function_call_arguments.delta")
                return
            }
        }
        XCTFail("expected named Responses SSE event")
    }

    func testClosingStreamCancelsNativeRequest() async throws {
        let handle = OpenStreamMeshNodeHandle()
        let node = Node(handle: handle)

        for try await _ in node.inference.streamChatCompletions(toolRequest) {
            break
        }

        try await waitUntil {
            handle.cancelledRequestIds == ["open-request"]
        }
    }

    func testFailurePreservesStatusAndBody() async throws {
        let node = Node(handle: FailedOpenAIStreamMeshNodeHandle())

        do {
            for try await _ in node.inference.streamChatCompletions(toolRequest) {}
            XCTFail("expected stream failure")
        } catch let failure as OpenAIStreamFailure {
            XCTAssertEqual(failure.requestId, "failed-request")
            XCTAssertEqual(failure.statusCode, 429)
            XCTAssertEqual(failure.message, "rate limited")
            XCTAssertEqual(failure.body, #"{"error":"slow down"}"#)
        }
    }
}
