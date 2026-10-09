import Foundation
import XCTest
@testable import MeshLLM

final class OpenAIStreamTests: XCTestCase {
    func testRequestPreservesAgentPayload() async throws {
        let handle = TestMeshNodeHandle()
        let node = Node(handle: handle)
        let response = try await node.inference.chatCompletions([
            "model": "test-model",
            "tools": [["type": "function", "function": ["name": "weather"]]],
        ])
        let requestBody = try XCTUnwrap(handle.lastOpenAIBody)
        let request = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(requestBody.utf8)) as? [String: Any])
        XCTAssertEqual(handle.lastOpenAIPath, "/v1/chat/completions")
        XCTAssertNotNil(request["tools"])
        XCTAssertEqual(request["stream"] as? Bool, false)
        XCTAssertEqual(response.statusCode, 200)
    }

    func testStreamPreservesNamedEventAndDoneFrame() async throws {
        let node = Node(handle: TestMeshNodeHandle())
        var events: [OpenAIStreamEvent] = []
        for try await event in node.inference.streamResponses(["model": "test-model", "input": "weather?"]) {
            events.append(event)
        }
        XCTAssertEqual(events.count, 3)
        guard case .sse(let delta) = events[1] else { return XCTFail("expected SSE delta") }
        XCTAssertEqual(delta.event, "response.function_call_arguments.delta")
        guard case .sse(let done) = events[2] else { return XCTFail("expected done frame") }
        XCTAssertTrue(done.isDone)
    }
}
