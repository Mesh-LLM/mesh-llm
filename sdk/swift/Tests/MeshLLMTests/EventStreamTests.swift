import XCTest
@testable import MeshLLM

final class EventStreamTests: XCTestCase {
    func testChatStreamEmitsCompletedEvent() async throws {
        let node = try makeTestNode()
        let request = ChatRequest(model: "test", messages: [])

        var events: [Event] = []
        for try await event in node.inference.chatStream(request) {
            events.append(event)
        }

        XCTAssertFalse(events.isEmpty)
        let hasCompleted = events.contains { if case .completed = $0 { return true }; return false }
        XCTAssertTrue(hasCompleted, "Stream should emit Completed event")
    }

    func testResponsesStreamEmitsCompletedEvent() async throws {
        let node = try makeTestNode()
        let request = ResponsesRequest(model: "test", input: "hello")

        var events: [Event] = []
        for try await event in node.inference.responsesStream(request) {
            events.append(event)
        }

        XCTAssertFalse(events.isEmpty)
        let hasCompleted = events.contains { if case .completed = $0 { return true }; return false }
        XCTAssertTrue(hasCompleted, "Stream should emit Completed event")
    }

    func testCancelOnTermination() async throws {
        let handle = OpenStreamMeshNodeHandle()
        let node = Node(handle: handle)
        let request = ChatRequest(model: "test", messages: [])

        for try await _ in node.inference.chatStream(request) {
            break
        }

        try await waitUntil {
            handle.cancelledRequestIds == ["open-request"]
        }
    }

    func testCompletionBeforeActivationDoesNotCancelCompletedRequest() async throws {
        let handle = TestMeshNodeHandle()
        let node = Node(handle: handle)
        let request = ChatRequest(model: "test", messages: [])

        var events: [Event] = []
        for try await event in node.inference.chatStream(request) {
            events.append(event)
        }

        let hasCompleted = events.contains { if case .completed = $0 { return true }; return false }
        XCTAssertTrue(hasCompleted, "Stream should emit Completed event")
        XCTAssertEqual(handle.cancelledRequestIds, [])
    }

    func testLateCompletionAfterCancellationIsIgnored() async throws {
        let handle = OpenStreamMeshNodeHandle()
        let node = Node(handle: handle)
        let request = ChatRequest(model: "test", messages: [])

        for try await _ in node.inference.chatStream(request) {
            break
        }

        try await waitUntil {
            handle.cancelledRequestIds == ["open-request"]
        }
        handle.completeChat()
        XCTAssertEqual(handle.cancelledRequestIds, ["open-request"])
    }
}
