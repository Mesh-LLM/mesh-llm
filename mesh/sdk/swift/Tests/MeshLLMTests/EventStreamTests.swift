import XCTest
@testable import MeshLLM

final class EventStreamTests: XCTestCase {
    func testSSEPayloadCanBeDecoded() throws {
        let event = OpenAISSEEvent(
            requestId: "test",
            event: nil,
            data: #"{"delta":{"content":"hello"}}"#,
            raw: "data: test\n\n"
        )
        XCTAssertNotNil(try event.jsonObject())
        XCTAssertFalse(event.isDone)
    }
}
