import XCTest
@testable import MeshLLM

final class NodeTests: XCTestCase {
    func testThreeRolesHaveDistinctWireValues() {
        XCTAssertEqual(NodeMode.client.rawValue, "client")
        XCTAssertEqual(NodeMode.serve.rawValue, "serve")
        XCTAssertEqual(NodeMode.combined.rawValue, "combined")
    }

    func testStatusReportsEmbeddedRole() async throws {
        let node = Node(handle: TestMeshNodeHandle())
        let status = try await node.status()
        XCTAssertFalse(status.running)
        XCTAssertEqual(status.mode, .client)
    }
}
