import XCTest
@testable import MeshLLM

final class NodeTests: XCTestCase {
    func testNodeCreation() throws {
        let node = try makeTestNode()
        XCTAssertNotNil(node)
    }

    func testMeshErrorAliasExposesTypedFfiErrors() {
        let error = MeshError.InvalidInviteToken(message: "bad token")
        guard case .InvalidInviteToken(let message) = error else {
            return XCTFail("expected InvalidInviteToken")
        }
        XCTAssertEqual(message, "bad token")
    }

    func testStatusBeforeStart() async throws {
        let node = try makeTestNode()
        let status = await node.status()
        XCTAssertFalse(status.connected)
    }

    func testStartAndStatus() async throws {
        let node = try makeTestNode()
        try await node.start()
        let status = await node.status()
        XCTAssertTrue(status.connected)
    }

    func testStop() async throws {
        let node = try makeTestNode()
        try await node.start()
        try await node.stop()
        let status = await node.status()
        XCTAssertFalse(status.connected)
    }

    func testReconnect() async throws {
        let node = try makeTestNode()
        try await node.start()
        try await node.reconnect()
        let status = await node.status()
        XCTAssertTrue(status.connected)
    }
}
