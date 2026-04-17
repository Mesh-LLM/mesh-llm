import XCTest
@testable import MeshLLM

final class MeshClientTests: XCTestCase {
    func testClientCreation() {
        let client = MeshClient(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: makeOwnerKeypairBytesHex())
        XCTAssertNotNil(client)
    }

    func testStatusBeforeJoin() async {
        let client = MeshClient(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: makeOwnerKeypairBytesHex())
        let status = await client.status()
        XCTAssertFalse(status.connected)
    }

    func testJoinAndStatus() async throws {
        let client = MeshClient(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: makeOwnerKeypairBytesHex())
        try await client.join()
        let status = await client.status()
        XCTAssertTrue(status.connected)
    }

    func testDisconnect() async throws {
        let client = MeshClient(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: makeOwnerKeypairBytesHex())
        try await client.join()
        await client.disconnect()
        let status = await client.status()
        XCTAssertFalse(status.connected)
    }

    func testReconnect() async throws {
        let client = MeshClient(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: makeOwnerKeypairBytesHex())
        try await client.join()
        try await client.reconnect()
        let status = await client.status()
        XCTAssertTrue(status.connected)
    }
}
