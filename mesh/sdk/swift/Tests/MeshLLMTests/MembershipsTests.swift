import Foundation
import XCTest
@testable import MeshLLM

final class MembershipsTests: XCTestCase {
    private static let status = #"{"name":"team","api_base_url":"http://localhost:9338/v1","invite_token":"invite","endpoint_id":"node-a","peers":2,"join_error":null}"#

    func testNativeMembershipControlsAndStatus() async throws {
        let memberships = Memberships(addJSON: { request in
            let value = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(request.utf8)) as? [String: Any])
            XCTAssertEqual(value["name"] as? String, "team")
            XCTAssertEqual(value["trusted_owners"] as? [String], ["owner"])
            XCTAssertEqual(value["api_port"] as? Int, 9338)
            XCTAssertEqual(value["enumerate_host"] as? Bool, false)
            return Self.status
        }, removeNative: { name in XCTAssertEqual(name, "team"); return false },
           listJSON: { "[\(Self.status)]" })
        let added = try await memberships.add(MeshMembershipConfig(name: "team", trustedOwners: ["owner"], apiPort: 9338))
        XCTAssertEqual(added.apiBaseUrl, "http://localhost:9338/v1")
        XCTAssertEqual(added.endpointId, "node-a")
        XCTAssertNil(added.joinError)
        let listed = try await memberships.list()
        XCTAssertEqual(listed.first?.peers, 2)
        let removed = try await memberships.remove("team")
        XCTAssertFalse(removed)
    }

    func testNativeFailuresPropagate() async throws {
        enum Failure: Error { case rejected }
        let memberships = Memberships(addJSON: { _ in throw Failure.rejected },
            removeNative: { _ in throw Failure.rejected }, listJSON: { throw Failure.rejected })
        do { _ = try await memberships.add(.init(name: "?", apiPort: 9338)); XCTFail("expected rejection") }
        catch Failure.rejected { }
        do { _ = try await memberships.remove("?"); XCTFail("expected rejection") }
        catch Failure.rejected { }
        do { _ = try await memberships.list(); XCTFail("expected rejection") }
        catch Failure.rejected { }
    }
}
