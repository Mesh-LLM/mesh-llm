import Foundation
import XCTest
@testable import MeshLLM

final class RuntimeCapacityTests: XCTestCase {
    func testCapacityPreservesUnsignedValuesChangesUnavailabilityAndErrors() async throws {
        let node = Node(handle: CapacityHandle())
        let firstValue = try await node.runtimeCapacity("org/model@revision#profile")
        let first = try XCTUnwrap(firstValue)
        XCTAssertEqual(first.executionLanes, 2)
        XCTAssertEqual(first.contextTokensPerLane, UInt64.max / 2)
        XCTAssertEqual(first.contextPoolTokens, UInt64.max - 1)
        XCTAssertEqual(first.capturedAtUnixMs, UInt64.max)
        let smallerValue = try await node.runtimeCapacity("org/model@revision#profile")
        let smaller = try XCTUnwrap(smallerValue)
        XCTAssertEqual(smaller.executionLanes, 1)
        XCTAssertEqual(smaller.contextTokensPerLane, 4096)
        XCTAssertEqual(smaller.contextPoolTokens, 4096)
        XCTAssertEqual(smaller.capturedAtUnixMs, 123)
        let unavailable = try await node.runtimeCapacity("org/model@revision#profile")
        XCTAssertNil(unavailable)
        do {
            _ = try await node.runtimeCapacity("org/model@revision#profile")
            XCTFail("stopped nodes must fail")
        } catch FfiError.HostUnavailable { }
    }
}

private final class CapacityHandle: MeshNodeHandle, @unchecked Sendable {
    private let lock = NSLock()
    private var reads = 0

    init() { super.init(noHandle: MeshNodeHandle.NoHandle()) }
    required init(unsafeFromHandle handle: UInt64) { super.init(unsafeFromHandle: handle) }

    override func runtimeCapacity(model: String) throws -> RuntimeCapacityNative? {
        lock.lock()
        defer { lock.unlock() }
        XCTAssertEqual(model, "org/model@revision#profile")
        reads += 1
        switch reads {
        case 1:
            return RuntimeCapacityNative(executionLanes: 2, contextTokensPerLane: UInt64.max / 2,
                contextPoolTokens: UInt64.max - 1, capturedAtUnixMs: UInt64.max)
        case 2:
            return RuntimeCapacityNative(executionLanes: 1, contextTokensPerLane: 4096,
                contextPoolTokens: 4096, capturedAtUnixMs: 123)
        case 3: return nil
        default: throw FfiError.HostUnavailable("node is not running")
        }
    }
}
