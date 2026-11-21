import Foundation
import XCTest
@testable import MeshLLM

private actor StatusQueue {
    var values: [String]
    var closed = false
    init(_ values: [String]) { self.values = values }
    func next() -> String? { values.isEmpty ? nil : values.removeFirst() }
    func close() { closed = true }
}

private actor PendingStatusRead {
    var waiter: CheckedContinuation<String?, Never>?
    var closed = false
    func next() async -> String? {
        if closed { return nil }
        return await withCheckedContinuation { waiter = $0 }
    }
    func close() {
        closed = true
        waiter?.resume(returning: nil)
        waiter = nil
    }
}

final class StatusTests: XCTestCase {
    private func snapshot(_ phase: String, error: String? = nil) throws -> String {
        var state = ["phase": phase]
        state["error"] = error
        let value: [String: Any] = ["state": state, "serving_models": ["chat"],
            "models": [], "local_models": [["identity": ["model_name": "chat"]]],
            "capacities": [["model": "chat", "context_per_lane": [4096, 8192], "last_success_unix_ms": 123]],
            "download": ["file": "weights", "downloaded_bytes": 10, "total_bytes": NSNull()]]
        return String(decoding: try JSONSerialization.data(withJSONObject: value), as: UTF8.self)
    }

    func testNativeSnapshotsPreserveProgressCapacityAndFailureThenClose() async throws {
        let queue = StatusQueue(try [snapshot("loading"), snapshot("downloading"), snapshot("ready"), snapshot("failed", error: "runtime failed")])
        let events = StatusEvents(nextJSON: { await queue.next() }, close: { await queue.close() })
        var seen: [NodeLifecycle] = []
        for try await status in events.events() {
            seen.append(status.state)
            XCTAssertEqual(status.download?.downloadedBytes, 10)
            XCTAssertNil(status.download?.totalBytes)
            XCTAssertEqual(status.capacities.first?.contextPerLane, [4096, 8192])
            XCTAssertEqual(status.capacities.first?.lastSuccessUnixMs, 123)
            XCTAssertEqual(status.localModels.first, .object(["identity": .object(["model_name": .string("chat")])]))
        }
        // Stream buffering deliberately coalesces progress; the terminal state must survive.
        XCTAssertEqual(seen.last, .failed("runtime failed"))
        await events.close()
        let closed = await queue.closed
        XCTAssertTrue(closed)
    }

    func testCancellingAPendingReadClosesTheNativeSubscription() async throws {
        let source = PendingStatusRead()
        let reading = expectation(description: "reading native status")
        let closed = expectation(description: "closed native subscription")
        let events = StatusEvents(nextJSON: {
            reading.fulfill()
            return await source.next()
        }, close: {
            await source.close()
            closed.fulfill()
        })
        let task = Task { try await events.nextEvent() }
        await fulfillment(of: [reading], timeout: 2)
        task.cancel()
        await fulfillment(of: [closed], timeout: 2)
        await source.close()
        let result = try await task.value
        XCTAssertNil(result)
    }

    func testNativeSourceErrorsPropagate() async throws {
        enum Failure: Error { case disconnected }
        let events = StatusEvents(nextJSON: { throw Failure.disconnected }, close: {})
        do { _ = try await events.nextEvent(); XCTFail("expected native failure") }
        catch Failure.disconnected { }
    }
}
