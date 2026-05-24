import Foundation
import XCTest
@testable import MeshLLM

func makeOwnerKeypairBytesHex() -> String {
    generateOwnerKeypairHex()
}

func makeTestNode() throws -> Node {
    Node(handle: TestMeshNodeHandle())
}

final class TestMeshNodeHandle: MeshNodeHandle, @unchecked Sendable {
    private let requestId = "test-request"
    private let lock = NSLock()
    private var cancelledRequestIdsStorage: [String] = []
    private var connected = false

    var cancelledRequestIds: [String] {
        lock.lock()
        defer { lock.unlock() }
        return cancelledRequestIdsStorage
    }

    init() {
        super.init(noHandle: MeshNodeHandle.NoHandle())
    }

    required init(unsafeFromHandle handle: UInt64) {
        super.init(unsafeFromHandle: handle)
    }

    override func chat(request: ChatRequestNative, listener: EventListener) throws -> String {
        listener.onEvent(event: .tokenDelta(requestId: requestId, delta: "hello"))
        listener.onEvent(event: .completed(requestId: requestId))
        return requestId
    }

    override func responses(request: ResponsesRequestNative, listener: EventListener) throws -> String {
        listener.onEvent(event: .tokenDelta(requestId: requestId, delta: "hello"))
        listener.onEvent(event: .completed(requestId: requestId))
        return requestId
    }

    override func cancel(requestId: String) throws {
        lock.lock()
        cancelledRequestIdsStorage.append(requestId)
        lock.unlock()
    }

    override func reconnect() throws {
        lock.lock()
        connected = true
        lock.unlock()
    }

    override func start() throws {
        lock.lock()
        connected = true
        lock.unlock()
    }

    override func status() -> ClientStatus {
        lock.lock()
        let isConnected = connected
        lock.unlock()
        return ClientStatus(connected: isConnected, peerCount: isConnected ? 1 : 0)
    }

    override func stop() throws {
        lock.lock()
        connected = false
        lock.unlock()
    }
}

final class OpenStreamMeshNodeHandle: MeshNodeHandle, @unchecked Sendable {
    private let requestId = "open-request"
    private let lock = NSLock()
    private var cancelledRequestIdsStorage: [String] = []
    private var chatListener: EventListener?

    var cancelledRequestIds: [String] {
        lock.lock()
        defer { lock.unlock() }
        return cancelledRequestIdsStorage
    }

    init() {
        super.init(noHandle: MeshNodeHandle.NoHandle())
    }

    required init(unsafeFromHandle handle: UInt64) {
        super.init(unsafeFromHandle: handle)
    }

    override func chat(request: ChatRequestNative, listener: EventListener) throws -> String {
        lock.lock()
        chatListener = listener
        lock.unlock()
        listener.onEvent(event: .tokenDelta(requestId: requestId, delta: "hello"))
        return requestId
    }

    override func responses(request: ResponsesRequestNative, listener: EventListener) throws -> String {
        listener.onEvent(event: .tokenDelta(requestId: requestId, delta: "hello"))
        return requestId
    }

    override func cancel(requestId: String) throws {
        lock.lock()
        cancelledRequestIdsStorage.append(requestId)
        lock.unlock()
    }

    func completeChat() {
        lock.lock()
        let listener = chatListener
        lock.unlock()
        listener?.onEvent(event: .completed(requestId: requestId))
    }
}

func waitUntil(
    timeout: Duration = .seconds(2),
    _ condition: @escaping () -> Bool
) async throws {
    let start = ContinuousClock.now
    while !condition() {
        if start.duration(to: .now) > timeout {
            XCTFail("condition was not met before timeout")
            return
        }
        try await Task.sleep(for: .milliseconds(10))
    }
}
