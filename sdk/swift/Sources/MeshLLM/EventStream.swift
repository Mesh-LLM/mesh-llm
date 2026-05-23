import Foundation

public extension Node.Inference {
    func chatStream(_ request: ChatRequest) -> AsyncThrowingStream<Event, Error> {
        chat(request)
    }

    func responsesStream(_ request: ResponsesRequest) -> AsyncThrowingStream<Event, Error> {
        responses(request)
    }
}

public extension Client.Inference {
    func chatStream(_ request: ChatRequest) -> AsyncThrowingStream<Event, Error> {
        chat(request)
    }

    func responsesStream(_ request: ResponsesRequest) -> AsyncThrowingStream<Event, Error> {
        responses(request)
    }
}

#if canImport(MeshLLMFFI)
import MeshLLMFFI

public final class EventStreamBridge: EventListener, @unchecked Sendable {
    private let continuation: AsyncThrowingStream<Event, Error>.Continuation
    private let onCancel: @Sendable (String) -> Void
    private let stateLock = NSLock()
    private var requestId: String?
    private var finished = false

    public init(
        continuation: AsyncThrowingStream<Event, Error>.Continuation,
        onCancel: @escaping @Sendable (String) -> Void
    ) {
        self.continuation = continuation
        self.onCancel = onCancel
        continuation.onTermination = { [weak self] _ in
            self?.cancelIfNeeded()
        }
    }

    public func activate(requestId: String) {
        stateLock.lock()
        guard !finished else {
            stateLock.unlock()
            return
        }
        self.requestId = requestId
        stateLock.unlock()
    }

    public func onEvent(event: ClientEvent) {
        let mapped = Node.mapEvent(event)
        switch mapped {
        case .completed, .failed, .disconnected:
            finish(with: mapped)
        default:
            stateLock.lock()
            let isFinished = finished
            stateLock.unlock()
            guard !isFinished else {
                return
            }
            continuation.yield(mapped)
        }
    }

    public func finish(throwing error: Error? = nil) {
        stateLock.lock()
        guard !finished else {
            stateLock.unlock()
            return
        }
        finished = true
        requestId = nil
        stateLock.unlock()

        if let error {
            continuation.finish(throwing: error)
        } else {
            continuation.finish()
        }
    }

    private func cancelIfNeeded() {
        stateLock.lock()
        guard !finished else {
            stateLock.unlock()
            return
        }
        let requestId = self.requestId
        finished = true
        self.requestId = nil
        stateLock.unlock()

        guard let requestId else {
            return
        }
        onCancel(requestId)
    }

    private func finish(with event: Event) {
        stateLock.lock()
        guard !finished else {
            stateLock.unlock()
            return
        }
        finished = true
        requestId = nil
        stateLock.unlock()

        continuation.yield(event)
        continuation.finish()
    }
}
#endif
