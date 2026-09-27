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
    private var cancellationPending = false

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
        if finished {
            let shouldCancel = cancellationPending
            cancellationPending = false
            stateLock.unlock()
            if shouldCancel {
                onCancel(requestId)
            }
        } else {
            self.requestId = requestId
            stateLock.unlock()
        }
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
        cancellationPending = requestId == nil
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

public final class OpenAIStreamBridge: OpenAiStreamListener, @unchecked Sendable {
    private let continuation: AsyncThrowingStream<OpenAIStreamEvent, Error>.Continuation
    private let onCancel: @Sendable (String) -> Void
    private let stateLock = NSLock()
    private var requestId: String?
    private var finished = false
    private var cancellationPending = false

    public init(
        continuation: AsyncThrowingStream<OpenAIStreamEvent, Error>.Continuation,
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
        if finished {
            let shouldCancel = cancellationPending
            cancellationPending = false
            stateLock.unlock()
            if shouldCancel {
                onCancel(requestId)
            }
        } else {
            self.requestId = requestId
            stateLock.unlock()
        }
    }

    public func onEvent(event: OpenAiStreamEventNative) {
        switch event {
        case .started(let requestId, let statusCode, let contentType):
            yield(.started(
                requestId: requestId,
                statusCode: statusCode,
                contentType: contentType
            ))
        case .sse(let requestId, let eventType, let data, let raw):
            yield(.sse(OpenAISSEEvent(
                requestId: requestId,
                event: eventType,
                data: data,
                raw: raw
            )))
        case .completed:
            finish()
        case .failed(let requestId, let statusCode, let error, let body):
            finish(throwing: OpenAIStreamFailure(
                requestId: requestId,
                statusCode: statusCode,
                message: error,
                body: body
            ))
        }
    }

    private func yield(_ event: OpenAIStreamEvent) {
        stateLock.lock()
        let isFinished = finished
        stateLock.unlock()
        guard !isFinished else { return }
        continuation.yield(event)
    }

    private func finish(throwing error: Error? = nil) {
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
        cancellationPending = requestId == nil
        stateLock.unlock()

        if let requestId {
            onCancel(requestId)
        }
    }
}
#endif
