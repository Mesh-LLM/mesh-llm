import Foundation

public actor ProviderRequestScheduler {
  private enum Acquisition {
    case granted
    case cancelled
    case overloaded
  }

  private var occupied = false
  private var waiterOrder: [UUID] = []
  private var waiters: [UUID: CheckedContinuation<Acquisition, Never>] = [:]
  private let maximumQueuedRequests: Int
  private let permitHandoffHook: (@Sendable () async -> Void)?

  public init(maximumQueuedRequests: Int = 64) {
    precondition(maximumQueuedRequests >= 0)
    self.maximumQueuedRequests = maximumQueuedRequests
    permitHandoffHook = nil
  }

  init(
    maximumQueuedRequests: Int = 64,
    permitHandoffHook: @escaping @Sendable () async -> Void
  ) {
    precondition(maximumQueuedRequests >= 0)
    self.maximumQueuedRequests = maximumQueuedRequests
    self.permitHandoffHook = permitHandoffHook
  }

  public func snapshot() -> AppleProviderLoad {
    AppleProviderLoad(
      maxConcurrentRequests: 1,
      activeRequests: occupied ? 1 : 0,
      queuedRequests: waiters.count
    )
  }

  public func withPermit<T: Sendable>(
    _ operation: @Sendable () async throws -> T
  ) async throws -> T {
    try Task.checkCancellation()
    switch await acquire() {
    case .cancelled:
      throw CancellationError()
    case .overloaded:
      throw AppleRuntimeFailure(
        code: "provider_busy",
        message: "Apple provider request queue is full",
        retryable: true
      )
    case .granted:
      break
    }
    do {
      if let permitHandoffHook {
        await permitHandoffHook()
      }
      try Task.checkCancellation()
      let result = try await operation()
      release()
      return result
    } catch {
      release()
      throw error
    }
  }

  private func acquire() async -> Acquisition {
    if Task.isCancelled {
      return .cancelled
    }
    if !occupied {
      occupied = true
      return .granted
    }
    if waiters.count >= maximumQueuedRequests {
      return .overloaded
    }
    let id = UUID()
    return await withTaskCancellationHandler {
      await withCheckedContinuation { continuation in
        if Task.isCancelled {
          continuation.resume(returning: .cancelled)
          return
        }
        waiterOrder.append(id)
        waiters[id] = continuation
      }
    } onCancel: {
      Task { await self.cancelWaiter(id) }
    }
  }

  private func cancelWaiter(_ id: UUID) {
    waiterOrder.removeAll { $0 == id }
    waiters.removeValue(forKey: id)?.resume(returning: .cancelled)
  }

  private func release() {
    while let id = waiterOrder.first {
      waiterOrder.removeFirst()
      if let continuation = waiters.removeValue(forKey: id) {
        continuation.resume(returning: .granted)
        return
      }
    }
    occupied = false
  }
}
