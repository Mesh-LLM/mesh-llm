import Foundation

public actor ProviderRequestScheduler {
  private var occupied = false
  private var waiterOrder: [UUID] = []
  private var waiters: [UUID: CheckedContinuation<Void, Never>] = [:]

  public init() {}

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
    try await acquire()
    do {
      try Task.checkCancellation()
      let result = try await operation()
      release()
      return result
    } catch {
      release()
      throw error
    }
  }

  private func acquire() async throws {
    try Task.checkCancellation()
    if !occupied {
      occupied = true
      return
    }
    let id = UUID()
    try await withTaskCancellationHandler {
      await withCheckedContinuation { continuation in
        waiterOrder.append(id)
        waiters[id] = continuation
      }
      try Task.checkCancellation()
    } onCancel: {
      Task { await self.cancelWaiter(id) }
    }
  }

  private func cancelWaiter(_ id: UUID) {
    waiterOrder.removeAll { $0 == id }
    waiters.removeValue(forKey: id)?.resume()
  }

  private func release() {
    while let id = waiterOrder.first {
      waiterOrder.removeFirst()
      if let continuation = waiters.removeValue(forKey: id) {
        continuation.resume()
        return
      }
    }
    occupied = false
  }
}
