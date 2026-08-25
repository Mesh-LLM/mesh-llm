import Foundation
import Testing

@testable import MeshAppleRuntime

private actor AsyncGate {
  private var isOpen = false
  private var continuation: CheckedContinuation<Void, Never>?

  func wait() async {
    if isOpen { return }
    await withCheckedContinuation { continuation = $0 }
  }

  func open() {
    isOpen = true
    continuation?.resume()
    continuation = nil
  }
}

private actor PermitHandoffGate {
  private var handoffCount = 0
  private var arrivalWaiters: [CheckedContinuation<Void, Never>] = []
  private var handoffContinuation: CheckedContinuation<Void, Never>?

  func arrive() async {
    handoffCount += 1
    guard handoffCount == 2 else { return }
    let waiters = arrivalWaiters
    arrivalWaiters.removeAll()
    for waiter in waiters {
      waiter.resume()
    }
    await withCheckedContinuation { handoffContinuation = $0 }
  }

  func waitForSecondHandoff() async {
    if handoffCount >= 2 { return }
    await withCheckedContinuation { arrivalWaiters.append($0) }
  }

  func releaseSecondHandoff() {
    handoffContinuation?.resume()
    handoffContinuation = nil
  }
}

@Test func cancelledPermitHandoffDoesNotStrandScheduler() async throws {
  let handoff = PermitHandoffGate()
  let operation = AsyncGate()
  let scheduler = ProviderRequestScheduler {
    await handoff.arrive()
  }

  let first = Task {
    try await scheduler.withPermit {
      await operation.wait()
      return 1
    }
  }
  while await scheduler.snapshot().activeRequests == 0 {
    await Task.yield()
  }

  let cancelled = Task { try await scheduler.withPermit { 2 } }
  while await scheduler.snapshot().queuedRequests == 0 {
    await Task.yield()
  }
  await operation.open()
  #expect(try await first.value == 1)

  await handoff.waitForSecondHandoff()
  cancelled.cancel()
  await handoff.releaseSecondHandoff()
  do {
    _ = try await cancelled.value
    Issue.record("cancelled handoff unexpectedly ran")
  } catch is CancellationError {
    // The handed-off permit was returned before cancellation escaped.
  }

  #expect(try await scheduler.withPermit { 3 } == 3)
  #expect(await scheduler.snapshot().activeRequests == 0)
  #expect(await scheduler.snapshot().queuedRequests == 0)
}

@Test func providerSchedulerBoundsQueuedContinuations() async throws {
  let scheduler = ProviderRequestScheduler(maximumQueuedRequests: 1)
  let operation = AsyncGate()
  let first = Task {
    try await scheduler.withPermit {
      await operation.wait()
      return 1
    }
  }
  while await scheduler.snapshot().activeRequests == 0 {
    await Task.yield()
  }
  let queued = Task { try await scheduler.withPermit { 2 } }
  while await scheduler.snapshot().queuedRequests == 0 {
    await Task.yield()
  }

  do {
    _ = try await scheduler.withPermit { 3 }
    Issue.record("request queue exceeded its configured bound")
  } catch let failure as AppleRuntimeFailure {
    #expect(failure.code == "provider_busy")
    #expect(failure.retryable)
  }

  queued.cancel()
  await operation.open()
  #expect(try await first.value == 1)
  do {
    _ = try await queued.value
    Issue.record("cancelled queued work unexpectedly ran")
  } catch is CancellationError {
    // Expected.
  }
}

@Test func oversizedHTTPHeadersAreRejectedBeforeBufferGrowth() {
  let oversized = Data(repeating: 0x61, count: HTTPRequest.maximumRequestHeaderBytes + 1)
  do {
    _ = try HTTPRequest.parse(oversized)
    Issue.record("oversized incomplete headers unexpectedly passed")
  } catch let failure as HTTPFailure {
    #expect(failure.status == 431)
    #expect(failure.code == "headers_too_large")
  } catch {
    Issue.record("unexpected header parsing error: \(error)")
  }
}

@Test func streamingErrorsTerminateTheEventStream() throws {
  let failure = AppleRuntimeFailure(
    code: "provider_error",
    message: "fixture failure",
    retryable: false
  )
  let data = try #require(streamErrorData(failure))
  let body = try #require(String(data: data, encoding: .utf8))
  #expect(body.contains("\"code\":\"provider_error\""))
  #expect(body.hasSuffix("data: [DONE]\n\n"))
}

@Test func providerBusyMapsToRetryableRateLimit() {
  let failure = httpFailure(
    from: AppleRuntimeFailure(
      code: "provider_busy",
      message: "queue full",
      retryable: true
    )
  )
  #expect(failure.status == 429)
  #expect(failure.retryable)
}

@Test func prewarmedSessionsAreSingleUse() {
  var prepared = SingleUsePreparedSession<Int>()
  prepared.store(42)
  #expect(prepared.take() == 42)
  #expect(prepared.take() == nil)
}
