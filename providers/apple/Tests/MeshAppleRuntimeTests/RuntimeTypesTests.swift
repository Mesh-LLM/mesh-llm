import Foundation
import Testing

@testable import MeshAppleRuntime

@Test func incrementalSnapshotsBecomeDeltas() {
  #expect(incrementalDelta(previous: "mesh", snapshot: "mesh-llm") == "-llm")
}

@Test func nonPrefixSnapshotIsPreserved() {
  #expect(incrementalDelta(previous: "old", snapshot: "replacement") == "replacement")
}

@Test func runtimeStatusIsStableJSON() throws {
  let model = AppleModelStatus(
    modelID: AppleRuntimeIdentifiers.systemModelID,
    providerKind: "system",
    availability: "available",
    unavailableReason: nil,
    contextSize: 4_096,
    supportedLanguages: ["en"],
    variant: "test",
    modelVersion: "27.0",
    versionSource: "apple_os_release_band",
    versionedModelID: "apple/system@27.0",
    capabilities: ["tool_calling"],
    load: AppleProviderLoad(
      maxConcurrentRequests: 1,
      activeRequests: 0,
      queuedRequests: 0
    )
  )
  let status = AppleRuntimeStatus(
    runtimeID: AppleRuntimeIdentifiers.runtimeID,
    protocolVersion: AppleRuntimeIdentifiers.protocolVersion,
    operatingSystem: "macOS",
    models: [model]
  )
  let data = try JSONEncoder().encode(status)
  let decoded = try JSONDecoder().decode(AppleRuntimeStatus.self, from: data)
  #expect(decoded == status)
}

@Test func providerSchedulerAdvertisesOneSlot() async {
  let load = await ProviderRequestScheduler().snapshot()
  #expect(load.maxConcurrentRequests == 1)
  #expect(load.activeRequests == 0)
  #expect(load.queuedRequests == 0)
}

private actor SchedulerGate {
  private var continuation: CheckedContinuation<Void, Never>?

  func wait() async {
    await withCheckedContinuation { continuation = $0 }
  }

  func open() {
    continuation?.resume()
    continuation = nil
  }
}

@Test func providerSchedulerSerializesAndCancelsQueuedWork() async throws {
  let scheduler = ProviderRequestScheduler()
  let gate = SchedulerGate()
  let first = Task {
    try await scheduler.withPermit {
      await gate.wait()
      return 1
    }
  }
  while await scheduler.snapshot().activeRequests == 0 {
    await Task.yield()
  }
  let second = Task { try await scheduler.withPermit { 2 } }
  while await scheduler.snapshot().queuedRequests == 0 {
    await Task.yield()
  }

  second.cancel()
  do {
    _ = try await second.value
    Issue.record("cancelled queued work unexpectedly ran")
  } catch is CancellationError {
    // Expected: cancellation removes the waiter without waiting for the active request.
  }
  #expect(await scheduler.snapshot().queuedRequests == 0)

  await gate.open()
  #expect(try await first.value == 1)
  #expect(await scheduler.snapshot().activeRequests == 0)
}

@Test func appleContextBudgetReservesRequestedOutputTokens() throws {
  try validateAppleContextBudget(
    contextSize: 4_096,
    inputTokens: 3_584,
    maximumResponseTokens: 512
  )

  do {
    try validateAppleContextBudget(
      contextSize: 4_096,
      inputTokens: 3_585,
      maximumResponseTokens: 512
    )
    Issue.record("over-budget request unexpectedly passed")
  } catch let failure as AppleRuntimeFailure {
    #expect(failure.code == "context_exceeded")
  }
}

@Test func generationDefaultsToSystemModel() {
  let request = AppleGenerationRequest(prompt: "hello")
  #expect(request.modelID == AppleRuntimeIdentifiers.systemModelID)
}

@Test func documentedSystemModelVersionsFollowAppleReleaseBands() {
  #expect(
    AppleRuntimeIdentifiers.documentedSystemModelVersion(
      for: OperatingSystemVersion(majorVersion: 26, minorVersion: 3, patchVersion: 1)
    ) == "26.0"
  )
  #expect(
    AppleRuntimeIdentifiers.documentedSystemModelVersion(
      for: OperatingSystemVersion(majorVersion: 26, minorVersion: 4, patchVersion: 0)
    ) == "26.4"
  )
  #expect(
    AppleRuntimeIdentifiers.documentedSystemModelVersion(
      for: OperatingSystemVersion(majorVersion: 27, minorVersion: 0, patchVersion: 0)
    ) == "27.0"
  )
  #expect(
    AppleRuntimeIdentifiers.documentedSystemModelVersion(
      for: OperatingSystemVersion(majorVersion: 28, minorVersion: 0, patchVersion: 0)
    ) == nil
  )
}

@Test func systemModelIDsAcceptOnlyTheInstalledDocumentedGeneration() {
  #expect(AppleRuntimeIdentifiers.isSystemModelID("apple/system"))
  if let versioned = AppleRuntimeIdentifiers.versionedSystemModelID {
    #expect(AppleRuntimeIdentifiers.isSystemModelID(versioned))
  }
  #expect(!AppleRuntimeIdentifiers.isSystemModelID("apple/system@999.0"))
}

@Test func coreAIModelIDsUseAnExplicitArtifactNamespace() {
  #expect(AppleRuntimeIdentifiers.isCoreAIModelID("apple/coreai/qwen3-4b"))
  #expect(AppleRuntimeIdentifiers.isCoreAIModelID("apple/coreai/qwen3-4b@2026-08-01"))
  #expect(!AppleRuntimeIdentifiers.isCoreAIModelID("apple/system"))
}
